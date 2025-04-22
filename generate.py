import os
import math
import datetime
import torch.nn as nn
import torch
import torch_scatter
import numpy as np
from tqdm import tqdm
from einops import rearrange
from transformers import logging

from plugin.VidToMe.utils import CONTROLNET_DICT
from plugin.VidToMe.utils import load_config, save_config
from plugin.VidToMe.utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from plugin.VidToMe.utils import prepare_control, load_latent, load_video, prepare_depth, save_video, save_loss_curve
from plugin.VidToMe.utils import register_time, register_attention_control, register_conv_control
from plugin.VidToMe import vidtome

from utils.general_utils import get_expon_lr_func, adaptive_instance_normalization
from utils.dataloader import OptDataset
from utils.flow_utils import warp_flow

from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
    run_chat_completion,
)

# suppress partial model loading warning
logging.set_verbosity_error()

class Generator(nn.Module):
    def __init__(self, pipe, scheduler, config):
        super().__init__()

        self.device = config.device
        self.seed = config.seed
        self.model_key = config.model_key

        self.config = config
        gene_config = config.generation
        float_precision = gene_config.float_precision if "float_precision" in gene_config else config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")

        post_opt_config = config.post_opt
        self.dataset = None
        self.apply_opt = post_opt_config.apply_opt
        self.lambda_dssim = post_opt_config.lambda_dssim
        self.lambda_flow = post_opt_config.lambda_flow
        self.lambda_tv = post_opt_config.lambda_tv
        self.lambda_exp = post_opt_config.lambda_exp
        self.epochs_exposure = post_opt_config.epochs_exposure
        self.epochs = post_opt_config.epochs
        self.opt_batch_size = post_opt_config.batch_size

        self.feature_lr = post_opt_config.feature_lr
        self.exposure_lr_init = post_opt_config.exposure_lr_init
        self.exposure_lr_final = post_opt_config.exposure_lr_final
        self.exposure_lr_delay_steps = post_opt_config.exposure_lr_delay_steps
        self.exposure_lr_delay_mult = post_opt_config.exposure_lr_delay_mult
        
        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        if config.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                print("[WARNING] xformers not found. Disable xformers attention.")
        self.n_timesteps = gene_config.n_timesteps
        scheduler.set_timesteps(gene_config.n_timesteps, device=self.device)
        self.scheduler = scheduler

        self.batch_size = 2
        self.control = gene_config.control
        self.noise_mode = gene_config.noise_mode  # vanilla, mixed, same
        self.use_depth = config.sd_version == "depth"
        self.use_controlnet = self.control in CONTROLNET_DICT.keys()
        self.use_pnp = self.control == "pnp"
        if self.use_controlnet:
            self.controlnet = pipe.controlnet
            self.controlnet_scale = gene_config.control_scale
        elif self.use_pnp:
            pnp_f_t = int(gene_config.n_timesteps * gene_config.pnp_f_t)
            pnp_attn_t = int(gene_config.n_timesteps * gene_config.pnp_attn_t)
            self.batch_size += 1
            self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)

        self.chunk_size = gene_config.chunk_size
        self.chunk_ord = gene_config.chunk_ord
        self.merge_global = gene_config.merge_global
        self.local_merge_ratio = gene_config.local_merge_ratio
        self.global_merge_ratio = gene_config.global_merge_ratio
        self.global_rand = gene_config.global_rand
        self.align_batch = gene_config.align_batch
        self.max_downsample = gene_config.max_downsample
        self.win_size_t = gene_config.win_size_t
        self.alpha_t = gene_config.alpha_t
        self.final_factor_t = gene_config.final_factor_t

        data_config = config.data
        data_config.apply_opt = config.post_opt.apply_opt
        if data_config.scene_type.lower() == "sceneflow":
            from utils.dataparsers import SceneFlowDataParser
            self.data_parser = SceneFlowDataParser(data_config, self.device)
        elif data_config.scene_type.lower() == "carla":
            from utils.dataparsers import CarlaDataParser
            self.data_parser = CarlaDataParser(data_config, self.device)
        elif data_config.scene_type.lower() == "robotrix":
            from utils.dataparsers import RobotrixDataParser
            self.data_parser = RobotrixDataParser(data_config, self.device)
        elif data_config.scene_type.lower() == "interiornet":
            from utils.dataparsers import InteriorNetDataParser
            self.data_parser = InteriorNetDataParser(data_config, self.device)
        elif data_config.scene_type.lower() == "video":
            from utils.dataparsers import VideoDataParser
            self.data_parser = VideoDataParser(data_config, self.device)
        else:
            raise NotImplementedError(f"Scene type {data_config.scene_type} is not supported.")          

        self.prompt = gene_config.prompt
        self.negative_prompt = gene_config.negative_prompt
        self.prompt_t = gene_config.prompt_t
        self.negative_prompt_t = gene_config.negative_prompt_t
        self.guidance_scale = gene_config.guidance_scale
        self.save_frame = gene_config.save_frame

        self.work_dir = config.work_dir

        self.chunk_ord = gene_config.chunk_ord
        if "mix" in self.chunk_ord:
            self.perm_div = float(self.chunk_ord.split("-")[-1]) if "-" in self.chunk_ord else 3.
            self.chunk_ord = "mix"
        # Patch VidToMe to model
        self.activate_vidtome()

        if gene_config.use_lora:
            self.pipe.load_lora_weights(**gene_config.lora)
    
    def activate_vidtome(self):
        vidtome.apply_patch(self.pipe, self.local_merge_ratio, self.merge_global, self.global_merge_ratio, max_downsample=self.max_downsample,
            seed = self.seed, batch_size = self.batch_size, align_batch = self.use_pnp or self.align_batch, global_rand = self.global_rand)        

    @torch.inference_mode()
    def encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.text_encoder(token_ids).last_hidden_state

        return conds

    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    @torch.inference_mode()
    def get_text_embeds_input(self, prompt, negative_prompt):
        text_embeds = self.get_text_embeds(
            prompt, negative_prompt, self.device)
        if self.use_pnp:
            pnp_guidance_embeds = self.get_text_embeds("", device=self.device)
            text_embeds = torch.cat(
                [pnp_guidance_embeds, text_embeds], dim=0)
        return text_embeds

    @torch.inference_mode()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        if negative_prompt is not None:
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                          return_tensors='pt')
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.inference_mode()
    def prepare_data(self, latent_path, frame_ids):
        self.frames, _, _, flows, past_flows, mask_bwds = self.data_parser.load_video(frame_ids=frame_ids)

        self.dataset = OptDataset(
            self.frames,
            past_flows,
            mask_bwds,
            device=self.device
        )
        
        if latent_path is None:
            if self.noise_mode.lower() == "vanilla":
                self.init_noise = self.pipe.prepare_latents(
                    self.frames.shape[0],
                    self.pipe.unet.config.in_channels,
                    self.frames.shape[2],
                    self.frames.shape[3],
                    self.dtype,
                    self.frames.device,
                    generator=self.rng[0],
                    latents=None,
                )
            elif self.noise_mode.lower() == "same":
                self.init_noise = self.pipe.prepare_latents(
                    1,
                    self.pipe.unet.config.in_channels,
                    self.frames.shape[2],
                    self.frames.shape[3],
                    self.dtype,
                    self.frames.device,
                    generator=self.rng[0],
                    latents=None,
                ).repeat(self.frames.shape[0], 1, 1, 1)
            elif self.noise_mode.lower() == "mixed":
                alpha = 1.0
                init_noise_ind = self.pipe.prepare_latents(
                    self.frames.shape[0],
                    self.pipe.unet.config.in_channels,
                    self.frames.shape[2],
                    self.frames.shape[3],
                    self.dtype,
                    self.frames.device,
                    generator=self.rng,
                    latents=None,
                )
                init_noise_shared = self.pipe.prepare_latents(
                    1,
                    self.pipe.unet.config.in_channels,
                    self.frames.shape[2],
                    self.frames.shape[3],
                    self.dtype,
                    self.frames.device,
                    generator=torch.Generator(device=self.device).manual_seed(int(self.seed)),
                    latents=None,
                )
                self.init_noise = init_noise_ind / math.sqrt(1 + alpha**2) + alpha * init_noise_shared / math.sqrt(1 + alpha**2)
            else:
                raise NotImplementedError(f"Noise mode {self.noise_mode} is not supported.")

        else:
            self.init_noise = load_latent(
                latent_path, t=self.scheduler.timesteps[0], frame_ids=frame_ids).to(self.dtype).to(self.device)

            control_save_path = os.path.dirname(os.path.dirname(latent_path))

        if self.use_depth:
            self.depths = prepare_depth(
                self.pipe, self.frames, frame_ids, control_save_path).to(self.init_noise)

        if self.use_controlnet:
            self.controlnet_images = prepare_control(
                self.control, self.frames, frame_ids, control_save_path).to(self.init_noise)

    @torch.inference_mode()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.inference_mode()
    def decode_latents_batch(self, latents):
        imgs = []
        batch_latents = latents.split(self.batch_size, dim=0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs

    @torch.inference_mode()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mode() * 0.18215
        return latents

    @torch.inference_mode()
    def encode_imgs_batch(self, imgs):
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim=0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)]
        latents = torch.cat(latents)
        return latents
    
    def get_chunks(self, flen):
        x_index = torch.arange(flen)

        # The first chunk has a random length
        rand_first = np.random.randint(0, self.chunk_size) + 1
        chunks = x_index[rand_first:].split(self.chunk_size, dim=0)
        chunks = [x_index[:rand_first]] + list(chunks) if len(chunks[0]) > 0 else [x_index[:rand_first]]
        if np.random.rand() > 0.5:
            chunks = chunks[::-1]
        
        # Chunk order only matter when we do global token merging
        if self.merge_global == False:
            return chunks

        # Chunk order. "seq": sequential order. "rand": full permutation. "mix": partial permutation.
        if self.chunk_ord == "rand":
            order = torch.randperm(len(chunks))
        elif self.chunk_ord == "mix":
            randord = torch.randperm(len(chunks)).tolist()
            rand_len = int(len(randord) / self.perm_div)
            seqord = sorted(randord[rand_len:])
            if rand_len > 0:
                randord = randord[:rand_len]
                if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                    seqord = seqord[::-1]
                order = randord + seqord
            else:
                order = seqord
        else:
            order = torch.arange(len(chunks))
        chunks = [chunks[i] for i in order]
        return chunks

    @torch.inference_mode()
    def ddim_sample(self, x, conds, conds_t, concat_conds=None):
        print("[INFO] denoising frames...")
        timesteps = self.scheduler.timesteps.to(self.device)

        noises = torch.zeros_like(x)
        noises_t = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):

            self.pre_iter(x, t)

            # Split video into chunks and denoise
            chunks = self.get_chunks(len(x))
            for chunk in chunks:
                # torch.cuda.empty_cache()
                chunk_concat_conds = concat_conds[chunk] if concat_conds is not None else None
                noises[chunk] = self.pred_noise(
                    x[chunk], conds, t, chunk_concat_conds, batch_idx=chunk)

            # Temporal denoising
            if self.alpha_t > 0:
                alpha_t = self.alpha_t * (self.final_factor_t ** min(i / len(timesteps), 1))
                noises_t, noises = self.temporal_denoise(x, conds_t, t, concat_conds, alpha_t, noises_t, noises)

            # x = self.pred_next_x(x, noises, t, i, inversion=False)
            x = self.scheduler.step(noises, t, x, generator=self.rng, return_dict=False)[0]

            self.post_iter(x, t)

        return x

    def pre_iter(self, x, t):
        if self.use_pnp:
            # Prepare PnP
            register_time(self, t.item())
            cur_latents = load_latent(self.latent_path, t=t, frame_ids = self.frame_ids)
            self.cur_latents = cur_latents

    def post_iter(self, x, t):
        if self.merge_global:
            # Reset global tokens
            vidtome.update_patch(self.pipe, global_tokens = None)
    
    def temporal_denoise(self, x, conds_t, t, concat_conds, alpha_t, noises_t, noises):
        
        n_slices = math.ceil((len(x) - 1) / (self.win_size_t - 1))
        # calculate the overlaps between windows
        if n_slices > 1:
            overlap = (n_slices * self.win_size_t - len(x)) // (n_slices-1)
            overlap_last = overlap + (n_slices * self.win_size_t - len(x)) % (n_slices-1)
            overlap_list = [overlap]*(n_slices - 2) + [overlap_last]
            # calculate the start indices of each window
            cumsum_overlap = np.cumsum(overlap_list)
            sl_idxs = [0] + [(i + 1) * self.win_size_t - cumsum_overlap[i] for i in range(0, n_slices-1)]
        else:
            overlap = 0
            overlap_last = 0
            overlap_list = [0]
            sl_idxs = [0]
        
        chunks = self.get_chunks(x.shape[-1])
            
        for idx, sl_i in enumerate(sl_idxs):
            for chunk in chunks:
                concat_conds_t = rearrange(concat_conds[sl_i:sl_i+self.win_size_t, :, :, chunk], 'n c x y -> y c n x') if concat_conds is not None else None
                xt = rearrange(x[sl_i:sl_i+self.win_size_t, :, :, chunk], 'n c x y -> y c n x')
                noises_t[sl_i:sl_i+self.win_size_t, :, :, chunk] = rearrange(self.pred_noise(
                    xt, conds_t, t, concat_conds_t, batch_idx=chunk, sl_i=sl_i), 'y c n x -> n c x y')
                
            # normalize noise levels in overlapping frames
            if sl_i > 0:
                noises_t[sl_i:sl_i+overlap_list[idx-1], :, :, :] = (noises_t[sl_i:sl_i+overlap_list[idx-1], :, :, :]) * np.sqrt(1/2)

        # apply adaptive instance normalization to noises_t
        noises_t = adaptive_instance_normalization(noises_t, noises)
        noises = (np.sqrt(alpha_t)) * noises_t + (np.sqrt(1 - alpha_t)) * noises

        return noises_t, noises

    @torch.inference_mode()
    def pred_noise(self, x, cond, t, concat_conds=None, batch_idx=None, sl_i=None):

        flen = len(x)
        text_embed_input = cond.repeat_interleave(flen, dim=0)

        # For classifier-free guidance
        latent_model_input = torch.cat([x, x])
        batch_size = 2

        if self.use_pnp:
            # Cat latents from inverted source frames for PnP operation
            source_latents = self.cur_latents
            if batch_idx is not None:
                if sl_i is None:
                    source_latents = source_latents[batch_idx]
                else:
                    source_latents = rearrange(source_latents[sl_i:sl_i+self.win_size_t, :, :, batch_idx], 'n c x y -> y c n x')
            latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            batch_size += 1

        # For sd-depth model
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                if sl_i is None:
                    depth = depth[batch_idx]
                else:
                    depth = rearrange(depth[sl_i:sl_i+self.win_size_t, :, :, batch_idx], 'n c x y -> y c n x')
            depth = depth.repeat(batch_size, 1, 1, 1)
            latent_model_input = torch.cat([latent_model_input, depth.to(x)], dim=1)
        
        kwargs = dict()
        # Compute controlnet outputs
        if self.use_controlnet:
            controlnet_cond = self.controlnet_images
            if batch_idx is not None:
                controlnet_cond = controlnet_cond[batch_idx]
            controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1)
            controlnet_kwargs = get_controlnet_kwargs(
                self.controlnet, latent_model_input, text_embed_input, t, controlnet_cond, self.controlnet_scale)
            kwargs.update(controlnet_kwargs)
        
        if self.model_key == 'iclight':
            kwargs.update({'cross_attention_kwargs':{'concat_conds': concat_conds}})
        
        # Pred noise!
        eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, **kwargs).sample
        noise_pred_uncond, noise_pred_cond = eps.chunk(batch_size)[-2:]
        # CFG
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    @torch.inference_mode()
    def pred_next_x(self, x, eps, t, i, inversion=False):
        if inversion:
            timesteps = reversed(self.scheduler.timesteps)
        else:
            timesteps = self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        if inversion:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        if inversion:
            pred_x0 = (x - sigma_prev * eps) / mu_prev
            x = mu * pred_x0 + sigma * eps
        else:
            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

        return x

    def init_pnp(self, conv_injection_t, qk_injection_t):
        qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control(
            self, qk_injection_timesteps, num_inputs=self.batch_size)
        register_conv_control(
            self, conv_injection_timesteps, num_inputs=self.batch_size)

    def check_latent_exists(self, latent_path):
        if self.use_pnp:
            timesteps = self.scheduler.timesteps
        else:
            timesteps = [self.scheduler.timesteps[0]]

        for ts in timesteps:
            cur_latent_path = os.path.join(
                latent_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(cur_latent_path):
                return False
        return True

    def exposure_align(self):

        from utils.loss_utils import l1_loss, relaxed_ms_ssim

        data_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.opt_batch_size, 
            shuffle=True
        )
        
        loss_list_exposure = []
        N, _, H, W = self.dataset.edited_images.shape
        iterations = self.epochs_exposure * N // self.opt_batch_size
        pbar = tqdm(total=self.epochs_exposure, desc="Optimizing Exposures")

        exposure = nn.Parameter(torch.eye(3, 4, device="cuda")[None].repeat(N, 1, 1).requires_grad_(True))
        exposure_optimizer = torch.optim.Adam([exposure])
        exposure_scheduler_args = get_expon_lr_func(self.exposure_lr_init, self.exposure_lr_final,
                                                    lr_delay_steps=self.exposure_lr_delay_steps,
                                                    lr_delay_mult=self.exposure_lr_delay_mult,
                                                    max_steps=iterations)

        for epoch in range(self.epochs_exposure):
            for i, (idxs, _edited_images, _pre_edited_images, _past_flows, _mask_bwds) in enumerate(data_loader):

                iteration = epoch * N // self.opt_batch_size + i + 1
                for param_group in exposure_optimizer.param_groups:
                    param_group['lr'] = exposure_scheduler_args(iteration)

                cat_images = torch.cat([_edited_images, _pre_edited_images], dim=0)
                cat_idxs = torch.cat([idxs, idxs-1], dim=0)
                cat_idxs[cat_idxs < 0] = 0

                cat_images = torch.bmm(cat_images.permute(0, 2, 3, 1).reshape(-1, H*W, 3), exposure[cat_idxs, :3, :3]) + exposure[cat_idxs, None, :3, 3]
                cat_images = torch.clamp(cat_images, 0, 1).reshape(-1, H, W, 3).permute(0, 3, 1, 2)  # N x 3 x H x W

                images = cat_images[:len(idxs)]
                pre_images = cat_images[len(idxs):]

                loss_photometric = l1_loss(images, _edited_images) * (1 - self.lambda_dssim) + \
                                    (1.0 - relaxed_ms_ssim(images, _edited_images, data_range=1, start_level=1)) * self.lambda_dssim

                warped_images = warp_flow(pre_images, _past_flows)

                loss_flow = l1_loss(warped_images[idxs>0] * _mask_bwds[idxs>0], 
                                    images[idxs>0] * _mask_bwds[idxs>0])

                loss = (1 - self.lambda_exp) * loss_photometric + self.lambda_exp * loss_flow

                loss_list_exposure.append(loss.item())

                loss.backward()

                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none = True)
            
            pbar.set_postfix(
                loss='{:3f}'.format(loss.item()), 
                loss_flow='{:3f}'.format(loss_flow.item()),
                loss_photometric='{:3f}'.format(loss_photometric.item())
            )
            pbar.update()

        pbar.close()

        self.dataset.exposure_align(exposure)
        
        return self.dataset.edited_images, loss_list_exposure

    def unique_tensor_optimization(self):

        from utils.loss_utils import l1_loss, relaxed_ms_ssim, TVLoss
        from utils.sh_utils import RGB2SH, SH2RGB

        tv_loss = TVLoss(self.lambda_tv)

        data_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.opt_batch_size, 
            shuffle=True
        )

        with torch.no_grad(): 
            N, _, H, W = self.dataset.edited_images.shape
            feature_lr = self.feature_lr * self.opt_batch_size / N
            pil_tensor = self.dataset.edited_images.permute(0, 2, 3, 1).reshape(N*H*W, -1)
            pil_tensor = torch_scatter.scatter(pil_tensor, self.data_parser.unq_inv, dim=0, reduce='mean')

        fused_color = RGB2SH(pil_tensor)
        features_dc = nn.Parameter(fused_color.contiguous().requires_grad_(True))

        l = [
            {'params': [features_dc], 'lr': feature_lr, "name": "f_dc"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        loss_list = []

        pbar = tqdm(total=self.epochs, desc="Optimizing Unique Tensor")

        for epoch in range(self.epochs):
            for i, (idxs, _edited_images, _, _past_flows, _mask_bwds) in enumerate(data_loader):

                cat_idxs = torch.cat([idxs, idxs-1], dim=0)
                cat_idxs[cat_idxs < 0] = 0

                unq_inv = self.data_parser.unq_inv.reshape(N, H, W, -1)[cat_idxs].reshape(-1)
                cat_images = SH2RGB(features_dc)[unq_inv].reshape(len(cat_idxs), H*W, -1) # N x HW x 3
                cat_images = torch.clamp(cat_images, 0, 1).reshape(len(cat_idxs), H, W, 3).permute(0, 3, 1, 2)  # N x 3 x H x W

                images = cat_images[:len(idxs)]
                pre_images = cat_images[len(idxs):]

                warped_images = warp_flow(pre_images, _past_flows)
                
                loss_flow = l1_loss(warped_images[idxs>0] * _mask_bwds[idxs>0], 
                                    images[idxs>0] * _mask_bwds[idxs>0])

                loss_photometric = (1.0 - relaxed_ms_ssim(images, _edited_images, data_range=1, 
                                                        start_level=1)) * self.lambda_dssim

                loss = (1 - self.lambda_exp) * loss_photometric + self.lambda_exp * loss_flow + tv_loss(images)

                loss_list.append(loss.item())

                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none = True)

            pbar.set_postfix(
                loss='{:3f}'.format(loss.item()), 
                loss_flow='{:3f}'.format(loss_flow.item()),
                loss_photometric='{:3f}'.format(loss_photometric.item())
            )
            pbar.update()

        pbar.close()

        with torch.no_grad():
            images = SH2RGB(features_dc)[self.data_parser.unq_inv].reshape(N, H*W, -1) # N x HW x 3
            images = torch.clamp(images, 0, 1).reshape(N, H, W, 3).permute(0, 3, 1, 2)  # N x 3 x H x W

        return images, loss_list
    
    def __call__(self, latent_path, output_path, frame_ids):
        self.scheduler.set_timesteps(self.n_timesteps)
        latent_path = get_latents_dir(latent_path, self.model_key)
        latent_path = None if not self.check_latent_exists(latent_path) else latent_path
        if latent_path is None:
            print("[INFO] latent path not found. Generate new latents.")
        else:
            print(f"[INFO] latent path found at {latent_path}")
        
        self.rng = [torch.Generator(device=self.device).manual_seed(int(self.seed))] * len(frame_ids)
        self.latent_path = latent_path
        self.frame_ids = frame_ids
        self.prepare_data(latent_path, frame_ids)

        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")
        self.frames = self.frames.to(device=self.vae.device, dtype=self.vae.dtype)

        for edit_name, edit_prompt in self.prompt.items():
            # concat_conds = self.vae.encode(self.frames).latent_dist.mode() * self.vae.config.scaling_factor
            torch.cuda.reset_peak_memory_stats()
            start_time = datetime.datetime.now()

            opt_post_fix = "_opt" if self.apply_opt else ""
            opt_post_fix += "_upsampled" if edit_prompt is None else ""
            save_name = f"{edit_name}_lmr_{self.local_merge_ratio}_gmr_{self.global_merge_ratio}_alpha_t_{self.alpha_t}"+opt_post_fix
            cur_output_path = os.path.join(output_path, save_name)

            if edit_prompt is None:
                if not self.data_parser.rgb_path.endswith(".mp4"):
                    save_video(self.frames, cur_output_path, save_frame=False, post_fix = "_gt", gif=False)
                    self.data_parser.rgb_path = os.path.join(cur_output_path, "output_gt.mp4")
                
                with torch.no_grad():
                    dialog = prepare_dialog(self.data_parser.rgb_path)
                    prompt_upsampler = create_vlm_prompt_upsampler(
                        checkpoint_dir=self.config.generation.prompt_upsampler_ckpt,
                    )
                    edit_prompt = run_chat_completion(
                        prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
                    )
                self.config.generation.prompt[edit_name] = edit_prompt
                del prompt_upsampler
                torch.cuda.empty_cache()

            print(f"[INFO] current prompt: {edit_prompt}")

            if self.model_key == 'iclight':
                concat_conds = self.encode_imgs_batch(self.frames)
                # clean_frames = self.decode_latents_batch(concat_conds)  # reconstruct to check results

                assert concat_conds.shape[1] == self.pipe.unet.config.in_channels, f"Expected {self.pipe.unet.config.in_channels} channels, got {concat_conds.shape[1]}"
                conds, unconds = self.encode_prompt_pair(positive_prompt=edit_prompt, negative_prompt=self.negative_prompt)
                conds_t, unconds_t = self.encode_prompt_pair(positive_prompt=self.prompt_t, negative_prompt=self.negative_prompt_t)

                prompt_embeds = torch.cat([unconds, conds])
                prompt_embeds_t = torch.cat([unconds_t, conds_t])
            else:
                concat_conds = None
                prompt_embeds = self.get_text_embeds_input(edit_prompt, self.negative_prompt)
                prompt_embeds_t = self.get_text_embeds_input(self.prompt_t, self.negative_prompt_t)

            # Comment this if you have enough GPU memory
            clean_latent = self.ddim_sample(self.init_noise, prompt_embeds, prompt_embeds_t, concat_conds)
            torch.cuda.empty_cache()
            clean_frames = self.decode_latents_batch(clean_latent)

            if self.apply_opt:
                self.dataset = OptDataset(
                    clean_frames,
                    self.dataset.past_flows,
                    self.dataset.mask_bwd,
                    device=self.device
                )  # update dataset

                clean_frames, loss_list_exposure = self.exposure_align()
                clean_frames, loss_list = self.unique_tensor_optimization()

            end_time = datetime.datetime.now()
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            self.config.max_memory_allocated = max(max_memory_allocated, self.config.max_memory_allocated)
            self.config.total_time = (end_time - start_time).total_seconds() + self.config.total_time
            self.config.sec_per_frame = self.config.total_time / len(frame_ids)

            save_config(self.config, cur_output_path, gene = True)
            save_video(clean_frames, cur_output_path, save_frame=self.save_frame, fps=self.data_parser.fps, gif=False)

            if not os.path.exists(os.path.join(output_path, save_name, "gt")) or \
                len(os.listdir(os.path.join(output_path, edit_name, "gt"))) != len(self.frames):
                self.frames = self.frames.to(device=clean_frames.device, dtype=clean_frames.dtype)
                save_video(self.frames, cur_output_path, save_frame=False, post_fix = "_gt", fps=self.data_parser.fps, gif=False)
            
            if self.apply_opt:
                save_loss_curve(loss_list_exposure, os.path.join(output_path, save_name), "loss_exposure")
                save_loss_curve(loss_list, os.path.join(output_path, save_name), "loss_unique_tensor")
