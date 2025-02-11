import os
import math
import torch.nn as nn
import torch
import torch_scatter
import numpy as np
from tqdm import tqdm
from transformers import logging

from plugin.VidToMe.utils import CONTROLNET_DICT
from plugin.VidToMe.utils import load_config, save_config
from plugin.VidToMe.utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from plugin.VidToMe.utils import prepare_control, load_latent, load_video, prepare_depth, save_video, save_frames
from plugin.VidToMe.utils import register_time, register_attention_control, register_conv_control

from plugin.VidToMe import vidtome

from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
    run_chat_completion,
)

# suppress partial model loading warning
logging.set_verbosity_error()

class Generator(nn.Module):
    def __init__(self, vae, pipe, scheduler, config):
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

        self.pipe = pipe
        self.vae = vae
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
        num_frames = len(list(range(*config.generation.frame_range)))
        self.rng = [torch.Generator(device=self.device).manual_seed(int(self.seed))] * num_frames

        self.batch_size = 2
        self.control = gene_config.control
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
        self.vox_ratio = gene_config.vox_ratio

        data_config = config.data
        if  data_config.scene_type.lower() == "sceneflow":
            from utils.dataparsers import SceneFlowDataParser
            self.data_parser = SceneFlowDataParser(data_config, self.device)
            config.input_path = self.data_parser.rgb_path
        else:
            raise NotImplementedError(f"Scene type {data_config.scene_type} is not supported.")

        for prompt_name in gene_config.prompt.keys():
            if gene_config.prompt[prompt_name] is None:
                dialog = prepare_dialog(self.data_parser.rgb_path)
                prompt_upsampler = create_vlm_prompt_upsampler(
                    checkpoint_dir=gene_config.prompt_upsampler_ckpt,
                )
                gene_config.prompt[prompt_name] = run_chat_completion(
                    prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
                )

        self.prompt = gene_config.prompt
        self.negative_prompt = gene_config.negative_prompt
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
        vidtome.apply_patch(self.pipe, self.local_merge_ratio, self.merge_global, self.global_merge_ratio, 
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

    @torch.no_grad()
    def get_text_embeds_input(self, prompt, negative_prompt):
        text_embeds = self.get_text_embeds(
            prompt, negative_prompt, self.device)
        if self.use_pnp:
            pnp_guidance_embeds = self.get_text_embeds("", device=self.device)
            text_embeds = torch.cat(
                [pnp_guidance_embeds, text_embeds], dim=0)
        return text_embeds

    @torch.no_grad()
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

    @torch.no_grad()
    def prepare_data(self, latent_path, frame_ids, apply_unq_inv):
        if apply_unq_inv:
            self.frames = self.data_parser.load_video(frame_ids=frame_ids)
        else:
            self.frames = self.data_parser.load_video_flow(frame_ids=frame_ids)
        
        if latent_path is None:
            self.init_noise = self.pipe.prepare_latents(
                self.frames.shape[0],
                self.pipe.unet.config.in_channels,
                self.frames.shape[2],
                self.frames.shape[3],
                self.dtype,
                self.frames.device,
                generator=self.rng,
                latents=None,
            )
        else:
            self.init_noise = load_latent(
                latent_path, t=self.scheduler.timesteps[0], frame_ids=frame_ids).to(self.dtype).to(self.device)

        if self.use_depth:
            self.depths = prepare_depth(
                self.pipe, self.frames, frame_ids, self.work_dir).to(self.init_noise)

        if self.use_controlnet:
            self.controlnet_images = prepare_control(
                self.control, self.frames, frame_ids, self.work_dir).to(self.init_noise)
        


    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        batch_latents = latents.split(self.batch_size, dim=0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mode() * 0.18215
        return latents

    @torch.no_grad()
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

    @torch.no_grad()
    def ddim_sample(self, x, conds, concat_conds):
        print("[INFO] denoising frames...")
        timesteps = self.scheduler.timesteps.to(self.device)

        noises = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):

            x = self.pre_iter(x, t)

            # Split video into chunks and denoise
            chunks = self.get_chunks(len(x))
            for chunk in chunks:
                torch.cuda.empty_cache()
                noises[chunk] = self.pred_noise(
                    x[chunk], conds, t, concat_conds[chunk], batch_idx=chunk)

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
        
        # Note t decreases from num_train_timesteps to 0
        if t < self.scheduler.config.num_train_timesteps * self.vox_ratio:
            N, _, H, W = self.frames.shape
            decoded_x = self.decode_latents_batch(x).permute(0, 2, 3, 1).reshape(N*H*W, -1)
            decoded_x = torch_scatter.scatter(decoded_x, self.data_parser.unq_inv, dim=0, reduce='mean')
            decoded_x = decoded_x[self.data_parser.unq_inv].reshape(N, H, W, -1).permute(0, 3, 1, 2)
            x = self.encode_imgs_batch(decoded_x)
        
        return x

    def post_iter(self, x, t):
        pass
        # if self.merge_global:
        #     # Reset global tokens
        #     vidtome.update_patch(self.pipe, global_tokens = None)

    @torch.no_grad()
    def pred_noise(self, x, cond, t, concat_conds, batch_idx=None):

        flen = len(x)
        text_embed_input = cond.repeat_interleave(flen, dim=0)

        # For classifier-free guidance
        latent_model_input = torch.cat([x, x])
        batch_size = 2

        if self.use_pnp:
            # Cat latents from inverted source frames for PnP operation
            source_latents = self.cur_latents
            if batch_idx is not None:
                source_latents = source_latents[batch_idx]
            latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            batch_size += 1

        # For sd-depth model
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                depth = depth[batch_idx]
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
        
        # Pred noise!
        eps = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={'concat_conds': concat_conds},
            **kwargs).sample
        noise_pred_uncond, noise_pred_cond = eps.chunk(batch_size)[-2:]
        # CFG
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    @torch.no_grad()
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

    @torch.no_grad()
    def __call__(self, data_path, latent_path, output_path, frame_ids, unet_bg=None):
        self.scheduler.set_timesteps(self.n_timesteps)
        latent_path = get_latents_dir(latent_path, self.model_key)
        latent_path = None if not self.check_latent_exists(latent_path) else latent_path
        if latent_path is None:
            print("[INFO] latent path not found. Generate new latents.")
        else:
            print(f"[INFO] latent path found at {latent_path}")
        
        self.latent_path = latent_path
        self.frame_ids = frame_ids
        self.prepare_data(latent_path, frame_ids, apply_unq_inv=unet_bg is None)

        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")
        self.frames = self.frames.to(device=self.vae.device, dtype=self.vae.dtype)

        for edit_name, edit_prompt in self.prompt.items():
            print(f"[INFO] current prompt: {edit_prompt}")
            # concat_conds = self.vae.encode(self.frames).latent_dist.mode() * self.vae.config.scaling_factor
            concat_conds = self.encode_imgs_batch(self.frames)
            conds, unconds = self.encode_prompt_pair(positive_prompt=edit_prompt, negative_prompt=self.negative_prompt)

            if unet_bg is None:
                prompt_embeds = torch.cat([unconds, conds])
                # Comment this if you have enough GPU memory
                clean_latent = self.ddim_sample(self.init_noise, prompt_embeds, concat_conds)  
                torch.cuda.empty_cache()
                clean_frames = self.decode_latents_batch(clean_latent)
            else:
                frames_list = []
                rng = torch.Generator(device=self.device).manual_seed(int(self.seed))
                latent = self.pipe(
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=self.frames.shape[3],
                    height=self.frames.shape[2],
                    num_inference_steps=self.n_timesteps,
                    num_images_per_prompt=1,
                    generator=rng,
                    output_type='latent',
                    guidance_scale=self.guidance_scale,
                    cross_attention_kwargs={'concat_conds': concat_conds[0:1]},
                ).images.to(self.vae.dtype)
                pre_frame = self.decode_latents_batch(latent)
                frames_list.append(pre_frame)

                # replace the network
                self.pipe.unet = unet_bg
                self.unet = self.pipe.unet
                pre_y, pre_x = torch.meshgrid(torch.arange(pre_frame.shape[2]), torch.arange(pre_frame.shape[3]))
                pre_y = pre_y.to(device=self.device)
                pre_x = pre_x.to(device=self.device)

                for i in range(1, self.frames.shape[0]):
                    rng = torch.Generator(device=self.device).manual_seed(int(self.seed))
                    bg = pre_frame
                    x = (pre_x + self.data_parser.flows[i-1, 0]).to(torch.int64)
                    y = (pre_y + self.data_parser.flows[i-1, 1]).to(torch.int64)
                    mask = (x >= 0) & (x < bg.shape[3]) & (y >= 0) & (y < bg.shape[2])
                    # bg[..., y[mask], x[mask]] = pre_frame[..., pre_y[mask], pre_x[mask]]
                    fg = self.frames[i:i+1].clone()
                    fg[..., y[mask], x[mask]] = 0.5
                    concat_conds = self.encode_imgs_batch(torch.concat([fg, bg], dim=0))
                    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

                    latent = self.pipe(
                        prompt_embeds=conds,
                        negative_prompt_embeds=unconds,
                        width=self.frames.shape[3],
                        height=self.frames.shape[2],
                        num_inference_steps=self.n_timesteps,
                        num_images_per_prompt=1,
                        generator=rng,
                        output_type='latent',
                        guidance_scale=self.guidance_scale,
                        cross_attention_kwargs={'concat_conds': concat_conds},
                    ).images.to(self.vae.dtype)
                    
                    pre_frame = self.decode_latents_batch(latent)
                    frames_list.append(pre_frame)

                clean_frames = torch.cat(frames_list)

            # rng = torch.Generator(device=device).manual_seed(int(self.seed))
            # clean_latent = self.pipe(
            #     prompt_embeds=conds,
            #     negative_prompt_embeds=unconds,
            #     width=self.frames.shape[3],
            #     height=self.frames.shape[2],
            #     num_inference_steps=self.n_timesteps,
            #     num_images_per_prompt=self.frames.shape[0],
            #     generator=[rng] * self.frames.shape[0],
            #     output_type='latent',
            #     guidance_scale=self.guidance_scale,
            #     cross_attention_kwargs={'concat_conds': concat_conds},
            # ).images.to(vae.dtype)
            # clean_frames = self.decode_latents_batch(clean_latent)

            save_name = f"{edit_name}_lmr_{self.local_merge_ratio}_gmr_{self.global_merge_ratio}"
            
            cur_output_path = os.path.join(output_path, save_name)
            save_config(self.config, cur_output_path, gene = True)
            save_video(clean_frames, cur_output_path, save_frame=self.save_frame)

            if not os.path.exists(os.path.join(output_path, save_name, "gt")) or \
                len(os.listdir(os.path.join(output_path, edit_name, "gt"))) != len(self.frames):
                self.frames = self.frames.to(device=clean_frames.device, dtype=clean_frames.dtype)
                save_video(self.frames, os.path.join(output_path, save_name), save_frame=False, post_fix = "_gt")
