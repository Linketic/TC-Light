import os
import math
import datetime

# === Third-Party Libraries ===
import torch
import torch.nn as nn
import torch_scatter
import numpy as np
from tqdm import tqdm
from einops import rearrange
from transformers import logging

# === Project Utilities: VidToMe ===
from utils.VidToMe import *

# === Project Utilities: General ===
from utils.general_utils import (
    get_expon_lr_func,
    adaptive_instance_normalization,
)

# === Dataset and Flow Utilities ===
from utils.dataloader import OptDataset
from utils.flow_utils import warp_flow
from utils.loss_utils import l1_loss, relaxed_ms_ssim, TVLoss
from utils.sh_utils import RGB2SH, SH2RGB
from utils.dataparsers import (
    SceneFlowDataParser, CarlaDataParser,
    InteriorNetDataParser, VideoDataParser,
)

# === COSMOS Prompt Upsampler ===
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
    run_chat_completion,
)


class Generator(VidToMeGenerator):
    def __init__(self, pipe, scheduler, config):
        super().__init__(pipe, scheduler, config)

        self.config = config
        self._init_dataset(config.data)
        self._init_generation(config.generation)
        self._init_post_optimization(config.post_opt)

    def _init_post_optimization(self, post_opt_config):
        """Initialize post-optimization related hyperparameters."""
        self.apply_opt = post_opt_config.apply_opt
        self.lambda_dssim = post_opt_config.lambda_dssim
        self.lambda_flow = post_opt_config.lambda_flow
        self.lambda_tv = post_opt_config.lambda_tv
        self.epochs_exposure = post_opt_config.epochs_exposure
        self.epochs = post_opt_config.epochs
        self.opt_batch_size = post_opt_config.batch_size

        self.feature_lr = post_opt_config.feature_lr
        self.exposure_lr_init = post_opt_config.exposure_lr_init
        self.exposure_lr_final = post_opt_config.exposure_lr_final
        self.exposure_lr_delay_steps = post_opt_config.exposure_lr_delay_steps
        self.exposure_lr_delay_mult = post_opt_config.exposure_lr_delay_mult

    def _init_generation(self, gene_config):
        """Initialize generation settings."""
        self.background_cond = gene_config.background_cond
        self.background_image_path = gene_config.background_image_path
        self.noise_mode = gene_config.noise_mode  # e.g., 'mixed', 'same'

        self.max_downsample = gene_config.max_downsample
        self.win_size_t = gene_config.win_size_t
        self.alpha_t = gene_config.alpha_t
        self.final_factor_t = gene_config.final_factor_t

        self.prompt_t = gene_config.prompt_t
        self.negative_prompt_t = gene_config.negative_prompt_t

    def _init_dataset(self, data_config):
        """Initialize dataset and data parser."""
        self.dataset = None

        scene_type = data_config.scene_type.lower()
        parser_map = {
            "sceneflow": SceneFlowDataParser,
            "carla": CarlaDataParser,
            "interiornet": InteriorNetDataParser,
            "video": VideoDataParser,
        }

        if scene_type not in parser_map:
            raise NotImplementedError(f"Scene type '{scene_type}' is not supported.")

        self.data_parser = parser_map[scene_type](data_config, self.device)
    
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
    def prepare_data(self, latent_path, frame_ids):
        """Prepare video frames, background blending, and latent noise for sampling."""
        
        # === Load raw frames ===
        self.frames = self.data_parser.load_video(frame_ids=frame_ids)
        N, C, H, W = self.frames.shape
        assert C == 3, "Input frames must be 3-channel RGB"

        # === Optional: Background Compositing ===
        if self.background_cond:
            from briarmbg import BriaRMBG
            rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4").to(self.frames.device, dtype=self.frames.dtype)

            # Resize input for RMBG network
            scale = (256.0 / float(H * W)) ** 0.5
            resized_size = (int(64 * round(W * scale)), int(64 * round(H * scale)))
            resized_frames = torch.nn.functional.interpolate(self.frames, size=resized_size, mode="bilinear")

            # Estimate alpha masks in batches
            alphas = []
            for img_batch in resized_frames.split(self.batch_size, dim=0):
                alpha = rmbg(img_batch * 255.0)[0][0]  # RMBG expects inputs in [0, 255]
                alphas.append(alpha)

            alpha = torch.cat(alphas, dim=0)
            alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear").clamp(0, 1)

            # Blend foreground and background frames
            background_frames = self.data_parser.load_video(path=self.background_image_path)
            self.frames = alpha * self.frames + (1 - alpha) * background_frames

        torch.cuda.empty_cache()

        # === Prepare latent noise ===
        self.init_noise = None
        if latent_path is None:
            in_channels = self.pipe.unet.config.in_channels
            noise_shape = (self.frames.shape[2], self.frames.shape[3])
            n_frames = self.frames.shape[0]

            if self.noise_mode.lower() == "vanilla":
                self.init_noise = self.pipe.prepare_latents(
                    n_frames, in_channels, *noise_shape,
                    dtype=self.dtype, device=self.frames.device, generator=self.rng[0], latents=None
                )
            elif self.noise_mode.lower() == "same":
                noise = self.pipe.prepare_latents(
                    1, in_channels, *noise_shape,
                    dtype=self.dtype, device=self.frames.device, generator=self.rng[0], latents=None
                )
                self.init_noise = noise.repeat(n_frames, 1, 1, 1)
            else:
                raise NotImplementedError(f"Noise mode '{self.noise_mode}' is not supported.")
        else:
            self.init_noise = load_latent(
                latent_path, t=self.scheduler.timesteps[0], frame_ids=frame_ids
            ).to(self.dtype).to(self.device)
            control_save_path = os.path.dirname(os.path.dirname(latent_path))

        # === Optional: Depth and ControlNet Conditioning, inherited from VidToMe ===
        if self.use_depth:
            self.depths = prepare_depth(self.pipe, self.frames, frame_ids, control_save_path).to(self.init_noise)

        if self.use_controlnet:
            self.controlnet_images = prepare_control(
                self.control, self.frames, frame_ids, control_save_path
            ).to(self.init_noise)


    @torch.inference_mode()
    def ddim_sample(self, x, conds, conds_t, concat_conds=None):
        """Perform DDIM sampling on latent x with optional temporal denoising."""
        print("[INFO] Denoising frames...")
        timesteps = self.scheduler.timesteps.to(self.device)

        noises = torch.zeros_like(x)
        noises_t = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            self.pre_iter(x, t)

            # === Denoise each chunk of the batch ===
            for chunk in self.get_chunks(len(x)):
                chunk_cond = concat_conds[chunk] if concat_conds is not None else None
                noises[chunk] = self.pred_noise(
                    x[chunk], conds, t, chunk_cond, batch_idx=chunk
                )

            # === Temporal denoising step ===
            if self.alpha_t > 0:
                factor = self.final_factor_t ** min(i / len(timesteps), 1)
                alpha_t = self.alpha_t * factor
                noises_t, noises = self.temporal_denoise(
                    x, conds_t, t, concat_conds, alpha_t, noises_t, noises
                )

            # === DDIM update ===
            x = self.scheduler.step(noises, t, x, generator=self.rng, return_dict=False)[0]

            self.post_iter(x, t)

        return x
    
    @torch.inference_mode()
    def temporal_denoise(self, x, conds_t, t, concat_conds, alpha_t, noises_t, noises):
        """
        Apply temporal denoising over overlapping frame windows.
        """
        num_frames = len(x)
        win = self.win_size_t
        n_slices = math.ceil((num_frames - 1) / (win - 1))

        # === Calculate overlapping indices for window slicing ===
        if n_slices > 1:
            total_overlap = n_slices * win - num_frames
            overlap = total_overlap // (n_slices - 1)
            last_overlap = overlap + total_overlap % (n_slices - 1)
            overlap_list = [overlap] * (n_slices - 2) + [last_overlap]
            cumsum_overlap = np.cumsum(overlap_list)
            sl_idxs = [0] + [(i + 1) * win - cumsum_overlap[i] for i in range(n_slices - 1)]
        else:
            sl_idxs = [0]
            overlap_list = [0]

        chunks = self.get_chunks(x.shape[-1])

        # === Denoise per time window and per chunk ===
        for idx, sl_i in enumerate(sl_idxs):
            for chunk in chunks:
                xt = rearrange(x[sl_i:sl_i + win, :, :, chunk], 'n c h w -> w c n h')
                concat_conds_t = (
                    rearrange(concat_conds[sl_i:sl_i + win, :, :, chunk], 'n c h w -> w c n h')
                    if concat_conds is not None else None
                )
                pred = self.pred_noise(xt, conds_t, t, concat_conds_t, batch_idx=chunk, sl_i=sl_i)
                noises_t[sl_i:sl_i + win, :, :, chunk] = rearrange(pred, 'w c n h -> n c h w')

            # === Normalize overlapping slices ===
            if sl_i > 0:
                overlap_len = overlap_list[idx - 1]
                noises_t[sl_i:sl_i + overlap_len] *= np.sqrt(0.5)

        # === Fuse temporal noise with original noise ===
        noises_t = adaptive_instance_normalization(noises_t, noises)
        noises = (alpha_t**0.5) * noises_t + ((1 - alpha_t)**0.5) * noises

        return noises_t, noises


    @torch.inference_mode()
    def pred_noise(self, x, cond, t, concat_conds=None, batch_idx=None, sl_i=None):
        """
        Predict noise for a batch of latents `x` with conditioning.
        Supports controlnet, depth, and PnP guidance.
        """
        flen = len(x)
        batch_size = 2
        text_embed_input = cond.repeat_interleave(flen, dim=0)

        # === Classifier-Free Guidance setup ===
        latent_model_input = torch.cat([x, x])

        # === Plug-and-Play (PnP) Support, inherited from VidToMe ===
        if self.use_pnp:
            source_latents = self.cur_latents
            if batch_idx is not None:
                if sl_i is None:
                    source_latents = source_latents[batch_idx]
                else:
                    source_latents = rearrange(
                        source_latents[sl_i:sl_i + self.win_size_t, :, :, batch_idx], 'n c h w -> w c n h')
            latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            batch_size += 1

        # === Depth Conditioning, inherited from VidToMe ===
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                if sl_i is None:
                    depth = depth[batch_idx]
                else:
                    depth = rearrange(
                        depth[sl_i:sl_i + self.win_size_t, :, :, batch_idx], 'n c h w -> w c n h')
            depth = depth.repeat(batch_size, 1, 1, 1)
            latent_model_input = torch.cat([latent_model_input, depth.to(x)], dim=1)

        # === ControlNet Conditioning, inherited from VidToMe ===
        kwargs = {}
        if self.use_controlnet:
            controlnet_cond = self.controlnet_images
            if batch_idx is not None:
                controlnet_cond = controlnet_cond[batch_idx]
            controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1)
            controlnet_kwargs = get_controlnet_kwargs(
                self.controlnet, latent_model_input, text_embed_input, t,
                controlnet_cond, self.controlnet_scale
            )
            kwargs.update(controlnet_kwargs)

        # === IC-Light Conditioning ===
        if self.model_key == 'iclight':
            kwargs['cross_attention_kwargs'] = {'concat_conds': concat_conds}

        # === UNet Forward ===
        eps = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            **kwargs
        ).sample

        noise_uncond, noise_cond = eps.chunk(batch_size)[-2:]
        noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
        
        return noise_pred

    def exposure_align(self):
        """
        Optimize per-frame exposure correction matrices using photometric loss and flow consistency.
        """

        # === Clear GPU cache ===
        torch.cuda.empty_cache()

        # === Prepare DataLoader ===
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt_batch_size,
            shuffle=True
        )

        # === Initialization ===
        loss_list_exposure = []
        N, _, H, W = self.dataset.edited_images.shape
        total_iters = self.epochs_exposure * N // self.opt_batch_size

        pbar = tqdm(total=self.epochs_exposure, desc="Optimizing Exposures")

        # === Exposure parameters: one affine matrix per frame ===
        exposure = nn.Parameter(
            torch.eye(3, 4, device="cuda")[None].repeat(N, 1, 1).requires_grad_(True)
        )

        exposure_optimizer = torch.optim.Adam([exposure])
        lr_schedule_fn = get_expon_lr_func(
            self.exposure_lr_init,
            self.exposure_lr_final,
            lr_delay_steps=self.exposure_lr_delay_steps,
            lr_delay_mult=self.exposure_lr_delay_mult,
            max_steps=total_iters
        )

        # === Optimization Loop ===
        for epoch in range(self.epochs_exposure):
            for i, (idxs, edited, pre_edited, past_flows, masks_bwd) in enumerate(data_loader):

                iter_idx = epoch * N // self.opt_batch_size + i + 1
                lr = lr_schedule_fn(iter_idx)
                for param_group in exposure_optimizer.param_groups:
                    param_group['lr'] = lr

                # === Concatenate edited and previous-edited frames ===
                cat_images = torch.cat([edited, pre_edited], dim=0)
                cat_idxs = torch.cat([idxs, idxs - 1], dim=0)
                cat_idxs[cat_idxs < 0] = 0  # Clamp indices

                # === Apply affine exposure transform ===
                flat_images = cat_images.permute(0, 2, 3, 1).reshape(-1, H * W, 3)
                transform = torch.bmm(flat_images, exposure[cat_idxs, :3, :3]) + exposure[cat_idxs, None, :3, 3]
                cat_images = torch.clamp(transform, 0, 1).reshape(-1, H, W, 3).permute(0, 3, 1, 2)

                # === Split back into edited and pre-edited ===
                images = cat_images[:len(idxs)]
                pre_images = cat_images[len(idxs):]

                # === Compute photometric loss ===
                loss_photometric = (
                    l1_loss(images, edited) * (1 - self.lambda_dssim)
                    + (1.0 - relaxed_ms_ssim(images, edited, data_range=1, start_level=1)) * self.lambda_dssim
                )

                # === Warp pre-edited frames with optical flow ===
                warped_images = warp_flow(pre_images, past_flows)

                # === Compute flow-consistency loss ===
                valid = idxs > 0
                loss_flow = l1_loss(
                    warped_images[valid] * masks_bwd[valid],
                    images[valid] * masks_bwd[valid]
                )

                # === Combine losses ===
                loss = (1 - self.lambda_flow) * loss_photometric + self.lambda_flow * loss_flow
                loss_list_exposure.append(loss.item())

                # === Optimization step ===
                loss.backward()
                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none=True)

            # === Progress Bar ===
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                loss_flow=f'{loss_flow.item():.4f}',
                loss_photometric=f'{loss_photometric.item():.4f}'
            )
            pbar.update()

        pbar.close()

        # === Save final exposure parameters ===
        self.dataset.exposure_align(exposure)

        return self.dataset.edited_images, loss_list_exposure
    
    def unique_tensor_optimization(self):
        """
        Optimizes unique video tensor using photometric loss, flow consistency,
        and TV loss for spatial smoothness.
        """
        if self.epochs <= 0:
            return self.dataset.edited_images, []

        torch.cuda.empty_cache()

        # === Setup ===
        tv_loss = TVLoss(self.lambda_tv)

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt_batch_size,
            shuffle=True
        )

        with torch.no_grad():
            N, _, H, W = self.dataset.edited_images.shape
            feature_lr = self.feature_lr * self.opt_batch_size / N

            # Aggregate per-UV color
            pil_tensor = self.dataset.edited_images.permute(0, 2, 3, 1).reshape(N * H * W, -1)
            pil_tensor = torch_scatter.scatter(pil_tensor, self.data_parser.unq_inv, dim=0, reduce='mean')
            fused_color = RGB2SH(pil_tensor)

        features_dc = nn.Parameter(fused_color.contiguous().requires_grad_(True))

        optimizer = torch.optim.Adam(
            [{'params': [features_dc], 'lr': feature_lr, "name": "f_dc"}],
            lr=0.0,
            eps=1e-15
        )

        loss_list = []
        pbar = tqdm(total=self.epochs, desc="Optimizing Unique Tensor")

        # === Optimization loop ===
        for epoch in range(self.epochs):
            for idxs, _edited_images, _, _past_flows, _mask_bwds in data_loader:

                cat_idxs = torch.cat([idxs, idxs - 1], dim=0)
                cat_idxs[cat_idxs < 0] = 0

                unq_inv = self.data_parser.unq_inv.reshape(N, H, W, -1)[cat_idxs].reshape(-1)
                cat_images = torch.index_select(SH2RGB(features_dc), 0, unq_inv)
                cat_images = torch.clamp(cat_images, 0, 1).reshape(len(cat_idxs), H, W, 3).permute(0, 3, 1, 2)

                images = cat_images[:len(idxs)]
                pre_images = cat_images[len(idxs):]

                warped_images = warp_flow(pre_images, _past_flows)

                valid = idxs > 0
                loss_flow = l1_loss(warped_images[valid] * _mask_bwds[valid], images[valid] * _mask_bwds[valid])
                loss_photometric = (1.0 - relaxed_ms_ssim(images, _edited_images, data_range=1, start_level=1)) * self.lambda_dssim

                loss = (1 - self.lambda_flow) * loss_photometric + self.lambda_flow * loss_flow + tv_loss(images)
                loss_list.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                loss_flow=f'{loss_flow.item():.4f}',
                loss_photometric=f'{loss_photometric.item():.4f}'
            )
            pbar.update()

        pbar.close()

        # === Final image synthesis ===
        with torch.no_grad():
            images = SH2RGB(features_dc)[self.data_parser.unq_inv].reshape(N, H * W, -1)
            images = torch.clamp(images, 0, 1).reshape(N, H, W, 3).permute(0, 3, 1, 2)

        return images, loss_list

    @torch.inference_mode()
    def _handle_missing_prompt(self, output_path):
        if not self.data_parser.rgb_path.endswith(".mp4"):
            save_video(self.frames, output_path, save_frame=False, post_fix="_gt", gif=False)
            self.data_parser.rgb_path = os.path.join(output_path, "output_gt.mp4")
        
        with torch.no_grad():
            dialog = prepare_dialog(self.data_parser.rgb_path)
            prompt_upsampler = create_vlm_prompt_upsampler(self.config.generation.prompt_upsampler_ckpt)
            edit_prompt = run_chat_completion(prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9)
            del prompt_upsampler
            torch.cuda.empty_cache()
        return edit_prompt
    
    @torch.inference_mode()
    def _prepare_prompts_and_conditions(self, edit_prompt):
        if self.model_key == 'iclight':
            concat_conds = self.encode_imgs_batch(self.frames)
            conds, unconds = self.encode_prompt_pair(edit_prompt, self.negative_prompt)
            conds_t, unconds_t = self.encode_prompt_pair(self.prompt_t, self.negative_prompt_t)
            return concat_conds, torch.cat([unconds, conds]), torch.cat([unconds_t, conds_t])
        else:
            return None, self.get_text_embeds_input(edit_prompt, self.negative_prompt), self.get_text_embeds_input(self.prompt_t, self.negative_prompt_t)

    
    def __call__(self, latent_path, output_path, frame_ids):
        self.scheduler.set_timesteps(self.n_timesteps)

        latent_path = get_latents_dir(latent_path, self.model_key)
        latent_path = None if not self.check_latent_exists(latent_path) else latent_path

        print(f"[INFO] latent path {'not found, generating new latents.' if latent_path is None else f'found at {latent_path}'}")

        self.rng = [torch.Generator(device=self.device).manual_seed(int(self.seed))] * len(frame_ids)
        self.latent_path = latent_path
        self.frame_ids = frame_ids
        self.prepare_data(latent_path, frame_ids)

        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")
        self.frames = self.frames.to(device=self.vae.device, dtype=self.vae.dtype)

        for edit_name, edit_prompt in self.prompt.items():
            # === Start time and memory tracking ===
            torch.cuda.reset_peak_memory_stats()
            start_time = datetime.now()

            # === Handle null prompt (upsample scenario) ===
            if edit_prompt is None:
                edit_prompt = self._handle_missing_prompt(output_path)
                self.config.generation.prompt[edit_name] = edit_prompt

            print(f"[INFO] current prompt: {edit_prompt}")
            concat_conds, prompt_embeds, prompt_embeds_t = self._prepare_prompts_and_conditions(edit_prompt)

            # === Sample and decode ===
            clean_latent = self.ddim_sample(self.init_noise, prompt_embeds, prompt_embeds_t, concat_conds)
            clean_frames = self.decode_latents_batch(clean_latent)

            if self.apply_opt:
                torch.cuda.empty_cache()
                _, _, _, _, past_flows, mask_bwds = self.data_parser.load_data(frame_ids)
                self.dataset = OptDataset(
                    clean_frames.to(past_flows.dtype),
                    past_flows,
                    mask_bwds,
                    device=self.device
                )

                clean_frames, loss_list_exposure = self.exposure_align()
                clean_frames, loss_list = self.unique_tensor_optimization()

            # === Record time & memory ===
            end_time = datetime.now()
            max_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            self.config.max_memory_allocated = max(max_memory, self.config.max_memory_allocated)
            self.config.total_time += (end_time - start_time).total_seconds()
            self.config.sec_per_frame = self.config.total_time / len(frame_ids)

            # === Save ===
            opt_suffix = f"_opt{'_upsampled' if edit_prompt is None else ''}"
            save_name = f"lmr_{self.local_merge_ratio}_gmr_{self.global_merge_ratio}_alpha_t_{self.alpha_t}{opt_suffix}_{edit_name}"
            cur_output_path = os.path.join(output_path, save_name)

            save_config(self.config, cur_output_path, gene=True)
            save_video(clean_frames, cur_output_path, save_frame=self.save_frame, fps=self.data_parser.fps, gif=False)

            # Save GT if not already saved
            gt_path = os.path.join(cur_output_path, "gt")
            if not os.path.exists(gt_path) or len(os.listdir(gt_path)) != len(self.frames):
                self.frames = self.frames.to(device=clean_frames.device, dtype=clean_frames.dtype)
                save_video(self.frames, cur_output_path, save_frame=False, post_fix="_gt", fps=self.data_parser.fps, gif=False)

            # Save losses if applicable
            if self.apply_opt:
                save_loss_curve(loss_list_exposure, cur_output_path, "loss_exposure")
                save_loss_curve(loss_list, cur_output_path, "loss_unique_tensor")
