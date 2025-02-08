import os
import math
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from transformers import logging

from plugin.VidToMe.utils import CONTROLNET_DICT
from plugin.VidToMe.utils import load_config, save_config
from plugin.VidToMe.utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from plugin.VidToMe.utils import control_preprocess, prepare_control, load_latent, load_video, prepare_depth, save_video, save_frames
from plugin.VidToMe.utils import register_time, register_attention_control, register_conv_control

from plugin.VidToMe import vidtome

from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
    run_chat_completion,
)

# suppress partial model loading warning
logging.set_verbosity_error()

class Inverter(nn.Module):
    def __init__(self, vae, pipe, scheduler, config):
        super().__init__()

        self.device = config.device
        self.seed = config.seed
        self.use_depth = config.sd_version == "depth"
        self.model_key = config.model_key

        self.config = config
        inv_config = config.inversion

        float_precision = inv_config.float_precision if "float_precision" in inv_config else config.float_precision
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
        num_frames = len(list(range(*config.generation.frame_range)))
        self.rng = [torch.Generator(device=self.device).manual_seed(int(self.seed))] * num_frames

        self.control = inv_config.control
        if self.control != "none":
            self.controlnet = pipe.controlnet

        self.controlnet_scale = inv_config.control_scale

        # scheduler.set_timesteps(inv_config.save_steps)
        self.timesteps_to_save = inv_config.save_steps
        scheduler.set_timesteps(inv_config.steps)

        self.scheduler = scheduler

        self.prompt=inv_config.prompt
        self.recon=inv_config.recon
        self.save_latents=inv_config.save_intermediate
        self.use_blip=inv_config.use_blip
        self.steps=inv_config.steps
        self.batch_size = inv_config.batch_size
        self.force = inv_config.force

        self.n_frames = inv_config.n_frames
        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir

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
    def encode_prompt_pair(self, positive_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        # uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        # uc_len = float(len(uc))
        max_count = max(c_len, 0)
        c_repeat = int(math.ceil(max_count / c_len))
        # uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), 0)

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        # uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        # uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c

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

    @torch.no_grad()
    def ddim_inversion(self, x, conds, save_path):
        print("[INFO] start DDIM Inversion!")
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds, timesteps[i], concat_conds=x[batch], batch_idx=batch)
                    noises += [noise]
                noises = torch.cat(noises)
                
                # x = self.scheduler.step(noises, t, x, generator=self.rng, return_dict=False)[0]
                x = self.pred_next_x(x, noises, t, i, inversion=True)

                if self.save_latents and t in self.timesteps_to_save:
                    torch.save(x, os.path.join(
                        save_path, f'noisy_latents_{t}.pt'))

        # Save inverted noise latents
        pth = os.path.join(save_path, f'noisy_latents_{t}.pt')
        torch.save(x, pth)
        print(f"[INFO] inverted latent saved to: {pth}")
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x, conds):
        print("[INFO] reconstructing frames...")
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds[batch], t, batch_idx=batch)
                    noises += [noise]
                noises = torch.cat(noises)
                x = self.pred_next_x(x, noises, t, i, inversion=False)
        return x

    @torch.no_grad()
    def pred_noise(self, x, cond, t, concat_conds, batch_idx=None):

        flen = len(x)
        text_embed_input = cond.repeat_interleave(flen, dim=0)

        # For sd-depth model
        if self.use_depth:
            depth = self.depths
            if batch_idx is not None:
                depth = depth[batch_idx]
            x = torch.cat([x, depth.to(x)], dim=1)

        kwargs = dict()
        # Compute controlnet outputs
        if self.control != "none":
            if batch_idx is None:
                controlnet_cond = self.controlnet_images
            else:
                controlnet_cond = self.controlnet_images[batch_idx]
            controlnet_kwargs = get_controlnet_kwargs(self.controlnet, x, cond, t, controlnet_cond, self.controlnet_scale)
            kwargs.update(controlnet_kwargs)
 
        eps = self.unet(
            x, 
            t, 
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={'concat_conds': concat_conds},
            **kwargs).sample
        return eps

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

    @torch.no_grad()
    def prepare_cond(self, prompts, n_frames):
        if isinstance(prompts, str):
            prompts = [prompts] * n_frames
            cond = self.get_text_embeds(prompts[0])
            conds = torch.cat([cond] * n_frames)
        elif isinstance(prompts, list):
            cond_ls = []
            for prompt in prompts:
                cond = self.get_text_embeds(prompt)
                cond_ls += [cond]
            conds = torch.cat(cond_ls)
        return conds, prompts
    
    def check_latent_exists(self, save_path):
        save_timesteps = [self.scheduler.timesteps[0]]
        if self.save_latents:
            save_timesteps += self.timesteps_to_save
        for ts in save_timesteps:
            latent_path = os.path.join(
                save_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(latent_path):
                return False
        return True


    @torch.no_grad()
    def __call__(self, data_path, save_path):
        # self.scheduler.set_timesteps(self.steps)
        save_path = get_latents_dir(save_path, self.model_key)
        os.makedirs(save_path, exist_ok = True)
        if self.check_latent_exists(save_path) and not self.force:
            print(f"[INFO] inverted latents exist at: {save_path}. Skip inversion! Set 'inversion.force: True' to invert again.")
            return

        frames = load_video(data_path, self.frame_height, self.frame_width, device=self.device, base=8)
        frames = frames.to(device=self.vae.device, dtype=self.vae.dtype)

        frame_ids = list(range(len(frames)))
        if self.n_frames is not None:
            frame_ids = frame_ids[:self.n_frames]
        frames = frames[frame_ids]

        if self.use_depth:
            self.depths = prepare_depth(self.pipe, frames, frame_ids, self.work_dir)
        if isinstance(self.prompt, str):
            prompts = [self.prompt] * len(frames)
        conds = self.encode_prompt_pair(positive_prompt=self.prompt)
        with open(os.path.join(save_path, 'inversion_prompts.txt'), 'w') as f:
            f.write('\n'.join(prompts))

        if self.control != "none":
            images = control_preprocess(
                frames, self.control)
            self.controlnet_images = images.to(self.device)

        latents = self.encode_imgs_batch(frames)
        torch.cuda.empty_cache()
        print(f"[INFO] clean latents shape: {latents.shape}")

        inverted_x = self.ddim_inversion(latents, conds, save_path)
        save_config(self.config, save_path, inv = True)
        if self.recon:
            latent_reconstruction = self.ddim_sample(inverted_x, conds)

            torch.cuda.empty_cache()
            recon_frames = self.decode_latents_batch(
                latent_reconstruction)

            recon_save_path = os.path.join(save_path, 'recon_frames')
            save_frames(recon_frames, recon_save_path, frame_ids = frame_ids)