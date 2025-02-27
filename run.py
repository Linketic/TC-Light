import os
import math
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from transformers import logging

import safetensors.torch as sf
from torch.hub import download_url_to_file
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import DDIMInverseScheduler, DPMSolverMultistepInverseScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG

from plugin.VidToMe.utils import load_config, get_frame_ids, seed_everything

from invert import Inverter
from generate_geom import Generator

from utils.common_utils import instantiate_from_config
from plugin.VideoVAE.CV_VAE.models.modeling_vae import CVVAEModel

if __name__ == "__main__":
    config = load_config()
    # pipe, scheduler, model_key = init_model(
    #     config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
    # manually change the pipe and scheduler
    sd15_name = 'stablediffusionapi/realistic-vision-v51'
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    # Change UNet

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward


    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


    unet.forward = hooked_unet_forward

    # Load

    model_path = './models/iclight_sd15_fc.safetensors'

    if not os.path.exists(model_path):
        download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    keys = sd_origin.keys()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, keys

    # Device

    device = torch.device('cuda')
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    rmbg = rmbg.to(device=device, dtype=torch.float32)

    # SDP

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # Samplers

    dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )

    # Pipelines

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=dpmpp_2m_sde_karras_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
 
    config.model_key = None
    seed_everything(config.seed)

    # inversion = Inverter(vae, pipe, dpmpp_2m_sde_karras_scheduler_inv, config)
    # inversion(config.input_path, config.inversion.save_path)

    # vae = CVVAEModel.from_pretrained('models', subfolder="vae3d_v1-1")

    # from omegaconf import OmegaConf
    # vae = instantiate_from_config(OmegaConf.load("models/vidtok/vidtok_kl_noncausal_488_4chn.yaml").model)

    # vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX1.5-5B", subfolder="vae")
    # vae.enable_slicing()
    # vae.enable_tiling()

    # vae = vae.to(device=device, dtype=torch.bfloat16)

    generator = Generator(vae, pipe, dpmpp_2m_sde_karras_scheduler, config, video_vae=False)

    frame_ids = get_frame_ids(
        config.generation.frame_range, generator.data_parser.n_frames, config.generation.frame_ids)
    config.total_number_of_frames = len(frame_ids)

    generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
