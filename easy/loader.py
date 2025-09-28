import torch
from copy import deepcopy
from hcpdiff.easy import SD15_auto_loader, PixArt_auto_loader, SDXL_auto_loader
from models import SD15ConvModel, SDXLConvModel, PixArtConv2DModel
from diffusers import UNet2DConditionModel
from pathlib import Path

def SD15_dist_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    models = SD15_auto_loader(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision, dtype=dtype, **kwargs)
    models['denoiser_T'] = models['denoiser']

    ckpt_path = Path(ckpt_path)
    # 为什么始终会判断为false? -- 全部都是针对的本地吗?
    is_hf_struct = (unet_path := ckpt_path/'unet').is_dir()
    print("--------------------------------------------------")
    print(f"SD15_dist_auto_loader loads from {ckpt_path}")
    print("--------------------------------------------------") 
    if ckpt_path.exists() and is_hf_struct:
        denoiser_conv = SD15ConvModel.from_pretrained(str(unet_path), revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False)
    elif ckpt_path.exists():
        denoiser_conv = SD15ConvModel.from_single_file(str(ckpt_path), revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False)
    else:
        denoiser_conv = SD15ConvModel.from_pretrained(str(ckpt_path), subfolder='unet', revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False)

    models['denoiser'] = denoiser_conv
    return models

def PixArt_dist_auto_loader(ckpt_path, denoiser=None, denoiser_conv=None, TE=None, vae=None, noise_sampler=None,
                     tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    models = PixArt_auto_loader(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision, dtype=dtype, **kwargs)
    models['denoiser_T'] = models['denoiser']

    ckpt_path = Path(ckpt_path)
    denoiser_conv = denoiser_conv or PixArtConv2DModel.from_pretrained(str(ckpt_path/'transformer'), revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False)
    models['denoiser'] = denoiser_conv

    return models

def SDXL_dist_auto_loader(ckpt_path, denoiser=None, TE=None, vae=None, noise_sampler=None,
                         tokenizer=None, revision=None, dtype=torch.float32, **kwargs):
    models = SDXL_auto_loader(ckpt_path, denoiser=denoiser, TE=TE, vae=vae, noise_sampler=noise_sampler, tokenizer=tokenizer, revision=revision, dtype=dtype, **kwargs)
    models['denoiser_T'] = deepcopy(models['denoiser'])

    ckpt_path = Path(ckpt_path)
    if (unet_path := ckpt_path/'unet').is_dir():
        denoiser_conv = SDXLConvModel.from_pretrained(unet_path, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False, device_map=None)
    else:
        denoiser_conv = SDXLConvModel.from_single_file(ckpt_path, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=False, device_map=None)

    models['denoiser'] = denoiser_conv

    return models