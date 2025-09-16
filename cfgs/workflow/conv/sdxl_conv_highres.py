import torch
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL

from models import SDXLConvModel

from cfgs.workflow.text2img import *

from rainbowneko.parser import neko_cfg
from rainbowneko.utils import KeyMapper
from rainbowneko.infer import Actions, PrepareAction, LambdaAction, LoadModelAction, LoopAction
from rainbowneko.ckpt_manager import auto_ckpt_loader, NekoLoader, LocalCkptSource

from hcpdiff.diffusion.sampler import DiffusersSampler
from hcpdiff.ckpt_manager import DiffusersSDXLFormat
from hcpdiff.models.compose import SDXLTextEncoder, SDXLTokenizer
from hcpdiff.workflow import LatentResizeAction, ImageResizeAction, EncodeAction
from hcpdiff.easy import SDXL_auto_loader, Diffusers_SD

prompt = ('best quality, reality-shot, realism, realistic photography of a an elaborate robot made out of butterfly pea flowers, magical fairytale landscape, fantasy style art, tropical color schemed theme, intr')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

pretrained_model='stabilityai/stable-diffusion-xl-base-1.0'
# vae_pretrained_model_name_or_path='/mnt/data1/pretrained_models/sdxl-vae-fp16-fix'

CKPT_PATH = 'ConvFusion-sdxl-xxx.safetensors'
SAVE_ROOT = 'output_sdxl/'
BATCH_SIZE=4
SEED = 42
N_STEPS = 20
WIDTH = 1024
HEIGHT = 1024
GUIDANCE_SCALE = 7.0

@neko_cfg
def build_model(pretrained_model=pretrained_model, noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Easy config
        BuildModelsAction(
            model_loader=SDXL_auto_loader(
                _partial_=True, 
                ckpt_path=pretrained_model,
                denoiser=SDXLConvModel.from_pretrained(
                    pretrained_model,
                    subfolder='unet',
                    low_cpu_mem_usage=False,
                    init_weights=True,
                ),
                # vae=AutoencoderKL.from_pretrained(vae_pretrained_model_name_or_path, torch_dtype=torch.float16),
                ## for model in single file format (xxx.ckpt or xxx.safetensors)
                # denoiser=SDXLConvModel.from_single_file(
                #     pretrained_model,
                #     low_cpu_mem_usage=False,
                #     init_weights=True,
                # ),
                noise_sampler=noise_sampler,
            )
        ),
        LambdaAction(
            lambda denoiser, vae, TE, device, dtype, **states: {
                'TE': TE.to(device=device, dtype=dtype),
                'denoiser': denoiser.to(device='cpu', dtype=dtype),
                'vae': vae.to(device='cpu', dtype=dtype),
            }
        ),
        LoadModelAction(
            cfg=dict(
                model=auto_ckpt_loader(
                    path=CKPT_PATH,
                ),
            ), 
            key_map_in=KeyMapper(key_map=('denoiser -> model',), move_mode=True),
        ),
    ])

@neko_cfg
def optimize_model(amp=torch.float16) -> Actions:
    return Actions([
        PrepareDiffusionAction(amp=amp, model_offload=True),
        XformersEnableAction(),
        VaeOptimizeAction(slicing=True),
    ])

# @neko_cfg
# def resize(width=1024, height=1024):
#     return Actions([
#         LatentResizeAction(width=width, height=height)
#     ])

# @neko_cfg
# def config_highres(seed=42, N_steps=20, strength=0.6, width=1024, height=1024):
#     return Actions([
#         SeedAction(seed),
#         MakeTimestepsAction(N_steps=N_steps, strength=strength),
#         MakeLatentAction(width=width, height=height)
#     ])

@neko_cfg
def resize(width=1024, height=1024):
    return Actions([
        DecodeAction(),
        SaveImageAction(save_root='output_pipe/', image_type='webp'),
        ImageResizeAction(width=width, height=height, mode='lanczos'),
        EncodeAction(),
    ])

@neko_cfg
def config_highres(seed=42, N_steps=20, strength=0.6, width=1024, height=1024):
    return Actions([
        SeedAction(seed),
        MakeTimestepsAction(N_steps=N_steps, strength=strength),
        MakeLatentAction(width=width, height=height)
    ])

@neko_cfg
def make_cfg(pretrained_model=pretrained_model):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=BATCH_SIZE, TE_final_norm=False),
        config_diffusion(seed=SEED, N_steps=N_STEPS, width=WIDTH, height=HEIGHT),
        diffusion(guidance_scale=GUIDANCE_SCALE),
        # >>> highres fix >>>
        resize(width=WIDTH*2, height=HEIGHT*2),
        config_highres(seed=SEED, N_steps=N_STEPS, width=WIDTH*2, height=HEIGHT*2),
        diffusion(guidance_scale=GUIDANCE_SCALE),
        # <<< highres fix <<<
        decode(save_root=SAVE_ROOT)
    ]))