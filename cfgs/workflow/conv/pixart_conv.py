import torch
from rainbowneko.infer import LoadModelAction, Actions, PrepareAction, LambdaAction
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader, NekoLoader, LocalCkptSource
from rainbowneko.utils import KeyMapper

from cfgs.workflow.text2img import *
from hcpdiff.workflow import BuildModelsAction
from hcpdiff.diffusion.sampler import DiffusersSampler
from hcpdiff.ckpt_manager import DiffusersPixArtFormat

from models import PixArtConv2DModel

prompt = 'stockholm city, Pixar style, by Tristan Eaton Stanley Artgerm and Tom Bagshaw, wrench_elven_arch, outdoors, indoors, tree, leaves, forest, water'
negative_prompt = 'nsfw,ng_deepnegative_v1_75t,badhandv4 (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)),watermark, (monotone), (multiple angles), 3D, low quality, lowres, mutated hands and fingers, long body, mutation, poorly drawn, black-white,'

pretrained_model='PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'
pretrained_model_512='PixArt-alpha/PixArt-Sigma-XL-2-512-MS'

# CKPT_PATH = 'exps/pixart-conv-lr3e-4/ckpts/model-36000.safetensors'
CKPT_PATH = '/mnt/data_center/data2/models/ConvFusion/unet-180000.safetensors'
SAVE_ROOT = 'output_pixart/'
BATCH_SIZE=4
SEED = 42
N_STEPS = 30
GUIDANCE_SCALE = 4.5

@neko_cfg
def build_model(pretrained_model=pretrained_model, dtype=torch.bfloat16) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=dtype),
        ## Easy config
        BuildModelsAction(
            model_loader=NekoLoader(
                source=LocalCkptSource(),
                format=DiffusersPixArtFormat()
            ).load(
                _partial_=True, 
                path=pretrained_model,
                denoiser=PixArtConv2DModel.from_pretrained(
                    pretrained_model_512,
                    subfolder='transformer',
                    low_cpu_mem_usage=False,
                    init_weights=True,
                ),
                noise_sampler=DiffusersSampler(
                    DPMSolverMultistepScheduler(
                        beta_start=0.0001,
                        beta_end=0.02,
                        beta_schedule='linear',
                        algorithm_type='dpmsolver++',
                        use_karras_sigmas=True,
                    )
                ),
            )
        ),
        LambdaAction(
            lambda denoiser, vae, TE, device, dtype, **states: {
                'TE': TE.to(device=device, dtype=dtype),
                'denoiser': denoiser.to(device='cpu'),
                'vae': vae.to(device='cpu'),
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
def make_cfg(pretrained_model=pretrained_model, width=512, height=512):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        optimize_model(amp=torch.bfloat16),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=BATCH_SIZE, TE_final_norm=False, layer_skip=0),
        config_diffusion(seed=SEED, N_steps=N_STEPS, width=width, height=height),
        diffusion(guidance_scale=GUIDANCE_SCALE),
        decode(save_root=SAVE_ROOT)
    ]))
