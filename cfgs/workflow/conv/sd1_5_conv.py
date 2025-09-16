import torch
from rainbowneko.infer import LoadModelAction, Actions, PrepareAction
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import auto_ckpt_loader, NekoLoader, LocalCkptSource
from rainbowneko.utils import KeyMapper

from cfgs.workflow.text2img import *
from hcpdiff.workflow import BuildModelsAction
from hcpdiff.easy import SD15_auto_loader, Diffusers_SD

from models import SD15ConvModel

prompt = ('masterpiece, best quality, 1girl, cat ears, outside')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

pretrained_model='Lykon/DreamShaper'


CKPT_PATH = 'ConvFusion-sd15-132000.safetensors'
SAVE_ROOT = 'output_pipe/'
BATCH_SIZE=4
SEED = 42
N_STEPS = 20
WIDTH = 768
HEIGHT = 768
GUIDANCE_SCALE = 7.0

@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Easy config
        BuildModelsAction(
            model_loader=SD15_auto_loader(
                _partial_=True, 
                ckpt_path=pretrained_model,
                denoiser=SD15ConvModel.from_pretrained(
                    pretrained_model,
                    subfolder='unet',
                    low_cpu_mem_usage=False,
                    init_weights=True,
                ),
                ## for model in single file format (xxx.ckpt or xxx.safetensors)
                # denoiser=SD15ConvModel.from_single_file(
                #     pretrained_model,
                #     low_cpu_mem_usage=False,
                #     init_weights=True,
                # ),
                noise_sampler=noise_sampler,
            )
        ),
        LoadModelAction(
            cfg=dict(
                model=auto_ckpt_loader(
                    path="${CKPT_PATH}",
                ),
            ), 
            key_map_in=KeyMapper(key_map=('denoiser -> model',), move_mode=True),
        ),
    ])

@neko_cfg
def make_cfg(pretrained_model=pretrained_model):
    return dict(
        CKPT_PATH=CKPT_PATH,
        workflow=Actions(actions=[
            build_model(pretrained_model=pretrained_model),
            optimize_model(),
            text(prompt=prompt, negative_prompt=negative_prompt, bs=BATCH_SIZE),
            config_diffusion(seed=SEED, N_steps=N_STEPS, width=WIDTH, height=HEIGHT),
            diffusion(guidance_scale=GUIDANCE_SCALE),
            decode(save_root=SAVE_ROOT)
        ])
    )
