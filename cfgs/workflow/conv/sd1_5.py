import torch
from rainbowneko.infer import Actions, PrepareAction
from rainbowneko.parser import neko_cfg
from rainbowneko.ckpt_manager import NekoLoader, LocalCkptSource

from cfgs.workflow.text2img import *
from hcpdiff.workflow import BuildModelsAction

prompt = ('masterpiece, best quality, 1girl, cat ears, outside')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

pretrained_model='Lykon/DreamShaper'

SAVE_ROOT = 'output_sd15/'
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
            model_loader=NekoLoader(
                source=LocalCkptSource(),
                format=DiffusersSD15Format()
            ).load(
                _partial_=True, 
                path=pretrained_model,
                noise_sampler=noise_sampler,
            )
        ),
    ])

@neko_cfg
def make_cfg(pretrained_model=pretrained_model):
    return dict(workflow=Actions(actions=[
        build_model(pretrained_model=pretrained_model),
        optimize_model(),
        text(prompt=prompt, negative_prompt=negative_prompt, bs=BATCH_SIZE),
        config_diffusion(seed=SEED, N_steps=N_STEPS, width=WIDTH, height=HEIGHT),
        diffusion(guidance_scale=GUIDANCE_SCALE),
        decode(save_root=SAVE_ROOT)
    ]))
