import torch
from diffusers import DPMSolverMultistepScheduler

from cfgs.workflow.text2img import *

from rainbowneko.parser import neko_cfg
from rainbowneko.utils import KeyMapper
from rainbowneko.infer import Actions, PrepareAction, LambdaAction, LoadModelAction, LoopAction
from rainbowneko.ckpt_manager import auto_ckpt_loader, NekoLoader, LocalCkptSource

from hcpdiff.diffusion.sampler import DiffusersSampler
from hcpdiff.ckpt_manager import DiffusersSDXLFormat
from hcpdiff.models.compose import SDXLTextEncoder, SDXLTokenizer

prompt = ('best quality, reality-shot, realism, realistic photography of a an elaborate robot made out of butterfly pea flowers, magical fairytale landscape, fantasy style art, tropical color schemed theme, intr')
negative_prompt = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality,'
                   ' normal quality, jpeg artifacts, signature, watermark, username, blurry')

pretrained_model='stabilityai/stable-diffusion-xl-base-1.0'

SAVE_ROOT = 'output_sdxl/'
BATCH_SIZE=4
SEED = 42
N_STEPS = 20
WIDTH = 1024
HEIGHT = 1024
GUIDANCE_SCALE = 7.0

@neko_cfg
def build_model(pretrained_model='ckpts/any5', noise_sampler=Diffusers_SD.dpmpp_2m_karras) -> Actions:
    return Actions([
        PrepareAction(device='cuda', dtype=torch.float16),
        ## Easy config
        BuildModelsAction(
            model_loader=NekoLoader(
                source=LocalCkptSource(),
                format=DiffusersSDXLFormat()
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
        text(prompt=prompt, negative_prompt=negative_prompt, bs=BATCH_SIZE, TE_final_norm=False),
        config_diffusion(seed=SEED, N_steps=N_STEPS, width=WIDTH, height=HEIGHT),
        diffusion(guidance_scale=GUIDANCE_SCALE),
        decode(save_root=SAVE_ROOT)
    ]))