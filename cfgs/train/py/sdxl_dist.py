import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL
from bitsandbytes.optim import AdamW8bit

from models import SDXLDistWrapper, SDXLConvModel
from data.source.lmdb_text2img import LmdbText2ImageSource
from cfgs.train.py import train_base, tuning_base
from cfgs.train.py.dataset import base_dataset
from easy import SDXL_dist_auto_loader

from hcpdiff.diffusion.sampler import VPSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.evaluate import HCPPreviewer
from hcpdiff.data import TextImagePairDataset, StableDiffusionHandler
from hcpdiff.models.compose import SDXLTextEncoder, SDXLTokenizer
from hcpdiff.loss import DiffusionLossContainer

from rainbowneko.parser import CfgWDModelParser, neko_cfg, CfgWDPluginParser
from rainbowneko.ckpt_manager import ckpt_saver, LAYERS_TRAINABLE, NekoResumer, NekoModelLoader
from rainbowneko.loggers import CLILogger
from rainbowneko.data import RatioBucket
from rainbowneko.utils import ConstantLR
from rainbowneko.train.loss import LossGroup, LossContainer

from cfgs.workflow.conv import sdxl_conv

pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0'
# vae_pretrained_model_name_or_path='/mnt/data1/pretrained_models/sdxl-vae-fp16-fix'
data_root='/dataset/mjv5'

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/sdxl-conv-v4-lr3e-4',
        mixed_precision='fp16',

        model_part=CfgWDModelParser(
            [
                dict(
                    lr=3e-4,
                    layers=['re:^denoiser\..*transformer_blocks.*\.conv1$'],
                )
            ],  
            weight_decay=1e-2,
        ),

        ckpt_saver=dict(
            model=ckpt_saver(
                layers=LAYERS_TRAINABLE,
                target_module='denoiser',
            )
        ),

        train=dict(
            train_steps=200000,
            gradient_accumulation_steps=2,
            save_step=2000,

            # resume=NekoResumer(
            #     start_step=6000,
            #     loader=dict(
            #         model=NekoModelLoader(
            #             path='exps/sdxl-conv-v4-lr3e-4/ckpts/model-6000.safetensors',
            #             target_module='denoiser'
            #         )
            #     )
            # ),

            optimizer=AdamW8bit(
                _partial_=True,
                weight_decay=0.03,
                betas=[0.9, 0.99],
            ),
            loss=LossGroup([
                # latent supervised loss
                DiffusionLossContainer(nn.MSELoss(reduction='none')),
                # latent distillation loss
                DiffusionLossContainer(nn.MSELoss(reduction='none'), key_map=('pred.model_pred -> 0','pred.model_pred_T -> 1')),
                # feature distillation loss
                LossContainer(lambda x:x, key_map=('pred.loss_feat -> 0',), weight=0.001),
            ]),
            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=500,
            )
        ),

        model=dict(
            name='SDXL_conv',
            wrapper=SDXLDistWrapper.from_pretrained(
                _partial_=True,
                models=SDXL_dist_auto_loader(_partial_=True, 
                    ckpt_path=pretrained_model_name_or_path,
                    # vae=AutoencoderKL.from_pretrained(vae_pretrained_model_name_or_path, torch_dtype=torch.float16),
                    low_cpu_mem_usage=False,
                    device_map=None
                ),
                # low_vram=True, # for VRAM<30G
            )
        ),

        logger=[
            CLILogger(
                _partial_=True,
                out_path='train.log',
                log_step=20,
            ),
        ],

        data_train=dict(
            dataset1=TextImagePairDataset(
                _partial_=True,
                batch_size=4,
                loss_weight=1.0,

                source=dict(
                    data_source1=LmdbText2ImageSource(
                        img_root=os.path.join(data_root, 'mjv5.lmdb'),
                        label_file=os.path.join(data_root, 'image_captions_prune.json'),
                        prompt_template='prompt_tuning_template/caption.txt'
                    )
                ),
                handler=StableDiffusionHandler(
                    erase=0.15,
                    bucket=RatioBucket
                ),
                bucket=RatioBucket.from_files(
                    target_area=1024*1024,
                    step_size=8,
                    num_bucket=6,
                    pre_build_bucket=os.path.join(data_root, 'mjv5-bucket-s1024-bs4-gpu1-step8.pkl')
                ),
            )
        ),

        evaluator=HCPPreviewer(
            _partial_=True,
            interval=2000,
            workflow=sdxl_conv.make_cfg(pretrained_model='${model.wrapper.models.ckpt_path}'),
        )
    )