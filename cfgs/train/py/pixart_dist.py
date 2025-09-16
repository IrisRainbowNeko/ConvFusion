import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from diffusers import AutoencoderKL, PixArtTransformer2DModel
from transformers import T5EncoderModel, AutoTokenizer

from models import PixArtDistWrapper, PixArtConv2DModel
from data.source.lmdb_text2img import LmdbText2ImageSource
from cfgs.train.py import train_base, tuning_base
from cfgs.train.py.dataset import base_dataset
from cfgs.workflow.conv import pixart_conv
from easy import PixArt_dist_auto_loader

from hcpdiff.diffusion.sampler import VPSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.evaluate import HCPPreviewer
from hcpdiff.data import TextImagePairDataset, StableDiffusionHandler
from hcpdiff.loss import DiffusionLossContainer

from rainbowneko.parser import CfgWDModelParser, neko_cfg, CfgWDPluginParser
from rainbowneko.ckpt_manager import ckpt_saver, LAYERS_TRAINABLE, NekoResumer, NekoModelLoader
from rainbowneko.loggers import CLILogger
from rainbowneko.data import RatioBucket
from rainbowneko.utils import ConstantLR
from rainbowneko.train.loss import LossGroup, LossContainer

pretrained_model_512='PixArt-alpha/PixArt-Sigma-XL-2-512-MS'
pretrained_model= 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'
data_root='/data_center/dataset/mjv5'

@neko_cfg
def make_cfg():
    return dict(
        _base_=[base_dataset, train_base, tuning_base],
        exp_dir=f'exps/pixart-conv-lr3e-4',
        mixed_precision='bf16',

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
            gradient_accumulation_steps=1,
            save_step=2000,

            # resume=NekoResumer(
            #     start_step=36000,
            #     loader=dict(
            #         model=NekoModelLoader(
            #             path='exps/pixart-conv-lr3e-4/ckpts/model-36000.safetensors',
            #             target_module='denoiser'
            #         )
            #     )
            # ),

            optimizer=AdamW(
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
            name='Pixart_conv',
            wrapper=PixArtDistWrapper.from_pretrained(
                _partial_=True,
                models=PixArt_dist_auto_loader(
                    _partial_=True,
                    ckpt_path=pretrained_model,

                    # for pixart-512
                    # denoiser=PixArtTransformer2DModel.from_pretrained(pretrained_model_512, low_cpu_mem_usage=False),
                    # denoiser_conv=PixArtConv2DModel.from_pretrained(pretrained_model_512, low_cpu_mem_usage=False),
                ),
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
                batch_size=16,
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
                    target_area=512*512,
                    step_size=16,
                    num_bucket=6,
                    pre_build_bucket=os.path.join(data_root, 'mjv5-bucket-s512-bs16-gpu1-step16.pkl')
                ),
            )
        ),

        evaluator=HCPPreviewer(
            _partial_=True,
            interval=2000,
            workflow=pixart_conv.make_cfg(pretrained_model='${model.wrapper.models.ckpt_path}'),
            # workflow=pixart_conv.make_cfg(pretrained_model='${model.wrapper.models.ckpt_path}', width=1024, height=1024), # for PixArt-1024
        )
    )