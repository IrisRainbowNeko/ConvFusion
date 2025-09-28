import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer

from models import StableDiffusionDistWrapper, SD15ConvModel
from data.source.lmdb_text2img import LmdbText2ImageSource
from cfgs.train.py import train_base, tuning_base
from cfgs.train.py.dataset import base_dataset
from easy import SD15_dist_auto_loader

from hcpdiff.evaluate import HCPPreviewer
from hcpdiff.diffusion.sampler import VPSampler, DDPMDiscreteSigmaScheduler
from hcpdiff.data import TextImagePairDataset, StableDiffusionHandler
from hcpdiff.loss import DiffusionLossContainer

from rainbowneko.parser import CfgWDModelParser, neko_cfg, CfgWDPluginParser
from rainbowneko.ckpt_manager import ckpt_saver, LAYERS_TRAINABLE, NekoResumer, NekoModelLoader
from rainbowneko.loggers import CLILogger
from rainbowneko.data import RatioBucket
from rainbowneko.utils import ConstantLR
from rainbowneko.train.loss import LossGroup, LossContainer

from cfgs.workflow.conv import sd1_5_conv

from cfgs.train.py.gemma3_enc import Gemma3Encoder

pretrained_model_name_or_path= 'Lykon/DreamShaper'
data_root='/data_center/data2/dataset/mjv5'

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir=f'exps/sd1_5-conv-v4-lr3e-4',
        mixed_precision='fp16',
        
        CKPT_PATH='${exp_dir}/ckpts/model-2000.safetensors',

        model_part=CfgWDModelParser(
            [
                dict(
                    lr=3e-4,
                    layers=[r're:^denoiser\..*transformer_blocks.*\.conv1$'],
                ),
                
                dict(
                    lr=1e-4,
                    layers=[
                        r're:^TE(\.text_model)?\.(x0_proj|connector|score_proj|layer_ht)(\.|$)'
                    ],
                ),
            ],  
            weight_decay=1e-2,
        ),

        ckpt_saver=dict(
            model=ckpt_saver(
                layers=LAYERS_TRAINABLE,
                target_module='denoiser',
            ),
            
            te=ckpt_saver(
                layers=LAYERS_TRAINABLE,
                target_module='TE',
            )
        ),

        train=dict(
            train_steps=60000,
            gradient_accumulation_steps=1,
            save_step=2000,

            # resume=NekoResumer(
            #     start_step=6000,
            #     loader=dict(
            #         model=NekoModelLoader(
            #             path='exps/sd1_5-conv-v4-lr3e-4/ckpts/model-6000.safetensors',
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
            name='SD1_5_conv',
            wrapper=StableDiffusionDistWrapper.from_pretrained(
                _partial_=True,
                models=SD15_dist_auto_loader(
                    _partial_=True,
                    ckpt_path=pretrained_model_name_or_path,
                    
                    TE=Gemma3Encoder.from_pretrained(
                            pretrained_model_name_or_path='/data_center/data2/gemma-3-4b-it',
                            diffusion_dim=768,
                            with_noise=True,
                            ignore_mismatched_sizes=True,
                            device_map='auto',
                            offload_folder='offload',
                            low_cpu_mem_usage=True,
                        ),
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
                batch_size=1,
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
                    step_size=8,
                    num_bucket=6,
                    pre_build_bucket=os.path.join(data_root, 'mjv5-bucket-s512-bs16-gpu1-step8.pkl')
                ),
            )
        ),

        evaluator=HCPPreviewer(
            _partial_=True,
            interval=2000,
            workflow=sd1_5_conv.make_cfg(pretrained_model='${model.wrapper.models.ckpt_path}'), 
        )
    )