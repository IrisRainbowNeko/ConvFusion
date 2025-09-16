from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoPipelineForText2Image
)

import torch
import argparse
import os
import json
from typing import List
import numpy as np
from icecream import ic
from torch.cuda.amp import autocast

import sys
sys.path.append('/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton')
from experiment.evaluation import evaluate_quantitative_scores_text2img
# from visualizer import ConvVisualizer
from linfusion import LinFusion

from hcpdiff import Visualizer
from hcpdiff.utils.utils import load_config_with_cli, is_list, prepare_seed, pad_attn_bias
from hcpdiff.deprecated.cfg_converter import InferCFGConverter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_TYPE = torch.float16
MODEL_PATH = {
    'sd1_5': '/mnt/data1/pretrained_models/DreamShaper',
    'sdxl': '/mnt/data1/pretrained_models/stable-diffusion-xl-base-1.0',
    'convdiff_sd1_5': '/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton/cfgs/infer/t2i-sd1_5-conv.yaml',
    'convdiff_sdxl': '/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton/cfgs/infer/t2i-sdxl-conv.yaml',
    'linfusion_sd1_5': ['/mnt/data1/pretrained_models/DreamShaper-8', '/mnt/data1/pretrained_models/LinFusion/LinFusion-1-5'],
    'linfusion_sdxl': ['/mnt/data1/pretrained_models/stable-diffusion-xl-base-1.0', '/mnt/data1/pretrained_models/LinFusion/LinFusion-XL'],
    'convdiff_pixart512': '/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton/cfgs/infer/t2i-pixart-conv.yaml',
    'convdiff_pixart1024': '/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton/cfgs/infer/t2i-pixart-conv-1024.yaml',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sd1_5', 
        help='sd1_5, sdxl, convdiff_sd1_5, convdiff_sdxl, linfusion_sd1_5, \
        linfusion_sdxl, convdiff_pixart512, convdiff_pixart1024')
    parser.add_argument('--data_name', type=str, default='laion',
        help='laion, coco')
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--eval_n_images", type=int, default=10000)
    parser.add_argument("--eval_batchsize", type=int, default=1)
    parser.add_argument("--calc_infer_cost", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--output_dir", type=str, default="")
    args, cfg_args = parser.parse_known_args()

    model_paths = MODEL_PATH[args.model_name]
    infer_args = None

    scheduler = DPMSolverMultistepScheduler(
        beta_start=0.00085,       
        beta_end=0.012,           
        beta_schedule='scaled_linear',
        algorithm_type='dpmsolver++',
        use_karras_sigmas=True
    )

    if args.model_name in ['sd1_5', 'sdxl']:
        pipe = DiffusionPipeline.from_pretrained(
            model_paths,
            torch_dtype=WEIGHT_TYPE,
        ).to(DEVICE)
        pipe.scheduler = scheduler
    elif args.model_name in ['linfusion_sd1_5', 'linfusion_sdxl']:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_paths[0],
            torch_dtype=WEIGHT_TYPE,
        ).to(DEVICE)
        pipe.scheduler = scheduler
        _ = LinFusion.construct_for(
            pipe,
            pretrained_model_name_or_path=model_paths[1]
        ) # module mounted to origin pipe
    elif args.model_name in ['convdiff_sd1_5', 'convdiff_sdxl', 'convdiff_pixart512', 'convdiff_pixart1024']:
        cfgs = load_config_with_cli(model_paths, args_list=cfg_args)
        cfgs = InferCFGConverter().convert(cfgs)
        infer_args = cfgs.infer_args

        # pipe = ConvVisualizer(cfgs)
        pipe = Visualizer(cfgs)
        if 'sd' in args.model_name:
            pipe.pipe.scheduler = scheduler
    else:
        raise NotImplementedError()

    if infer_args is not None:
        infer_args.pop('guidance_scale', None) 
        infer_args.pop('num_inference_steps', None) 
        
        result = evaluate_quantitative_scores_text2img(
            pipe,
            args.data_name,
            n_images=args.eval_n_images,
            batchsize=args.eval_batchsize,
            seed=args.seed,
            calc_infer_cost=args.calc_infer_cost,
            num_inference_steps=args.n_steps,
            guidance_scale=args.cfg_scale,
            output_dir=args.output_dir,
            device=DEVICE,
            **infer_args
        )
    else:
        result = evaluate_quantitative_scores_text2img(
            pipe,
            args.data_name,
            n_images=args.eval_n_images,
            batchsize=args.eval_batchsize,
            seed=args.seed,
            calc_infer_cost=args.calc_infer_cost,
            num_inference_steps=args.n_steps,
            guidance_scale=args.cfg_scale,
            output_dir=args.output_dir,
            device=DEVICE,
        )
    
    print(result)
    with open(f"{args.output_dir}/results.txt", "a+") as f:
        f.write(
            f"{args}\n{result}\n\n"
        )

if __name__ == "__main__":
    main()
