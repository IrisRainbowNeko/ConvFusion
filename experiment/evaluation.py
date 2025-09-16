from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional
import torch
import json
from functools import wraps
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.cuda import Event
from icecream import ic
from tqdm import tqdm
from torchanalyzer import ModelTimeMemAnalyzer, TorchViser
from hcpdiff import Visualizer
from diffusers.models.attention import BasicTransformerBlock
import itertools
from torch.utils.data import Dataset , DataLoader

import sys
sys.path.append('/mnt/data1/cxzhou/Cnn_as_Attn/conv-diff_triton')
from experiment.fd_dinov2.fd_score import calculate_fd_given_paths
from experiment.IS_dinov2.ds_dino import DINOScore
from models.block_conv import BasicConvFormerBlock

from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.multimodal.clip_score import CLIPScore

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = {
    'coco': [
        '/mnt/dataset/MSCOCO2014/annotations/captions_val2014.json', 
        '/mnt/data1/cxzhou/Cnn_as_Attn/CompareWith/DiTFastAttn/data/real_images_coco_30k'
    ],
    'laion': [
        '/mnt/dataset/10000_of_LAION/LAION_10000.json', 
        '/mnt/dataset/10000_of_LAION/imgs'
    ]
}
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"
TIME_RECORDER = []


class LAIONDataset(Dataset):
    def __init__(self , ann_path):
        assert ann_path.endswith('')
        with open(ann_path, 'r') as f:
            anno = json.load(f)

        self.filename_list = [d['img_path'].split('/')[-1] for d in anno]
        self.caption_list = [d["prompt"] for d in anno]

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self , idx):
        return self.filename_list[idx], self.caption_list[idx]

def collate_fn(batch):
    transposed = list(zip(*batch))
    return list(transposed[0]), list(transposed[1])

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def cuda_timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        if func.__name__ == 'forward':
            TIME_RECORDER.append(round(elapsed_time*1000, 4))
            # print(f"{RED}{func.__name__} took {elapsed_time*1000:.6f} ms{RESET}")
            # print(func.__module__)
        return result
    return wrapper


def add_timekeeper(module):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, BasicConvFormerBlock):
            if hasattr(sub_module, "conv1"):
                sub_module.conv1.forward = cuda_timing_decorator(sub_module.conv1.forward)
                # print(f"{GREEN}Added timing decorator to {name}.conv1{RESET}")
        elif isinstance(sub_module, BasicTransformerBlock):
            if hasattr(sub_module, "attn1"):
                sub_module.attn1.forward = cuda_timing_decorator(sub_module.attn1.forward)
                # print(f"{GREEN}Added timing decorator to {name}.attn1{RESET}")


def evaluate_quantitative_scores_text2img(
    pipe,
    data_name,
    n_images=10000,
    batchsize=1,
    seed=3,
    calc_infer_cost=False,
    num_inference_steps=20,
    reuse_generated=True,
    guidance_scale=4.5,
    output_dir='',
    device=None,
    **kwargs
):
    global TIME_RECORDER

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    # disable_torch_init()

    dataset = LAIONDataset(DATA_PATH[data_name][0])
    dataloader = DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=batchsize,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if not calc_infer_cost:
        results = {}
        inception = InceptionScore().to(device)
        dino = DINOScore().to(device)
        clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

        if os.path.exists(output_dir) and not reuse_generated:
            os.system(f"rm -rf {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    for batch in tqdm(dataloader):
        # prepare text prompts in batch
        # if data_name == 'coco':
        #     slice = anno["annotations"][index : index + batchsize]
        #     filename_list = [str(d["id"]).zfill(12)+'.jpg' for d in slice]
        #     caption_list = [d["caption"] for d in slice]
        # elif data_name == 'laion':
        #     slice = anno[index : index + batchsize]
        #     filename_list = [d['img_path'].split('/')[-1] for d in slice]
        #     caption_list = [d["prompt"] for d in slice]

        filename_list, caption_list = batch
        if calc_infer_cost:
            if isinstance(pipe, Visualizer):
                pipe.pipe.text_encoder.to(device)
                # pipe.pipe.text_encoder = pipe.pipe.text_encoder.to(torch.bfloat16)
                pipe.pipe.unet.to(device)
                add_timekeeper(pipe.pipe.unet)

                _ = pipe.vis_images(
                    prompt=caption_list,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=[""]*len(caption_list),
                    seeds=[seed]*len(caption_list),
                    **kwargs
                )
            else:
                add_timekeeper(pipe.unet)
                _ = pipe(
                    caption_list,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=[""]*len(caption_list),
                    guidance_scale=guidance_scale,
                )
        else:
            torch_images = []
            for filename in filename_list:
                image_file = os.path.join(output_dir, filename)
                if os.path.exists(image_file):
                    image = Image.open(image_file)
                    image_np = np.array(image)
                    torch_image = torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2)
                    torch_images.append(torch_image)

            if len(torch_images) > 0:
                torch_images = torch.cat(torch_images, dim=0)
                torch_images = torch.nn.functional.interpolate(
                    torch_images, size=(299, 299), mode="bilinear", align_corners=False
                ).to(device)

                inception.update(torch_images)
                dino.update(torch_images)
                clip.update(torch_images, caption_list[: len(torch_images)])
                continue
            elif isinstance(pipe, Visualizer):
                generated_imgs = pipe.vis_images(
                    prompt=caption_list,
                    negative_prompt=[""]*len(caption_list),
                    num_inference_steps=num_inference_steps,
                    seeds=[seed]*len(caption_list),
                    output_type="np",
                    **kwargs
                )
            else:
                generated_imgs = pipe(
                    caption_list,
                    output_type="np",
                    num_inference_steps=num_inference_steps,
                    negative_prompt=[""]*len(caption_list),
                    guidance_scale=guidance_scale,
                ).images
            
            img_tensor = torch.tensor(
                generated_imgs * 255, 
            ).byte().permute(0, 3, 1, 2).contiguous()
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)

            inception.update(img_tensor)
            dino.update(img_tensor)
            clip.update(img_tensor, caption_list)

            for idx, image in enumerate(generated_imgs):
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(os.path.join(
                    output_dir, filename_list[idx]
                ))

    if  int(os.getenv('WORLD_SIZE', '1')) > 1:
        ic('Generate finished ', n_images, batchsize)
        sys.exit(0)
    
    if calc_infer_cost:
        len_recorder = len(TIME_RECORDER)
        ic(len_recorder)
        TIME_RECORDER = TIME_RECORDER[-len_recorder//2:]
        mean_infer_cost = sum(TIME_RECORDER) / len(TIME_RECORDER)
        ic(round(mean_infer_cost, 4), 'milliseconds')
        sys.exit(0)

    IS = inception.compute()
    DS = dino.compute()
    CLIP = clip.compute()

    results["IS"] = IS
    results["DS"] = DS
    results["CLIP"] = CLIP
    ic(f"Inception Score: {IS}")
    ic(f"DINO Score: {DS}")
    ic(f"CLIP Score: {CLIP}")

    # fid_value = calculate_fid_given_paths(
    #     [DATA_PATH[data_name][1], output_dir],
    #     1,
    #     device,
    #     dims=2048,
    #     num_workers=0,
    # )
    fdd_value = calculate_fd_given_paths(
        [DATA_PATH[data_name][1], output_dir],
        1,
        device,
        dims=768,
        num_workers=0,
    )

    # results["FID"] = fid_value
    results["FDD"] = fdd_value
    # ic(f"FID: {fid_value}")
    ic(f"FDD: {fdd_value}")
    return results