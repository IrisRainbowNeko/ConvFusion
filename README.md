# Distilling Self-Attention into Convolutions ($\Delta$-ConvBlock)

Official PyTorch Implementation of "Can We Achieve Efficient Diffusion without Self-Attention? Distilling Self-Attention into Convolutions"

## Prepare and Installation

Clone the repo:
```bash
git clone 
cd conv-diff
```

Install python environment:
```bash
conda create -n conv-diff python=3.11
conda activate conv-diff
```

Install requirements:
```bash
pip install -r requirements.txt
```

## Pyramid ConvBlock

In `conv-diff/pyramid_dwconv`, files are all about our presented $\Delta$-ConvBlock, which is composed with depth-wise CNN and pooling. We use [triton](https://github.com/triton-lang/triton) to rewrite some acceleration operators for our block. You can have an overall view of this module in `conv-diff/pyramid_dwconv/conv_block.py` (It is a test demo of this module, the officially used code laying at `conv-diff/model/block_conv.py`).

Using $\Delta$-ConvBlock for you custom model:
```python
import torch
from pyramid_dwconv import PyramidConvBlockTriton

conv_block = PyramidConvBlockTriton(
    dim=512,
    kernel_size=9,
    scales=(1., 1/2, 1/3),
)

x = torch.rand(2,64,64,512) # [B,H,W,C]
y = conv_block(x) # [2,64,64,512]
```

### Triton Kernel

Using our depth-wise convolution triton kernel:
```python
from pyramid_dwconv import DepthwiseConv2DFunction

x = torch.rand(2,64,64,512) # [B,H,W,C]
w = torch.rand(9,9,512) # [K_h,K_w,C]
y = DepthwiseConv2DFunction.apply(x, w, stride=1, padding=4, dilation=1)
```

## Quick Inference

### With HCP-Diffusion (recommend)

**Step 1: Download the pretrained models**

🤗 [Huggingface](https://huggingface.co/7eu7d7/ConvFusion/tree/main)


**Step 2: Running inference command**

Our project is build with HCP-Diffusion and the model inference using HCP-Diffusion workflow:
```bash
CUDA_VISIBLE_DEVICES=0 hcp_run --cfg cfgs/workflow/conv/sd1_5_conv.py CKPT_PATH="path_to_model_ckpt"
```
where `cfgs/workflow/conv/sd1_5_conv.py` is the text-to-image workflow config file of $\Delta$-ConvFusion with SD1.5 structure.

For more models please refer to `infer.sh` and config files in `cfgs/workflow/conv/`.

### With diffusers

Coming Soon

## Training

**Step 1: Pretrained model download**
| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| SD1.5    | 🤗 [Huggingface](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)    🤖 [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)    | Stable Diffusion v1.5, supports 512P |
| DreamShaper*    | 🤗 [Huggingface](https://huggingface.co/Lykon/DreamShaper)    🤖 [ModelScope](https://www.modelscope.cn/models/MusePublic/DreamShaper_SD_1_5)    | Finetuned version of SD1.5 from community, supports 512P |
| SDXL   | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://www.modelscope.cn/models/MusePublic/47_ckpt_SD_XL)    | Stable Diffusion XL, supports 1K |
|  SDXL-VAE-FP16-Fix*  | 🤗 [Huggingface](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)    🤖 [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix)    | Modified from SDXL-VAE to run in fp16 precision without generating NaNs |
| Pixart-Sigma 512     | 🤗 [Huggingface](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS)     🤖 [ModelScope](https://www.modelscope.cn/models/fq980207/PixArt-Sigma/summary)     | PixArt-Sigma-XL-2-512-MS, supports 512P |
| Pixart-Sigma 1024     | 🤗 [Huggingface](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)     🤖 [ModelScope](https://www.modelscope.cn/models/fq980207/PixArt-Sigma/summary)     | PixArt-Sigma-XL-2-1024-MS, supports 1K |

**Step 2: Training data download**
| Dataset              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| midjourney-v5-202304    | 🤗 [Huggingface](https://huggingface.co/datasets/JohnTeddy3/midjourney-v5-202304)    | JohnTeddy3/midjourney-v5-202304 |
| midjourney-v5-202304-lmdb    | 🤗 [Huggingface](https://huggingface.co/datasets/7eu7d7/midjourney-v5-2M)    | we store it in lmdb format for efficient loading |

**Step 3: Train with HCP-Diffusion**

Train $\Delta$-ConvFusion with SD1.5 structure:
```bash
hcp_train --cfg cfgs/train/py/sd1_5_dist.py
# or
bash train.sh
```

more details please refer to [HCP-Diffusion](https://github.com/IrisRainbowNeko/HCP-Diffusion).

> This project is establshed on `HCP-Diffusion` and `RainbowNeko Engine`. You can get some introduction and guidance below.
> https://hcpdiff.readthedocs.io/en/latest/
> https://rainbownekoengine.readthedocs.io/en/latest/


## Config Files

`HCP-Diffusion` and `RainbowNeko Engine` support 'yaml' and 'python' format config files. 

We place all training config files in `cfgs/train/yaml` and `cfgs/train/py`, you can use it follow `train.sh`.

For inference, you can use config files in `cfgs/workflow` following commands in `infer.sh`. 
> Note that, `HCP-Diffusion` and `RainbowNeko Engine` support previewing training effects by inference during training. It is achieved by given inference config files in training config files. (Detail about preview stage to see `hcpdiff.evaluate.HCPPreviewer`)

### Quick Use
Taking `cfgs/train/py/sd1_5_dist.py` as an example, you need to change `pretrained_model_name_or_path` and `data_root`.

After modified training or inference config files, you can run with 
```bash
hcp_run --cfg cfgs/workflow/conv/sd1_5_conv.py
```
More command examples can be found in `train.sh` and `infer.sh`