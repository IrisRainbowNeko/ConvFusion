from torch import nn
import torch
import numpy as np
from transformers.models.llama.modeling_llama import LlamaRMSNorm

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.LayerNorm(in_ch),
            nn.SiLU(),
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
        )

        if in_ch!=out_ch:
            self.skip = nn.Linear(in_ch, out_ch)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.skip(x)

class SimpleMLP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        norm = LlamaRMSNorm(out_ch)
        norm.weight.data.fill_(1.8)

        self.proj = nn.Sequential(
            nn.Linear(in_ch, in_ch, bias=False),
            nn.SiLU(),
            nn.Linear(in_ch, out_ch, bias=False),
            norm,
        )

    def forward(self, x):
        return self.proj(x)

class NormMLP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, out_ch, bias=False),
        )

    def forward(self, x):
        return self.proj(x)

class SimpleMLPLora(nn.Module):
    def __init__(self, in_ch, out_ch, rank=32):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_ch, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, out_ch, bias=False),
        )

    def forward(self, x):
        return self.proj(x)

class FFNLora(nn.Module):
    def __init__(self, in_ch, out_ch, rank=32):
        super().__init__()

        self.norm = nn.LayerNorm(in_ch, elementwise_affine=True)
        self.proj = nn.Sequential(
            nn.Linear(in_ch, rank),
            nn.SiLU(),
            nn.Linear(rank, out_ch),
        )

    def forward(self, x):
        x = self.norm(x)
        return self.proj(x) + x

class FeatScale(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(*shape)*np.log(1.))

    def forward(self, x):
        #return x*self.alpha
        return x*self.alpha.exp()