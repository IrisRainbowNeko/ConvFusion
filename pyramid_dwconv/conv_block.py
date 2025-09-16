import torch
from torch import nn
from torch.nn import functional as F

from pyramid_dwconv.dwconv_triton import OptimizedDepthwiseConv2d
from pyramid_dwconv.up_triton import OptimizedBilinearInterp2d
from pyramid_dwconv.avgpool_triton import OptimizedAdaptiveAvgPool2d

class SimpleGateTriton(nn.Module):
    def __init__(self, dim: int, kernel_size=13, scale_factor=1., padding=None, bias=False, simple=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale = (dim/2)**-0.5
        padding = kernel_size // 2 if padding is None else padding
        self.conv_spatial = OptimizedDepthwiseConv2d(dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.simple = simple

    def forward(self, x):
        B, H, W, C = x.shape
        if self.scale_factor != 1.:
            x = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (int(H*self.scale_factor), int(W*self.scale_factor))).permute(0, 2, 3, 1)
            # x = OptimizedAdaptiveAvgPool2d((int(H*self.scale_factor), int(W*self.scale_factor)))(x)
        x = self.conv_spatial(x)
        x1, x2 = x.chunk(2, dim=-1)
        if self.simple:
            x = x1 * (x2/self.scale)
        else:
            x = x1 * torch.sigmoid(x2) # PixArt used old setting
        if self.scale_factor != 1.:
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(H,W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            # x = OptimizedBilinearInterp2d((H,W))(x)
        return x


class PyramidConvBlockTriton(nn.Module):
    def __init__(self, dim: int, bias=True, out_bias=True, kernel_size=13, padding=None, scales=(1.,1/2), inter_ratio=0.5, simple=True):
        super().__init__()

        inter_dim = int(dim*inter_ratio)
        self.conv_up = nn.Linear(dim, inter_dim, bias=bias)
        self.gates = nn.ModuleList([SimpleGateTriton(inter_dim, kernel_size=kernel_size, scale_factor=scale, padding=padding, bias=bias, simple=simple) for scale in scales])
        self.conv_out = nn.Linear(inter_dim//2*len(scales), dim, bias=out_bias)

        self.conv_pool = nn.Linear(dim, dim, bias=out_bias)

    def forward(self, x):
        x_conv = self.conv_up(x)

        x_conv_list = []
        for gate in self.gates:
            x_conv_list.append(gate(x_conv))

        x_conv = torch.cat(x_conv_list, dim=-1)
        x_conv = self.conv_out(x_conv)

        x_pool = x.mean(dim=(1,2), keepdim=True)
        x_pool = self.conv_pool(x_pool)

        return x_conv + x_pool


class SimpleGate(nn.Module):
    def __init__(self, dim: int, kernel_size=13, scale_factor=1., padding=None, bias=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale = (dim/2)**-0.5
        padding = kernel_size // 2 if padding is None else padding
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=padding,
                                      bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.scale_factor != 1.:
            x = F.adaptive_avg_pool2d(x, (int(H*self.scale_factor), int(W*self.scale_factor)))
        x = self.conv_spatial(x)
        x1, x2 = x.chunk(2, dim=1)
        # x = x1 * (x2/self.scale)
        x = x1 * torch.sigmoid(x2) # PixArt used old setting
        if self.scale_factor != 1.:
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        return x


class PyramidConvBlock(nn.Module):
    def __init__(self, dim: int, bias=False, out_bias=True, kernel_size=9, padding=None, scales=(1.,1/2), inter_ratio=0.5):
        super().__init__()

        inter_dim = int(dim*inter_ratio)
        self.conv_up = nn.Conv2d(dim, inter_dim, kernel_size=1, bias=bias)
        self.gates = nn.ModuleList([SimpleGate(inter_dim, kernel_size=kernel_size, scale_factor=scale, padding=padding, bias=bias) for scale in scales])
        self.conv_out = nn.Conv2d(inter_dim//2*len(scales), dim, kernel_size=1, bias=out_bias)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(dim, dim, kernel_size=1, bias=out_bias)

    def forward(self, x):
        x_conv = self.conv_up(x)

        x_conv_list = []
        for gate in self.gates:
            x_conv_list.append(gate(x_conv))

        x_conv = torch.cat(x_conv_list, dim=1)
        x_conv = self.conv_out(x_conv)

        x_pool = self.pool(x)
        x_pool = self.conv_pool(x_pool)

        return x_conv + x_pool
    
def benchmark(batch_size=8, height=224, width=224, channels=64, kernel_size=13, dtype=torch.float16):
    import time
    
    # Create input
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()
    
    # Create our implementation
    conv_triton = PyramidConvBlockTriton(
        dim=channels,
        kernel_size=kernel_size,
    ).cuda().to(dtype)

    out_triton = conv_triton(x)
    grad = torch.randn_like(out_triton)
    
    # Create PyTorch implementation (need to convert to BCHW format)
    x_torch = x.permute(0, 3, 1, 2).clone()  # BHWC -> BCHW
    grad_torch = grad.permute(0, 3, 1, 2)
    conv_torch = PyramidConvBlock(
        dim=channels,
        kernel_size=kernel_size,
    ).cuda().to(dtype)

    #conv_torch.weight.data = conv_triton.weight.permute(2,0,1).unsqueeze(1).data.clone()
    
    # Warmup
    for _ in range(10):
        y1 = conv_triton(x)
        torch.autograd.backward(y1, grad)
        y2 = conv_torch(x_torch)
        torch.autograd.backward(y2, grad_torch)
    
    torch.cuda.synchronize()
    
    # Test our implementation
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        y = conv_triton(x)
        torch.autograd.backward(y, grad)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # Test PyTorch implementation
    start = time.time()
    for _ in range(iterations):
        y = conv_torch(x_torch)
        torch.autograd.backward(y, grad_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # Convert results to milliseconds
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"Batch size={batch_size}, Height={height}, Width={width}, Channels={channels}, Kernel size={kernel_size}x{kernel_size}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"Speedup: {torch_ms/triton_ms:.2f}x")
    
    return triton_ms, torch_ms


if __name__ == "__main__":
    # Usage example test
    batch_size = 4
    height = 64
    width = 64
    channels = 1152
    benchmark(batch_size, height, width, channels, kernel_size=9)
    benchmark(batch_size, height, width, channels, kernel_size=13)
    benchmark(batch_size, height, width, channels, kernel_size=17)