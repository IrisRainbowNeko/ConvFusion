import torch
import gc
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd


# Unified CUDA memory cleanup utility
def release_cuda_memory(*objects):
    # Delete Python references, trigger garbage collection and clear CUDA cache
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        # May not be available in some environments
        pass


# Forward kernel (NHWC, 2D grid by batch and channel tile, spatial dimension processed in streaming with BLOCK_N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_kernel_optimized(
        input_ptr, weight_ptr, output_ptr,
        batch, in_height, in_width, channels,
        out_height, out_width,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)  # (c-group, n-tile)
    pid_b = tl.program_id(1)  # batch id

    n_tiles = tl.cdiv(out_height * out_width, BLOCK_N)
    pid_cg = pid0 // n_tiles
    pid_nt = pid0 % n_tiles

    c_start = pid_cg * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    n_start = pid_nt * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (out_height * out_width)

    oh = offs_n // out_width
    ow = offs_n % out_width

    # Accumulator [BLOCK_N, BLOCK_C]
    acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)

    # Accumulate convolution kernel contributions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h - padding_h + kh * dilation_h
            iw = ow * stride_w - padding_w + kw * dilation_w
            valid = (ih >= 0) & (ih < in_height) & (iw >= 0) & (iw < in_width) & mask_n

            x_base = pid_b * (in_height * in_width * channels) + ih * (in_width * channels) + iw * channels
            x_ptrs = input_ptr + x_base[:, None] + offs_c[None, :]
            w_ptrs = weight_ptr + (kh * kernel_w + kw) * channels + offs_c

            m = valid[:, None] & mask_c[None, :]
            x = tl.load(x_ptrs, mask=m, other=0.0)
            w = tl.load(w_ptrs, mask=mask_c, other=0.0)
            acc += (x * w[None, :]).to(tl.float32)

    # Write back output [B, OH, OW, C]
    y_base = pid_b * (out_height * out_width * channels) + oh * (out_width * channels) + ow * channels
    y_ptrs = output_ptr + y_base[:, None] + offs_c[None, :]
    tl.store(y_ptrs, acc, mask=(mask_n[:, None] & mask_c[None, :]))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_backward_input_kernel(
        g_output_ptr, weight_ptr, g_input_ptr,
        batch, out_height, out_width, channels,
        in_height, in_width,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)  # (c-group, n_in-tile)
    pid_b = tl.program_id(1)  # batch id

    n_tiles = tl.cdiv(in_height * in_width, BLOCK_N)
    pid_cg = pid0 // n_tiles
    pid_nt = pid0 % n_tiles

    c_start = pid_cg * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    n_start = pid_nt * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (in_height * in_width)

    ih = offs_n // in_width
    iw = offs_n % in_width

    # Accumulator [BLOCK_N, BLOCK_C]
    acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            oh_num = ih + padding_h - kh * dilation_h
            ow_num = iw + padding_w - kw * dilation_w

            cond_h = (oh_num % stride_h) == 0
            cond_w = (ow_num % stride_w) == 0
            oh = oh_num // stride_h
            ow = ow_num // stride_w

            valid = cond_h & cond_w & (oh >= 0) & (oh < out_height) & (ow >= 0) & (ow < out_width) & mask_n

            dy_base = pid_b * (out_height * out_width * channels) + oh * (out_width * channels) + ow * channels
            dy_ptrs = g_output_ptr + dy_base[:, None] + offs_c[None, :]
            w_ptrs = weight_ptr + (kh * kernel_w + kw) * channels + offs_c

            m = valid[:, None] & mask_c[None, :]
            dy = tl.load(dy_ptrs, mask=m, other=0.0)
            w = tl.load(w_ptrs, mask=mask_c, other=0.0)
            acc += (dy * w[None, :]).to(tl.float32)

    x_base = pid_b * (in_height * in_width * channels) + ih * (in_width * channels) + iw * channels
    x_ptrs = g_input_ptr + x_base[:, None] + offs_c[None, :]
    tl.store(x_ptrs, acc, mask=(mask_n[:, None] & mask_c[None, :]))


# Backward propagation weight gradient kernel (NHWC, parallel by batch, no atomic add), compute partial sums for each batch, finally reduce on host side
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 512, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 1024, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_weight_grad_kernel(
        input_ptr, grad_output_ptr, grad_weight_b_ptr,
        batch, in_height, in_width, channels,
        out_height, out_width,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)  # Map to (kh, kw, c-tile)
    pid_b = tl.program_id(1)  # batch id

    # Decode (kh, kw) and channel tile
    pid_k = pid0 % (kernel_h * kernel_w)
    pid_cg = pid0 // (kernel_h * kernel_w)
    kh = pid_k // kernel_w
    kw = pid_k % kernel_w

    c_start = pid_cg * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    # Accumulator (fp32)
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Streaming reduction over spatial dimension N = out_height * out_width
    Ntot = out_height * out_width
    for n in range(0, Ntot, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Ntot

        oh = offs_n // out_width
        ow = offs_n % out_width

        ih = oh * stride_h - padding_h + kh * dilation_h
        iw = ow * stride_w - padding_w + kw * dilation_w
        valid = (ih >= 0) & (ih < in_height) & (iw >= 0) & (iw < in_width) & mask_n

        # Linear address, assuming tensor is BHWC contiguous
        x_base = pid_b * in_height * in_width * channels + ih * in_width * channels + iw * channels
        dy_base = pid_b * out_height * out_width * channels + oh * out_width * channels + ow * channels

        x_ptrs = input_ptr + x_base[None, :] + offs_c[:, None]
        dy_ptrs = grad_output_ptr + dy_base[None, :] + offs_c[:, None]

        m = mask_c[:, None] & valid[None, :]

        x = tl.load(x_ptrs, mask=m, other=0.0)
        dy = tl.load(dy_ptrs, mask=m, other=0.0)

        acc += tl.sum((x * dy).to(tl.float32), axis=1)

    # Write per-batch partial sums: [B, KH, KW, C] contiguous layout (C contiguous)
    out_ptrs = (
            grad_weight_b_ptr
            + pid_b * (kernel_h * kernel_w * channels)
            + kh * (kernel_w * channels)
            + kw * channels
            + offs_c
    )
    tl.store(out_ptrs, acc, mask=mask_c)


class DepthwiseConv2DFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, stride, padding, dilation, dtype):
        # Save parameters
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.dtype = dtype
        amp_enabled = torch.is_autocast_enabled()
        compute_dtype = torch.get_autocast_dtype('cuda') if amp_enabled else input.dtype

        # Calculate output dimensions
        # Prepare tensors according to autocast compute dtype
        x = input if input.dtype == compute_dtype else input.to(compute_dtype)
        w = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)

        batch, in_h, in_w, channels = x.shape
        kernel_h, kernel_w = w.shape[:2]
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

        output = torch.empty((batch, out_h, out_w, channels), device=x.device, dtype=compute_dtype)

        # Launch forward kernel
        def grid(meta):
            n_tiles = triton.cdiv(out_h * out_w, meta['BLOCK_N'])
            return (
                n_tiles * triton.cdiv(channels, meta['BLOCK_C']),
                batch,
            )

        _depthwise_conv_kernel_optimized[grid](
            x, w, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
            kernel_h, kernel_w,
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )

        ctx.save_for_backward(x, w)
        ctx.compute_dtype = compute_dtype
        return output

    @staticmethod
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        compute_dtype = getattr(ctx, 'compute_dtype', grad_output.dtype)

        # Initialize gradients
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)

        # Calculate input gradients
        def input_grid(meta):
            in_h, in_w = input.shape[1:3]
            n_tiles = triton.cdiv(in_h * in_w, meta['BLOCK_N'])
            return (
                n_tiles * triton.cdiv(input.shape[3], meta['BLOCK_C']),
                input.shape[0],
            )

        go = grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)
        _depthwise_conv_backward_input_kernel[input_grid](
            go, weight, grad_input,
            grad_output.shape[0], grad_output.shape[1], grad_output.shape[2], grad_output.shape[3],
            input.shape[1], input.shape[2],
            weight.shape[0], weight.shape[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )

        # Calculate weight gradients
        def weight_grid(meta):
            kH, kW = weight.shape[:2]
            return (
                triton.cdiv(input.shape[3], meta['BLOCK_C']) * kH * kW,
                input.shape[0],
            )

        # per-batch partial sums [B, KH, KW, C] in fp32 (C contiguous)
        wgrad_b = torch.empty(
            (input.shape[0], weight.shape[0], weight.shape[1], input.shape[3]),
            device=input.device, dtype=torch.float32
        )

        _depthwise_conv_weight_grad_kernel[weight_grid](
            input, go, wgrad_b,
            input.shape[0], input.shape[1], input.shape[2], input.shape[3],
            grad_output.shape[1], grad_output.shape[2],
            weight.shape[0], weight.shape[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )

        # reduce over batch and cast to target dtype/layout [KH, KW, C]
        grad_weight_fp32 = wgrad_b.sum(dim=0)
        grad_weight.copy_(grad_weight_fp32.to(grad_weight.dtype))

        return grad_input, grad_weight, None, None, None, None


class OptimizedDepthwiseConv2d(torch.nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, dtype=None):
        super(OptimizedDepthwiseConv2d, self).__init__()

        # Handle convolution kernel parameters
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Set default data type, supporting fp16 and bf16

        # Initialize weights
        self.weight = torch.nn.Parameter(
            torch.empty(kernel_size[0], kernel_size[1], channels)
        )
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        # Check input format, must be BHWC
        assert x.dim() == 4 and x.shape[3] == self.channels, "Input must be BHWC format with matching channel count"

        return DepthwiseConv2DFunction.apply(
            x, self.weight,
            self.stride, self.padding, self.dilation,
            self.weight.dtype
        )


# Performance testing
def benchmark(batch_size=8, height=224, width=224, channels=64, kernel_size=13, dtype=torch.float16):
    import time

    # Create input
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()

    # Create our implementation
    conv_triton = OptimizedDepthwiseConv2d(
        channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).cuda()

    # Create PyTorch implementation (needs conversion to BCHW format)
    x_torch = x.permute(0, 3, 1, 2).clone()  # BHWC -> BCHW
    conv_torch = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=channels,
        bias=False
    ).cuda()

    conv_torch.weight.data = conv_triton.weight.permute(2, 0, 1).unsqueeze(1).data.clone()

    with torch.cuda.amp.autocast(dtype=dtype):
        # Warmup
        for _ in range(10):
            y1 = conv_triton(x)
            y1.mean().backward()
            y2 = conv_torch(x_torch)
            y2.mean().backward()

        torch.cuda.synchronize()

        # Test our implementation
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            y = conv_triton(x)
            y.mean().backward()
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / iterations

        # Test PyTorch implementation
        start = time.time()
        for _ in range(iterations):
            y = conv_torch(x_torch)
            y.mean().backward()
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / iterations

    # Convert results to milliseconds
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000

    print(f"Batch size={batch_size}, Height={height}, Width={width}, Channels={channels}, Kernel={kernel_size}x{kernel_size}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")

    # Verify result correctness
    x.requires_grad = True
    x_torch.requires_grad = True
    conv_torch.zero_grad()
    conv_triton.zero_grad()

    with torch.cuda.amp.autocast(dtype=dtype):
        out_triton = conv_triton(x)
        out_torch = conv_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC

        max_diff = torch.max(torch.abs(out_triton - out_torch))
        print(f"Maximum absolute error: {max_diff.item()}")

        grad = torch.randn_like(out_triton)
        torch.autograd.backward(out_triton, grad)
        torch.autograd.backward(out_torch, grad)

    max_diff = torch.max(torch.abs(x.grad - x_torch.grad.permute(0, 2, 3, 1)))
    print(f"Grad maximum absolute error: {max_diff.item()}")

    wg_triton = conv_triton.weight.grad.permute(2, 0, 1).unsqueeze(1)
    max_diff = torch.max(torch.abs(wg_triton - conv_torch.weight.grad))
    max_diff_p = torch.max(torch.abs(wg_triton - conv_torch.weight.grad) / conv_torch.weight.grad.max())
    print(f"W_grad maximum absolute error: {max_diff.item()}, {100 * max_diff_p.item():.4f}%")

    return triton_ms, torch_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['kernel_size'],
        x_vals=list(range(5, 31, 2)),  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['torch', 'triton', 'attn'],
        # label name for the lines
        line_names=["Torch", "Triton", "Attention"],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="dwconv-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_dwconv(kernel_size, provider):
    from diffusers.models.attention_processor import Attention

    batch_size = 8
    height, width = 128, 128
    channels = 1152

    def run_model(x, conv):
        y = conv(x)
        y.mean().backward()
        conv.zero_grad()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float16).cuda()
        conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        ).cuda().to(dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_model(x, conv), quantiles=quantiles)

    if provider == 'triton':
        x = torch.randn(batch_size, height, width, channels, dtype=torch.float16).cuda()
        conv = OptimizedDepthwiseConv2d(
            channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,  # same padding
        ).cuda().to(dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_model(x, conv), quantiles=quantiles)

    if provider == 'attn':
        x = torch.randn(batch_size, height * width, channels, dtype=torch.float16).cuda()
        attn = Attention(
            channels,
            heads=36,
            dim_head=32,
        ).cuda().to(dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_model(x, attn), quantiles=quantiles)

    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['kernel_size'],
        x_vals=list(range(5, 31, 2)),  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['torch', 'triton', 'attn'],
        # label name for the lines
        line_names=["Torch", "Triton", 'Attention'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="memory(MB)",  # label name for the y-axis
        plot_name="dwconv-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_dwconv_VRAM(kernel_size, provider):
    from diffusers.models.attention_processor import Attention

    batch_size = 32
    height, width = 64, 64
    channels = 1152

    if provider == 'torch':
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float16).cuda()
        conv = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        ).cuda().to(dtype=torch.float16)

        y = conv(x)
        torch.cuda.synchronize()
        allocated_memory = torch.cuda.memory_allocated() / 1e6

        # Clean up VRAM after getting results, avoid accumulation from multiple runs
        release_cuda_memory(y, conv, x)

    if provider == 'attn':
        x = torch.randn(batch_size, height * width, channels, dtype=torch.float16).cuda()
        attn = Attention(
            channels,
            heads=36,
            dim_head=32,
        ).cuda().to(dtype=torch.float16)
        y = attn(x)
        torch.cuda.synchronize()
        allocated_memory = torch.cuda.memory_allocated() / 1e6

        # Clean up VRAM after getting results, avoid accumulation from multiple runs
        release_cuda_memory(y, attn, x)

    if provider == 'triton':
        x = torch.randn(batch_size, height, width, channels, dtype=torch.float16).cuda()
        conv = OptimizedDepthwiseConv2d(
            channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,  # same padding
        ).cuda().to(dtype=torch.float16)

        y_triton = conv(x)
        torch.cuda.synchronize()
        allocated_memory = torch.cuda.memory_allocated() / 1e6

        # Clean up VRAM after getting results, avoid accumulation from multiple runs
        release_cuda_memory(y_triton, conv, x)

    return allocated_memory, allocated_memory, allocated_memory


if __name__ == "__main__":
    # # Usage case testing
    # batch_size = 16
    # height = 64
    # width = 64
    # channels = 1152
    # benchmark(batch_size, height, width, channels, kernel_size=9)
    # benchmark(batch_size, height, width, channels, kernel_size=13)
    # benchmark(batch_size, height, width, channels, kernel_size=17)

    benchmark_dwconv.run(save_path='./dwconv_performance_v2', show_plots=True, print_data=True)