import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['batch', 'in_h', 'in_w', 'channels', 'out_h', 'out_w'],
)
@triton.jit
def _adaptive_avg_pool_kernel_optimized(
        input_ptr, output_ptr,
        batch, in_h, in_w, channels,
        out_h, out_w,
        DTYPE: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: pid0 covers (c-group, n-tile), pid1 is batch
    pid0 = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_tiles = tl.cdiv(out_h * out_w, BLOCK_N)
    pid_cg = pid0 // n_tiles
    pid_nt = pid0 % n_tiles

    c_start = pid_cg * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    n_start = pid_nt * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (out_h * out_w)

    oh = offs_n // out_w
    ow = offs_n % out_w

    # compute pooling windows per output index (pure integer math)
    h_start = (oh * in_h) // out_h
    h_end = ((oh + 1) * in_h + out_h - 1) // out_h
    w_start = (ow * in_w) // out_w
    w_end = ((ow + 1) * in_w + out_w - 1) // out_w

    h_len = tl.maximum(h_end - h_start, 0)
    w_len = tl.maximum(w_end - w_start, 0)

    # compute dtype
    dtype = tl.float16 if DTYPE == 0 else (tl.bfloat16 if DTYPE == 1 else tl.float32)

    # accumulator over window [BLOCK_N, BLOCK_C]
    acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=dtype)

    max_kh = tl.max(h_len)
    max_kw = tl.max(w_len)

    for kh in range(0, max_kh):
        ih = h_start + kh
        h_valid = (kh < h_len) & (ih >= 0) & (ih < in_h) & mask_n
        for kw in range(0, max_kw):
            iw = w_start + kw
            w_valid = (kw < w_len) & (iw >= 0) & (iw < in_w) & mask_n
            valid = h_valid & w_valid

            x_base = (
                    pid_b * (in_h * in_w * channels)
                    + ih * (in_w * channels)
                    + iw * channels
            )
            x_ptrs = input_ptr + x_base[:, None] + offs_c[None, :]
            m = valid[:, None] & mask_c[None, :]
            x = tl.load(x_ptrs, mask=m, other=0.0).to(dtype)
            acc += x

    area = tl.maximum(h_len * w_len, 1).to(dtype)[:, None]
    out_val = (acc / area).to(dtype)

    y_base = (
            pid_b * (out_h * out_w * channels)
            + oh * (out_w * channels)
            + ow * channels
    )
    y_ptrs = output_ptr + y_base[:, None] + offs_c[None, :]
    tl.store(y_ptrs, out_val, mask=(mask_n[:, None] & mask_c[None, :]))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['batch', 'in_h', 'in_w', 'channels', 'out_h', 'out_w'],
)
@triton.jit
def _adaptive_avg_pool_backward_input_kernel(
        grad_output_ptr, grad_input_ptr,
        batch, in_h, in_w, channels,
        out_h, out_w,
        DTYPE: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: pid0 covers (c-group, n_in-tile), pid1 is batch
    pid0 = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_tiles = tl.cdiv(in_h * in_w, BLOCK_N)
    pid_cg = pid0 // n_tiles
    pid_nt = pid0 % n_tiles

    c_start = pid_cg * BLOCK_C
    offs_c = c_start + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    n_start = pid_nt * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < (in_h * in_w)

    ih = offs_n // in_w
    iw = offs_n % in_w

    # Overlapped contributions: for each input (ih, iw), accumulate all output bins including it
    ih0 = ih
    iw0 = iw

    oh_start = (ih0 * out_h) // in_h
    oh_end = ((ih0 + 1) * out_h + in_h - 1) // in_h
    ow_start = (iw0 * out_w) // in_w
    ow_end = ((iw0 + 1) * out_w + in_w - 1) // in_w

    oh_len = tl.maximum(oh_end - oh_start, 0)
    ow_len = tl.maximum(ow_end - ow_start, 0)

    dtype = tl.float16 if DTYPE == 0 else (tl.bfloat16 if DTYPE == 1 else tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=dtype)

    max_oh = tl.max(oh_len)
    max_ow = tl.max(ow_len)

    for dho in range(0, max_oh):
        oh = oh_start + dho
        oh_valid = (dho < oh_len) & (oh >= 0) & (oh < out_h) & mask_n
        # window extents for this oh
        h_s = (oh * in_h) // out_h
        h_e = ((oh + 1) * in_h + out_h - 1) // out_h
        h_ws = tl.maximum(h_e - h_s, 1).to(tl.float32)
        h_mem = (ih0 >= h_s) & (ih0 < h_e)

        for dwo in range(0, max_ow):
            ow = ow_start + dwo
            ow_valid = (dwo < ow_len) & (ow >= 0) & (ow < out_w) & mask_n
            w_s = (ow * in_w) // out_w
            w_e = ((ow + 1) * in_w + out_w - 1) // out_w
            w_ws = tl.maximum(w_e - w_s, 1).to(tl.float32)
            w_mem = (iw0 >= w_s) & (iw0 < w_e)

            valid = oh_valid & ow_valid & h_mem & w_mem

            area = (h_ws * w_ws).to(dtype)[:, None]
            go_base = (
                    pid_b * (out_h * out_w * channels)
                    + oh * (out_w * channels)
                    + ow * channels
            )
            go_ptrs = grad_output_ptr + go_base[:, None] + offs_c[None, :]
            m = valid[:, None] & mask_c[None, :]
            go = tl.load(go_ptrs, mask=m, other=0.0).to(dtype)
            acc += (go / area).to(dtype)

    # write gradients back (no atomics)
    x_base = (
            pid_b * (in_h * in_w * channels)
            + ih * (in_w * channels)
            + iw * channels
    )
    x_ptrs = grad_input_ptr + x_base[:, None] + offs_c[None, :]
    tl.store(x_ptrs, acc, mask=(mask_n[:, None] & mask_c[None, :]))


class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, output_size):
        batch, in_h, in_w, channels = input.shape
        ctx.output_size = output_size

        out_h, out_w = output_size

        amp_enabled = torch.is_autocast_enabled()
        compute_dtype = torch.get_autocast_dtype('cuda') if amp_enabled else input.dtype
        x = input if input.dtype == compute_dtype else input.to(compute_dtype)

        # Output tensor initialization
        output = torch.empty((batch, out_h, out_w, channels), device=input.device, dtype=compute_dtype)

        # Grid: (c-group * n-tiles (out)), batch)
        def grid(meta):
            n_tiles = triton.cdiv(out_h * out_w, meta['BLOCK_N'])
            return (
                n_tiles * triton.cdiv(channels, meta['BLOCK_C']),
                batch,
            )

        # DTYPE flag: 0=fp16,1=bf16,2=fp32
        dtype_flag = 2
        if compute_dtype == torch.float16:
            dtype_flag = 0
        elif compute_dtype == torch.bfloat16:
            dtype_flag = 1

        # Launch optimized kernel
        _adaptive_avg_pool_kernel_optimized[grid](
            x, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
            DTYPE=dtype_flag,
        )
        ctx.save_for_backward(x)
        ctx.compute_dtype = compute_dtype
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        out_h, out_w = ctx.output_size

        batch, in_h, in_w, channels = input.shape
        compute_dtype = getattr(ctx, 'compute_dtype', grad_output.dtype)
        go = grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)

        # Initialize gradients (write in input dtype)
        grad_input = torch.zeros((batch, in_h, in_w, channels), device=input.device, dtype=input.dtype)

        # Grid: (c-group * n_in-tiles, batch)
        def grid(meta):
            n_tiles = triton.cdiv(in_h * in_w, meta['BLOCK_N'])
            return (
                n_tiles * triton.cdiv(channels, meta['BLOCK_C']),
                batch,
            )

        dtype_flag = 2
        if compute_dtype == torch.float16:
            dtype_flag = 0
        elif compute_dtype == torch.bfloat16:
            dtype_flag = 1

        _adaptive_avg_pool_backward_input_kernel[grid](
            go, grad_input,
            batch, in_h, in_w, channels,
            out_h, out_w,
            DTYPE=dtype_flag,
        )

        return grad_input, None


class OptimizedAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def forward(self, x):
        # Input validation
        assert x.dim() == 4, "Input must be a 4D tensor in BHWC format"

        return AdaptiveAvgPool2dFunction.apply(x, self.output_size)


# Performance testing
def benchmark(batch_size=8, height=224, width=224, channels=64, scale=0.5, dtype=torch.float16):
    import time

    # Create input
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()

    # Create our implementation
    conv_triton = OptimizedAdaptiveAvgPool2d(
        (int(height * scale), int(width * scale))
    ).cuda()

    with torch.cuda.amp.autocast(dtype=dtype):
        out_triton = conv_triton(x)
    grad = torch.randn_like(out_triton)

    # Create PyTorch implementation (need to convert to BCHW format)
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    grad_torch = grad.permute(0, 3, 1, 2)
    conv_torch = torch.nn.AdaptiveAvgPool2d(
        (int(height * scale), int(width * scale))
    ).cuda().to(dtype)

    x.requires_grad = True
    x_torch.requires_grad = True

    # Warmup
    for _ in range(10):
        with torch.cuda.amp.autocast(dtype=dtype):
            y1 = conv_triton(x)
            y2 = conv_torch(x_torch)
            torch.autograd.backward(y1, grad)
            torch.autograd.backward(y2, grad_torch)

    torch.cuda.synchronize()

    # Test our implementation
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        with torch.cuda.amp.autocast(dtype=dtype):
            y = conv_triton(x)
            torch.autograd.backward(y, grad)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations

    # Test PyTorch implementation
    start = time.time()
    for _ in range(iterations):
        with torch.cuda.amp.autocast(dtype=dtype):
            y = conv_torch(x_torch)
            torch.autograd.backward(y, grad_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations

    # Convert results to milliseconds
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000

    print(f"Batch size={batch_size}, Height={height}, Width={width}, Channels={channels}, Scale={scale}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")

    # Verify result correctness
    conv_torch.zero_grad()
    conv_triton.zero_grad()
    # Clear input grads to avoid accumulation from warmup/bench loops
    x.grad = None
    x_torch.grad = None

    with torch.cuda.amp.autocast(dtype=dtype):
        out_triton = conv_triton(x)
        out_torch = conv_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC

    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"Max absolute error: {max_diff.item()}")

    torch.autograd.backward(out_triton, grad)
    torch.autograd.backward(out_torch, grad)

    max_diff = torch.max(torch.abs(x.grad - x_torch.grad.permute(0, 2, 3, 1)))
    print(f"Grad max absolute error: {max_diff.item()}")

    return triton_ms, torch_ms


if __name__ == "__main__":
    # Usage case testing
    batch_size = 8
    height = 64
    width = 64
    channels = 1152
    benchmark(batch_size, height, width, channels, scale=1 / 6)