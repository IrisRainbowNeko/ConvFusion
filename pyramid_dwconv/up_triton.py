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
def _bilinear_interp_kernel(
    input_ptr, output_ptr,
    batch, in_h, in_w, channels,
    out_h, out_w,
    ALIGN_CORNERS: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: pid0 -> (c-group, n-tile), pid1 -> batch
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

    # compute source coords
    oh_f = oh.to(tl.float32)
    ow_f = ow.to(tl.float32)
    in_h_f = tl.full([1], in_h, tl.float32)
    in_w_f = tl.full([1], in_w, tl.float32)
    out_h_f = tl.full([1], out_h, tl.float32)
    out_w_f = tl.full([1], out_w, tl.float32)

    dtype = tl.float16 if DTYPE == 0 else (tl.bfloat16 if DTYPE == 1 else tl.float32)

    if ALIGN_CORNERS:
        scale_h = (in_h_f - 1.0) / tl.maximum(out_h_f - 1.0, 1.0)
        scale_w = (in_w_f - 1.0) / tl.maximum(out_w_f - 1.0, 1.0)
        src_h_f32 = oh_f * scale_h
        src_w_f32 = ow_f * scale_w
    else:
        scale_h = in_h_f / out_h_f
        scale_w = in_w_f / out_w_f
        src_h_f32 = (oh_f + 0.5) * scale_h - 0.5
        src_w_f32 = (ow_f + 0.5) * scale_w - 0.5

    # indices from fp32 for stability
    y0 = tl.floor(src_h_f32).to(tl.int32)
    x0 = tl.floor(src_w_f32).to(tl.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy1 = (src_h_f32 - y0.to(tl.float32)).to(dtype)
    wx1 = (src_w_f32 - x0.to(tl.float32)).to(dtype)
    wy0 = (1.0 - wy1).to(dtype)
    wx0 = (1.0 - wx1).to(dtype)

    # clamp
    y0 = tl.maximum(tl.minimum(y0, in_h - 1), 0)
    y1 = tl.maximum(tl.minimum(y1, in_h - 1), 0)
    x0 = tl.maximum(tl.minimum(x0, in_w - 1), 0)
    x1 = tl.maximum(tl.minimum(x1, in_w - 1), 0)

    # pointers and load
    base_b = pid_b * (in_h * in_w * channels)
    base00 = base_b + y0 * (in_w * channels) + x0 * channels
    base01 = base_b + y0 * (in_w * channels) + x1 * channels
    base10 = base_b + y1 * (in_w * channels) + x0 * channels
    base11 = base_b + y1 * (in_w * channels) + x1 * channels

    ptr00 = input_ptr + base00[:, None] + offs_c[None, :]
    ptr01 = input_ptr + base01[:, None] + offs_c[None, :]
    ptr10 = input_ptr + base10[:, None] + offs_c[None, :]
    ptr11 = input_ptr + base11[:, None] + offs_c[None, :]

    m = mask_n[:, None] & mask_c[None, :]
    v00 = tl.load(ptr00, mask=m, other=0.0).to(dtype)
    v01 = tl.load(ptr01, mask=m, other=0.0).to(dtype)
    v10 = tl.load(ptr10, mask=m, other=0.0).to(dtype)
    v11 = tl.load(ptr11, mask=m, other=0.0).to(dtype)

    wy0 = wy0[:, None]
    wy1 = wy1[:, None]
    wx0 = wx0[:, None]
    wx1 = wx1[:, None]

    out = (
        v00 * (wy0 * wx0) +
        v01 * (wy0 * wx1) +
        v10 * (wy1 * wx0) +
        v11 * (wy1 * wx1)
    )

    # store
    y_base = (
        pid_b * (out_h * out_w * channels)
        + oh * (out_w * channels)
        + ow * channels
    )
    out_ptrs = output_ptr + y_base[:, None] + offs_c[None, :]
    tl.store(out_ptrs, out, mask=m)

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
def _bilinear_interp_backward_input_kernel(
    grad_output_ptr, grad_input_ptr,
    batch, in_h, in_w, channels,
    out_h, out_w,
    ALIGN_CORNERS: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: pid0 -> (c-group, n_in-tile), pid1 -> batch
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

    ih_f = ih.to(tl.float32)
    iw_f = iw.to(tl.float32)
    in_h_f = tl.full([1], in_h, tl.float32)
    in_w_f = tl.full([1], in_w, tl.float32)
    out_h_f = tl.full([1], out_h, tl.float32)
    out_w_f = tl.full([1], out_w, tl.float32)

    dtype = tl.float16 if DTYPE == 0 else (tl.bfloat16 if DTYPE == 1 else tl.float32)

    if ALIGN_CORNERS:
        scale_h = (in_h_f - 1.0) / tl.maximum(out_h_f - 1.0, 1.0)
        scale_w = (in_w_f - 1.0) / tl.maximum(out_w_f - 1.0, 1.0)
        # open interval: ( (ih-1)/s, (ih+1)/s ) -> integers
        oh_lo = tl.floor((ih_f - 1.0) / scale_h).to(tl.int32) + 1
        oh_hi = tl.ceil((ih_f + 1.0) / scale_h).to(tl.int32) - 1
        ow_lo = tl.floor((iw_f - 1.0) / scale_w).to(tl.int32) + 1
        ow_hi = tl.ceil((iw_f + 1.0) / scale_w).to(tl.int32) - 1
    else:
        scale_h = in_h_f / out_h_f
        scale_w = in_w_f / out_w_f
        # open interval: ( ((ih-0.5)/s - 0.5), ((ih+1.5)/s - 0.5) )
        oh_lo = tl.floor(((ih_f - 0.5) / scale_h) - 0.5).to(tl.int32) + 1
        oh_hi = tl.ceil(((ih_f + 1.5) / scale_h) - 0.5).to(tl.int32) - 1
        ow_lo = tl.floor(((iw_f - 0.5) / scale_w) - 0.5).to(tl.int32) + 1
        ow_hi = tl.ceil(((iw_f + 1.5) / scale_w) - 0.5).to(tl.int32) - 1

    oh_lo = tl.maximum(oh_lo, 0)
    ow_lo = tl.maximum(ow_lo, 0)
    oh_hi = tl.minimum(oh_hi, out_h - 1)
    ow_hi = tl.minimum(ow_hi, out_w - 1)

    oh_len = tl.maximum(oh_hi - oh_lo + 1, 0)
    ow_len = tl.maximum(ow_hi - ow_lo + 1, 0)

    acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=dtype)

    max_oh = tl.max(oh_len)
    max_ow = tl.max(ow_len)

    for dho in range(0, max_oh):
        oh = oh_lo + dho
        h_valid = (dho < oh_len) & (oh >= 0) & (oh < out_h) & mask_n
        if ALIGN_CORNERS:
            src_h_f32 = oh.to(tl.float32) * scale_h
        else:
            src_h_f32 = (oh.to(tl.float32) + 0.5) * scale_h - 0.5
        y0 = tl.floor(src_h_f32).to(tl.int32)
        y1 = y0 + 1
        wy1 = (src_h_f32 - y0.to(tl.float32)).to(dtype)
        wy0 = (1.0 - wy1).to(dtype)
        y0c = tl.maximum(tl.minimum(y0, in_h - 1), 0)
        y1c = tl.maximum(tl.minimum(y1, in_h - 1), 0)
        eq0 = (y0c == ih)
        eq1 = (y1c == ih)
        wy = (tl.where(eq0, wy0, 0.0) + tl.where(eq1, wy1, 0.0)).to(dtype)
        wy = wy[:, None]

        for dwo in range(0, max_ow):
            ow = ow_lo + dwo
            w_valid = (dwo < ow_len) & (ow >= 0) & (ow < out_w) & mask_n
            valid = h_valid & w_valid

            if ALIGN_CORNERS:
                src_w_f32 = ow.to(tl.float32) * scale_w
            else:
                src_w_f32 = (ow.to(tl.float32) + 0.5) * scale_w - 0.5
            x0 = tl.floor(src_w_f32).to(tl.int32)
            x1 = x0 + 1
            wx1 = (src_w_f32 - x0.to(tl.float32)).to(dtype)
            wx0 = (1.0 - wx1).to(dtype)
            x0c = tl.maximum(tl.minimum(x0, in_w - 1), 0)
            x1c = tl.maximum(tl.minimum(x1, in_w - 1), 0)
            ex0 = (x0c == iw)
            ex1 = (x1c == iw)
            wx = (tl.where(ex0, wx0, 0.0) + tl.where(ex1, wx1, 0.0)).to(dtype)
            wx = wx[:, None]

            w = wy * wx

            go_base = (
                pid_b * (out_h * out_w * channels)
                + oh * (out_w * channels)
                + ow * channels
            )
            go_ptrs = grad_output_ptr + go_base[:, None] + offs_c[None, :]
            m = valid[:, None] & mask_c[None, :]
            go = tl.load(go_ptrs, mask=m, other=0.0).to(dtype)
            acc += go * w

    x_base = (
        pid_b * (in_h * in_w * channels)
        + ih * (in_w * channels)
        + iw * channels
    )
    x_ptrs = grad_input_ptr + x_base[:, None] + offs_c[None, :]
    tl.store(x_ptrs, acc, mask=(mask_n[:, None] & mask_c[None, :]))

class BilinearInterp2dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, output_size, align_corners=False):
        batch, in_h, in_w, channels = input.shape
        out_h, out_w = output_size

        amp_enabled = torch.is_autocast_enabled()
        compute_dtype = torch.get_autocast_dtype('cuda') if amp_enabled else input.dtype

        x = input if input.dtype == compute_dtype else input.to(compute_dtype)
        output = torch.empty((batch, out_h, out_w, channels), device=input.device, dtype=compute_dtype)

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

        _bilinear_interp_kernel[grid](
            x, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
            ALIGN_CORNERS=1 if align_corners else 0,
            DTYPE=dtype_flag,
        )
        ctx.save_for_backward(x)
        ctx.output_size = output_size
        ctx.align_corners = align_corners
        ctx.compute_dtype = compute_dtype
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        batch, in_h, in_w, channels = input.shape
        out_h, out_w = ctx.output_size
        align_corners = ctx.align_corners
        compute_dtype = getattr(ctx, 'compute_dtype', grad_output.dtype)

        go = grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)

        # write in input dtype; kernel accumulates in fp32 then casts on store
        grad_input = torch.zeros((batch, in_h, in_w, channels), device=input.device, dtype=input.dtype)

        def grid(meta):
            n_tiles = triton.cdiv(out_h * out_w, meta['BLOCK_N'])
            return (
                n_tiles * triton.cdiv(channels, meta['BLOCK_C']),
                batch,
            )

        dtype_flag = 2
        if compute_dtype == torch.float16:
            dtype_flag = 0
        elif compute_dtype == torch.bfloat16:
            dtype_flag = 1

        _bilinear_interp_backward_input_kernel[grid](
            go, grad_input,
            batch, in_h, in_w, channels,
            out_h, out_w,
            ALIGN_CORNERS=1 if align_corners else 0,
            DTYPE=dtype_flag,
        )

        return grad_input, None, None

class OptimizedBilinearInterp2d(torch.nn.Module):
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.align_corners = align_corners

    def forward(self, x):
        assert x.dim() == 4, "Input must be BHWC"
        return BilinearInterp2dFunction.apply(x, self.output_size, self.align_corners)

def benchmark(batch_size=8, height=64, width=64, channels=256, scale=2.0, dtype=torch.float16, align_corners=False):
    import time
    import torch.nn.functional as F

    x = torch.randn(batch_size, height, width, channels, dtype=dtype, device='cuda')
    out_h, out_w = int(height * scale), int(width * scale)

    up_triton = OptimizedBilinearInterp2d((out_h, out_w), align_corners=align_corners).cuda()

    x_torch = x.permute(0, 3, 1, 2).contiguous()
    grad_shape = (batch_size, out_h, out_w, channels)
    grad = torch.randn(grad_shape, dtype=dtype, device='cuda')
    grad_torch = grad.permute(0, 3, 1, 2).contiguous()

    x.requires_grad = True
    x_torch.requires_grad = True

    # warmup
    for _ in range(10):
        with torch.cuda.amp.autocast(dtype=dtype):
            y1 = up_triton(x)
            y2 = F.interpolate(x_torch, size=(out_h, out_w), mode='bilinear', align_corners=align_corners)
            torch.autograd.backward(y1, grad)
            torch.autograd.backward(y2, grad_torch)

    torch.cuda.synchronize()

    # triton timing
    iters = 100
    start = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast(dtype=dtype):
            y = up_triton(x)
            torch.autograd.backward(y, grad)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iters

    # torch timing
    start = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast(dtype=dtype):
            y = F.interpolate(x_torch, size=(out_h, out_w), mode='bilinear', align_corners=align_corners)
            torch.autograd.backward(y, grad_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iters

    print(f"Batch size={batch_size}, Height={height}, Width={width}, Channels={channels}, Scale={scale}")
    print(f"Triton: {triton_time*1000:.3f}ms, PyTorch: {torch_time*1000:.3f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

    # correctness
    up_triton.zero_grad()
    x.grad = None
    x_torch.grad = None

    with torch.cuda.amp.autocast(dtype=dtype):
        out_triton = up_triton(x)
        out_torch = F.interpolate(x_torch, size=(out_h, out_w), mode='bilinear', align_corners=align_corners).permute(0, 2, 3, 1)
    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"Max absolute error: {max_diff}")

    with torch.cuda.amp.autocast(dtype=dtype):
        torch.autograd.backward(out_triton, grad)
        torch.autograd.backward(out_torch, grad)
    grad_diff = (x.grad - x_torch.grad.permute(0, 2, 3, 1)).abs().max().item()
    grad_diff_p = (grad_diff/x_torch.grad.permute(0, 2, 3, 1).abs().max().item())*100
    print(f"Grad max absolute error: {grad_diff}, {grad_diff_p:.4f}%")

if __name__ == "__main__":
    benchmark(batch_size=8, height=64, width=64, channels=1152, scale=1.5, dtype=torch.float16, align_corners=False)