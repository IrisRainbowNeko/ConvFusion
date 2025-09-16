import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=8),
    ],
    key=['batch', 'in_h', 'in_w', 'channels', 'out_h', 'out_w'],
)
@triton.jit
def _bilinear_interp_kernel(
    input_ptr, output_ptr,
    batch, in_h, in_w, channels,
    out_h, out_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 3D parallelization
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2)
    
    # Decompose combined dimensions
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_bc // num_channel_blocks
    pid_c = pid_bc % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    # Channel mask
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # Output spatial positions
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Calculate input floating point coordinates
    y_scale = in_h.to(tl.float32) / out_h
    x_scale = in_w.to(tl.float32) / out_w
    
    y = (h_idx.to(tl.float32) + 0.5) * y_scale - 0.5
    x = (w_idx.to(tl.float32) + 0.5) * x_scale - 0.5
    y = tl.maximum(y,0)
    x = tl.maximum(x,0)
    
    # Calculate four neighboring point coordinates
    y_low = tl.floor(y).to(tl.int32)
    y_low = tl.minimum(tl.maximum(y_low, 0), in_h - 1)
    y_high = tl.minimum(y_low + 1, in_h - 1)
    dy = y - y_low.to(tl.float32)
    
    x_low = tl.floor(x).to(tl.int32)
    x_low = tl.minimum(tl.maximum(x_low, 0), in_w - 1)
    x_high = tl.minimum(x_low + 1, in_w - 1)
    dx = x - x_low.to(tl.float32)
    
    # Calculate weights (broadcast to 3D)
    w_lt = (1 - dx[None, :, None]) * (1 - dy[:, None, None])  # (BLOCK_H, BLOCK_W, 1)
    w_rt = dx[None, :, None] * (1 - dy[:, None, None])
    w_lb = (1 - dx[None, :, None]) * dy[:, None, None]
    w_rb = dx[None, :, None] * dy[:, None, None]
    
    # Input pointer calculation
    batch_offset = pid_batch * in_h * in_w * channels
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    
    # Pointers for four neighboring points
    ptrs_lt = (
        batch_offset +
        (y_low[:, None, None] * in_w * channels) +
        (x_low[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_rt = (
        batch_offset +
        (y_low[:, None, None] * in_w * channels) +
        (x_high[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_lb = (
        batch_offset +
        (y_high[:, None, None] * in_w * channels) +
        (x_low[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_rb = (
        batch_offset +
        (y_high[:, None, None] * in_w * channels) +
        (x_high[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    
    # Load data with mask
    data_lt = tl.load(input_ptr + ptrs_lt, mask=c_mask[None, None, :], other=0.0)
    data_rt = tl.load(input_ptr + ptrs_rt, mask=c_mask[None, None, :], other=0.0)
    data_lb = tl.load(input_ptr + ptrs_lb, mask=c_mask[None, None, :], other=0.0)
    data_rb = tl.load(input_ptr + ptrs_rb, mask=c_mask[None, None, :], other=0.0)
    
    # Interpolation calculation
    interp = (
        data_lt * w_lt +
        data_rt * w_rt +
        data_lb * w_lb +
        data_rb * w_rb
    )
    
    # Output pointer and mask
    output_ptrs = (
        pid_batch * out_h * out_w * channels +
        (h_idx[:, None, None] * out_w * channels) +
        (w_idx[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    output_mask = (
        (h_idx[:, None, None] < out_h) & 
        (w_idx[None, :, None] < out_w) & 
        c_mask[None, None, :]
    )
    
    # Result storage
    tl.store(output_ptr + output_ptrs, interp.to(output_ptr.dtype.element_ty), mask=output_mask)

class OptimizedBilinearInterp2d(torch.nn.Module):
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners
        
    def forward(self, x):
        # Input validation (BHWC format)
        assert x.dim() == 4, "Input must be a 4D tensor in BHWC format"
        batch, in_h, in_w, channels = x.shape
        
        # Parse output dimensions
        if isinstance(self.output_size, int):
            out_h = out_w = self.output_size
        else:
            out_h, out_w = self.output_size
            
        # Output tensor initialization
        output = torch.empty((batch, out_h, out_w, channels), 
                           device=x.device, dtype=x.dtype)
        
        # Dynamic grid division
        def grid(meta):
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                batch * triton.cdiv(channels, meta['BLOCK_C'])
            )
        
        # Launch kernel
        _bilinear_interp_kernel[grid](
            x, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
        )
        return output

# Correctness verification and performance testing function
def benchmark_interp(batch=2, in_h=64, in_w=64, channels=1152, scale=3.0, dtype=torch.float16):
    import time
    
    # Create input
    x = torch.randn(batch, in_h, in_w, channels, dtype=dtype).cuda()
    
    # Create our implementation
    interp_triton = OptimizedBilinearInterp2d(
        (int(in_h*scale), int(in_w*scale))
    ).cuda()
    
    # Create PyTorch implementation (need to convert to BCHW format)
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    interp_torch = torch.nn.Upsample(
        scale_factor=scale, 
        mode='bilinear',
        align_corners=False
    ).cuda().to(dtype)
    
    # Warmup
    for _ in range(10):
        _ = interp_triton(x)
        _ = interp_torch(x_torch)
    
    torch.cuda.synchronize()
    
    # Test performance
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = interp_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        _ = interp_torch(x_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # Convert results to milliseconds
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"Batch size={batch}, Input size={in_h}x{in_w}, Channels={channels}, Scale factor={scale}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"Speedup: {torch_ms/triton_ms:.2f}x")
    
    # Verify result correctness
    out_triton = interp_triton(x)
    out_torch = interp_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC

    # print(out_triton.view(12,12))
    # print(out_torch.view(12,12))

    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"Max absolute error: {max_diff.item()}")
    
    return triton_ms, torch_ms

if __name__ == "__main__":
    benchmark_interp(batch=2, in_h=64, in_w=64, channels=1152, scale=3.0)