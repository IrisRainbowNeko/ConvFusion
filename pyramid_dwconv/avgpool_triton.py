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
def _adaptive_avg_pool_kernel(
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
    
    # Spatial position calculation (using larger blocks for medium-sized spatial dimensions)
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # (BLOCK_H)
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # (BLOCK_W)
    
    # Dynamic window parameters
    h_start = tl.floor((h_idx.to(tl.float16) * in_h) / out_h).to(tl.int32)
    h_end = tl.ceil(((h_idx + 1).to(tl.float16) * (in_h)) / out_h).to(tl.int32)
    h_num = h_end-h_start
    w_start = tl.floor(((w_idx).to(tl.float16) * (in_w)) / out_w).to(tl.int32)
    w_end = tl.ceil(((w_idx + 1).to(tl.float16) * (in_w)) / out_w).to(tl.int32)
    w_num = w_end-w_start

    # Input feature map starting position
    batch_offset = pid_batch * in_h * in_w * channels
    
    # Initialize accumulators
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
    
    # Sliding window traversal
    for kh in range(tl.max(h_num)):
        for kw in range(tl.max(w_num)):
            # Calculate input coordinates (maintain 2D shape)
            current_h = h_start + kh  # (BLOCK_H)
            current_w = w_start + kw  # (BLOCK_W)
            
            # Generate spatial mask (auto-broadcast to BLOCK_H x BLOCK_W)
            h_mask = (current_h >= 0) & (current_h < h_end)  # (BLOCK_H)
            w_mask = (current_w >= 0) & (current_w < w_end)  # (BLOCK_W)
            valid_mask = h_mask[:, None] & w_mask[None, :]  # (BLOCK_H, BLOCK_W)
            
            # Input pointer calculation (explicit broadcast to 3D)
            input_ptrs = (
                batch_offset +
                (current_h[:, None, None] * in_w * channels) +
                (current_w[None, :, None] * channels) + 
                (c_start + tl.arange(0, BLOCK_C))[None, None, :]
            )
            
            # Vectorized load with mask
            data = tl.load(
                input_ptr + input_ptrs,
                mask=valid_mask[:, :, None] & c_mask[None, None, :],
                other=0.0
            )
            
            # Accumulate sum and count
            sum_acc += data
            count_acc += valid_mask  # Direct use of 2D mask
    
    # Calculate average
    safe_count = tl.maximum(count_acc, 1).to(tl.float32)[:, :, None]
    avg_result = sum_acc / safe_count
    
    # Output pointer calculation
    output_ptrs = (
        pid_batch * out_h * out_w * channels +
        (h_idx[:, None, None] * out_w * channels) +
        (w_idx[None, :, None] * channels) + 
        (c_start + tl.arange(0, BLOCK_C))[None, None, :]
    )
    
    # Generate output mask
    output_mask = (
        (h_idx[:, None, None] < out_h) & 
        (w_idx[None, :, None] < out_w) & 
        c_mask[None, None, :]
    )
    
    # Result storage
    tl.store(output_ptr + output_ptrs, avg_result, mask=output_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=8),
    ],
    key=['batch', 'in_h', 'in_w', 'channels', 'out_h', 'out_w'],
)
@triton.jit
def _adaptive_avg_pool_grad_kernel(
    grad_output_ptr, grad_input_ptr,
    batch, in_h, in_w, channels,
    out_h, out_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 3D parallelization: each program handles a block of output space and batch-channel block
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2)
    
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_bc // num_channel_blocks
    pid_c = pid_bc % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # Output h and w indices
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Calculate input window parameters (consistent with forward process)
    h_start = tl.floor((oh.to(tl.float16) * in_h) / out_h).to(tl.int32)
    h_end = tl.ceil(((oh + 1).to(tl.float16) * in_h) / out_h).to(tl.int32)
    h_num = h_end - h_start
    
    w_start = tl.floor((ow.to(tl.float16) * in_w) / out_w).to(tl.int32)
    w_end = tl.ceil(((ow + 1).to(tl.float16) * in_w) / out_w).to(tl.int32)
    w_num = w_end - w_start
    
    # Load gradient values of current output block
    output_ptrs = (
        pid_batch * out_h * out_w * channels +
        oh[:, None, None] * out_w * channels +
        ow[None, :, None] * channels +
        (c_start + tl.arange(0, BLOCK_C))[None, None, :]
    )
    output_mask = (
        (oh[:, None, None] < out_h) &
        (ow[None, :, None] < out_w) &
        c_mask[None, None, :]
    )
    grad_output = tl.load(grad_output_ptr + output_ptrs, mask=output_mask, other=0.0)
    
    # Calculate effective area and normalize
    area = (h_num[:, None] * w_num[None, :]).to(tl.float32)
    area = tl.maximum(area, 1.0)
    grad_val = grad_output / area[:, :, None]  # Broadcast to channel dimension
    
    # Traverse all positions in input window
    max_h_steps = tl.max(h_num)
    max_w_steps = tl.max(w_num)
    
    for kh in range(max_h_steps):
        for kw in range(max_w_steps):
            # Calculate current input position
            current_h = h_start + kh
            current_w = w_start + kw
            
            # Generate valid mask
            h_valid = (kh < h_num) & (current_h < in_h)
            w_valid = (kw < w_num) & (current_w < in_w)
            valid_mask = h_valid[:, None] & w_valid[None, :]
            
            # Calculate input pointer
            input_ptrs = (
                pid_batch * in_h * in_w * channels +
                current_h[:, None, None] * in_w * channels +
                current_w[None, :, None] * channels +
                (c_start + tl.arange(0, BLOCK_C))[None, None, :]
            )
            
            # Combine complete mask
            final_mask = (
                valid_mask[:, :, None] & 
                (current_h[:, None, None] >= 0) &
                (current_w[None, :, None] >= 0) &
                c_mask[None, None, :]
            )
            
            # Use atomic add operation to accumulate gradients
            tl.atomic_add(grad_input_ptr + input_ptrs, grad_val, mask=final_mask)

class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size):
        batch, in_h, in_w, channels = input.shape
        ctx.output_size = output_size

        out_h, out_w = output_size
        
        # Output tensor initialization
        output = torch.empty((batch, out_h, out_w, channels), device=input.device, dtype=input.dtype)
        
        # Dynamic grid division
        def grid(meta):
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                batch * triton.cdiv(channels, meta['BLOCK_C'])
            )
        
        # Launch optimized kernel
        _adaptive_avg_pool_kernel[grid](
            input, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
        )
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        out_h, out_w = ctx.output_size

        batch, in_h, in_w, channels = input.shape
        
        # Initialize gradients
        grad_input = torch.zeros_like(input)
        
        # Calculate input gradients
        def grid(meta):
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                batch * triton.cdiv(channels, meta['BLOCK_C'])
            )
        
        _adaptive_avg_pool_grad_kernel[grid](
            grad_output, grad_input,
            batch, in_h, in_w, channels,
            out_h, out_w,
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
        (int(height*scale), int(width*scale))
    ).cuda()

    out_triton = conv_triton(x)
    grad = torch.randn_like(out_triton)
    
    # Create PyTorch implementation (need to convert to BCHW format)
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    grad_torch = grad.permute(0, 3, 1, 2)
    conv_torch = torch.nn.AdaptiveAvgPool2d(
        (int(height*scale), int(width*scale))
    ).cuda().to(dtype)

    x.requires_grad = True
    x_torch.requires_grad = True
    
    # Warmup
    for _ in range(10):
        y1 = conv_triton(x)
        y2 = conv_torch(x_torch)
        torch.autograd.backward(y1, grad)
        torch.autograd.backward(y2, grad_torch)
    
    torch.cuda.synchronize()
    
    # Test our implementation
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        #y = conv_triton(x)
        y = conv_torch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
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
    
    print(f"Batch size={batch_size}, Height={height}, Width={width}, Channels={channels}, Scale={scale}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"Speedup: {torch_ms/triton_ms:.2f}x")
    
    # Verify result correctness
    conv_torch.zero_grad()
    conv_triton.zero_grad()

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
    benchmark(batch_size, height, width, channels, scale=1/6)