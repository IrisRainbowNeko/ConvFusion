import torch
import triton
import triton.language as tl


# Optimized version for specific convolution kernel sizes and hardware
@triton.autotune(
    configs=[
        # Optimized configuration for large channels
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 256}, num_warps=8),
        # Configuration for medium spatial dimensions
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_kernel_optimized(
    input_ptr, weight_ptr, output_ptr,
    batch, height, width, channels,
    out_height, out_width,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 3D parallelization: spatial dimensions + channel/batch combined dimension
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_cb = tl.program_id(2)
    
    # Decompose combined dimensions
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_cb // num_channel_blocks
    pid_c = pid_cb % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    # Calculate initial positions
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # Spatial position calculation (using larger blocks for medium-sized spatial dimensions)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Input feature map starting position
    batch_offset = pid_batch * height * width * channels
    
    # Use 3D tensor to store accumulator, leveraging register reuse
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    
    # Convolution kernel loop unrolling for faster computation (supports arbitrary kernel sizes)
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Coordinate calculation with dilation
            ih_base = h_offsets * stride_h - padding_h + kh * dilation_h
            iw_base = w_offsets * stride_w - padding_w + kw * dilation_w
            
            # Create spatial mask (handle boundaries)
            h_mask = (ih_base >= 0) & (ih_base < height)
            w_mask = (iw_base >= 0) & (iw_base < width)
            space_mask = h_mask[:, None] & w_mask[None, :]  # Logical operations + broadcast acceleration
            
            # Input data loading (with spatial mask)
            input_ptrs = (
                batch_offset + 
                (ih_base[:, None, None] * width * channels) + 
                (iw_base[None, :, None] * channels) + 
                (c_start + tl.arange(0, BLOCK_C)[None, None, :])
            )
            inputs = tl.load(input_ptr + input_ptrs, 
                           mask=(space_mask[:, :, None] & c_mask[None, None, :]), 
                           other=0.0)
            
            # Weight loading (optimized memory layout)
            weight_ptrs = (
                kh * kernel_w * channels + 
                kw * channels + 
                c_start + tl.arange(0, BLOCK_C)
            )
            weights = tl.load(weight_ptr + weight_ptrs, mask=c_mask, other=0.0)
            
            # Matrix accumulation (leveraging broadcast mechanism)
            acc += inputs * weights
    
    # Calculate output positions and store
    h_out_offsets = tl.where(h_offsets < out_height, h_offsets, 0)
    w_out_offsets = tl.where(w_offsets < out_width, w_offsets, 0)
    
    output_ptrs = (
        pid_batch * out_height * out_width * channels + 
        h_out_offsets[:, None, None] * out_width * channels + 
        w_out_offsets[None, :, None] * channels + 
        (c_start + tl.arange(0, BLOCK_C)[None, None, :])
    )
    
    output_mask = (
        (h_offsets[:, None, None] < out_height) & 
        (w_offsets[None, :, None] < out_width) & 
        (c_mask[None, None, :])
    )
    tl.store(output_ptr + output_ptrs, acc, mask=output_mask)

@triton.autotune(
    configs=[
        # Optimized configuration for large channels
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 256}, num_warps=8),
        # Configuration for medium spatial size
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_backward_input_kernel(
    g_output_ptr, weight_ptr, g_input_ptr,
    batch, height, width, channels,
    out_height, out_width,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 3D parallelization: spatial dimensions + channel/batch combined dimension
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_cb = tl.program_id(2)
    
    # Decompose combined dimensions
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_cb // num_channel_blocks
    pid_c = pid_cb % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    # Calculate initial positions
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # Spatial position calculation (using larger blocks for medium-sized spatial size)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Input feature map starting position
    batch_offset = pid_batch * height * width * channels
    
    # Use 3D tensor to store accumulator, leveraging register reuse
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    
    # Convolution kernel loop unrolling for faster computation (supports arbitrary kernel sizes)
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Coordinate calculation with dilation
            ih_base = h_offsets * stride_h - padding_h + kh * dilation_h
            iw_base = w_offsets * stride_w - padding_w + kw * dilation_w
            
            # Create spatial mask (handle boundaries)
            h_mask = (ih_base >= 0) & (ih_base < height)
            w_mask = (iw_base >= 0) & (iw_base < width)
            space_mask = h_mask[:, None] & w_mask[None, :]  # Logical operations + broadcast acceleration
            
            # Input data loading (with spatial mask)
            gout_ptrs = (
                batch_offset + 
                (ih_base[:, None, None] * width * channels) + 
                (iw_base[None, :, None] * channels) + 
                (c_start + tl.arange(0, BLOCK_C)[None, None, :])
            )
            grad_out = tl.load(g_output_ptr + gout_ptrs, 
                           mask=(space_mask[:, :, None] & c_mask[None, None, :]), 
                           other=0.0)
            
            # Weight loading (optimized memory layout)
            weight_ptrs = (
                (kernel_h-kh-1) * kernel_w * channels + 
                (kernel_w-kw-1) * channels + 
                c_start + tl.arange(0, BLOCK_C)
            )
            weights = tl.load(weight_ptr + weight_ptrs, mask=c_mask, other=0.0)
            
            # Matrix accumulation (leveraging broadcast mechanism)
            acc += grad_out * weights
    
    # Calculate output positions and store
    h_out_offsets = tl.where(h_offsets < out_height, h_offsets, 0)
    w_out_offsets = tl.where(w_offsets < out_width, w_offsets, 0)
    
    output_ptrs = (
        pid_batch * out_height * out_width * channels + 
        h_out_offsets[:, None, None] * out_width * channels + 
        w_out_offsets[None, :, None] * channels + 
        (c_start + tl.arange(0, BLOCK_C)[None, None, :])
    )
    
    output_mask = (
        (h_offsets[:, None, None] < out_height) & 
        (w_offsets[None, :, None] < out_width) & 
        (c_mask[None, None, :])
    )
    tl.store(g_input_ptr + output_ptrs, acc, mask=output_mask)

# Backward propagation weight gradient kernel
@triton.autotune(
    configs=[
        # Optimized configuration for large channels
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 256}, num_warps=8, num_stages=3),
        # Configuration for medium spatial dimensions
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4, num_stages=3),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_weight_grad_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    batch, height, width, channels,
    out_height, out_width,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
        # 3D parallelization: spatial dimensions + channel/batch combined dimension
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_cb = tl.program_id(2)
    
    # Decompose combined dimensions
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_cb // num_channel_blocks
    pid_c = pid_cb % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    # Calculate initial positions
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # Spatial position calculation (using larger blocks for medium-sized spatial dimensions)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_center = h_offsets * stride_h
    w_center = w_offsets * stride_w

    # Input feature map starting position
    batch_offset = pid_batch * height * width * channels

    # Input data loading (with spatial mask)
    ch_mask = (h_center >= 0) & (h_center < height)
    cw_mask = (w_center >= 0) & (w_center < width)
    cspace_mask = ch_mask[:, None] & cw_mask[None, :]  # Logical operations + broadcast acceleration

    in_ptrs = (
        batch_offset + 
        (h_center[:, None, None] * width * channels) + 
        (w_center[None, :, None] * channels) + 
        (c_start + tl.arange(0, BLOCK_C)[None, None, :])
    )
    inputs = tl.load(input_ptr + in_ptrs, mask=cspace_mask[:, :, None] &c_mask[None, None, :], other=0.0)
    
    # Convolution kernel loop unrolling for faster computation (supports arbitrary kernel sizes)
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Coordinate calculation with dilation
            ih_base = h_center - padding_h + kh * dilation_h
            iw_base = w_center - padding_w + kw * dilation_w
            
            # Create spatial mask (handle boundaries)
            h_mask = (ih_base >= 0) & (ih_base < height)
            w_mask = (iw_base >= 0) & (iw_base < width)
            space_mask = h_mask[:, None] & w_mask[None, :]  # Logical operations + broadcast acceleration
            
            # Output gradient loading (with spatial mask)
            gout_ptrs = (
                batch_offset + 
                (ih_base[:, None, None] * width * channels) + 
                (iw_base[None, :, None] * channels) + 
                (c_start + tl.arange(0, BLOCK_C)[None, None, :])
            )
            grad_out = tl.load(grad_output_ptr + gout_ptrs, 
                           mask=(space_mask[:, :, None] & c_mask[None, None, :]), 
                           other=0.0)
            
            # Weight loading (optimized memory layout)
            weight_ptrs = (
                (kernel_h-kh-1) * kernel_w * channels + 
                (kernel_w-kw-1) * channels + 
                c_start + tl.arange(0, BLOCK_C)
            )

            grad_w = tl.sum(tl.sum(grad_out*inputs, axis=0), axis=0)
            tl.atomic_add(grad_weight_ptr + weight_ptrs, grad_w, mask=c_mask)


class DepthwiseConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        # Save parameters
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.dtype = input.dtype
        
        # Calculate output dimensions
        batch, in_h, in_w, channels = input.shape
        kernel_h, kernel_w = weight.shape[:2]
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
        
        output = torch.empty((batch, out_h, out_w, channels), 
                           device=input.device, dtype=input.dtype)
        
        # Launch forward kernel
        def grid(meta):
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                batch * triton.cdiv(channels, meta['BLOCK_C'])
            )
        
        _depthwise_conv_kernel_optimized[grid](
            input, weight, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
            kernel_h, kernel_w,
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )
        
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        # Initialize gradients
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Calculate input gradients
        def input_grid(meta):
            out_h, out_w = grad_output.shape[1:3]
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                grad_output.shape[0] * triton.cdiv(grad_output.shape[3], meta['BLOCK_C'])
            )
        
        _depthwise_conv_backward_input_kernel[input_grid](
            grad_output, weight, grad_input,
            *grad_output.shape[:4],
            *input.shape[1:3],
            *weight.shape[:2],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )
        
        # Calculate weight gradients
        def weight_grid(meta):
            out_h, out_w = grad_output.shape[1:3]
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                grad_output.shape[0] * triton.cdiv(grad_output.shape[3], meta['BLOCK_C'])
            )

        _depthwise_conv_weight_grad_kernel[weight_grid](
            input, grad_output, grad_weight,
            *grad_output.shape[:4],
            *input.shape[1:3],
            *weight.shape[:2],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1]
        )
        
        return grad_input, grad_weight, None, None, None, None

class OptimizedDepthwiseConv2d(torch.nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
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
        
        # Initialize weights
        self.weight = torch.nn.Parameter(
            torch.empty(kernel_size[0], kernel_size[1], channels)
        )
        torch.nn.init.kaiming_uniform_(self.weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("bias", None)
        
    def forward(self, x):
        # Check input format, must be BHWC
        assert x.dim() == 4 and x.shape[3] == self.channels, "Input must be in BHWC format and channel count must match"
        
        if self.bias is None:
            return DepthwiseConv2DFunction.apply(
                x, self.weight, 
                self.stride, self.padding, self.dilation,
            )
        else:
            return DepthwiseConv2DFunction.apply(
                x, self.weight, 
                self.stride, self.padding, self.dilation,
            ) + self.bias


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
        dtype=dtype
    ).cuda()
    
    # Create PyTorch implementation (need to convert to BCHW format)
    x_torch = x.permute(0, 3, 1, 2).clone()  # BHWC -> BCHW
    conv_torch = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=channels,
        bias=False
    ).cuda().to(dtype)

    conv_torch.weight.data = conv_triton.weight.permute(2,0,1).unsqueeze(1).data.clone()
    
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
    print(f"Speedup: {torch_ms/triton_ms:.2f}x")

    
    # Verify result correctness
    x.requires_grad = True
    x_torch.requires_grad = True
    conv_torch.zero_grad()
    conv_triton.zero_grad()

    out_triton = conv_triton(x)
    out_torch = conv_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"Max absolute error: {max_diff.item()}")

    grad = torch.randn_like(out_triton)
    torch.autograd.backward(out_triton, grad)
    torch.autograd.backward(out_torch, grad)

    max_diff = torch.max(torch.abs(x.grad - x_torch.grad.permute(0, 2, 3, 1)))
    print(f"Grad max absolute error: {max_diff.item()}")

    wg_triton = conv_triton.weight.grad.permute(2,0,1).unsqueeze(1)
    max_diff = torch.max(torch.abs(wg_triton - conv_torch.weight.grad))
    max_diff_p = torch.max(torch.abs(wg_triton - conv_torch.weight.grad)/conv_torch.weight.grad.max())
    print(f"W_grad max absolute error: {max_diff.item()}, {100*max_diff_p.item():.4f}%")
    
    return triton_ms, torch_ms

# Simple usage demonstration
def example_usage():
    # Create input tensor in BHWC format
    batch_size = 2
    height, width = 64, 64
    channels = 512
    
    # Create input, BHWC format (2, 64, 64, 512)
    x = torch.randn(batch_size, height, width, channels, dtype=torch.float16).cuda()
    
    # Common large kernel sizes
    kernel_size = 13  # Between 11x11 to 17x17
    
    # Create depth-wise convolution layer
    conv = OptimizedDepthwiseConv2d(
        channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,  # same padding
        dtype=torch.float16  # Can be changed to torch.bfloat16
    ).cuda()
    
    # Forward pass
    output = conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Data type: {output.dtype}")
    
    return output

if __name__ == "__main__":
    # Usage case testing
    batch_size = 8
    height = 64
    width = 64
    channels = 1152
    benchmark(batch_size, height, width, channels, kernel_size=9)
    benchmark(batch_size, height, width, channels, kernel_size=13)
    benchmark(batch_size, height, width, channels, kernel_size=17)