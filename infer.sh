# use 'hcpdiff' to run
CUDA_VISIBLE_DEVICES=0 hcp_run --cfg cfgs/workflow/conv/sd1_5_conv.py
# CUDA_VISIBLE_DEVICES=0 hcp_run --cfg cfgs/workflow/conv/sdxl_conv.py
# CUDA_VISIBLE_DEVICES=0 hcp_run --cfg cfgs/workflow/conv/pixart_conv.py

# High resolution
# CUDA_VISIBLE_DEVICES=1 hcp_run --cfg cfgs/workflow/conv/sdxl_conv_highres.py