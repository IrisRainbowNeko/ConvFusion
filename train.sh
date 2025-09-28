# cd XXX/conv-diff
# You can change py config with corresponding yaml config files.
# config files placed in `conv-diff/cfgs/train/py` and `conv-diff/cfgs/train/yaml`

## Single-GPU
# hcp_train_1gpu --cfg cfgs/train/py/sd1_5_dist.py
# hcp_train_1gpu --cfg cfgs/train/py/pixart_dist.py
# hcp_train_1gpu --cfg cfgs/train/py/sdxl_dist.py

## Multi-GPUs
hcp_train --cfg cfgs/train/py/gemma3_dist.py
# hcp_train --cfg cfgs/train/py/sd1_5_dist.py
# hcp_train --cfg cfgs/train/py/pixart_dist.py
# hcp_train --cfg cfgs/train/py/sdxl_dist.py