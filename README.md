## Overview

Use swin transformer to denoise

check [here](https://github.com/Dr0sophila/swin_diffusion/blob/main/models.py)

code base on DiT

## Setup

First, download and set up the repo:

```
git clone https://github.com/Dr0sophila/swin_diffusion
cd swin_diffusion
```

create env

```
conda env create -f environment.yml
conda activate SWD
```

## Train

```
torchrun --nnodes=1 --nproc_per_node=1 train.py --model Swin --data-path /path/to/imagenet/train
```

**~~not tested~~**

## References:
Swin Transformer: https://github.com/microsoft/Swin-Transformer

DiT: https://github.com/facebookresearch/DiT

GLIDE: https://github.com/openai/glide-text2im

MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py

