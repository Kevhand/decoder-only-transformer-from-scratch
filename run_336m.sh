#!/bin/bash
# Training command used for the 336M model (2 GPUs). The 110M model uses train.py's defaults:
#   torchrun --standalone --nproc_per_node=2 train.py --compile
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=2 train.py \
  --n-layers 24 --d-model 1024 --n-heads 16 --d-ff 4096 --lr 3e-4 --max-tokens 7e9 \
  --micro-batch 8 --compile --out-dir checkpoints_350m --ckpt-every 100
