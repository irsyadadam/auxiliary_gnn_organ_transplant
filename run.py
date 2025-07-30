#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyg_CUDA

export CUDA_VISIBLE_DEVICES=6,7

python3 main.py \
  --mode train \
  --batch_size 128 \
  --lr 1e-4 \
  --epochs 20 \
  --checkpoint checkpoints/ \
  --link_target "TARGET_HERE" \
  --data_path "file_path" \
