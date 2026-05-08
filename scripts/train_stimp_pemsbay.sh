#!/bin/bash
# STIMP Baseline — PEMS-BAY
conda activate stimp
cd "$(dirname "$0")/.."

python train_stimp_traffic.py \
    --dataset      pemsbay \
    --data_root    /remote-home/share/dmb_nas/wangtengbo/PriSTI-main/data \
    --eval_length  24 \
    --missing_pattern point \
    --missing_ratio   0.9 \
    --task_name    stimp-pemsbay \
    --epochs       200 \
    --batch_size   16 \
    --lr           1e-3 \
    --wd           1e-4 \
    --test_freq    50 \
    --num_workers  4 \
    --beta_start   0.0001 \
    --beta_end     0.2 \
    --num_steps    50 \
    --num_samples  10 \
    --schedule     quad \
    --hidden_channels          32 \
    --diffusion_embedding_size 64
