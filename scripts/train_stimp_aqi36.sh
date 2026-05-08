#!/bin/bash
# STIMP Baseline — AQI-36
conda activate stimp
cd "$(dirname "$0")/.."

python train_stimp_traffic.py \
    --dataset      aqi36 \
    --data_root    /remote-home/share/dmb_nas/wangtengbo/PriSTI-main/data \
    --eval_length  36 \
    --missing_pattern point \
    --missing_ratio   0.9 \
    --task_name    stimp-aqi36 \
    --epochs       500 \
    --batch_size   16 \
    --lr           1e-3 \
    --wd           1e-4 \
    --test_freq    100 \
    --num_workers  4 \
    --beta_start   0.0001 \
    --beta_end     0.2 \
    --num_steps    50 \
    --num_samples  10 \
    --schedule     quad \
    --hidden_channels          32 \
    --diffusion_embedding_size 64
