#!/bin/bash
# ============================================================
# STAD-Imputer — AQI-36 Training Script
# Dataset  : AQI-36 (36 air quality stations, PM2.5)
# Run with : bash scripts/train_aqi36.sh
# ============================================================

conda activate stimp

cd "$(dirname "$0")/.."   # enter project root

python train_traffic.py \
    --dataset      aqi36 \
    --data_root    /remote-home/share/dmb_nas/wangtengbo/PriSTI-main/data \
    --eval_length  36 \
    --missing_ratio   0.9 \
    --task_name    stad-aqi36 \
    \
    --epochs       200 \
    --batch_size   16 \
    --lr           1e-3 \
    --wd           1e-4 \
    --test_freq    200 \
    --num_workers  4 \
    \
    --beta_start   0.0001 \
    --beta_end     0.2 \
    --num_steps    50 \
    --num_samples  10 \
    --schedule     quad \
    \
    --hidden_channels          32 \
    --diffusion_embedding_size 64 \
    --balance_weight           0.01 \
    \
    --ATI_dim             32 \
    --ATI_dropout         0.1 \
    --ATI_dilation_choices 1,2,4,8 \
    --ATI_tcn_layers      2 \
    \
    --ANA_in_dim       32 \
    --ANA_out_dim      32 \
    --ANA_k_phys       8 \
    --ANA_k_feat       8 \
    --ANA_num_prototypes 32 \
    --ANA_dropout      0.1 \
    --Add_ANA_Residual true \
    \
    --AHM_hidden_dim  32 \
    --AHM_pos_dim     8 \
    --AHM_num_experts 8 \
    --AHM_r           8 \
    --AHM_top_k       3 \
    --AHM_dropout     0.1 \
    --AHM_num_scales  3
