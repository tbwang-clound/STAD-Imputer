#!/bin/bash
# DINEOF Baseline on Chl-a (PRE)
python train_baseline_dineof.py \
    --data_root /remote-home/share/dmb_nas/wangtengbo/zone_chla_data \
    --area PRE \
    --datasets_type chla \
    --missing_ratio 0.9 \
    --model_K 10
