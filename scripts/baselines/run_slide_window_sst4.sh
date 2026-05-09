#!/bin/bash
# Slide Window Baseline on SST4 (PRE)
python train_baseline_slide_window.py \
    --data_root /remote-home/share/dmb_nas/wangtengbo/zone_sst4_data \
    --area PRE \
    --datasets_type sst4 \
    --missing_ratio 0.9
