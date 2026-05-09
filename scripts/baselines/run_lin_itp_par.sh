#!/bin/bash
# Lin-ITP Baseline on PAR (PRE)
python train_baseline_lin_itp.py \
    --data_root /remote-home/share/dmb_nas/wangtengbo/zone_par_data \
    --area PRE \
    --datasets_type par \
    --missing_ratio 0.9
