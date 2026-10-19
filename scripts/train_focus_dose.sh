#!/usr/bin/env bash
# PADE, multi-variable (focus-dose) study: full model (Sections 3.2-3.3 of the paper).
# DATA must contain mask/ and SEM/ with identical file names <cell>_f{focus}_d{dose}.png,
# here the 3 x 3 anchor conditions (focus -45/0/45 nm x dose 40/50/60) of every training cell.
set -euo pipefail
DATA=${DATA:-./datasets/focus_dose_anchors}
python train.py --name pade_focus_dose \
    --dataset_mode custom --norm_mode clade --label_nc 256 --no_instance \
    --label_dir ${DATA}/mask --image_dir ${DATA}/SEM \
    --theta_type dose_focus --use_vae --use_pade --z_dim 512 \
    --lambda_kld 0.01 --lambda_gan 10 \
    --batchSize 32 --niter 1000 --niter_decay 500 \
    --crop_size 256 --load_size 256 --vflip \
    --save_epoch_freq 100 --gpu_ids 0
