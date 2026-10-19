#!/usr/bin/env bash
# Single-variable (dose) study of Section 3.1. P = number of dose anchors per cell (2 to 5);
# DATA_P must hold the masks and SEM images at those anchor doses (file names ..._f0_d{dose}.png).
set -euo pipefail
P=${P:-2}
DATA_P=${DATA_P:-./datasets/dose_${P}points}
python train.py --name pade_dose_${P}points \
    --dataset_mode custom --norm_mode clade --label_nc 256 --no_instance \
    --label_dir ${DATA_P}/mask --image_dir ${DATA_P}/SEM \
    --theta_type dose --use_vae --use_pade --z_dim 512 \
    --lambda_kld 0.01 --lambda_gan 10 \
    --batchSize 32 --niter 600 --niter_decay 300 \
    --crop_size 256 --load_size 256 --vflip \
    --save_epoch_freq 100 --gpu_ids 0
