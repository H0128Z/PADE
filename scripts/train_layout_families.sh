#!/usr/bin/env bash
# Additional layout families of Section 3.5: unchanged configuration, trained on the training layouts
# at the 3 x 3 anchor conditions focus -30/20/70 nm x dose 40/50/60.
set -euo pipefail
DATA=${DATA:-./datasets/layout_families_anchors}
python train.py --name pade_layout_families \
    --dataset_mode custom --norm_mode clade --label_nc 256 --no_instance \
    --label_dir ${DATA}/mask --image_dir ${DATA}/SEM \
    --theta_type dose_focus2d --use_vae --use_pade --z_dim 512 \
    --lambda_kld 0.01 --lambda_gan 10 \
    --batchSize 32 --niter 1000 --niter_decay 500 \
    --crop_size 256 --load_size 256 --vflip \
    --save_epoch_freq 100 --gpu_ids 0
