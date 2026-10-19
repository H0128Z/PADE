#!/usr/bin/env bash
# Ablation models of Section 3.4 (same data as train_focus_dose.sh). The latent-only model was
# trained for 1000 + 500 epochs like the full model, the others for 600 + 300 epochs.
set -euo pipefail
DATA=${DATA:-./datasets/focus_dose_anchors}
COMMON="--dataset_mode custom --norm_mode clade --label_nc 256 --no_instance
        --label_dir ${DATA}/mask --image_dir ${DATA}/SEM --theta_type dose_focus
        --lambda_kld 0.01 --lambda_gan 10 --batchSize 32
        --crop_size 256 --load_size 256 --vflip --save_epoch_freq 100 --gpu_ids 0"
python train.py --name abl_no_pade      ${COMMON} --niter 1000 --niter_decay 500 --use_vae --z_dim 512   # latent only, no PADE
python train.py --name abl_no_latent    ${COMMON} --niter 600 --niter_decay 300 --use_pade              # PADE only, no encoder
python train.py --name abl_neither      ${COMMON} --niter 600 --niter_decay 300                         # neither pathway
python train.py --name abl_zdim128      ${COMMON} --niter 600 --niter_decay 300 --use_vae --use_pade --z_dim 128
python train.py --name abl_zdim256      ${COMMON} --niter 600 --niter_decay 300 --use_vae --use_pade --z_dim 256
