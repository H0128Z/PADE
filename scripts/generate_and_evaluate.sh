#!/usr/bin/env bash
# Generate the dense 7 x 11 focus-dose grid for every cell and compare CD and EL with the simulation.
set -euo pipefail
CKPT=${CKPT:-./checkpoints/pade_focus_dose}
MASK=${MASK:-./datasets/mask}               # <cell>.png
GRID=${GRID:-./datasets/focus_dose_grid}    # ground truth <cell>_f{focus}_d{dose}.png (also used as anchors)
OUT=${OUT:-./results/pade_focus_dose}
python generate.py --checkpoint ${CKPT} --mask_dir ${MASK} --anchor_dir ${GRID} --out_dir ${OUT}/SEM \
    --anchor_focus -45 0 45 --anchor_dose 40 50 60 \
    --focus -45 -30 -15 0 15 30 45 --dose 40 42 44 46 48 50 52 54 56 58 60
python evaluation/evaluate_cd_el.py --gt_dir ${GRID} --gen_dir ${OUT}/SEM --mask_dir ${MASK} --out ${OUT}/cd_el
