# PADE: Process-Adaptive Denormalization for Focus–Dose Conditioned SEM Image Generation

Code for the paper

> H. Lee, S.-W. Lee, D.-N. Kim, "PADE: Process-Adaptive Denormalization for Focus–Dose Conditioned SEM Image Generation," submitted to *Journal of Micro/Nanopatterning, Materials, and Metrology* (JM3).

PADE generates SEM images of a layout at arbitrary focus and dose from a small set of anchor conditions.
The generator is built on CLADE (class-adaptive normalization). In every normalization block, PADE maps the
normalized process condition θ = (dose, focus) through a small MLP to a channel-wise scale and shift of the
class-indexed affine parameters, so the process condition acts on every block of the generator. A VAE encoder
provides a latent code z for the layout-specific appearance. At inference, z at a target condition is the
bilinear interpolation of the latent means of the anchor images of the same cell.

## Repository layout

```
train.py                     training (CLADE generator + PADE, optional VAE encoder)
generate.py                  generation at arbitrary focus-dose conditions (latent interpolation between anchors)
evaluation/cd_measure.py     row-wise CD and LWR measurement of line-space images (Appendix B)
evaluation/evaluate_cd_el.py CD agreement on held-out conditions and exposure latitude (EL) from Bossung surfaces
scripts/                     commands used for the experiments of the paper
models/ data/ options/ trainers/ util/   network, dataset, and option code
```

PADE itself is implemented in `models/networks/normalization.py` (`ClassAffine`, option `--use_pade`); the
normalization of θ is in `data/pix2pix_dataset.py` (`normalize_theta`).

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.11 and one NVIDIA GPU.

## Data format

Masks and SEM images are 8-bit PNG files with matching names `<cell>_f{focus}_d{dose}.png`
(focus in nm, may be negative; dose in the units of the simulator). The mask stores one class index per pixel.

* Training: `--label_dir` holds the mask of every training image (one copy per condition) and `--image_dir`
  the SEM images, here only the anchor conditions.
* Generation and evaluation: one mask per cell, `<cell>.png`, and the SEM images of the cell at the anchor
  conditions (and, for evaluation, at all conditions), either in one folder or in `<cell>/` subfolders.

The process condition is parsed from the file names and normalized according to `--theta_type`:

| `--theta_type` | study | θ |
|---|---|---|
| `dose_focus`   | line-space focus-dose study (Sections 3.2–3.4) | ((dose − 40)/20, focus/45) |
| `dose_focus2d` | additional layout families (Section 3.5)       | ((dose − 40)/20, (focus − 20)/50) |
| `dose`         | single-variable dose study (Section 3.1)       | ((dose − 20)/20) |

The simulated SEM dataset and the trained checkpoints are available from the corresponding author upon
reasonable request.

## Usage

Train the full model on the 3 × 3 anchor conditions (focus −45/0/45 nm × dose 40/50/60):

```bash
DATA=./datasets/focus_dose_anchors bash scripts/train_focus_dose.sh
```

Generate the dense 7 × 11 focus-dose grid and compare CD and EL with the simulation:

```bash
CKPT=./checkpoints/pade_focus_dose MASK=./datasets/mask GRID=./datasets/focus_dose_grid \
    bash scripts/generate_and_evaluate.sh
```

`evaluate_cd_el.py` prints the CD correlation, RMSE, and bias over the held-out (non-anchor) conditions and
the mean absolute EL error and EL correlation over all cells. The ablation models, the single-variable study,
and the additional layout families are trained with the other scripts in `scripts/`.

## Acknowledgments and license

This code is built on [CLADE](https://github.com/tzt101/CLADE) (Tan et al.) and
[SPADE](https://github.com/NVlabs/SPADE) (Park et al.). Like these projects, it is released under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (see `LICENSE.md`).
