"""Generate SEM images at arbitrary focus-dose conditions with a trained PADE model.

For each layout cell, the anchor SEM images of that cell are encoded once; the latent
code z at a target condition is the bilinear interpolation of the anchor latent means
over (focus, dose), extended linearly outside the anchor range, and the target condition
itself is supplied to the generator as theta.

Expected files
  --mask_dir    <cell>.png                          layout mask (class index per pixel)
  --anchor_dir  <cell>_f{focus}_d{dose}.png         anchor SEM images (flat, or in <cell>/ subfolders)
  --checkpoint  directory with opt.pkl and <epoch>_net_G.pth, <epoch>_net_E.pth written by train.py
Output
  --out_dir     <cell>_f{focus}_d{dose}.png

Example (multi-variable study, anchors at focus {-45, 0, 45} nm x dose {40, 50, 60}):
  python generate.py --checkpoint checkpoints/pade_focus_dose --mask_dir data/mask \
      --anchor_dir data/anchor_sem --out_dir results/generated \
      --anchor_focus -45 0 45 --anchor_dose 40 50 60 \
      --focus -45 -30 -15 0 15 30 45 --dose 40 42 44 46 48 50 52 54 56 58 60
"""
import argparse
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.networks.encoder import ConvEncoder
from models.networks.generator import SPADEGenerator
from data.pix2pix_dataset import normalize_theta

_T = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])])


def interp_weights(v, grid):
    """Linear interpolation neighbours (lo, hi, weight of hi) of v on a sorted 1-D grid.
    Outside the grid, the end segment is extended linearly (weight < 0 or > 1)."""
    if len(grid) == 1:
        return grid[0], grid[0], 0.0
    if v <= grid[0]:
        lo, hi = grid[0], grid[1]
    elif v >= grid[-1]:
        lo, hi = grid[-2], grid[-1]
    else:
        lo, hi = next((lo, hi) for lo, hi in zip(grid[:-1], grid[1:]) if lo <= v <= hi)
    return lo, hi, (v - lo) / (hi - lo)


def bilinear_latent(mu, f, d, grid_f, grid_d):
    fl, fh, a = interp_weights(f, grid_f)
    dl, dh, b = interp_weights(d, grid_d)
    return ((1 - a) * (1 - b) * mu[(fl, dl)] + a * (1 - b) * mu[(fh, dl)]
            + (1 - a) * b * mu[(fl, dh)] + a * b * mu[(fh, dh)])


def anchor_path(anchor_dir, cell, f, d):
    name = f"{cell}_f{f}_d{d}.png"
    for p in (os.path.join(anchor_dir, cell, name), os.path.join(anchor_dir, name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"anchor image {name} not found in {anchor_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--which_epoch', default='latest')
    ap.add_argument('--mask_dir', required=True)
    ap.add_argument('--anchor_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--cells', nargs='*', default=None, help='cell names (default: every mask in --mask_dir)')
    ap.add_argument('--anchor_focus', nargs='+', type=int, default=[-45, 0, 45])
    ap.add_argument('--anchor_dose', nargs='+', type=int, default=[40, 50, 60])
    ap.add_argument('--focus', nargs='+', type=int, default=[-45, -30, -15, 0, 15, 30, 45])
    ap.add_argument('--dose', nargs='+', type=int, default=list(range(40, 61, 2)))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    opt = pickle.load(open(os.path.join(args.checkpoint, 'opt.pkl'), 'rb'))
    opt.semantic_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
    opt.gpu_ids = []
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    netG = SPADEGenerator(opt).to(dev)
    netG.load_state_dict(torch.load(os.path.join(args.checkpoint, f'{args.which_epoch}_net_G.pth'), map_location=dev))
    netG.eval()
    netE = None
    if opt.use_vae:
        netE = ConvEncoder(opt).to(dev)
        netE.load_state_dict(torch.load(os.path.join(args.checkpoint, f'{args.which_epoch}_net_E.pth'), map_location=dev))
        netE.eval()

    grid_f, grid_d = sorted(args.anchor_focus), sorted(args.anchor_dose)
    cells = args.cells or sorted(os.path.splitext(n)[0] for n in os.listdir(args.mask_dir) if n.endswith('.png'))
    os.makedirs(args.out_dir, exist_ok=True)

    for cell in cells:
        # latent means of this cell's anchor images
        mu = {}
        if netE is not None:
            for fa in grid_f:
                for da in grid_d:
                    img = _T(Image.open(anchor_path(args.anchor_dir, cell, fa, da)).convert('RGB')).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        m, _ = netE(img)
                    mu[(fa, da)] = m.squeeze(0)

        seg = Image.open(os.path.join(args.mask_dir, f'{cell}.png')).convert('L').resize((256, 256), Image.NEAREST)
        tseg = torch.tensor(np.array(seg), dtype=torch.long, device=dev).unsqueeze(0)
        one_hot = torch.zeros((1, opt.label_nc, 256, 256), device=dev).scatter_(1, tseg.unsqueeze(1), 1.0)

        for f in args.focus:
            for d in args.dose:
                z = bilinear_latent(mu, f, d, grid_f, grid_d).unsqueeze(0) if netE is not None else None
                theta = normalize_theta(float(d), float(f), opt.theta_type).unsqueeze(0).to(dev) \
                    if getattr(opt, 'use_pade', False) else None
                with torch.no_grad():
                    fake = netG(one_hot, z=z, theta=theta)
                save_image((fake + 1) / 2, os.path.join(args.out_dir, f'{cell}_f{f}_d{d}.png'))
        print(f'{cell}: {len(args.focus) * len(args.dose)} images')


if __name__ == '__main__':
    main()
