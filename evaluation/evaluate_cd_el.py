"""CD and exposure-latitude (EL) agreement between generated and ground-truth SEM images.

For every cell and focus-dose condition, CD and LWR are measured on both images
(cd_measure.py). CD agreement (correlation, RMSE, bias) is reported on the held-out
conditions, i.e. all conditions that are not anchors.

EL: each cell's CD values form a Bossung surface over (dose, focus), interpolated with a
bicubic spline on a 101 x 91 grid. The in-spec region is where the CD lies within +/-10%
of the CD at the nominal condition (focus 0 nm, dose 50) of the same surface; EL is the
dose width of the largest axis-aligned in-spec rectangle, in percent of the nominal dose.

Example:
  python evaluation/evaluate_cd_el.py --gt_dir data/sem_grid --gen_dir results/generated \
      --mask_dir data/mask --out results/cd_el
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

from cd_measure import measure_image, pitch_of

NOMINAL_FOCUS, NOMINAL_DOSE, TOL = 0, 50, 0.10


def image_path(root, cell, f, d):
    name = f'{cell}_f{f}_d{d}.png'
    for p in (os.path.join(root, cell, name), os.path.join(root, name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{name} not found in {root}')


def _measure(job):
    key, path, mask, pitch = job
    return key, measure_image(path, mask, pitch)


def max_rect(mask):
    """(rows, cols) of the largest all-True axis-aligned rectangle of a 2-D boolean array."""
    H, W = mask.shape
    heights = np.zeros(W, dtype=int)
    best = (0, 0, 0)
    for i in range(H):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []
        for j in range(W + 1):
            h = heights[j] if j < W else 0
            start = j
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                if sh * (j - s) > best[0]:
                    best = (sh * (j - s), sh, j - s)
                start = s
            stack.append((start, h))
    return best[1], best[2]


def exposure_latitude(g, col, focus, dose):
    """EL (%) of one Bossung surface: column `col` of the per-condition table g of one cell."""
    Z = g.pivot_table(index='dose', columns='focus', values=col).reindex(index=dose, columns=focus).values
    target = float(g[(g.focus == NOMINAL_FOCUS) & (g.dose == NOMINAL_DOSE)][col].iloc[0])
    D = np.linspace(min(dose), max(dose), 101)
    F = np.linspace(min(focus), max(focus), 91)
    Zf = RectBivariateSpline(dose, focus, Z, kx=3, ky=3)(D, F)
    nr, _ = max_rect((Zf >= (1 - TOL) * target) & (Zf <= (1 + TOL) * target))
    return nr * (D[1] - D[0]) / NOMINAL_DOSE * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_dir', required=True, help='ground-truth SEM images <cell>_f{focus}_d{dose}.png (flat or in <cell>/)')
    ap.add_argument('--gen_dir', required=True, help='generated SEM images, same names')
    ap.add_argument('--mask_dir', required=True, help='layout masks <cell>.png')
    ap.add_argument('--cells', nargs='*', default=None, help='default: every mask in --mask_dir')
    ap.add_argument('--focus', nargs='+', type=int, default=[-45, -30, -15, 0, 15, 30, 45])
    ap.add_argument('--dose', nargs='+', type=int, default=list(range(40, 61, 2)))
    ap.add_argument('--anchor_focus', nargs='+', type=int, default=[-45, 0, 45])
    ap.add_argument('--anchor_dose', nargs='+', type=int, default=[40, 50, 60])
    ap.add_argument('--out', default='cd_el')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    cells = args.cells or sorted(os.path.splitext(n)[0] for n in os.listdir(args.mask_dir) if n.endswith('.png'))
    jobs = []
    for c in cells:
        mask, pitch = os.path.join(args.mask_dir, f'{c}.png'), pitch_of(c)
        for f in args.focus:
            for d in args.dose:
                jobs.append((('gt', c, f, d), image_path(args.gt_dir, c, f, d), mask, pitch))
                jobs.append((('gen', c, f, d), image_path(args.gen_dir, c, f, d), mask, pitch))
    with Pool(args.workers) as p:
        res = dict(p.imap_unordered(_measure, jobs, chunksize=16))

    rows = [dict(cell=c, focus=f, dose=d,
                 split='anchor' if (f in args.anchor_focus and d in args.anchor_dose) else 'held-out',
                 cd_gt=res[('gt', c, f, d)][0], cd_gen=res[('gen', c, f, d)][0],
                 lwr_gt=res[('gt', c, f, d)][1], lwr_gen=res[('gen', c, f, d)][1])
            for c in cells for f in args.focus for d in args.dose]
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'cd_per_image.csv'), index=False)

    el = pd.DataFrame([dict(cell=c, EL_gt=exposure_latitude(g, 'cd_gt', args.focus, args.dose),
                            EL_gen=exposure_latitude(g, 'cd_gen', args.focus, args.dose))
                       for c, g in df.groupby('cell')])
    el.to_csv(os.path.join(args.out, 'el_per_cell.csv'), index=False)

    h = df[df.split == 'held-out'].dropna(subset=['cd_gt', 'cd_gen'])
    e = h.cd_gen - h.cd_gt
    print(f'held-out images: {len(h)}')
    print(f'CD  r = {np.corrcoef(h.cd_gt, h.cd_gen)[0, 1]:.4f}   RMSE = {np.sqrt((e ** 2).mean()):.2f} nm   bias = {e.mean():+.2f} nm')
    print(f'EL  MAE = {(el.EL_gen - el.EL_gt).abs().mean():.2f} %   r = {np.corrcoef(el.EL_gt, el.EL_gen)[0, 1]:.3f}   ({len(el)} cells)')


if __name__ == '__main__':
    main()
