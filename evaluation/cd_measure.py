"""Row-wise CD and LWR measurement of line-space SEM images (Appendix B of the paper).

Images are analyzed at 256 x 256 px (1 px = 6 nm). The center of every printed line is
taken from the layout mask; for each image row, the edges of the line are located on
the 5-px moving-average intensity profile as the positions where the profile crosses
70% of the way from the space minimum to the bright edge peak, with linear
interpolation between pixels. Rows within 12 px of the top and bottom borders are
excluded, lines cut by the left or right border are skipped, and lines with fewer than
10 valid rows are discarded.

  CD  of an image = mean over lines of the mean row width
  LWR of an image = mean over lines of 3 sigma of the row width
"""
import re

import numpy as np
from PIL import Image

SIZE = 256          # analysis resolution (px)
NM_PER_PX = 6.0     # pixel size at SIZE
BUFFER = 12         # excluded rows at the top and bottom (px)
LEVEL = 0.70        # threshold level between space minimum and edge peak
SMOOTH = 5          # moving-average window (px)


def pitch_of(cell):
    """Pitch in nm, parsed from a cell name such as ls_p200_fill_cd100."""
    return int(re.search(r'_p(\d+)', cell).group(1))


def load_gray(path):
    im = Image.open(path).convert('L')
    if im.size[0] != SIZE:
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return np.array(im)


def line_centers(mask_path):
    """Column centers of the lines of a layout mask, read on its middle row."""
    a = np.array(Image.open(mask_path).convert('L').resize((SIZE, SIZE), Image.NEAREST))
    mid = (a[SIZE // 2, :] > 127).astype(int)
    e = np.diff(np.concatenate([[0], mid, [0]]))
    s, t = np.where(e == 1)[0], np.where(e == -1)[0]
    return [int((p + q) / 2) for p, q in zip(s, t) if q > p]


def threshold_edges(prof, mid, level=LEVEL):
    """Left and right edge (px) of the line around index `mid` of a smoothed 1-D profile."""
    n = len(prof)
    if mid < 3 or mid > n - 4:
        return np.nan, np.nan
    pL = int(np.argmax(prof[:mid]))                       # bright edge peak, left side
    fL = int(np.argmin(prof[:pL + 1])) if pL > 0 else 0   # space minimum, left side
    pR = mid + int(np.argmax(prof[mid:]))
    fR = pR + int(np.argmin(prof[pR:]))
    if pL <= fL or fR <= pR:
        return np.nan, np.nan
    tL = prof[fL] + level * (prof[pL] - prof[fL])
    tR = prof[fR] + level * (prof[pR] - prof[fR])
    segL = prof[fL:pL + 1]
    iL = np.where(segL >= tL)[0]
    if len(iL) == 0:
        return np.nan, np.nan
    i = iL[0]
    xl = fL + (i if i == 0 else i - 1 + (tL - segL[i - 1]) / (segL[i] - segL[i - 1]))
    segR = prof[pR:fR + 1]
    iR = np.where(segR <= tR)[0]
    if len(iR) == 0:
        return np.nan, np.nan
    j = iR[0]
    xr = pR + (j if j == 0 else j - 1 + (tR - segR[j - 1]) / (segR[j] - segR[j - 1]))
    return float(xl), float(xr)


def row_edges(img, centers, pitch_nm):
    """Per-line, per-row edge positions (px). Returns L, R of shape (n_lines, n_rows), NaN for rejected rows."""
    ppx = int(round(pitch_nm / NM_PER_PX))
    half = int(ppx * 0.5)
    h, w = img.shape
    rows = range(BUFFER, h - BUFFER)
    k = np.ones(SMOOTH) / SMOOTH
    Ls, Rs = [], []
    for xc in centers:
        if xc - half < 0 or xc + half > w - 1:
            continue
        x0, x1 = xc - half, xc + half
        L = np.full(len(rows), np.nan)
        R = np.full(len(rows), np.nan)
        for i, y in enumerate(rows):
            sm = np.convolve(img[y, x0:x1].astype(float), k, mode='same')
            lx, rx = threshold_edges(sm, xc - x0)
            if np.isfinite(lx) and np.isfinite(rx) and 0 < rx - lx < (x1 - x0):
                L[i], R[i] = x0 + lx, x0 + rx
        if np.isfinite(L).sum() >= 10:
            Ls.append(L)
            Rs.append(R)
    return np.array(Ls).reshape(-1, len(rows)), np.array(Rs).reshape(-1, len(rows))


def measure_image(path, mask_path, pitch_nm):
    """(CD, LWR) of one image in nm; NaN if no line can be measured."""
    L, R = row_edges(load_gray(path), line_centers(mask_path), pitch_nm)
    if len(L) == 0:
        return np.nan, np.nan
    width = (R - L) * NM_PER_PX
    cd = float(np.nanmean(width, axis=1).mean())
    lwr = float(np.mean(3 * np.nanstd(width, axis=1)))
    return cd, lwr
