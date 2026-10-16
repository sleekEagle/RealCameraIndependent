"""Helpers shared by the stage-1 blur diagnostics (reports/blur_learning.md)."""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import data
from ..models import BLUR_K, SIGMA0_PX

OUT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "blur_learning" / "diag"
BINS = [0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 64]   # blur sigma bins (px)


def out_dir(name, out=None):
    p = Path(out) if out else OUT_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def blur_scale(row):
    """C = k f^2 / (N (s - f)) (px per unit |s - d| / d) and the focus distance s of a manifest row."""
    v, N, s = (float(t) for t in data.camera(row))
    f = v * s / (v + s)
    return BLUR_K * f ** 2 / (N * (s - f)), s


def label_sigma(depth, row, sigma0=SIGMA0_PX):
    """Label blur sigma (native px) for a depth map: sqrt(sigma0^2 + (C |s - d| / d)^2)."""
    C, s = blur_scale(row)
    return np.sqrt(sigma0 ** 2 + (C * np.abs(s - depth) / depth) ** 2)


def spearman(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def gauss_blur(x, s):
    """Separable Gaussian blur (sigma s px) of a (B, C, H, W) tensor, reflect padding."""
    k = int(math.ceil(3 * s))
    t = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-t ** 2 / (2 * s * s))
    g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k, k, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k, k), mode="reflect"), g.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def rows_with_focus(rows):
    return [r for r in rows if np.isfinite(float(data.camera(r)[2]))]
