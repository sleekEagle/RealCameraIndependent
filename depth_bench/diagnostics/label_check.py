"""
Do the blur labels match the images? A classical blur estimate from the image alone (re-blur gradient
ratio, Zhuo & Sim 2011: R = |grad I| / |grad (I * G_sr)|, sigma = sr / sqrt(R^2 - 1)) at strong edges,
compared with the label sigma = sqrt(sigma0^2 + (C |s - d| / d)^2). Only texture edges inside a surface
(no depth step in a 9x9 window). The estimate is reliable for about 1-8 px (it reads above ~10 px low).

    python -m depth_bench.diagnostics.label_check --per_f 20
Writes images.csv, bins.csv, by_fnumber.csv, label_check.png to --out.
"""
import argparse
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import data
from ..models import SIGMA0_PX
from .common import blur_scale, out_dir, rows_with_focus, spearman

SR = 2.0   # re-blur sigma (px)
BINS = [0, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 64]


def srgb_lin(u8):
    c = u8.astype(np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def grad(img):
    return np.hypot(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))


def edge_blur(img, depth):
    """Edge pixels (mask) and their blur estimate (px)."""
    y = srgb_lin(img) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    g1, g2 = grad(y), grad(cv2.GaussianBlur(y, (0, 0), SR))
    u8 = cv2.GaussianBlur((np.clip(y, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), (0, 0), 1.0)
    edges = cv2.Canny(u8, 20, 60) > 0
    dmax = cv2.dilate(np.nan_to_num(depth, nan=0), np.ones((9, 9)))
    dmin = -cv2.dilate(-np.nan_to_num(depth, nan=1e3), np.ones((9, 9)))
    ok = edges & np.isfinite(depth) & ((dmax - dmin) / depth < 0.03) & (g1 > np.percentile(g1[edges], 50))
    R = g1[ok] / np.maximum(g2[ok], 1e-6)
    return ok, np.minimum(np.where(R > 1.0005, SR / np.sqrt(np.maximum(R ** 2 - 1, 1e-6)), 64.0), 64.0)


def synthetic_check():
    """The estimator on a blurred step edge with known sigma."""
    out = []
    for s in [0.5, 1, 2, 4, 8, 12]:
        im = np.zeros((200, 400), np.float32)
        im[:, 200:] = 1
        b = cv2.GaussianBlur(im, (0, 0), s) + np.random.default_rng(0).normal(0, 0.002, im.shape).astype(np.float32)
        R = grad(b)[100, 195:205].max() / grad(cv2.GaussianBlur(b, (0, 0), SR))[100, 195:205].max()
        out.append({"true_sigma": s, "estimate": SR / np.sqrt(R * R - 1)})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_f", type=int, default=20, help="images per F-number (all folds, all views)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("label_check", a.out)
    syn = synthetic_check()
    syn.to_csv(out / "estimator_synthetic.csv", index=False)
    print("estimator on synthetic edges:\n", syn.round(2).to_string(index=False))

    rows = rows_with_focus(data.read_manifest())
    rng = random.Random(0)
    by_f = {}
    for r in rows:
        by_f.setdefault(float(r["f_number"]), []).append(r)
    recs, ims = [], []
    for N, rs in sorted(by_f.items()):
        for r in rng.sample(rs, min(a.per_f, len(rs))):
            img, depth = data.load_pair(r)
            C, s = blur_scale(r)
            ok, est = edge_blur(img, depth)
            if ok.sum() < 200:
                continue
            d = depth[ok]
            true = np.sqrt(SIGMA0_PX ** 2 + (C * np.abs(s - d) / d) ** 2)
            recs.append(pd.DataFrame({"F": N, "est": est, "true": true}))
            ims.append({"view": r["color_path"], "F": N, "C": C, "s": s, "edge_pixels": int(ok.sum()),
                        "est_median": float(np.median(est)), "true_median": float(np.median(true)),
                        "within_spearman": spearman(est, true)})
            print(f"F{N:g} C={C:5.2f} n={ok.sum():7d} est {np.median(est):5.2f} label {np.median(true):5.2f} "
                  f"rho {ims[-1]['within_spearman']:+.2f}  {r['color_path']}", flush=True)
    df, im = pd.concat(recs), pd.DataFrame(ims)
    im.to_csv(out / "images.csv", index=False)
    byf = [{"F": N, "edge_pixels": len(g), "spearman": spearman(g.est, g.true), "est_median": g.est.median(),
            "label_median": g.true.median()} for N, g in df.groupby("F")]
    byf.append({"F": "all", "edge_pixels": len(df), "spearman": spearman(df.est, df.true),
                "est_median": df.est.median(), "label_median": df.true.median(),
                "across_image_spearman": spearman(im.est_median, im.true_median),
                "median_within_image_spearman": im.within_spearman.median()})
    pd.DataFrame(byf).to_csv(out / "by_fnumber.csv", index=False)
    df["bin"] = pd.cut(df.true, BINS)
    bins = df.groupby("bin", observed=True).est.agg(
        pixels="size", est_median="median", est_p25=lambda x: x.quantile(.25), est_p75=lambda x: x.quantile(.75))
    bins["share_of_pixels"] = bins.pixels / bins.pixels.sum()
    bins.to_csv(out / "bins.csv")
    print(pd.DataFrame(byf).round(3).to_string(index=False), "\n", bins.round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    mid = [iv.mid for iv in bins.index]
    ax[0].plot(mid, bins.est_median, "o-", label="median estimate")
    ax[0].fill_between(mid, bins.est_p25, bins.est_p75, alpha=.3, label="25-75%")
    ax[0].plot([0.7, 16], [0.7, 16], "k--", lw=1, label="estimate = label")
    ax[0].set(xscale="log", yscale="log", xlabel="label sigma (px)", ylabel="edge-based estimate (px)",
              title="edge pixels, all F-numbers")
    ax[0].legend()
    for N, g in im.groupby("F"):
        ax[1].scatter(g.true_median, g.est_median, label=f"F{N:g}", s=18)
    ax[1].plot([0.5, 16], [0.5, 16], "k--", lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="label sigma, view median (px)", ylabel="estimate, view median (px)",
              title="per view")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "label_check.png", dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
