"""
Blur statistics of the MODEST labels on a fold's training views (no network involved).

Label blur sigma = sqrt(sigma0^2 + (C |s - d| / d)^2), C = k f^2 / (N (s - f)), native px. Reports
  - per F-number: C, label sigma (median, 95th pct), share of pixels whose defocus C |s-d|/d is below sigma0;
  - the share of the x-space stage-1 loss each F-number carries (a blur error e costs (e / (C x_scale))^2);
  - losses of constant outputs: best constant under the x-space loss (camind_corr0), under the pixel
    loss, and under the log loss (one constant for all pixels / one per image);
  - the blur range inside 518 px crops: log2(p95 / p5) of the label sigma.

    python -m depth_bench.diagnostics.blur_stats --folds S1 S2 --images 120
Writes per_fnumber.csv, constants.csv, crop_range.csv, blur_stats.png to --out.
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .. import data
from ..models import SIGMA0_PX, X_SCALE
from .common import blur_scale, out_dir, rows_with_focus

EPS = 0.05   # as CamindW.LOG_EPS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["S1", "S2"])
    ap.add_argument("--images", type=int, default=120, help="training views sampled per fold")
    ap.add_argument("--crops", type=int, default=10, help="518 px crop windows per view for the range")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("blur_stats", a.out)

    per_f, consts, ranges, px_all = [], [], [], []
    for fold in a.folds:
        rows = rows_with_focus(data.select(data.read_manifest(), fold, ("train",)))
        rng = random.Random(0)
        recs = []
        for r in rng.sample(rows, min(a.images, len(rows))):
            C, s = blur_scale(r)
            d = data.load_depth(r)[::8, ::8]
            x = np.abs(s - d) / d
            sig = np.sqrt(SIGMA0_PX ** 2 + (C * x) ** 2)
            ok = np.isfinite(sig)
            N = float(r["f_number"])
            recs.append(pd.DataFrame({"view": r["color_path"], "F": N, "C": C, "x": x[ok], "sigma": sig[ok]}))
            ls = np.log2(sig)
            H, W = ls.shape
            for _ in range(a.crops):   # 518 px crop = 65 px at stride 8
                y0, x0 = rng.randrange(H - 65), rng.randrange(W - 65)
                w = ls[y0:y0 + 65, x0:x0 + 65]
                w = w[np.isfinite(w)]
                if w.size >= 2000:
                    ranges.append({"fold": fold, "F": N, "log2_p95_p5": np.percentile(w, 95) - np.percentile(w, 5)})
        df = pd.concat(recs)
        df["fold"] = fold
        px_all.append(df)
        df["w_x"] = 1 / (df.C * X_SCALE) ** 2
        for N, g in df.groupby("F"):
            per_f.append({"fold": fold, "F": N, "C_median_px": g.C.median(), "sigma_median_px": g.sigma.median(),
                          "sigma_p95_px": g.sigma.quantile(0.95),
                          "share_defocus_below_sigma0": float((g.C * g.x < SIGMA0_PX).mean()),
                          "share_sigma_below_1.5px": float((g.sigma < 1.5).mean()),
                          "share_of_x_loss_weight": g.w_x.sum() / df.w_x.sum()})
        a_ = 1 / (df.C * X_SCALE)
        l = np.log(df.sigma + EPS)
        consts.append({"fold": fold,
                       "x_loss_best_constant_sigma_px": float((a_ * df.x / X_SCALE).sum() / (a_ ** 2).sum()),
                       "pixel_loss_best_constant_sigma_px": float(df.sigma.mean()),
                       "log_loss_single_constant_mse": float(l.var()),
                       "log_loss_constant_per_view_mse": float(df.assign(l=l).groupby("view").l.var(ddof=0).mean()),
                       "sigma_median_px": float(df.sigma.median()), "sigma_p95_px": float(df.sigma.quantile(0.95))})
    per_f, consts, rg = pd.DataFrame(per_f), pd.DataFrame(consts), pd.DataFrame(ranges)
    per_f.to_csv(out / "per_fnumber.csv", index=False)
    consts.to_csv(out / "constants.csv", index=False)
    rows_r = []
    for (fold, N), g in list(rg.groupby(["fold", "F"])) + [((f, "all"), g) for f, g in rg.groupby("fold")]:
        v = g.log2_p95_p5
        rows_r.append({"fold": fold, "F": N, "crops": len(v), "ratio_median": 2 ** v.median(),
                       "ratio_p75": 2 ** v.quantile(.75), "ratio_p90": 2 ** v.quantile(.9),
                       **{f"share_ratio_ge_{2 ** t:g}x": float((v >= t).mean()) for t in (1, 1.5, 2, 3)}})
    pd.DataFrame(rows_r).to_csv(out / "crop_range.csv", index=False)
    pd.set_option("display.width", 220)
    print(per_f.round(3).to_string(index=False), "\n")
    print(consts.round(3).to_string(index=False), "\n")
    print(pd.DataFrame(rows_r).round(3).to_string(index=False))

    px = pd.concat(px_all)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.geomspace(0.7, 40, 50)
    for (fold, N), g in px.groupby(["fold", "F"]):
        ax[0].hist(g.sigma, bins=bins, histtype="step", density=True, label=f"{fold} F{N:g}",
                   ls="-" if fold == a.folds[0] else "--")
    ax[0].axvline(SIGMA0_PX, color="k", lw=0.8)
    ax[0].set(xscale="log", xlabel="label blur sigma (native px)", ylabel="density",
              title="label blur per F-number (line: base blur)")
    ax[0].legend(fontsize=7, ncol=2)
    for (fold, N), g in rg.groupby(["fold", "F"]):
        v = np.sort(2 ** g.log2_p95_p5.values)
        ax[1].plot(v, np.arange(1, len(v) + 1) / len(v), ls="-" if fold == a.folds[0] else "--", label=f"{fold} F{N:g}")
    ax[1].set(xscale="log", xlabel="blur ratio inside a 518 px crop (p95 / p5)", ylabel="share of crops (CDF)",
              title="blur range within crops")
    ax[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%gx"))
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "blur_stats.png", dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
