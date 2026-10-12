"""
Analysis of the full benchmark (all models, zero-shot and trained, 5 setting folds).

Every evaluation used the same 8196 fixed test crops, and every crop carries its label in each fold
(test_seen / test_unseenF / test_unseenfl / test_both). So besides fold-averaged tables, the same crop
can be compared when its camera setting was trained on (seen folds) and when it was held out (the
fold that hides its F-number and/or focal length): a paired comparison without scene effects.

    python -m depth_bench.analyze --runs <MODEST_runs folder> --out reports/benchmark

Writes CSV tables and PNG figures to --out.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BLUR_BINS, FOLDS

ZS = {"da2": "Depth Anything V2", "unidepth": "UniDepth V2", "unidepth_noK": "UniDepth V2 (no K)",
      "metric3d": "Metric3D v2", "da3": "Depth Anything 3"}
FT = {"da2": "Depth Anything V2", "unidepth": "UniDepth V2", "metric3d_c6000": "Metric3D v2",
      "da3": "Depth Anything 3", "camind": "Camind CNN (scratch)"}
GROUPS = ["test_seen", "test_unseenF", "test_unseenfl", "test_both"]
GNAME = {"test_seen": "seen", "test_unseenF": "unseen F", "test_unseenfl": "unseen fl", "test_both": "both unseen"}
KEY = ["color_path", "crop_index"]
METRICS = ["absrel", "delta1", "si_absrel", "silog", "rmse", "scale", "bf1"]
NB = len(BLUR_BINS) - 1
BIN_NAMES = [f"{BLUR_BINS[k]:g}-{BLUR_BINS[k + 1]:g}" for k in range(NB)]


def load(runs):
    """Long table: one row per (model, stage, fold, crop), with that fold's label."""
    parts = []
    for stage, models in (("zero-shot", ZS), ("trained", FT)):
        for m in models:
            for fold in FOLDS:
                name = f"zs_{m}" if stage == "zero-shot" else f"ft_{m}_{fold}"
                pc = pd.read_csv(Path(runs) / "eval" / name / "per_crop.csv")
                pc = pc.assign(model=m, stage=stage, fold=fold, label=pc[fold])
                parts.append(pc)
    return pd.concat(parts, ignore_index=True)


def fold_table(d):
    """Mean over crops per fold and group, then mean and std over the 5 folds."""
    per_fold = d.groupby(["stage", "model", "fold", "label"])[METRICS].mean()
    per_fold["absrel_med"] = d.groupby(["stage", "model", "fold", "label"])["absrel"].median()
    agg = per_fold.groupby(["stage", "model", "label"]).agg(["mean", "std"])
    return agg


def boot_ratio(df, num, den, n=2000, seed=0):
    """95% interval of sum(num)/sum(den) - 1 by resampling views (crops of one view are correlated)."""
    v = df.groupby("view_id")[[num, den]].sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    r = v[idx, 0].sum(1) / v[idx, 1].sum(1) - 1
    return np.percentile(r, [2.5, 97.5])


def paired(d):
    """Per crop: error when its setting was held out vs mean error over the folds where it was seen."""
    t = d[d.stage == "trained"]
    seen = t[t.label == "test_seen"].groupby(["model", *KEY])["absrel"].mean().rename("seen")
    rows = []
    for g in ["test_unseenF", "test_unseenfl", "test_both"]:
        u = t[t.label == g].set_index(["model", *KEY])[["absrel", "view_id", "f_number", "fl_mm"]]
        j = u.join(seen, how="inner").reset_index()
        for m, jm in j.groupby("model"):
            lo, hi = boot_ratio(jm, "absrel", "seen")
            rows.append({"model": m, "group": GNAME[g], "n_crops": len(jm), "n_views": jm["view_id"].nunique(),
                         "seen_absrel": jm["seen"].mean(), "unseen_absrel": jm["absrel"].mean(),
                         "rel_increase": jm["absrel"].mean() / jm["seen"].mean() - 1, "ci_lo": lo, "ci_hi": hi,
                         "frac_crops_worse": float((jm["absrel"] > jm["seen"]).mean())})
    return pd.DataFrame(rows)


def per_setting(d, col):
    """Trained models: AbsRel per F-number (or focal length) when seen vs when held out (paired by crop)."""
    t = d[d.stage == "trained"]
    held = "test_unseenF" if col == "f_number" else "test_unseenfl"
    seen = t[t.label == "test_seen"].groupby(["model", *KEY])["absrel"].mean().rename("seen")
    u = t[t.label.isin([held, "test_both"])].set_index(["model", *KEY])[["absrel", col]]
    j = u.join(seen, how="inner").reset_index()
    out = j.groupby(["model", col]).agg(seen=("seen", "mean"), unseen=("absrel", "mean"), n=("absrel", "size"))
    out["rel_increase"] = out["unseen"] / out["seen"] - 1
    return out.reset_index()


def blur_table(d):
    """Pixel-weighted AbsRel per blur bin: zero-shot (all crops), trained seen and trained unseen F."""
    rows = []
    sel = {"zero-shot": (d.stage == "zero-shot") & (d.fold == "S1"),   # zero-shot: each crop once
           "trained, seen": (d.stage == "trained") & (d.label == "test_seen"),
           "trained, unseen F": (d.stage == "trained") & d.label.isin(["test_unseenF", "test_both"])}
    for name, mask in sel.items():
        for m, g in d[mask].groupby("model"):
            for k in range(NB):
                n = g[f"npx_b{k}"].fillna(0)
                if n.sum() > 0:
                    rows.append({"set": name, "model": m, "blur_px": BIN_NAMES[k], "bin": k,
                                 "absrel": float((g[f"absrel_b{k}"].fillna(0) * n).sum() / n.sum()),
                                 "n_pixels": int(n.sum()), "n_crops": int((n > 0).sum())})
    return pd.DataFrame(rows)


def training_table(runs):
    rows = []
    for m in FT:
        for fold in FOLDS:
            rd = Path(runs) / "finetune" / f"{m}_{fold}"
            done = json.loads((rd / "done.json").read_text())
            log = pd.read_csv(rd / "log.csv")
            log = log[log.val_absrel.notna()]
            rows.append({"model": m, "fold": fold, "hours": done["time_s"] / 3600, "steps": done["steps"],
                         "val_start": log.val_absrel.iloc[0], "val_best": done["best_val_absrel"],
                         "best_step": int(log.loc[log.val_absrel.idxmin(), "step"])})
    return pd.DataFrame(rows)


def fig_groups(agg, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, (stage, models) in zip(axes, (("zero-shot", ZS), ("trained", FT))):
        x = np.arange(len(models))
        w = 0.2
        for i, g in enumerate(GROUPS):
            mu = [agg.loc[(stage, m, g), ("absrel", "mean")] for m in models]
            sd = [agg.loc[(stage, m, g), ("absrel", "std")] for m in models]
            ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, label=GNAME[g])
        ax.set_xticks(x, [models[m] for m in models], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{stage}: AbsRel per test group (mean ± std over 5 folds)", fontsize=10)
        ax.set_ylabel("AbsRel (lower is better)")
        if stage == "zero-shot":
            ax.set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_absrel_groups.png", dpi=150)
    plt.close(fig)


def fig_paired(pr, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(FT)
    x = np.arange(len(models))
    for i, g in enumerate(["unseen F", "unseen fl", "both unseen"]):
        s = pr[pr.group == g].set_index("model").loc[models]
        y = 100 * s.rel_increase
        err = [100 * (s.rel_increase - s.ci_lo), 100 * (s.ci_hi - s.rel_increase)]
        ax.bar(x + (i - 1) * 0.27, y, 0.27, yerr=err, capsize=3, label=g)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [FT[m] for m in models], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("AbsRel increase vs. seen (%)")
    ax.set_title("Same crops: error when the setting was held out vs. when it was trained on\n(95% interval, resampling views)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_paired_gap.png", dpi=150)
    plt.close(fig)


def fig_blur(bt, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in ZS:
        s = bt[(bt.set == "zero-shot") & (bt.model == m)]
        axes[0].plot(s.bin, s.absrel, "o-", label=ZS[m])
    for m in FT:
        for name, ls in (("trained, seen", "-"), ("trained, unseen F", "--")):
            s = bt[(bt.set == name) & (bt.model == m)]
            axes[1].plot(s.bin, s.absrel, "o" + ls, color=f"C{list(FT).index(m)}",
                         label=f"{FT[m]} ({name.split(', ')[1]})")
    for ax, t in zip(axes, ("zero-shot", "trained (solid: seen F-number, dashed: held-out F-number)")):
        ax.set_xticks(range(NB), BIN_NAMES)
        ax.set_xlabel("blur-circle diameter (px, from true depth and focus distance)")
        ax.set_ylabel("AbsRel (pixel-weighted)")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "fig_blur.png", dpi=150)
    plt.close(fig)


def fig_scale_bf1(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    zs = d[(d.stage == "zero-shot") & (d.fold == "S1")]
    for m in ZS:
        s = zs[zs.model == m].groupby("fl_mm")["scale"].median()
        axes[0].plot(s.index, s.values, "o-", label=ZS[m])
    axes[0].axhline(1, color="k", lw=0.8)
    axes[0].set_yscale("log")
    ticks = [0.15, 0.2, 0.3, 0.5, 1, 2, 3, 5]
    axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel("focal length (mm)")
    axes[0].set_ylabel("median scale = true / predicted")
    axes[0].set_title("zero-shot: overall scale error by focal length (1 = right)", fontsize=10)
    axes[0].legend(fontsize=7)
    for stage, models, ls in (("zero-shot", ZS, "--"), ("trained", FT, "-")):
        s = d[(d.stage == stage) & ((d.fold == "S1") if stage == "zero-shot" else True)]
        for m in models:
            if stage == "zero-shot" and m not in ("da2", "da3", "unidepth"):
                continue
            v = s[s.model == m].groupby("f_number")["bf1"].mean()
            name = (ZS if stage == "zero-shot" else FT)[m]
            axes[1].plot(v.index, v.values, "o" + ls, label=f"{name} ({stage})")
    axes[1].set_xscale("log")
    axes[1].set_xticks([2.8, 5, 9, 16, 22], ["2.8", "5", "9", "16", "22"])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_xlabel("F-number")
    axes[1].set_ylabel("boundary F1 (higher = sharper edges)")
    axes[1].set_title("depth-edge accuracy by F-number", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_scale_bf1.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="reports/benchmark")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.runs)
    agg = fold_table(d)
    agg.to_csv(out / "table_groups.csv")
    pr = paired(d)
    pr.to_csv(out / "table_paired.csv", index=False)
    pf = per_setting(d, "f_number")
    pf.to_csv(out / "table_by_fnumber.csv", index=False)
    pl = per_setting(d, "fl_mm")
    pl.to_csv(out / "table_by_focal.csv", index=False)
    bt = blur_table(d)
    bt.to_csv(out / "table_blur.csv", index=False)
    tr = training_table(a.runs)
    tr.to_csv(out / "table_training.csv", index=False)
    zs = d[(d.stage == "zero-shot") & (d.fold == "S1")]
    zs.groupby(["model", "fl_mm"])["scale"].median().unstack().to_csv(out / "table_zs_scale_by_focal.csv")
    d[(d.stage == "zero-shot") & (d.fold == "S1") | (d.stage == "trained")].groupby(
        ["stage", "model", "f_number"])["bf1"].mean().unstack().to_csv(out / "table_bf1_by_fnumber.csv")

    fig_groups(agg, out)
    fig_paired(pr, out)
    fig_blur(bt, out)
    fig_scale_bf1(d, out)
    print(f"tables and figures in {out}")


if __name__ == "__main__":
    main()
