"""
Figures and tables for reports/blur_learning.md, from the local diagnostics (reports/blur_learning/diag)
and a copy of the pod results folder (--pod: blur_check/, depth_vs_distance/, eval/, finetune/).

    python -m depth_bench.diagnostics.report_figures --pod <copy of MODEST_runs>
Writes fig_*.png and table_*.csv to reports/blur_learning/.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "blur_learning"
DIAG = OUT / "diag"

PROBES = [  # label, check_blur folder, crops or whole images
    ("S1 crops, full model (dropout 0.2)", "camind_corr_S1_step10250"),
    ("S1 crops, sigma loss", "camind_corr_S1_probe_sigma_step3125"),
    ("S1 crops, log loss", "camind_corr_S1_probe_log_step3125"),
    ("S1 crops, log loss, dropout 0", "camind_corr_S1_probe_log_d0_step3125"),
    ("S1 crops, + texture mask", "camind_corr_S1_probe_mask_step3125"),
    ("S2 crops, log loss, dropout 0", "camind_corr_S2_probe_log_d0_step3125"),
    ("S2 crops, + texture mask", "camind_corr_S2_probe_mask_step3125"),
    ("S2 crops, wide-range curriculum, 10k", "camind_corr_S2_probe_wide_step10000"),
    ("S2 whole images 1036, 10k", "camind_corr_S2_r1036_probe_whole_step10000"),
    ("S2 whole, scene 5 held out", "camind_corr_S2_r1036_hold5_step7750"),
    ("S2 whole, full model (depth + blur)", "camind_corr_S2_r1036_whole_step12500"),
]
EVALS = [  # label, eval folder pattern ({f} = fold)
    ("Camind plain, crops (dropout 0.2)", "ft_camind_{f}_r518"),
    ("Camind plain, crops (dropout 0)", "ft_camind_{f}_d0"),
    ("Camind corrected, crops (dropout 0.2)", "ft_camind_corr_{f}"),
    ("Camind plain, whole 1036", "ft_camind_{f}_r1036_whole"),
    ("Camind corrected, whole 1036", "ft_camind_corr_{f}_r1036_whole"),
    ("Camind corrected + SNR, whole 1036", "ft_camind_corr_{f}_r1036_whole_snr"),
    ("DA3 fine-tuned", "ft_da3_{f}_r518"),
    ("UniDepth fine-tuned", "ft_unidepth_{f}_r518"),
    ("DA2 fine-tuned", "ft_da2_{f}_r518"),
    ("Metric3D fine-tuned", "ft_metric3d_{f}_r518"),
]
GROUPS = [("test_seen", "seen"), ("test_unseenF", "unseen F"), ("test_unseenfl", "unseen fl"), ("test_both", "both unseen")]
FIXES = [("corrected (reference)", "camind_corr_S1_r1036_whole"), ("+ multi-scale stage 1", "camind_corr_S1_r1036_whole_ms"),
         ("+ blur augmentation", "camind_corr_S1_r1036_whole_aug"), ("+ F-dependent base blur", "camind_corr_S1_r1036_whole_s0n")]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def synth():
    rows = []
    for f in sorted((DIAG / "synth_blur").glob("curve_*.csv")):
        _, variant, seed = f.stem.split("_")
        c = pd.read_csv(f)
        rows.append({"variant": variant, "seed": seed, "final_spearman_textured": c.spearman_textured.iloc[-1],
                     "final_median_rel_err": c.median_rel_err.iloc[-1]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_synth.csv", index=False)
    names = {"base": "base\n(wide range,\nsmooth map)", "A": "A\n(+ sharpening,\nnoise, JPEG)",
             "B": "B\n(one blur\nper crop)", "C": "C\n(only small\nblurs 0.6-2.5 px)"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, v in enumerate(["base", "A", "B", "C"]):
        y = t[t.variant == v].final_spearman_textured.values
        ax.scatter(np.full(len(y), i) + np.linspace(-0.1, 0.1, len(y)), y, s=50, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4), [names[v] for v in ["base", "A", "B", "C"]])
    ax.set_ylabel("Spearman, predicted vs true blur\n(test crops, after 3000 steps)")
    ax.set_title("Synthetic blur (diagnostic only): one dot per seed")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_synth.png")


def fit_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for f in sorted((DIAG / "fit_one_batch").glob("curve_*.csv")):
        c = pd.read_csv(f)
        c = c[c.precision == "bf16"]
        axes[0].plot(c.step, c.spearman, label=f"seed {f.stem.split('_')[1]}")
    axes[0].set(title="One fixed batch of 4 crops (memorize)", xlabel="step", ylabel="Spearman on that batch", ylim=(-0.5, 1))
    for f in sorted((DIAG / "stream_fit").glob("curve_lr0.001_drop0_jit0_*.csv")):
        c = pd.read_csv(f)
        axes[1].plot(c.step, c.pixel_spearman, label=f"seed {f.stem.split('_')[-1]}")
    axes[1].set(title="Random crops of 20 views (learn), lr 1e-3, dropout 0", xlabel="step", ylabel="Spearman on held-out crops",
                ylim=(-0.5, 1))
    for a in axes:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    save(fig, "fig_fit.png")


def probes(pod):
    rows = []
    for label, folder in PROBES:
        s = pd.read_csv(pod / "blur_check" / folder / "summary.csv")
        s = s[s.group == "all"]
        get = lambda q: s[s.quantity == q].spearman.iloc[0] if (s.quantity == q).any() else np.nan
        rows.append({"probe": label, "spearman_all": get("blur sigma (stage-1 output, px)"),
                     "spearman_textured": get("blur sigma, textured pixels"),
                     "spearman_x_stage2_input": get("|s-d|/d (stage-2 input)")})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_probes.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(t))[::-1]
    ax.barh(y + 0.18, t.spearman_all, 0.36, label="all pixels")
    ax.barh(y - 0.18, t.spearman_textured, 0.36, label="textured pixels")
    ax.set_yticks(y, t.probe, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman, stage-1 blur vs label blur (val views)")
    ax.set_title("Stage-1 probes on real MODEST images")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig_probes.png")
    shutil.copy(pod / "blur_check" / "camind_corr_S2_r1036_probe_whole_step10000" / "examples.png", OUT / "fig_probe_whole_examples.png")


def main_results(pod):
    rows = []
    for fold in ("S1", "S2"):
        for label, pat in EVALS:
            p = pod / "eval" / pat.format(f=fold) / "summary.csv"
            if not p.exists():
                continue
            s = pd.read_csv(p)
            for g, gname in GROUPS:
                r = s[s.value == g]
                rows.append({"fold": fold, "model": label, "group": gname, "absrel": r.absrel.iloc[0], "scale": r.scale.iloc[0]})
    t = pd.DataFrame(rows)
    t.pivot_table(index=["fold", "model"], columns="group", values="absrel", sort=False).round(3).to_csv(OUT / "table_main.csv")
    show = ["Camind plain, crops (dropout 0.2)", "Camind plain, whole 1036", "Camind corrected, whole 1036",
            "DA3 fine-tuned", "UniDepth fine-tuned"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fold in zip(axes, ("S1", "S2")):
        for i, m in enumerate(show):
            v = t[(t.fold == fold) & (t.model == m)].set_index("group").reindex([g for _, g in GROUPS]).absrel
            ax.bar(np.arange(4) + (i - 2) * 0.16, v.values, 0.16, label=m)
        ax.set_xticks(range(4), [g for _, g in GROUPS])
        ax.set_title(f"Fold {fold} test")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AbsRel (lower is better)")
    axes[1].legend(fontsize=7)
    save(fig, "fig_main.png")


def by_fnumber(pod):
    rows = []
    for fold in ("S1", "S2"):
        b = pd.read_csv(pod / "depth_vs_distance" / fold / "bins.csv")
        b = b[(b.F != "all") & ~b.model.str.contains(" x scale")]
        b["w"] = b.absrel * b.pixels
        g = b.groupby(["model", "F"]).sum(numeric_only=True)
        g = (g.w / g.pixels).rename("absrel").reset_index()
        g["fold"] = fold
        rows.append(g)
    t = pd.concat(rows)
    t["F"] = t.F.astype(float)
    t.pivot_table(index=["fold", "model"], columns="F", values="absrel").round(3).to_csv(OUT / "table_by_fnumber.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    held = {"S1": 2.8, "S2": 22.0}
    for ax, fold in zip(axes, ("S1", "S2")):
        for m in ("Camind plain", "Camind corrected", "DA3 fine-tuned", "DA3 zero-shot"):
            v = t[(t.fold == fold) & (t.model == m)].sort_values("F")
            ax.plot(range(len(v)), v.absrel, "o-", label=m)
        Fs = sorted(t.F.unique())
        ax.set_xticks(range(len(Fs)), [f"F{f:g}" + ("\n(held out)" if f == held[fold] else "") for f in Fs])
        ax.set_title(f"Fold {fold} test, all focal lengths")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AbsRel")
    axes[0].legend(fontsize=8)
    save(fig, "fig_by_fnumber.png")


def by_distance(pod):
    order = ["0-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-6", "6-20"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    out = []
    for ax, (fold, F, title) in zip(axes, (("S2", ["2.8", "5.0"], "Fold S2, F2.8 + F5 (seen in training)"),
                                           ("S1", ["2.8"], "Fold S1, F2.8 (held out of training)"))):
        b = pd.read_csv(pod / "depth_vs_distance" / fold / "bins.csv")
        b = b[b.F.astype(str).isin(F)].copy()
        b["w"] = b.absrel * b.pixels
        g = b.groupby(["model", "bin"]).sum(numeric_only=True)
        g = (g.w / g.pixels).rename("absrel").reset_index()
        for m in ("Camind plain", "Camind corrected", "DA3 fine-tuned", "DA3 zero-shot", "UniDepth zero-shot"):
            v = g[g.model == m].set_index("bin").reindex(order).absrel
            ax.plot(range(len(order)), v.values, "o-", label=m)
            out.append(pd.Series(v.values, index=order, name=f"{fold} {'+'.join(F)} | {m}"))
        ax.set_xticks(range(len(order)), [o + " m" for o in order], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 0.8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AbsRel (clipped at 0.8)")
    axes[0].legend(fontsize=8)
    save(fig, "fig_by_distance.png")
    pd.DataFrame(out).round(3).to_csv(OUT / "table_by_distance.csv")


def fixes(pod):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, run in FIXES:
        c = pd.read_csv(pod / "finetune" / run / "log.csv").dropna(subset=["val_absrel"])
        c = c[c.step > 0].drop_duplicates("step", keep="last")
        ax.plot(c.step, c.val_absrel.rolling(4, min_periods=1).mean(), label=label)
    ax.set(xlabel="step", ylabel="val AbsRel (rolling mean of 4 checks)", title="Fold S1: fix runs (stopped early) vs reference",
           ylim=(0.15, 0.7))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig_fixes_val.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True, help="local copy of the pod's MODEST_runs (results only)")
    a = ap.parse_args(argv)
    pod = Path(a.pod)
    synth()
    fit_curves()
    probes(pod)
    main_results(pod)
    by_fnumber(pod)
    by_distance(pod)
    fixes(pod)
    for src in ("blur_stats/blur_stats.png", "label_check/label_check.png", "label_maps/label_maps.png",
                "label_maps/texture_mask.png", "synth_blur/example_base_s0r.png", "synth_blur/example_B_s0r.png"):
        shutil.copy(DIAG / src, OUT / ("fig_" + Path(src).name))
    for fold in ("S1", "S2"):
        shutil.copy(pod / "depth_vs_distance" / fold / "error_vs_distance.png", OUT / f"fig_error_vs_distance_{fold}.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
