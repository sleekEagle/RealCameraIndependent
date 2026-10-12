"""
def_calibration/scene_motion_report.py -- tables and figures for reports/scene_motion_blur.md, from
scene_motion_blur.py's repeat_pairs.csv and the scene EXIF table.

    python def_calibration/scene_motion_report.py          (torch_plot env)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path(r"D:\datasets\MODEST_focus\scene_motion")
OUT = Path(__file__).resolve().parents[1] / "reports" / "scene_motion"
MAIN_N = [2.8, 5.0, 9.0, 16.0, 22.0]


def q(x, p):
    return float(np.nanpercentile(x, p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(D / "repeat_pairs.csv")
    df = df[df.dvx.notna() & df.N.isin(MAIN_N)].copy()
    df["ax"], df["ay"] = df.dvx.abs(), df.dvy.abs()
    # quadrature difference in blur between the two shots (px): a lower bound on the more-moved shot's motion
    df["mx"], df["my"] = np.sqrt(df.ax), np.sqrt(df.ay)
    null = df[df.N == 2.8]
    thr_x, thr_y = q(null.ax, 95), q(null.ay, 95)

    rows = []
    for N, g in df.groupby("N"):
        rows.append({"N": N, "pairs": len(g), "exposure_median_s": q(g.exp_s, 50), "exposure_p90_s": q(g.exp_s, 90),
                     "dx_median_px": q(g.mx, 50), "dx_p90_px": q(g.mx, 90), "dy_median_px": q(g.my, 50),
                     "dy_p90_px": q(g.my, 90), "share_dx_over_0.5px": float((g.mx > 0.5).mean()),
                     "share_dx_over_1px": float((g.mx > 1).mean()), "share_dy_over_0.5px": float((g.my > 0.5).mean()),
                     "share_above_F2.8_p95_x": float((g.ax > thr_x).mean())})
    byN = pd.DataFrame(rows).round(3)
    byN.to_csv(OUT / "table_by_fnumber.csv", index=False)
    print(byN.to_string(index=False))

    # by exposure time (all F-numbers): motion should grow with exposure
    bins = [0, 1 / 250, 1 / 60, 1 / 15, 1 / 4, 1 / 2, 1.01, 10]
    labels = ["<1/250", "1/250-1/60", "1/60-1/15", "1/15-1/4", "1/4-1/2", "1/2-1", ">1"]
    df["exp_bin"] = pd.cut(df.exp_s, bins, labels=labels)
    byE = df.groupby("exp_bin", observed=True).agg(pairs=("mx", "size"), dx_median=("mx", "median"),
                                                  dx_p90=("mx", lambda x: q(x, 90)), dy_median=("my", "median"),
                                                  share_dx_over_1px=("mx", lambda x: float((x > 1).mean()))).round(3)
    byE.to_csv(OUT / "table_by_exposure.csv")
    print(byE.to_string())

    # by focal length, long exposures only (F16 and F22)
    lg = df[df.N >= 16]
    byF = lg.groupby("fl_mm").agg(pairs=("mx", "size"), dx_median=("mx", "median"), dx_p90=("mx", lambda x: q(x, 90)),
                                  dy_median=("my", "median")).round(3)
    byF.to_csv(OUT / "table_by_focal_F16_F22.csv")
    print(byF.to_string())

    # by scene, F16 + F22
    byS = lg.groupby("scene").agg(pairs=("mx", "size"), exposure_median=("exp_s", "median"),
                                  dx_median=("mx", "median"), dx_p90=("mx", lambda x: q(x, 90)),
                                  dy_median=("my", "median")).round(3)
    byS.to_csv(OUT / "table_by_scene_F16_F22.csv")
    print(byS.to_string())

    # by camera body
    byC = lg.groupby("camera").agg(pairs=("mx", "size"), dx_median=("mx", "median"), dy_median=("my", "median")).round(3)
    print(byC.to_string())
    # left and right camera of the same repeat (same manifest row, so same dup_score): do they move together?
    lr = df.pivot_table(index=["scene", "fl_mm", "N", "dup_score"], columns="side", values="dvx").dropna()
    corr = {N: round(float(g.Left.corr(g.Right, method="spearman")), 3)
            for N, g in lr.groupby(level="N")}
    print("left-right rank correlation of horizontal blur difference, per F-number:", corr)
    pd.Series(corr, name="spearman_left_right_dvx").to_csv(OUT / "table_left_right_corr.csv")

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, col, name in zip(axs, ("mx", "my"), ("horizontal", "vertical")):
        data = [df[df.N == N][col].clip(upper=4) for N in MAIN_N]
        ax.boxplot(data, labels=[f"F{N:g}\n{q(df[df.N == N].exp_s, 50):.3g} s" for N in MAIN_N],
                   showfliers=False, whis=(10, 90))
        ax.set_title(f"{name} blur difference between repeat shots")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("sqrt(|difference in blur variance|) (px)")
    fig.suptitle("Repeat shots of the same view: boxes 25-75%, whiskers 10-90%; median exposure under each F-number")
    fig.tight_layout()
    fig.savefig(OUT / "fig_by_fnumber.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N in MAIN_N:
        g = df[df.N == N]
        ax.scatter(g.exp_s, g.mx.clip(upper=5), s=3, alpha=0.3, label=f"F{N:g}")
    t = df.groupby("exp_bin", observed=True).agg(e=("exp_s", "median"), m=("mx", "median"))
    ax.plot(t.e, t.m, "k-o", label="median")
    ax.set_xscale("log")
    ax.set_xlabel("exposure time (s)")
    ax.set_ylabel("horizontal blur difference (px)")
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(alpha=0.3)
    ax.set_title("horizontal blur difference between repeat shots vs exposure")
    fig.tight_layout()
    fig.savefig(OUT / "fig_vs_exposure.png", dpi=120)
    plt.close(fig)
    print(f"null (F2.8) 95th percentile: dvx {thr_x:.3f}, dvy {thr_y:.3f} px^2 -> {OUT}")


if __name__ == "__main__":
    main()
