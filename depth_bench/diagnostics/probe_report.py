"""
Figures and a table from stage-1 probe runs (pod outputs): training-loss curves from the finetune logs,
and from check_blur summaries the predicted vs true median blur per true-blur bin and the overall
Spearman correlations.

    python -m depth_bench.diagnostics.probe_report \
        --logs "S2 whole 1036=probes/probe_whole_S2.log" "S2 wide crops=probes/probe_wide_S2.log" \
        --checks "S2 whole 1036=probes/blur_check/camind_corr_S2_r1036_probe_whole_step10000" ...
Writes loss_curves.png, bins.png, checks.csv to --out.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import out_dir

STEP = re.compile(r"^step (\d+)/(\d+) loss ([0-9.eE+-]+)")


def read_log(path):
    pts = [(int(m.group(1)), float(m.group(3))) for m in map(STEP.match, Path(path).read_text().splitlines()) if m]
    return pd.DataFrame(pts, columns=["step", "loss"])


def blur_rows(summary):
    """Rows of the blur quantity (textured pixels when present) of a check_blur summary."""
    q = "blur sigma, textured pixels" if (summary.quantity == "blur sigma, textured pixels").any() \
        else "blur sigma (stage-1 output, px)"
    return summary[summary.quantity == q], q


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[], help="label=finetune log")
    ap.add_argument("--checks", nargs="*", default=[], help="label=check_blur output folder")
    ap.add_argument("--smooth", type=int, default=25, help="rolling mean over this many log lines")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("probe_report", a.out)

    if a.logs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for spec in a.logs:
            label, path = spec.split("=", 1)
            d = read_log(path)
            ax.plot(d.step, d.loss.rolling(a.smooth, min_periods=1).mean(), label=label)
        ax.set(xlabel="optimizer step", ylabel="training loss (rolling mean)", yscale="log", title="stage-1 probes")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "loss_curves.png", dpi=110)

    if a.checks:
        table = []
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for spec in a.checks:
            label, folder = spec.split("=", 1)
            s = pd.read_csv(Path(folder) / "summary.csv")
            rows, q = blur_rows(s)
            allrow = rows[rows.group == "all"].iloc[0]
            alls = s[(s.group == "all") & (s.quantity == "blur sigma (stage-1 output, px)")].iloc[0]
            table.append({"run": label, "quantity": q, "spearman_textured_or_all": allrow.spearman,
                          "spearman_all_pixels": alls.spearman, "true_median": allrow.true_median,
                          "pred_median": allrow.pred_median,
                          **{f"spearman_{g}": r.spearman for g, r in rows.set_index("group").iterrows()
                             if g.startswith("F")}})
            b = rows[rows.group.str.startswith("true sigma") & (rows.pixels >= 2000)]
            ax.plot(b.true_median, b.pred_median, "o-", label=label)
        lim = [0.3, 20]
        ax.plot(lim, lim, "k--", lw=1, label="prediction = truth")
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="true median blur in bin (px)",
               ylabel="predicted median blur in bin (px)", title="stage-1 output per true-blur bin")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "bins.png", dpi=110)
        t = pd.DataFrame(table)
        t.to_csv(out / "checks.csv", index=False)
        pd.set_option("display.width", 220)
        print(t.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
