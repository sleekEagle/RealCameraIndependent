"""
Depth error vs true distance for several models on the same test views and pixels, plus two checks for
fusing a defocus model (camera-corrected Camind) with a SOTA depth model:

  1. AbsRel and delta1 per true-depth bin (all F-numbers, and per F-number) for every model; for each
     model also with the best scale per view (median gt / pred), which separates shape from scale error.
  2. Scale from defocus: per view, the scale a defocus model implies, median(pred / gt) over all pixels
     and over near pixels; and a fusion preview: each baseline rescaled per view by
     alpha = median(pred_defocus / pred_baseline) over the pixels the defocus model puts nearer than
     --near m (all pixels if too few), scored like the other models.

Every model sees the whole view (resized to its short side), the prediction is upsampled to full
resolution and scored in the benchmark's fixed crop windows (data.eval_crop_positions), as evaluate.py.
All models are loaded at once and run on each view in turn.

    python -m depth_bench.diagnostics.depth_vs_distance --fold S2 [--max_images 400]
Writes bins.csv, views.csv, summary.txt and figures to <RUNS_DIR>/depth_vs_distance/<fold>/.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import data, models
from ..config import RUNS_DIR, TEST_LABELS
from ..evaluate import stratified_subset
from ..train_utils import autocast_ctx, load_trainable

DEPTH_BINS = [0, 1, 1.5, 2, 3, 4, 6, 20]
# label, model, input short side, run folder under finetune/ ({fold} filled in; None = zero-shot), kwargs
MODELS = [
    ("Camind plain", "camind", 1036, "camind_{fold}_r1036_whole", {}),
    ("Camind corrected", "camind_corr", 1036, "camind_corr_{fold}_r1036_whole", {}),
    ("Camind corrected+SNR", "camind_corr", 1036, "camind_corr_{fold}_r1036_whole_snr", {"stage2_snr": True}),
    ("DA3 fine-tuned", "da3", 518, "da3_{fold}_r518", {}),
    ("UniDepth fine-tuned", "unidepth", 518, "unidepth_{fold}_r518", {}),
    ("DA2 fine-tuned", "da2", 518, "da2_{fold}_r518", {}),
    ("Metric3D fine-tuned", "metric3d", 518, "metric3d_{fold}_r518", {}),
    ("DA3 zero-shot", "da3", 518, None, {}),
    ("UniDepth zero-shot", "unidepth", 518, None, {}),
    ("DA2 zero-shot", "da2", 518, None, {}),
    ("Metric3D zero-shot", "metric3d", 518, None, {}),
]
DEFOCUS = ("Camind corrected", "Camind corrected+SNR")


class Views(Dataset):
    """Full-resolution depth, crop windows and the view resized to each needed short side."""

    def __init__(self, rows, shorts, crop=518, crops=6):
        self.rows, self.shorts, self.crop, self.crops = rows, sorted(shorts), crop, crops

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img, depth = data.load_pair(row)
        out = {"i": i, "depth": torch.from_numpy(depth), "cam": data.camera(row),
               "positions": data.eval_crop_positions(row, depth, self.crop, self.crops)}
        for s in self.shorts:
            im, sc = data.resize_image(img, s)
            out[s] = {"image": torch.from_numpy(im).permute(2, 0, 1).float() / 255.0,
                      "K": torch.from_numpy(data.resize_intrinsics(row, sc)), "px_scale": torch.tensor(float(sc))}
        return out


def build(spec, fold):
    label, name, short, run, kw = spec
    m = models.build(name, **kw).cuda().eval()
    if run:
        ck = RUNS_DIR / "finetune" / run.format(fold=fold) / "best.pt"
        if not ck.exists():
            return None
        models.set_trainable(m, 0)
        load_trainable(m, ck)
    return m


class Acc:
    """Pixel sums per (model, F, depth bin): AbsRel, delta1, AbsRel at the best per-view scale."""

    def __init__(self):
        self.s = defaultdict(lambda: np.zeros(4))   # pixels, sum absrel, sum delta1 hits, sum oracle absrel

    def add(self, label, N, pred, gt, bins):
        ok = torch.isfinite(pred) & (pred > 0)
        p, g, b = pred[ok], gt[ok], bins[ok]
        if p.numel() < 100:
            return None
        s_or = torch.median(g / p)
        ar = (p - g).abs() / g
        ar_or = (s_or * p - g).abs() / g
        d1 = (torch.maximum(p / g, g / p) < 1.25).float()
        nb = len(DEPTH_BINS) - 1
        cnt = torch.bincount(b, minlength=nb)
        sums = [torch.bincount(b, weights=w, minlength=nb) for w in (ar, d1, ar_or)]
        for k in range(nb):
            if cnt[k] > 0:
                v = np.array([cnt[k].item()] + [x[k].item() for x in sums])
                self.s[(label, N, k)] += v
                self.s[(label, "all", k)] += v
        return float(s_or)

    def table(self):
        rows = []
        for (label, N, k), (n, a, d, o) in self.s.items():
            rows.append({"model": label, "F": N, "bin": f"{DEPTH_BINS[k]:g}-{DEPTH_BINS[k + 1]:g}", "bin_index": k,
                         "pixels": int(n), "absrel": a / n, "delta1": d / n, "absrel_best_scale": o / n})
        return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S2")
    ap.add_argument("--max_images", type=int, help="stratified subset of the fold's test views")
    ap.add_argument("--near", type=float, default=2.0, help="m: pixels the defocus model puts nearer give the scale")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else RUNS_DIR / "depth_vs_distance" / a.fold
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in data.read_manifest() if r[a.fold] in TEST_LABELS]
    if a.max_images:
        rows = stratified_subset(rows, a.fold, a.max_images)
    nets = []
    for spec in MODELS:
        m = build(spec, a.fold)
        if m is None:
            print(f"skipped {spec[0]}: no checkpoint", flush=True)
            continue
        nets.append((spec[0], spec[2], m))
        print(f"loaded {spec[0]}", flush=True)
    labels = [l for l, _, _ in nets]
    shorts = {s for _, s, _ in nets}
    defocus = [d for d in DEFOCUS if d in labels]
    baselines = [l for l in labels if not l.startswith("Camind")]
    print(f"{len(rows)} test views of {a.fold}; models: {labels}", flush=True)

    acc, views = Acc(), []
    dl = DataLoader(Views(rows, shorts), batch_size=None, num_workers=a.workers)
    edges = torch.tensor(DEPTH_BINS[1:-1]).cuda()
    for n_done, it in enumerate(dl):
        if not it["positions"]:
            continue
        row = rows[it["i"]]
        gt_full = it["depth"].cuda()
        H, W = gt_full.shape
        win = torch.zeros(H, W, dtype=torch.bool, device="cuda")
        for (x0, y0) in it["positions"]:
            win[y0:y0 + 518, x0:x0 + 518] = True
        win &= torch.isfinite(gt_full)
        gt = gt_full[win]
        bins = torch.bucketize(gt, edges)
        N = float(row["f_number"])
        preds = {}
        with torch.no_grad(), autocast_ctx():
            for label, short, m in nets:
                b = {k: it[short][k][None] for k in ("image", "K", "px_scale")}
                b["cam"] = it["cam"][None]
                p = models.predict(m, b).float()
                p = F.interpolate(p[:, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                preds[label] = p[win]
        rec = {"view": row["color_path"], "F": N, "fl_mm": row["fl_mm"], "group": row[a.fold], "pixels": int(gt.numel())}
        for label in labels:
            s_or = acc.add(label, N, preds[label], gt, bins)
            rec[f"{label} | best scale"] = s_or
            rec[f"{label} | median pred/gt"] = float(torch.median(preds[label] / gt))
        for d in defocus:
            pd_ = preds[d]
            near_gt = gt < a.near
            if near_gt.sum() > 1000:
                rec[f"{d} | median pred/gt near"] = float(torch.median(pd_[near_gt] / gt[near_gt]))
            sel = (pd_ < a.near) & torch.isfinite(pd_)
            if sel.sum() < 1000:
                sel = torch.isfinite(pd_)
            for bl in baselines:
                alpha = torch.median(pd_[sel] / preds[bl][sel])
                fused = f"{bl} x scale({d.replace('Camind ', '')})"
                acc.add(fused, N, alpha * preds[bl], gt, bins)
                rec[f"{fused} | alpha"] = float(alpha)
        views.append(rec)
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(rows)} views", flush=True)

    t = acc.table()
    t.to_csv(out / "bins.csv", index=False)
    v = pd.DataFrame(views)
    v.to_csv(out / "views.csv", index=False)

    lines = []
    allf = t[t.F == "all"].pivot_table(index="model", columns="bin_index", values="absrel")
    allf.columns = [f"{DEPTH_BINS[k]:g}-{DEPTH_BINS[k + 1]:g} m" for k in allf.columns]
    order = labels + [m for m in allf.index if m not in labels]
    allf = allf.reindex([m for m in order if m in allf.index])
    pix = t[(t.F == "all") & (t.model == labels[0])].sort_values("bin_index").pixels.values
    lines += ["== AbsRel per true-depth bin, all F-numbers", allf.round(3).to_string(),
              "pixels per bin: " + ", ".join(f"{c}: {int(n)}" for c, n in zip(allf.columns, pix)), ""]
    best = t[t.F == "all"].pivot_table(index="model", columns="bin_index", values="absrel_best_scale")
    best.columns = allf.columns
    lines += ["== AbsRel with the best per-view scale (shape error only)",
              best.reindex([m for m in labels if m in best.index]).round(3).to_string(), ""]
    for N in sorted(t[t.F != "all"].F.unique(), key=float):
        f = t[t.F == N].pivot_table(index="model", columns="bin_index", values="absrel")
        f.columns = [f"{DEPTH_BINS[k]:g}-{DEPTH_BINS[k + 1]:g} m" for k in f.columns]
        keep = [m for m in labels if m in f.index and (m.startswith("Camind") or "fine-tuned" in m)]
        lines += [f"== AbsRel per depth bin at F{N:g}", f.reindex(keep).round(3).to_string(), ""]
    for d in defocus:
        for col in (f"{d} | median pred/gt", f"{d} | median pred/gt near"):
            if col in v:
                e = v.groupby("F")[col].apply(lambda x: np.median(np.abs(np.log(x.dropna())))) if v[col].notna().any() else None
                if e is not None:
                    lines += [f"== {col}: median |log scale error| per F-number (0.1 = ~10% off)", e.round(3).to_string(), ""]
    over = t[t.F == "all"].groupby("model").apply(lambda g: (g.absrel * g.pixels).sum() / g.pixels.sum())
    lines += ["== AbsRel over all pixels (incl. fusion previews)", over.reindex([m for m in order if m in over.index]).round(3).to_string()]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    cols = list(allf.columns)
    x = np.arange(len(cols))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for m in labels:
        ls = "-" if m.startswith("Camind") else ("--" if "fine-tuned" in m else ":")
        ax[0].plot(x, allf.loc[m].values, ls, marker="o", label=m)
    ax[0].set(xticks=x, xticklabels=cols, xlabel="true depth", ylabel="AbsRel", yscale="log",
              title=f"{a.fold}: error vs distance (all F-numbers)")
    ax[0].legend(fontsize=7)
    for m in [m for m in allf.index if " x scale(" in m] + [b for b in baselines if "zero-shot" in b]:
        ax[1].plot(x, allf.loc[m].values, "-" if " x scale(" in m else ":", marker="o", label=m)
    ax[1].set(xticks=x, xticklabels=cols, xlabel="true depth", ylabel="AbsRel", yscale="log",
              title="fusion preview: baselines rescaled by the defocus model")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "error_vs_distance.png", dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
