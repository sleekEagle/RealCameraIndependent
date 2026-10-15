"""
Check how well a camera-corrected Camind (camind_corr / camind_corr0) has learned the blur, from a
checkpoint (last.pt is written at every validation, so this works while training runs).

On fixed crops of the fold's val views it compares
  - the stage-2 input (the model's |s - d| / d / x_scale) with the true value from depth and focus distance;
  - the stage-1 output (read as the blur sigma, px) with the blur the blur model gives for the true depth:
    sqrt(sigma0^2 + (C |s - d| / d)^2), C = k f^2 / (N (s - f))   (camind_corr0: sigma0 = 0).
Reported overall, per F-number and per true-blur bin: Pearson and Spearman correlation, mean absolute
error, median ratio predicted / true. Also writes example maps (image, true and predicted |s - d| / d,
true and predicted blur, depth).

    python -m depth_bench.check_blur --model camind_corr --fold S1                 # uses last.pt
    python -m depth_bench.check_blur --model camind_corr --fold S1 --ckpt <path>/best.pt --images 120
    python -m depth_bench.check_blur --model camind_corr --fold S1 --tag overfit --split train --train_images 20
        (overfit test: the same training images as finetune --train_images 20)
Outputs in <RUNS_DIR>/blur_check/<run name>_step<step>/: summary.csv, examples.png
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data, models
from .config import RUNS_DIR
from .train_utils import load_trainable

SIGMA_BINS = [0, 0.5, 1, 2, 4, 8, 16, np.inf]


def stats(p, t):
    if len(t) < 50:
        return {}
    rank = lambda x: np.argsort(np.argsort(x))
    return {"pixels": len(t), "pearson": float(np.corrcoef(p, t)[0, 1]),
            "spearman": float(np.corrcoef(rank(p), rank(t))[0, 1]),
            "mae": float(np.mean(np.abs(p - t))), "median_ratio": float(np.median(p / np.maximum(t, 1e-3))),
            "true_median": float(np.median(t)), "pred_median": float(np.median(p))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["camind_corr", "camind_corr0"])
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", default="", help="as used for training (run name suffix)")
    ap.add_argument("--ckpt", help="default: <RUNS_DIR>/finetune/<model>_<fold>[_<tag>]/last.pt")
    ap.add_argument("--images", type=int, default=60, help="images (2 fixed crops each)")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--train_images", type=int, default=0, help="with --split train: as for finetune")
    ap.add_argument("--stride", type=int, default=4, help="pixel subsampling for the statistics")
    models.add_model_args(ap)
    a = ap.parse_args(argv)

    run = f"{a.model}_{a.fold}" + (f"_{a.tag}" if a.tag else "")
    ckpt = a.ckpt or str(RUNS_DIR / "finetune" / run / "last.pt")
    model = models.build(a.model, **models.model_kwargs(a.model, a)).cuda().eval()
    models.set_trainable(model, 0)
    step = load_trainable(model, ckpt).get("step", "?")
    rows = data.select(data.read_manifest(), a.fold, (a.split,))
    if a.split == "train":
        rows = data.subset(rows, a.train_images)
    rows = random.Random(0).sample(rows, min(a.images, len(rows)))
    ds = data.EvalCrops(rows, 518, 2)
    out = RUNS_DIR / "blur_check" / f"{run}_step{step}"
    out.mkdir(parents=True, exist_ok=True)

    recs, examples = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = data.collate([ds[i]])
            if b is None:
                continue
            bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            pred = models.predict(model, bb)
            gt = b["depth"][0].numpy()
            v_, N, s = (float(x) for x in b["cam"][0])
            C = float(model.blur_scale(b["cam"])[0])
            x_true = np.abs(s - gt) / gt / model.x_scale
            sig_true = np.sqrt(model.sigma0 ** 2 + (C * np.abs(s - gt) / gt) ** 2)
            x_pred = model.last_mul[0].float().cpu().numpy()
            sig_pred = np.abs(model.last_blur[0].float().cpu().numpy())
            ok = np.isfinite(gt)
            sub = np.zeros_like(ok)
            sub[::a.stride, ::a.stride] = True
            ok &= sub
            recs.append(pd.DataFrame({"F": N, "x_true": x_true[ok], "x_pred": x_pred[ok],
                                      "sig_true": sig_true[ok], "sig_pred": sig_pred[ok]}))
            if len(examples) < 6 and i % max(1, len(ds) // 6) == 0:
                examples.append((b["image"][0].permute(1, 2, 0).numpy(), x_true, x_pred, sig_true, sig_pred, gt,
                                 pred[0].float().cpu().numpy(), N, C))
    df = pd.concat(recs)
    df["sig_bin"] = pd.cut(df.sig_true, SIGMA_BINS, right=False)

    res = []
    for name, g in [("all", df)] + [(f"F{N:g}", g) for N, g in df.groupby("F")] + \
            [(f"true sigma {iv.left:g}-{iv.right:g} px", g) for iv, g in df.groupby("sig_bin", observed=True)]:
        res.append({"group": name, "quantity": "|s-d|/d (stage-2 input)", **stats(g.x_pred.values, g.x_true.values)})
        res.append({"group": name, "quantity": "blur sigma (stage-1 output, px)", **stats(g.sig_pred.values, g.sig_true.values)})
    res = pd.DataFrame(res)
    res.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{run}, checkpoint step {step}, {len(rows)} {a.split} images")
    print(res.round(3).to_string(index=False))

    fig, axs = plt.subplots(len(examples), 6, figsize=(18, 3 * len(examples)), squeeze=False)
    for r, (im, xt, xp, st, sp, gt, dp, N, C) in enumerate(examples):
        vmax_x, vmax_s = np.nanpercentile(xt, 99), np.nanpercentile(st, 99)
        for c, (img, title, kw) in enumerate([
                (im, f"F{N:g}, C={C:.2f} px", {}),
                (xt, "true |s-d|/d / x_scale", {"vmin": 0, "vmax": vmax_x}), (xp, "predicted", {"vmin": 0, "vmax": vmax_x}),
                (st, "true blur sigma (px)", {"vmin": 0, "vmax": vmax_s}), (sp, "stage-1 output", {"vmin": 0, "vmax": vmax_s}),
                (np.hstack([gt, dp]), "depth: true | predicted", {})]):
            axs[r, c].imshow(img, **kw)
            axs[r, c].set_title(title, fontsize=8)
            axs[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=80)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
