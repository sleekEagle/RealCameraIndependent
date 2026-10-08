"""
Evaluate a model (zero-shot, or with fine-tuned weights) on fixed native-resolution crops.

Zero-shot needs no fold: it is evaluated once on all test views, and the per-crop CSV keeps
every fold's label, so results can be grouped for any fold afterwards. A fine-tuned model is
evaluated on the test views with the labels of the fold it was trained on.

Examples (depthbench env):
    python -m depth_bench.evaluate --model da3 --out zeroshot_da3                    # zero-shot
    python -m depth_bench.evaluate --model da3 --fold S1 --ckpt <run>/best.pt --out ft_da3_S1
    python -m depth_bench.evaluate --model da2 --fold S1 --max_images 40 --crops 2    # quick subset

Writes <RUNS_DIR>/eval/<out>/per_crop.csv, summary.csv and blur_summary.csv (AbsRel per blur bin).
"""
import argparse
import csv
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, models
from .config import BLUR_BINS, FOCUS_CSV, FOLDS, RUNS_DIR, TEST_LABELS
from .metrics import blur_binned_absrel, boundary_f1_si, depth_metrics
from .train_utils import autocast_ctx, load_trainable

METRICS = ("absrel", "rmse", "delta1", "delta2", "delta3", "silog", "si_absrel", "scale", "bf1", "bprec", "brec")
NB = len(BLUR_BINS) - 1
BLUR_COLS = [c for k in range(NB) for c in (f"absrel_b{k}", f"npx_b{k}")]
SUMMARY = (*METRICS, "absrel_med")  # median AbsRel: robust to a few outlier crops


def stratified_subset(rows, fold, n, seed=0):
    """Up to n images, spread evenly over the fold's test groups (for quick runs)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r[fold]].append(r)
    out = []
    for g in TEST_LABELS:
        rs = groups.get(g, [])
        out += rng.sample(rs, min(len(rs), max(1, n // len(TEST_LABELS))))
    return out


def focus_table(path=FOCUS_CSV):
    """(scene, fl_mm, side) -> focus distance in metres; empty if the file is missing."""
    if not path.exists():
        print(f"no focus file {path}: blur bins skipped")
        return {}
    return {(int(r["scene"]), int(r["fl_mm"]), r["side"]): float(r["d_focus"])
            for r in csv.DictReader(open(path)) if r["status"] == "ok"}


def blur_map(depth, row, focus):
    """Blur-circle diameter in rectified pixels from the true depth (thin lens):
    fx * (f / N) * |1/d - 1/s|, with f the real lens focal length, N the F-number, s the focus distance."""
    s = focus.get((int(row["scene"]), int(row["fl_mm"]), row["side"]))
    if s is None:
        return torch.full_like(depth, float("nan"))
    aperture_m = float(row["f_true_mm"]) / 1000 / float(row["f_number"])
    return float(row["fx"]) * aperture_m * torch.abs(1 / depth - 1 / s)


def blur_summary(per_crop, group_cols):
    """Pixel-weighted AbsRel per blur bin and group."""
    acc = defaultdict(lambda: [0.0, 0, 0])  # (group, bin) -> [sum of AbsRel, pixels, crops]
    for r in per_crop:
        for k in range(NB):
            n = r.get(f"npx_b{k}", 0)
            if n:
                a = acc[(tuple(r[c] for c in group_cols), k)]
                a[0] += r[f"absrel_b{k}"] * n
                a[1] += n
                a[2] += 1
    out = []
    for (g, k), (sm, n, nc) in sorted(acc.items()):
        out.append({**dict(zip(group_cols, g)), "blur_px": f"{BLUR_BINS[k]:g}-{BLUR_BINS[k + 1]:g}",
                    "n_crops": nc, "n_pixels": n, "absrel": round(sm / n, 4)})
    return out


def summarize(per_crop, group_cols):
    groups = defaultdict(list)
    for r in per_crop:
        if "absrel" in r:
            groups[tuple(r[c] for c in group_cols)].append(r)
    out = []
    for k, rs in sorted(groups.items()):
        row = dict(zip(group_cols, k), n_crops=len(rs))
        for m in METRICS:
            v = [r[m] for r in rs if r.get(m) is not None and np.isfinite(r[m])]  # boundary F1 can be NaN
            row[m] = round(float(np.mean(v)), 4) if v else ""
        row["absrel_med"] = round(float(np.median([r["absrel"] for r in rs])), 4)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--fold", choices=FOLDS, help="evaluate this fold's test groups (default: all test views)")
    ap.add_argument("--ckpt", help="fine-tuned weights (best.pt / last.pt from finetune)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--crops", type=int, default=6, help="fixed crops per image")
    ap.add_argument("--max_images", type=int, help="stratified subset (needs --fold) for quick runs")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    rows = data.read_manifest()
    if a.fold:
        rows = [r for r in rows if r[a.fold] in TEST_LABELS]
    else:
        rows = [r for r in rows if r["view_role"] == "test"]
    if a.max_images:
        rows = stratified_subset(rows, a.fold or "S1", a.max_images)
    out_dir = RUNS_DIR / "eval" / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model = models.build(a.model).cuda().eval()
    if a.ckpt:
        models.set_trainable(model, 0)  # only to make all params frozen; the checkpoint restores weights
        load_trainable(model, a.ckpt)
    focus = focus_table()
    ds = data.EvalCrops(rows, a.crop, a.crops)
    dl = DataLoader(ds, batch_size=a.crops, num_workers=a.workers, collate_fn=data.collate)

    keep = ["color_path", "scene", "fl_mm", "f_number", "side", "view_id", "f_true_mm", "rect_turn_deg", *FOLDS]
    per_crop, t0 = [], time.time()
    for bi, b in enumerate(dl):
        if b is None:
            continue
        with torch.no_grad(), autocast_ctx():
            pred = model(b["image"].cuda(non_blocking=True), b["K"].cuda(non_blocking=True))
        pred = pred.float().cpu()
        blur = torch.stack([blur_map(d, rows[i], focus) for d, i in zip(b["depth"], b["row_index"])])
        bins = blur_binned_absrel(pred, b["depth"], blur, BLUR_BINS)
        for m, bb, c, pr, gt, i, j in zip(depth_metrics(pred, b["depth"]), bins, blur, pred, b["depth"],
                                          b["row_index"], b["crop_index"]):
            r = {**{k: rows[i][k] for k in keep}, "crop_index": j, **m}
            if "absrel" in m:
                r.update(boundary_f1_si(pr.numpy(), gt.numpy()))
            c = c[torch.isfinite(c)]
            r["blur_med_px"] = round(float(c.median()), 3) if c.numel() else ""
            for k, (v, n) in enumerate(bb):
                r[f"absrel_b{k}"], r[f"npx_b{k}"] = (round(v, 5) if v is not None else ""), n
            per_crop.append(r)
        if (bi + 1) % 50 == 0:
            print(f"  {bi + 1}/{len(rows)} images, {time.time() - t0:.0f}s", flush=True)

    cols = keep + ["crop_index", "n_valid", *METRICS, "blur_med_px", *BLUR_COLS]
    with open(out_dir / "per_crop.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_crop)
    group = [a.fold] if a.fold else ["f_number"]
    summary = summarize(per_crop, group) + summarize(per_crop, ["f_number"]) + summarize(per_crop, ["fl_mm"])
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "value", "n_crops", *SUMMARY])
        w.writeheader()
        for s in summary:
            g = [k for k in s if k not in ("n_crops", *SUMMARY)][0]
            w.writerow({"group": g, "value": s[g], **{k: s[k] for k in ("n_crops", *SUMMARY)}})
    blur = blur_summary(per_crop, group) + blur_summary(per_crop, ["f_number"])
    with open(out_dir / "blur_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "value", "blur_px", "n_crops", "n_pixels", "absrel"])
        w.writeheader()
        for s in blur:
            g = [k for k in s if k not in ("blur_px", "n_crops", "n_pixels", "absrel")][0]
            w.writerow({"group": g, "value": s[g], **{k: s[k] for k in ("blur_px", "n_crops", "n_pixels", "absrel")}})
    print(f"{a.model} {a.out}: {len(per_crop)} crops in {time.time() - t0:.0f}s")
    for s in summarize(per_crop, group):
        print("  " + "  ".join(f"{k}={v}" for k, v in s.items()))
    return out_dir


if __name__ == "__main__":
    main()
