"""
def_calibration/focus_distance_survey.py -- estimate the focus distance (D_focus) of every
(scene, focal length, camera) in MODEST_depth, and check whether it stays constant.

Uses the patch sampling, blur model, binning and joint fit of focaldist_estimation_imgs.py
unchanged (same defaults):
    blur_metric_i(d) ~= K * N_i^(-p) * (|1/d - 1/D_focus| + EPS)^p
fitted jointly over all F-numbers of a group of images. This script only adds parallel
sampling with a per-image cache, grouping by (scene, focal length, side) and by view, and
summary tables.

Two levels of fit:
  1. per (scene, focal length, side): the main D_focus estimate;
  2. per view (one camera position, shot at all F-numbers) inside each setting: if focus
     was not touched during a session, the per-view values should agree.

Image list (incl. view ids) comes from MODEST_depth/manifest.csv, so excluded images stay
excluded. Per-image samples are cached, so groupings can be re-fitted quickly.

Outputs in OUT_DIR:
    samples/<scene>_<fl>_<F>_<side>_<index>.npy   cached (depth, blur) samples per image
    dfocus_per_setting.csv                        one row per (scene, fl, side)
    dfocus_per_view.csv                           one row per (scene, fl, side, view)
    plots/<scene>_<fl>_<side>.png                 fit plots per setting

Run (opencv_env):
    python def_calibration/focus_distance_survey.py
"""
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focaldist_estimation_imgs import (MIN_BINS, MIN_SAMPLES, bin_samples,  # noqa: E402
                                       extract_patch_samples, fit_dfocus, plot_fit)

DEPTH_ROOT = Path(r"D:\datasets\MODEST_depth")
MANIFEST = DEPTH_ROOT / "manifest.csv"
OUT_DIR = Path(r"D:\datasets\MODEST_focus")
WORKERS = 12

# same defaults as focaldist_estimation_imgs.py
PATCH_SIZE = 32
MIN_VALID_FRAC = 0.9
MAX_DEPTH_CV = 0.05
MIN_TEXTURE_STD = 2.0
N_DEPTH_BINS = 40
BLUR_PERCENTILE = 90.0
MIN_BIN_SAMPLES = 5


def patch_samples(color_path, depth_path):
    """(depth, Laplacian variance) per valid, depth-flat, textured non-overlapping patch."""
    s = extract_patch_samples(Path(color_path), Path(depth_path), PATCH_SIZE, MIN_VALID_FRAC,
                              MAX_DEPTH_CV, MIN_TEXTURE_STD)
    return np.array(s, dtype=np.float32).reshape(-1, 2)


def cache_path(r):
    return OUT_DIR / "samples" / f"s{r['scene']}_fl{r['fl_mm']}_F{float(r['f_number']):g}_{r['side']}_{r['index']}.npy"


def compute(r):
    p = cache_path(r)
    if not p.exists():
        np.save(p, patch_samples(DEPTH_ROOT / r["color_path"], DEPTH_ROOT / r["depth_path"]))
    return str(p)


def fit_group(rows, plot_to=None, title=""):
    """Joint fit over all F-numbers of the given images (same procedure as process_setting)."""
    by_f = defaultdict(list)
    for r in rows:
        by_f[float(r["f_number"])].append(np.load(cache_path(r)))
    raw_d, raw_b, bd_all, bb_all, bn_all, fs = [], [], [], [], [], []
    for f_num, arrs in sorted(by_f.items()):
        s = np.concatenate(arrs)
        if len(s) < MIN_SAMPLES:
            continue
        bd, bb = bin_samples(s[:, 0], s[:, 1], N_DEPTH_BINS, BLUR_PERCENTILE, MIN_BIN_SAMPLES)
        if len(bd) == 0:
            continue
        raw_d.append(s[:, 0]); raw_b.append(s[:, 1])
        bd_all.append(bd); bb_all.append(bb); bn_all.append(np.full(len(bd), f_num)); fs.append(f_num)
    n_samples = int(sum(len(x) for x in raw_d))
    if not bd_all or sum(len(x) for x in bd_all) < MIN_BINS:
        return {"status": "too_few_bins", "n_samples": n_samples}
    bd, bb, bn = np.concatenate(bd_all), np.concatenate(bb_all), np.concatenate(bn_all)
    try:
        fit = fit_dfocus(bd, bb, bn)
    except Exception as e:  # noqa: BLE001 - report and continue
        return {"status": f"fit_failed: {e}", "n_samples": n_samples}
    w = int(np.argmin(fs))
    argmax = float(bd_all[w][np.argmax(bb_all[w])])
    if plot_to is not None:
        plot_fit(np.concatenate(raw_d), np.concatenate(raw_b), bd, bb, bn, fit, argmax, plot_to, title)
    depths = np.concatenate(raw_d)
    return {"status": "ok", "n_samples": n_samples, "n_bins": len(bd), "n_fstops": len(fs),
            "d_focus": round(fit["d_focus"], 3), "d_focus_stderr": round(fit["d_focus_stderr"], 3),
            "d_focus_argmax": round(argmax, 3), "p": round(fit["p"], 2), "rmse": round(fit["rmse"], 3),
            "depth_p5": round(float(np.percentile(depths, 5)), 2),
            "depth_p95": round(float(np.percentile(depths, 95)), 2)}


def main():
    (OUT_DIR / "samples").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST)))
    todo = [r for r in rows if not cache_path(r).exists()]
    print(f"{len(rows)} images, {len(todo)} not yet sampled", flush=True)
    if todo:  # start worker processes only when there is sampling left to do
        with ProcessPoolExecutor(WORKERS) as ex:
            for i, _ in enumerate(ex.map(compute, todo, chunksize=4)):
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)

    settings = defaultdict(list)
    for r in rows:
        settings[(int(r["scene"]), int(r["fl_mm"]), r["side"])].append(r)

    cols = ["scene", "fl_mm", "side", "f_true_mm", "status", "n_fstops", "n_samples", "n_bins",
            "d_focus", "d_focus_stderr", "d_focus_argmax", "p", "rmse", "depth_p5", "depth_p95"]
    with open(OUT_DIR / "dfocus_per_setting.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for (s, fl, side), rs in sorted(settings.items()):
            res = fit_group(rs, OUT_DIR / "plots" / f"s{s}_fl{fl}_{side}.png", f"Scene{s} {fl}mm {side}")
            w.writerow({"scene": s, "fl_mm": fl, "side": side, "f_true_mm": rs[0]["f_true_mm"], **res})
            f.flush()
            print(f"Scene{s} {fl}mm {side}: {res.get('status')} D_focus {res.get('d_focus')} "
                  f"(+-{res.get('d_focus_stderr')}, argmax {res.get('d_focus_argmax')})", flush=True)

    vcols = ["scene", "fl_mm", "side", "view_id", "n_images", "status", "n_fstops", "n_samples",
             "d_focus", "d_focus_stderr", "d_focus_argmax", "depth_p5", "depth_p95"]
    with open(OUT_DIR / "dfocus_per_view.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vcols, extrasaction="ignore")
        w.writeheader()
        for (s, fl, side), rs in sorted(settings.items()):
            views = defaultdict(list)
            for r in rs:
                views[r["view_id"]].append(r)
            for vid, vr in sorted(views.items()):
                if len({r["f_number"] for r in vr}) < 3:  # need several F-numbers for the joint fit
                    continue
                w.writerow({"scene": s, "fl_mm": fl, "side": side, "view_id": vid, "n_images": len(vr), **fit_group(vr)})
    print(f"Done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
