"""
def_calibration/focus_from_aperture_pairs.py -- focus distance (D_focus) per (scene, focal
length, camera) from pairs of images of the SAME view at a wide and a narrow aperture.

Why: a single image's sharpness measure (e.g. Laplacian variance) mixes blur with how much
texture a patch has, which makes focus fits unreliable when the blur signal is weak (short
focal lengths, deep depth of field). Every MODEST view is shot at several F-numbers, so the
narrow-aperture image (F22, nearly all in focus) is a reference with the same texture. For
each patch we find the Gaussian blur sigma that turns the reference into the wide-aperture
image (F2.8). Texture cancels; sigma measures the extra defocus blur directly.

Model (thin lens, all in half-resolution pixels; x = 1/d - 1/s in 1/m):
    blur std of target    sigma_t = k * |x|,            k = fx_half * (f / N_t) / 4
    blur std of reference sigma_r = k * (N_t / N_r) * |x|  (+ a small constant, e.g. diffraction)
    measured extra blur   dsigma^2 = max(0, alpha * x^2 - c0),  alpha = k^2 * (1 - (N_t/N_r)^2)
s (the focus distance), alpha and c0 >= 0 are found by a robust grid search on binned medians.
alpha / alpha_pred (from fx, f_true_mm and the F-numbers in the manifest) is reported as a
self-check: it should be similar across settings (the factor 1/4 for disc -> Gaussian blur
is approximate, so its absolute value is not exactly 1).

Steps per view and camera (cached in OUT_DIR/pairs/):
  1. load target (F2.8) and reference (F22, else F16) at half resolution, plus target depth;
  2. align the reference to the target (ECC affine at 1/4 resolution);
  3. per 32x32 patch: best sigma on a 0-12 px grid (best per-patch gain, means removed),
     refined by a parabola; keep textured, well-matched, depth-flat patches.

Outputs in OUT_DIR:
    pairs/<view_id>_<side>.npy          (1/d, sigma) samples per view
    dfocus_aperture_pairs.csv           one row per (scene, fl, side)
    plots_pairs/<scene>_<fl>_<side>.png fit plots

Run (opencv_env; needs only numpy, OpenCV, tifffile, matplotlib):
    python def_calibration/focus_from_aperture_pairs.py                     # all settings
    python def_calibration/focus_from_aperture_pairs.py --settings 3:40 5:28 # scene:fl subset
"""
import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEPTH_ROOT = Path(r"D:\datasets\MODEST_depth")
MANIFEST = DEPTH_ROOT / "manifest.csv"
OUT_DIR = Path(r"D:\datasets\MODEST_focus")

P = 32                                   # patch size at half resolution (64 px full resolution)
SIGMAS = np.arange(0.0, 12.01, 0.25)     # blur grid, half-resolution pixels
MIN_VALID_FRAC, MAX_DEPTH_CV = 0.9, 0.05
MIN_HP_STD = 1.5                         # high-pass std of the reference patch (gray levels)
MIN_MATCH_R2 = 0.9                       # share of target patch variance explained by best match
TARGET_F, REF_F = 2.8, (22.0, 16.0)
N_BINS, MIN_BIN = 25, 8


def blocks_sum(a, nh, nw):
    return a[:nh * P, :nw * P].reshape(nh, P, nw, P).sum(axis=(1, 3))


def view_samples(job):
    """job = (out_path, target_color, ref_color, target_depth) -> saves (1/d, sigma) samples."""
    out_path, tc, rc, td = job
    if Path(out_path).exists():
        return out_path
    T = cv2.imread(str(tc), cv2.IMREAD_REDUCED_GRAYSCALE_2).astype(np.float32)
    R = cv2.imread(str(rc), cv2.IMREAD_REDUCED_GRAYSCALE_2).astype(np.float32)
    D = tifffile.imread(str(td)).astype(np.float32)[::2, ::2][:T.shape[0], :T.shape[1]]

    # align reference to target at 1/4 resolution (scale the translation back up)
    t4, r4 = cv2.resize(T, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.resize(R, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(cv2.GaussianBlur(t4, (0, 0), 2), cv2.GaussianBlur(r4, (0, 0), 2), W,
                                    cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        np.save(out_path, np.zeros((0, 2), np.float32))
        return out_path
    W[:, 2] *= 2
    R = cv2.warpAffine(R, W, (T.shape[1], T.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

    nh, nw, n = T.shape[0] // P, T.shape[1] // P, P * P
    sT, sTT = blocks_sum(T, nh, nw), blocks_sum(T * T, nh, nw)
    var_t = sTT - sT ** 2 / n
    resid = np.empty((len(SIGMAS), nh, nw), np.float32)
    for i, s in enumerate(SIGMAS):
        B = R if s == 0 else cv2.GaussianBlur(R, (0, 0), float(s))
        sB, sBB, sTB = blocks_sum(B, nh, nw), blocks_sum(B * B, nh, nw), blocks_sum(T * B, nh, nw)
        cov, var_b = sTB - sT * sB / n, sBB - sB ** 2 / n
        resid[i] = var_t - np.where(cov > 0, cov ** 2 / np.maximum(var_b, 1e-6), 0)
    best = resid.argmin(axis=0)
    # parabolic refinement between grid neighbours
    i0, i1, i2 = np.clip(best - 1, 0, len(SIGMAS) - 1), best, np.clip(best + 1, 0, len(SIGMAS) - 1)
    r0, r1, r2 = (np.take_along_axis(resid, i[None], 0)[0] for i in (i0, i1, i2))
    den = r0 - 2 * r1 + r2
    off = np.where((den > 0) & (i0 != i1) & (i2 != i1), 0.5 * (r0 - r2) / np.where(den > 0, den, 1), 0)
    sigma = SIGMAS[best] + np.clip(off, -0.5, 0.5) * (SIGMAS[1] - SIGMAS[0])
    r2_match = 1 - r1 / np.maximum(var_t, 1e-6)

    hp = R - cv2.GaussianBlur(R, (0, 0), 1.5)
    hp_std = np.sqrt(np.maximum(blocks_sum(hp * hp, nh, nw) / n - (blocks_sum(hp, nh, nw) / n) ** 2, 0))
    Db = D[:nh * P, :nw * P].reshape(nh, P, nw, P).swapaxes(1, 2).reshape(nh, nw, n)
    valid = np.isfinite(Db) & (Db > 0)
    Dz = np.where(valid, Db, np.nan)
    with np.errstate(all="ignore"):
        dmed, dmean, dstd = np.nanmedian(Dz, axis=2), np.nanmean(Dz, axis=2), np.nanstd(Dz, axis=2)
    keep = ((valid.mean(axis=2) >= MIN_VALID_FRAC) & (dstd / dmean <= MAX_DEPTH_CV) & (hp_std >= MIN_HP_STD)
            & (r2_match >= MIN_MATCH_R2) & (best < len(SIGMAS) - 1))
    np.save(out_path, np.column_stack([1.0 / dmed[keep], sigma[keep]]).astype(np.float32))
    return out_path


def fit_setting(samples, alpha_pred, c0_fixed=None):
    """Robust grid fit of dsigma^2 = max(0, alpha*(x - 1/s)^2 - c0) on binned medians (x = 1/d).

    c0 (the reference image's own blur, mostly diffraction) trades off against s when the data
    only shows one side of the V (focus beyond most of the scene). It depends on the F-numbers
    and pixel size, not on the scene, so it can be fixed to a common value with c0_fixed."""
    x, sg = samples[:, 0], samples[:, 1]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    bx, by, bw = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= MIN_BIN:
            bx.append(np.median(x[m])); by.append(np.median(sg[m]) ** 2); bw.append(np.sqrt(m.sum()))
    bx, by, bw = map(np.array, (bx, by, bw))
    if len(bx) < 6:
        return {"status": "too_few_bins"}
    inv_s = np.arange(0.0, 2.5001, 0.0025)                 # 1/s in 1/m (s from 0.4 m to infinity)
    alphas = alpha_pred * np.logspace(np.log10(0.1), np.log10(10), 61)
    c0s = np.array([c0_fixed]) if c0_fixed is not None else np.linspace(0, max(4.0, float(by.min()) * 2 + 1), 21)
    xx = (bx[None, :] - inv_s[:, None]) ** 2                # (S, B)
    cost = np.empty((len(alphas), len(c0s), len(inv_s)))
    for i, al in enumerate(alphas):                         # loop keeps memory small
        pred = np.maximum(0, al * xx[None] - c0s[:, None, None])           # (C, S, B)
        cost[i] = (bw * np.abs(by - pred)).sum(axis=-1) / bw.sum()
    a_i, c_i, s_i = np.unravel_index(cost.argmin(), cost.shape)
    best = cost[a_i, c_i, s_i]
    prof = cost.min(axis=(0, 1))                             # best cost for each 1/s
    ok = inv_s[prof <= best * 1.1 + 1e-9]
    s_hat = 1 / inv_s[s_i] if inv_s[s_i] > 0 else np.inf
    return {"status": "ok", "d_focus": round(float(s_hat), 3),
            "d_focus_lo": round(float(1 / ok.max()), 3) if ok.max() > 0 else np.inf,
            "d_focus_hi": round(float(1 / ok.min()), 3) if ok.min() > 0 else np.inf,
            "alpha_ratio": round(float(alphas[a_i] / alpha_pred), 3), "c0": round(float(c0s[c_i]), 2),
            "cost": round(float(best), 3), "n_bins": len(bx),
            "depth_p5": round(float(1 / np.quantile(x, 0.95)), 2), "depth_p95": round(float(1 / np.quantile(x, 0.05)), 2),
            "_bins": (bx, by), "_params": (alphas[a_i], c0s[c_i], inv_s[s_i])}


def plot(res, samples, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bx, by = res["_bins"]
    a, c0, i_s = res["_params"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.08, color="gray", label="patches")
    ax.scatter(bx, np.sqrt(by), color="C0", s=14, zorder=3, label="bin medians")
    xs = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 300)
    ax.plot(xs, np.sqrt(np.maximum(0, a * (xs - i_s) ** 2 - c0)), color="C3", label=f"fit, D_focus = {res['d_focus']} m")
    ax.axvline(i_s, color="C3", ls="--", lw=1)
    ax.set_xlabel("1 / depth (1/m)"); ax.set_ylabel("extra blur sigma (half-res px)"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", help="scene:fl pairs, e.g. 3:40 5:28 (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--c0", type=float, default=None,
                    help="fix the reference blur constant (half-res px^2); free fits give a median of ~0.5")
    a = ap.parse_args()
    only = {tuple(int(v) for v in s.split(":")) for s in a.settings} if a.settings else None
    (OUT_DIR / "pairs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots_pairs").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    views = defaultdict(dict)  # (scene, fl, side, view) -> {F: row}
    for r in rows:
        if only and (int(r["scene"]), int(r["fl_mm"])) not in only:
            continue
        views[(int(r["scene"]), int(r["fl_mm"]), r["side"], r["view_id"])][float(r["f_number"])] = r
    jobs, setting_views = [], defaultdict(list)
    for (s, fl, side, vid), byf in views.items():
        ref = next((byf[f] for f in REF_F if f in byf), None)
        if TARGET_F not in byf or ref is None:
            continue
        t = byf[TARGET_F]
        out = OUT_DIR / "pairs" / f"{vid}_{side}.npy"
        jobs.append((str(out), DEPTH_ROOT / t["color_path"], DEPTH_ROOT / ref["color_path"], DEPTH_ROOT / t["depth_path"]))
        setting_views[(s, fl, side)].append((out, t, float(ref["f_number"])))
    todo = [j for j in jobs if not Path(j[0]).exists()]
    print(f"{len(jobs)} view pairs, {len(todo)} to measure", flush=True)
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(view_samples, todo, chunksize=2)):
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)

    out_csv = OUT_DIR / "dfocus_aperture_pairs.csv"
    cols = ["scene", "fl_mm", "side", "n_views", "n_patches", "status", "d_focus", "d_focus_lo", "d_focus_hi",
            "alpha_ratio", "c0", "cost", "n_bins", "depth_p5", "depth_p95"]
    old = {}
    if only and out_csv.exists():  # keep rows of settings not processed in this run
        old = {(int(r["scene"]), int(r["fl_mm"]), r["side"]): r for r in csv.DictReader(open(out_csv))}
    results = dict(old)
    for (s, fl, side), vl in sorted(setting_views.items()):
        samples = np.concatenate([np.load(p) for p, _, _ in vl])
        t = vl[0][1]
        n_ref = np.median([nr for _, _, nr in vl])
        k = float(t["fx"]) / 2 * (float(t["f_true_mm"]) / 1000 / TARGET_F) / 4
        alpha_pred = k ** 2 * (1 - (TARGET_F / n_ref) ** 2)
        res = fit_setting(samples, alpha_pred, a.c0) if len(samples) >= 100 else {"status": "too_few_patches"}
        if res["status"] == "ok":
            plot(res, samples, OUT_DIR / "plots_pairs" / f"s{s}_fl{fl}_{side}.png", f"Scene{s} {fl}mm {side}")
        row = {"scene": s, "fl_mm": fl, "side": side, "n_views": len(vl), "n_patches": len(samples),
               **{c: v for c, v in res.items() if not c.startswith("_")}}
        results[(s, fl, side)] = row
        print(f"Scene{s} {fl}mm {side}: {res['status']} D_focus {res.get('d_focus')} "
              f"[{res.get('d_focus_lo')}, {res.get('d_focus_hi')}] alpha_ratio {res.get('alpha_ratio')} ({len(samples)} patches)", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in sorted(results):
            w.writerow(results[k])
    print(f"Results: {out_csv}")


if __name__ == "__main__":
    main()
