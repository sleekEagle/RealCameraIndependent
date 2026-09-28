"""
def_calibration/focaldist_estimation.py -- per-camera-setting focus-distance (D_focus)
calibration from paired color/depth images.

Physical basis: the circle-of-confusion diameter for a point at true depth d, with the
camera focused at D_focus, is c(d) = k_i * |1/d - 1/D_focus|, where k_i = A_i*f*D_focus /
(D_focus - f) depends on the aperture diameter A_i (thin-lens defocus model). Since aperture
diameter and f-number are related by A_i = f/N_i, this simplifies to k_i = K0/N_i, where
K0 = f^2*D_focus/(D_focus-f) is a single constant shared by every f-stop of the same
(focal_length, side) -- i.e. each f-stop's blur scale is tied to the others through its known
f-number N_i (parsed straight from the "F<N>" folder name), not independently free.

Laplacian variance (this script's blur/sharpness metric) responds to blur with a power law in
c whose exponent depends on the scene's own texture spectrum (not derivable in closed form),
and it is a sharpness metric -- peaked at D_focus, not zero there, unlike c(d) itself.
Combining that power law with k_i = K0/N_i above gives, per f-stop i:

    blur_metric_i(d) ~= K * N_i^(-p) * (|1/d - 1/D_focus| + EPS)^p

with K, D_focus, p all free but shared across every f-stop (allowing p < 0 reproduces the
peaked shape), and EPS a small fixed numerical floor (in depth-normalized units) that keeps
the model finite exactly at d = D_focus instead of singular. This is a properly constrained
joint fit across all f-stops of a (focal_length, side) -- rather than fitting each f-stop
independently (throwing away the known relationship between them) or pooling raw samples
into a single shared k regardless of aperture (physically wrong, since k genuinely scales
with 1/N) -- using the f-number labels already present in the folder structure.

For each (focal_length, side): each f-stop's (depth, Laplacian-variance) samples, from small,
depth-flat, sufficiently textured patches across that f-stop's own images, are first reduced
to one robust point per depth bin *within that f-stop* (bin_samples() -- see its docstring
for why a high percentile per bin, not the raw samples or their mean, is used -- this has to
happen per f-stop, before combining, since mixing different f-stops' raw samples in one bin
would let the largest-aperture f-stop's bigger blur values dominate the percentile). Only
then are all f-stops' binned points pooled and fit jointly to the model above.

    python def_calibration/focaldist_estimation.py
    python def_calibration/focaldist_estimation.py --scene_dir ... --out_dir ...

Expects: <scene_dir>/fl_<X>mm/F<Y>/{color,depth}/{L,R}/<idx>.{jpg,tiff}, color/depth pairs
matched by filename stem (0.jpg <-> 0.tiff, etc), and F<Y> parseable as a float f-number.

Output:
    <out_dir>/dfocus_results.txt       -- one row per (focal_length, side) (CSV):
        focal, side, status, n_fstops, n_images, n_samples, n_bins,
        d_focus, d_focus_stderr, d_focus_argmax, k, p, rmse
    <out_dir>/plots/<focal>_<side>.png -- raw + binned data points (colored per f-stop) and
                                           each f-stop's fitted curve

d_focus_stderr is the fit's parameter standard error for D_focus (from the covariance
matrix); rmse is the overall residual error of the fitted curve against the binned points --
both are useful, differently-scoped "fitting error" numbers. d_focus_argmax is a simple,
fit-free sanity check: the depth of maximum blur within the widest-aperture f-stop's own
binned points (see process_setting()'s comment). Large disagreement between d_focus and
d_focus_argmax flags an unreliable fit (e.g. extrapolation past the observed depth range).

Requires: numpy, scipy, Pillow, tifffile, matplotlib.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit

EPS = 1e-3  # numerical floor, in depth-normalized (d / median(d)) units -- see module docstring
MIN_SAMPLES = 20  # below this, a 3-parameter fit isn't meaningful; the setting is skipped
MIN_BINS = 5  # below this many populated depth bins, the fit is skipped as unreliable


def discover_settings(scene_dir: Path):
    """scene_dir -> list of (focal_name, side, [(fstop_name, color_dir, depth_dir), ...]),
    one entry per (focal_length, L/R side), pooling every f-stop found under that focal
    length."""
    settings = []
    for focal_dir in sorted(scene_dir.glob("fl_*mm")):
        if not focal_dir.is_dir():
            continue
        for side in ("L", "R"):
            fstop_dirs = []
            for fstop_dir in sorted(focal_dir.iterdir()):
                color_dir = fstop_dir / "color" / side
                depth_dir = fstop_dir / "depth" / side
                if color_dir.is_dir() and depth_dir.is_dir():
                    fstop_dirs.append((fstop_dir.name, color_dir, depth_dir))
            if fstop_dirs:
                settings.append((focal_dir.name, side, fstop_dirs))
    return settings


def parse_fstop(fstop_name: str) -> float:
    """'F2.8' -> 2.8"""
    return float(fstop_name.lstrip("Ff"))


def paired_files(color_dir: Path, depth_dir: Path):
    """Match color/<stem>.jpg <-> depth/<stem>.tiff by filename stem."""
    depth_by_stem = {p.stem: p for p in depth_dir.glob("*.tiff")}
    return [(c, depth_by_stem[c.stem]) for c in sorted(color_dir.glob("*.jpg"))
            if c.stem in depth_by_stem]


def extract_patch_samples(color_path: Path, depth_path: Path, patch_size: int,
                           min_valid_frac: float, max_depth_cv: float, min_texture_std: float):
    """One color/depth image pair -> list of (depth, laplacian_variance) samples, one per
    valid, depth-flat, sufficiently textured non-overlapping patch."""
    gray = np.asarray(Image.open(color_path).convert("L"), dtype=np.float64)
    depth = tifffile.imread(depth_path).astype(np.float64)
    if depth.ndim == 3:  # occasional (H,W,1) tiffs
        depth = depth[..., 0]

    h, w = gray.shape
    samples = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            gpatch = gray[y:y + patch_size, x:x + patch_size]
            dpatch = depth[y:y + patch_size, x:x + patch_size]

            valid = np.isfinite(dpatch) & (dpatch > 0)

            if valid.mean() < min_valid_frac:
                continue
            dvalid = dpatch[valid]
            if dvalid.mean() == 0 or dvalid.std() / dvalid.mean() > max_depth_cv:
                continue
            if gpatch.std() < min_texture_std:
                continue

            blur_metric = ndimage.laplace(gpatch).var()
            samples.append((float(np.median(dvalid)), float(blur_metric)))
    return samples


def bin_samples(depths: np.ndarray, blur: np.ndarray, n_bins: int, percentile: float,
                 min_bin_samples: int):
    """Reduce noisy per-patch samples to one robust (depth, blur) point per depth bin, using
    a high percentile of blur within each bin rather than every raw sample or the bin mean.

    Laplacian variance depends heavily on how much texture a patch happens to contain, not
    just on blur -- a weakly-textured patch reads as "blurry" even in perfect focus, simply
    because it has little high-frequency content to lose. That's the dominant noise source in
    the raw scatter, and neither more samples nor bigger patches fixes it, since it's a bias
    per patch, not estimation noise. Taking a high percentile within each depth bin instead of
    the mean favors the best-textured patches at that depth, which most faithfully reveal the
    true blur level there, and largely cancels this texture-content bias rather than just
    averaging over it.
    """
    edges = np.linspace(depths.min(), depths.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(depths, edges) - 1, 0, n_bins - 1)

    binned_d, binned_b = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() < min_bin_samples:
            continue
        binned_d.append(float(np.median(depths[mask])))
        binned_b.append(float(np.percentile(blur[mask], percentile)))
    return np.array(binned_d), np.array(binned_b)


def _model(X, k, s, p):
    d, n = X
    return k * n ** (-p) * (np.abs(1.0 / d - 1.0 / s) + EPS) ** p


def fit_dfocus(depths: np.ndarray, blur: np.ndarray, fstops: np.ndarray) -> dict:
    """Binned (depth, laplacian_variance, f-number) points, pooled across every f-stop of one
    (focal_length, side) -> joint fit result dict. See module docstring for why each f-stop's
    scale is tied to the others via its known f-number rather than fit independently."""
    d_scale = np.median(depths)
    d_norm = depths / d_scale
    n_scale = np.median(fstops)
    n_norm = fstops / n_scale
    X = np.vstack([d_norm, n_norm])

    s0 = d_norm[np.argmax(blur)]  # depth (normalized) of the sharpest observed sample
    k0 = np.median(blur)
    p0 = -3.0

    popt, pcov = curve_fit(
        _model, X, blur, p0=[k0, s0, p0],
        bounds=([1e-12, d_norm.min() * 0.5, -15.0], [np.inf, d_norm.max() * 2.0, 15.0]),
        maxfev=20000,
    )
    k, s_fit, p = popt
    stderr_norm = np.sqrt(np.diag(pcov))

    pred = _model(X, *popt)
    rmse = float(np.sqrt(np.mean((pred - blur) ** 2)))

    return {
        "d_focus": float(s_fit * d_scale), "d_focus_stderr": float(stderr_norm[1] * d_scale),
        "k": float(k), "p": float(p), "rmse": rmse,
        "d_scale": float(d_scale), "n_scale": float(n_scale), "popt_norm": popt.tolist(),
    }


def plot_fit(raw_d, raw_b, binned_d, binned_b, binned_n, fit: dict, d_focus_argmax: float,
             out_path: Path, title: str):
    d_dense = np.linspace(raw_d.min(), raw_d.max(), 400)
    unique_n = sorted(set(binned_n.tolist()))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_n), 2)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(raw_d, raw_b, s=3, alpha=0.06, color="gray", label="raw patch samples")

    for n_val, color in zip(unique_n, colors):
        mask = binned_n == n_val
        ax.scatter(binned_d[mask], binned_b[mask], s=10, color=color, zorder=3,
                   label=f"F{n_val:g} (binned)")
        X_dense = np.vstack([d_dense / fit["d_scale"],
                              np.full_like(d_dense, n_val / fit["n_scale"])])
        pred = _model(X_dense, *fit["popt_norm"])
        ax.plot(d_dense, pred, color=color, linewidth=1.5)

    ax.axvline(fit["d_focus"], color="black", linestyle="--", linewidth=1,
               label=f"D_focus (fit) = {fit['d_focus']:.2f}")
    ax.axvline(d_focus_argmax, color="firebrick", linestyle=":", linewidth=1,
               label=f"D_focus (argmax) = {d_focus_argmax:.2f}")
    ax.set_xlabel("depth")
    ax.set_ylabel("Laplacian variance (blur metric)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_depth_masked(depth):
    """depth: a path to a depth tiff, or an already-loaded (H,W) array -> masked array with
    invalid pixels (non-finite or <= 0) masked out."""
    if isinstance(depth, (str, Path)):
        depth = tifffile.imread(depth).astype(np.float64)
        if depth.ndim == 3:  # occasional (H,W,1) tiffs
            depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    return np.ma.masked_where(~valid, depth)


def plot_depth_map(depth, out_path: Path, title: str = "depth map", cmap: str = "viridis"):
    """Saves a colored visualization of a depth map (path or array), invalid pixels in gray."""
    masked = _load_depth_masked(depth)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap=cmap_obj)
    fig.colorbar(im, ax=ax, label="depth")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def show_depth_map(depth, title: str = "depth map", cmap: str = "viridis", block: bool = True):
    """Interactively displays a depth map (path or array) in a popup window, invalid pixels in
    gray. This module sets the non-interactive Agg backend at import (needed for the batch
    pipeline's headless plot saving above), so this switches to an interactive backend first --
    safe here since it's called standalone, before any figure has been created in Agg mode."""
    plt.switch_backend("TkAgg")
    masked = _load_depth_masked(depth)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap=cmap_obj)
    fig.colorbar(im, ax=ax, label="depth")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show(block=block)


def process_setting(focal, side, fstop_dirs, out_dir: Path, patch_size,
                     min_valid_frac, max_depth_cv, min_texture_std, n_depth_bins,
                     blur_percentile, min_bin_samples) -> dict:
    setting_name = f"{focal}_{side}"
    base = {"focal": focal, "side": side, "n_fstops": len(fstop_dirs)}

    n_images = n_raw_samples = 0
    raw_d_all, raw_b_all = [], []
    binned_d_all, binned_b_all, binned_n_all = [], [], []
    fstop_ns = []  # one n_number per included f-stop, parallel to binned_d_all/binned_b_all

    for fstop_name, color_dir, depth_dir in fstop_dirs:
        n_number = parse_fstop(fstop_name)
        pairs = paired_files(color_dir, depth_dir)
        n_images += len(pairs)

        fstop_samples = []
        for i, (color_path, depth_path) in enumerate(pairs):
            s = extract_patch_samples(color_path, depth_path, patch_size, min_valid_frac,
                                       max_depth_cv, min_texture_std)
            if i == 0 and s:
                depths_preview = [d for d, _ in s]
                print(f"    {setting_name} ({fstop_name}): depth range in first image's "
                      f"valid patches = [{min(depths_preview):.1f}, {max(depths_preview):.1f}]")
            fstop_samples.extend(s)
        n_raw_samples += len(fstop_samples)

        if len(fstop_samples) < MIN_SAMPLES:
            print(f"    {setting_name} ({fstop_name}): too few samples "
                  f"({len(fstop_samples)}), excluding this f-stop from the joint fit")
            continue

        d = np.array([s[0] for s in fstop_samples])
        b = np.array([s[1] for s in fstop_samples])
        raw_d_all.append(d)
        raw_b_all.append(b)

        bd, bb = bin_samples(d, b, n_depth_bins, blur_percentile, min_bin_samples)
        if len(bd) == 0:
            continue
        binned_d_all.append(bd)
        binned_b_all.append(bb)
        binned_n_all.append(np.full(len(bd), n_number))
        fstop_ns.append(n_number)

    base["n_images"] = n_images

    if not binned_d_all or sum(len(x) for x in binned_d_all) < MIN_BINS:
        return {**base, "status": "skipped_too_few_bins", "n_samples": n_raw_samples}

    binned_d = np.concatenate(binned_d_all)
    binned_b = np.concatenate(binned_b_all)
    binned_n = np.concatenate(binned_n_all)
    raw_d = np.concatenate(raw_d_all)
    raw_b = np.concatenate(raw_b_all)

    try:
        fit = fit_dfocus(binned_d, binned_b, binned_n)
    except Exception as e:
        return {**base, "status": f"fit_failed: {e}", "n_samples": n_raw_samples}

    # Simple sanity-check estimate: the depth of maximum blur within the widest-aperture
    # (smallest f-number) f-stop's own binned points. The widest aperture gives the
    # strongest, most peaked blur-vs-depth signal, so its own argmax is the most reliable
    # simple check; mixing f-stops here would bias toward whichever has the largest raw
    # blur scale rather than reflecting the true depth of sharpest focus. Large disagreement
    # between this and the curve fit's D_focus is a red flag for the fit (e.g. extrapolation
    # past the observed depth range, or a spurious local optimum).
    widest_idx = int(np.argmin(fstop_ns))
    bd_widest, bb_widest = binned_d_all[widest_idx], binned_b_all[widest_idx]
    d_focus_argmax = float(bd_widest[np.argmax(bb_widest)])

    plot_fit(raw_d, raw_b, binned_d, binned_b, binned_n, fit, d_focus_argmax,
             out_dir / "plots" / f"{setting_name}.png", setting_name)
    return {**base, "status": "ok", "n_samples": n_raw_samples, "n_bins": len(binned_d),
            "d_focus": fit["d_focus"], "d_focus_stderr": fit["d_focus_stderr"],
            "d_focus_argmax": d_focus_argmax,
            "k": fit["k"], "p": fit["p"], "rmse": fit["rmse"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", default=r"D:\datasets\MODEST_processed\scene4")
    parser.add_argument("--out_dir", default=r"D:\datasets\MODEST_processed\scene4_dfocus")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--min_valid_frac", type=float, default=0.9)
    parser.add_argument("--max_depth_cv", type=float, default=0.05)
    parser.add_argument("--min_texture_std", type=float, default=2.0)
    parser.add_argument("--n_depth_bins", type=int, default=40)
    parser.add_argument("--blur_percentile", type=float, default=90.0)
    parser.add_argument("--min_bin_samples", type=int, default=5)
    args = parser.parse_args()

    scene_dir, out_dir = Path(args.scene_dir), Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    settings = discover_settings(scene_dir)
    print(f"Found {len(settings)} camera settings under {scene_dir}")

    fieldnames = ["focal", "side", "status", "n_fstops", "n_images", "n_samples", "n_bins",
                  "d_focus", "d_focus_stderr", "d_focus_argmax", "k", "p", "rmse"]
    results_path = out_dir / "dfocus_results.txt"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, (focal, side, fstop_dirs) in enumerate(settings):
            print(f"[{i + 1}/{len(settings)}] {focal} {side} "
                  f"({len(fstop_dirs)} f-stops)")
            result = process_setting(focal, side, fstop_dirs, out_dir,
                                      args.patch_size, args.min_valid_frac,
                                      args.max_depth_cv, args.min_texture_std,
                                      args.n_depth_bins, args.blur_percentile,
                                      args.min_bin_samples)
            writer.writerow(result)
            f.flush()

    print(f"Done. Results: {results_path}")


if __name__ == "__main__":
    main()
