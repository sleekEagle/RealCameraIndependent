"""
def_calibration/focaldist_estimation_calibimgs.py -- focus-distance (D_focus) calibration
from ChArUco calibration-target images, as a cross-check against focaldist_estimation.py's
scene-depth-map-based estimate.

Rather than reading a sensor depth map, per-point distance to the ChArUco board is computed
geometrically: cv2.aruco.CharucoDetector finds the board's corners in each image, cv2.solvePnP
recovers the board's pose using that focal length's calibrated K/D (from its .npz file) and
the known board geometry (pattern_info_charuco.json), and each detected corner's known 3D
position (on the board's flat Z=0 plane) is transformed into camera coordinates to read off
its exact depth. No separate depth sensor is involved, and since the board is often tilted, a
single image can already contribute a range of different depths from its different corners.

See focaldist_estimation.py's module docstring for the underlying defocus-blur theory
(thin-lens equation -> circle of confusion c(d) = k*|1/d - 1/D_focus| -> the free-exponent
power law relating Laplacian variance to c). The one difference here: every calibration image
at a given focal length shares a single, fixed f-stop (see pattern_info_charuco.json's
f_number) -- there is no aperture variation to jointly constrain across f-stops the way
focaldist_estimation.py does, so this fits the plain single-aperture model directly:

    blur_metric(d) ~= k * (|1/d - 1/D_focus| + EPS)^p

pooled over every calibration image at a given (focal_length, side).

    python def_calibration/focaldist_estimation_calibimgs.py

Expects, under --calib_root:
    <camera_dir>/fl_<X>mm/*.JPG                 -- calibration images (EOS_6D_A = L, _B = R)
    <camera_dir>/calibration/fl_<X>mm.npz        -- that focal length's K, D
    pattern_info_charuco.json                    -- board geometry (squares, lengths, dict)

Output (same shape as focaldist_estimation.py's):
    <out_dir>/dfocus_results.txt       -- one row per (focal_length, side) (CSV):
        focal, side, status, n_images, n_detected, n_samples, n_bins,
        d_focus, d_focus_stderr, k, p, rmse
    <out_dir>/plots/<focal>_<side>.png -- raw + binned corner samples and the fitted curve
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit

CALIB_ROOT = (r"D:\datasets\MODEST\Global_calibration_set\MODEST_ChArUco"
              r"\Global_calibration_set\ChArUco_pattern")
CAMERA_DIRS = {"L": "EOS_6D_A", "R": "EOS_6D_B"}

EPS = 1e-3  # numerical floor, in depth-normalized (d / median(d)) units
MIN_SAMPLES = 20  # below this, a 3-parameter fit isn't meaningful; the setting is skipped
MIN_BINS = 5  # below this many populated depth bins, the fit is skipped as unreliable


def load_pattern_info(calib_root: Path) -> dict:
    with open(calib_root / "pattern_info_charuco.json") as f:
        return json.load(f)["charuco"]


def build_board(info: dict):
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, info["dictionary"]))
    return aruco.CharucoBoard((info["squares_x"], info["squares_y"]),
                               info["square_length_m"], info["marker_length_m"], dictionary)


def load_intrinsics(camera_dir: Path, focal_name: str):
    npz = np.load(camera_dir / "calibration" / f"{focal_name}.npz")
    return npz["K"], npz["D"]


def extract_corner_samples(image_path: Path, detector, obj_points_all, K, D,
                            patch_size: int, min_texture_std: float):
    """One calibration image -> list of (depth, laplacian_variance) samples, one per detected
    ChArUco corner: depth comes from the board's PnP pose (exact, geometric), blur from a
    small patch centered on that corner (corners are guaranteed high-contrast, so no
    depth-flatness check is needed the way natural-scene patches require one)."""
    gray = np.asarray(Image.open(image_path).convert("L"))

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return []

    obj_points = obj_points_all[charuco_ids.flatten()]
    img_points = charuco_corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
    if not ok:
        return []

    R, _ = cv2.Rodrigues(rvec)
    depths = (R @ obj_points.T + tvec).T[:, 2]  # Z-depth along the optical axis, per corner

    h, w = gray.shape
    half = patch_size // 2
    samples = []
    for (x, y), d in zip(img_points, depths):
        if d <= 0:
            continue
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        patch = gray[yi - half:yi + half, xi - half:xi + half].astype(np.float64)
        if patch.std() < min_texture_std:
            continue
        blur_metric = ndimage.laplace(patch).var()
        samples.append((float(d), float(blur_metric)))
    return samples


def bin_samples(depths: np.ndarray, blur: np.ndarray, n_bins: int, percentile: float,
                 min_bin_samples: int):
    """Same texture-content-bias fix as focaldist_estimation.py's bin_samples() -- see its
    docstring. Kept here too since a stray weakly-textured patch can still slip through even
    on a checkerboard target (e.g. a corner near frame edge or under uneven exposure)."""
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


def _model(d, k, s, p):
    return k * (np.abs(1.0 / d - 1.0 / s) + EPS) ** p


def fit_dfocus(depths: np.ndarray, blur: np.ndarray) -> dict:
    d_scale = np.median(depths)
    d_norm = depths / d_scale

    s0 = d_norm[np.argmax(blur)]
    k0 = np.median(blur)
    p0 = -3.0

    popt, pcov = curve_fit(
        _model, d_norm, blur, p0=[k0, s0, p0],
        bounds=([1e-12, d_norm.min() * 0.5, -15.0], [np.inf, d_norm.max() * 2.0, 15.0]),
        maxfev=20000,
    )
    k, s_fit, p = popt
    stderr_norm = np.sqrt(np.diag(pcov))
    pred = _model(d_norm, *popt)
    rmse = float(np.sqrt(np.mean((pred - blur) ** 2)))

    return {
        "d_focus": float(s_fit * d_scale), "d_focus_stderr": float(stderr_norm[1] * d_scale),
        "k": float(k), "p": float(p), "rmse": rmse,
        "d_scale": float(d_scale), "popt_norm": popt.tolist(),
    }


def plot_fit(depths, blur, binned_d, binned_b, fit: dict, out_path: Path, title: str):
    d_dense = np.linspace(depths.min(), depths.max(), 400)
    pred = _model(d_dense / fit["d_scale"], *fit["popt_norm"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(depths, blur, s=3, alpha=0.05, color="gray", label="raw corner samples")
    ax.scatter(binned_d, binned_b, s=25, color="tab:blue", zorder=3,
               label="binned (used for fit)")
    ax.plot(d_dense, pred, color="red", linewidth=2, label="fitted curve")
    ax.axvline(fit["d_focus"], color="black", linestyle="--", linewidth=1,
               label=f"D_focus = {fit['d_focus']:.3f} m")
    ax.set_xlabel("distance to ChArUco corner (m)")
    ax.set_ylabel("Laplacian variance (blur metric)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_setting(side, focal_name, camera_dir: Path, board, obj_points_all, out_dir: Path,
                     patch_size, min_texture_std, n_depth_bins, blur_percentile,
                     min_bin_samples) -> dict:
    setting_name = f"{focal_name}_{side}"
    K, D = load_intrinsics(camera_dir, focal_name)
    images = sorted((camera_dir / focal_name).glob("*.JPG"))
    detector = aruco.CharucoDetector(board)

    samples = []
    n_detected = 0
    for img_path in images:
        s = extract_corner_samples(img_path, detector, obj_points_all, K, D,
                                    patch_size, min_texture_std)
        if s:
            n_detected += 1
        samples.extend(s)

    base = {"focal": focal_name, "side": side, "n_images": len(images), "n_detected": n_detected}

    if len(samples) < MIN_SAMPLES:
        return {**base, "status": "skipped_too_few_samples", "n_samples": len(samples)}

    depths = np.array([s[0] for s in samples])
    blur = np.array([s[1] for s in samples])
    binned_d, binned_b = bin_samples(depths, blur, n_depth_bins, blur_percentile,
                                      min_bin_samples)

    if len(binned_d) < MIN_BINS:
        return {**base, "status": "skipped_too_few_bins", "n_samples": len(samples)}

    try:
        fit = fit_dfocus(binned_d, binned_b)
    except Exception as e:
        return {**base, "status": f"fit_failed: {e}", "n_samples": len(samples)}

    plot_fit(depths, blur, binned_d, binned_b, fit,
             out_dir / "plots" / f"{setting_name}.png", setting_name)
    return {**base, "status": "ok", "n_samples": len(samples), "n_bins": len(binned_d),
            "d_focus": fit["d_focus"], "d_focus_stderr": fit["d_focus_stderr"],
            "k": fit["k"], "p": fit["p"], "rmse": fit["rmse"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_root", default=CALIB_ROOT)
    parser.add_argument("--out_dir", default=r"D:\datasets\MODEST_processed\scene4_dfocus_calibimgs")
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--min_texture_std", type=float, default=5.0)
    parser.add_argument("--n_depth_bins", type=int, default=40)
    parser.add_argument("--blur_percentile", type=float, default=90.0)
    parser.add_argument("--min_bin_samples", type=int, default=5)
    args = parser.parse_args()

    calib_root, out_dir = Path(args.calib_root), Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    info = load_pattern_info(calib_root)
    board = build_board(info)
    obj_points_all = board.getChessboardCorners()

    settings = []
    for side, cam_name in CAMERA_DIRS.items():
        camera_dir = calib_root / cam_name
        for focal_dir in sorted(camera_dir.glob("fl_*mm")):
            if focal_dir.is_dir():
                settings.append((side, focal_dir.name, camera_dir))
    print(f"Found {len(settings)} camera settings under {calib_root}")

    fieldnames = ["focal", "side", "status", "n_images", "n_detected", "n_samples", "n_bins",
                  "d_focus", "d_focus_stderr", "k", "p", "rmse"]
    results_path = out_dir / "dfocus_results.txt"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, (side, focal_name, camera_dir) in enumerate(settings):
            print(f"[{i + 1}/{len(settings)}] {focal_name} {side}")
            result = process_setting(side, focal_name, camera_dir, board, obj_points_all,
                                      out_dir, args.patch_size, args.min_texture_std,
                                      args.n_depth_bins, args.blur_percentile,
                                      args.min_bin_samples)
            writer.writerow(result)
            f.flush()

    print(f"Done. Results: {results_path}")


if __name__ == "__main__":
    main()
