"""
def_calibration/defocus_calib.py -- standalone edge-spread-function extraction for ChArUco
calibration images.

extract_esf_samples() measures the edge-spread function (ESF) directly in image pixels along
the boundary between adjacent ChArUco squares, together with the exact geometric depth (from
the board's PnP pose) of the point where each scan line crosses that boundary. Unlike Laplacian
variance, the ESF's width is provably linear in the circle-of-confusion diameter (see the
defocus-blur theory in focaldist_estimation.py's module docstring), so it is a lower-noise blur
proxy for D_focus estimation.
"""
import csv
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm


def _bilinear_sample(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate grayscale intensity at fractional pixel coords (xs, ys). Points
    that fall outside the image come back as NaN so the caller can drop that whole line."""
    h, w = gray.shape
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    out = np.full(xs.shape, np.nan)
    x0 = np.floor(xs[valid]).astype(int)
    y0 = np.floor(ys[valid]).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs[valid] - x0
    fy = ys[valid] - y0
    out[valid] = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x1] * fx * (1 - fy) +
                  gray[y1, x0] * (1 - fx) * fy + gray[y1, x1] * fx * fy)
    return out


def extract_esf_samples(image_path: Path, detector, obj_points_all, K, D, info: dict,
                         lines_per_square: int = 3, margin_frac: float = 0.1,
                         n_obj_samples: int = 300, sample_spacing_px: float = 0.5,
                         min_contrast: float = 15.0):
    """One calibration image -> list of edge-spread-function samples, one per (square, edge
    side, scan-line height). For each square boundary, a horizontal object-space segment
    straddling it is projected into the image through the board's PnP pose (K/D are applied via
    cv2.projectPoints, so the raw distorted image itself is never resampled/undistorted -- that
    would blur it and corrupt the very thing we're trying to measure). The projected polyline is
    then resampled onto a uniform *pixel*-arc-length grid (not a uniform object-space grid,
    since perspective makes the two spacings differ) and bilinearly sampled for intensity, which
    gives the edge-spread function directly in image pixels, centered on the edge crossing.
    Depth is the Z-coordinate (camera coordinates) of the object-space point where the scan
    line crosses the edge.

    detector: cv2.aruco.CharucoDetector for the board.
    obj_points_all: board.getChessboardCorners(), indexed by ChArUco corner id.
    K, D: that focal length's calibrated camera matrix and distortion coefficients.
    info: pattern_info_charuco.json's "charuco" section (squares_x, squares_y,
        square_length_m, ...).

    Each returned dict also carries "img_xy" (Nx2), the actual image pixel coordinates the
    ESF was sampled at, for overlaying on the source image as a sanity check. "scan" is
    "horizontal" for a horizontal scan line crossing one of the square's left/right (vertical)
    edges, or "vertical" for a vertical scan line crossing one of its top/bottom (horizontal)
    edges -- sampling both doubles coverage and lets horizontal- vs vertical-edge blur be
    compared separately.
    """
    gray_u8 = np.asarray(Image.open(image_path).convert("L"))

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_u8)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return []
    obj_points = obj_points_all[charuco_ids.flatten()]
    img_points = charuco_corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
    if not ok:
        return []
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3, 1)

    gray = gray_u8.astype(np.float64)
    sq = info["square_length_m"]
    squares_x, squares_y = info["squares_x"], info["squares_y"]
    margin = margin_frac * sq
    t = np.linspace(-margin, margin, n_obj_samples)
    fracs = (np.arange(lines_per_square) + 1) / (lines_per_square + 1)  # e.g. .25, .5, .75

    def sample_scan_line(obj_pts):
        """obj_pts: Nx3 object-space line straddling an edge at t=0 (the middle sample) ->
        (s, esf, px, py, contrast), or None if it fails the coverage/contrast checks."""
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        img_pts = img_pts.reshape(-1, 2)

        seg_len = np.linalg.norm(np.diff(img_pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        if arc[-1] < 2 * sample_spacing_px:
            return None
        s_edge = np.interp(0.0, t, arc)

        s_uniform = np.arange(0.0, arc[-1], sample_spacing_px)
        px = np.interp(s_uniform, arc, img_pts[:, 0])
        py = np.interp(s_uniform, arc, img_pts[:, 1])

        esf = _bilinear_sample(gray, px, py)
        if np.isnan(esf).any():
            return None

        k = max(3, len(esf) // 10)
        contrast = abs(float(esf[:k].mean()) - float(esf[-k:].mean()))
        if contrast < min_contrast:
            return None
        return s_uniform - s_edge, esf, px, py, contrast

    results = []
    for row in range(squares_y):
        for col in range(squares_x):
            # horizontal scan lines: cross the square's left/right (vertical) edges
            for edge, x_edge in (("left", col * sq), ("right", (col + 1) * sq)):
                if (edge == "left" and col == 0) or (edge == "right" and col == squares_x - 1):
                    continue
                for frac in fracs:
                    y_line = (row + frac) * sq
                    obj_pts = np.stack(
                        [x_edge + t, np.full_like(t, y_line), np.zeros_like(t)], axis=1)
                    sampled = sample_scan_line(obj_pts)
                    if sampled is None:
                        continue
                    s, esf, px, py, contrast = sampled

                    obj_edge = np.array([[x_edge, y_line, 0.0]])
                    depth = float((R @ obj_edge.T + tvec)[2, 0])
                    if depth <= 0:
                        continue

                    results.append({
                        "row": row, "col": col, "edge": edge, "scan": "horizontal",
                        "frac": float(frac), "depth": depth, "s": s, "esf": esf,
                        "contrast": contrast, "img_xy": np.stack([px, py], axis=1),
                    })

            # vertical scan lines: cross the square's top/bottom (horizontal) edges
            for edge, y_edge in (("top", row * sq), ("bottom", (row + 1) * sq)):
                if (edge == "top" and row == 0) or (edge == "bottom" and row == squares_y - 1):
                    continue
                for frac in fracs:
                    x_line = (col + frac) * sq
                    obj_pts = np.stack(
                        [np.full_like(t, x_line), y_edge + t, np.zeros_like(t)], axis=1)
                    sampled = sample_scan_line(obj_pts)
                    if sampled is None:
                        continue
                    s, esf, px, py, contrast = sampled

                    obj_edge = np.array([[x_line, y_edge, 0.0]])
                    depth = float((R @ obj_edge.T + tvec)[2, 0])
                    if depth <= 0:
                        continue

                    results.append({
                        "row": row, "col": col, "edge": edge, "scan": "vertical",
                        "frac": float(frac), "depth": depth, "s": s, "esf": esf,
                        "contrast": contrast, "img_xy": np.stack([px, py], axis=1),
                    })
    return results


def esf_width_10_90(s: np.ndarray, esf: np.ndarray, plateau_frac: float = 0.1):
    """10%-90% rise/fall width of one ESF curve (same units as `s` -- pixels, here). For a
    Gaussian PSF this equals exactly 2*Phi^-1(0.9)*sigma ~= 2.563*sigma (see this module's
    docstring), so it's a low-noise, linear proxy for blur scale. Returns None if the two
    plateaus can't be told apart (e.g. a flat/degenerate curve).

    Thresholds are set relative to the curve's own two plateau levels (averaged over
    `plateau_frac` of each end, same convention as the contrast check in
    extract_esf_samples()), not its raw min/max, so a single noisy sample can't skew them. The
    affine normalization below maps the left end to 0 and the right end to 1 regardless of
    whether the edge is rising or falling, so no direction handling is needed.
    """
    n = len(esf)
    k = max(3, int(n * plateau_frac))
    lo, hi = float(esf[:k].mean()), float(esf[-k:].mean())

    if hi == lo:
        return None

    norm = (esf - lo) / (hi - lo)
    s10 = float(np.interp(0.1, norm, s))
    s90 = float(np.interp(0.9, norm, s))
    return abs(s90 - s10)


def process_directory(image_dir: Path, detector, obj_points_all, K, D, info: dict, **kwargs):
    """Run extract_esf_samples() over every *.JPG in image_dir and reduce each sample to its
    10-90 width, pooling results across the whole directory (kwargs are forwarded to
    extract_esf_samples(), e.g. margin_frac). Returns a list of lightweight dicts (no s/esf/
    img_xy arrays, just the numbers needed for a D_focus fit): {"image", "depth", "width",
    "row", "col", "edge", "scan"}."""
    results = []
    image_paths = sorted(image_dir.glob("*.JPG"))
    for image_path in tqdm(image_paths, desc=f"ESF samples ({image_dir.name})", unit="img"):
        for s in extract_esf_samples(image_path, detector, obj_points_all, K, D, info, **kwargs):
            width = esf_width_10_90(s["s"], s["esf"])
            if width is None:
                continue
            results.append({
                "image": image_path.name, "depth": s["depth"], "width": width,
                "row": s["row"], "col": s["col"], "edge": s["edge"], "scan": s["scan"],
            })
    return results


def load_widths_by_focal(debug_dir: Path, scan: str = "both") -> dict:
    """Read every fl_<X>mm_widths.csv in debug_dir (as written by run_defocus_calib()) and pool
    each focal length's depth/width samples into arrays, filtered by scan direction.

    scan: "horizontal", "vertical", or "both" (no filtering, pools both scan directions
        together).

    Returns {focal_name: {"depth": np.ndarray, "width": np.ndarray}}, e.g.
        {"fl_28mm": {"depth": array([...]), "width": array([...])}, "fl_32mm": {...}, ...}
    """
    if scan not in ("horizontal", "vertical", "both"):
        raise ValueError(f"scan must be 'horizontal', 'vertical', or 'both', got {scan!r}")

    results = {}
    for csv_path in sorted(debug_dir.glob("fl_*mm_widths.csv")):
        focal_name = csv_path.stem[:-len("_widths")]  # "fl_28mm_widths" -> "fl_28mm"
        depths, widths = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if scan != "both" and row["scan"] != scan:
                    continue
                depths.append(float(row["depth"]))
                widths.append(float(row["width"]))
        results[focal_name] = {"depth": np.array(depths), "width": np.array(widths)}
    return results


def _load_setup(focal_name: str, camera_dir: Path, calib_root: Path):
    """Load the board geometry/detector and that focal length's intrinsics."""
    with open(calib_root / "pattern_info_charuco.json") as f:
        info = json.load(f)["charuco"]
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, info["dictionary"]))
    board = aruco.CharucoBoard((info["squares_x"], info["squares_y"]),
                                info["square_length_m"], info["marker_length_m"], dictionary)
    obj_points_all = board.getChessboardCorners()
    detector = aruco.CharucoDetector(board)
    npz = np.load(camera_dir / "calibration" / f"{focal_name}.npz")
    return detector, obj_points_all, npz["K"], npz["D"], info


def process_focal_length(image_dir: Path, out_plot: Path, n_plot: int = 6):
    """Batch-process every image in a directory (e.g. an fl_XXmm folder): extract ESF samples,
    reduce each to a 10-90 width, and write out a pooled (depth, width) CSV plus a width-vs-depth
    scatter plot. Also saves points-on-image and ESF-curve diagnostic plots for one
    representative image, as a sanity check. image_dir is expected under the usual
    <calib_root>/<camera_dir>/fl_<X>mm/*.JPG layout (see focaldist_estimation_calibimgs.py's
    module docstring), so calib_root and camera_dir are inferred from image_dir's own parent
    directories."""
    focal_name = image_dir.name  # e.g. "fl_28mm"
    camera_dir = image_dir.parent  # e.g. .../EOS_6D_A
    calib_root = camera_dir.parent  # .../ChArUco_pattern
    detector, obj_points_all, K, D, info = _load_setup(focal_name, camera_dir, calib_root)

    out_dir = out_plot or image_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = process_directory(image_dir, detector, obj_points_all, K, D, info)
    print(f"{image_dir}: {len(results)} (depth, width) samples")
    if not results:
        return
    for r in results:
        print(f"{r['depth']:.4f}\t{r['width']:.3f}\t{r['image']}")

    csv_path = out_dir / f"{focal_name}_widths.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "row", "col", "edge", "scan", "depth", "width"])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r["depth"] for r in results], [r["width"] for r in results], s=4, alpha=0.3)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("10-90 width (px)")
    ax.set_title(f"width vs depth: {focal_name}")
    fig.tight_layout()
    width_plot_path = out_dir / f"{focal_name}_width_vs_depth.png"
    fig.savefig(width_plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {width_plot_path}")

    # horizontal- vs vertical-scan bias check: depth and width distributions side by side
    horiz = [r for r in results if r["scan"] == "horizontal"]
    vert = [r for r in results if r["scan"] == "vertical"]
    fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(10, 5))
    ax_d.violinplot([[r["depth"] for r in horiz], [r["depth"] for r in vert]], showmeans=True)
    ax_d.set_xticks([1, 2], labels=["horizontal", "vertical"])
    ax_d.set_ylabel("depth (m)")
    ax_d.set_title("depth by scan direction")
    ax_w.violinplot([[r["width"] for r in horiz], [r["width"] for r in vert]], showmeans=True)
    ax_w.set_xticks([1, 2], labels=["horizontal", "vertical"])
    ax_w.set_ylabel("10-90 width (px)")
    ax_w.set_title("width by scan direction")
    fig.suptitle(f"horizontal vs vertical scan bias: {focal_name}")
    fig.tight_layout()
    bias_plot_path = out_dir / f"{focal_name}_scan_bias.png"
    fig.savefig(bias_plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {bias_plot_path}")

    # points-on-image + ESF-curve diagnostic plots, for one representative image
    image_path = sorted(image_dir.glob("*.JPG"))[0]
    samples = extract_esf_samples(image_path, detector, obj_points_all, K, D, info)
    if not samples:
        return

    gray_u8 = np.asarray(Image.open(image_path).convert("L"))
    points_path = out_dir / f"{image_path.stem}_points.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(gray_u8, cmap="gray", vmin=0, vmax=255)
    scan_colors = {"horizontal": "tab:orange", "vertical": "tab:cyan"}
    edge_xy = np.empty((len(samples), 2))
    labeled_scans = set()
    for i, s in enumerate(samples):
        xy = s["img_xy"]
        label = f"{s['scan']} scan" if s["scan"] not in labeled_scans else None
        labeled_scans.add(s["scan"])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.6, alpha=0.6, color=scan_colors[s["scan"]],
                label=label)
        edge_xy[i] = np.interp(0.0, s["s"], xy[:, 0]), np.interp(0.0, s["s"], xy[:, 1])
    ax.scatter(edge_xy[:, 0], edge_xy[:, 1], s=2, color="red", zorder=3,
               label="edge crossing (depth sample)")
    ax.set_title(f"scan lines considered: {image_path.name} ({len(samples)} samples)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(points_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {points_path}")

    # ESF curves for a handful of samples
    esf_path = out_dir / f"{image_path.stem}_esf.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in samples[:n_plot]:
        width = esf_width_10_90(s["s"], s["esf"])
        ax.plot(s["s"], s["esf"], marker=".", markersize=2, linewidth=0.8,
                label=f"row{s['row']} col{s['col']} {s['edge']} d={s['depth']:.2f}m "
                      f"w10-90={width:.2f}px")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("distance along scan line, centered on edge (px)")
    ax.set_ylabel("intensity")
    ax.set_title(f"ESF samples: {image_path.name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(esf_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {esf_path}")


def extract_depth_v_blur():
    """Single-focal-length smoke test: process_focal_length() on one hardcoded fl_XXmm folder
    (EOS_6D_A/fl_32mm). See run_all_esf_extraction() to process every camera/focal length."""
    process_focal_length(
        Path(r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
             r"\Global_calibration_set\ChArUco_pattern\EOS_6D_A\fl_32mm"),
        Path(r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
             r"\Global_calibration_set\ChArUco_pattern\EOS_6D_A\debug"),
    )


CAMERA_DIRS = {"L": "EOS_6D_A", "R": "EOS_6D_B"}


def run_all_esf_extraction():
    """Run process_focal_length() for every fl_<X>mm folder under both EOS_6D_A (L) and
    EOS_6D_B (R), writing each camera's widths CSVs to its own debug/ subfolder. Skips any
    (camera, focal length) whose widths CSV already exists, so this is safe to re-run after
    only some of the cameras/focal lengths have been processed."""
    calib_root = Path(r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
                       r"\Global_calibration_set\ChArUco_pattern")

    for side, cam_name in CAMERA_DIRS.items():
        camera_dir = calib_root / cam_name
        out_dir = camera_dir / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_dir in sorted(camera_dir.glob("fl_*mm")):
            if not image_dir.is_dir():
                continue
            csv_path = out_dir / f"{image_dir.name}_widths.csv"
            if csv_path.exists():
                print(f"[skip] {side} {image_dir.name}: {csv_path.name} already exists")
                continue
            print(f"=== {side} ({cam_name}) {image_dir.name} ===")
            process_focal_length(image_dir, out_dir)


def bin_samples(depth: np.ndarray, width: np.ndarray, n_bins: int = 40,
                 min_bin_samples: int = 20):
    """Collapse raw (depth, width) samples into n_bins equal-width depth bins, taking the
    median depth/width within each populated bin (bins with fewer than min_bin_samples are
    dropped). Same denoising idea as focaldist_estimation_calibimgs.py's bin_samples() --
    individual ESF-width samples are noisy enough that fitting them directly leaves s and k
    almost unconstrained (near-flat loss across a wide range of s), so the fit needs the
    per-depth median, not the raw scatter."""
    depth = np.asarray(depth, dtype=float)
    width = np.asarray(width, dtype=float)
    edges = np.linspace(depth.min(), depth.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(depth, edges) - 1, 0, n_bins - 1)

    binned_d, binned_w = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() < min_bin_samples:
            continue
        binned_d.append(float(np.median(depth[mask])))
        binned_w.append(float(np.median(width[mask])))
    return np.array(binned_d), np.array(binned_w)


def h(s, d, f, N):
    """Squared circle-of-confusion term from the thin-lens defocus model: k * h(s, d, f, N) +
    gamma**2 = width**2, with s the focus distance, d the object depth, f the focal length, N
    the f-number (all in consistent units -- meters here). Shared by fit_S_gamma() and
    fit_k_gamma_pooled()."""
    return ((s - d) / d * 1 / (s - f) * f**2 / N) ** 2


def fit_S_gamma(depth, width, f, N, n_bins: int = 40, min_bin_samples: int = 20,
                 s_upper_factor: float = 3.0):
    """Fit the focus distance s, blur-scale k, and blur floor gamma in

        k * h(s, d, f, N) + gamma**2 = width**2

    by nonlinear least squares, with f (focal length, meters) and N (f-number) fixed scalars.
    h(s, d, f, N) is the (squared) circle-of-confusion term from the thin-lens defocus model,
    so gamma is the residual blur width at the in-focus depth (s == d), where h vanishes.

    depth/width are first reduced with bin_samples() (median per depth bin) -- fitting the raw,
    per-scan-line samples directly leaves s and k barely constrained by the optimizer, since
    per-sample noise swamps the true width-vs-depth shape (see bin_samples()'s docstring).

    Since f, s, d are all in meters, h(s, d, f, N) comes out on the order of 1e-8 to 1e-10 (a
    real circle-of-confusion diameter, in meters), while width**2 is in px**2 (order 1-30) --
    k is really a (pixels per meter)**2-ish unit-conversion factor, so it has to land around
    1e9-1e10 to matter at all. A k0 of 1.0 is off by that many orders of magnitude, which
    leaves curve_fit stuck where k*h is negligible regardless of s (s then looks
    "unidentifiable" -- the fit is just insensitive to it, not genuinely unconstrained by the
    data). k0 is instead back-solved from the single most-defocused bin, matching k0*h(s0, d) to
    that bin's (width**2 - gamma0**2) so the optimizer starts in the right regime.

    h(s, d, f, N) flattens out to a fixed shape (independent of s) once s is well beyond the
    sampled depth range, so if the real focus distance lies past the farthest sampled depth,
    width keeps decreasing across the whole range with no interior minimum -- any sufficiently
    large s then fits equally well, and s is genuinely not identifiable from this data (not a
    numeric bug). s is capped at s_upper_factor * max(depth) so that case reports a bounded,
    honest "at least this far" number instead of an arbitrary huge one, and is flagged via
    "s_at_upper_bound" so it isn't mistaken for a real point estimate.

    k is fit in log space (log_k = log(k), k = exp(log_k)) rather than directly: s and gamma
    live on an O(1-10) scale while k lives on an O(1e9) scale, and that ~9-order-of-magnitude
    mismatch is itself a bad-conditioning problem for curve_fit's gradient-based search,
    independent of (and in addition to) the k0 initial-guess fix above. Fitting log_k puts all
    three parameters within a few orders of magnitude of each other, which is a more general fix
    than picking "nicer" physical units for f/s/d (that only rescales k, it doesn't change how
    far its scale sits from s/gamma's). k_stderr is recovered from log_k's stderr via the delta
    method (k_stderr ~= k * log_k_stderr), which is only a first-order approximation -- accurate
    when log_k_stderr is small, increasingly so otherwise.

    Returns {"s", "k", "gamma", "s_stderr", "k_stderr", "gamma_stderr", "rmse", "n_bins",
    "s_at_upper_bound"}.
    """
    binned_d, binned_w = bin_samples(depth, width, n_bins=n_bins,
                                      min_bin_samples=min_bin_samples)

    def model(d, s, log_k, gamma):
        return np.sqrt(np.exp(log_k) * h(s, d, f, N) + gamma**2)

    s0 = float(binned_d[np.argmin(binned_w)])
    gamma0 = float(binned_w.min())

    h0 = h(s0, binned_d, f, N)
    resid = binned_w**2 - gamma0**2
    i = int(np.argmax(np.abs(resid)))
    k0 = abs(resid[i]) / h0[i] if h0[i] > 0 else 1.0
    log_k0 = np.log(k0)

    s_upper = s_upper_factor * float(binned_d.max())
    lower = [f * 1.001, np.log(1e-12), 0.0]
    upper = [s_upper, np.inf, np.inf]

    popt, pcov = curve_fit(model, binned_d, binned_w,
                            p0=[min(s0, s_upper * 0.5), log_k0, gamma0],
                            bounds=(lower, upper), maxfev=20000)
    s_fit, log_k_fit, gamma_fit = popt
    perr = np.sqrt(np.diag(pcov))
    pred = model(binned_d, *popt)
    rmse = float(np.sqrt(np.mean((pred - binned_w) ** 2)))
    s_at_upper_bound = bool(s_fit > 0.99 * s_upper)

    k_fit = float(np.exp(log_k_fit))
    k_stderr = float(k_fit * perr[1])  # delta method: d(exp(log_k))/d(log_k) = exp(log_k) = k

    return {
        "s": float(s_fit), "k": k_fit, "gamma": float(gamma_fit),
        "s_stderr": float(perr[0]), "k_stderr": k_stderr, "gamma_stderr": float(perr[2]),
        "rmse": rmse, "n_bins": len(binned_d), "s_at_upper_bound": s_at_upper_bound,
    }


def load_dfocus_results(results_path: Path, side: str = "L") -> dict:
    """Read focaldist_estimation_calibimgs.py's dfocus_results.txt and return
    {focal_name: d_focus} for the given side ("L" == EOS_6D_A, "R" == EOS_6D_B), one independent
    D_focus estimate per focal length. Rows whose status isn't "ok" are skipped."""
    results = {}
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["side"] != side or row["status"] != "ok":
                continue
            results[row["focal"]] = float(row["d_focus"])
    return results


def fit_k_gamma_pooled(settings, n_bins: int = 40, min_bin_samples: int = 20):
    """Given several (depth, width, s, f, N) settings -- e.g. one per (camera side, focal
    length), each with its own known focus distance s (an independent D_focus estimate) and
    focal length f -- fit a SINGLE k and gamma shared across all of them:

        k * h(s, d, f, N) + gamma**2 = width**2

    on the assumption that k and gamma (the blur-scale and blur-floor terms) are camera- and
    focal-length-independent -- i.e. the same lens/sensor blur characteristics throughout --
    while s and f still vary per setting. With every setting's own s and f already fixed, the
    equation is linear in x = h(s, d, f, N) and y = width**2 for every setting, so all settings'
    (x, y) points are pooled and solved with a single ordinary-least-squares fit, rather than
    fitting k/gamma separately per setting.

    Each setting's (depth, width) is first reduced with bin_samples() (median per depth bin,
    denoising -- see fit_S_gamma()'s docstring) before pooling.

    settings: iterable of dicts, each with "depth", "width", "s", "f", "N", and optionally
        "label" (kept only for the per-setting diagnostics below).

    Returns {"k", "k_stderr", "c", "c_stderr", "gamma", "gamma_stderr", "rmse", "n_points",
    "n_settings", "per_setting"}, where "c" = gamma**2 is the raw OLS intercept (gamma is None
    if c < 0, since a negative intercept has no real square root), and "per_setting" is a list
    of {"label", "n_bins", "rmse"} -- each setting's own residual against the *shared* k/gamma,
    for checking whether the shared fit is actually a good match to every setting individually.
    """
    all_x, all_y, per_setting = [], [], []
    for setting in settings:
        binned_d, binned_w = bin_samples(setting["depth"], setting["width"],
                                          n_bins=n_bins, min_bin_samples=min_bin_samples)
        x = h(setting["s"], binned_d, setting["f"], setting["N"])
        y = binned_w ** 2
        all_x.append(x)
        all_y.append(y)
        per_setting.append({"label": setting.get("label"), "n_bins": len(binned_d),
                             "x": x, "w": binned_w})

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    fit = linregress(all_x, all_y)
    k, c = fit.slope, fit.intercept
    gamma = float(np.sqrt(c)) if c >= 0 else None
    gamma_stderr = float(fit.intercept_stderr / (2 * np.sqrt(c))) if c > 0 else None

    all_w = np.concatenate([s["w"] for s in per_setting])
    pred_w_all = np.sqrt(np.clip(k * all_x + c, 0, None))
    rmse = float(np.sqrt(np.mean((pred_w_all - all_w) ** 2)))

    for s in per_setting:
        pred_w = np.sqrt(np.clip(k * s["x"] + c, 0, None))
        s["rmse"] = float(np.sqrt(np.mean((pred_w - s["w"]) ** 2)))
        del s["x"], s["w"]

    return {
        "k": float(k), "k_stderr": float(fit.stderr),
        "c": float(c), "c_stderr": float(fit.intercept_stderr),
        "gamma": gamma, "gamma_stderr": gamma_stderr,
        "rmse": rmse, "n_points": int(len(all_x)), "n_settings": len(per_setting),
        "per_setting": per_setting,
    }


def run_k_gamma_pooled_report():
    """Fit a single shared k and gamma across every (camera side, focal length) setting (see
    fit_k_gamma_pooled()), using each setting's independent D_focus estimate (from
    focaldist_estimation_calibimgs.py's dfocus_results.txt) as its known focus distance s. Both
    EOS_6D_A (L) and EOS_6D_B (R)'s ESF-width CSVs must already exist under <camera_dir>/debug
    (see run_all_esf_extraction()). Saves a per-setting diagnostics CSV (with the shared k/gamma
    repeated on every row) and a markdown report to out_dir."""
    calib_root = Path(r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
                       r"\Global_calibration_set\ChArUco_pattern")
    dfocus_path = Path(r"D:\datasets\MODEST_processed\scene4_dfocus\dfocus_results.txt")
    out_dir = Path(r"D:\datasets\MODEST_processed\global_def_calib")
    scan = "both"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(calib_root / "pattern_info_charuco.json") as f:
        N = json.load(f)["camera_params"]["f_number"]

    settings = []
    for side, cam_name in CAMERA_DIRS.items():
        debug_dir = calib_root / cam_name / "debug"
        dfocus = load_dfocus_results(dfocus_path, side=side)
        res = load_widths_by_focal(debug_dir, scan)
        for focal_name in sorted(res, key=lambda name: int(name[3:-2])):
            if focal_name not in dfocus:
                print(f"[skip] {side} {focal_name}: no D_focus entry in {dfocus_path.name}")
                continue
            data = res[focal_name]
            focal_mm = int(focal_name[3:-2])
            settings.append({
                "label": f"{side}_{focal_name}", "side": side, "focal": focal_name,
                "focal_mm": focal_mm, "depth": data["depth"], "width": data["width"],
                "s": dfocus[focal_name], "f": focal_mm / 1000, "N": N,
            })

    fit = fit_k_gamma_pooled(settings)
    print(f"pooled: k={fit['k']:.4e} +- {fit['k_stderr']:.2e}   "
          f"gamma={fit['gamma']:.4f} +- {fit['gamma_stderr']:.4f} px   "
          f"rmse={fit['rmse']:.4f} px  (n_points={fit['n_points']}, "
          f"n_settings={fit['n_settings']})")

    rows = []
    for setting, per in zip(settings, fit["per_setting"]):
        rows.append({
            "side": setting["side"], "focal": setting["focal"], "focal_mm": setting["focal_mm"],
            "f_number": setting["N"], "s_m": setting["s"], "n_raw_samples": len(setting["depth"]),
            "n_bins": per["n_bins"], "rmse_px": per["rmse"],
            "k": fit["k"], "k_stderr": fit["k_stderr"],
            "gamma_px": fit["gamma"], "gamma_stderr": fit["gamma_stderr"],
            "c": fit["c"], "c_stderr": fit["c_stderr"],
            "pooled_rmse_px": fit["rmse"], "n_points": fit["n_points"],
        })
        print(f"  {setting['label']:>12}  n_bins={per['n_bins']:3d}  rmse={per['rmse']:.3f} px")

    fieldnames = ["side", "focal", "focal_mm", "f_number", "s_m", "n_raw_samples", "n_bins",
                  "rmse_px", "k", "k_stderr", "gamma_px", "gamma_stderr", "c", "c_stderr",
                  "pooled_rmse_px", "n_points"]
    csv_path = out_dir / "k_gamma_pooled_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    fig, (ax_fit, ax_res) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("viridis")
    focal_mms = sorted({r["focal_mm"] for r in rows})
    for setting, per in zip(settings, fit["per_setting"]):
        color = cmap(focal_mms.index(setting["focal_mm"]) / max(1, len(focal_mms) - 1))
        marker = "o" if setting["side"] == "L" else "^"
        binned_d, binned_w = bin_samples(setting["depth"], setting["width"])
        x = h(setting["s"], binned_d, setting["f"], setting["N"])
        ax_fit.scatter(x, binned_w**2, s=10, color=color, marker=marker, alpha=0.7)
    x_line = np.linspace(0, max(h(s["s"], bin_samples(s["depth"], s["width"])[0], s["f"], s["N"]).max()
                                 for s in settings), 200)
    ax_fit.plot(x_line, fit["k"] * x_line + fit["c"], color="black", linewidth=2,
                label=f"k*x + c  (k={fit['k']:.2e}, gamma={fit['gamma']:.2f}px)")
    ax_fit.set_xlabel("x = h(s, d, f, N)")
    ax_fit.set_ylabel("y = width^2 (px^2)")
    ax_fit.set_title("pooled linear fit (circle o = L, triangle ^ = R, color = focal length)")
    ax_fit.legend(fontsize=8)

    side_colors = {"L": "tab:blue", "R": "tab:red"}
    for side in ("L", "R"):
        side_rows = [r for r in rows if r["side"] == side]
        ax_res.plot([r["focal_mm"] for r in side_rows], [r["rmse_px"] for r in side_rows],
                    marker="o", color=side_colors[side], label=side)
    ax_res.axhline(fit["rmse"], color="black", linestyle="--", linewidth=1,
                    label=f"pooled rmse={fit['rmse']:.3f}px")
    ax_res.set_xlabel("focal length (mm)")
    ax_res.set_ylabel("per-setting rmse (px), using shared k/gamma")
    ax_res.set_title("per-setting fit quality under the shared k/gamma")
    ax_res.legend(fontsize=8)
    fig.suptitle(f"pooled k/gamma fit across all (side, focal length) settings (f/{N})")
    fig.tight_layout()
    plot_path = out_dir / "k_gamma_pooled_fit.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {plot_path}")

    report_lines = [
        "# Pooled k / gamma calibration report",
        "",
        f"f-number: {N}, scan direction pooled: {scan}",
        f"S values (per side, per focal length): {dfocus_path}",
        f"Settings pooled: {fit['n_settings']} (both EOS_6D_A/L and EOS_6D_B/R, "
        f"{len(focal_mms)} focal lengths each)",
        "",
        "k and gamma are assumed shared across camera side and focal length (same lens/sensor "
        "blur characteristics), so a single k, gamma is fit from every setting's pooled "
        "(h(s, d, f, N), width^2) points at once (see fit_k_gamma_pooled()), rather than fitting "
        "them separately per setting.",
        "",
        f"## Pooled result",
        "",
        f"- k = {fit['k']:.4e} +/- {fit['k_stderr']:.2e}",
        f"- gamma = {fit['gamma']:.4f} +/- {fit['gamma_stderr']:.4f} px",
        f"- c (= gamma^2, raw OLS intercept) = {fit['c']:.4f} +/- {fit['c_stderr']:.4f}",
        f"- pooled rmse = {fit['rmse']:.4f} px over {fit['n_points']} pooled bin-points",
        "",
        "## Per-setting fit quality (using the shared k, gamma)",
        "",
        "| side | focal | S (m) | n_raw | n_bins | rmse (px) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(
            f"| {r['side']} | {r['focal']} | {r['s_m']:.3f} | {r['n_raw_samples']} | "
            f"{r['n_bins']} | {r['rmse_px']:.3f} |"
        )
    report_lines += [
        "",
        f"See {plot_path.name} for the pooled linear fit and per-setting residuals, and "
        f"{csv_path.name} for the full numbers.",
    ]
    report_path = out_dir / "k_gamma_pooled_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"wrote {report_path}")


def main():
    run_all_esf_extraction()


if __name__ == "__main__":
    main()
