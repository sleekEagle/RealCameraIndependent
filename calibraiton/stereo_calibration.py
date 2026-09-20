"""
stereo_calibration.py

Stereo extrinsics (R, T, rectification) from paired checkerboard images,
reusing per-camera intrinsics (K, D) already computed by charuco_calibration.py.

Expects a directory layout like:

    <right_dir>/fl_<N>mm/calibration/*.JPG   (+ optional ignore.txt)
    <left_dir>/fl_<N>mm/calibration/*.JPG    (+ optional ignore.txt)

and a checkerboard pattern_info.json like:

    {
      "checkerboard": {
        "inner_corners_per_row": 6,
        "inner_corners_per_col": 4,
        "cell_height_m": 0.06,
        "cell_width_m": 0.06
      }
    }

ignore.txt (one per calibration dir) lists bare filename tokens to drop, e.g.:

    0652
    0661

A token is matched as a substring of the image basename on its own side. Left
and right file lists are sorted independently, then a single keep-mask (a
position is dropped if EITHER side's ignore.txt flags it) is applied to both,
so filenames never need to match across cameras but the pairing by position
stays intact.

Calibrates one fl_<N>mm focal length per run. Set the constants below, then:
    python stereo_calibration.py
"""

import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

FL_DIR_PREFIX = "fl_"


def clear_dir(path):
    """Delete all files directly inside path (subdirectories are left alone)."""
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            os.remove(fp)


def read_ignore_tokens(dir_path):
    """Read ignore.txt in dir_path, if present, as a set of stripped tokens."""
    path = os.path.join(dir_path, "ignore.txt")
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def list_paired_images(left_dir, right_dir, ext):
    """Sorted, ignore-filtered (left_path, right_path) pairs.

    A pair is dropped if either side's ignore.txt flags the image at that
    position, so left/right stay aligned by index.
    """
    left_paths = sorted(glob.glob(os.path.join(left_dir, f"*.{ext}")))
    right_paths = sorted(glob.glob(os.path.join(right_dir, f"*.{ext}")))
    if len(left_paths) != len(right_paths):
        sys.exit(
            f"Mismatch: {len(left_paths)} left images vs {len(right_paths)} right "
            f"images in {left_dir} / {right_dir}. Filenames must correspond 1:1 "
            f"in sorted order."
        )
    if not left_paths:
        sys.exit(f"No images found in {left_dir} / {right_dir} with extension .{ext}")

    left_tokens = read_ignore_tokens(left_dir)
    right_tokens = read_ignore_tokens(right_dir)

    def ignored(path, tokens):
        name = os.path.basename(path)
        return any(tok in name for tok in tokens)

    pairs = [
        (lp, rp) for lp, rp in zip(left_paths, right_paths)
        if not (ignored(lp, left_tokens) or ignored(rp, right_tokens))
    ]
    n_dropped = len(left_paths) - len(pairs)
    if n_dropped:
        print(f"Dropped {n_dropped} ignored pair(s) per ignore.txt")
    return pairs


def load_pattern_config(pattern_info_path):
    with open(pattern_info_path) as f:
        info = json.load(f)["checkerboard"]
    board_size = (info["inner_corners_per_row"], info["inner_corners_per_col"])
    return board_size, info["cell_width_m"], info["cell_height_m"]


def load_intrinsics(calib_dir, fl_name, expected_image_size):
    data = np.load(os.path.join(calib_dir, f"{fl_name}.npz"))
    K, D, image_size = data["K"], data["D"], tuple(data["image_size"])
    if image_size != expected_image_size:
        sys.exit(
            f"Intrinsics image_size {image_size} from {calib_dir} does not match "
            f"the stereo image size {expected_image_size} - K/D would be invalid "
            f"at this resolution."
        )
    return K, D


def find_corners(gray, board_size):
    """Detect checkerboard corners and refine them to sub-pixel accuracy."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def robust_outlier_threshold(errors, k):
    """median + k * MAD (MAD scaled by 1.4826 to be a consistent estimator of
    the std dev under a normal assumption). Returns None if the spread is
    degenerate (MAD == 0), in which case thresholding should be skipped.
    """
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    if mad == 0:
        return None
    return median + k * 1.4826 * mad


def stereo_calibrate_fl(fl_name, right_dir, left_dir, intrinsics_a_dir, intrinsics_b_dir,
                         pattern_info_path, out_dir, report_path=None, ext="JPG",
                         debug_dir=None, outlier_k=3.5):
    """Stereo-calibrate one fl_<N>mm focal length and save the result.

    right_dir/left_dir are the EOS6D_A_Right/EOS6D_B_Left roots (each containing
    an fl_<N>mm/calibration subdirectory). intrinsics_a_dir/intrinsics_b_dir hold
    the fl_<N>mm.npz files from charuco_calibration.py for the right/left cameras
    respectively.
    """
    os.makedirs(out_dir, exist_ok=True)
    if debug_dir:
        debug_dir = os.path.join(debug_dir, fl_name)
        os.makedirs(debug_dir, exist_ok=True)
        clear_dir(debug_dir)

    right_image_dir = os.path.join(right_dir, fl_name, "calibration")
    left_image_dir = os.path.join(left_dir, fl_name, "calibration")

    print(f"\n************** {fl_name} **************")
    pairs = list_paired_images(left_image_dir, right_image_dir, ext)

    board_size, cell_width_m, cell_height_m = load_pattern_config(pattern_info_path)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp[:, 0] *= cell_width_m
    objp[:, 1] *= cell_height_m

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    used_names = []
    image_size = None
    skipped = 0

    for lp, rp in pairs:
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            print(f"[skip] could not read {lp} or {rp}")
            skipped += 1
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray_l.shape[::-1]  # (width, height)

        found_l, corners_l = find_corners(gray_l, board_size)
        found_r, corners_r = find_corners(gray_r, board_size)

        if not (found_l and found_r):
            print(f"[skip] corners not found: {os.path.basename(lp)} / {os.path.basename(rp)}")
            skipped += 1
            continue

        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
        used_names.append((os.path.basename(lp), os.path.basename(rp)))

        if debug_dir:
            vis_l = cv2.drawChessboardCorners(img_l.copy(), board_size, corners_l, found_l)
            vis_r = cv2.drawChessboardCorners(img_r.copy(), board_size, corners_r, found_r)
            cv2.imwrite(os.path.join(debug_dir, f"L_{os.path.basename(lp)}"), vis_l)
            cv2.imwrite(os.path.join(debug_dir, f"R_{os.path.basename(rp)}"), vis_r)

    used = len(objpoints)
    print(f"Usable pairs: {used}/{len(pairs)}   Skipped: {skipped}")
    if used < 10:
        print("Warning: fewer than 10 usable pairs - calibration accuracy will suffer.")

    K_l, D_l = load_intrinsics(intrinsics_b_dir, fl_name, image_size)  # Left = label B
    K_r, D_r = load_intrinsics(intrinsics_a_dir, fl_name, image_size)  # Right = label A

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    ret, K_l, D_l, K_r, D_r, R, T, E, F, rvecs, tvecs, per_view_errors = cv2.stereoCalibrateExtended(
        objpoints, imgpoints_left, imgpoints_right,
        K_l, D_l, K_r, D_r, image_size,
        None, None, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"Stereo calibration RMS error: {ret:.4f} px")

    if outlier_k is not None:
        pair_errors = np.max(per_view_errors, axis=1)
        print(f"Per-pair reprojection error (px): min={pair_errors.min():.4f} "
              f"mean={pair_errors.mean():.4f} max={pair_errors.max():.4f}")

        threshold = robust_outlier_threshold(pair_errors, outlier_k)
        if threshold is None:
            print("Per-pair errors have zero MAD (all essentially equal) - skipping "
                  "outlier removal.")
            keep = np.ones(used, dtype=bool)
            n_drop = 0
        else:
            print(f"Adaptive outlier threshold (median + {outlier_k}*MAD): {threshold:.4f} px")
            keep = pair_errors < threshold
            n_drop = int(np.sum(~keep))

        if n_drop == used:
            print(f"All {used} pairs are above the {threshold:.4f}px adaptive threshold - "
                  f"keeping the un-filtered calibration instead of dropping everything.")
            n_drop = 0
            keep = np.ones(used, dtype=bool)

        if debug_dir:
            selected_dir = os.path.join(debug_dir, "selected")
            rejected_dir = os.path.join(debug_dir, "rejected")
            for d in (selected_dir, rejected_dir):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
            for (name_l, name_r), k in zip(used_names, keep):
                dst_dir = selected_dir if k else rejected_dir
                for prefix, name in (("L_", name_l), ("R_", name_r)):
                    src = os.path.join(debug_dir, f"{prefix}{name}")
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(dst_dir, f"{prefix}{name}"))

        if n_drop:
            print(f"Dropping {n_drop}/{used} pairs above {threshold:.4f}px reprojection "
                  f"error, recalibrating...")
            objpoints = [p for p, k in zip(objpoints, keep) if k]
            imgpoints_left = [p for p, k in zip(imgpoints_left, keep) if k]
            imgpoints_right = [p for p, k in zip(imgpoints_right, keep) if k]
            used -= n_drop
            ret, K_l, D_l, K_r, D_r, R, T, E, F, rvecs, tvecs, per_view_errors = cv2.stereoCalibrateExtended(
                objpoints, imgpoints_left, imgpoints_right,
                K_l, D_l, K_r, D_r, image_size,
                None, None, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
            )
            print(f"Stereo calibration RMS error after outlier removal: {ret:.4f} px")

    baseline = np.linalg.norm(T)
    print(f"Baseline (camera separation): {baseline:.4f} m")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

    out_path = os.path.join(out_dir, f"{fl_name}.npz")
    np.savez(
        out_path,
        K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        image_size=image_size, reproj_error=ret,
    )
    print(f"Saved stereo calibration to {out_path}")

    if debug_dir and used_names:
        sample_name_l, sample_name_r = used_names[0]
        sample_l = cv2.imread(os.path.join(left_image_dir, sample_name_l))
        sample_r = cv2.imread(os.path.join(right_image_dir, sample_name_r))
        if sample_l is not None and sample_r is not None:
            rect_l = cv2.remap(sample_l, map1x, map1y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(sample_r, map2x, map2y, cv2.INTER_LINEAR)
            combo = np.hstack([rect_l, rect_r])
            for y in range(0, combo.shape[0], 40):
                cv2.line(combo, (0, y), (combo.shape[1], y), (0, 255, 0), 1)
            sanity_path = os.path.join(debug_dir, "rectified_check.jpg")
            cv2.imwrite(sanity_path, combo)
            print(f"Saved rectification sanity check to {sanity_path} "
                  f"(green lines should cross the same real-world feature in both halves)")

    fl_label = fl_name[len(FL_DIR_PREFIX):]  # e.g. "28mm"
    report_lines = [
        fl_label,
        "R:",
        str(R),
        "T:",
        str(T.ravel()),
        f"Baseline: {baseline:.4f} m",
        f"Stereo reprojection RMS error: {ret:.4f} px",
        f"Pairs used: {used}/{len(pairs)}",
        "",
        "",
    ]
    if report_path:
        with open(report_path, "a") as f:
            f.write("\n".join(report_lines))
        print(f"Appended report entry to {report_path}")


SCENE_DIR = r"C:\Users\lahir\MODEST\Scene4\Scene4"
RIGHT_DIR = os.path.join(SCENE_DIR, "EOS6D_A_Right")
LEFT_DIR = os.path.join(SCENE_DIR, "EOS6D_B_Left")
INTRINSICS_A_DIR = r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_A\calibration"
INTRINSICS_B_DIR = r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_B\calibration"
PATTERN_INFO_PATH = os.path.join(SCENE_DIR, "pattern_info.json")

OUT_DIR = os.path.join(SCENE_DIR, "stereo_calibration")
DEBUG_DIR = os.path.join(SCENE_DIR, "stereo_calibration", "debug")
REPORT_PATH = os.path.join(OUT_DIR, "stereo_report.txt")
EXT = "JPG"
OUTLIER_K = 3.5  # adaptive outlier cutoff: median + OUTLIER_K * MAD of per-pair reprojection error

FL_NAMES = [
    "fl_28mm",
    "fl_32mm",
    "fl_36mm",
    "fl_40mm",
    "fl_45mm",
    "fl_50mm",
    "fl_55mm",
    "fl_60mm",
    "fl_65mm",
    "fl_70mm",
]


if __name__ == "__main__":
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    for fl_name in FL_NAMES:
        stereo_calibrate_fl(
            fl_name, RIGHT_DIR, LEFT_DIR, INTRINSICS_A_DIR, INTRINSICS_B_DIR,
            PATTERN_INFO_PATH, OUT_DIR,
            report_path=REPORT_PATH, ext=EXT,
            debug_dir=DEBUG_DIR, outlier_k=OUTLIER_K,
        )
