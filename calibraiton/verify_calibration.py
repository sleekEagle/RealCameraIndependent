"""
verify_calibration.py

Independent check on an existing calibration report: parse the K/D matrices
out of a monocal_report_*.txt file, then re-detect ChArUco corners in the
original calibration images and recompute the reprojection error via
solvePnP + projectPoints (pose-only - K/D are held fixed, not re-optimized).

This is a sanity check on the numbers already in the report, not a
recalibration: solvePnP finds the best-fit pose for each image given the
report's K/D, then measures how well that pose reprojects the detected
corners.

Set REPORT_PATH / CAMERA_DIR / PATTERN_CONFIG below, then:
    python verify_calibration.py
"""

import glob
import os
import re

import cv2
import numpy as np

from calibraiton.charuco_calibration import FL_DIR_PREFIX, build_board, detect_charuco

FLOAT_RE = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
LABEL_RE = re.compile(r"^(\d+(?:\.\d+)?mm)$")


def parse_report(report_path):
    """Parse a monocal_report .txt file into {focal_length_label: {"K", "D"}}."""
    with open(report_path, "r") as f:
        lines = [line.rstrip() for line in f]

    results = {}
    i = 0
    while i < len(lines):
        match = LABEL_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue
        label = match.group(1)

        while i < len(lines) and lines[i].strip() != "Camera matrix:":
            i += 1
        i += 1
        k_lines = []
        while i < len(lines) and lines[i].strip():
            k_lines.append(lines[i])
            i += 1
        K = np.array([float(x) for x in re.findall(FLOAT_RE, " ".join(k_lines))]).reshape(3, 3)

        while i < len(lines) and lines[i].strip() != "Distortion coefficients:":
            i += 1
        i += 1
        d_lines = []
        while i < len(lines) and lines[i].strip():
            d_lines.append(lines[i])
            i += 1
        D = np.array([float(x) for x in re.findall(FLOAT_RE, " ".join(d_lines))])

        results[label] = {"K": K, "D": D}

    return results


def verify_focal_length(image_dir, board, detector, K, D, ext="JPG", min_corners=6):
    """Re-detect corners and recompute reprojection error against a fixed K/D."""
    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))

    per_image_rms = []
    sq_errors = []
    skipped = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obj_points, img_points = detect_charuco(detector, board, gray, min_corners)
        if obj_points is None:
            skipped += 1
            continue

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        if not ok:
            skipped += 1
            continue

        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
        residuals = img_points.reshape(-1, 2) - proj.reshape(-1, 2)
        image_sq_errors = np.sum(residuals ** 2, axis=1)

        sq_errors.append(image_sq_errors)
        per_image_rms.append(float(np.sqrt(image_sq_errors.mean())))

    used = len(per_image_rms)
    overall_rms = float(np.sqrt(np.concatenate(sq_errors).mean())) if sq_errors else float("nan")

    return {
        "overall_rms": overall_rms,
        "per_image_rms": per_image_rms,
        "used": used,
        "total": len(image_paths),
        "skipped": skipped,
    }


def verify_report(report_path, camera_dir, config, ext="JPG"):
    parsed = parse_report(report_path)
    board = build_board(config)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 100
    detector_params.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), detector_params
    )

    fl_dirs = sorted(
        (d for d in os.listdir(camera_dir) if d.lower().startswith(FL_DIR_PREFIX)),
        key=lambda d: float(d[len(FL_DIR_PREFIX):-2]),
    )

    for fl_name in fl_dirs:
        label = fl_name[len(FL_DIR_PREFIX):]  # e.g. "28mm"
        if label not in parsed:
            print(f"\n{label}: no matching entry in report, skipping")
            continue

        K = parsed[label]["K"]
        D = parsed[label]["D"]

        result = verify_focal_length(
            os.path.join(camera_dir, fl_name), board, detector, K, D, ext=ext
        )

        print(f"\n{label}")
        print(f"  Images used for verification: {result['used']}/{result['total']} "
              f"(skipped {result['skipped']})")
        print(f"  Independently recomputed RMS reprojection error: {result['overall_rms']:.4f} px")
        pass


CAMERA_DIR = r"D:\datasets\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_A"
REPORT_PATH = os.path.join(CAMERA_DIR, "monocal_report_EOS_6D_A.txt")
EXT = "JPG"

PATTERN_CONFIG = {
    "charuco": {
        "squares_x": 12,
        "squares_y": 16,
        "square_length_m": 0.0435,
        "marker_length_m": 0.030,
        "dictionary": "DICT_4X4_100",
    },
}


if __name__ == "__main__":
    verify_report(REPORT_PATH, CAMERA_DIR, PATTERN_CONFIG, ext=EXT)
