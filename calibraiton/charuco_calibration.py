"""
charuco_calibration.py

Single-camera calibration from ChArUco board images.

Expects a directory layout like:

    <camera_dir>/
        fl_28mm/*.JPG
        fl_32mm/*.JPG
        ...

and a pattern config dict with a "charuco" block, e.g.:

    {
      "charuco": {
        "squares_x": 12,
        "squares_y": 16,
        "square_length_m": 0.0435,
        "marker_length_m": 0.030,
        "dictionary": "DICT_4X4_100"
      }
    }

The board geometry (squares/markers/dictionary) is a property of the
physical board, not of the focal length, so one config applies to every
fl_<N>mm subdirectory of a camera.

Walks CAMERA_DIR and calibrates every fl_<N>mm subdirectory it finds.
Set CAMERA_DIR / OUT_DIR / PATTERN_CONFIG etc. below, then:
    python charuco_calibration.py

REPORT_PATH is cleared at the start of the run and accumulates one entry
per fl_<N>mm directory processed.
"""

import glob
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


def build_board(config):
    charuco_cfg = config["charuco"]
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, charuco_cfg["dictionary"])
    )
    board = cv2.aruco.CharucoBoard(
        (charuco_cfg["squares_x"], charuco_cfg["squares_y"]),
        charuco_cfg["square_length_m"],
        charuco_cfg["marker_length_m"],
        dictionary,
    )
    return board


def detect_charuco(detector, board, gray, min_corners):
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    n_found = 0 if charuco_ids is None else len(charuco_ids)
    if charuco_corners is None or charuco_ids is None or n_found < min_corners:
        return None, None, n_found
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < min_corners:
        return None, None, n_found
    return obj_points, img_points, n_found


def per_view_errors(objpoints, imgpoints, K, D, rvecs, tvecs):
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        errors.append(err)
    return np.array(errors)


def calibrate_charuco(image_dir, config, ext="JPG", debug_dir=None, min_corners=6,
                       outlier_threshold=1.0, resize_factor=1.0):
    """Calibrate a single camera from the ChArUco images in image_dir.

    resize_factor scales every image (e.g. 0.1 -> 10% of original width/height)
    before corner detection and calibration, so the resulting K/D and
    image_size are for the resized resolution, not the original.
    """
    board = build_board(config)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 100
    detector_params.cornerRefinementMinAccuracy = 0.01

    # Loosened marker-candidate detection (defaults in comments) so markers are
    # still found when small/far-away, blurry, or under uneven lighting.
    detector_params.adaptiveThreshWinSizeMin = 3        # default 3
    detector_params.adaptiveThreshWinSizeMax = 35        # default 23
    detector_params.adaptiveThreshWinSizeStep = 4        # default 10
    detector_params.minMarkerPerimeterRate = 0.01        # default 0.03
    detector_params.polygonalApproxAccuracyRate = 0.05   # default 0.03

    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), detector_params
    )

    # detector = cv2.aruco.CharucoDetector(
    #     board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters()
    # )

    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))
    if not image_paths:
        sys.exit(f"No images found in {image_dir} with extension .{ext}")

    undetected_dir = None
    if debug_dir:
        undetected_dir = os.path.join(debug_dir, "not_enough_corners")
        if os.path.isdir(undetected_dir):
            shutil.rmtree(undetected_dir)
        os.makedirs(undetected_dir, exist_ok=True)

    all_obj_points = []
    all_img_points = []
    used_paths = []
    image_size = None
    skipped = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] could not read {path}")
            skipped += 1
            continue

        if resize_factor != 1.0:
            img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor,
                              interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        obj_points, img_points, n_found = detect_charuco(detector, board, gray, min_corners)
        if obj_points is None:
            print(f"[skip] only {n_found}/{min_corners} charuco corners found: "
                  f"{os.path.basename(path)}")
            skipped += 1
            if undetected_dir:
                cv2.imwrite(os.path.join(undetected_dir, os.path.basename(path)), img)
            continue

        all_obj_points.append(obj_points)
        all_img_points.append(img_points)
        used_paths.append(path)

        if debug_dir:
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            vis = cv2.aruco.drawDetectedCornersCharuco(img.copy(), charuco_corners, charuco_ids)
            cv2.imwrite(os.path.join(debug_dir, os.path.basename(path)), vis)

    used = len(all_obj_points)
    print(f"Usable images: {used}/{len(image_paths)}   Skipped: {skipped}")
    if used < 10:
        print("Warning: fewer than 10 usable images - calibration accuracy will suffer.")

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None, flags=cv2.CALIB_FIX_K3
    )
    print(f"Reprojection RMS error: {ret:.4f} px")

    if outlier_threshold is not None:
        errors = per_view_errors(all_obj_points, all_img_points, K, D, rvecs, tvecs)
        keep = errors < outlier_threshold
        n_drop = int(np.sum(~keep))

        if debug_dir:
            selected_dir = os.path.join(debug_dir, "selected")
            rejected_dir = os.path.join(debug_dir, "rejected")
            for d in (selected_dir, rejected_dir):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
            for path, k in zip(used_paths, keep):
                src = os.path.join(debug_dir, os.path.basename(path))
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(selected_dir if k else rejected_dir,
                                                    os.path.basename(path)))

        if n_drop:
            print(f"Dropping {n_drop}/{used} images above {outlier_threshold}px reprojection "
                  f"error, recalibrating...")
            all_obj_points = [p for p, k in zip(all_obj_points, keep) if k]
            all_img_points = [p for p, k in zip(all_img_points, keep) if k]
            used -= n_drop
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points, all_img_points, image_size, None, None
            )
            print(f"Reprojection RMS error after outlier removal: {ret:.4f} px")

    return {
        "K": K,
        "D": D,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reproj_error": ret,
        "image_size": image_size,
        "used": used,
        "total": len(image_paths),
    }


def calibrate_one_fl(camera_dir, fl_name, config, out_dir, report_path=None, ext="JPG",
                      debug_dir=None, outlier_threshold=1.0, resize_factor=1.0):
    """Calibrate a single fl_<N>mm subdirectory of camera_dir and save the result."""
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(debug_dir, fl_name)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        clear_dir(debug_dir)

    image_dir = os.path.join(camera_dir, fl_name)
    if not os.path.isdir(image_dir):
        sys.exit(f"{image_dir} does not exist")

    print(f"\n************** {fl_name} **************")
    result = calibrate_charuco(
        image_dir, config, ext=ext,
        debug_dir=debug_dir, outlier_threshold=outlier_threshold,
        resize_factor=resize_factor,
    )

    out_path = os.path.join(out_dir, f"{fl_name}.npz")
    np.savez(
        out_path,
        K=result["K"], D=result["D"],
        image_size=result["image_size"], reproj_error=result["reproj_error"],
    )
    print(f"Saved calibration to {out_path}")

    fl_label = fl_name[len(FL_DIR_PREFIX):]  # e.g. "28mm"
    report_lines = [
        fl_label,
        "Camera matrix:",
        str(result["K"]),
        "",
        "Distortion coefficients:",
        str(result["D"].ravel()),
        "",
        f"Mean reprojection error: {result['reproj_error']:.4f} pixels",
        f"Images used: {result['used']}/{result['total']}",
        "",
        "",
    ]

    if report_path:
        with open(report_path, "a") as f:
            f.write("\n".join(report_lines))
        print(f"Appended report entry to {report_path}")


def walk_dir_calibrate(camera_dir, config, out_dir, report_path=None, ext="JPG",
                        debug_dir=None, outlier_threshold=1.0, resize_factor=1.0):
    """Calibrate every fl_<N>mm subdirectory found directly inside camera_dir."""
    fl_names = sorted(
        name for name in os.listdir(camera_dir)
        if name.startswith(FL_DIR_PREFIX) and os.path.isdir(os.path.join(camera_dir, name))
    )
    if not fl_names:
        sys.exit(f"No {FL_DIR_PREFIX}* subdirectories found in {camera_dir}")

    if report_path and os.path.exists(report_path):
        os.remove(report_path)

    for fl_name in fl_names:
        calibrate_one_fl(
            camera_dir, fl_name, config, out_dir,
            report_path=report_path, ext=ext,
            debug_dir=debug_dir, outlier_threshold=outlier_threshold,
            resize_factor=resize_factor,
        )


CAMERA_DIR = r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_B"
OUT_DIR = r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_B\calibration"
DEBUG_DIR = r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\debug"
EXT = "JPG"
OUTLIER_THRESHOLD = 1.7
RESIZE_FACTOR = 1.0

# Set to a single subdirectory name (e.g. "fl_28mm") to calibrate only that
# focal length instead of walking every fl_<N>mm dir under CAMERA_DIR.
FL_NAME = "fl_70mm"
REPORT_PATH = os.path.join(OUT_DIR, f"monocal_report{FL_NAME}.txt")

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
    if FL_NAME:
        calibrate_one_fl(
            CAMERA_DIR, FL_NAME, PATTERN_CONFIG, OUT_DIR,
            report_path=REPORT_PATH, ext=EXT,
            debug_dir=DEBUG_DIR, outlier_threshold=OUTLIER_THRESHOLD,
            resize_factor=RESIZE_FACTOR,
        )
    else:
        walk_dir_calibrate(
            CAMERA_DIR, PATTERN_CONFIG, OUT_DIR,
            report_path=REPORT_PATH, ext=EXT,
            debug_dir=DEBUG_DIR, outlier_threshold=OUTLIER_THRESHOLD,
            resize_factor=RESIZE_FACTOR,
        )
