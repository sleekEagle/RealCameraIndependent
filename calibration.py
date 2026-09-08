"""
stereo_calibration.py

Stereo camera calibration from paired checkerboard images.


Notes:
    - board_cols/board_rows = number of INNER corners (not squares) along
      each dimension of the checkerboard, e.g. a 10x7-square board has
      9x6 inner corners.
    - square_size = physical size of one checkerboard square (any unit,
      e.g. mm) - the baseline/translation output will be in that unit.
    - Left and right image filenames must correspond 1:1 when both
      directories are sorted (e.g. left/0001.jpg <-> right/0001.jpg).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


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

def square():
    left_dir = r'C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_B_Left\fl_28mm\calibration'
    right_dir = r'C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_A_Right\fl_28mm\calibration'
    debug_dir = r'C:\Users\lahir\MODEST\debug' # Optional folder for corner/rectification visualizations
    out_dir = r'C:\Users\lahir\MODEST\calibrtion\scene4'
    board_cols = 6 # Inner corners along board width
    board_rows = 4 # Inner corners along board height
    square_size = 0.06 # hysical size of one square (e.g. mm)
    ext = 'JPG'

    board_size = (board_cols, board_rows)

    left_paths = sorted(glob.glob(os.path.join(left_dir, f"*.{ext}")))
    right_paths = sorted(glob.glob(os.path.join(right_dir, f"*.{ext}")))

    if len(left_paths) != len(right_paths):
        sys.exit(
            f"Mismatch: {len(left_paths)} left images vs {len(right_paths)} right images. "
            f"Filenames must correspond 1:1 in sorted order."
        )
    if not left_paths:
        sys.exit("No images found. Check --left_dir/--right_dir/--ext.")

    # 3D reference points for the checkerboard (same for every valid pair)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    used, skipped = 0, 0
    for lp, rp in zip(left_paths, right_paths):
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

        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            used += 1
            if debug_dir:
                vis_l = cv2.drawChessboardCorners(img_l.copy(), board_size, corners_l, found_l)
                vis_r = cv2.drawChessboardCorners(img_r.copy(), board_size, corners_r, found_r)
                cv2.imwrite(os.path.join(debug_dir, f"L_{os.path.basename(lp)}"), vis_l)
                cv2.imwrite(os.path.join(debug_dir, f"R_{os.path.basename(rp)}"), vis_r)
        else:
            print(f"[skip] corners not found: {os.path.basename(lp)} / {os.path.basename(rp)}")
            skipped += 1

    print(f"\nUsable pairs: {used}   Skipped: {skipped}")
    if used < 10:
        print(
            "Warning: fewer than 10 valid pairs - calibration accuracy will suffer. "
            "Aim for 20-40+ pairs covering the whole frame and varied board tilts/distances."
        )

    # Step 1: calibrate each camera individually
    print("\nCalibrating left camera...")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
    print(f"Left camera reprojection RMS error: {ret_l:.4f} px")

    print("Calibrating right camera...")
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)
    print(f"Right camera reprojection RMS error: {ret_r:.4f} px")

    # Step 2: stereo calibration - solves for R, T between the two cameras
    print("\nRunning stereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_l, D_l, K_r, D_r, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"Stereo calibration RMS error: {ret_stereo:.4f} px")
    print(f"Baseline (camera separation): {np.linalg.norm(T):.2f} (same unit as --square_size)")

    # Step 3: rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

    # Step 4: save results
    np.savez(
        out_dir,
        K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        image_size=image_size, reproj_error=ret_stereo,
    )
    print(f"\nSaved calibration to {out_dir}")

    # Step 5: visual sanity check - epipolar lines should align across both views
    if used > 0 and debug_dir:
        sample_l = cv2.imread(left_paths[0])
        sample_r = cv2.imread(right_paths[0])
        rect_l = cv2.remap(sample_l, map1x, map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(sample_r, map2x, map2y, cv2.INTER_LINEAR)
        combo = np.hstack([rect_l, rect_r])
        for y in range(0, combo.shape[0], 40):
            cv2.line(combo, (0, y), (combo.shape[1], y), (0, 255, 0), 1)
        out_path = os.path.join(debug_dir, "rectified_check.jpg")
        cv2.imwrite(out_path, combo)
        print(f"Saved rectification sanity check to {out_path} "
              f"(green lines should cross the same real-world feature in both halves)")


if __name__ == "__main__":
    main()