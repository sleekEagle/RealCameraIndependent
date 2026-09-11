"""
compute_depth.py

Compute a depth map from a stereo image pair, using the calibration file
saved by stereo_calibration_charuco.py.

Requires:
    pip install opencv-contrib-python numpy

Usage:
    python compute_depth.py \
        --calib calibration.npz \
        --left_img left_0001.jpg --right_img right_0001.jpg \
        --out_dir ./depth_out

Outputs (written to --out_dir):
    rectified_left.png / rectified_right.png - rectified inputs (sanity check)
    disparity_raw.npy      - raw float32 disparity map (pixels)
    disparity_vis.png      - color-mapped disparity for visual inspection
    depth_mm.npy           - float32 depth map, in the SAME UNIT as the
                              square_length you used during calibration
                              (e.g. mm), with invalid pixels set to 0
    depth_vis.png          - color-mapped depth for visual inspection
"""

import argparse
import os

import cv2
import numpy as np


def load_calibration(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def rectify_pair(img_l, img_r, calib):
    rect_l = cv2.remap(img_l, calib["map1x"], calib["map1y"], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, calib["map2x"], calib["map2y"], cv2.INTER_LINEAR)
    return rect_l, rect_r


def compute_disparity(rect_l_gray, rect_r_gray, num_disparities=256, block_size=5,
                       use_wls=True):
    """Semi-global block matching, optionally refined with a WLS filter."""
    min_disp = 0
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disparities,   # must be divisible by 16
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    disp_l = left_matcher.compute(rect_l_gray, rect_r_gray).astype(np.float32) / 16.0

    if use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        disp_r = right_matcher.compute(rect_r_gray, rect_l_gray).astype(np.float32) / 16.0
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        disp_l = wls_filter.filter(disp_l.astype(np.int16), rect_l_gray,
                                    disparity_map_right=disp_r.astype(np.int16))
        disp_l = disp_l.astype(np.float32) / 16.0  # WLS filter output is also in 16x fixed-point

    return disp_l


def disparity_to_depth(disparity, Q, valid_mask):
    """Returns a depth map (same unit as calibration's square_length), 0 where invalid."""
    points_3d = cv2.reprojectImageTo3D(disparity, Q)  # H x W x 3 -> (X, Y, Z)
    depth = points_3d[:, :, 2]
    depth[~valid_mask] = 0
    return depth


def colorize(arr, mask=None, cmap=cv2.COLORMAP_JET):
    valid = arr[mask] if mask is not None else arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])  # robust range, ignores outliers
    norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cmap)
    if mask is not None:
        vis[~mask] = 0
    return vis


def main():
    right_img = r"C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_A_Right\fl_28mm\inference\F5.0\IMG_0678.JPG"
    left_img = r"C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_B_Left\fl_28mm\inference\F5.0\IMG_1601.JPG"
    calib = r"C:\Users\lahir\MODEST\calibrtion\scene4.npz"
    out_dir = r'C:\Users\lahir\MODEST\depth'
    num_disparities = 128
    block_size = 5
    min_depth = None #"Optional: clip depths below this (same unit as calibration)"
    max_depth = None # Optional: clip depths above this (same unit as calibration)
    no_wls = False

    # parser.add_argument("--no_wls", action="store_true", help="Disable WLS disparity refinement")

    os.makedirs(out_dir, exist_ok=True)
    calib = load_calibration(calib)

    img_l = cv2.imread(left_img)
    img_r = cv2.imread(right_img)
    if img_l is None or img_r is None:
        raise SystemExit(f"Could not read {left_img} or {right_img}")

    rect_l, rect_r = rectify_pair(img_l, img_r, calib)
    cv2.imwrite(os.path.join(out_dir, "rectified_left.png"), rect_l)
    cv2.imwrite(os.path.join(out_dir, "rectified_right.png"), rect_r)

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    disparity = compute_disparity(
        gray_l, gray_r,
        num_disparities=num_disparities,
        block_size=block_size,
        use_wls=not no_wls,
    )
    np.save(os.path.join(out_dir, "disparity_raw.npy"), disparity)

    valid_mask = disparity > 0.0  # SGBM marks unmatched pixels as <= 0
    disp_vis = colorize(disparity, valid_mask)
    cv2.imwrite(os.path.join(out_dir, "disparity_vis.png"), disp_vis)

    depth = disparity_to_depth(disparity, calib["Q"], valid_mask)

    if min_depth is not None:
        valid_mask &= (depth >= min_depth)
    if max_depth is not None:
        valid_mask &= (depth <= max_depth)
    depth[~valid_mask] = 0

    np.save(os.path.join(out_dir, "depth_mm.npy"), depth)
    depth_vis = colorize(depth, valid_mask)
    cv2.imwrite(os.path.join(out_dir, "depth_vis.png"), depth_vis)

    valid_depths = depth[valid_mask]
    coverage = 100.0 * valid_mask.sum() / valid_mask.size
    print(f"Valid pixel coverage: {coverage:.1f}%")
    if valid_depths.size:
        print(f"Depth range (valid pixels): {valid_depths.min():.1f} - {valid_depths.max():.1f} "
              f"(same unit as calibration's square_length)")
    print(f"Saved outputs to {out_dir}")


def get_intrinsic(path):
    calib = np.load(path)
    K_rect = calib["P1"][:, :3]
    baseline = np.linalg.norm(calib["T"])

    map1x, map1y = calib['map1x'], calib['map1y']
    map2x, map2y = calib['map2x'], calib['map2y']

    return {
        'K': K_rect,
        'baseline': baseline,
        'map1x': map1x,
        'map1y': map1y,
        'map2x': map2x,
        'map2y': map2y
    }

def rectify_img(img_path_l, img_path_r, calib):
    img_l = cv2.imread(img_path_l)
    img_r = cv2.imread(img_path_r)
    if img_l is None or img_r is None:
        raise SystemExit(f"Could not read {img_path_l} or {img_path_r}")
    
    rect_l = cv2.remap(img_l, calib['map1x'], calib['map1y'], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_l, calib['map2x'], calib['map2y'], cv2.INTER_LINEAR)

    side_by_side = cv2.hconcat([rect_l, rect_r])
    max_width, max_height = 1600, 900
    scale = min(max_width / side_by_side.shape[1], max_height / side_by_side.shape[0], 1.0)
    display = cv2.resize(side_by_side, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for y in range(0, display.shape[0], 40):
        cv2.line(display, (0, y), (display.shape[1], y), (0, 255, 0), 1)
    cv2.imshow("Original (left) vs Rectified (right)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rect


if __name__ == "__main__":
    intr = get_intrinsic(r"C:\Users\lahir\MODEST\calibrtion\scene4.npz")
    rectify_img(r"C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_B_Left\fl_28mm\inference\F2.8\IMG_1596.JPG",
                r"C:\Users\lahir\MODEST\Scene4\Scene4\EOS6D_A_Right\fl_28mm\inference\F2.8\IMG_0673.JPG",
                intr)