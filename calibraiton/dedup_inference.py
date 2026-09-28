"""
dedup_inference.py

Detects near-duplicate consecutive shots (same scene photographed twice in a
row without moving the camera) inside inference image sets, and copies a
de-duplicated dataset to a new location. The source tree is never modified.

Expects a directory layout like:

    <right_dir>/fl_<N>mm/inference/F<aperture>/*.JPG
    <left_dir>/fl_<N>mm/inference/F<aperture>/*.JPG

For each fl_<N>mm/F<aperture> leaf folder, left and right file lists are
sorted independently (same convention as stereo_calibration.py). Similarity
is measured as the mean absolute grayscale difference (on a small downscaled
copy) between a candidate frame and the last *kept* frame, taken separately
for left and right and combined with max() so a position is only dropped when
neither camera's view changed. A position is dropped if that combined score
is below MAD_THRESHOLD. The same keep-mask is applied to both sides so
left/right pairing by index stays intact.

Run:
    python dedup_inference.py
"""

import glob
import os
import shutil

import cv2
import numpy as np

SCENE_DIR = r"C:\Users\lahir\MODEST\Scene4\Scene4"
RIGHT_DIR = os.path.join(SCENE_DIR, "EOS6D_A_Right")
LEFT_DIR = os.path.join(SCENE_DIR, "EOS6D_B_Left")
OUT_SCENE_DIR = r"C:\Users\lahir\MODEST\Scene4_dedup\Scene4"

EXT = "JPG"
THUMB_SIZE = (128, 128)
MAD_THRESHOLD = 0.08  # empirically: duplicate pairs ~0.001-0.06, real scene changes ~0.19-0.29


def load_thumb(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(img, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def mad(a, b):
    return float(np.mean(np.abs(a - b))) / 255.0


def find_leaf_dirs(camera_dir):
    """All fl_<N>mm/inference/F<aperture> dirs under camera_dir, relative paths."""
    pattern = os.path.join(camera_dir, "fl_*mm", "inference", "F*")
    return sorted(
        os.path.relpath(p, camera_dir)
        for p in glob.glob(pattern)
        if os.path.isdir(p)
    )


def keep_mask(left_files, right_files):
    """Boolean keep-mask by position: keep[0] is always True; keep[i] is True
    only if either camera's thumbnail differs enough from the last kept frame.
    """
    left_thumbs = [load_thumb(f) for f in left_files]
    right_thumbs = [load_thumb(f) for f in right_files]

    n = len(left_files)
    keep = [True] * n
    last_l, last_r = left_thumbs[0], right_thumbs[0]
    for i in range(1, n):
        score = max(mad(left_thumbs[i], last_l), mad(right_thumbs[i], last_r))
        if score < MAD_THRESHOLD:
            keep[i] = False
        else:
            last_l, last_r = left_thumbs[i], right_thumbs[i]
    return keep


def dedup_leaf(rel_dir):
    left_dir = os.path.join(LEFT_DIR, rel_dir)
    right_dir = os.path.join(RIGHT_DIR, rel_dir)

    left_files = sorted(glob.glob(os.path.join(left_dir, f"*.{EXT}")))
    right_files = sorted(glob.glob(os.path.join(right_dir, f"*.{EXT}")))

    out_left_dir = os.path.join(OUT_SCENE_DIR, os.path.relpath(LEFT_DIR, SCENE_DIR), rel_dir)
    out_right_dir = os.path.join(OUT_SCENE_DIR, os.path.relpath(RIGHT_DIR, SCENE_DIR), rel_dir)
    os.makedirs(out_left_dir, exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)

    if len(left_files) != len(right_files):
        print(f"[warn] {rel_dir}: {len(left_files)} left vs {len(right_files)} right "
              f"images - skipping dedup, copying all as-is.")
        for f in left_files:
            shutil.copy2(f, os.path.join(out_left_dir, os.path.basename(f)))
        for f in right_files:
            shutil.copy2(f, os.path.join(out_right_dir, os.path.basename(f)))
        return len(left_files), len(left_files)

    if not left_files:
        return 0, 0

    mask = keep_mask(left_files, right_files)
    n_kept = sum(mask)
    print(f"{rel_dir}: kept {n_kept}/{len(left_files)}")

    for f, k in zip(left_files, mask):
        if k:
            shutil.copy2(f, os.path.join(out_left_dir, os.path.basename(f)))
    for f, k in zip(right_files, mask):
        if k:
            shutil.copy2(f, os.path.join(out_right_dir, os.path.basename(f)))

    return len(left_files), n_kept


if __name__ == "__main__":
    left_leaves = set(find_leaf_dirs(LEFT_DIR))
    right_leaves = set(find_leaf_dirs(RIGHT_DIR))
    common = sorted(left_leaves & right_leaves)

    if left_leaves - right_leaves:
        print(f"[warn] leaf dirs only in Left: {sorted(left_leaves - right_leaves)}")
    if right_leaves - left_leaves:
        print(f"[warn] leaf dirs only in Right: {sorted(right_leaves - left_leaves)}")

    total_before = 0
    total_after = 0
    for rel_dir in common:
        n_before, n_after = dedup_leaf(rel_dir)
        total_before += n_before
        total_after += n_after

    print(f"\nTotal (per side): {total_before} -> {total_after} "
          f"({total_before - total_after} dropped)")
    print(f"Output written to: {OUT_SCENE_DIR}")
