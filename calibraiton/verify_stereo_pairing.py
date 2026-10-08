"""
verify_stereo_pairing.py

Sanity-checks the assumption (used by stereo_calibration.py and
dedup_inference.py) that sorting each camera's file list independently and
pairing by index gives true left/right stereo pairs.

Raw pixel diff can't be used across cameras (parallax + baseline shift the
image content), so this instead compares HSV color histograms, which are
tolerant to the shift but still sensitive to an actual scene change. For each
position i it compares:
    aligned    = hist_corr(L[i],   R[i])
    prev-shift = hist_corr(L[i],   R[i-1])
    next-shift = hist_corr(L[i],   R[i+1])
A correctly paired dataset should have "aligned" consistently as high as or
higher than the shifted comparisons, especially at scene-change boundaries
(where a real off-by-one would show a large drop). Flags:
  - any position whose aligned score is below LOW_SCORE_THRESHOLD
  - any folder where the shifted-by-one score beats aligned on average
    (a systematic pairing offset)

Run:
    python verify_stereo_pairing.py
"""

import glob
import os
import sys

import cv2
import numpy as np

SCENE_DIR = sys.argv[1] if len(sys.argv) > 1 else r"D:\datasets\MODEST\Scene4_dedup\Scene4"
RIGHT_DIR = os.path.join(SCENE_DIR, "EOS6D_A_Right")
LEFT_DIR = os.path.join(SCENE_DIR, "EOS6D_B_Left")

EXT = "JPG"
HIST_SIZE = (256, 256)
LOW_SCORE_THRESHOLD = 0.5


def hist(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, HIST_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def corr(a, b):
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def find_leaf_dirs(camera_dir):
    pattern = os.path.join(camera_dir, "fl_*mm", "inference", "F*")
    return sorted(
        os.path.relpath(p, camera_dir)
        for p in glob.glob(pattern)
        if os.path.isdir(p)
    )


def verify_leaf(rel_dir):
    left_dir = os.path.join(LEFT_DIR, rel_dir)
    right_dir = os.path.join(RIGHT_DIR, rel_dir)
    left_files = sorted(glob.glob(os.path.join(left_dir, f"*.{EXT}")))
    right_files = sorted(glob.glob(os.path.join(right_dir, f"*.{EXT}")))

    if len(left_files) != len(right_files):
        print(f"[skip] {rel_dir}: {len(left_files)} left vs {len(right_files)} right "
              f"- count mismatch, can't verify by-index pairing here.")
        return []

    n = len(left_files)
    if n == 0:
        return []

    left_h = [hist(f) for f in left_files]
    right_h = [hist(f) for f in right_files]

    aligned = [corr(left_h[i], right_h[i]) for i in range(n)]
    prev_shift = [corr(left_h[i], right_h[i - 1]) if i > 0 else None for i in range(n)]
    next_shift = [corr(left_h[i], right_h[i + 1]) if i < n - 1 else None for i in range(n)]

    flags = []
    for i in range(n):
        if aligned[i] < LOW_SCORE_THRESHOLD:
            flags.append(
                f"  [low score] i={i} {os.path.basename(left_files[i])} <-> "
                f"{os.path.basename(right_files[i])}  aligned={aligned[i]:.3f}"
            )

    valid_prev = [p for p in prev_shift if p is not None]
    valid_next = [nx for nx in next_shift if nx is not None]
    mean_aligned = float(np.mean(aligned))
    mean_prev = float(np.mean(valid_prev)) if valid_prev else None
    mean_next = float(np.mean(valid_next)) if valid_next else None

    systematic_warning = None
    if (mean_prev is not None and mean_prev > mean_aligned) or \
       (mean_next is not None and mean_next > mean_aligned):
        systematic_warning = (
            f"  [systematic offset?] mean aligned={mean_aligned:.3f} "
            f"mean prev-shift={mean_prev:.3f} mean next-shift={mean_next:.3f}"
        )

    if flags or systematic_warning:
        print(f"{rel_dir}: {n} pairs, mean aligned={mean_aligned:.3f}")
        for f in flags:
            print(f)
        if systematic_warning:
            print(systematic_warning)
    else:
        print(f"{rel_dir}: OK ({n} pairs, mean aligned={mean_aligned:.3f})")

    return flags


if __name__ == "__main__":
    left_leaves = set(find_leaf_dirs(LEFT_DIR))
    right_leaves = set(find_leaf_dirs(RIGHT_DIR))
    common = sorted(left_leaves & right_leaves)

    if left_leaves - right_leaves:
        print(f"[warn] leaf dirs only in Left: {sorted(left_leaves - right_leaves)}")
    if right_leaves - left_leaves:
        print(f"[warn] leaf dirs only in Right: {sorted(right_leaves - left_leaves)}")

    total_flags = 0
    for rel_dir in common:
        total_flags += len(verify_leaf(rel_dir))

    print(f"\nDone. {len(common)} folders checked, {total_flags} low-score pairs flagged.")
