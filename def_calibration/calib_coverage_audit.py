"""
def_calibration/calib_coverage_audit.py -- what depths do the ChArUco calibration images cover,
per camera, focal length, session (capture date) and F-number?

Needed before fitting blur vs depth (base blur sigma0, blur constant k, focus distance s): a group
can only pin its focus distance if its board depths lie on both sides of it. The earlier analysis
(focaldist_estimation_calibimgs.py) pooled all images per focal length; the images, however, come
from two sessions (2026-01-31 and 2026-05-27..06-01) and two F-numbers (F16 and F20, F22 at 70 mm),
which may have different focus settings.

Per image: EXIF date/time, F-number and lens focal length; ChArUco corners detected at full
resolution; board pose by solvePnP with that focal length's calibrated K, D; depth (Z along the
optical axis) of every detected corner.

Outputs in --out:
    images.csv       one row per image: camera, fl, date, time, N, n_corners, depth min/median/max,
                     reprojection error
    groups.csv       one row per (camera, fl, date, N): images, detected, depth range, quantiles
    coverage_<camera>.png   depth of every detected corner per focal length, coloured by date/N

    python def_calibration/calib_coverage_audit.py          (opencv_env)
"""
import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = r"D:\datasets\MODEST_focus\calib_audit"
# same as focaldist_estimation_calibimgs.py (not imported: that module needs scipy, opencv_env has none)
CALIB_ROOT = (r"D:\datasets\MODEST\Global_calibration_set\MODEST_ChArUco"
              r"\Global_calibration_set\ChArUco_pattern")
CAMERA_DIRS = {"L": "EOS_6D_A", "R": "EOS_6D_B"}


def load_pattern_info(calib_root):
    import json
    with open(Path(calib_root) / "pattern_info_charuco.json") as f:
        return json.load(f)["charuco"]


def build_board(info):
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, info["dictionary"]))
    return aruco.CharucoBoard((info["squares_x"], info["squares_y"]),
                              info["square_length_m"], info["marker_length_m"], dictionary)


def load_intrinsics(camera_dir, focal_name):
    npz = np.load(Path(camera_dir) / "calibration" / f"{focal_name}.npz")
    return npz["K"], npz["D"]


def exif(path):
    e = Image.open(path)._getexif() or {}
    dt = e.get(36867, "") or ""
    n = e.get(33437)
    fl = e.get(37386)
    return dt[:10].replace(":", "-"), dt[11:19], float(n) if n else None, float(fl) if fl else None


def process(args):
    path, side, fl_name = args
    root = Path(CALIB_ROOT)
    info = load_pattern_info(root)
    board = build_board(info)
    detector = aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()
    K, D = load_intrinsics(root / CAMERA_DIRS[side], fl_name)
    date, time_, n, lens_fl = exif(path)
    row = {"camera": side, "fl_mm": int(fl_name[3:-2]), "image": path.name, "date": date, "time": time_,
           "N": n, "lens_fl": lens_fl, "n_corners": 0, "d_min": "", "d_med": "", "d_max": "", "reproj_px": ""}
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) < 6:
        return row, None
    obj = obj_all[ids.flatten()]
    img = corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D)
    if not ok:
        return row, None
    R, _ = cv2.Rodrigues(rvec)
    depth = (R @ obj.T + tvec).T[:, 2]
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    row.update(n_corners=len(ids), d_min=round(float(depth.min()), 3), d_med=round(float(np.median(depth)), 3),
               d_max=round(float(depth.max()), 3), reproj_px=round(err, 3))
    return row, depth.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(CALIB_ROOT)
    jobs = []
    for side, cam in CAMERA_DIRS.items():
        for fl_dir in sorted((root / cam).glob("fl_*mm"), key=lambda p: int(p.name[3:-2])):
            jobs += [(p, side, fl_dir.name) for p in sorted(fl_dir.iterdir()) if p.suffix.lower() == ".jpg"]
    print(f"{len(jobs)} images, {a.workers} workers", flush=True)

    rows, depths = [], {}
    with Pool(a.workers) as pool:
        for i, (row, d) in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
            rows.append(row)
            if d is not None:
                depths[(row["camera"], row["fl_mm"], row["image"])] = (row["date"], row["N"], d)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: (r["camera"], r["fl_mm"], r["date"], r["time"]))
    with open(out / "images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(lambda: {"images": 0, "detected": 0, "d": []})
    for r in rows:
        g = groups[(r["camera"], r["fl_mm"], r["date"], r["N"])]
        g["images"] += 1
    for (cam, fl, img), (date, n, d) in depths.items():
        g = groups[(cam, fl, date, n)]
        g["detected"] += 1
        g["d"].append(d)
    with open(out / "groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "fl_mm", "date", "N", "images", "detected", "corners",
                    "d_min", "d_p5", "d_p25", "d_med", "d_p75", "d_p95", "d_max"])
        for (cam, fl, date, n), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3] or 0)):
            if g["d"]:
                d = np.concatenate(g["d"])
                q = np.percentile(d, [0, 5, 25, 50, 75, 95, 100]).round(2)
                w.writerow([cam, fl, date, n, g["images"], g["detected"], len(d), *q])
            else:
                w.writerow([cam, fl, date, n, g["images"], 0, 0] + [""] * 7)

    # coverage plot: one row of points per focal length, coloured by (date, N)
    keys = sorted({(date, n) for (_, _, _), (date, n, _) in depths.items()}, key=lambda x: (x[0], x[1] or 0))
    colors = {k: f"C{i}" for i, k in enumerate(keys)}
    for cam in CAMERA_DIRS:
        fig, ax = plt.subplots(figsize=(10, 6))
        fls = sorted({fl for (c, fl, _) in depths if c == cam})
        for (c, fl, img), (date, n, d) in depths.items():
            if c != cam:
                continue
            off = (keys.index((date, n)) - len(keys) / 2) * 0.35
            ax.scatter(d, np.full(len(d), fls.index(fl) + off * 0.25), s=2, alpha=0.3, color=colors[(date, n)])
        for k in keys:
            ax.scatter([], [], color=colors[k], label=f"{k[0]}  F{k[1]:g}")
        ax.set_yticks(range(len(fls)), [f"{fl} mm" for fl in fls])
        ax.set_xlabel("depth of detected board corners (m)")
        ax.set_title(f"camera {cam} ({CAMERA_DIRS[cam]}): board depths per focal length, by date and F-number")
        ax.legend(fontsize=7, markerscale=4, ncol=2)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"coverage_{cam}.png", dpi=130)
        plt.close(fig)
    print(f"done: {sum(r['n_corners'] > 0 for r in rows)}/{len(rows)} images with a board pose -> {out}")


if __name__ == "__main__":
    main()
