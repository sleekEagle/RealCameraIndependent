"""
refine_rectification.py

Fixes stereo calibrations whose rectification turns the cameras a lot.

Problem: in some (scene, focal length) settings the stereo calibration reports a
forward/back offset Tz of several cm between the cameras. cv2.stereoRectify then
turns both virtual cameras by 10-25 deg, which stretches the rectified images
(and their defocus blur) unevenly; Scene9 36mm is zoomed 6x.

Cause: the two zoom lenses are not set to exactly the same focal length (32, 36,
40, ... mm are not click-stops). A few percent of magnification difference
between left and right is indistinguishable from one camera standing closer to
the scene, so it shows up as a false Tz.

Fix, per setting:
  1. Measure the right/left magnification ratio from inference-image matches
     (vertical image coordinates scale with focal length only).
  2. Scale the right camera's focal length by that ratio.
  3. Re-estimate R and the direction of T from the inference matches (essential
     matrix with the corrected intrinsics); keep the calibrated |Tx| as scale.
  4. Rectify again and compare with the original on held-out inference pairs.
A setting is fixed only if its original turn is > TURN_LIMIT and the new
rectification has both a smaller turn and a smaller vertical misalignment;
otherwise the original calibration is kept.

Writes <scene>\\stereo_calibration_v2\\fl_<N>mm.npz (same keys as
stereo_calibration.py, incl. full-resolution remap tables) for fixed settings,
and refine_rectification_report.csv in CALIB_ROOT.

Run (opencv_env):
    python refine_rectification.py                        # all settings
    python refine_rectification.py --only 5:50 6:65 --force
        # only these (scene:focal length) settings; --force keeps the fix even
        # if the rule above would not (rows are updated in the existing report)
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

CALIB_ROOT = r"D:\datasets\MODEST_calibration"
DEDUP_ROOT = r"D:\datasets\MODEST_dedup"
FLS = [28, 32, 36, 40, 45, 50, 55, 60, 65, 70]
TURN_LIMIT = 5.0  # deg; settings below this are left as they are
S = 4  # matching runs at 1/4 resolution
SIFT = cv2.SIFT_create(4000)


def scene_dir(root, s):
    p = os.path.join(root, f"Scene{s}")
    return os.path.join(p, f"Scene{s}") if os.path.isdir(os.path.join(p, f"Scene{s}")) else p


def child(parent, name):
    for c in os.listdir(parent):
        if c.lower() == name.lower():
            return os.path.join(parent, c)
    return None


def inference_pairs(s, fl):
    """All (left, right) raw inference pairs of a setting (name-sorted = true pairs in MODEST_dedup)."""
    root = scene_dir(DEDUP_ROOT, s)
    infs = [child(os.path.join(glob.glob(os.path.join(root, f"EOS6D_*_{side}"))[0], f"fl_{fl}mm"), "inference")
            for side in ("Left", "Right")]
    pairs = []
    for ap in sorted(os.listdir(infs[0])):
        rd = child(infs[1], ap)
        if rd:
            pairs += list(zip(sorted(glob.glob(os.path.join(infs[0], ap, "*.JPG"))),
                              sorted(glob.glob(os.path.join(rd, "*.JPG")))))
    return pairs


def matches(lp, rp):
    """SIFT matches between a raw left/right pair, in full-resolution pixel coordinates."""
    a, b = (cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4) for p in (lp, rp))
    ka, da = SIFT.detectAndCompute(a, None)
    kb, db = SIFT.detectAndCompute(b, None)
    if da is None or db is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    g = [x for x, y in (t for t in m if len(t) == 2) if x.distance < 0.7 * y.distance]
    return (np.float32([ka[x.queryIdx].pt for x in g]) * S, np.float32([kb[x.trainIdx].pt for x in g]) * S)


def normalized(pts, K, D):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def vertical_scale(c, match_list):
    """Right/left magnification ratio: robust fit y_R = scale * y_L + offset in normalized coords."""
    ratios = []
    for pa, pb in match_list:
        na, nb = normalized(pa, c["K_l"], c["D_l"]), normalized(pb, c["K_r"], c["D_r"])
        k = np.abs(na[:, 1]) > 0.05
        if k.sum() < 30:
            continue
        A = np.column_stack([na[k, 1], np.ones(k.sum())])
        sol, *_ = np.linalg.lstsq(A, nb[k, 1], rcond=None)
        r = nb[k, 1] - A @ sol
        inl = np.abs(r) < 3 * np.median(np.abs(r)) + 1e-9
        sol, *_ = np.linalg.lstsq(A[inl], nb[k, 1][inl], rcond=None)
        ratios.append(sol[0])
    return float(np.median(ratios))


def refine_extrinsics(c, match_list):
    """R and T from the essential matrix of the inference matches; |Tx| kept from calibration."""
    PA = np.concatenate([normalized(pa, c["K_l"], c["D_l"]) for pa, _ in match_list])
    PB = np.concatenate([normalized(pb, c["K_r"], c["D_r"]) for _, pb in match_list])
    E, mask = cv2.findEssentialMat(PA, PB, np.eye(3), method=cv2.RANSAC, prob=0.9999,
                                   threshold=1.0 / c["K_l"][0, 0])
    _, R, t, _ = cv2.recoverPose(E, PA, PB, np.eye(3), mask=mask)
    T_cal = c["T"].ravel()
    t = t.ravel() * np.sign(t.ravel()[0]) * np.sign(T_cal[0])
    return R, (t / abs(t[0]) * abs(T_cal[0])).reshape(3, 1)


def rectify(c, R, T):
    size = tuple(int(x) for x in c["image_size"])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(c["K_l"], c["D_l"], c["K_r"], c["D_r"], size, R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return dict(R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)


def turn_deg(R):
    return float(np.linalg.norm(cv2.Rodrigues(R)[0]) * 180 / np.pi)


def stretch(K, R1, P1, size):
    """Min/max local magnification of the original->rectified mapping over a 5x5 grid."""
    H = P1[:, :3] @ R1 @ np.linalg.inv(K)
    f = lambda x, y: (H @ [x, y, 1.0])[:2] / (H @ [x, y, 1.0])[2]
    sv = []
    for x in np.linspace(0, size[0] - 1, 5):
        for y in np.linspace(0, size[1] - 1, 5):
            J = np.column_stack([(f(x + 1, y) - f(x - 1, y)) / 2, (f(x, y + 1) - f(x, y - 1)) / 2])
            sv += list(np.linalg.svd(J, compute_uv=False))
    return min(sv), max(sv)


def vertical_error(c, rect, match_list):
    """Median |dy| (full-res px) of the given raw matches after rectification."""
    dys = []
    for pa, pb in match_list:
        if len(pa) < 20:
            continue
        ra = cv2.undistortPoints(pa.reshape(-1, 1, 2), c["K_l"], c["D_l"], R=rect["R1"], P=rect["P1"]).reshape(-1, 2)
        rb = cv2.undistortPoints(pb.reshape(-1, 1, 2), c["K_r"], c["D_r"], R=rect["R2"], P=rect["P2"]).reshape(-1, 2)
        dy = rb[:, 1] - ra[:, 1]
        dys.append(dy[np.abs(dy - np.median(dy)) < 40])  # drop wrong matches
    return float(np.median(np.abs(np.concatenate(dys))))


def full_maps(c, rect):
    size = tuple(int(x) for x in c["image_size"])
    m = {}
    for k, (K, D, Rr, P) in {"1": (c["K_l"], c["D_l"], rect["R1"], rect["P1"]),
                             "2": (c["K_r"], c["D_r"], rect["R2"], rect["P2"])}.items():
        m[f"map{k}x"], m[f"map{k}y"] = cv2.initUndistortRectifyMap(K, D, Rr, P, size, cv2.CV_32FC1)
    return m


def main(only=None, force=False):
    cv2.setRNGSeed(0)  # reproducible RANSAC
    report = []
    print(f"{'setting':14s} {'old turn':>8s} {'ratio':>6s} | {'Tz old->new (cm)':>17s} {'turn new':>8s} "
          f"{'zoom old->new':>13s} {'dy old->new (px)':>17s} {'stretch new':>12s}  decision")
    for s in range(1, 10):
        base = os.path.join(scene_dir(CALIB_ROOT, s), "stereo_calibration")
        for fl in FLS:
            p = os.path.join(base, f"fl_{fl}mm.npz")
            if not os.path.isfile(p) or (only and (s, fl) not in only):
                continue
            c0 = np.load(p)
            old = {k: c0[k] for k in ("R1", "R2", "P1", "P2", "Q")}
            t_old = turn_deg(old["R1"])
            row = dict(scene=s, fl_mm=fl, turn_old=round(t_old, 2), decision="kept (small turn)")
            if t_old > TURN_LIMIT or (only and force):
                pairs = inference_pairs(s, fl)
                ml = [matches(lp, rp) for lp, rp in pairs]
                est, chk = ml[0::2], ml[1::2]  # estimate on half the pairs, check on the other half
                c = {k: c0[k] for k in c0.files}
                ratio = vertical_scale(c, est)
                c["K_r"] = c["K_r"].copy()
                c["K_r"][0, 0] *= ratio
                c["K_r"][1, 1] *= ratio
                R, T = refine_extrinsics(c, est)
                new = rectify(c, R, T)
                size = tuple(int(x) for x in c["image_size"])
                t_new = turn_deg(new["R1"])
                dy_old, dy_new = vertical_error(c0, old, chk), vertical_error(c, new, chk)
                z_old, z_new = old["P1"][0, 0] / c0["K_l"][0, 0], new["P1"][0, 0] / c["K_l"][0, 0]
                st = stretch(c["K_l"], new["R1"], new["P1"], size)
                by_rule = t_new < t_old and dy_new < dy_old
                fixed = by_rule or force
                row.update(ratio=round(ratio, 4), tz_old_cm=round(float(c0["T"][2, 0]) * 100, 2),
                           tz_new_cm=round(float(T[2, 0]) * 100, 2), turn_new=round(t_new, 2),
                           zoom_old=round(z_old, 3), zoom_new=round(z_new, 3), dy_old_px=round(dy_old, 2),
                           dy_new_px=round(dy_new, 2), stretch_new=f"{st[0]:.2f}-{st[1]:.2f}",
                           decision="FIXED" if by_rule else "FIXED (manual)" if fixed else "kept (no improvement)")
                print(f"Scene{s} {fl}mm".ljust(14) + f" {t_old:8.1f} {ratio:6.3f} | {row['tz_old_cm']:7.1f} -> {row['tz_new_cm']:5.1f}"
                      f" {t_new:8.1f} {z_old:6.2f} -> {z_new:4.2f} {dy_old:7.2f} -> {dy_new:5.2f} {row['stretch_new']:>12s}  {row['decision']}",
                      flush=True)
                if fixed:
                    out_dir = os.path.join(scene_dir(CALIB_ROOT, s), "stereo_calibration_v2")
                    os.makedirs(out_dir, exist_ok=True)
                    np.savez(os.path.join(out_dir, f"fl_{fl}mm.npz"),
                             K_l=c["K_l"], D_l=c["D_l"], K_r=c["K_r"], D_r=c["D_r"], R=R, T=T,
                             R1=new["R1"], R2=new["R2"], P1=new["P1"], P2=new["P2"], Q=new["Q"],
                             image_size=c["image_size"], right_focal_ratio=ratio,
                             **full_maps(c, new))
            report.append(row)
    cols = ["scene", "fl_mm", "turn_old", "ratio", "tz_old_cm", "tz_new_cm", "turn_new", "zoom_old", "zoom_new",
            "dy_old_px", "dy_new_px", "stretch_new", "decision"]
    path = os.path.join(CALIB_ROOT, "refine_rectification_report.csv")
    if only and os.path.isfile(path):  # update the processed rows of the existing report
        new = {(r["scene"], r["fl_mm"]): r for r in report}
        rows = [new.pop((int(r["scene"]), int(r["fl_mm"])), r) for r in csv.DictReader(open(path))]
        report = rows + list(new.values())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(report)
    n = sum(str(r["decision"]).startswith("FIXED") for r in report)
    print(f"\n{n} settings fixed in total, written to <scene>\\stereo_calibration_v2. Report: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="settings as scene:focal_length, e.g. 5:50 6:65")
    ap.add_argument("--force", action="store_true", help="keep the fix for --only settings regardless of the rule")
    a = ap.parse_args()
    only = {tuple(int(x) for x in s.split(":")) for s in a.only} if a.only else None
    main(only, a.force)
