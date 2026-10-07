"""
make_manifest.py

Builds manifest.csv for MODEST_depth: one row per (color, depth) image with its
camera setting, view id, rectified intrinsics and a split label for each of the
5 setting folds (camera-generalization study).

Steps
  1. List every image in MODEST_depth (drops Scene9 28mm F4.5/F8/F13 and any
     other F-number outside the main five, e.g. the stray Scene6 28mm F11, and
     the settings in EXCLUDE).
  2. MODEST_depth images have no EXIF, so to get capture times, image <i> is
     looked up as the i-th file (sorted by name) of the same folder in
     MODEST_dedup, which is how the depth script enumerated them (checked by
     image content). MODEST_dedup is only used here; the manifest itself only
     references MODEST_depth files.
  3. Group shots of one (scene, focal length) into views: in capture order, a
     new view starts when the scene clearly changes (thumbnail correlation below
     VIEW_NCC_MIN) or when an F-number repeats within the current view.
  3b. Neighbouring views that share content (complementary F-number sets, i.e.
     one pass split in two, or very similar thumbnails) are merged into one
     split group, so near-identical content never lands on both sides.
  4. Rectified intrinsics: P1 (left) / P2 (right) from the scene's stereo
     calibration npz (the images in MODEST_depth are rectified); the fixed
     calibration in stereo_calibration_v2 is used where it exists (column
     calib = v1/v2). Also rect_zoom = rectified fx / original fx,
     rect_turn_deg = how far rectification rotated that camera, and
     f_true_mm = the camera's real lens focal length (original fx * pixel
     pitch), for computing defocus blur.
  5. Per (scene, focal length): ~20% of split groups -> test, ~10% -> val,
     rest -> train (fixed seed). Fold labels (see LABELS below).

Labels per fold (fold hides 2 focal lengths + 1 F-number):
  train            train view, seen fl, seen F
  val              val view,   seen fl, seen F
  test_seen        test view,  seen fl, seen F
  test_unseenF     test view,  seen fl, hidden F
  test_unseenfl    test view,  hidden fl, seen F
  test_both        test view,  hidden fl, hidden F
  extra_unseenfl   train/val view at a hidden fl, seen F (clean: nothing at a hidden fl is trained)
  extra_both       train/val view at a hidden fl, hidden F (clean)
  unused           train/val view, seen fl, hidden F (same view is trained at other F-numbers)

Run (opencv_env):
    python make_manifest.py
"""

import csv
import glob
import os
import random
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibraiton"))
from stereo_calibration import capture_time  # noqa: E402

DEPTH_ROOT = r"D:\datasets\MODEST_depth"
DEDUP_ROOT = r"D:\datasets\MODEST_dedup"
CALIB_ROOT = r"D:\datasets\MODEST_calibration"
OUT_CSV = os.path.join(DEPTH_ROOT, "manifest.csv")

SCENES = [f"Scene{i}" for i in range(1, 10)]
F_NUMBERS = (2.8, 5.0, 9.0, 16.0, 22.0)
FOLDS = {
    "S1": ({50, 70}, 2.8),
    "S2": ({28, 45}, 22.0),
    "S3": ({32, 60}, 5.0),
    "S4": ({36, 65}, 9.0),
    "S5": ({40, 55}, 16.0),
}
TEST_FRAC, VAL_FRAC, SEED = 0.20, 0.10, 0
PIXEL_PITCH_MM = 35.8 / 5472  # Canon EOS 6D: 35.8 mm sensor width, 5472 px
# (scene, focal length) settings left out entirely
EXCLUDE = {
    (9, 65): "rectification turns 10.9 deg and zooms 1.35x; refine_rectification.py could not fix it",
}
# Normalized correlation of 128x128 thumbnails between consecutive shots. Brightness differs
# between F-numbers, so a plain pixel difference fails. Measured: same view 0.83-1.00, new view
# mostly 0.01-0.6. Scene9 28mm neighbouring views reach ~0.9, but there an F-number repeat
# (F22->F22 or F2.8->F2.8 at the turn of each pass) marks the boundary.
VIEW_NCC_MIN = 0.7
# Neighbouring views are put in the same split group (so they land on the same side of the
# train/test split) when their F-number sets are disjoint (one pass split in two, e.g. the
# camera moved mid-pass) or when they look very similar (best thumbnail NCC >= this value).
GROUP_NCC_MIN = 0.75


def f_number(name):
    m = re.fullmatch(r"F([\d.]+)", name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def nested(root, scene):
    """Scene4 is nested as Scene4\\Scene4 in some trees."""
    p = os.path.join(root, scene)
    return os.path.join(p, scene) if os.path.isdir(os.path.join(p, scene)) else p


def child(parent, name):
    """Case-insensitive child directory."""
    for c in os.listdir(parent):
        if c.lower() == name.lower() and os.path.isdir(os.path.join(parent, c)):
            return os.path.join(parent, c)
    return None


def thumb(path):
    """128x128 grayscale thumbnail, normalized to zero mean and unit std."""
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def list_images():
    """Step 1 + 2: rows with depth paths, setting, and the linked dedup files."""
    rows = []
    for scene in SCENES:
        droot = nested(DEPTH_ROOT, scene)
        sroot = nested(DEDUP_ROOT, scene)
        cams = {side: glob.glob(os.path.join(sroot, f"EOS6D_*_{side}"))[0] for side in ("Left", "Right")}
        for fl in sorted(os.listdir(droot), key=lambda x: int(x[3:-2])):
            if (int(scene[5:]), int(fl[3:-2])) in EXCLUDE:
                continue
            for ap in sorted(os.listdir(os.path.join(droot, fl))):
                fnum = f_number(ap)
                if fnum not in F_NUMBERS:
                    continue
                src = {}
                for side, cam in cams.items():
                    inf = child(os.path.join(cam, fl), "inference")
                    adir = child(inf, ap) if inf else None
                    src[side] = sorted(os.listdir(adir)) if adir else []
                    src[side + "_dir"] = adir
                n = len(glob.glob(os.path.join(droot, fl, ap, "color", "L", "*.jpg")))
                if not (n == len(src["Left"]) == len(src["Right"])):
                    sys.exit(f"Count mismatch {scene} {fl} {ap}: depth {n}, dedup {len(src['Left'])}/{len(src['Right'])}")
                for i in range(n):
                    for side, s in (("L", "Left"), ("R", "Right")):
                        rel = os.path.relpath(os.path.join(droot, fl, ap), DEPTH_ROOT)
                        rows.append(dict(
                            color_path=os.path.join(rel, "color", side, f"{i}.jpg"),
                            depth_path=os.path.join(rel, "depth", side, f"{i}.tiff"),
                            scene=int(scene[5:]), fl_mm=int(fl[3:-2]), f_number=fnum, side=side, index=i,
                            source_left=os.path.join(src["Left_dir"], src["Left"][i]),
                            source_right=os.path.join(src["Right_dir"], src["Right"][i]),
                        ))
    return rows


def assign_views(rows):
    """Step 3: view ids from capture order + scene-change score."""
    by_setting = defaultdict(dict)  # (scene, fl) -> {(F, index): row-pair key}
    for r in rows:
        by_setting[(r["scene"], r["fl_mm"])][(r["f_number"], r["index"])] = r["source_left"]
    view_of, scores, sizes = {}, [], Counter()
    view_thumbs = defaultdict(list)  # view_id -> [(F, thumb)]
    for (scene, fl), shots in by_setting.items():
        seq = sorted(((capture_time(p), F, i, p) for (F, i), p in shots.items()))
        vid, cur, prev = 0, set(), None
        for t, F, i, p in seq:
            th = thumb(p)
            s = float(np.mean(th * prev)) if prev is not None else None  # NCC
            if s is not None:
                scores.append(s)
            if prev is not None and (F in cur or s < VIEW_NCC_MIN):
                sizes[len(cur)] += 1
                vid, cur = vid + 1, set()
            cur.add(F)
            view_id = f"s{scene}_fl{fl}_v{vid:02d}"
            view_of[(scene, fl, F, i)] = view_id
            view_thumbs[view_id].append((F, th))
            prev = th
        sizes[len(cur)] += 1
    for r in rows:
        r["view_id"] = view_of[(r["scene"], r["fl_mm"], r["f_number"], r["index"])]
    return np.array(scores), sizes, view_thumbs


def assign_split_groups(rows, view_thumbs):
    """Merge neighbouring views that share content into one split group."""
    views = defaultdict(set)
    for r in rows:
        views[(r["scene"], r["fl_mm"])].add(r["view_id"])
    group_of, merges = {}, []
    for key in sorted(views):
        vids = sorted(views[key])
        g = 0
        group_of[vids[0]] = f"{vids[0][:-4]}_g{g:02d}"
        for a, b in zip(vids, vids[1:]):
            fa = {F for F, _ in view_thumbs[a]}
            fb = {F for F, _ in view_thumbs[b]}
            ncc = max(float(np.mean(x * y)) for _, x in view_thumbs[a] for _, y in view_thumbs[b])
            if not (fa & fb) or ncc >= GROUP_NCC_MIN:
                merges.append((a, b, round(ncc, 2), "disjoint F" if not (fa & fb) else "similar"))
            else:
                g += 1
            group_of[b] = f"{b[:-4]}_g{g:02d}"
    for r in rows:
        r["split_group"] = group_of[r["view_id"]]
    return merges


def add_intrinsics(rows):
    """Step 4: rectified intrinsics from P1 (L) / P2 (R).

    Uses <scene>/stereo_calibration_v2 (fixed rectification, see
    calibraiton/refine_rectification.py) when it exists for the setting; the
    MODEST_depth images of those settings were recomputed with it.
    """
    cache = {}
    for r in rows:
        key = (r["scene"], r["fl_mm"])
        if key not in cache:
            base = nested(CALIB_ROOT, f"Scene{r['scene']}")
            v2 = os.path.join(base, "stereo_calibration_v2", f"fl_{r['fl_mm']}mm.npz")
            npz = v2 if os.path.isfile(v2) else os.path.join(base, "stereo_calibration", f"fl_{r['fl_mm']}mm.npz")
            c = np.load(npz)
            turn = lambda R: float(np.linalg.norm(cv2.Rodrigues(R)[0]) * 180 / np.pi)
            cache[key] = {"L": (c["P1"], c["K_l"], turn(c["R1"])), "R": (c["P2"], c["K_r"], turn(c["R2"])),
                          "size": tuple(int(x) for x in c["image_size"]),
                          "calib": "v2" if npz == v2 else "v1"}
        P, K, rot = cache[key][r["side"]]
        r.update(fx=float(P[0, 0]), fy=float(P[1, 1]), cx=float(P[0, 2]), cy=float(P[1, 2]),
                 width=cache[key]["size"][0], height=cache[key]["size"][1],
                 rect_zoom=round(float(P[0, 0] / K[0, 0]), 3), rect_turn_deg=round(rot, 2),
                 # real lens focal length of this camera (unrectified K), for blur: f = fx * pixel pitch
                 f_true_mm=round(float(K[0, 0]) * PIXEL_PITCH_MM, 3),
                 calib=cache[key]["calib"])


def assign_splits(rows):
    """Step 5: split-group-level test/val/train, then fold labels."""
    groups = defaultdict(set)
    for r in rows:
        groups[(r["scene"], r["fl_mm"])].add(r["split_group"])
    role = {}
    for key in sorted(groups):
        g = sorted(groups[key])
        random.Random(f"{SEED}-{key[0]}-{key[1]}").shuffle(g)  # own seed per setting: stable if others change
        n_test = max(1, round(TEST_FRAC * len(g)))
        n_val = max(1, round(VAL_FRAC * len(g))) if len(g) - n_test >= 3 else 0
        for j, gid in enumerate(g):
            role[gid] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    for r in rows:
        r["view_role"] = role[r["split_group"]]
        for fold, (hid_fl, hid_F) in FOLDS.items():
            ufl, uF = r["fl_mm"] in hid_fl, r["f_number"] == hid_F
            if r["view_role"] == "test":
                r[fold] = ("test_both" if ufl and uF else "test_unseenfl" if ufl
                           else "test_unseenF" if uF else "test_seen")
            elif ufl:
                r[fold] = "extra_both" if uF else "extra_unseenfl"
            elif uF:
                r[fold] = "unused"
            else:
                r[fold] = r["view_role"]  # train or val


if __name__ == "__main__":
    rows = list_images()
    print(f"step 1-2: {len(rows)} images")
    scores, sizes, view_thumbs = assign_views(rows)
    print("step 3: consecutive-shot NCC, histogram:")
    h, e = np.histogram(scores, bins=[-1, 0, .3, .5, .6, .7, .8, .85, .9, .95, .98, 1.01])
    for k in range(len(h)):
        print(f"   {e[k]:.2f}-{e[k + 1]:.2f}: {h[k]}")
    print("   views by number of F-numbers:", dict(sorted(sizes.items())))
    merges = assign_split_groups(rows, view_thumbs)
    print(f"step 3b: {len(merges)} neighbouring-view merges into split groups:")
    for m in merges:
        print("   ", *m)
    add_intrinsics(rows)
    assign_splits(rows)
    cols = ["color_path", "depth_path", "scene", "fl_mm", "f_number", "side", "index", "view_id", "split_group", "view_role",
            "fx", "fy", "cx", "cy", "width", "height", "rect_zoom", "rect_turn_deg", "f_true_mm", "calib", *FOLDS]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")
