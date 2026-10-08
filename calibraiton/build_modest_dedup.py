"""
build_modest_dedup.py

Builds a de-duplicated copy of MODEST in which every image is part of a true
left/right stereo pair. The source tree is never modified.

For every scene and focal length:
  1. Left and right images are paired by EXIF capture time (same method as
     stereo_calibration.py: per-focal-length clock offset, 0.5 s tolerance).
     Calibration images are paired within the calibration folder. Inference
     images are paired across all aperture folders of the focal length, so a
     pair whose two halves were filed under different aperture folders is still
     found; it is placed in the folder of the aperture recorded in EXIF.
     Images without a partner are not copied.
  2. Inference only: pairs are walked in capture order per aperture folder and
     a pair is dropped as a near-duplicate of the last kept pair when neither
     camera's view changed (mean absolute difference of 128x128 grayscale
     thumbnails, max over both cameras), using the burst-aware thresholds
     below. Calibration pairs are all kept.
  3. Kept pairs are copied to the same relative path under OUT_ROOT, together
     with scene metadata (about.txt, pattern_info.json, ...) and the stereo
     calibration results (*.npz, stereo_report.txt; not the debug folder).
     Where a camera's file counter wrapped inside a folder (IMG_9996 ->
     IMG_0010), that camera's files in that folder are renamed to a continuous
     5-digit counter (IMG_09996, IMG_10010) so name order equals capture order.
  4. Every output folder is verified: sorting left and right files by name and
     zipping them must give exactly the kept pairs.

A per-pair manifest (pairs_manifest.csv) and a summary (dedup_summary.txt) are
written to OUT_ROOT.

Run:
    python build_modest_dedup.py [out_root] [scene ...]
"""

import csv
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

from stereo_calibration import PAIR_TIME_TOL_S, camera_dir, capture_time

MODEST_DIR = r"D:\datasets\MODEST"
OUT_ROOT = r"D:\datasets\MODEST_dedup"
SCENES = [
    "Scene1", "Scene2", "Scene3", os.path.join("Scene4", "Scene4"), "Scene5",
    "Scene6", "Scene7", "Scene7_illusions", "Scene8", "Scene9",
]
METADATA_FILES = ["about.txt", "pattern_info.json", "sample_selection.txt"]
EXT = "JPG"
THUMB_SIZE = (128, 128)
# A pair is a duplicate of the last kept pair when either
#   - it was shot within BURST_GAP_S of the previous pair and scores < BURST_DUP_THRESHOLD, or
#   - it scores < DUP_THRESHOLD whatever the gap (same view re-shot after a pause).
# Measured over all 7,539 inference pairs: burst shots (1.3-3.5 s apart) all score < 0.074;
# re-shots after a pause score <= 0.037; real new positions score >= 0.066 (low-texture
# scenes such as Scene9 go as low as 0.066, so the old single 0.08 threshold dropped them).
BURST_GAP_S = 3.5
BURST_DUP_THRESHOLD = 0.08
DUP_THRESHOLD = 0.05


def thumb(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(img, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def mad(a, b):
    return float(np.mean(np.abs(a - b))) / 255.0


def clock_offset(left_times, right_times):
    """Left-minus-right clock offset that the most image combinations agree on."""
    diffs = np.array([lt - rt for lt in left_times for rt in right_times])
    support = [np.sum(np.abs(diffs - d) < PAIR_TIME_TOL_S) for d in diffs]
    best = diffs[int(np.argmax(support))]
    return float(np.median(diffs[np.abs(diffs - best) < PAIR_TIME_TOL_S]))


def pair_by_time(left_paths, right_paths, times, offset):
    """[(left, right, |dt|)] in left capture order; unmatched images are left out."""
    pairs, used = [], set()
    for lp in sorted(left_paths, key=times.get):
        target = times[lp] - offset
        rp = min((p for p in right_paths if p not in used), key=lambda p: abs(times[p] - target), default=None)
        if rp is not None and abs(times[rp] - target) < PAIR_TIME_TOL_S:
            pairs.append((lp, rp, abs(times[rp] - target)))
            used.add(rp)
    return pairs


def dedup(pairs, times):
    """Per-pair (keep, score) in capture order; score is vs the last kept pair."""
    out, last, prev_t = [], None, None
    for lp, rp, _ in pairs:
        tl, tr = thumb(lp), thumb(rp)
        t = times[lp]
        if last is None:
            out.append((True, None))
            last, prev_t = (tl, tr), t
            continue
        score = max(mad(tl, last[0]), mad(tr, last[1]))
        burst = t - prev_t < BURST_GAP_S
        keep = not ((burst and score < BURST_DUP_THRESHOLD) or score < DUP_THRESHOLD)
        out.append((keep, score))
        prev_t = t
        if keep:
            last = (tl, tr)
    return out


def aperture(path):
    return float(Image.open(path).getexif().get_ifd(0x8769).get(33437))


def folder_aperture(name):
    """'F16.0' / 'F16' -> 16.0; None if the name is not an aperture."""
    try:
        return float(name[1:]) if name[:1].upper() == "F" else None
    except ValueError:
        return None


def sub_dir(parent, name):
    """Case-insensitive child directory lookup (folder-name case varies per camera)."""
    if os.path.isdir(parent):
        for c in os.listdir(parent):
            if c.lower() == name.lower() and os.path.isdir(os.path.join(parent, c)):
                return os.path.join(parent, c)
    return None


def out_path(src_dir, name):
    return os.path.join(OUT_ROOT, os.path.relpath(src_dir, MODEST_DIR), name)


def copy(src, rel_root):
    dst = os.path.join(OUT_ROOT, os.path.relpath(src, rel_root))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def output_names(paths, times):
    """Output file names for one camera's kept files in one folder.

    Original names, unless the camera counter wrapped inside the folder so that
    name order != capture order; then a continuous 5-digit counter is used.
    """
    by_time = sorted(paths, key=times.get)
    if [os.path.basename(p) for p in by_time] == sorted(os.path.basename(p) for p in by_time):
        return {p: os.path.basename(p) for p in by_time}
    names, wraps, prev = {}, 0, None
    for p in by_time:
        m = re.fullmatch(rf"IMG_(\d{{4}})\.{EXT}", os.path.basename(p), re.IGNORECASE)
        if not m:
            sys.exit(f"Name order != capture order in {os.path.dirname(p)} and {p} is not IMG_####.{EXT}")
        num = int(m.group(1))
        if prev is not None and num < prev:
            wraps += 1
        prev = num
        names[p] = f"IMG_{num + 10000 * wraps:05d}.{EXT}"
    return names


def build_scene(scene, writer, summary):
    scene_dir = os.path.join(MODEST_DIR, scene)
    left_dir, _ = camera_dir(scene_dir, "Left")
    right_dir, _ = camera_dir(scene_dir, "Right")

    for name in METADATA_FILES:
        if os.path.isfile(os.path.join(scene_dir, name)):
            copy(os.path.join(scene_dir, name), MODEST_DIR)
    for f in glob.glob(os.path.join(scene_dir, "stereo_calibration", "*")):
        if os.path.isfile(f):
            copy(f, MODEST_DIR)

    for fl in sorted(set(os.listdir(left_dir)) & set(os.listdir(right_dir))):
        lfl, rfl = os.path.join(left_dir, fl), os.path.join(right_dir, fl)
        if not (os.path.isdir(lfl) and os.path.isdir(rfl)):
            continue
        times = {p: capture_time(p) for d in (lfl, rfl)
                 for p in glob.glob(os.path.join(d, "**", f"*.{EXT}"), recursive=True)}
        lt = [t for p, t in times.items() if p.startswith(lfl + os.sep)]
        rt = [t for p, t in times.items() if p.startswith(rfl + os.sep)]
        if not lt or not rt:
            continue
        offset = clock_offset(lt, rt)

        groups = {}  # (left_src_dir, right_src_dir) -> [(left, right, dt)]

        # calibration: pair within the calibration folder
        lcal, rcal = sub_dir(lfl, "calibration"), sub_dir(rfl, "calibration")
        if lcal and rcal:
            lp, rp = glob.glob(os.path.join(lcal, f"*.{EXT}")), glob.glob(os.path.join(rcal, f"*.{EXT}"))
            groups[(lcal, rcal)] = pair_by_time(lp, rp, times, offset)

        # inference: pair across all aperture folders, then file each pair by aperture
        linf, rinf = sub_dir(lfl, "inference"), sub_dir(rfl, "inference")
        if linf and rinf:
            ldirs = {d: folder_aperture(os.path.basename(d)) for d in glob.glob(os.path.join(linf, "*")) if os.path.isdir(d)}
            rdirs = {d: folder_aperture(os.path.basename(d)) for d in glob.glob(os.path.join(rinf, "*")) if os.path.isdir(d)}
            lp = [p for d in ldirs for p in glob.glob(os.path.join(d, f"*.{EXT}"))]
            rp = [p for d in rdirs for p in glob.glob(os.path.join(d, f"*.{EXT}"))]
            inf_pairs = pair_by_time(lp, rp, times, offset)
            n_unmatched = len(lp) + len(rp) - 2 * len(inf_pairs)
            if n_unmatched:
                paired = {x for a, b, _ in inf_pairs for x in (a, b)}
                lost = [f"L:{os.path.relpath(p, lfl)}" for p in lp if p not in paired] + \
                       [f"R:{os.path.relpath(p, rfl)}" for p in rp if p not in paired]
                print(f"[unmatched] {scene} {fl} inference: {len(lost)} image(s) without a partner: "
                      + ", ".join(lost))
                summary.append((f"{scene} {fl} inference (all apertures)", len(lp), len(rp),
                                len(inf_pairs), "", f"{n_unmatched} unmatched: " + ", ".join(lost)))
            for a, b, dt in inf_pairs:
                la, ra = os.path.dirname(a), os.path.dirname(b)
                if ldirs[la] != rdirs[ra]:
                    # halves filed under different apertures: use the aperture recorded in EXIF
                    f = aperture(a)
                    if aperture(b) != f:
                        msg = (f"L:{os.path.relpath(a, lfl)} (F{f:g}) + R:{os.path.relpath(b, rfl)} "
                               f"(F{aperture(b):g}) same moment but different apertures")
                        print(f"[drop] {scene} {fl}: {msg}")
                        summary.append((f"{scene} {fl} inference (all apertures)", "", "", "", "", "dropped pair: " + msg))
                        continue
                    la = next((d for d, v in ldirs.items() if v == f), la)
                    ra = next((d for d, v in rdirs.items() if v == f), ra)
                    msg = (f"L:{os.path.relpath(a, lfl)} + R:{os.path.relpath(b, rfl)} -> F{f:g} "
                           f"({os.path.relpath(la, lfl)}, {os.path.relpath(ra, rfl)})")
                    print(f"[refile] {scene} {fl}: {msg}")
                    summary.append((f"{scene} {fl} inference (all apertures)", "", "", "", "", "refiled pair: " + msg))
                groups.setdefault((la, ra), []).append((a, b, dt))

        for (ld, rd), pairs in sorted(groups.items()):
            pairs.sort(key=lambda x: times[x[0]])
            label = f"{scene} {fl} {os.path.relpath(ld, lfl)}"
            is_inference = os.path.relpath(ld, lfl).lower().startswith("inference")
            flags = dedup(pairs, times) if is_inference else [(True, None)] * len(pairs)
            kept = [(a, b) for (a, b, _), (k, _) in zip(pairs, flags) if k]
            lnames = output_names([a for a, _ in kept], times)
            rnames = output_names([b for _, b in kept], times)

            out = []
            for (a, b, dt), (keep, score) in zip(pairs, flags):
                writer.writerow([scene, fl, os.path.relpath(ld, lfl), os.path.relpath(a, lfl), os.path.relpath(b, rfl),
                                 lnames.get(a, ""), rnames.get(b, ""),
                                 f"{dt:.3f}", "" if score is None else f"{score:.4f}", int(keep)])
                if keep:
                    dl, dr = out_path(ld, lnames[a]), out_path(rd, rnames[b])
                    for src, dst in ((a, dl), (b, dr)):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                    out.append((dl, dr))

            # verify: sorted-by-name order of the output folders reproduces the kept pairs
            if out:
                out_l = sorted(glob.glob(os.path.join(os.path.dirname(out[0][0]), f"*.{EXT}")))
                out_r = sorted(glob.glob(os.path.join(os.path.dirname(out[0][1]), f"*.{EXT}")))
                if list(zip(out_l, out_r)) != out:
                    sys.exit(f"Verification failed for {label}: name-sorted output is not the kept pairs")

            renamed = sum(lnames[a] != os.path.basename(a) for a, _ in kept) + \
                      sum(rnames[b] != os.path.basename(b) for _, b in kept)
            n_l = len(glob.glob(os.path.join(ld, f"*.{EXT}")))
            n_r = len(glob.glob(os.path.join(rd, f"*.{EXT}")))
            notes = []
            if not is_inference and n_l + n_r > 2 * len(pairs):
                notes.append(f"{n_l + n_r - 2 * len(pairs)} unmatched")
            if renamed:
                notes.append(f"{renamed} renamed (counter wrap)")
            note = "; ".join(notes)
            summary.append((label, n_l, n_r, len(pairs), len(kept), note))
            print(f"{label}: L{n_l} R{n_r} -> {len(pairs)} pairs -> kept {len(kept)}"
                  + (f"  ({note})" if note else ""), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT_ROOT = sys.argv[1]
    if len(sys.argv) > 2:
        SCENES = sys.argv[2:]
    if os.path.exists(OUT_ROOT):
        sys.exit(f"{OUT_ROOT} already exists - remove it or choose another OUT_ROOT")
    os.makedirs(OUT_ROOT)
    summary = []
    with open(os.path.join(OUT_ROOT, "pairs_manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "focal_length", "output_folder", "left_source", "right_source",
                         "left_output_name", "right_output_name",
                         "time_diff_s", "dup_score_vs_last_kept", "kept"])
        for scene in SCENES:
            build_scene(scene, writer, summary)
    with open(os.path.join(OUT_ROOT, "dedup_summary.txt"), "w") as f:
        f.write("folder\tleft_in\tright_in\tpairs\tkept_pairs\tnote\n")
        for row in summary:
            f.write("\t".join(str(x) for x in row) + "\n")
    folders = [r for r in summary if isinstance(r[4], int)]
    print(f"\nKept {sum(r[4] for r in folders)} pairs from {sum(r[3] for r in folders)} "
          f"true pairs in {len(folders)} folders. Output: {OUT_ROOT}")
