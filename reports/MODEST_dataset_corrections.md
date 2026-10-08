# MODEST: corrections and filtering of the dataset

Last updated: 2026-10-07

This report lists every correction and filtering step applied to the original MODEST data, from the raw photos to the manifest used for training. It explains what was wrong, how it was found, what was done, and what is still open.

More detail on some steps is in three separate reports:
- `D:\datasets\MODEST\STEREO_CALIBRATION_REPORT.md`: stereo pairing and calibration fixes.
- `D:\datasets\MODEST_dedup\INFERENCE_DEDUP_REPORT.md`: building the de-duplicated inference set.
- `D:\datasets\MODEST_calibration\RECTIFICATION_FIX_REPORT.md`: the rectification fix.

## 1. Overview

### 1.1 The data

Two Canon EOS 6D cameras (labels A and B) on a stereo rig. 9 scenes. Each scene is shot at 10 focal lengths (28–70 mm). At each focal length there are:
- **calibration images**: a checkerboard in different poses, at F16 (Scene 9 at 28 mm: F13);
- **inference images**: the scene at 5 F-numbers (F2.8, F5, F9, F16, F22).

Which camera is on the left depends on the scene: B is left in Scenes 1, 3, 4, 6; A is left in Scenes 2, 5, 7, 8, 9.

### 1.2 Pipeline

| Step | Output | Script |
|---|---|---|
| 1. Single-camera calibration (ChArUco), per focal length | `MODEST\Global_calibration_set\...\EOS_6D_{A,B}\calibration\fl_<N>mm.npz` | `calibraiton/charuco_calibration.py` |
| 2. Stereo calibration, per scene and focal length | `MODEST_calibration\<scene>\stereo_calibration\fl_<N>mm.npz` | `calibraiton/stereo_calibration.py` |
| 3. Build true stereo pairs, remove repeated shots | `D:\datasets\MODEST_dedup` (inference), `D:\datasets\MODEST_calibration` (calibration) | `calibraiton/build_modest_dedup.py` |
| 4. Depth from each stereo pair (FoundationStereo, on Kaggle) | `D:\datasets\MODEST_depth` | Kaggle notebook (not in this repo) |
| 5. Fix bad rectifications, recompute their depth | `MODEST_calibration\<scene>\stereo_calibration_v2`, replaced folders in `MODEST_depth` | `calibraiton/refine_rectification.py` |
| 6. Training manifest and splits | `D:\datasets\MODEST_depth\manifest.csv` | `make_manifest.py` |

### 1.3 Numbers at each stage

| Stage | Inference stereo pairs | Images (L + R) |
|---|---|---|
| Raw `MODEST` inference folders | 7,539 true pairs (after pairing by time) | about 15,100 photos |
| `MODEST_dedup` (repeats removed) | 3,771 | 7,542 |
| `MODEST_depth` (with depth) | 3,548 | 7,096 |
| `manifest.csv` (used for training and testing) | 3,465 | 6,930 |

Calibration: 1,897 stereo pairs in `MODEST_calibration`.

## 2. Stereo calibration (step 2)

### 2.1 Left/right camera labels

**Problem:** `stereo_calibration.py` always loaded camera B's lens parameters for the left camera. That is wrong for Scenes 2, 5, 7, 8 and 9, where A is on the left.

**Fix:** the script reads the side from the folder name (`EOS6D_<label>_<Left|Right>`) and loads the lens parameters of that camera.

### 2.2 Pairing left and right images

**Problem:** left and right images were paired by their position in each sorted folder (1st with 1st, 2nd with 2nd). This breaks when one folder has an image the other does not: every later pair then shows two different checkerboard poses.
- In 7 of the 76 calibration settings, some pairs were wrong (4–12 wrong pairs each, in Scene 5 at 40–70 mm and Scene 6 at 32 mm).
- In 2 more settings the folders had different image counts and the script stopped (Scene 5 at 28 mm, Scene 8 at 65 mm).
- Wrong pairs gave errors of up to 183 px and camera separations of up to 1.19 m (true: about 0.2 m). In Scene 6 at 32 mm the error stayed at a normal-looking 1.07 px despite 7 wrong pairs.

**Fix:** pairs are matched by **EXIF capture time** (`list_paired_images()` in `stereo_calibration.py`):
1. Read the capture time of every image (`DateTimeOriginal` + `SubSecTimeOriginal`).
2. Estimate the clock difference between the two cameras (about 2 h): the left-minus-right time difference that the most image combinations agree on.
3. Pair each left image with the right image taken at the same moment, within 0.5 s.

Over 1,719 checked pairs, true pairs differ by a median of 0.01 s (maximum 0.45 s). Consecutive shots of one camera are at least 1.25 s apart.

Pairing by file number was tried first and rejected. The number offset between the cameras changes when one camera takes an extra shot (this happens in Scenes 3 and 6).

### 2.3 Other calibration fixes

- **Scene 1 at 50–65 mm:** the right camera has no calibration images. No stereo calibration is possible, so these focal lengths have no depth. The script now skips missing focal lengths instead of stopping.
- **Scene 4's calibration cannot be reused for Scene 1:** the camera separation differs (about 0.17 m vs 0.22 m).
- **Scene 9 has no `pattern_info.json`:** Scene 1's is used. All scenes use the same checkerboard (6 × 4 inner corners, 6 cm squares).

## 3. Images filed in the wrong folder (raw `MODEST`)

**Problem:** some images were in the wrong folder:
- calibration shots stored under `inference/F16.0` (some settings had only 4–6 calibration pairs);
- F16 inference shots stored under `calibration`;
- in some cases, one camera's copy of a calibration shot was under `F16.0` while the other camera's copy was correctly under `calibration`.

**How they were found:**
1. **Capture timeline:** each session first takes calibration shots at F16, then inference shots in a fixed aperture cycle (F22, F16, F9, F5, F2.8 and back). Images whose folder did not fit their place in the timeline were flagged.
2. **Cross-camera timing:** flagged single-camera images were matched by time to the other camera's calibration folder (all within 0.02 s).
3. **Folder counts:** after a move, each F16 inference folder must have as many images as the other apertures, in both cameras.
4. **Checkerboard check:** every moved image was checked for a visible checkerboard, by the detector and by eye where the detector failed.

**What was done:** 237 files were moved to the correct folder in the raw data (`D:\datasets\MODEST`). A first version of the rule also moved 68 F16 scene photos without a checkerboard into `calibration`. Step 4 caught this, and they were moved back.

**Record:** every move, including the reverted ones, is in `D:\datasets\MODEST\moved_calibration_files.log`.

**Left as is (not filing errors):** extra or repeated shots inside the inference cycle (Scene 2 65 mm, Scene 8 65 mm, Scene 5 50/65 mm), and two F2.8 test shots before calibration in Scene 2 at 65 mm.

## 4. De-duplicated inference set, `MODEST_dedup` (step 3)

Built by `calibraiton/build_modest_dedup.py`. The raw data is not changed.

### 4.1 Pairing

Inference images are paired by capture time (as in 2.2), across **all aperture folders of a focal length at once**. This finds pairs whose two halves are in differently named folders.

- **38 images had no partner** on the other camera and were left out. The largest cases are Scene 5 at 65 mm (one scene position shot only by the right camera, 10 images) and Scene 8 at 50 mm (11 images).
- **5 pairs were taken at different F-numbers on the two cameras** and were dropped (for example left F16, right F18). This explains the stray `F18` and `F8` folders.

| Scene | fl | Pair dropped | Retaken correctly? |
|---|---|---|---|
| 6 | 50 mm | left F16 + right F18 (2 pairs) | yes |
| 6 | 70 mm | left F8 + right F9 (1 pair) | **no**: one F9 scene position is missing |
| 9 | 28 mm | left F13 + right F8 (2 pairs) | yes |

### 4.2 Removing repeated shots

Almost every inference view was photographed twice in a row (1.3–3.5 s apart), and a few views were re-shot later. These repeats were removed.

**Rule:** pairs are walked in capture order. Each pair is compared with the last kept pair, using the mean absolute difference of 128 × 128 grayscale thumbnails (the larger of the left and right values). A pair is a repeat if:
- it was shot less than 3.5 s after the previous pair and scores below 0.08, or
- it scores below 0.05, whatever the time gap.

**Why these thresholds:** over all 7,539 pairs, burst repeats scored below 0.074, re-shots after a pause at most 0.037, and new scene positions at least 0.066. The older single threshold of 0.08 (in `dedup_inference.py`) would have dropped real new views in Scene 9, whose plain white walls give low scores.

**Result:** 7,539 pairs → **3,771 kept**; 3,768 repeats removed.

### 4.3 File names

In 7 folders, one camera's file counter wrapped from IMG_9999 to IMG_0001 inside the folder. Sorting by name then no longer matched capture order. In these folders the wrapped camera's files were renamed to a continuous 5-digit counter (for example IMG_9975 → IMG_09975, IMG_0010 → IMG_10010).
- Scene 7 at 36 mm, left camera, all 5 apertures.
- Scene 8 at 40 mm, right camera, F2.8 and F5.

After this, sorting each folder by name gives the correct stereo pairs everywhere.

### 4.4 Left out

- `Scene2_illusions`: only the left camera has images.
- `Scene9\EOS6D_A_Left\_no_exif`: processed PNG files, not camera images.

### 4.5 Checks

- Every output folder: equal left/right counts, and name-sorted order equals the time-based pairs (538 folders).
- Every image's EXIF aperture matches its folder. The only calibration images not at F16 are Scene 9 at 28 mm (F13), as its `about.txt` says.

### 4.6 Calibration data split off

The calibration image pairs, the stereo calibration results and `pattern_info.json` were moved from `MODEST_dedup` to `D:\datasets\MODEST_calibration`, with the same folder structure and their own manifest.

## 5. Depth maps, `MODEST_depth` (step 4)

Depth is computed with FoundationStereo on Kaggle, for every rectified inference pair, for both the left and the right image.

### 5.1 Missing depth for Scene 4 at 45 mm

**Problem:** depth was missing for Scene 4 at 45 mm at F2.8, F9 and F22, and F5 was incomplete (6 of 8 pairs).

**Cause:** the processing script does not skip single folders, so the run most likely stopped partway (probably the Kaggle disk filled up near the end of Scene 4). Kaggle lists folders in no fixed order, so 45 mm was processed last.

**Fix:** Scene 4 was re-run. All 411 pairs now have depth.

### 5.2 Labels: depth from each F-number's own pair

Each image uses the depth computed from its own F-number's stereo pair. We checked whether reusing the sharp F16 depth for all F-numbers would be better:
- The typical difference between an F-number's own depth and the (aligned) F16 depth is 0.1% (F9) to 0.3% (F2.8).
- Pixels that differ by more than 5%: 1.2% at F9, 3.3% at F2.8 (6.7% inside blurred areas).

So blur makes the stereo depth slightly worse at low F-numbers, mostly in blurred areas. The difference is small, and the simpler option was chosen: **each image uses its own depth.** This should be mentioned in the paper.

## 6. Rectification fix (step 5)

Full detail: `D:\datasets\MODEST_calibration\RECTIFICATION_FIX_REPORT.md`.

### 6.1 Problem

In some settings, rectification turned the two cameras by 10–25°. This stretched the rectified images, and their defocus blur, unevenly. In Scene 9 at 36 mm the images were zoomed in 6.2×. 31 of 86 settings turned more than 5°.

### 6.2 Cause

The two zoom lenses were not always set to exactly the same focal length (32, 36, 40 mm and so on are not click-stops). A 2–4% size difference between the left and right images looks the same as one camera standing closer to the scene. The calibration reported it as a forward/back offset (Tz) of several cm, and rectification turned the cameras to match.

A first attempt, using each scene's median camera geometry, failed: the turns went away, but the rows no longer lined up (vertical error up to 36 px). Vertical alignment needs each focal length's own rotation.

### 6.3 Fix

Per affected setting:
1. Measure the right/left size ratio from matched points in the inference photos.
2. Correct the right camera's focal length by that ratio.
3. Re-estimate the camera geometry from the photos (essential matrix), keeping the calibrated camera separation.
4. Rectify again, and keep the result only if both the turn and the vertical error improve.

### 6.4 Result

- **27 settings fixed** (25 by the rule, 2 by hand: Scene 5 at 50 mm and Scene 6 at 65 mm). For example, Scene 9 at 36 mm: turn 24.8° → 1.4°, zoom 6.22× → 1.05×, vertical error 63 px → 1.0 px.
- **Depth recomputed** for these 27 settings with the new calibration: 1,096 pairs (2,192 images). The new data replaced the old folders in `MODEST_depth`. The old folders were moved to `D:\datasets\MODEST_depth_replaced_before_rectfix`, with a README.
- Before replacing, every setting was checked: the new images match the new rectification, and the depth files read correctly.
- **Scene 9 at 65 mm could not be fixed** (turn 10.9°, zoom 1.35×) and is left out of the manifest.
- Scene 1 at 40 mm, Scene 4 at 32 mm and Scene 8 at 36 mm keep their original calibration (turns of 5–7°).

## 7. Filtering for training: `manifest.csv` (step 6)

Built by `make_manifest.py`. One row per image (left and right separately). Paths point to files in `MODEST_depth`.

### 7.1 Left out

| What | Images | Why |
|---|---|---|
| Scene 9, 65 mm | 110 | rectification could not be fixed (section 6.4) |
| Scene 9, 28 mm, F4.5 / F8 / F13 | 54 | F-numbers outside the main five |
| Scene 6, 28 mm, F11 | 2 | a single stray pair |
| Scene 1, 50–65 mm | (no depth) | no stereo calibration (section 2.3) |

Result: **6,930 images** (3,465 pairs).

### 7.2 Views and split groups

A **view** is one camera position and scene arrangement, shot at all five F-numbers. Images of one view show the same content with nearly the same depth, so they must stay on the same side of the train/test split.

- **Linking to capture times:** images in `MODEST_depth` have no EXIF, so image `i` is linked to the `i`-th file (sorted by name) of the same folder in `MODEST_dedup`. This was confirmed by comparing image content on a sample (difference about 0.005, versus about 0.2 for the neighbouring file).
- **Grouping into views:** within each scene and focal length, images are put in capture order. A new view starts when an F-number repeats or the image content clearly changes. Content is compared with a brightness-independent correlation (NCC) of thumbnails, because brightness differs between F-numbers. A plain pixel difference split most views apart. Result: 746 views; 683 have all five F-numbers.
- **Split groups:** neighbouring views are merged into one split group when their F-numbers complement each other (one view split in two, for example when the camera moved mid-pass) or when they look very similar (NCC ≥ 0.75). Result: 682 groups.

### 7.3 Splits (camera-generalization study)

**Held-out views:** per scene and focal length, about 20% of split groups are test views and about 10% are validation views (fixed seed per setting). The rest are training views.

**5 setting folds:** each fold hides 2 focal lengths and 1 F-number. Every focal length and every F-number is hidden exactly once. The depth-of-field extremes are fully unseen in one fold each: 70 mm at F2.8 (S1) and 28 mm at F22 (S2).

| Fold | Hidden focal lengths | Hidden F-number |
|---|---|---|
| S1 | 50, 70 mm | F2.8 |
| S2 | 28, 45 mm | F22 |
| S3 | 32, 60 mm | F5 |
| S4 | 36, 65 mm | F9 |
| S5 | 40, 55 mm | F16 |

**Labels per fold:**
- `train`, `val`;
- `test_seen`, `test_unseenF`, `test_unseenfl`, `test_both` (test views);
- `extra_unseenfl`, `extra_both` (training views at a hidden focal length; clean, because nothing at a hidden focal length is trained);
- `unused` (training views at a hidden F-number; the same view is trained at other F-numbers).

| Fold | Train | Val | Test seen | Test unseen F | Test unseen fl | Test both |
|---|---|---|---|---|---|---|
| S1 | 2,920 | 582 | 876 | 220 | 218 | 52 |
| S2 | 2,976 | 568 | 868 | 216 | 224 | 58 |
| S3 | 3,010 | 554 | 888 | 218 | 208 | 52 |
| S4 | 3,046 | 568 | 904 | 220 | 194 | 48 |
| S5 | 3,010 | 578 | 836 | 218 | 248 | 64 |

### 7.4 Camera information per image

- `fx`, `fy`, `cx`, `cy`: intrinsics of the **rectified** image (P1 for left, P2 for right). Taken from `stereo_calibration_v2` for the 27 fixed settings, otherwise from the original calibration (column `calib` = `v2` / `v1`).
- `rect_zoom`, `rect_turn_deg`: how much rectification zoomed and turned the image. Maximum now 1.19× and 12.4°; 207 images still turn more than 10° (Scene 5 at 50 and 55 mm, Scene 8 at 40 mm).
- `f_true_mm`: the camera's real lens focal length from calibration (fx × 6.54 µm pixel size). It differs from the folder name, sometimes by more than 1 mm (for example "28 mm" is 29.1–29.2 mm, "65 mm" is 63.8–66.1 mm). **Use `f_true_mm` and `fx`, not the folder name, to compute defocus blur:** blur in rectified pixels ≈ fx × (f_true / N) × |1/s − 1/d|, where s is the focus distance.

### 7.5 Checks

- No split group is on more than one side.
- No hidden setting appears in any fold's training or validation data.
- About 16–21% of each scene's images are test images.
- Left and right rectified intrinsics are identical within each setting.

## 8. Known remaining issues

1. **Vertical misalignment in some original calibrations.** In settings that kept their original calibration, matched points can be off by a few pixels vertically after rectification (typically 1–6 px; Scene 4 at 40–50 mm showed up to 10 px in later shots of a session). The two cameras seem to shift slightly during a session. This lowers stereo depth quality somewhat.
2. **Some fixed settings still turn 10–12°** (207 images). They are much better aligned than before, but somewhat stretched. They can be filtered with `rect_turn_deg`.
3. **Absolute focal length.** The fix corrected the right lens relative to the left lens. If the left lens itself is off (estimated within about 1%), the depth scale and the intrinsics of that setting are off by about the same amount.
4. **Camera movement within a view.** Across the F-numbers of one view, the camera usually stays within a few pixels (median 4 px), but in about 11% of views it moved more than 20 px. This does not affect the labels (each image has its own depth), but it matters for any method that compares images of the same view across F-numbers.
5. **Depth labels at low F-numbers** are slightly less accurate in blurred areas (section 5.2).
6. **One missing scene position:** Scene 6 at 70 mm has no F9 image for one view (section 4.1).

## 9. Files

| File or folder | What it is |
|---|---|
| `D:\datasets\MODEST` | Raw data, with the folder corrections of section 3 |
| `D:\datasets\MODEST\moved_calibration_files.log` | Every file move in the raw data |
| `D:\datasets\MODEST_dedup` | De-duplicated inference pairs; `pairs_manifest.csv`, `dedup_summary.txt` |
| `D:\datasets\MODEST_calibration` | Calibration pairs, `stereo_calibration\` (original) and `stereo_calibration_v2\` (fixed); `refine_rectification_report.csv` |
| `D:\datasets\modest_stereo_calib_v2` | Small copies of the v2 calibrations without remap tables (for Kaggle) |
| `D:\datasets\MODEST_depth` | Rectified images and depth maps; `manifest.csv` |
| `D:\datasets\MODEST_depth_replaced_before_rectfix` | Old folders replaced by the rectification fix (safe to delete) |

| Script (this repo) | Purpose |
|---|---|
| `calibraiton/stereo_calibration.py` | Stereo calibration with time-based pairing |
| `calibraiton/build_modest_dedup.py` | Builds `MODEST_dedup` |
| `calibraiton/refine_rectification.py` | Fixes bad rectifications, writes `stereo_calibration_v2` |
| `make_manifest.py` | Builds `manifest.csv` with views, splits and intrinsics |
