# Dataset splits for the camera-generalization study

Last updated: 2026-10-08

This report explains how MODEST_depth is split into training, validation and test data. The goal of the study is to test whether a monocular metric depth model works on **camera settings it has not seen**. A camera setting is a focal length and an F-number. Together they set how wide the view is and how much defocus blur the image has.

The split is stored in `D:\datasets\MODEST_depth\manifest.csv`, built by `make_manifest.py`. The training and evaluation code in `depth_bench/` reads it.

## 1. Summary

- **Five setting folds (S1–S5).** Each fold hides 2 focal lengths and 1 F-number from training. Across the 5 folds, every focal length and every F-number is hidden exactly once.
- **All 9 scenes are used for training.** This study is only about camera generalization, not new scenes.
- **About 20% of views are held out for testing and about 10% for validation.** A view is one camera position, shot at all five F-numbers. All shots of a view stay on the same side of the split.
- **Four test groups per fold:**
  - seen setting;
  - unseen F-number;
  - unseen focal length;
  - both unseen.
- **Ground truth:** each image uses the stereo depth computed from its own F-number's image pair.
- **Size:** 3,465 stereo pairs in total. Each fold trains on about 1,500 pairs and tests on about 690 pairs.

## 2. The data

MODEST was shot with two Canon EOS 6D cameras as a stereo rig.

| | Values |
|---|---|
| Scenes | 9 |
| Focal lengths | 28, 32, 36, 40, 45, 50, 55, 60, 65, 70 mm |
| F-numbers | 2.8, 5, 9, 16, 22 |
| Settings (scene × focal length) | 85 |
| Stereo pairs | 3,465 (6,930 images, Left and Right) |

Not every scene has every focal length:

- **Scene 1** has no 50, 55, 60 or 65 mm data.
- **Scene 9 at 65 mm** is left out. Its stereo rectification could not be fixed (see `MODEST_dataset_corrections.md`).
- **Scene 9 at 28 mm** keeps only two F-numbers. Its F4.5, F8 and F13 shots are not part of the five main F-numbers and are left out.
- **Scene 6 at 28 mm:** one stray F11 pair is left out.

Pairs per F-number are almost equal (681–712). Pairs per focal length range from 298 (65 mm) to 421 (70 mm).

## 3. The five setting folds

| Fold | Hidden focal lengths | Hidden F-number |
|---|---|---|
| S1 | 50, 70 mm | F2.8 |
| S2 | 28, 45 mm | F22 |
| S3 | 32, 60 mm | F5 |
| S4 | 36, 65 mm | F9 |
| S5 | 40, 55 mm | F16 |

How the folds were chosen:

- **Each value is hidden exactly once.** Every focal length and every F-number is tested as "unseen" in one fold, so the results cover the whole range.
- **The extremes are tested as "both unseen".** S1 hides 70 mm with F2.8, the setting with the shallowest depth of field (the most blur). S2 hides 28 mm with F22, the deepest depth of field (the least blur). So the hardest blur cases are tested with nothing similar in training.
- **Each fold mixes a short and a long focal length.** This way the hidden focal lengths are not all at one end of the range.

## 4. Views and split groups

### 4.1 Why split by view

Each camera position was shot at all five F-numbers, one after the other. These five images show the same content. If one of them were in training and another in testing, the model would be tested on a scene it has already seen from the same position. So the split is done by **view**, not by single image. The Left and Right images of a pair also stay together.

### 4.2 How views are found

The MODEST_depth images have no EXIF data. Each image is linked to its original file in MODEST_dedup to get its capture time. Then, within one scene and focal length, the shots are walked in capture order. A new view starts when:

- the image content changes, measured as the correlation of 128 × 128 grayscale thumbnails (below 0.7 means a new view); or
- an F-number repeats within the current view.

The result is 735 views. 672 of them have all five F-numbers. The rest are partial passes, for example when the camera moved partway through.

### 4.3 Split groups

Neighbouring views that share content are merged into one **split group**. Two neighbours are merged when:

- they have no F-number in common (one pass split in two); or
- they look very similar (thumbnail correlation of 0.75 or more).

40 split groups contain more than one view. There are 672 split groups in total. The train / val / test decision is made per split group.

### 4.4 Assigning train, val and test

Within each setting (scene × focal length), the split groups are shuffled with a fixed seed and assigned:

- 20% to **test** (at least 1);
- 10% to **val** (at least 1, if the setting has enough groups);
- the rest to **train**.

Each setting has its own seed. Rebuilding one setting does not change the split of the others.

| Role | Split groups | Pairs | Share of pairs |
|---|---|---|---|
| train | 455 | 2,337 | 67% |
| val | 85 | 445 | 13% |
| test | 132 | 683 | 20% |

Validation is a bit above 10% because of the "at least 1" rule in small settings. One setting (Scene 9, 28 mm) has only 2 split groups, so it has no validation group.

This view role is the same for all five folds. The folds differ only in which settings are hidden.

## 5. Labels per fold

Each row of the manifest has one label per fold (columns S1–S5). The label depends on the view role and on whether the image's focal length and F-number are hidden in that fold.

| Label | View role | Focal length | F-number | Used for |
|---|---|---|---|---|
| `train` | train | seen | seen | training |
| `val` | val | seen | seen | picking the best checkpoint |
| `test_seen` | test | seen | seen | test: seen setting |
| `test_unseenF` | test | seen | hidden | test: unseen F-number |
| `test_unseenfl` | test | hidden | seen | test: unseen focal length |
| `test_both` | test | hidden | hidden | test: both unseen |
| `extra_unseenfl` | train / val | hidden | seen | optional extra test data |
| `extra_both` | train / val | hidden | hidden | optional extra test data |
| `unused` | train / val | seen | hidden | not used |

Notes on the last three labels:

- **`extra_*` views are clean extra test data.** In a fold, nothing at a hidden focal length is ever trained. A train or val view at a hidden focal length is therefore new to the model, both in content and in setting. These views can be added to the unseen-focal-length tests to get more test data.
- **`unused` views are left out.** These are train or val views at a seen focal length but the hidden F-number. The same view is used for training at the other four F-numbers, so the model has already seen its content. It is not a clean test of the hidden F-number, and it cannot be trained on either.

## 6. Size of each fold (stereo pairs)

| Label | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| train | 1,460 | 1,488 | 1,505 | 1,523 | 1,505 |
| val | 291 | 284 | 277 | 284 | 289 |
| test_seen | 438 | 434 | 444 | 452 | 418 |
| test_unseenF | 110 | 108 | 109 | 110 | 109 |
| test_unseenfl | 109 | 112 | 104 | 97 | 124 |
| test_both | 26 | 29 | 26 | 24 | 32 |
| extra_unseenfl | 478 | 447 | 452 | 428 | 417 |
| extra_both | 114 | 117 | 112 | 106 | 111 |
| unused | 439 | 446 | 436 | 441 | 460 |

Each pair gives two training or test images, Left and Right, each with its own depth map. So the image counts are twice the numbers above.

## 7. Ground truth and inputs

- **Depth per F-number.** Each image's ground truth is the stereo depth computed from the image pair at the same F-number. Depth from F16 is never moved over to other F-numbers. Valid depth is 0.3–20 m; other pixels are ignored.
- **No resizing.** Resizing would shrink the defocus blur, which is what the study is about. Models see native-resolution crops (518 × 518 by default). A crop keeps fx and fy and only shifts the principal point.
- **Intrinsics.** Each row stores the rectified fx, fy, cx and cy. For 27 settings these come from the fixed calibration (`calib = v2`); the other 58 use the original one (`v1`). `f_true_mm` is the real lens focal length, used to compute blur.
- **Fixed test crops.** Each test image gets the same crop positions in every run and for every model. They come from a seed based on the image path, and each crop must have at least 50% valid depth.

## 8. Limitations

- **`test_both` is small:** 24–32 pairs per fold. Results for this group will be noisy. The `extra_both` views (about 110 pairs per fold) can be added to make it bigger.
- **Scenes are shared.** Train and test use different views of the same scenes. The results show generalization to new camera settings, not to new scenes.
- **The hidden F-number is still seen through other views.** Within the hidden focal lengths nothing is trained. But the hidden F-number's blur range overlaps with its neighbours (for example F2.8 and F5). The "unseen F-number" test measures how well the model handles values between and beyond the trained F-numbers.
- **Focus distance changes between scenes** (median 1.8–2.8 m, see `focus_distance_estimation.md`). The same F-number can therefore give different blur in different scenes.

## 9. Files

| File | Content |
|---|---|
| `make_manifest.py` | builds the manifest (views, split groups, roles, fold labels, intrinsics) |
| `D:\datasets\MODEST_depth\manifest.csv` | one row per image, with columns `view_id`, `split_group`, `view_role` and `S1`–`S5` |
| `depth_bench/config.py` | the fold definitions and test labels used by training and evaluation |
| `depth_bench/data.py` | reads the manifest and makes the crops |
