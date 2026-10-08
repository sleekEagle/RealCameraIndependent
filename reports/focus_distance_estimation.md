# Focus distance estimation for MODEST

Last updated: 2026-10-08

This report explains how we estimated the focus distance (the distance the lens was focused at) of the MODEST images. It covers what the metadata says, the methods we tried, what failed and why, and the final results.

The focus distance is needed to compute the defocus blur in each image:

blur in rectified pixels ≈ fx × (f / N) × |1/d − 1/s|

where fx is the rectified focal length in pixels, f the real lens focal length, N the F-number, d the depth of a point and s the focus distance.

## 1. Summary

- **The files do not contain the focus distance.** All images were shot in manual focus, and the lens does not report a distance.
- **Method 1 (sharpness of single images):** a fit of a sharpness measure against depth. It was unreliable for 53 of 170 camera settings, mostly at short focal lengths. Using only the low F-numbers improved it to 149 of 170.
- **Method 2 (aperture pairs, final):** compare each view at F2.8 with the same view at F22, which removes the effect of texture. All 170 fits work, with a median uncertainty of 2.4%.
- **Main finding:** the focus distance was **not** kept constant. It differs between scenes (scene medians 1.8–2.8 m), and also changes with focal length within a scene. Left and Right cameras are usually close (median difference 5.9%).
- **Results:** `D:\datasets\MODEST_focus\dfocus_aperture_pairs.csv`, one value per scene, focal length and camera.

## 2. What the metadata tells us

### 2.1 Focus mode

Canon stores the focus mode in its EXIF maker notes (CameraSettings, index 7). For **all 11,336 images** in `MODEST_dedup` and `MODEST_calibration`, the value is 3, which means **manual focus**. The camera never refocused by itself; focus only changed if someone turned the focus ring.

### 2.2 No focus distance in the files

- The lens (Canon EF 28–70 mm f/2.8L USM) does not report its focus distance: the maker-note fields FocusDistanceUpper/Lower are 0, and SubjectDistance is 0.
- The ChArUco calibration configuration has a field `"focal_distance_m": ""`, which was never filled in.
- Each scene's `about.txt` gives only the depth range of the scene (for example "1.9–3.9 m from checkerboard").

### 2.3 The global calibration does not imply a fixed focus

The lens intrinsics were calibrated once per focal length, for all scenes. This is a common simplification: zoom changes the camera matrix a lot, focus only a little. It does not mean the focus was kept fixed. If focus changes from 1.5 m to 4 m, the effective focal length in pixels changes by about 1.2% at 28 mm and about 3.0% at 70 mm.

### 2.4 The lenses were sometimes adjusted during a session

The lens reports its zoom position (in whole mm) in every image. In 6 settings this number changes partway through a session (for example Scene 4 at 50 mm, left camera: 50 mm for 19 images, 48 mm for 53). So the zoom ring was sometimes touched mid-session, and the focus ring next to it may have been too. In 9 more settings the left and right lenses report different values for the whole session (for example Scene 9 at 36 mm: 37 vs 36 mm), which matches the zoom mismatch found in the rectification fix.

### 2.5 Focus within one view does not change

Each view is shot at all five F-numbers in quick succession. Across these, the image scale changes by only 0.01–0.04%. A focus change would change it by roughly 0.5–1%. So within a view the focus was constant.

## 3. Method 1: sharpness of single images

### 3.1 How it works

Script: `def_calibration/focaldist_estimation_imgs.py` (the per-image measurement and the fit), driven over all data by `def_calibration/focus_distance_survey.py`.

1. Cut each image into 32 × 32 patches. Keep patches with valid, nearly flat depth and enough texture.
2. Per patch, measure sharpness as the variance of the Laplacian, and record the patch depth.
3. Per F-number, bin the patches by depth and take a high percentile of sharpness per bin. The high percentile favours the best-textured patches.
4. Fit all F-numbers of a setting together with one shared focus distance:
   sharpness ≈ K × N^(−p) × (|1/d − 1/s| + ε)^p
5. Mark a fit as **reliable** when:
   - it agrees with a simple check (the depth of the sharpest bin at the lowest F-number) within 25%;
   - its uncertainty is below 20%;
   - the focus distance lies within (or near) the depth range of the scene.

### 3.2 Results

| Fit uses | Reliable fits | Reliable at 28–40 mm | Left vs Right (both reliable) | Spread between views (median) |
|---|---|---|---|---|
| All 5 F-numbers | 117 / 170 | 41 / 72 | 3.7% (50 pairs) | 10.6% |
| F2.8 + F5 | 149 / 170 | 57 / 72 | 4.7% (68 pairs) | 8.1% |
| F2.8 only | 154 / 170 | 59 / 72 | 4.2% (72 pairs) | – |

### 3.3 Why it fails

1. **Sharpness mixes blur with texture.** A plain wall has a low Laplacian variance even when it is in focus. The binning reduces this effect, but cannot remove it.
2. **Weak blur signal at short focal lengths.** At 28 mm, focused at 2 m, the depth of field at F22 runs from about 0.75 m to infinity: the image is sharp everywhere and carries no information about focus. At F2.8 it is about 1.65–2.5 m. With all five F-numbers weighted equally, the flat F16 and F22 curves add noise without information.
3. **A free exponent extrapolates badly.** When the focus lies beyond most of the scene, the fit extends a power law with a free exponent and can return absurd values (24 m, 140 m, 202 m in Scene 5 at 28–40 mm).

Using only the low F-numbers addresses reason 2, which is why it helps. Reasons 1 and 3 remain.

### 3.4 Other things tried in method 1

- **A faster, vectorized patch measurement** (one Laplacian per image instead of per patch). It turned out slower (2.4 s vs 1.3 s per image, because the original skips filtered patches early) and gave slightly different values (about 5% higher). It was dropped; the original function is used unchanged.
- **Software problems:**
  - `pip install scipy` into `opencv_env` produced a scipy that crashed Python on import (it does not match that environment's conda numpy). It was removed again; the fitting runs in the `torch_plot` environment.
  - Starting 12 worker processes that each load scipy and matplotlib ran Windows out of memory ("paging file too small"). Worker processes are now started only for images that are not yet cached.

## 4. Considered and rejected: edge spread functions

The repository also has an edge-spread approach (`def_calibration/defocus_calib.py`), which measures blur across sharp, straight edges. It works on the calibration board, but natural scene photos have few sharp straight edges, so it cannot be used on the inference images. The calibration images are not useful either: they are shot at F16, where nearly everything is in focus.

## 5. Method 2: aperture pairs (final method)

### 5.1 Idea

Every view is shot at several F-numbers. The F22 image of a view is nearly in focus everywhere, so it is a sharp reference with exactly the same texture as the F2.8 image. For each patch we find how much blur turns the F22 patch into the F2.8 patch. Texture cancels out, and the result is a direct measure of the extra blur at F2.8.

Script: `def_calibration/focus_from_aperture_pairs.py` (runs in `opencv_env`; numpy, OpenCV, tifffile and matplotlib only).

### 5.2 Steps per view and camera

1. Load the F2.8 image (target) and the F22 image (reference; F16 if a view has no F22) at half resolution, and the target's depth map.
2. Align the reference to the target (ECC affine registration). The camera moves a median of 4 px within a view.
3. For each 32 × 32 patch (64 × 64 at full resolution), blur the reference with Gaussians of σ = 0 to 12 px (step 0.25) and find the σ that best matches the target. Brightness differs between F-numbers, so each patch gets its own best gain, and means are removed. The best σ is refined with a parabola.
4. Keep patches that:
   - have valid, nearly flat depth (90% valid, depth variation under 5%);
   - have enough texture in the reference (high-pass standard deviation ≥ 1.5 gray levels);
   - are matched well (the best match explains at least 90% of the target patch's variance);
   - are not at the end of the σ range.

### 5.3 Model and fit

Thin-lens defocus, in half-resolution pixels, with x = 1/d − 1/s:
- blur of the target: σ_t = k × |x|, with k = fx_half × (f / N_t) / 4;
- blur of the reference: σ_r = k × (N_t / N_r) × |x|, plus a small constant (mostly diffraction at F22);
- measured extra blur: Δσ² = max(0, α × x² − c0), with α = k² × (1 − (N_t / N_r)²).

Per setting, the patches are binned by inverse depth (25 bins, median Δσ per bin), and s, α and c0 are found by a robust grid search (weighted absolute error). The uncertainty of s is the range of s whose cost is within 10% of the best.

**Self-check:** α can be predicted from fx, the real focal length (`f_true_mm` in the manifest) and the F-numbers. The ratio α_fit / α_pred came out consistent across settings (median 1.17, 5–95%: 0.76–1.58). The factor 1/4 that converts a blur disc to a Gaussian is approximate, so the ratio is not exactly 1.

### 5.4 What failed first: a free c0

With c0 free, the left and right cameras of Scene 9 at 32 mm gave 4.65 m and 2.65 m. The fit plots showed why:

- Near the focus distance, F2.8 is not measurably blurrier than F22 (the reference has its own small blur), so Δσ is flat at zero over a range of depths.
- In many settings the focus lies beyond most of the scene, so only the near side of the V is visible.
- With only one side visible, the position of the tip (s) and the width of the flat zone (c0) trade off against each other, and the fit can pick either.

**Fix:** c0 depends on the F-numbers and the pixel size, not on the scene, so it can be fixed. Free fits gave a median c0 of 0.5 half-pixel² (0.6 among the most precise fits). Diffraction alone at F22 gives about 0.15; demosaicing, JPEG compression and the reference's own small defocus add the rest. With c0 fixed at 0.5:

| | Agreement with method 1 on reliable settings (6) | Left vs Right, previously unreliable settings |
|---|---|---|
| c0 free | median 2.6%, max 7.3% | median 9.9%, max 55% |
| c0 = 0.5 | median 1.7%, max 7.3% | median 9.0%, max 35% |

### 5.5 Validation

1. **Against method 1 where that was reliable.** On 6 settings where method 1 (F2.8 + F5) was reliable for both cameras, method 2 agrees within a median of 1.7–2.6%. Over all 149 settings where method 1 (F2.8 + F5) was reliable, the agreement is a median of 4.6% (90% within 13%).
2. **On the settings where method 1 failed.** All 17 settings that were unreliable for at least one camera now give plausible values (1.6–3.9 m) with narrow uncertainty ranges.

## 6. Results

All 170 fits (85 settings × 2 cameras) succeeded.

| Check | Value |
|---|---|
| Uncertainty of each fit (upper / lower bound − 1) | median 2.4%, 90% under 9.1% |
| Slope vs optics (α_fit / α_pred) | median 1.17, 5–95%: 0.76–1.58 |
| Left vs Right difference | median 5.9%, 75% under 11.1%, 90% under 16%, max 35% |
| Agreement with method 1 (F2.8 + F5) where reliable | median 4.6%, 90% under 13% |

### 6.1 Focus distance per scene, focal length and camera

A dash means there is no data (Scene 1 has no 50–65 mm; Scene 9 at 65 mm is excluded from the dataset).

**Left camera** (metres)

| Scene | 28 mm | 32 mm | 36 mm | 40 mm | 45 mm | 50 mm | 55 mm | 60 mm | 65 mm | 70 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.29 | 2.38 | 2.72 | 2.56 | 3.31 | – | – | – | – | 3.60 |
| 2 | 1.72 | 1.66 | 1.99 | 2.22 | 2.20 | 1.86 | 2.13 | 2.06 | 1.98 | 2.16 |
| 3 | 2.82 | 2.34 | 2.42 | 3.03 | 2.45 | 2.69 | 2.55 | 2.05 | 2.55 | 2.08 |
| 4 | 2.20 | 1.76 | 1.79 | 1.84 | 1.67 | 1.81 | 2.01 | 1.92 | 1.98 | 1.96 |
| 5 | 2.56 | 2.67 | 2.08 | 2.08 | 2.20 | 1.45 | 2.03 | 1.75 | 1.97 | 1.79 |
| 6 | 1.64 | 1.72 | 1.85 | 1.92 | 1.59 | 1.72 | 1.63 | 1.60 | 1.50 | 1.85 |
| 7 | 2.13 | 2.34 | 1.93 | 2.02 | 2.09 | 2.21 | 2.22 | 2.74 | 2.58 | 2.53 |
| 8 | 1.57 | 1.75 | 2.29 | 1.56 | 1.85 | 2.13 | 2.14 | 2.31 | 2.23 | 2.09 |
| 9 | 2.70 | 3.33 | 2.78 | 2.80 | 2.82 | 2.88 | 2.80 | 2.69 | – | 2.82 |

**Right camera** (metres)

| Scene | 28 mm | 32 mm | 36 mm | 40 mm | 45 mm | 50 mm | 55 mm | 60 mm | 65 mm | 70 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.31 | 3.17 | 2.70 | 2.72 | 3.10 | – | – | – | – | 3.51 |
| 2 | 1.89 | 1.77 | 1.75 | 2.07 | 1.98 | 1.65 | 2.17 | 2.04 | 1.99 | 2.15 |
| 3 | 2.82 | 2.53 | 2.82 | 3.05 | 2.65 | 3.15 | 2.65 | 2.67 | 2.52 | 2.19 |
| 4 | 1.68 | 2.00 | 2.13 | 2.04 | 1.64 | 1.86 | 2.00 | 1.85 | 1.95 | 1.90 |
| 5 | 2.92 | 2.65 | 2.21 | 1.94 | 2.12 | 1.53 | 2.00 | 1.96 | 1.87 | 1.75 |
| 6 | 1.84 | 2.04 | 2.30 | 2.30 | 1.78 | 1.77 | 1.72 | 1.79 | 1.43 | 1.99 |
| 7 | 1.97 | 2.23 | 1.95 | 2.03 | 2.12 | 2.20 | 2.14 | 2.67 | 2.55 | 2.52 |
| 8 | 1.34 | 1.79 | 2.06 | 1.75 | 1.81 | 2.31 | 2.25 | 2.14 | 2.09 | 2.14 |
| 9 | 3.85 | 2.84 | 2.55 | 2.42 | 2.78 | 2.67 | 2.45 | 2.80 | – | 2.53 |

| Scene | Lowest | Highest | Median |
|---|---|---|---|
| 1 | 2.29 | 3.60 | 2.72 |
| 2 | 1.65 | 2.22 | 1.99 |
| 3 | 2.05 | 3.15 | 2.60 |
| 4 | 1.64 | 2.20 | 1.91 |
| 5 | 1.45 | 2.92 | 2.01 |
| 6 | 1.43 | 2.30 | 1.77 |
| 7 | 1.93 | 2.74 | 2.20 |
| 8 | 1.34 | 2.31 | 2.09 |
| 9 | 2.42 | 3.85 | 2.79 |

### 6.2 Findings

1. **The focus was not fixed across scenes.** Typical focus per scene ranges from 1.8 m (Scene 6) to 2.8 m (Scene 9). For the same focal length it differs a lot between scenes (at 50 mm: 1.45–2.88 m).
2. **Within a scene it changes moderately with focal length.** Scene 4 stays at 1.6–2.2 m, while Scene 1 goes from 2.3 to 3.6 m. Either the lens was refocused after each zoom change, or zooming shifted the focus (this lens is not parfocal).
3. **Left and Right are usually close but not the same** (median 5.9%). Each lens was focused by hand.
4. **Method 1 had over-estimated far focus distances.** For example, it put Scene 3 at 3–5 m; method 2 gives 2.0–3.2 m.

## 7. Limitations

- **One-sided data.** Where the focus lies beyond most of the scene, the estimate depends on the fixed c0. A different c0 (0.2–0.8) would shift those estimates by a few percent.
- **Left/Right differences of 10–35%** in a few settings (for example Scene 9 at 28 mm: 2.70 / 3.85 m, Scene 6 at 40 mm: 1.92 / 2.30 m). The data cannot tell whether the lenses were really focused differently or the estimates are less precise. Treat these settings as having about ±10% uncertainty.
- **One value per setting.** Method 2 assumes the focus is constant over the views of a session. Method 1's per-view fits agreed within 20% in 90% of settings, but a few sessions may have been refocused mid-way (for example Scene 2 at 45–65 mm). Method 2 has not yet been checked per view.
- **Depth labels at F2.8** come from stereo matching on blurred images and are slightly less accurate in blurred areas, which adds some noise to the patch depths.
- **The blur model** is a thin-lens approximation with a Gaussian blur kernel; the 1/4 disc-to-Gaussian factor is approximate. This does not affect s, only the absolute scale of α.

## 8. Files and how to reproduce

| File | Content |
|---|---|
| `D:\datasets\MODEST_focus\dfocus_aperture_pairs.csv` | **Final results** (method 2, c0 = 0.5): focus distance, uncertainty range, α ratio, c0, cost per scene, focal length and camera |
| `D:\datasets\MODEST_focus\plots_pairs\` | Fit plot per setting (method 2) |
| `D:\datasets\MODEST_focus\pairs\` | Cached per-view patch measurements (method 2) |
| `D:\datasets\MODEST_focus\dfocus_aperture_pairs_freec0.csv` | Method 2 with free c0, for 23 test settings |
| `D:\datasets\MODEST_focus\dfocus_per_setting.csv`, `dfocus_per_view.csv` | Method 1 (all F-numbers) |
| `D:\datasets\MODEST_focus\dfocus_fnumber_comparison.csv` | Method 1 with all F, F2.8 + F5 and F2.8 only |
| `D:\datasets\MODEST_focus\samples\`, `plots\` | Method 1 cached samples and fit plots |

Commands:

```
# method 2 (final), in opencv_env
python def_calibration/focus_from_aperture_pairs.py --c0 0.5
python def_calibration/focus_from_aperture_pairs.py --c0 0.5 --settings 9:32 9:36   # subset

# method 1, in torch_plot (needs scipy)
python def_calibration/focus_distance_survey.py
```

Both read the image list from `D:\datasets\MODEST_depth\manifest.csv`, so excluded images stay excluded.
