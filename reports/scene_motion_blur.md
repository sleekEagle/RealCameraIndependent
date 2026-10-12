# Camera motion blur in the MODEST scene photos

Last updated: 2026-10-12

Depth-from-defocus methods read depth from the size of the blur. They usually assume that all blur comes from defocus, plus a small fixed base blur. This report asks whether that is safe for MODEST. In other words: did the camera move during the exposures, and how much blur did that add?

The calibration images already showed camera motion (`base_blur.md`, section 4.2). This report measures it directly in the scene photos.

Scripts:
- `def_calibration/scene_motion_blur.py` (measurement);
- `def_calibration/scene_motion_report.py` (tables and figures).

Tables and figures are in `reports/scene_motion/`. Raw results: `D:\datasets\MODEST_focus\scene_motion\repeat_pairs.csv`.

## 1. Summary

- **Exposure times grow strongly with the F-number.** The ISO was fixed within each scene, so a smaller aperture means a longer exposure. Median exposures are:

  | F-number | F2.8 | F5 | F9 | F16 | F22 |
  |---|---|---|---|---|---|
  | Median exposure | 1/160 s | 1/50 s | 1/15 s | 1/5 s | 1/2 s |

  The longest exposure is 3.2 s.
- **The camera moved during the long exposures.** Two shots of the same view, taken about 2 s apart, differ in sharpness at F9–F22, but not at F2.8.
  - Left and right cameras get sharper or blurrier together in the same shots (rank correlation 0.7 at F16 and F22, 0.1 at F2.8). So the whole stereo rig moved; this is not measurement noise.
- **How much (lower bound):** at F16 and F22, the blur difference between the two repeat shots is:
  - median 0.3–0.4 px horizontally and 0.5–0.6 px vertically;
  - for 10% of shots, more than 0.9–1.0 px horizontally and 1.3 px vertically.

  A difference between two shots only gives a lower bound: the more-moved shot has at least this much motion blur.
- **Motion grows with exposure time and focal length.** It is negligible up to 1/60 s, appears from about 1/15 s, and is largest at 1/4–1 s. Exposures over 1 s show less of it, which fits a vibration that dies down during the exposure.
- **It differs a lot between scenes.** In Scene 1 and Scene 9 the median difference is about 0.6 px horizontally and 0.9–1.0 px vertically. In Scene 6, with short exposures, it is 0.15 px.
- **Answer:** no motion blur is a **safe assumption at F2.8 and F5**. It is **not safe at F9–F22**, where motion blur is about as large as σ₀ (0.62 px) and often as large as the defocus blur itself (section 4).

## 2. Exposure settings of the scene photos

The values come from the EXIF data of the original photos in `D:\datasets\MODEST`. The rectified images in `MODEST_depth` have no EXIF.

- Each scene used one ISO: 1,000–1,600 in most scenes, 200–500 in Scene 5.
- The exposure was set by the light. Changing the F-number from F2.8 to F22 is 6 stops, so the exposure time grows about 64×.
- Scenes 3, 5 and 6 were bright, with short exposures (F16 and F22: 1/60–1/5 s).
- Scenes 1, 7, 8 and 9 were dark (F16 and F22: 0.3–0.8 s medians, up to 3.2 s).

| Scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Median exposure at F16 + F22 (s) | 0.40 | 0.30 | 0.04 | 0.25 | 0.17 | 0.017 | 0.80 | 0.30 | 0.30 |

For comparison, the calibration images used 1/13–1/6 s and already showed 0.4–1 px of horizontal motion (`base_blur.md`).

## 3. Method: repeat shots

Almost every scene view was photographed twice in a row, 1–3.5 s apart, with nothing changed. `MODEST_dedup` kept one shot of each such pair and dropped the other; `pairs_manifest.csv` lists both. In total there are 7,536 such image pairs (3,768 repeats × 2 cameras), at every focal length and F-number.

The two shots have the same content, the same focus and the same exposure, so their defocus blur is identical. A difference in blur between them must come from motion during the exposure, or from a change in the scene.

**The measurement:**

1. Take the green channel of each shot in linear light, and the central 2,048 × 2,048 px.
2. Compute the power spectrum, averaged over 64 tiles, and remove the noise floor.
3. Compare the two spectra. The ratio of the spectra cancels the scene content, and also the JPEG sharpening, which is the same in both shots. What remains is fitted as

   ln(P₂ / P₁) = c − Δσx² · ωx² − Δσy² · ωy²

   - Δσx² is the difference in horizontal blur variance between the two shots; Δσy² is the vertical one.
   - Only frequencies with good signal are used: 0.1 < |ω| < 1.2 rad/px, signal at least 5× the noise.
4. Report √|Δσ²| in pixels. Because both shots may have moved, this is a **lower bound** on the motion of the more-moved shot.

**Checks:**
- **Known blur added to F2.8 shots** (30 pairs):

  | Added | Expected Δσ² (px²) | Measured median (px²) |
  |---|---|---|
  | 1 px horizontal | 1.00 | 0.99 |
  | 2 px horizontal | 4.00 | 4.04 |
  | 1 px vertical | 1.00 | 0.98 |
  | 0.5 px horizontal | 0.25 | 0.20 |

  Blur in one direction does not leak into the other direction.
- **F2.8 repeat pairs as the baseline.** These have exposures of 1/160 s and should show no motion. Their medians are 0.13 px horizontally and 0.14 px vertically, which is the noise level of the method. 95% of them are below 0.30 px horizontally and 0.39 px vertically.
- **Left vs right camera.** Both cameras fire at the same moment. If the measured differences were noise, they would not agree between the two cameras. At F16 and F22 they agree strongly (rank correlation 0.70 and 0.69); at F2.8 they do not (0.10).

## 4. Results

### 4.1 By F-number

The blur difference between repeat shots, √|Δσ²|, in px. Figure: `fig_by_fnumber.png`.

| F-number | Median exposure | Horizontal: median | Horizontal: 90th pct | Vertical: median | Vertical: 90th pct | Pairs with horizontal difference > 0.5 px | Pairs with vertical difference > 0.5 px |
|---|---|---|---|---|---|---|---|
| F2.8 | 1/160 s | 0.13 | 0.25 | 0.14 | 0.29 | 1% | 3% |
| F5 | 1/50 s | 0.17 | 0.50 | 0.19 | 0.77 | 10% | 20% |
| F9 | 1/15 s | 0.28 | 0.86 | 0.44 | 1.15 | 30% | 47% |
| F16 | 1/5 s | 0.38 | 1.03 | 0.61 | 1.31 | 37% | 58% |
| F22 | 1/2 s | 0.29 | 0.90 | 0.54 | 1.32 | 32% | 53% |

At F16, 11% of pairs differ by more than 1 px horizontally.

### 4.2 By exposure time

All F-numbers together. Figure: `fig_vs_exposure.png`.

| Exposure | <1/250 s | 1/250–1/60 | 1/60–1/15 | 1/15–1/4 | 1/4–1/2 | 1/2–1 | >1 s |
|---|---|---|---|---|---|---|---|
| Horizontal median (px) | 0.12 | 0.13 | 0.21 | 0.44 | 0.48 | 0.45 | 0.32 |
| Vertical median (px) | 0.13 | 0.15 | 0.32 | 0.71 | 0.73 | 0.72 | 0.56 |

- Motion starts at about 1/60 s and is largest from 1/15 s to 1 s.
- It is smaller beyond 1 s, although that bin has only 82 pairs. This fits a short vibration after the shutter fires: in a very long exposure, the moving part is a smaller share of the exposure.

### 4.3 By focal length (F16 and F22)

| Focal length (mm) | 28 | 32 | 36 | 40 | 45 | 50 | 55 | 60 | 65 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|
| Horizontal median (px) | 0.19 | 0.23 | 0.26 | 0.32 | 0.33 | 0.41 | 0.42 | 0.46 | 0.42 | 0.41 |
| Vertical median (px) | 0.43 | 0.40 | 0.47 | 0.62 | 0.54 | 0.69 | 0.63 | 0.67 | 0.68 | 0.63 |

Motion blur grows with focal length up to about 50 mm and then levels off. A turning of the camera gives blur proportional to focal length, so this points to rotation.

### 4.4 Direction

- In the scene photos the vertical difference is larger than the horizontal one (medians 0.61 vs 0.38 px at F16).
- In the calibration images it was the other way round: mostly horizontal.
- The rig or tripod setup may have differed between the scene and calibration sessions. So **neither direction can be assumed free of motion in the scene photos.**
- The two camera bodies (A and B) are affected equally.

### 4.5 By scene (F16 and F22)

| Scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Horizontal median (px) | 0.65 | 0.36 | 0.17 | 0.47 | 0.21 | 0.15 | 0.31 | 0.45 | 0.59 |
| Vertical median (px) | 0.88 | 0.44 | 0.31 | 0.51 | 0.71 | 0.16 | 0.72 | 0.67 | 0.96 |

Scenes 3 and 6, with short exposures, are nearly free of motion. Scenes 1 and 9 are affected most.

## 5. What this means for depth from defocus

Compare the motion blur with the defocus blur. The example is a 50 mm lens focused at 2.2 m (a typical scene focus distance), with k ≈ 36,000 /m (`base_blur.md`). The defocus blur is then σ_defocus ≈ m · |s − d| / d.

| F-number | m (px) | Depth range where defocus blur < 0.6 px | Typical motion blur |
|---|---|---|---|
| F2.8 | 15 | 2.1–2.3 m | ≈ 0 |
| F5 | 8.4 | 2.05–2.37 m | small (median 0.2 px, 10% of shots > 0.5 px) |
| F9 | 4.7 | 1.95–2.5 m | 0.3–0.4 px median, often > 0.8 px |
| F16 | 2.6 | 1.8–2.9 m | 0.4–0.6 px median, often > 1 px |
| F22 | 1.9 | 1.7–3.2 m | 0.3–0.5 px median, often > 1 px |

The "typical motion blur" column gives the measured differences between shots (lower bounds).

- **F2.8 and F5:** defocus is far larger than motion, except very close to the focal plane, where depth from blur is unreliable anyway. **Ignoring motion is safe.**
- **F9:** motion blur matters within about ±15% of the focus distance. Away from focus, defocus dominates.
- **F16 and F22:** over a large part of the scene (here 1.7–3.2 m), the defocus blur is no larger than the motion blur. A method that reads all blur as defocus would place sharp-looking regions too close to the focus distance. That error changes from shot to shot and is not the same in the horizontal and vertical directions. **Ignoring motion is not safe.**

**Practical options:**
- Treat F16 and F22 blur values below about 1–1.5 px as uncertain.
- Or model motion explicitly: one blur variance per shot and direction, added to the defocus blur. Section 3 shows it can be estimated.
- For training data, the repeat shots can flag strongly moved views (a large difference within a pair).
- The benchmark's blur-binned results (`benchmark_results.md`) compute blur from defocus only. At F16 and F22 their 0–2 px bins also contain motion blur.

## 6. Limitations

- **Lower bounds only.** Repeat shots give differences in motion between two shots, not the motion of each shot. If both shots moved by a similar amount, the difference underestimates the motion.
- **Gaussian approximation.** Real motion paths are lines or curves, not Gaussian blur. The fit uses the low frequencies, where the variance is what matters.
- **Scene changes** between the two shots (people, flicker, a moved object) would look like a blur difference. The strong left-right agreement suggests this is rare.
- **Only the image centre** (2,048 × 2,048 px) is measured. A rotation about the optical axis would blur the corners more.

## 7. Files and commands

| | |
|---|---|
| Measurement | `def_calibration/scene_motion_blur.py`, `measure` (~35 min, 8 processes) and `validate` |
| Tables and figures | `def_calibration/scene_motion_report.py` → `reports/scene_motion/` (`table_by_fnumber.csv`, `table_by_exposure.csv`, `table_by_focal_F16_F22.csv`, `table_by_scene_F16_F22.csv`, `table_left_right_corr.csv`, `fig_by_fnumber.png`, `fig_vs_exposure.png`) |
| Raw results | `D:\datasets\MODEST_focus\scene_motion\repeat_pairs.csv`: one row per repeat pair and camera, with exposure, ISO, time gap, Δσx², Δσy² and fit quality. `validate.csv` holds the known-blur check. |

```
python def_calibration/scene_motion_blur.py validate     # torch_plot env
python def_calibration/scene_motion_blur.py measure
python def_calibration/scene_motion_report.py
```
