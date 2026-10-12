# Base blur (σ₀) of the MODEST cameras

Last updated: 2026-10-12

This report measures the **base blur σ₀**: the blur that is in an image even where the lens is perfectly focused. It also answers whether σ₀ can be treated as one constant per camera. The blur model is

σ(d)² = σ₀² + (k · f² / (N (s − f)) · |s − d| / d)²

where σ is the standard deviation of the blur (px), d the depth, s the focus distance, f the focal length, N the F-number, and k the blur constant (px per metre of blur-circle diameter on the sensor).

Everything is measured on the ChArUco calibration images, where the depth of every board edge is known from the board pose. Script: `def_calibration/base_blur_calib.py`. Figures and tables: `reports/base_blur/`. All outputs: `D:\datasets\MODEST_focus\base_blur`.

All σ values are in **raw camera pixels** (6.54 µm), and they are the blur **before the camera's JPEG sharpening** (section 3.2), unless stated otherwise.

## 1. Summary

- **σ₀ ≈ 0.62 px for both cameras at F16.** Weighted mean over focal lengths: camera L 0.623 px (8 focal lengths), camera R 0.629 px (4 focal lengths). It is measured at the image centre, in the vertical direction, on the sharpest 20% of shots.
- **Over focal length it is constant within the noise.** At F16 the values are 0.60–0.67 px (L) and 0.60–0.69 px (R), from 28 to 60 mm. Their spread matches their uncertainty (χ²/dof 1.3 and 0.6). At 65–70 mm it could not be measured, because no boards were placed close to the focus there.
- **Between F16 and F20 the data show no clear change.** F20: camera L 0.71 px, camera R 0.59 px. Diffraction theory predicts about +0.13 px. The two sessions differ in more than the F-number, so this comparison is weak.
- **The real blur in a shot is often larger than σ₀, because the camera moved during the exposure.** This motion blur:
  - changes from shot to shot;
  - is mostly horizontal;
  - grows with focal length.

  In the January session the median shot has about 0.4 px (28–40 mm) to 1.0–1.1 px (60–70 mm) of extra horizontal blur. The worst 10% of shots have more than 2 px. In May it is about half as much.
- **Blur grows towards the image corners.** Near focus, compared with the centre of the same shot, it is 1–4% higher at half the distance to the corner, 3–13% higher at 70%, and 4–16% higher in the corners.
- **The camera sharpens its JPEGs.** Ignoring this underestimates the blur near focus by about 20%.
- **k ≈ 36,000 /m** (median of the January fits; 31,000–40,000 per focal length). The theoretical value for a uniform blur disc is 38,200 /m. So σ ≈ 0.24 × the blur-circle diameter in pixels.
- **Not measured: F2.8, F5 and F9.** There are no calibration images at these apertures.

**Answer to the question:** yes, the camera's own σ₀ can be treated as **one value per camera and F-number**, about 0.62 px at F16. Both cameras have the same value, and it does not change with focal length within the measurement error. But the total blur that is "always there" in a real shot is larger and not constant. It also includes:

- camera motion, which changes per shot;
- the rise towards the corners;
- in rectified images, the resampling (section 6).

## 2. Data

The calibration images come from two sessions, and **each session used one F-number only**:

| Session | F-number | Board depth (5–95% of corners) | Images with a board pose | Exposure (median) |
|---|---|---|---|---|
| 31 Jan 2026 | F16 | 0.8–1.7 m | 1,214 of 1,266 | 1/8 s (L), 1/6 s (R) |
| 27 May – 1 Jun 2026 | F20 (F22 for camera L at 70 mm) | 2.3–4.8 m, mostly 3.3–4.5 m | 580 of 685 | 1/13–1/8 s (1/5 s at F22) |

The board is 12 × 16 squares of 43.5 mm with 30 mm ArUco markers. In May the board was found in only 2–6 of about 25 images at 28 and 32 mm; it is too small in the image at that distance. Coverage per group: `D:\datasets\MODEST_focus\calib_audit` (made by `def_calibration/calib_coverage_audit.py`).

What each session can measure:

- **January:** the focus lies inside the board depths, and the V-shape of blur against depth is clear. This session gives σ₀ at F16, s and k.
- **May:** the boards are 3–4.6 m away, where defocus is very small, so the curve is nearly flat. This session gives σ₀ at F20 roughly, but cannot fix s or k on its own.

## 3. Method

### 3.1 Blur of one edge (slanted-edge method)

For every inner edge between two squares of the board:

1. Find the board and its pose (`solvePnP` with the calibrated K and distortion of that focal length).
2. Take the pixels around the middle 70% of the edge.
   - On the white side, the window stops halfway to the ArUco marker, which sits 6.75 mm inside the white square.
   - On the black side, the window is up to 40 px wide.
3. Fit a blurred step to the raw pixel values: I = a + b · Φ((v − c₀ − c₁u) / σ). Here v is the distance from the edge, u the position along the edge, and c₀, c₁ absorb small pose errors.
   - The edge is slightly tilted against the pixel grid, so its pixels sample the step at many sub-pixel offsets.
   - No interpolation is used.
4. Use the green channel, converted from sRGB to linear light. Blur acts on light, not on gamma-coded values; fitting the coded values would overestimate σ by about 8%.
5. Take the depth of the edge midpoint from the board pose.

So σ contains everything the camera adds: defocus, diffraction, lens aberrations, pixel area, anti-alias filter and demosaicing. On synthetic edges the fit recovers σ within 0.01 px for σ from 0.3 to 4 px.

In total, 434,512 edges were measured on 1,786 images (about 50 min with 8 processes). 357,976 edges (82%) pass the quality checks:

- no clipped pixels;
- at least 2 px of sub-pixel phase coverage;
- fit error under 8%;
- a window of at least 2.5 σ on the white side.

The main loss is edges lying almost exactly along the pixel grid.

### 3.2 In-camera sharpening

Edges near focus show a bright overshoot and a dark undershoot next to the edge. Lens blur cannot do this, because it only averages light. It is the camera's JPEG sharpening. Two example sets of fitted edges are in `reports/base_blur/edge_profiles_near.png` and `edge_profiles_far.png`.

The sharpening is modelled as an unsharp mask on the gamma-coded values: S = E + α (E − E ∗ Gauss(r)). It was estimated with a grid search over (α, r):

- 1,586 sharp edges (a random 20%) from 82 sampled images, 2 per camera, focal length and session; 52 of them have sharp edges;
- each edge fitted at every grid point with α and r fixed;
- `reports/base_blur/sharpening_grid.png` shows the error at each grid point.

Results:

- Sharpening on coded values fits better than sharpening on linear values, and much better than no sharpening (squared error −7%).
- **Best overall: α = 0.75, r = 0.85 px.** All edges are then fitted with this model.
- α and r trade off along one valley of the error surface. All cameras and sessions lie in the same valley (α·r² ≈ 0.54–0.63), so the sharpening was probably the same throughout.
- Moving along the valley shifts σ near focus by about ±0.06 px. This is a systematic uncertainty of all σ₀ values here.
- The plain fit (no sharpening model, column `sigma_plain`) gives σ about 20% lower near focus; the median of sharp edges is 0.67 px instead of 0.82 px. For blurred edges (σ > 2 px) the two fits agree.

### 3.3 σ₀ per group, with camera motion per shot

A group is one camera, focal length, session and F-number. Fitting one V-curve with one floor σ₀ per group gave σ₀ rising from 0.73 px (28 mm) to 1.1 px (70 mm). This turned out to be camera motion, not lens blur:

- Horizontal blur near focus is much larger than vertical blur. At the image centre (camera L, January), horizontal is 1.1× vertical at 28 mm and 1.5–1.9× at 55–70 mm.
- The extra horizontal blur varies strongly between shots of the same group. Its spread is as large as its median. A fixed cause, such as the lens or the print quality of the board, would give the same blur in every shot.
- In the shots with the least motion, horizontal and vertical blur are almost equal.

So the final fit has two steps. The bootstrap over images repeats both steps (100 resamples, 90% intervals).

1. **V-curve shape (s and m = k·f²/(N(s−f))) from all shots of a group together,** with one shared floor. Motion does not depend on depth, so it raises the floor but hardly changes s and m.
   - When the focus is not inside the board depths, or the slope is not determined (all May groups, and January R 28/32/70 mm), m is instead computed from k.
   - That k comes from the well-determined January groups of the same camera, interpolated over focal length. k belongs to the lens, so it carries over between sessions.
2. **Given the V-curve, a floor per shot and per direction** (horizontal hx, vertical hy), from the central edges (less than half the distance to the corner). The floor is the shot's base blur plus its motion.
   - **σ₀ = the 20th percentile of the floors over shots:** the blur of the least-moved shots.
   - The motion of a shot is √(floor² − σ₀²).

The method was checked on fake edges with known σ₀, s, k and per-shot motion. It recovered σ₀ within about 0.01 px.

The values are reported as **measured** only when the 90% interval is at most 0.2 px wide and its lower end is at least 0.3 px; the pixel area alone gives 0.29 px, so no real camera is sharper. Groups with a steep V-curve and no boards near focus fail this test: L 65 and 70 mm and R 40–55, 65 and 70 mm in January, plus 11 of the 16 May groups. Their floor is simply not sampled. Example: L 70 mm, where all boards were at 1.3–2 m and the focus was at 1.16 m (`vcurves_L.png`).

## 4. Results

### 4.1 σ₀ per focal length and F-number

Vertical blur (least affected by motion), at the image centre, on the sharpest 20% of shots, with 90% intervals:

| Focal length | L, F16 | R, F16 | L, F20 | R, F20 |
|---|---|---|---|---|
| 28 mm | 0.62 [0.58, 0.64] | 0.62 [0.54, 0.63] | – | – |
| 32 mm | 0.60 [0.53, 0.63] | 0.60 [0.44, 0.62] | – | – |
| 36 mm | 0.61 [0.53, 0.63] | 0.64 [0.56, 0.69] | – | – |
| 40 mm | 0.60 [0.56, 0.62] | not measured | 0.70 [0.63, 0.72] | not measured |
| 45 mm | 0.66 [0.60, 0.74] | not measured | 0.73 [0.66, 0.74] | not measured |
| 50 mm | 0.64 [0.51, 0.67] | not measured | not measured | not measured |
| 55 mm | 0.67 [0.64, 0.71] | not measured | 0.69 [0.59, 0.71] | 0.60 [0.57, 0.67] |
| 60 mm | 0.63 [0.54, 0.69] | 0.69 [0.65, 0.83] | not measured | 0.55 [0.48, 0.66] |
| 65–70 mm | not measured | not measured | not measured | not measured |
| **Weighted mean** | **0.623** | **0.629** | **0.712** | **0.592** |

"–" = no images with a detected board. "not measured" = the floor is not pinned down (section 3.3). All groups: `reports/base_blur/groups_fit.csv`; means and trends: `sigma0_summary.csv`; figure: `sigma0_vs_fl.png`.

- **Focal length:** at F16 the values do not change beyond their uncertainty. The fitted trend is +0.02 px per 10 mm, which adds up to about 0.06 px over 28–60 mm. This also shows that the printed board is sharp enough. Its own blur would grow with focal length (it scales as f/d), and the data allow at most about 0.02 mm of print blur.
- **Camera:** L and R agree within 0.01 px at F16.
- **F-number:** F20 is +0.09 px for camera L and −0.04 px for camera R. The Gaussian approximation of diffraction (σ ≈ 0.42 λN/pixel = 0.57 px at F16, 0.71 px at F20) predicts +0.13 px. The data do not show this clearly. Two reasons make the comparison weak:
  - every F20 image is from another session, at 3–4.5 m, with a nearly flat V-curve;
  - its σ₀ depends on k carried over from January.

  A safe statement is: **σ₀ at F20 lies within about ±0.1 px of σ₀ at F16.**
- **Horizontal blur** is higher, even for the sharpest shots:
  - L F16: 0.66–0.69 px at 28–40 mm, rising to 0.75–0.83 px at 45–60 mm;
  - R F16: 0.73–0.78 px at 28–32 mm;
  - F20: 0.73–0.77 px (L), 0.67–0.71 px (R).

  The rise with focal length is residual motion: even the least-moved 20% of January shots still move. In May, with less motion, horizontal is close to vertical for camera L (+0.03 px).

### 4.2 Camera motion during the exposure

Extra horizontal blur per shot (√(floor² − σ₀²)), median over shots:

| | 28–40 mm | 45–55 mm | 60–70 mm |
|---|---|---|---|
| January, L | 0.38 px | 0.76 px | 1.04 px |
| January, R | 0.82 px | 1.12 px | 1.13 px |
| May, L | 0.26 px | 0.28 px | 0.68 px |
| May, R | 0.18 px | 0.50 px | 0.51 px |

- In January, 10% of shots have more than 2.1 px (L) or 2.9 px (R) of extra horizontal blur.
- Vertical motion is smaller: median 0.3–0.4 px in January and 0.2–0.6 px in May.
- Growth with focal length means the camera turned slightly during the exposure (about 20 arcseconds). The rig most likely sways sideways when the shutter fires.
- January had longer exposures (median 1/8–1/6 s) than most May groups (1/13 s), and more motion.

Per shot: `D:\datasets\MODEST_focus\base_blur\per_image_floor.csv`.

### 4.3 Position in the image

This compares each edge near focus with the centre floor of the same shot, in the same direction. So motion and defocus are removed. Figure: `sigma0_vs_position.png`.

| Distance from centre (1 = corner) | 0.5 | 0.7 | 0.9 |
|---|---|---|---|
| F16 (January), horizontal / vertical | +4% / +3% | +9–13% / +8% | +8–13% / +15–16% |
| F20 (May), horizontal / vertical | +1% / +1% | +5% / +2–3% | +8–9% / +4–7% |

- The rise is larger in January.
- There, the V-curve is steep. So a slightly curved surface of best focus adds noticeable defocus away from the centre, and part of this rise is likely field curvature rather than pure lens blur.
- For horizontal blur, edges whose blur runs along the radius are 3–4% blurrier than edges whose blur runs across it. For vertical blur there is no such difference.

### 4.4 The blur constant k (by-product)

k from the January groups whose V-curve has its bottom inside the board depths:

| Focal length (mm) | 28 | 32 | 36 | 40 | 45 | 50 | 55 | 60 | 65 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|
| L (×1000 /m) | 30.7 | 31.7 | 32.2 | 36.9 | 34.3 | 35.6 | 36.3 | 37.8 | 34.2 | 38.3 |
| R (×1000 /m) | – | – | 32.3 | 40.0 | 31.3 | 39.6 | 39.1 | 36.0 | 38.4 | – |

- The 90% intervals are typically ±10–20%.
- **Median: 36,000 /m.** The theory for a uniform blur disc gives 1/(4 × 6.54 µm) = 38,200 /m.
- In the units of the benchmark's blur (blur-circle diameter in pixels, fx·f/N·|1/d − 1/s|), this means **σ ≈ 0.24 × diameter**.
- Camera L shows lower values at 28–36 mm (31–32k) than at 40–70 mm (34–38k). The difference is about the size of the intervals.

## 5. Is σ₀ constant for a camera?

| Question | Answer |
|---|---|
| Same for camera L and camera R? | Yes (0.623 vs 0.629 px at F16) |
| Same over focal length? | Yes within the noise, 28–60 mm (±0.05 px); not measured at 65–70 mm |
| Same over F-number? | F16 vs F20: within about ±0.1 px. F2.8–F9: unknown. Diffraction is smaller there, but lens aberrations are larger at wide apertures, so σ₀ at F2.8 could go either way |
| Same across the image? | Within ~5% up to half-way to the corner; up to 10–16% more in the corners |
| Same in every shot? | **No.** Camera motion adds a shot-dependent, mostly horizontal blur, often larger than σ₀ itself at long focal lengths |

A practical model for the scene images:

σ² = σ₀² + σ_motion² + (k · f² / (N (s − f)) · |s − d| / d)², with σ₀ ≈ 0.62 px and k ≈ 36,000 /m,

where σ_motion is unknown per shot: typically 0.3–1 px horizontally, larger at long focal lengths and long exposures. At wide apertures the defocus term dominates almost everywhere. At F16–F22 it does not: σ₀ and motion are a large part of the blur over much of the scene.

## 6. Using σ₀ on the rectified scene images

The images in `MODEST_depth` are rectified: magnified by `rect_zoom` (1.02–1.19, median 1.075) and resampled with bilinear interpolation. Both add blur. A rough conversion, not measured here:

σ_rect ≈ rect_zoom × √(σ₀² + 0.17)

Here 0.17 px² is the average variance that bilinear resampling adds; it varies across the image between 0 and 0.25. With σ₀ = 0.62 px and zoom 1.075, this gives **σ_rect ≈ 0.80 rectified px**.

To simulate how the camera's JPEG looks, apply the blur in linear light, convert to sRGB, then apply the unsharp mask (α = 0.75, r = 0.85 px) to the coded values.

## 7. Limitations

- **Two F-numbers only.** F16 and F20 (F22 for one group) come from different sessions, so F-number and session cannot be separated. F2.8, F5 and F9 are not covered.
- **Gaussian blur model.** The real blur has a sharp core and a faint wide tail; it shows as a small shoulder 2–8 px from the edge. σ is the best-fitting Gaussian.
- **Sharpening model.** It is approximate, and the α–r trade-off leaves ±0.06 px of systematic uncertainty in σ near focus.
- **σ₀ = the sharpest 20% of shots.** If even those shots moved a little, the true motion-free σ₀ is slightly lower.
- **Long focal lengths.** At 65–70 mm, and in several other steep groups, no boards were placed near focus, so σ₀ is not measured there.
- **Green channel only.** Colour fringes (lateral chromatic aberration) in red and blue were not studied.

## 8. Files and commands

| | |
|---|---|
| Coverage audit | `def_calibration/calib_coverage_audit.py` → `D:\datasets\MODEST_focus\calib_audit` |
| Measurement and fits | `def_calibration/base_blur_calib.py` (torch_plot env), steps `sharpen` (~15 min), `measure` (~50 min), `fit` (~15 min) |
| Edge measurements | `edges.csv`: one row per edge (σ, plain σ, depth, position, direction, quality fields) |
| Sharpening | `sharpening.json`, `sharpening_grid_best.csv`, `sharpening_grid.png`, `sharpen_edges.csv` |
| Fits | `groups_fit.csv` (per group), `sigma0_summary.csv`, `per_image_floor.csv` (floor and motion per shot), `position.txt` |
| Figures | `vcurves_L.png`, `vcurves_R.png` (blur vs depth per group), `sigma0_vs_fl.png`, `sigma0_vs_position.png` |

```
python def_calibration/calib_coverage_audit.py            # opencv_env
python def_calibration/base_blur_calib.py sharpen         # torch_plot env
python def_calibration/base_blur_calib.py measure
python def_calibration/base_blur_calib.py fit
python def_calibration/base_blur_calib.py measure --debug L 40 IMG_7785.JPG   # edge plots for one image
```
