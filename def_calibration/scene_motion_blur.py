"""
def_calibration/scene_motion_blur.py -- camera motion blur in the MODEST scene photos, measured from repeat shots.

Almost every scene view was shot twice in a row, 1-3.5 s apart, with nothing changed (MODEST_dedup kept
one and dropped the other; pairs_manifest.csv lists both). The two shots have the same content, focus,
F-number and exposure, so their defocus blur is identical. Any difference in blur between them comes from
camera motion during the exposure (or the scene changing).

For two shots I1 = I * h1 and I2 = I * h2 the ratio of their power spectra cancels the scene:
    ln(P2 / P1)(w) = c - dvx * wx^2 - dvy * wy^2        (Gaussian / low-frequency approximation)
with dvx = sx2^2 - sx1^2 and dvy = sy2^2 - sy1^2 the differences in horizontal and vertical blur variance
(px^2), and c a brightness term. The same JPEG sharpening in both shots cancels too. Per shot: green
channel in linear light, central 2048 x 2048 px, power spectrum averaged over 64 Hann-windowed 256 x 256
tiles, noise floor (mean power at |w| > 2.6 rad/px) subtracted; the fit uses frequencies 0.1 < |w| < 1.2
rad/px where both shots have signal at least 5x their noise.

sqrt(|dvx|) is the difference in horizontal blur between the two shots in quadrature, so a LOWER BOUND
on the horizontal motion blur of the more-moved shot (both shots may have moved).

    python def_calibration/scene_motion_blur.py measure            (torch_plot env)
    python def_calibration/scene_motion_blur.py validate           (known blur added to F2.8 shots)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

SRC = Path(r"D:\datasets\MODEST")
MANIFEST = Path(r"D:\datasets\MODEST_dedup\pairs_manifest.csv")
OUT = Path(r"D:\datasets\MODEST_focus\scene_motion")
CROP, TILE = 2048, 256
W_LO, W_HI, W_NOISE, SNR = 0.1, 1.2, 2.6, 5.0
LUT = np.arange(256) / 255.0
LUT = np.where(LUT <= 0.04045, LUT / 12.92, ((LUT + 0.055) / 1.055) ** 2.4).astype(np.float32)

_wy, _wx = np.meshgrid(np.fft.fftfreq(TILE) * 2 * np.pi, np.fft.fftfreq(TILE) * 2 * np.pi, indexing="ij")
_wr = np.hypot(_wx, _wy)
_HANN = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)


def camera_dirs():
    """(scene, 'Left'/'Right') -> camera folder (contains fl_XXmm/inference/F*), from the MODEST layout."""
    out = {}
    for d in SRC.glob("Scene*/**/EOS6D_*_*"):
        if d.is_dir() and any(d.glob("fl_*mm/inference")):
            scene = d.relative_to(SRC).parts[0]
            out[(scene, d.name.split("_")[-1])] = d
    return out


def load_linear(path, crop=CROP):
    g = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, 1]
    h, w = g.shape
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return LUT[g[y0:y0 + crop, x0:x0 + crop]]


def spectrum(img):
    """Mean power spectrum of Hann-windowed tiles, and its noise floor."""
    n = img.shape[0] // TILE
    t = img[:n * TILE, :n * TILE].reshape(n, TILE, n, TILE).transpose(0, 2, 1, 3).reshape(-1, TILE, TILE)
    t = (t - t.mean(axis=(1, 2), keepdims=True)) * _HANN
    P = np.mean(np.abs(np.fft.fft2(t)) ** 2, axis=0)
    return P, float(P[_wr > W_NOISE].mean())


def blur_difference(img1, img2):
    """Fit ln(P2/P1) = c - dvx wx^2 - dvy wy^2. Returns dvx, dvy (px^2), c, number of bins, rms residual."""
    (P1, n1), (P2, n2) = spectrum(img1), spectrum(img2)
    S1, S2 = P1 - n1, P2 - n2
    use = (_wr > W_LO) & (_wr < W_HI) & (S1 > SNR * n1) & (S2 > SNR * n2)
    if use.sum() < 50:
        return np.nan, np.nan, np.nan, int(use.sum()), np.nan
    A = np.stack([np.ones(use.sum()), -_wx[use] ** 2, -_wy[use] ** 2], 1)
    y = np.log(S2[use] / S1[use])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    return float(coef[1]), float(coef[2]), float(coef[0]), int(use.sum()), float(np.sqrt(np.mean(res ** 2)))


def exif(path):
    e = Image.open(path)._getexif() or {}
    t = e.get(33434)
    return (float(t) if t else np.nan), e.get(34855), e.get(36867)


def pairs():
    """One row per repeat: the kept shot and the dropped repeat, for each camera."""
    m = pd.read_csv(MANIFEST)
    dirs = camera_dirs()
    rows = []
    for (scene_path, fl, folder), g in m.groupby(["scene", "focal_length", "output_folder"], sort=False):
        scene, last = scene_path.split(chr(92))[0], None
        for r in g.itertuples():
            if r.kept == 1:
                last = r
                continue
            if last is None or pd.isna(r.dup_score_vs_last_kept):
                continue
            for side, a, b in (("Left", last.left_source, r.left_source), ("Right", last.right_source, r.right_source)):
                base = dirs[(scene, side)] / fl
                rows.append({"scene": scene, "fl_mm": int(fl[3:-2]), "N": float(folder.split("F")[-1]),
                             "side": side, "camera": dirs[(scene, side)].name.split("_")[1],
                             "kept": str(base / a), "repeat": str(base / b), "dup_score": r.dup_score_vs_last_kept})
    return pd.DataFrame(rows)


def measure_pair(row):
    try:
        i1, i2 = load_linear(row["kept"]), load_linear(row["repeat"])
    except Exception as ex:  # missing or unreadable file
        return {**row, "error": type(ex).__name__}
    dvx, dvy, c, nb, rms = blur_difference(i1, i2)
    t1, iso, d1 = exif(row["kept"])
    t2, _, d2 = exif(row["repeat"])
    gap = (pd.Timestamp(d2.replace(":", "-", 2)) - pd.Timestamp(d1.replace(":", "-", 2))).total_seconds() if d1 and d2 else np.nan
    return {**row, "exp_s": t1, "exp_s_repeat": t2, "iso": iso, "gap_s": gap, "dvx": dvx, "dvy": dvy, "c": c,
            "bins": nb, "rms": rms}


def validate_pair(args):
    """Known horizontal / vertical Gaussian blur added (in linear light) to the repeat of an F2.8 pair."""
    row, sig, axis = args
    i1, i2 = load_linear(row["kept"]), load_linear(row["repeat"])
    k = (0, 0)
    if sig > 0:
        i2 = cv2.GaussianBlur(i2, (0, 0), sigmaX=sig if axis == "x" else 1e-6, sigmaY=sig if axis == "y" else 1e-6)
    dvx, dvy, c, nb, rms = blur_difference(i1, i2)
    return {"kept": row["kept"], "sigma_added": sig, "axis": axis, "dvx": dvx, "dvy": dvy, "bins": nb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "validate"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pr = pairs()
    print(f"{len(pr)} repeat pairs (both cameras)", flush=True)
    t0 = time.time()
    if a.step == "validate":
        base = pr[pr.N == 2.8].sample(30, random_state=0).to_dict("records")
        jobs = [(r, s, ax) for r in base for s, ax in [(0, "x"), (0.5, "x"), (1.0, "x"), (2.0, "x"), (1.0, "y")]]
        with Pool(a.workers) as pool:
            df = pd.DataFrame(pool.map(validate_pair, jobs))
        df.to_csv(a.out / "validate.csv", index=False)
        t = df.groupby(["axis", "sigma_added"])[["dvx", "dvy"]].median().round(3)
        t["expected"] = t.index.get_level_values(1) ** 2
        print(t.to_string())
        return
    if a.limit:
        pr = pr.sample(a.limit, random_state=0)
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(measure_pair, pr.to_dict("records"), chunksize=4)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(pr)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "repeat_pairs.csv", index=False)
    print(f"done: {len(df)} pairs, {time.time() - t0:.0f}s -> {a.out / 'repeat_pairs.csv'}")


if __name__ == "__main__":
    main()
