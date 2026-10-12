"""
def_calibration/base_blur_calib.py -- base blur sigma0 (the blur that is there even in perfect focus)
and the defocus V-curve, measured on the ChArUco calibration images.

Step "measure": for every calibration image, find the board and its pose (solvePnP with that focal
length's calibrated K, D). For every inner square edge (between two neighbouring chessboard corners)
take the pixels around the middle of the edge and fit a blurred step:

    I = a + b * Phi((v - c0 - c1 * u) / sigma)

v = signed distance of the pixel centre from the edge (px), u = position along the edge, Phi = normal
CDF; c0, c1 absorb small pose errors. This is the slanted-edge method: the edge is tilted against the
pixel grid, so its pixels sample the step at many sub-pixel offsets. Raw pixel values are used (no
interpolation), so sigma holds everything the camera adds: defocus, diffraction, aberrations, pixel
area, anti-alias filter, demosaicing, JPEG processing. Values: green channel, sRGB -> linear light
(blur acts on light, not on gamma-encoded values). Depth = Z of the edge midpoint from the board pose.
Units: raw camera pixels (6.54 um). The scene images in MODEST_depth are rectified (zoom 1.02-1.19,
bilinear resampling), so their blur in rectified pixels is larger.

The window across the edge stops halfway to the ArUco marker on the white side (the marker sits
(square - marker)/2 inside the white square), so only blur up to ~1/5 of that margin is measurable.

The camera sharpens its JPEGs (halo next to sharp edges). Step "sharpen" estimates that sharpening as an
unsharp mask on the gamma-coded values, S = E + alpha (E - E * Gauss(r)), by a grid search over (alpha, r)
on sharp edges; "measure" then fits each edge with that model, so sigma is the blur BEFORE sharpening
(sigma_plain = a plain step fit in linear light, for comparison). alpha and r trade off against each
other (one valley in the error surface); along the valley sigma near focus moves by about +-0.06 px.

Step "fit": per group (camera, focal length, session date, F-number), the V-curve
    sigma(d)^2 = (m (s - d) / d)^2 + floor^2
with the floor (blur at focus) per image and direction, because camera motion during the exposure
differs from shot to shot (mostly horizontal). sigma0 = blur at focus of the sharpest 20% of shots.

    python def_calibration/base_blur_calib.py sharpen        (torch_plot env; needs calib_audit/images.csv)
    python def_calibration/base_blur_calib.py measure
    python def_calibration/base_blur_calib.py measure --debug L 70 IMG_7085.JPG
    python def_calibration/base_blur_calib.py fit
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

CALIB_ROOT = Path(r"D:\datasets\MODEST\Global_calibration_set\MODEST_ChArUco"
                  r"\Global_calibration_set\ChArUco_pattern")
CAMERA_DIRS = {"L": "EOS_6D_A", "R": "EOS_6D_B"}
AUDIT_CSV = Path(r"D:\datasets\MODEST_focus\calib_audit\images.csv")
OUT = Path(r"D:\datasets\MODEST_focus\base_blur")
PIXEL_M = 6.54e-6                       # EOS 6D pixel pitch
KAPPA = 0.42 * 0.55e-6 / PIXEL_M        # diffraction sigma per F-number unit, px (Gaussian fit to Airy, 550 nm)

_INFO = json.loads((CALIB_ROOT / "pattern_info_charuco.json").read_text())["charuco"]
SX, SY = _INFO["squares_x"], _INFO["squares_y"]
SQ, MK = _INFO["square_length_m"], _INFO["marker_length_m"]
MARGIN_M = (SQ - MK) / 2                # white border between a square edge and its marker
LUT = np.arange(256) / 255.0
LUT = np.where(LUT <= 0.04045, LUT / 12.92, ((LUT + 0.055) / 1.055) ** 2.4).astype(np.float64)
SQ2PI = np.sqrt(2 * np.pi)

_W = {}  # per-process detector, board corners, intrinsics


def _init():
    cv2.setNumThreads(1)
    board = aruco.CharucoBoard((SX, SY), SQ, MK, aruco.getPredefinedDictionary(getattr(aruco, _INFO["dictionary"])))
    _W["det"] = aruco.CharucoDetector(board)
    _W["obj"] = board.getChessboardCorners().astype(np.float64)


def intrinsics(cam, fl):
    key = (cam, fl)
    if key not in _W:
        npz = np.load(CALIB_ROOT / CAMERA_DIRS[cam] / "calibration" / f"fl_{fl}mm.npz")
        _W[key] = (npz["K"].astype(np.float64), npz["D"].astype(np.float64))
    return _W[key]


def fit_step(u, v, y, sigma_init=1.0):
    """Least-squares (Levenberg-Marquardt) fit of y = a + b*Phi((v - c0 - c1*u)/sigma); bright side at v > 0.
    Returns (a, b, c0, c1, sigma, sigma_se, rmse) or None."""
    lo, hi = y[v < -1.5], y[v > 1.5]
    if len(lo) < 10 or len(hi) < 10:
        return None
    p = np.array([np.median(lo), np.median(hi) - np.median(lo), 0.0, 0.0, np.log(sigma_init)])

    def rj(p):
        a, b, c0, c1, ls = p
        s = np.exp(ls)
        z = (v - c0 - c1 * u) / s
        phi = np.exp(-0.5 * z * z) / SQ2PI
        g = -b * phi / s
        J = np.stack([np.ones_like(z), ndtr(z), g, g * u, -b * phi * z], axis=1)
        return a + b * J[:, 1] - y, J

    r, J = rj(p)
    cost, lam = r @ r, 1e-3
    for _ in range(60):
        A, g = J.T @ J, J.T @ r
        try:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        except np.linalg.LinAlgError:
            return None
        pn = p + step
        pn[4] = np.clip(pn[4], np.log(0.05), np.log(200.0))
        rn, Jn = rj(pn)
        cn = rn @ rn
        if cn < cost:
            done = cost - cn < 1e-9 * cost or np.abs(step).max() < 1e-6
            p, r, J, cost, lam = pn, rn, Jn, cn, max(lam / 3, 1e-9)
            if done:
                break
        else:
            lam *= 5
            if lam > 1e8:
                break
    n = len(y)
    rmse = np.sqrt(cost / n)
    try:
        cov = np.linalg.inv(J.T @ J) * cost / max(n - 5, 1)
        se = float(np.exp(p[4]) * np.sqrt(max(cov[4, 4], 0)))
    except np.linalg.LinAlgError:
        se = np.nan
    return p[0], p[1], p[2], p[3], float(np.exp(p[4])), se, float(rmse)


def encode(x):
    """Linear light -> sRGB code value (0..1)."""
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


DG = 0.05  # grid step (px) for the sharpened edge profile


def sharpened_profile(g, a, b, sigma, alpha, r, space):
    """Edge profile on the grid g (px) as sRGB code values: a step in linear light (dark a, bright a + b)
    blurred by a Gaussian sigma, gamma-coded, and sharpened by an unsharp mask
    S = E + alpha * (E - E * Gauss(r)) applied to the coded values (space "code") or before coding ("lin")."""
    E = a + b * ndtr(g / sigma)
    if space == "code":
        E = encode(E)
    if alpha > 0:
        n = int(np.ceil(4 * r / DG))
        k = np.exp(-0.5 * (np.arange(-n, n + 1) * DG / r) ** 2)
        E = E + alpha * (E - np.convolve(np.pad(E, n, mode="edge"), k / k.sum(), "valid"))
    return E if space == "code" else encode(E)


def fit_sharpened(u, v, y, alpha, r, space="code", init=None):
    """LM fit of the sharpened-edge model (sharpening alpha, r fixed) to code values y (0..1). Parameters
    a, b (linear light), c0, c1, log sigma. Returns (params, sigma_se, rmse) or None."""
    lo, hi = y[v < -1.5], y[v > 1.5]
    if len(lo) < 10 or len(hi) < 10:
        return None
    if init is None:
        a0 = float(LUT[int(round(np.median(lo) * 255))])
        init = [a0, float(LUT[int(round(np.median(hi) * 255))]) - a0, 0.0, 0.0, 0.0]
    p = np.array(init, dtype=np.float64)
    npar = len(p)

    def prof(q, g):
        return sharpened_profile(g, q[0], q[1], np.exp(q[4]), alpha, r, space)

    def rj(q):
        w = v - q[2] - q[3] * u
        g = np.arange(w.min() - 1, w.max() + 1 + DG, DG)
        S = prof(q, g)
        m = np.interp(w, g, S)
        dS = np.interp(w, g, np.gradient(S, DG))
        J = np.empty((len(y), npar))
        J[:, 2], J[:, 3] = -dS, -dS * u
        for j in (0, 1, 4):
            h = 1e-4 if j in (0, 1) else 1e-3
            q2 = q.copy()
            q2[j] += h
            J[:, j] = (np.interp(w, g, prof(q2, g)) - m) / h
        return m - y, J

    r_, J = rj(p)
    cost, lam = r_ @ r_, 1e-3
    for _ in range(60):
        A, gr = J.T @ J, J.T @ r_
        try:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -gr)
        except np.linalg.LinAlgError:
            return None
        pn = p + step
        pn[4] = np.clip(pn[4], np.log(0.05), np.log(200.0))
        rn, Jn = rj(pn)
        cn = rn @ rn
        if cn < cost:
            done = cost - cn < 1e-9 * cost or np.abs(step).max() < 1e-6
            p, r_, J, cost, lam = pn, rn, Jn, cn, max(lam / 3, 1e-9)
            if done:
                break
        else:
            lam *= 5
            if lam > 1e8:
                break
    n = len(y)
    try:
        cov = np.linalg.inv(J.T @ J) * cost / max(n - npar, 1)
        se = float(np.exp(p[4]) * np.sqrt(max(cov[4, 4], 0)))
    except np.linalg.LinAlgError:
        se = np.nan
    return p, se, float(np.sqrt(cost / n))


def edge_samples(g8, P, Q, rvec, tvec, K, D):
    """Pixels around the middle of the board edge P-Q (object points). Returns a dict with u (scaled
    along-edge position), v (signed distance, bright side > 0), y (linear green), geometry; or None."""
    t = np.linspace(0, 1, 41)[:, None]
    d = (Q - P) / np.linalg.norm(Q - P)
    nrm = np.array([-d[1], d[0], 0.0])
    mid = (P + Q) / 2
    pts = np.vstack([P + t * (Q - P), mid + MARGIN_M * nrm, mid - MARGIN_M * nrm])
    img, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    img = img.reshape(-1, 2)
    poly, m1, m2 = img[:41], img[41], img[42]
    A, B = poly[0], poly[-1]
    L = float(np.linalg.norm(B - A))
    if L < 20:
        return None
    e = (B - A) / L
    n = np.array([-e[1], e[0]])
    uu, hh = (poly - A) @ e, (poly - A) @ n
    margin_px = float((np.linalg.norm(m1 - poly[20]) + np.linalg.norm(m2 - poly[20])) / 2)
    trim, ww, wb = 0.15 * L, 0.5 * margin_px, min(0.3 * L, 40.0)
    if ww < 2.5:
        return None
    wmax = max(ww, wb) + abs(hh).max() + 2
    corners = np.array([A + e * uq + n * vq for uq in (trim, L - trim) for vq in (-wmax, wmax)])
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    H, W = g8.shape
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return None
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    rel = np.stack([xs.ravel() - A[0], ys.ravel() - A[1]], axis=1)
    u = rel @ e
    v = rel @ n - np.interp(u, uu, hh)
    vals = g8[y0:y1 + 1, x0:x1 + 1].ravel()
    inside = (u >= trim) & (u <= L - trim)
    side_p = inside & (v > ww / 2) & (v < ww)
    side_n = inside & (v < -ww / 2) & (v > -ww)
    if side_p.sum() < 10 or side_n.sum() < 10:
        return None
    if np.median(vals[side_n]) > np.median(vals[side_p]):  # bright (white) side to v > 0
        v = -v
    keep = inside & (v <= ww) & (v >= -wb)
    white = keep & (v > 0)
    ang = np.degrees(np.arctan2(abs(e[1]), abs(e[0])))
    tilt = min(ang, 90 - ang)  # angle to the nearest pixel axis
    return {"u": (u[keep] - L / 2) / L, "v": v[keep], "y": LUT[vals[keep]], "yc": vals[keep] / 255.0,
            "sat": float((vals[white] >= 250).mean()), "L_px": L, "margin_px": margin_px, "w_white": ww,
            "w_black": wb, "tilt_deg": tilt, "phase_cov": (L - 2 * trim) * np.sin(np.radians(tilt)),
            "mid": poly[20], "normal": n, "xs": xs.ravel()[keep], "ys": ys.ravel()[keep]}


def sharp_edge(row):
    """Edges used to estimate the camera's sharpening: near focus (the halo is visible), a wide window on
    the white side (the plateau after the halo is reached), no clipping, enough sub-pixel phases."""
    return (row["sigma_plain"] <= 1.2 and row["w_white"] >= 6 and row["sat"] < 0.001
            and row["phase_cov"] >= 2 and row["rel_rmse_plain"] < 0.1 and row["contrast"] > 0.1)


USM_GRID = [(0.0, 1.0)] + [(al, rr) for rr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3, 1.7)
                           for al in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)]


def measure_image(job, mode="measure", usm=None, max_px=4000, keep_samples=False, keep_frac=1.0):
    """All inner edges of one calibration image -> list of row dicts (one per edge).
    mode "measure": plain fit (sigma_plain) and the sharpened-edge fit with the camera's sharpening usm
    {"alpha", "r"} (sigma). mode "sharpen": a random keep_frac of the sharp_edge() edges, fitted with every
    (alpha, r) of USM_GRID: squared error sse_i and sigma sig_i per grid point i."""
    cam, fl, name, date, N = job
    if not _W:
        _init()
    K, D = intrinsics(cam, fl)
    bgr = cv2.imread(str(CALIB_ROOT / CAMERA_DIRS[cam] / f"fl_{fl}mm" / name), cv2.IMREAD_COLOR)
    corners, ids, _, _ = _W["det"].detectBoard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 12:
        return []
    g8 = np.ascontiguousarray(bgr[:, :, 1])
    del bgr
    ids = ids.ravel()
    obj, img = _W["obj"][ids], corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D)
    if not ok:
        return []
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    cx, cy = K[0, 2], K[1, 2]
    half_diag = np.hypot(g8.shape[1] / 2, g8.shape[0] / 2)
    rng = np.random.default_rng(int(ids.sum()))
    have, nxc = set(ids.tolist()), SX - 1  # nxc inner corners per row
    rows = []
    for i in sorted(have):
        r, c = divmod(i, nxc)
        for j, orient in ((i + 1, "x"), (i + nxc, "y")):
            if (orient == "x" and c == nxc - 1) or j not in have:
                continue
            P, Q = _W["obj"][i], _W["obj"][j]
            s = edge_samples(g8, P, Q, rvec, tvec, K, D)
            if s is None:
                continue
            k = len(s["y"])
            sub = rng.choice(k, max_px, replace=False) if k > max_px else slice(None)
            uu, vv, yy, yc = s["u"][sub], s["v"][sub], s["y"][sub], s["yc"][sub]
            f = fit_step(uu, vv, yy)
            if f is None:
                continue
            a, b, c0, c1, sigma, se, rmse = f
            depth = float((R @ ((P + Q) / 2) + tvec.ravel())[2])
            rad = s["mid"] - (cx, cy)
            rn = np.linalg.norm(rad)
            row = {"camera": cam, "fl_mm": fl, "date": date, "N": N, "image": name, "edge": f"{orient}{i}",
                   "depth_m": round(depth, 4), "x_px": round(float(s["mid"][0]), 1),
                   "y_px": round(float(s["mid"][1]), 1), "r_norm": round(float(rn / half_diag), 4),
                   # |cos| of the angle between the edge normal (the direction blur is measured in) and the radius
                   "cos_radial": round(float(abs(s["normal"] @ rad) / rn) if rn > 0 else np.nan, 3),
                   # direction blur is measured in: 0 = across a vertical edge (horizontal blur), 90 = vertical
                   "normal_deg": round(float(np.degrees(np.arctan2(s["normal"][1], s["normal"][0])) % 180), 1),
                   "L_px": round(s["L_px"], 1), "margin_px": round(s["margin_px"], 2),
                   "w_white": round(s["w_white"], 2), "tilt_deg": round(s["tilt_deg"], 2),
                   "phase_cov": round(s["phase_cov"], 2), "n_px": int(min(k, max_px)),
                   "sat": round(s["sat"], 4), "dark": round(a, 4), "contrast": round(b, 4),
                   "sigma_plain": round(sigma, 4), "rel_rmse_plain": round(rmse / max(abs(b), 1e-6), 4)}
            init = [a, b, c0, c1, np.log(max(sigma, 0.1))]
            if mode == "sharpen":
                if not sharp_edge(row) or rng.random() > keep_frac:
                    continue
                for gi, (al, rr) in enumerate(USM_GRID):
                    g = fit_sharpened(uu, vv, yc, al, rr, "code", init=init)
                    if g is None:
                        break
                    row[f"sse_{gi}"] = g[2] ** 2 * len(yc)
                    row[f"sig_{gi}"] = float(np.exp(g[0][4]))
                else:
                    rows.append(row)
                continue
            g = fit_sharpened(uu, vv, yc, usm["alpha"], usm["r"], "code", init=init)
            if g is None:
                continue
            p, se2, rmse2 = g
            cc = float(encode(p[0] + p[1]) - encode(p[0]))  # contrast in code values
            row.update({"dark": round(p[0], 4), "contrast": round(p[1], 4),  # linear levels of this fit
                        "c0": round(p[2], 3), "c1": round(p[3], 3), "sigma": round(float(np.exp(p[4])), 4),
                        "sigma_se": round(se2, 4), "rel_rmse": round(rmse2 / max(abs(cc), 1e-6), 4)})
            if keep_samples:
                row["_s"] = {kk: (vx[sub] if isinstance(vx, np.ndarray) and vx.shape == s["y"].shape else vx)
                             for kk, vx in s.items()}
            rows.append(row)
    return rows


def debug_plot(cam, fl, name, out):
    """Overlay of the measured edge regions on one image, and a few fitted edge profiles."""
    aud = pd.read_csv(AUDIT_CSV)
    r = aud[(aud.camera == cam) & (aud.fl_mm == fl) & (aud.image == name)].iloc[0]
    usm = json.loads((out / "sharpening.json").read_text())
    rows = measure_image((cam, fl, name, r.date, r.N), usm=usm, keep_samples=True)
    print(f"{len(rows)} edges; sigma median {np.median([x['sigma'] for x in rows]):.3f} px")
    g8 = cv2.imread(str(CALIB_ROOT / CAMERA_DIRS[cam] / f"fl_{fl}mm" / name), cv2.IMREAD_COLOR)[:, :, 1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(g8, cmap="gray")
    for x in rows:
        ax.plot(x["_s"]["xs"][::7], x["_s"]["ys"][::7], ",", alpha=0.5)
        ax.text(x["x_px"], x["y_px"], f"{x['sigma']:.2f}", fontsize=5, color="red")
    ax.set_title(f"{cam} {fl}mm {name} F{r.N:g}: measured regions, sigma (px)")
    fig.savefig(out / f"debug_{cam}_{fl}_{name}.png", dpi=150)
    plt.close(fig)
    pick = sorted(rows, key=lambda x: x["sigma"])
    pick = [pick[i] for i in np.linspace(0, len(pick) - 1, 6).astype(int)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, x in zip(axs.ravel(), pick):
        s = x["_s"]
        vv = s["v"] - x["c0"] - x["c1"] * s["u"]
        ax.plot(vv, s["yc"], ".", ms=1, alpha=0.4)
        g = np.arange(vv.min(), vv.max(), DG)
        ax.plot(g, sharpened_profile(g, x["dark"], x["contrast"], x["sigma"], usm["alpha"], usm["r"], "code"),
                "r-", label=f"sharpened model, sigma {x['sigma']:.2f}")
        ax.plot(g, encode(x["dark"] + x["contrast"] * ndtr(g / x["sigma_plain"])), "g--", lw=0.8,
                label=f"plain fit, sigma {x['sigma_plain']:.2f}")
        ax.set_title(f"{x['edge']} d={x['depth_m']:.2f}m rel_rmse={x['rel_rmse']:.3f}", fontsize=8)
        ax.set_xlabel("distance from edge (px)")
        ax.set_ylabel("code value (0-1)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"debug_{cam}_{fl}_{name}_profiles.png", dpi=110)
    plt.close(fig)


def _jobs(aud):
    return list(aud[["camera", "fl_mm", "image", "date", "N"]].itertuples(index=False, name=None))


def _run(jobs, workers, **kw):
    rows, t0 = [], time.time()
    print(f"{len(jobs)} images, {workers} workers", flush=True)
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(partial(measure_image, **kw), jobs, chunksize=1)):
            rows += r
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} images, {len(rows)} edges, {time.time() - t0:.0f}s", flush=True)
    print(f"  {len(rows)} edges, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows).sort_values(["camera", "fl_mm", "date", "image", "edge"])


def sharpen(a):
    """Estimate the in-camera sharpening (unsharp mask strength alpha, radius r, and whether it acts on
    gamma-coded or linear values) from sharp edges of 2 images per camera, focal length and session."""
    aud = pd.read_csv(AUDIT_CSV)
    aud = aud[aud.n_corners >= 60]
    aud = aud.groupby(["camera", "fl_mm", "date"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 2), random_state=0))
    if a.limit:
        aud = aud.sample(a.limit, random_state=0)
    if not (a.out / "sharpen_edges.csv").exists():
        df = _run(_jobs(aud), a.workers, mode="sharpen", keep_frac=a.keep_frac)
        df.to_csv(a.out / "sharpen_edges.csv", index=False)
    df = pd.read_csv(a.out / "sharpen_edges.csv")
    ng = len(USM_GRID)
    sse, sig = df[[f"sse_{i}" for i in range(ng)]].to_numpy(), df[[f"sig_{i}" for i in range(ng)]].to_numpy()
    print(f"\n{len(df)} sharp edges from {df.image.nunique()} images; error relative to no sharpening "
          f"(1 = no better), and median sigma, at the best grid point:")
    best, res = {}, []
    for (cam, date), idx in [(("all", ""), np.arange(len(df)))] + \
            [(k, np.flatnonzero((df.camera == k[0]).to_numpy() & (df.date == k[1]).to_numpy()))
             for k in df.groupby(["camera", "date"]).groups]:
        tot = sse[idx].sum(0)
        i = int(np.argmin(tot))
        best[(cam, date)] = i
        # grid points within 1% of the best total error: how far sigma moves along the alpha-r valley
        near = np.flatnonzero(tot <= tot[i] * 1.01)
        med = np.median(sig[idx], axis=0)
        res.append({"camera": cam, "date": date, "edges": len(idx), "alpha": USM_GRID[i][0], "r": USM_GRID[i][1],
                    "rel_error": round(tot[i] / tot[0], 4), "sigma_med": round(med[i], 3),
                    "sigma_med_none": round(med[0], 3), "sigma_med_range_1pct": f"{med[near].min():.3f}-{med[near].max():.3f}",
                    "sigma_med_overall_best": round(med[best[("all", "")]], 3)})
    res = pd.DataFrame(res)
    print(res.to_string(index=False))
    res.to_csv(a.out / "sharpening_grid_best.csv", index=False)
    i = best[("all", "")]
    usm = {"alpha": USM_GRID[i][0], "r": USM_GRID[i][1], "space": "code", "edges": len(df),
           "note": "unsharp mask S = E + alpha (E - E * Gauss(r)) on sRGB code values; grid search over sharp edges"}
    (a.out / "sharpening.json").write_text(json.dumps(usm, indent=2))
    print(f"\nsharpening: {usm} -> {a.out / 'sharpening.json'}")
    # error surface per camera and session
    keys = list(best)
    al_v = sorted({g[0] for g in USM_GRID[1:]})
    r_v = sorted({g[1] for g in USM_GRID[1:]})
    fig, axs = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.3), squeeze=False)
    for ax, k in zip(axs[0], keys):
        idx = np.arange(len(df)) if k[0] == "all" else \
            np.flatnonzero((df.camera == k[0]).to_numpy() & (df.date == k[1]).to_numpy())
        tot = sse[idx].sum(0)
        Z = np.full((len(r_v), len(al_v)), np.nan)
        for gi, (al, rr) in enumerate(USM_GRID[1:], start=1):
            Z[r_v.index(rr), al_v.index(al)] = tot[gi] / tot[0]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmax=np.nanmin(Z) + 0.05)
        ax.set_xticks(range(len(al_v)), [f"{x:g}" for x in al_v], fontsize=6)
        ax.set_yticks(range(len(r_v)), [f"{x:g}" for x in r_v], fontsize=6)
        bi = best[k]
        if bi:
            ax.plot(al_v.index(USM_GRID[bi][0]), r_v.index(USM_GRID[bi][1]), "r*")
        ax.set_title(f"{k[0]} {k[1]} ({len(idx)})", fontsize=8)
        ax.set_xlabel("alpha", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axs[0][0].set_ylabel("r (px)")
    fig.suptitle("squared error relative to no sharpening, per (alpha, r); star = best", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "sharpening_grid.png", dpi=120)
    plt.close(fig)


def measure(a):
    usm = json.loads((a.out / "sharpening.json").read_text())
    aud = pd.read_csv(AUDIT_CSV)
    aud = aud[aud.n_corners >= 12]
    if a.limit:
        aud = aud.sample(a.limit, random_state=0)
    df = _run(_jobs(aud), a.workers, mode="measure", usm=usm)
    df.to_csv(a.out / "edges.csv", index=False)
    print(f"done: {len(df)} edges from {df.image.nunique()} images -> {a.out / 'edges.csv'}")


GROUP = ["camera", "fl_mm", "date", "N"]


def good_edges(df):
    """Edges with a trustworthy sigma: no clipping, enough sub-pixel phases, a good fit, a window wide
    enough for the blur (2.5 sigma on the white side), a small fit error."""
    ok = ((df.sat < 0.002) & (df.phase_cov >= 2) & (df.rel_rmse < 0.08) & (df.contrast > 0.1)
          & (df.sigma <= df.w_white / 2.5) & (df.sigma_se < 0.05 + 0.05 * df.sigma) & (df.sigma > 0.2))
    return df[ok].copy()


def v_curve(d, s, m, s0):
    return np.sqrt(s0 ** 2 + (m * (s - d) / d) ** 2)


def m_from_k(s, k, v_img, N):
    """Defocus slope m (px) from the blur constant k: sigma = k f^2 / (N (s - f)) |s - d| / d, with the
    focal length f from the image distance v and the focus distance s (thin lens, 1/f = 1/v + 1/s)."""
    f = v_img * s / (v_img + s)
    return k * f ** 2 / (N * (s - f))


def fit_group(d, sig, kfix=None):
    """V-curve sigma(d)^2 = sigma0^2 + (m (s - d) / d)^2 with one floor sigma0 for all shots of a group,
    fitted in log sigma with a robust loss. Camera motion does not depend on depth, so it raises the floor
    but hardly changes s and m. With kfix = (k, v_img, N), m = m_from_k(s, ...): for groups whose V-curve
    is flat or one-sided. Start: a grid over s, with sigma0^2 (and m^2) from a linear fit of sigma^2.
    Returns s, m, sigma0."""
    from scipy.optimize import least_squares
    ls, best = np.log(sig), None
    for s in np.geomspace(d.min() / 1.5, d.max() * (5 if kfix else 1.5), 150):
        x2 = ((s - d) / d) ** 2
        if kfix:
            mk = m_from_k(s, *kfix)
            c = [max(float(np.median(sig ** 2 - mk ** 2 * x2)), 1e-4), mk ** 2]
        else:
            c = np.maximum(np.linalg.lstsq(np.stack([np.ones_like(x2), x2], 1), sig ** 2, rcond=None)[0], 1e-4)
        cost = np.sum(np.minimum((0.5 * np.log(c[0] + c[1] * x2) - ls) ** 2, 0.04))
        if best is None or cost < best[0]:
            best = (cost, s, np.sqrt(c[1]), np.sqrt(c[0]))
    if kfix:
        res = least_squares(lambda p: np.log(v_curve(d, p[0], m_from_k(p[0], *kfix), np.exp(p[1]))) - ls,
                            [best[1], np.log(best[3])], loss="soft_l1", f_scale=0.05,
                            bounds=([0.3, -np.inf], [np.inf, np.inf]))
        return res.x[0], float(m_from_k(res.x[0], *kfix)), float(np.exp(res.x[1]))
    res = least_squares(lambda p: np.log(v_curve(d, p[0], np.exp(p[1]), np.exp(p[2]))) - ls,
                        [best[1], np.log(best[2]), np.log(best[3])], loss="soft_l1", f_scale=0.05)
    return res.x[0], float(np.exp(res.x[1])), float(np.exp(res.x[2]))


def fit_floors(d, sig, theta, img, s, m, floor0):
    """Given a group's V-curve (s, m), the blur floor of every image, per direction:
        sigma^2 = (m (s - d) / d)^2 + hx_j^2 cos^2(theta) + hy_j^2 sin^2(theta)
    hx_j, hy_j = blur at focus of image j, horizontal / vertical (base blur + that shot's camera motion);
    theta = direction the edge's blur is measured in (0 = horizontal). Robust fit in log sigma.
    img: image index (0..n-1) per edge. Returns hx, hy."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    n_img, n = int(img.max()) + 1, len(d)
    c2, s2, ls = np.cos(theta) ** 2, np.sin(theta) ** 2, np.log(sig)
    dfc = (m * (s - d) / d) ** 2

    def resid(p):
        hx, hy = np.exp(p[:n_img])[img], np.exp(p[n_img:])[img]
        return 0.5 * np.log(np.clip(dfc + hx ** 2 * c2 + hy ** 2 * s2, 1e-6, 1e8)) - ls

    r = np.arange(n)
    sp = coo_matrix((np.ones(2 * n, dtype=int), (np.r_[r, r], np.r_[img, n_img + img])), shape=(n, 2 * n_img))
    p = least_squares(resid, np.full(2 * n_img, np.log(floor0)), loss="soft_l1", f_scale=0.05,
                      jac_sparsity=sp, x_scale="jac").x
    return np.exp(p[:n_img]), np.exp(p[n_img:])


CENTRE = 0.5  # per-image floors use edges within this distance from the image centre (1 = corner)
SHARP_Q = 20  # sigma0 = this percentile over images of the per-image floor (the least-moved shots)


def group_task(args):
    """One group: V-curve (s, m) from all shots together (fit_group), then the floor of each shot per
    direction from its central edges (fit_floors); sigma0 per direction = SHARP_Q-th percentile of the
    floors. Bootstrap over images repeats both steps. Returns key, (s, m, sigma0 shared floor),
    ((s, m, hx, hy, None), bootstrap rows (s, m, sigma0_v, sigma0_h), images with floors)."""
    key, g, n_boot, seed, kfix = args
    rng = np.random.default_rng(seed)
    start = fit_group(g.depth_m.to_numpy(), g.sigma.to_numpy(), kfix)
    c = g[g.r_norm < CENTRE]
    vert = c.normal_deg.between(45, 135)
    nv, nh = c[vert].groupby("image").size(), c[~vert].groupby("image").size()
    imgs = np.array([im for im in c.image.unique() if nv.get(im, 0) >= 5 and nh.get(im, 0) >= 5])
    if len(imgs) < 4:
        return key, start, None
    cblocks = {im: c[c.image == im] for im in imgs}
    gblocks = {im: b for im, b in g.groupby("image")}

    def floors(names, s, m, s0):
        parts = [cblocks[im] for im in names]
        hx, hy = fit_floors(np.concatenate([p.depth_m for p in parts]), np.concatenate([p.sigma for p in parts]),
                            np.radians(np.concatenate([p.normal_deg for p in parts])),
                            np.concatenate([np.full(len(p), j) for j, p in enumerate(parts)]), s, m, s0)
        return hx, hy

    hx, hy = floors(imgs, *start)
    boot, all_imgs, has_floor = [], np.array(list(gblocks)), set(imgs)
    for _ in range(n_boot):
        draw = rng.choice(all_imgs, len(all_imgs))
        try:
            bg = pd.concat([gblocks[im] for im in draw])
            st = fit_group(bg.depth_m.to_numpy(), bg.sigma.to_numpy(), kfix)
            names, counts = np.unique([im for im in draw if im in has_floor], return_counts=True)
            if len(names) < 3:
                continue
            bx, by = floors(names, *st)
            boot.append((st[0], st[1], np.percentile(np.repeat(by, counts), SHARP_Q),
                         np.percentile(np.repeat(bx, counts), SHARP_Q)))
        except ValueError:  # a degenerate resample (non-finite values); skipped
            pass
    return key, start, ((start[0], start[1], hx, hy, None), np.array(boot), imgs)


def fit(a):
    """Fits on edges.csv, per group (camera, focal length, session date, F-number):
    1. V-curve from all shots together, one shared floor: sigma(d)^2 = floor^2 + (m (s - d) / d)^2
       (fit_group; the shared floor is the plain_sigma0 column). Pass 2: groups whose focus is not inside
       their board depths are fitted again with k fixed (see below).
    2. Given (s, m), the floor of each shot per direction from its central edges (fit_floors). sigma0
       per direction = SHARP_Q-th percentile of the floors (the least-moved shots); motion of a shot =
       sqrt(floor^2 - sigma0^2). 90% intervals from a bootstrap over images (repeats 1 and 2).
       k = m / (f^2 / (N (s - f))) in px/m, with f from the calibrated image distance v = fx * pixel and s.
    3. sigma0 summary per camera and F-number (spread over focal lengths), and blur against position in
       the image relative to the same shot's centre (edges near focus)."""
    df = pd.read_csv(a.out / "edges.csv")
    ed = good_edges(df)
    print(f"{len(ed)}/{len(df)} edges pass the quality checks", flush=True)
    groups = {key: g for key, g in ed.groupby(GROUP) if len(g) >= 200 and g.image.nunique() >= 4}

    def v_of(cam, fl):  # image distance from the calibrated focal length in pixels
        return np.load(CALIB_ROOT / CAMERA_DIRS[cam] / "calibration" / f"fl_{fl}mm.npz")["K"][0, 0] * PIXEL_M

    def inside(key, s):
        d = groups[key].depth_m
        return bool(np.percentile(d, 5) < s < np.percentile(d, 95))

    # pass 1: s and m free. Pass 2: groups whose focus is not inside their board depths, or whose slope is
    # not determined (flat or one-sided V-curve), again with k fixed to that camera's well-determined
    # pass-1 values at this focal length (interpolated over focal length); k belongs to the lens, so it
    # carries over between sessions
    results = {}
    with Pool(a.workers) as pool:
        tasks = [(key, g, a.boot, i, None) for i, (key, g) in enumerate(groups.items())]
        for key, start, mres in pool.imap_unordered(group_task, tasks):
            results[key] = (start, mres, None)
        kref = {}
        for key, (start, mres, _) in results.items():
            if mres is None:
                continue
            s, m = mres[0][:2]
            mb = mres[1][:, 1]
            if inside(key, s) and np.percentile(mb, 95) - np.percentile(mb, 5) < 0.3 * m:  # k well determined
                f = v_of(key[0], key[1]) * s / (v_of(key[0], key[1]) + s)
                kref.setdefault(key[0], []).append((key[1], m * key[3] * (s - f) / f ** 2))
        redo = []
        for i, (key, (start, mres, _)) in enumerate(results.items()):
            if mres is None or key[0] not in kref:
                continue
            mb, m = mres[1][:, 1], mres[0][1]
            if inside(key, mres[0][0]) and np.percentile(mb, 95) - np.percentile(mb, 5) < 0.75 * m:
                continue  # focus inside the board depths and the slope at least roughly determined
            fls, ks = zip(*sorted(kref[key[0]]))
            k = float(np.interp(key[1], fls, ks))
            redo.append((key, groups[key], a.boot, 1000 + i, (k, v_of(key[0], key[1]), key[3])))
        print(f"pass 2: {len(redo)} groups with k fixed", flush=True)
        for (key, start, mres), t in zip(pool.imap(group_task, redo), redo):
            results[key] = (start, mres, t[4][0])

    rows, gfit, per_img = [], {}, []
    for key in sorted(results):
        start, mres, kfixed = results[key]
        cam, fl, date, N = key
        g = groups[key]
        d = g.depth_m.to_numpy()
        row = {"camera": cam, "fl_mm": fl, "date": date, "N": N, "images": g.image.nunique(), "edges": len(g),
               "d_p5": round(np.percentile(d, 5), 3), "d_p95": round(np.percentile(d, 95), 3),
               "plain_s_m": round(start[0], 3), "plain_sigma0": round(start[2], 3),
               "k_fixed": round(kfixed) if kfixed else np.nan}
        if mres is not None:
            (s, m, hx, hy, _), boot, imgs = mres
            s0v, s0h = np.percentile(hy, SHARP_Q), np.percentile(hx, SHARP_Q)
            v_img = v_of(cam, fl)
            f = v_img * s / (v_img + s)  # thin lens: 1/f = 1/v + 1/s
            k = m / (f ** 2 / (N * (s - f))) if s > f else np.nan
            mx = np.sqrt(np.maximum(hx ** 2 - s0h ** 2, 0))
            my = np.sqrt(np.maximum(hy ** 2 - s0v ** 2, 0))

            def q(x, p):
                return round(float(np.percentile(x, p)), 3)
            row.update({"s_m": round(s, 3), "s_lo": q(boot[:, 0], 5), "s_hi": q(boot[:, 0], 95),
                        "sigma0_v": round(s0v, 3), "sigma0_v_lo": q(boot[:, 2], 5), "sigma0_v_hi": q(boot[:, 2], 95),
                        "sigma0_h": round(s0h, 3), "sigma0_h_lo": q(boot[:, 3], 5), "sigma0_h_hi": q(boot[:, 3], 95),
                        "floor_v_med": round(float(np.median(hy)), 3), "floor_h_med": round(float(np.median(hx)), 3),
                        "motion_h_med": round(float(np.median(mx)), 3), "motion_v_med": round(float(np.median(my)), 3),
                        "m_px": round(m, 3), "k_per_m": round(k) if np.isfinite(k) else np.nan,
                        "k_lo": round(q(boot[:, 1], 5) / m * k) if np.isfinite(k) else np.nan,
                        "k_hi": round(q(boot[:, 1], 95) / m * k) if np.isfinite(k) else np.nan,
                        "fit_images": len(imgs), "s_inside": bool(np.percentile(d, 5) < s < np.percentile(d, 95))})
            gfit[key] = (s, m, s0v, s0h, dict(zip(imgs, zip(hx, hy))))
            per_img += [{"camera": cam, "fl_mm": fl, "date": date, "N": N, "image": im,
                         "floor_h": round(float(x), 3), "floor_v": round(float(y), 3),
                         "motion_h": round(float(u), 3), "motion_v": round(float(w), 3)}
                        for im, x, y, u, w in zip(imgs, hx, hy, mx, my)]
            print(f"  {cam} {fl:2d}mm {date} F{N:g}: s={s:.2f} m  sigma0 vert {s0v:.3f} [{row['sigma0_v_lo']:.3f}, "
                  f"{row['sigma0_v_hi']:.3f}]  horiz {s0h:.3f} [{row['sigma0_h_lo']:.3f}, {row['sigma0_h_hi']:.3f}]  "
                  f"(plain {start[2]:.3f})  k={row['k_per_m']}  motion h/v median "
                  f"{row['motion_h_med']:.2f}/{row['motion_v_med']:.2f} px", flush=True)
        rows.append(row)
    gt = pd.DataFrame(rows).sort_values(GROUP)
    gt.to_csv(a.out / "groups_fit.csv", index=False)
    pi = pd.DataFrame(per_img).sort_values(GROUP + ["image"])
    pi.to_csv(a.out / "per_image_floor.csv", index=False)

    # sigma0 per camera, F-number and direction: weighted mean over focal lengths, spread, trend
    summ = []
    for direc in ("v", "h"):
        col, lo, hi = f"sigma0_{direc}", f"sigma0_{direc}_lo", f"sigma0_{direc}_hi"
        # measured floors only: a narrow interval, and a lower bound above the blur of the pixel area
        # alone (0.29 px), which no real camera can beat. Steep V-curves with no boards near focus fail this.
        ok = gt[(gt[hi] - gt[lo] <= 0.2) & (gt[lo] >= 0.3) & (gt.fit_images >= 8)]
        for (cam, N), g in ok.groupby(["camera", "N"]):
            w = 1 / np.maximum((g[hi] - g[lo]) / 3.29, 0.01) ** 2  # 90% interval -> sd
            mean = float(np.sum(w * g[col]) / np.sum(w))
            chi2 = float(np.sum(w * (g[col] - mean) ** 2))
            slope = np.polyfit(g.fl_mm, g[col], 1, w=np.sqrt(w))[0] if len(g) > 2 else np.nan
            summ.append({"direction": {"v": "vertical", "h": "horizontal"}[direc], "camera": cam, "N": N,
                         "groups": len(g), "fl_mm": " ".join(map(str, g.fl_mm)), "sigma0_mean": round(mean, 3),
                         "sigma0_min": g[col].min(), "sigma0_max": g[col].max(),
                         "chi2_per_dof": round(chi2 / max(len(g) - 1, 1), 2),
                         "slope_px_per_10mm": round(10 * slope, 3) if np.isfinite(slope) else np.nan,
                         "diffraction_sigma": round(KAPPA * N, 3)})
    summ = pd.DataFrame(summ)
    summ.to_csv(a.out / "sigma0_summary.csv", index=False)
    print("\n" + summ.to_string(index=False))

    # position: blur of an edge near focus relative to the same shot's blur floor in that direction
    pos = []
    for key, (s, m, s0v, s0h, flo) in gfit.items():
        g = ed[(ed.camera == key[0]) & (ed.fl_mm == key[1]) & (ed.date == key[2]) & (ed.N == key[3])
               & ed.image.isin(list(flo))].copy()
        th = np.radians(g.normal_deg.to_numpy())
        hx = np.array([flo[i][0] for i in g.image])
        hy = np.array([flo[i][1] for i in g.image])
        base = hx ** 2 * np.cos(th) ** 2 + hy ** 2 * np.sin(th) ** 2
        dfc = (m * (s - g.depth_m.to_numpy()) / g.depth_m.to_numpy()) ** 2
        g["ratio"] = np.sqrt(np.maximum(g.sigma.to_numpy() ** 2 - dfc, 0.01) / base)
        pos.append(g[dfc < 0.25 * base])
    pos = pd.concat(pos)
    pos["r_bin"] = pd.cut(pos.r_norm, [0, 0.2, 0.4, 0.6, 0.8, 1.01])
    pos["blur_dir"] = np.where(pos.normal_deg.between(45, 135), "vertical", "horizontal")
    pos["radial"] = pd.cut(pos.cos_radial, [-0.01, 0.3, 0.7, 1.0], labels=["across radius", "mixed", "along radius"])
    with open(a.out / "position.txt", "w") as fh:
        for col in ("r_bin", "radial"):
            t = pos.groupby(["camera", "N", "blur_dir", col], observed=True).ratio.agg(["size", "median"]).round(3)
            fh.write(f"== blur near focus by {col}, relative to the same shot's floor (centre, same direction)\n"
                     f"{t.to_string()}\n\n")
    print((a.out / "position.txt").read_text())
    plots(ed, gt, gfit, pos, a.out)
    print(f"-> {a.out}")


def plots(ed, gt, gfit, pos, out):
    # V-curves per camera: one panel per group; points = all edges, lines = floor of the sharpest shots
    for cam in ("L", "R"):
        keys = sorted(k for k in gfit if k[0] == cam)
        if not keys:
            continue
        nc = 6
        nr = int(np.ceil(len(keys) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
        for ax, key in zip(axs.ravel(), keys):
            g = ed[(ed.camera == key[0]) & (ed.fl_mm == key[1]) & (ed.date == key[2]) & (ed.N == key[3])]
            vert = g.normal_deg.between(45, 135)
            ax.plot(g.depth_m[~vert], g.sigma[~vert], ".", ms=1, alpha=0.15, color="C1")
            ax.plot(g.depth_m[vert], g.sigma[vert], ".", ms=1, alpha=0.15, color="C0")
            dd = np.linspace(g.depth_m.min(), g.depth_m.max(), 200)
            s, m, s0v, s0h, _ = gfit[key]
            ax.plot(dd, v_curve(dd, s, m, s0v), "b-", lw=1)
            ax.plot(dd, v_curve(dd, s, m, s0h), "r-", lw=1)
            ax.set_title(f"{key[1]}mm {key[2][5:]} F{key[3]:g}: s={s:.2f} s0={s0v:.2f}/{s0h:.2f}", fontsize=8)
            ax.set_ylim(0, min(np.percentile(g.sigma, 99) * 1.2, 15))
            ax.tick_params(labelsize=7)
        for ax in axs.ravel()[len(keys):]:
            ax.axis("off")
        fig.supxlabel("board depth (m)")
        fig.supylabel("sigma (px, before sharpening)")
        fig.suptitle(f"camera {cam}: blur vs depth (points: blue vertical, orange horizontal blur); lines: V-curve "
                     f"with the floor of the sharpest {SHARP_Q}% of shots (blue vertical, red horizontal)")
        fig.tight_layout()
        fig.savefig(out / f"vcurves_{cam}.png", dpi=110)
        plt.close(fig)
    # sigma0 per group vs focal length, per direction
    dates = sorted(gt.date.unique())
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey=True, sharex=True)
    for j, cam in enumerate(("L", "R")):
        for i, direc in enumerate(("v", "h")):
            ax = axs[i, j]
            g = gt[(gt.camera == cam) & gt[f"sigma0_{direc}"].notna()]
            for (date, N), h in g.groupby(["date", "N"]):
                off = {16.0: -0.6, 20.0: 0.6, 22.0: 1.2}.get(N, 0)
                col = f"C{dates.index(date)}"
                y, lo, hi = h[f"sigma0_{direc}"], h[f"sigma0_{direc}_lo"], h[f"sigma0_{direc}_hi"]
                ax.errorbar(h.fl_mm + off, y, yerr=[y - lo, hi - y], fmt="o", ms=4, capsize=2, color=col,
                            label=f"{date} F{N:g}")
                ax.plot(h.fl_mm + off, h[f"floor_{direc}_med"], "_", ms=8, color=col, alpha=0.7)
            ax.set_title(f"camera {cam}, {'vertical' if direc == 'v' else 'horizontal'} blur", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 2)
        axs[0, j].legend(fontsize=7)
        axs[1, j].set_xlabel("focal length (mm)")
    for i in range(2):
        axs[i, 0].set_ylabel("sigma0 (px, raw pixels)")
    fig.suptitle(f"sigma0 per group = blur at focus of the sharpest {SHARP_Q}% of shots (90% bootstrap over "
                 f"images); dash = median shot", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sigma0_vs_fl.png", dpi=120)
    plt.close(fig)
    # blur near focus vs position, relative to the same shot's centre
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, direc in zip(axs, ("horizontal", "vertical")):
        for (cam, N), g in pos[pos.blur_dir == direc].groupby(["camera", "N"]):
            t = g.groupby("r_bin", observed=True).ratio.median()
            ax.plot([iv.mid for iv in t.index], t.values, "o-", label=f"{cam} F{N:g} ({len(g)} edges)")
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xlabel("distance from image centre (1 = corner)")
        ax.set_title(f"{direc} blur")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("blur / same shot's blur at the centre")
    fig.suptitle("blur near focus across the image, relative to the same shot's centre (defocus removed)")
    fig.tight_layout()
    fig.savefig(out / "sigma0_vs_position.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sharpen", "measure", "fit"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="measure: only N random images (testing)")
    ap.add_argument("--debug", nargs=3, metavar=("CAMERA", "FL", "IMAGE"), help="measure: plots for one image")
    ap.add_argument("--boot", type=int, default=100, help="fit: bootstrap resamples per group")
    ap.add_argument("--keep_frac", type=float, default=0.2, help="sharpen: share of the sharp edges used")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.step == "sharpen":
        sharpen(a)
    elif a.step == "measure":
        if a.debug:
            debug_plot(a.debug[0], int(a.debug[1]), a.debug[2], a.out)
        else:
            measure(a)
    else:
        fit(a)


if __name__ == "__main__":
    main()
