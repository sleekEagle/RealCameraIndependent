"""
MODEST_depth data: manifest rows, native-resolution crops, crop-adjusted intrinsics.

Main setting: no resizing (resizing shrinks the defocus blur, which this study is about).
A crop keeps the focal length (fx, fy) and moves the principal point by the crop offset.

Resized setting (for comparison, TrainResized / EvalResized): the whole image is shrunk to a short
side of `short` px (518 -> 777 x 518) with area averaging, like a lower-resolution camera; the
depth for training is shrunk with nearest-neighbour sampling (no mixing of depths across edges);
the intrinsics are scaled by the same factor. Evaluation upsamples the prediction back to full
resolution and scores it in the same fixed crop windows as the native setting.
"""
import csv
import hashlib
import random

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from pathlib import Path

from .config import DEPTH_ROOT, FOCUS_CSV, MANIFEST, MAX_DEPTH, MIN_DEPTH, RESIZED_CACHE


def read_manifest(path=MANIFEST):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select(rows, fold=None, labels=None, view_role=None):
    """Rows with fold label in `labels` (e.g. ("train",)) and/or the given view_role."""
    out = rows
    if fold is not None and labels is not None:
        out = [r for r in out if r[fold] in labels]
    if view_role is not None:
        out = [r for r in out if r["view_role"] == view_role]
    return out


_FOCUS = None


def subset(rows, n):
    """A fixed random subset of n rows (all rows if n is 0 or larger than len(rows))."""
    return random.Random(0).sample(rows, n) if 0 < n < len(rows) else rows


def label_blur(depth, row, blur_k, sigma0):
    """Blur sigma (native px) that the blur model gives for the true depth:
    sqrt(sigma0^2 + (C |s - d| / d)^2), C = k f^2 / (N (s - f)) (as models.CamindW.blur_scale)."""
    v, N, s = (float(t) for t in camera(row))
    f = v * s / (v + s)
    C = blur_k * f ** 2 / (N * (s - f))
    return np.sqrt(sigma0 ** 2 + (C * np.abs(s - depth) / depth) ** 2)


def focus_table(path=FOCUS_CSV):
    """(scene, fl_mm, side) -> estimated focus distance in metres (def_calibration/focus_from_aperture_pairs.py).
    Empty if the file is missing."""
    global _FOCUS
    if _FOCUS is None:
        _FOCUS = {}
        if path.exists():
            with open(path, newline="") as f:
                _FOCUS = {(int(r["scene"]), int(r["fl_mm"]), r["side"]): float(r["d_focus"])
                          for r in csv.DictReader(f) if r["status"] == "ok"}
    return _FOCUS


def image_distance_m(row):
    """Calibrated image distance v (m): original fx x pixel pitch. Not the lens focal length f: the
    calibration fits a pinhole model, whose focal length is the lens-to-sensor distance, v = f s / (s - f),
    2-5% larger than f at our focus distances. Column v_mm (called f_true_mm in older manifests)."""
    return float(row["v_mm"] if "v_mm" in row else row["f_true_mm"]) / 1000


def camera(row):
    """Lens settings of an image: [image distance v (m), F-number, focus distance s (m, NaN if unknown)].
    The focal length follows from 1/f = 1/v + 1/s."""
    s = focus_table().get((int(row["scene"]), int(row["fl_mm"]), row["side"]), float("nan"))
    return torch.tensor([image_distance_m(row), float(row["f_number"]), s], dtype=torch.float32)


def _path(rel):
    """Manifest paths are written with backslashes; '/' works on Windows and Linux."""
    return DEPTH_ROOT / rel.replace("\\", "/")


def load_image(row):
    """Full-resolution RGB uint8 (H, W, 3)."""
    img = cv2.imread(str(_path(row["color_path"])), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(row["color_path"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(row):
    """Full-resolution depth float32 (H, W, metres, NaN = invalid)."""
    depth = tifffile.imread(_path(row["depth_path"])).astype(np.float32)
    depth[~np.isfinite(depth) | (depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = np.nan
    return depth


def load_pair(row):
    """Full-resolution RGB uint8 (H, W, 3) and depth float32 (H, W, metres, NaN = invalid)."""
    return load_image(row), load_depth(row)


def intrinsics(row, x0=0, y0=0, width=None, flip=False):
    """3x3 K for a crop starting at (x0, y0); flip mirrors the principal point horizontally."""
    fx, fy, cx, cy = (float(row[k]) for k in ("fx", "fy", "cx", "cy"))
    cx, cy = cx - x0, cy - y0
    if flip:
        cx = width - 1 - cx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def crop_sample(img, depth, row, x0, y0, size, flip=False):
    im = img[y0:y0 + size, x0:x0 + size]
    d = depth[y0:y0 + size, x0:x0 + size]
    if flip:
        im, d = im[:, ::-1], d[:, ::-1]
    return {
        "image": torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0,
        "depth": torch.from_numpy(np.ascontiguousarray(d)),
        "K": torch.from_numpy(intrinsics(row, x0, y0, size, flip)),
        "px_scale": torch.tensor(1.0),
        "cam": camera(row),
    }


def _stable_seed(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def eval_crop_positions(row, depth, size, n, min_valid=0.5, tries=200):
    """Deterministic crop positions for one image (same for every model and every run)."""
    rng = np.random.default_rng(_stable_seed(row["color_path"]))
    H, W = depth.shape
    valid = np.isfinite(depth)
    out = []
    for _ in range(tries):
        x0, y0 = int(rng.integers(0, W - size + 1)), int(rng.integers(0, H - size + 1))
        if valid[y0:y0 + size, x0:x0 + size].mean() >= min_valid:
            out.append((x0, y0))
            if len(out) == n:
                break
    return out


class EvalCrops(Dataset):
    """Fixed crops of the given rows; one item per crop. Loads each image once per crop index
    (rows are processed in order, so a small cache keeps decoding to once per image)."""

    def __init__(self, rows, size=518, crops_per_image=6):
        self.rows, self.size, self.n = rows, size, crops_per_image
        self._cache = (None, None, None, None)
        self.items = [(i, j) for i in range(len(rows)) for j in range(crops_per_image)]

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        if self._cache[0] != i:
            img, depth = load_pair(self.rows[i])
            pos = eval_crop_positions(self.rows[i], depth, self.size, self.n)
            self._cache = (i, img, depth, pos)
        return self._cache[1:]

    def __getitem__(self, k):
        i, j = self.items[k]
        img, depth, pos = self._load(i)
        if j >= len(pos):  # image has fewer valid crop positions than requested
            return None
        s = crop_sample(img, depth, self.rows[i], *pos[j], self.size)
        s.update(row_index=i, crop_index=j, x0=pos[j][0], y0=pos[j][1])
        return s


class TrainCrops(IterableDataset):
    """Endless stream of random native-resolution crops from the given rows.

    Each decoded image yields `crops_per_load` crops (decoding a 20 MP JPEG + depth TIFF takes
    ~0.4 s); a shuffle buffer mixes crops from different images.

    wide_range=R > 0: keep only crops whose label blur spans a wide range, log2(p95 / p5) >= R (an
    image with no such crop is skipped). Such crops show sharp and blurred regions side by side."""

    def __init__(self, rows, size=518, crops_per_load=4, min_valid=0.5, flip=True, jitter=0.1,
                 buffer=64, seed=0, wide_range=0.0, blur_k=36000.0, sigma0=0.8):
        self.rows, self.size, self.k, self.min_valid = rows, size, crops_per_load, min_valid
        self.flip, self.jitter, self.buffer, self.seed = flip, jitter, buffer, seed
        self.wide_range, self.blur_k, self.sigma0 = wide_range, blur_k, sigma0

    def _crops(self, rng):
        while True:
            row = self.rows[rng.randrange(len(self.rows))]
            img, depth = load_pair(row)
            H, W = depth.shape
            valid = np.isfinite(depth)
            if self.wide_range > 0:
                lsig = np.log2(label_blur(depth[::8, ::8], row, self.blur_k, self.sigma0))
            got = 0
            for _ in range(self.k * 10):
                x0, y0 = rng.randrange(W - self.size + 1), rng.randrange(H - self.size + 1)
                if valid[y0:y0 + self.size, x0:x0 + self.size].mean() < self.min_valid:
                    continue
                if self.wide_range > 0:
                    w = lsig[y0 // 8:(y0 + self.size) // 8, x0 // 8:(x0 + self.size) // 8]
                    w = w[np.isfinite(w)]
                    if w.size < 100 or np.percentile(w, 95) - np.percentile(w, 5) < self.wide_range:
                        continue
                s = crop_sample(img, depth, row, x0, y0, self.size, flip=self.flip and rng.random() < 0.5)
                if self.jitter:  # mild brightness / contrast jitter; no geometric scaling
                    a = 1 + rng.uniform(-self.jitter, self.jitter)
                    b = rng.uniform(-self.jitter, self.jitter) / 2
                    s["image"] = (s["image"] * a + b).clamp(0, 1)
                yield s
                got += 1
                if got == self.k:
                    break

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000 + random.randrange(1 << 20))
        buf = []
        for s in self._crops(rng):
            buf.append(s)
            if len(buf) >= self.buffer:
                yield buf.pop(rng.randrange(len(buf)))


def resize_intrinsics(row, s, flip=False, width=None):
    """K of the image scaled by s (pixel centres: x' = (x + 0.5) s - 0.5)."""
    fx, fy, cx, cy = (float(row[k]) for k in ("fx", "fy", "cx", "cy"))
    fx, fy, cx, cy = fx * s, fy * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5
    if flip:
        cx = width - 1 - cx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def resize_image(img, short):
    """Whole image shrunk so the short side is `short` px (area averaging); returns image, scale."""
    H, W = img.shape[:2]
    s = short / min(H, W)
    size = (int(round(W * s)), int(round(H * s)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), s


def _cache_paths(row, short, cache=RESIZED_CACHE):
    base = Path(cache) / f"r{short}" / Path(row["color_path"].replace("\\", "/")).with_suffix("")
    return base.with_suffix(".png"), base.parent / (base.name + ".depth.npy")


def make_resized(row, short):
    """Resized image (area averaging) and depth (nearest), computed from the full-resolution files."""
    img, depth = load_pair(row)
    im, s = resize_image(img, short)
    h, w = im.shape[:2]
    return im, cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST), s


def load_resized(row, short, with_depth=True):
    """Resized image uint8, resized depth float32 (or None) and the scale factor. Read from the cache
    (MODEST_RESIZED_CACHE, made by resize_cache.py) when it has this image, else computed. The cache
    holds exactly what make_resized computes (lossless PNG; depth float16 like the source TIFFs)."""
    if RESIZED_CACHE:
        img_p, dep_p = _cache_paths(row, short)
        if img_p.exists() and dep_p.exists():
            im = cv2.cvtColor(cv2.imread(str(img_p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            d = np.load(dep_p).astype(np.float32) if with_depth else None
            return im, d, short / min(int(row["height"]), int(row["width"]))
    if not with_depth:
        im, s = resize_image(load_image(row), short)
        return im, None, s
    return make_resized(row, short)


class TrainResized(IterableDataset):
    """Endless stream of whole resized training images (flip + brightness/contrast jitter)."""

    def __init__(self, rows, short=518, flip=True, jitter=0.1, buffer=16, seed=0):
        self.rows, self.short, self.flip, self.jitter, self.buffer, self.seed = rows, short, flip, jitter, buffer, seed

    def _samples(self, rng):
        while True:
            row = self.rows[rng.randrange(len(self.rows))]
            im, d, s = load_resized(row, self.short)
            h, w = im.shape[:2]
            flip = self.flip and rng.random() < 0.5
            if flip:
                im, d = im[:, ::-1], d[:, ::-1]
            image = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
            if self.jitter:
                a = 1 + rng.uniform(-self.jitter, self.jitter)
                b = rng.uniform(-self.jitter, self.jitter) / 2
                image = (image * a + b).clamp(0, 1)
            yield {"image": image, "depth": torch.from_numpy(np.ascontiguousarray(d)),
                   "K": torch.from_numpy(resize_intrinsics(row, s, flip, w)), "cam": camera(row),
                   "px_scale": torch.tensor(float(s))}

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000 + random.randrange(1 << 20))
        buf = []
        for x in self._samples(rng):
            buf.append(x)
            if len(buf) >= self.buffer:
                yield buf.pop(rng.randrange(len(buf)))


class ValResized(Dataset):
    """Whole resized images with resized (nearest) depth: cheap validation for checkpoint selection."""

    def __init__(self, rows, short=518):
        self.rows, self.short = rows, short

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        im, d, s = load_resized(row, self.short)
        return {"image": torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0,
                "depth": torch.from_numpy(d), "K": torch.from_numpy(resize_intrinsics(row, s)), "cam": camera(row),
                "px_scale": torch.tensor(float(s))}


class EvalResized(Dataset):
    """One item per test image: the resized image and K, plus the full-resolution depth and the
    fixed crop positions (same as EvalCrops), so the upsampled prediction can be scored in the same
    windows as the native-resolution evaluation. Use batch_size=1."""

    def __init__(self, rows, short=518, crop=518, crops_per_image=6):
        self.rows, self.short, self.crop, self.n = rows, short, crop, crops_per_image

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        im, _, s = load_resized(row, self.short, with_depth=False)
        depth = load_depth(row)   # full resolution, for scoring
        return {"image": torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0,
                "K": torch.from_numpy(resize_intrinsics(row, s)), "cam": camera(row), "px_scale": torch.tensor(float(s)),
                "depth_full": torch.from_numpy(depth), "row_index": i,
                "positions": eval_crop_positions(row, depth, self.crop, self.n)}


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {k: torch.stack([b[k] for b in batch]) for k in ("image", "depth", "K", "cam", "px_scale")
           if all(k in b for b in batch)}
    for k in batch[0]:
        if k not in out:
            out[k] = [b[k] for b in batch]
    return out
