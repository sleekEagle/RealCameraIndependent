"""
MODEST_depth data: manifest rows, native-resolution crops, crop-adjusted intrinsics.

No resizing anywhere: resizing would shrink the defocus blur, which this study is about.
A crop keeps the focal length (fx, fy) and moves the principal point by the crop offset.
"""
import csv
import hashlib
import random

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .config import DEPTH_ROOT, FOCUS_CSV, MANIFEST, MAX_DEPTH, MIN_DEPTH


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


def camera(row):
    """Lens settings of an image: [real focal length (m), F-number, focus distance (m, NaN if unknown)]."""
    s = focus_table().get((int(row["scene"]), int(row["fl_mm"]), row["side"]), float("nan"))
    return torch.tensor([float(row["f_true_mm"]) / 1000, float(row["f_number"]), s], dtype=torch.float32)


def _path(rel):
    """Manifest paths are written with backslashes; '/' works on Windows and Linux."""
    return DEPTH_ROOT / rel.replace("\\", "/")


def load_pair(row):
    """Full-resolution RGB uint8 (H, W, 3) and depth float32 (H, W, metres, NaN = invalid)."""
    img = cv2.imread(str(_path(row["color_path"])), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(row["color_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = tifffile.imread(_path(row["depth_path"])).astype(np.float32)
    depth[~np.isfinite(depth) | (depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = np.nan
    return img, depth


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
    ~0.4 s); a shuffle buffer mixes crops from different images."""

    def __init__(self, rows, size=518, crops_per_load=4, min_valid=0.5, flip=True, jitter=0.1,
                 buffer=64, seed=0):
        self.rows, self.size, self.k, self.min_valid = rows, size, crops_per_load, min_valid
        self.flip, self.jitter, self.buffer, self.seed = flip, jitter, buffer, seed

    def _crops(self, rng):
        while True:
            row = self.rows[rng.randrange(len(self.rows))]
            img, depth = load_pair(row)
            H, W = depth.shape
            valid = np.isfinite(depth)
            got = 0
            for _ in range(self.k * 10):
                x0, y0 = rng.randrange(W - self.size + 1), rng.randrange(H - self.size + 1)
                if valid[y0:y0 + self.size, x0:x0 + self.size].mean() < self.min_valid:
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


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {k: torch.stack([b[k] for b in batch]) for k in ("image", "depth", "K", "cam")}
    for k in batch[0]:
        if k not in out:
            out[k] = [b[k] for b in batch]
    return out
