"""
scene_dataset.py

PyTorch Dataset for scanning a scene directory of the form:

    <root>/
        fl_<focal_length>mm/
            F<aperture>/
                color/
                    0.jpg
                    1.jpg
                    ...
                depth/
                    0.tiff
                    1.tiff
                    ...

Each color image has a matching depth map with the same filename stem
(e.g. color/12.jpg <-> depth/12.tiff). Every (focal_length, aperture, index)
triplet becomes one dataset sample.

Requires: torch, numpy, Pillow, tifffile, imagecodecs (tifffile needs
imagecodecs to decode the floating-point-predictor compressed depth tiffs).
"""

import glob
import os
import re

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

FL_DIR_RE = re.compile(r"^fl_([\d.]+)mm$", re.IGNORECASE)
APERTURE_DIR_RE = re.compile(r"^F([\d.]+)$", re.IGNORECASE)


class SceneDataset(Dataset):
    """Loads (depth, color, focal_length, aperture) samples from a scene directory.

    Args:
        root_dir: path to the scene directory (e.g. .../outputs/scene4).
        color_transform: optional callable applied to the color PIL.Image.
            Defaults to converting to a float32 CHW tensor scaled to [0, 1].
        depth_transform: optional callable applied to the raw depth numpy
            array (float32, HxW, may contain NaN for invalid pixels).
            Defaults to converting to a float32 1xHxW tensor.
    """

    def __init__(self, root_dir, color_transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.color_transform = color_transform or self._default_color_transform
        self.depth_transform = depth_transform or self._default_depth_transform
        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(f"No samples found under {root_dir}")

    @staticmethod
    def _default_color_transform(img):
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _default_depth_transform(arr):
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

    def _index_samples(self):
        samples = []
        for fl_name in sorted(os.listdir(self.root_dir)):
            fl_match = FL_DIR_RE.match(fl_name)
            fl_dir = os.path.join(self.root_dir, fl_name)
            if not fl_match or not os.path.isdir(fl_dir):
                continue
            focal_length = float(fl_match.group(1))

            for aperture_name in sorted(os.listdir(fl_dir)):
                aperture_match = APERTURE_DIR_RE.match(aperture_name)
                aperture_dir = os.path.join(fl_dir, aperture_name)
                if not aperture_match or not os.path.isdir(aperture_dir):
                    continue
                aperture = float(aperture_match.group(1))

                color_dir = os.path.join(aperture_dir, "color")
                depth_dir = os.path.join(aperture_dir, "depth")
                color_paths = sorted(
                    glob.glob(os.path.join(color_dir, "*.jpg")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
                )
                for color_path in color_paths:
                    stem = os.path.splitext(os.path.basename(color_path))[0]
                    depth_path = os.path.join(depth_dir, f"{stem}.tiff")
                    if not os.path.isfile(depth_path):
                        continue
                    samples.append((color_path, depth_path, focal_length, aperture))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        color_path, depth_path, focal_length, aperture = self.samples[idx]

        color_img = Image.open(color_path).convert("RGB")
        color = self.color_transform(color_img)

        depth_arr = tifffile.imread(depth_path)
        depth = self.depth_transform(depth_arr)

        return (
            depth,
            color,
            torch.tensor(focal_length, dtype=torch.float32),
            torch.tensor(aperture, dtype=torch.float32),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = SceneDataset(r"C:\Users\lahir\Downloads\outputs\scene4")
    print(f"Found {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    depth, color, focal_length, aperture = next(iter(loader))
    print("depth:", depth.shape, depth.dtype)
    print("color:", color.shape, color.dtype)
    print("focal_length:", focal_length)
    print("aperture:", aperture)
