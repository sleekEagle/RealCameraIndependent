"""
def_calibration/show_depth.py -- interactively display a depth map for a quick visual check.

Kept separate from focaldist_estimation.py, which forces the non-interactive Agg backend for
its headless batch plot-saving -- Agg can't open a window, so this needs its own script using
matplotlib's normal (interactive) backend.

    python def_calibration/show_depth.py D:\\datasets\\MODEST_processed\\scene4\\fl_28mm\\F2.8\\depth\\L\\0.tiff
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def show_depth_map(depth, title: str = "depth map", cmap: str = "viridis"):
    """depth: a path to a depth tiff, or an already-loaded (H,W) array -- displays it in a
    popup window with invalid pixels (non-finite or <= 0) shown in gray."""
    if isinstance(depth, (str, Path)):
        title = title if title != "depth map" else str(depth)
        depth = tifffile.imread(depth).astype(np.float64)
        if depth.ndim == 3:  # occasional (H,W,1) tiffs
            depth = depth[..., 0]

    valid = np.isfinite(depth) & (depth > 0)
    masked = np.ma.masked_where(~valid, depth)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(masked, cmap=cmap_obj)
    fig.colorbar(im, ax=ax, label="depth")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_depth_map(sys.argv[1])
