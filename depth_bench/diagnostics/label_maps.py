"""
Figures of the blur labels and the texture mask.
  label_maps.png: F2.8 / F5 image windows next to their label blur map (log scale), to see that the label
                  is large where the image is visibly soft.
  texture_mask.png: training crops and the pixels the stage-1 mask keeps (local contrast >= threshold).

    python -m depth_bench.diagnostics.label_maps --fold S2
"""
import argparse
import itertools
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import data
from ..models import CamindW
from .common import label_sigma, out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S2")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--window", type=int, default=1036, help="window size of the label figure (native px)")
    ap.add_argument("--mask", type=float, default=0.02, help="texture-mask threshold (as --stage1_mask)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("label_maps", a.out)
    rows = data.select(data.read_manifest(), a.fold, ("train",))

    big = [r for r in rows if r["f_number"] in ("2.8", "5.0")]
    rng, shown, Wn = random.Random(3), [], a.window
    while len(shown) < a.n:
        r = rng.choice(big)
        img, depth = data.load_pair(r)
        sig = label_sigma(depth, r)
        H, W = depth.shape
        for _ in range(300):
            x0, y0 = rng.randrange(W - Wn), rng.randrange(H - Wn)
            w = sig[y0:y0 + Wn, x0:x0 + Wn]
            if np.isfinite(w).mean() > 0.9 and np.nanpercentile(w, 10) < 1.5 and np.nanpercentile(w, 90) > 5:
                shown.append((img[y0:y0 + Wn, x0:x0 + Wn], w, f"{r['color_path']}  F{r['f_number']}"))
                break
    fig, ax = plt.subplots(a.n, 2, figsize=(10, 5 * a.n))
    for i, (im, w, t) in enumerate(shown):
        ax[i, 0].imshow(im)
        ax[i, 0].set_title(t, fontsize=8)
        h = ax[i, 1].imshow(np.log2(w), cmap="magma", vmin=np.log2(0.8), vmax=np.log2(16))
        ax[i, 1].set_title("label blur sigma (px, log scale)", fontsize=8)
        cb = fig.colorbar(h, ax=ax[i, 1], fraction=0.046)
        cb.set_ticks(np.log2([1, 2, 4, 8, 16]))
        cb.set_ticklabels(["1", "2", "4", "8", "16"])
        for x in ax[i]:
            x.axis("off")
    fig.tight_layout()
    fig.savefig(out / "label_maps.png", dpi=70)

    ds = data.TrainCrops(random.Random(2).sample(big, 6), 518, seed=2, jitter=0)
    b = data.collate(list(itertools.islice(iter(ds), a.n)))
    m = CamindW.texture_mask(b["image"], a.mask)
    fig, ax = plt.subplots(2, a.n, figsize=(4 * a.n, 8))
    for k in range(a.n):
        im = b["image"][k].permute(1, 2, 0).numpy()
        ax[0, k].imshow(im)
        ax[0, k].set_title(f"F{float(b['cam'][k, 1]):g}")
        o = im.copy()
        o[~m[k].numpy()] *= 0.25
        ax[1, k].imshow(o)
        ax[1, k].set_title(f"kept by the mask: {m[k].float().mean():.0%}")
        for x in ax[:, k]:
            x.axis("off")
    fig.tight_layout()
    fig.savefig(out / "texture_mask.png", dpi=70)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
