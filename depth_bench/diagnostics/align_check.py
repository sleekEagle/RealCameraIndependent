"""
Are image and depth aligned in training crops (also flipped ones)? For each crop, image-gradient energy
is summed over the depth discontinuities (top 3% of the log-depth gradient) for shifts of the depth map
by -15..15 px; aligned crops peak at (0, 0). Crops with no clear peak hit the edge of the search window.

    python -m depth_bench.diagnostics.align_check --fold S2 --images 12
Writes align_check.csv to --out.
"""
import argparse
import random

import cv2
import numpy as np
import pandas as pd

from .. import data
from .common import out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S2")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("align_check", a.out)
    rows = data.select(data.read_manifest(), a.fold, ("train",))
    rng, res, c = random.Random(1), [], 20
    for r in rng.sample(rows, a.images):
        img, depth = data.load_pair(r)
        H, W = depth.shape
        for flip in (False, True):
            for _ in range(50):
                x0, y0 = rng.randrange(W - 518), rng.randrange(H - 518)
                if np.isfinite(depth[y0:y0 + 518, x0:x0 + 518]).mean() > 0.9:
                    break
            s = data.crop_sample(img, depth, r, x0, y0, 518, flip=flip)
            im = s["image"].permute(1, 2, 0).numpy().mean(2).astype(np.float32)
            d = s["depth"].numpy()
            d = np.log(np.nan_to_num(d, nan=np.nanmedian(d))).astype(np.float32)
            ge = np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))
            de = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0), cv2.Sobel(d, cv2.CV_32F, 0, 1))
            de = (de > np.percentile(de, 97)).astype(np.float32)
            if de.sum() < 200:
                continue
            score = {(dy, dx): float((ge[c:-c, c:-c] * de[c + dy:518 - c + dy, c + dx:518 - c + dx]).sum())
                     for dy in range(-15, 16) for dx in range(-15, 16)}
            best = max(score, key=score.get)
            res.append({"view": r["color_path"], "flip": flip, "best_dy": best[0], "best_dx": best[1],
                        "score_best_over_zero": score[best] / score[(0, 0)],
                        "at_window_edge": max(abs(best[0]), abs(best[1])) == 15})
    df = pd.DataFrame(res)
    df.to_csv(out / "align_check.csv", index=False)
    print(df.to_string(index=False))
    print(f"\ncrops {len(df)}; peak within 2 px: {((df.best_dy.abs() <= 2) & (df.best_dx.abs() <= 2)).mean():.0%}; "
          f"no clear peak (window edge): {df.at_window_edge.mean():.0%}")
    print(df.groupby("flip")[["best_dy", "best_dx"]].mean().round(1).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
