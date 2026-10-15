"""
Build the cache of resized images for the resized setting, so training does not decode a 20 MP
JPEG and depth TIFF for every sample.

For every manifest image: the image shrunk to a short side of --short px (area averaging) as a
lossless PNG, and the depth shrunk with nearest-neighbour sampling as float16 .npy (the source TIFFs
are float16, so nothing is lost). This is exactly what data.make_resized computes; data.load_resized
reads it when MODEST_RESIZED_CACHE points to --out. Images already cached are skipped; files are
written under a temporary name and renamed, so an interrupted run leaves no broken files.

    python -m depth_bench.resize_cache --out /workspace/MODEST_resized --short 518
    MODEST_RESIZED_CACHE=/workspace/MODEST_resized python -m depth_bench.run_all --resize 518
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from . import data


def build_one(args):
    row, short, out = args
    img_p, dep_p = data._cache_paths(row, short, out)
    if img_p.exists() and dep_p.exists():
        return 0
    img_p.parent.mkdir(parents=True, exist_ok=True)
    im, d, _ = data.make_resized(row, short)
    tmp = img_p.with_name(img_p.stem + ".tmp.png")
    cv2.imwrite(str(tmp), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    os.replace(tmp, img_p)
    tmp = dep_p.with_name(dep_p.name + ".tmp.npy")
    np.save(tmp, d.astype(np.float16))
    os.replace(tmp, dep_p)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", type=int, default=518)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (for testing)")
    ap.add_argument("--fold", help="only this fold's train and val images (to save disk space)")
    a = ap.parse_args(argv)
    rows = data.read_manifest()
    if a.fold:
        rows = data.select(rows, a.fold, ("train", "val"))
    if a.limit:
        rows = rows[:a.limit]
    t0, done = time.time(), 0
    with Pool(a.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(build_one, [(r, a.short, a.out) for r in rows], chunksize=4)):
            done += n
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(rows)} images, {time.time() - t0:.0f}s", flush=True)
    print(f"resize cache: {done} images written, {len(rows) - done} already there, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
