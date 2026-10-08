"""
Pack the files the benchmark needs into one tar per scene, for upload to Google Drive.

Only the colour images and depth maps listed in manifest.csv are packed (36 GB), with the manifest
and the focus-distance file. Tars are not compressed (JPEG and the deflate TIFFs are already
compressed). Copying a few large files from Drive is much faster than many small ones.

    python -m depth_bench.pack_for_colab --out D:\\datasets\\MODEST_colab

In Colab, extract them to the local disk (see colab_run.ipynb):
    tar -xf <drive>/MODEST_colab/Scene1.tar -C /content/MODEST_depth      (paths inside are relative
                                                                           to MODEST_depth)
"""
import argparse
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

from . import data
from .config import FOCUS_CSV, MANIFEST


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\datasets\MODEST_colab")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    by_scene = defaultdict(list)
    for r in data.read_manifest():
        for k in ("color_path", "depth_path"):
            by_scene[int(r["scene"])].append(r[k].replace("\\", "/"))
    need = sum(os.path.getsize(data._path(p)) for ps in by_scene.values() for p in ps)
    free = shutil.disk_usage(out).free
    print(f"{sum(map(len, by_scene.values()))} files, {need / 1e9:.1f} GB; free on target {free / 1e9:.1f} GB")
    if free < need * 1.02:
        raise SystemExit("not enough free space")

    for scene, paths in sorted(by_scene.items()):
        tar_path = out / f"Scene{scene}.tar"
        tmp = tar_path.with_suffix(".tar.part")
        if tar_path.exists():
            print(f"{tar_path.name}: exists, skipped")
            continue
        with tarfile.open(tmp, "w") as tf:
            for p in sorted(paths):
                tf.add(data._path(p), arcname=p)
        tmp.rename(tar_path)  # only complete tars get the final name
        print(f"{tar_path.name}: {len(paths)} files, {tar_path.stat().st_size / 1e9:.1f} GB", flush=True)
    shutil.copy(MANIFEST, out / "manifest.csv")
    if FOCUS_CSV.exists():
        shutil.copy(FOCUS_CSV, out / FOCUS_CSV.name)
    print(f"done: upload the folder {out} to Google Drive")


if __name__ == "__main__":
    main()
