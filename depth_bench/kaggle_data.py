"""
Prepare MODEST data inside a Kaggle notebook and print the paths to use.

The Kaggle dataset is the folder made by pack_for_colab.py (Scene1.tar ... Scene9.tar, manifest.csv,
dfocus_aperture_pairs.csv). Kaggle may keep the tars as files or unpack them on upload, so both are
handled:
  - tar files found: they are extracted to --out (local disk; ~36 GB)
  - already unpacked: for each scene, the folder that contains "Scene<k>/fl_..." is found and linked
    into --out (no copy)
Either way --out ends up as a MODEST_depth root: <out>/Scene<k>/fl_<f>mm/F<N>/{color,depth}/...

    python -m depth_bench.kaggle_data --input /kaggle/input/<dataset> --out /kaggle/tmp/MODEST_depth
"""
import argparse
import csv
import os
import shutil
import tarfile
from pathlib import Path


def find_one(root, name):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under {root}")
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="the Kaggle dataset folder (/kaggle/input/<name>)")
    ap.add_argument("--out", default="/kaggle/tmp/MODEST_depth")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = find_one(a.input, "manifest.csv")
    focus = find_one(a.input, "dfocus_aperture_pairs.csv")
    shutil.copy(manifest, out / "manifest.csv")
    with open(manifest, newline="") as f:
        first = {}  # scene folder name -> one image path in it
        for r in csv.DictReader(f):
            p = r["color_path"].replace("\\", "/")
            first.setdefault(p.split("/")[0], p)

    tars = sorted(Path(a.input).rglob("Scene*.tar"))
    if tars:
        need = sum(t.stat().st_size for t in tars)
        free = shutil.disk_usage(out).free
        print(f"{len(tars)} tar files ({need / 1e9:.1f} GB); free on {out}: {free / 1e9:.1f} GB")
        for t in tars:
            flag = out / f".{t.name}.done"
            if flag.exists():
                continue
            if shutil.disk_usage(out).free < t.stat().st_size * 1.05:
                raise SystemExit(f"not enough disk space to extract {t.name}")
            print(f"extracting {t.name}", flush=True)
            with tarfile.open(t) as tf:
                tf.extractall(out)
            flag.touch()
    else:
        print("no tar files: the dataset is already unpacked; linking scene folders")
        scene_dirs = [d for d in Path(a.input).rglob("Scene*") if d.is_dir()]
        for scene, rel in first.items():
            # the "Scene<k>" folder whose parent holds the manifest's relative paths
            hits = [d for d in scene_dirs if d.name == scene and (d.parent / rel).exists()]
            if not hits:
                raise SystemExit(f"{rel} not found under {a.input}")
            src = hits[0]
            link = out / scene
            if not link.exists():
                os.symlink(src, link, target_is_directory=True)
            print(f"  {scene} -> {src}")

    missing = [rel for rel in first.values() if not (out / rel).exists()]
    if missing:
        raise SystemExit(f"files missing after preparing the data, e.g. {missing[:3]}")
    print(f"MODEST_DEPTH_ROOT={out}")
    print(f"MODEST_FOCUS_CSV={focus}")


if __name__ == "__main__":
    main()
