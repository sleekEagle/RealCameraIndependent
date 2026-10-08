"""Paths. Every value can be overridden with an environment variable (e.g. on Kaggle/Colab)."""
import os
from pathlib import Path

DEPTH_ROOT = Path(os.environ.get("MODEST_DEPTH_ROOT", r"D:\datasets\MODEST_depth"))
MANIFEST = Path(os.environ.get("MODEST_MANIFEST", str(DEPTH_ROOT / "manifest.csv")))
CODE_EXT = Path(os.environ.get("DEPTH_BENCH_CODE_EXT", r"D:\code_ext"))   # cloned model repos
RUNS_DIR = Path(os.environ.get("DEPTH_BENCH_RUNS", r"D:\datasets\MODEST_runs"))

# keep large downloads off C: by default
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TORCH_HOME", r"D:\torch_cache")

FOLDS = ("S1", "S2", "S3", "S4", "S5")
TEST_LABELS = ("test_seen", "test_unseenF", "test_unseenfl", "test_both")
EXTRA_LABELS = ("extra_unseenfl", "extra_both")

MIN_DEPTH, MAX_DEPTH = 0.3, 20.0   # metres; outside this range a pixel is not used

# Focus distance per (scene, focal length, camera side), from def_calibration/focus_from_aperture_pairs.py.
# Used only for the blur-binned evaluation; if the file is missing, blur bins are skipped.
FOCUS_CSV = Path(os.environ.get("MODEST_FOCUS_CSV", r"D:\datasets\MODEST_focus\dfocus_aperture_pairs.csv"))
# Blur-diameter bins in pixels for the blur-binned AbsRel
BLUR_BINS = (0, 1, 2, 4, 8, 16, 32, 64, float("inf"))
