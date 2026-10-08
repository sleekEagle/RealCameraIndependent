"""
Run the whole benchmark: zero-shot evaluation, then for every fold and model training and evaluation.
Safe to re-run after a disconnect (e.g. Colab): finished steps are skipped and stopped training runs
resume from last.pt.

Order:
  1. zero-shot evaluation of the pretrained models, once on all test views (results can be grouped by
     any fold afterwards): da2, unidepth, unidepth_noK, metric3d (official canonical focal), da3
  2. for each fold, for each model: train (fine-tune, or from scratch for camind), then evaluate
     the best checkpoint on that fold's test groups

    python -m depth_bench.run_all                     # everything
    python -m depth_bench.run_all --folds S1 S2 --models da2 camind
    python -m depth_bench.run_all --dry_run           # show what is done and what would run

Each step runs in its own process (frees GPU memory); its full output goes to <RUNS_DIR>/logs/.
"""
import argparse
import os
import subprocess
import sys
import time

from .config import FOLDS, RUNS_DIR

ZERO_SHOT = ["da2", "unidepth", "unidepth_noK", "metric3d", "da3"]
# benchmark name -> model name used for training (Metric3D: canonical focal 6000 px, see models.py)
TRAIN = {"da2": "da2", "unidepth": "unidepth", "metric3d": "metric3d_c6000", "da3": "da3", "camind": "camind"}
SCRATCH = {"camind"}


def _show(line, keys):
    """Key lines, plus training progress every 100 optimizer steps ("step 300/2500 loss ...")."""
    if any(k in line for k in keys):
        return True
    w = line.split()
    if len(w) > 1 and w[0] == "step" and "/" in w[1]:
        n = w[1].split("/")[0]
        return n.isdigit() and int(n) % 100 == 0
    return False


def run(args, log_name, dry):
    """Run `python -m <args>`; full output to logs/<log_name>.log, key lines to the console."""
    print(f">> {' '.join(args)}", flush=True)
    if dry:
        return 0, ""
    log = RUNS_DIR / "logs" / f"{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    keys = ("absrel=", "val AbsRel", "trainable params", "done:", "Error", "error", "crops in", "resumed")
    tail = []
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        p = subprocess.Popen([sys.executable, "-W", "ignore", "-u", "-m", *args], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            f.flush()
            tail = (tail + [line])[-50:]
            if _show(line, keys):
                print("   " + line.rstrip(), flush=True)
        p.wait()
    return p.returncode, "".join(tail)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=list(FOLDS))
    ap.add_argument("--models", nargs="*", default=list(TRAIN), help=f"from {list(TRAIN)}")
    ap.add_argument("--zero_shot", nargs="*", default=ZERO_SHOT, help="models for zero-shot ('none' to skip)")
    ap.add_argument("--ft_samples", type=int, default=40000, help="fine-tuning budget (crops)")
    ap.add_argument("--scratch_samples", type=int, default=200000, help="from-scratch budget (crops)")
    ap.add_argument("--crops", type=int, default=6, help="fixed evaluation crops per test image")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    ev = ["--crops", str(a.crops), "--workers", str(a.workers)]
    t0 = time.time()

    # 1. zero-shot
    for m in [z for z in a.zero_shot if z != "none"]:
        if (RUNS_DIR / "eval" / f"zs_{m}" / "summary.csv").exists():
            print(f"zero-shot {m}: done")
            continue
        code, out = run(["depth_bench.evaluate", "--model", m, *ev, "--out", f"zs_{m}"], f"zs_{m}", a.dry_run)
        if code:
            print(f"   zero-shot {m} FAILED (exit {code}):\n{out}", flush=True)

    # 2. train + evaluate per fold
    for fold in a.folds:
        for name in a.models:
            m = TRAIN[name]
            rd = RUNS_DIR / "finetune" / f"{m}_{fold}"
            ev_dir = RUNS_DIR / "eval" / f"ft_{m}_{fold}"
            if (ev_dir / "summary.csv").exists():
                print(f"{fold} {name}: done")
                continue
            if not (rd / "done.json").exists():
                scratch = name in SCRATCH
                tr = ["depth_bench.finetune", "--model", m, "--fold", fold,
                      "--samples", str(a.scratch_samples if scratch else a.ft_samples), "--workers", str(a.workers)]
                # same 16 crops per optimizer step for all; the small CNN fits 8 per batch
                batch, accum = (8, 2) if scratch else (2, 8)
                code, out = run(tr + ["--batch", str(batch), "--accum", str(accum)], f"train_{m}_{fold}", a.dry_run)
                if code and "out of memory" in out.lower():
                    print("   out of GPU memory, retrying with batch 1", flush=True)
                    code, out = run(tr + ["--batch", "1", "--accum", str(batch * accum)], f"train_{m}_{fold}", a.dry_run)
                if code:
                    print(f"   training {name} {fold} FAILED (exit {code}):\n{out}", flush=True)
                    continue
            ck = rd / "best.pt"
            code, out = run(["depth_bench.evaluate", "--model", m, "--fold", fold, *ev, "--ckpt", str(ck),
                             "--out", f"ft_{m}_{fold}"], f"eval_{m}_{fold}", a.dry_run)
            if code:
                print(f"   evaluation {name} {fold} FAILED (exit {code}):\n{out}", flush=True)
            print(f"   elapsed {(time.time() - t0) / 3600:.1f} h", flush=True)
    print("run_all finished")


if __name__ == "__main__":
    main()
