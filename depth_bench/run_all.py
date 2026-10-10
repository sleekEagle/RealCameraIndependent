"""
Run the whole benchmark: zero-shot evaluation, then for every fold and model training and evaluation.
Safe to re-run (after a disconnect or a new Kaggle session): finished steps are skipped and stopped
training runs resume from last.pt.

Jobs, in this order:
  1. zero-shot evaluation of the pretrained models, once on all test views (results can be grouped by
     any fold afterwards): da2, unidepth, unidepth_noK, metric3d (official canonical focal), da3
  2. for each fold, for each model: train (fine-tune, or from scratch for camind), then evaluate
     the best checkpoint on that fold's test groups

    python -m depth_bench.run_all                     # everything
    python -m depth_bench.run_all --folds S1 S2 --models da2 camind
    python -m depth_bench.run_all --dry_run           # show what is done and what would run

Several processes can share the work, e.g. one per GPU (Kaggle T4 x2):
    CUDA_VISIBLE_DEVICES=0 python -m depth_bench.run_all --clear_locks &
    CUDA_VISIBLE_DEVICES=1 python -m depth_bench.run_all &
A process claims a job with a lock file (<RUNS_DIR>/locks/); others skip it. --clear_locks removes
locks left by a killed session (use it only in the first process, before the others start).

--hours: time limit for this session. No job is started too close to it, and training saves last.pt
and stops at the limit, so the next session continues from there.

Each step runs in its own process (frees GPU memory); its full output goes to <RUNS_DIR>/logs/.
"""
import argparse
import os
import subprocess
import sys
import time

from .config import FOLDS, RUNS_DIR
from .finetune import EXIT_TIME_LIMIT

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


def claim(job):
    """Take a job with an exclusive lock file; False if another process holds it."""
    lock = RUNS_DIR / "locks" / f"{job}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        lock.write_text(f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except FileExistsError:
        return False


def release(job):
    (RUNS_DIR / "locks" / f"{job}.lock").unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=list(FOLDS))
    ap.add_argument("--models", nargs="*", default=list(TRAIN), help=f"from {list(TRAIN)}")
    ap.add_argument("--zero_shot", nargs="*", default=ZERO_SHOT, help="models for zero-shot ('none' to skip)")
    ap.add_argument("--ft_samples", type=int, default=40000, help="fine-tuning budget (crops)")
    ap.add_argument("--scratch_samples", type=int, default=200000, help="from-scratch budget (crops)")
    ap.add_argument("--crops", type=int, default=6, help="fixed evaluation crops per test image")
    # data loading decodes 20 MP images (~0.4 s each), so fast GPUs need several workers; on Windows
    # many workers can exhaust the paging file, so fewer there
    ap.add_argument("--workers", type=int, default=min(8 if os.name != "nt" else 4, os.cpu_count() or 2))
    ap.add_argument("--hours", type=float, default=0, help="session time limit in hours (0 = none)")
    ap.add_argument("--keep_last", action="store_true", help="keep last.pt of finished runs (default: delete)")
    ap.add_argument("--clear_locks", action="store_true", help="remove locks of a previous (killed) session")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    ev = ["--crops", str(a.crops), "--workers", str(a.workers)]
    t0 = time.time()
    if a.hours:
        # training stops 10 min before the limit, to leave time for saving
        os.environ["DEPTH_BENCH_DEADLINE"] = str(t0 + a.hours * 3600 - 600)
    if a.clear_locks:
        for f in (RUNS_DIR / "locks").glob("*.lock"):
            f.unlink()

    def left_h():
        return a.hours - (time.time() - t0) / 3600 if a.hours else float("inf")

    jobs = [("zs", m, None) for m in a.zero_shot if m != "none"]
    jobs += [("train", name, fold) for fold in a.folds for name in a.models]
    for kind, name, fold in jobs:
        if kind == "zs":
            job, ev_dir = f"zs_{name}", RUNS_DIR / "eval" / f"zs_{name}"
        else:
            m = TRAIN[name]
            job, ev_dir, rd = f"{m}_{fold}", RUNS_DIR / "eval" / f"ft_{m}_{fold}", RUNS_DIR / "finetune" / f"{m}_{fold}"
        if (ev_dir / "summary.csv").exists():
            print(f"{job}: done")
            continue
        if left_h() < 0.75:  # an evaluation takes up to ~30 min on a T4
            print(f"less than 45 min left in this session: not starting {job}")
            break
        if not a.dry_run and not claim(job):
            print(f"{job}: taken by another process")
            continue
        try:
            if kind == "zs":
                code, out = run(["depth_bench.evaluate", "--model", name, *ev, "--out", job], job, a.dry_run)
                if code:
                    print(f"   zero-shot {name} FAILED (exit {code}):\n{out}", flush=True)
                continue
            if not (rd / "done.json").exists():
                scratch = name in SCRATCH
                tr = ["depth_bench.finetune", "--model", m, "--fold", fold,
                      "--samples", str(a.scratch_samples if scratch else a.ft_samples), "--workers", str(a.workers)]
                # same 16 crops per optimizer step for all; the small CNN fits 8 per batch
                batch, accum = (8, 2) if scratch else (2, 8)
                code, out = run(tr + ["--batch", str(batch), "--accum", str(accum)], f"train_{job}", a.dry_run)
                if code and "out of memory" in out.lower():
                    print("   out of GPU memory, retrying with batch 1", flush=True)
                    code, out = run(tr + ["--batch", "1", "--accum", str(batch * accum)], f"train_{job}", a.dry_run)
                if code == EXIT_TIME_LIMIT:
                    print(f"   {job}: session time limit reached; training continues next session", flush=True)
                    break
                if code:
                    print(f"   training {name} {fold} FAILED (exit {code}):\n{out}", flush=True)
                    continue
            if not a.keep_last and not a.dry_run:
                (rd / "last.pt").unlink(missing_ok=True)  # only needed to resume; saves disk (Kaggle: 20 GB)
            if left_h() < 0.75:
                print(f"less than 45 min left in this session: evaluation of {job} next session")
                break
            code, out = run(["depth_bench.evaluate", "--model", m, "--fold", fold, *ev, "--ckpt", str(rd / "best.pt"),
                             "--out", f"ft_{job}"], f"eval_{job}", a.dry_run)
            if code:
                print(f"   evaluation {name} {fold} FAILED (exit {code}):\n{out}", flush=True)
        finally:
            if not a.dry_run:
                release(job)
            print(f"   elapsed {(time.time() - t0) / 3600:.1f} h", flush=True)
    print("run_all finished")


if __name__ == "__main__":
    main()
