"""
Pilot: check the whole pipeline for every baseline on one fold with a tiny budget.

For each model: zero-shot evaluation on a small fixed test subset, a short fine-tune, and
evaluation of the fine-tuned checkpoint on the same subset. Each step runs in its own
process so GPU memory is released between models. If fine-tuning runs out of GPU memory at
batch 2, it is retried with batch 1 (same number of crops per step via accumulation).

    python -m depth_bench.pilot                       # all models, fold S1
    python -m depth_bench.pilot --models da2 da3 --fold S2
Writes <RUNS_DIR>/pilot_summary.csv.
"""
import argparse
import csv
import subprocess
import sys

from .config import RUNS_DIR

ALL = ["da2", "unidepth", "metric3d_c6000", "da3", "camind"]
SCRATCH = {"camind"}


def run(args):
    print(">>", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", *args], capture_output=True, text=True)
    out = p.stdout + p.stderr
    for line in out.splitlines():
        if any(k in line for k in ("absrel=", "val AbsRel", "trainable params", "s/step", "done:", "Error", "error", "crops in")):
            print("   " + line.strip(), flush=True)
    return p.returncode, out


def summary(out_name, fold):
    path = RUNS_DIR / "eval" / out_name / "summary.csv"
    if not path.exists():
        return {}
    return {r["value"]: r for r in csv.DictReader(open(path)) if r["group"] == fold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ALL)
    ap.add_argument("--fold", default="S1")
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--samples", type=int, default=400)
    a = ap.parse_args()
    ev = ["--fold", a.fold, "--max_images", str(a.images), "--crops", "2"]
    results = []
    for m in a.models:
        if m not in SCRATCH:  # no zero-shot for models trained from scratch
            run(["depth_bench.evaluate", "--model", m, *ev, "--out", f"pilot_zs_{m}_{a.fold}"])
            results.append(("zero-shot", m, summary(f"pilot_zs_{m}_{a.fold}", a.fold)))
        if m == "unidepth":
            run(["depth_bench.evaluate", "--model", "unidepth_noK", *ev, "--out", f"pilot_zs_unidepth_noK_{a.fold}"])
            results.append(("zero-shot", "unidepth_noK", summary(f"pilot_zs_unidepth_noK_{a.fold}", a.fold)))
        ft = ["depth_bench.finetune", "--model", m, "--fold", a.fold, "--samples", str(a.samples), "--val_every", "25",
              "--val_images", "20", "--val_crops", "2", "--workers", "2", "--tag", "pilot"]
        code, out = run(ft + ["--batch", "2", "--accum", "2"])
        if code != 0 and "out of memory" in out.lower():
            print("   out of GPU memory at batch 2, retrying with batch 1", flush=True)
            code, out = run(ft + ["--batch", "1", "--accum", "4"])
        rd = RUNS_DIR / "finetune" / f"{m}_{a.fold}_pilot"
        ck = rd / "best.pt" if (rd / "best.pt").exists() else rd / "last.pt"
        if code == 0 and ck.exists():
            run(["depth_bench.evaluate", "--model", m, *ev, "--ckpt", str(ck), "--out", f"pilot_ft_{m}_{a.fold}"])
            results.append(("fine-tuned", m, summary(f"pilot_ft_{m}_{a.fold}", a.fold)))
        else:
            print(f"   fine-tuning {m} failed:\n" + out[-2000:], flush=True)

    path = RUNS_DIR / "pilot_summary.csv"
    cols = ["stage", "model", "fold", "group", "n_crops", "absrel", "absrel_med", "rmse", "delta1", "delta2", "delta3", "silog", "si_absrel", "scale"]
    done = {m for _, m, _ in results}
    old = [r for r in csv.DictReader(open(path))] if path.exists() else []
    old = [r for r in old if not (r["model"] in done and r.get("fold", a.fold) == a.fold)]  # replace re-run models
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)
        for stage, m, s in results:
            for g, r in s.items():
                w.writerow({"stage": stage, "model": m, "fold": a.fold, "group": g, **{k: r.get(k, "") for k in cols[4:]}})
    print(f"pilot summary: {RUNS_DIR / 'pilot_summary.csv'}")


if __name__ == "__main__":
    main()
