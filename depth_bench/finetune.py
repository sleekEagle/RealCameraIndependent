"""
Fine-tune one model on one setting fold (same recipe for every model).
Models without pretrained weights (camind, camind_nocorr) are trained from scratch with the same
loop: all their weights are trained, with --lr_head, and with their own loss (model.loss).

Recipe: train the head and the last `--unfreeze` fraction of encoder blocks; AdamW with a
lower learning rate for the encoder; linear warm-up then cosine decay; SILog loss on metric
depth; mixed precision; gradient accumulation; native-resolution random crops (no resizing,
horizontal flip + mild brightness/contrast jitter). The training budget is a number of crop
samples, so every model sees the same amount of data. Validation (AbsRel on fixed crops of
the fold's `val` views) picks the best checkpoint. Runs can be resumed.

Example (depthbench env):
    python -m depth_bench.finetune --model da3 --fold S1 --samples 40000 --batch 2 --accum 8
Writes <RUNS_DIR>/finetune/<model>_<fold>[_<tag>]/ : last.pt, best.pt, log.csv, config.json,
and done.json when finished.

Time limit (e.g. Kaggle's 12 h sessions): if DEPTH_BENCH_DEADLINE (unix time) is set, training saves
last.pt and exits with code 75 once that time is reached; the next run resumes from last.pt.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, models
from .config import FOLDS, RUNS_DIR
from .metrics import depth_metrics, silog_loss
from .train_utils import amp_dtype, autocast_ctx, load_trainable, trainable_state


def validate(model, loader):
    model.eval()
    vals = []
    with torch.no_grad(), autocast_ctx():
        for b in loader:
            if b is None:
                continue
            pred = models.predict(model, b)
            vals += [m["absrel"] for m in depth_metrics(pred.float().cpu(), b["depth"]) if "absrel" in m]
    return float(np.mean(vals)) if vals else float("nan")


def set_train_mode(model):
    """Frozen encoder in eval mode (no dropout / stochastic depth); trained heads in train mode."""
    model.eval()
    for m in model.head_modules():
        m.train()


EXIT_TIME_LIMIT = 75
DEADLINE = float(os.environ.get("DEPTH_BENCH_DEADLINE", 0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--fold", required=True, choices=FOLDS)
    ap.add_argument("--samples", type=int, default=40000, help="training budget in crop samples")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--unfreeze", type=float, default=0.25, help="fraction of encoder blocks to train")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--val_every", type=int, default=250, help="optimizer steps between validations")
    ap.add_argument("--val_images", type=int, default=120)
    ap.add_argument("--val_crops", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    run = RUNS_DIR / "finetune" / f"{a.model}_{a.fold}{'_' + a.tag if a.tag else ''}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))

    rows = data.read_manifest()
    train_rows = data.select(rows, a.fold, ("train",))
    val_rows = data.select(rows, a.fold, ("val",))
    val_rows = random.Random(0).sample(val_rows, min(a.val_images, len(val_rows)))
    print(f"{a.model} {a.fold}: {len(train_rows)} train images, {len(val_rows)} val images", flush=True)

    model = models.build(a.model).cuda()
    enc_p, head_p = models.set_trainable(model, a.unfreeze)
    n_tr = sum(p.numel() for p in enc_p + head_p)
    print(f"trainable params: {n_tr / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    groups = [{"params": head_p, "lr": a.lr_head}] + ([{"params": enc_p, "lr": a.lr_enc}] if enc_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in groups]
    steps = max(1, a.samples // (a.batch * a.accum))
    warm = max(1, int(steps * a.warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype() == torch.float16)

    step, best, log_rows = 0, float("inf"), []
    if (run / "last.pt").exists():  # resume
        ck = load_trainable(model, run / "last.pt")
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed at step {step} (best val {best:.4f})", flush=True)

    train_dl = DataLoader(data.TrainCrops(train_rows, a.crop, seed=a.seed + step), batch_size=a.batch,
                          num_workers=a.workers, collate_fn=data.collate, persistent_workers=a.workers > 0)
    val_dl = DataLoader(data.EvalCrops(val_rows, a.crop, a.val_crops), batch_size=a.val_crops,
                        num_workers=min(2, a.workers), collate_fn=data.collate)
    if step == 0:
        best = validate(model, val_dl)
        if not math.isfinite(best):
            raise RuntimeError("validation AbsRel is not finite before training (forward pass overflows? "
                               f"mixed precision: {amp_dtype()})")
        print(f"step 0 ({'zero-shot' if getattr(model, 'pretrained', True) else 'random init'}) val AbsRel {best:.4f}", flush=True)
        log_rows.append({"step": 0, "loss": "", "val_absrel": best, "time_s": 0})
        # if training never beats the starting weights, these are the best ones (else best.pt is missing)
        torch.save({"model": trainable_state(model), "step": 0, "val_absrel": best}, run / "best.pt")

    it = iter(train_dl)
    t0 = time.time()
    run_loss = []
    bad = 0  # optimizer steps in a row with a non-finite loss
    while step < steps:
        set_train_mode(model)
        mult = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * mult
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            b = next(it)
            with autocast_ctx():
                pred = models.predict(model, b)
            if hasattr(model, "loss"):  # model's own loss (camind: depth MSE)
                loss = model.loss(pred, b) / a.accum
            else:
                loss = silog_loss(pred.float(), b["depth"].cuda(non_blocking=True)) / a.accum
            scaler.scale(loss).backward()
            run_loss.append(loss.item() * a.accum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(enc_p + head_p, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if DEADLINE and time.time() > DEADLINE and step < steps:
            torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "step": step, "best": best},
                       run / "last.pt")
            if log_rows:
                with open(run / "log.csv", "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["step", "loss", "val_absrel", "time_s"])
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerows(log_rows)
            print(f"time limit reached: saved last.pt at step {step}/{steps}; re-run to resume", flush=True)
            sys.exit(EXIT_TIME_LIMIT)
        if not all(math.isfinite(x) for x in run_loss[-a.accum:]):
            bad += 1  # fp16 loss scaling may skip a few steps; a long run of them means overflow
            if bad >= 20:
                raise RuntimeError(f"loss not finite for 20 steps in a row (step {step}, mixed precision: {amp_dtype()})")
        else:
            bad = 0
        if step % 10 == 0:
            print(f"step {step}/{steps} loss {np.mean(run_loss[-10 * a.accum:]):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step, GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if step % a.val_every == 0 or step == steps:
            v = validate(model, val_dl)
            if not math.isfinite(v):
                raise RuntimeError(f"validation AbsRel is not finite at step {step} (mixed precision: {amp_dtype()})")
            log_rows.append({"step": step, "loss": float(np.mean(run_loss)), "val_absrel": v, "time_s": round(time.time() - t0)})
            run_loss = []
            state = trainable_state(model)
            if v < best:
                best = v
                torch.save({"model": state, "step": step, "val_absrel": v}, run / "best.pt")
            torch.save({"model": state, "opt": opt.state_dict(), "step": step, "best": best}, run / "last.pt")
            print(f"step {step}: val AbsRel {v:.4f} (best {best:.4f})", flush=True)
            with open(run / "log.csv", "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", "loss", "val_absrel", "time_s"])
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(log_rows)
            log_rows = []
    (run / "done.json").write_text(json.dumps({"best_val_absrel": best, "steps": step,
                                               "time_s": round(time.time() - t0)}))
    print(f"done: best val AbsRel {best:.4f}; checkpoints in {run}")
    return run


if __name__ == "__main__":
    main()
