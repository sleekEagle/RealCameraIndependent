"""
The dropout / max-pool train-eval mismatch of Camind's AENet (Dropout(0.2) right before each max-pool).

1. Toy: expected max of a 2x2 window after dropout (kept values x 1/0.8) vs the eval-mode max.
2. Network: max |train-mode output - eval-mode output| on one input, with dropout 0 and 0.2.
3. Trained checkpoints (built with dropout 0.2, as they were trained): depth metrics on fold val crops in
   eval mode (as validated / benchmarked) and in train mode (as trained; dropout noise, 1 and 8 samples).

    python -m depth_bench.diagnostics.dropout_mismatch \
        --ckpt camind_S1=camind=D:/.../finetune/camind_S1/best.pt camind_corr_S1=camind_corr=D:/.../best.pt
Writes toy.csv, network.csv, checkpoints.csv to --out.
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch

from .. import data, models
from ..metrics import depth_metrics
from ..train_utils import load_trainable
from .common import out_dir


def toy(p=0.2, n=200000):
    rng = np.random.default_rng(0)
    res = []
    for w in ([4, 4, 4, 4], [4, 3, 2, 1], [4, 0, 0, 0]):
        x = np.array(w, float)
        keep = rng.random((n, 4)) > p
        res.append({"window": str(w), "eval_max": x.max(), "train_mean_max": (keep * x / (1 - p)).max(1).mean()})
    df = pd.DataFrame(res)
    df["ratio"] = df.train_mean_max / df.eval_max
    return df


def network():
    x = torch.rand(1, 3, 64, 64).cuda()
    cam = torch.tensor([[0.05, 5.0, 2.0]]).cuda()
    res = []
    for d in (0.0, 0.2):
        m = models.build("camind_corr", dropout=d).cuda()
        torch.manual_seed(0)
        m.train()
        a = m(x, None, cam)
        m.eval()
        b = m(x, None, cam)
        res.append({"dropout": d, "max_abs_train_minus_eval": float((a - b).abs().max())})
    return pd.DataFrame(res)


def checkpoints(specs, fold, images):
    rows = random.Random(0).sample(data.select(data.read_manifest(), fold, ("val",)), images)
    ds = data.EvalCrops(rows, 518, 2)
    res = []
    for spec in specs:
        name, model_name, path = spec.split("=", 2)
        m = models.build(model_name, dropout=0.2).cuda()
        load_trainable(m, path)
        acc = {"eval": [], "train": [], "train x8": []}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(ds)):
                b = data.collate([ds[i]])
                if b is None:
                    continue
                m.eval()
                pe = models.predict(m, b).float().cpu()
                m.train()
                ps = torch.stack([models.predict(m, b).float().cpu() for _ in range(8)])
                for k, p in (("eval", pe), ("train", ps[0]), ("train x8", ps.mean(0))):
                    r = depth_metrics(p, b["depth"])[0]
                    if "absrel" in r:
                        acc[k].append((r["absrel"], r["si_absrel"], r["scale"]))
        for k, v in acc.items():
            v = np.array(v)
            res.append({"checkpoint": name, "mode": k, "crops": len(v), "absrel": v[:, 0].mean(),
                        "si_absrel": v[:, 1].mean(), "median_scale": np.median(v[:, 2])})
    return pd.DataFrame(res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=model=path (trained with dropout 0.2)")
    ap.add_argument("--fold", default="S1")
    ap.add_argument("--images", type=int, default=30, help="val views (2 fixed crops each)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("dropout_mismatch", a.out)
    for name, df in (("toy", toy()), ("network", network())) + \
            ((("checkpoints", checkpoints(a.ckpt, a.fold, a.images)),) if a.ckpt else ()):
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"== {name}\n{df.round(4).to_string(index=False)}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
