"""
Can stage 1 fit ONE fixed batch of real crops? (camind_corr, log-blur loss, no depth loss, dropout 0,
own loop: AdamW lr 1e-3, no clipping.) Compared with the loss of the best single constant and of the best
constant per crop, so a fit that only finds each crop's level is told apart from learning the blur map.

    python -m depth_bench.diagnostics.fit_one_batch --steps 2000
Writes curve_s<seed>.csv and baselines_s<seed>.csv to --out.
"""
import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch

from .. import data, models
from .common import out_dir, spearman


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="which crops form the batch, and the initial weights")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("fit_one_batch", a.out)
    random.seed(a.seed)   # TrainCrops draws its stream seed from the global generator
    torch.manual_seed(a.seed)
    rows = data.subset(data.select(data.read_manifest(), a.fold, ("train",)), 20)
    batch = data.collate(list(itertools.islice(iter(data.TrainCrops(rows, 518, seed=0)), 4)))
    cam, gt = batch["cam"], batch["depth"]
    m0 = models.build("camind_corr")
    C, s = m0.blur_scale(cam).view(-1, 1, 1), cam[:, 2].view(-1, 1, 1)
    t = torch.sqrt(m0.sigma0 ** 2 + (C * (s - gt).abs() / gt) ** 2)
    v = torch.isfinite(gt) & (gt > 0)
    lt = torch.log(t + m0.LOG_EPS)
    base = {"seed": a.seed, "best_single_constant": float(((lt[v] - lt[v].mean()) ** 2).mean()),
            "best_constant_per_crop": float(np.mean([((lt[k][v[k]] - lt[k][v[k]].mean()) ** 2).mean() for k in range(4)])),
            "F": str(cam[:, 1].tolist())}
    pd.DataFrame([base]).to_csv(out / f"baselines_s{a.seed}.csv", index=False)
    print("log-loss baselines:", base)
    curve = []
    for prec, amp in (("fp32", False), ("bf16", True)):
        torch.manual_seed(a.seed)
        m = models.build("camind_corr", stage1_space="logsigma", stage1_weight=1.0, depth_weight=0.0).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
        for i in range(a.steps + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = models.predict(m, batch)
            loss = m.loss(pred, batch)
            if i % 50 == 0:
                b = m.last_blur.float().abs().detach().cpu()
                curve.append({"precision": prec, "step": i, "loss": loss.item(), "spearman": spearman(b[v], t[v]),
                              "median_rel_err": float(((b[v] - t[v]).abs() / t[v]).median())})
                if i % 250 == 0:
                    print(prec, curve[-1], flush=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
    pd.DataFrame(curve).to_csv(out / f"curve_s{a.seed}.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
