"""
Stage 1 alone on a stream of random native crops from 20 training views (camind_corr, log-blur loss),
checked on 20 fixed crops of the same views: pixel and crop-median Spearman with the label blur, and the
spread of the predicted crop medians. Step 0 is the untrained network (it already ranks some blur).

    python -m depth_bench.diagnostics.stream_fit --lr 1e-3 --dropout 0 --jitter 0 --steps 800
Writes curve_<tag>.csv to --out.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from .. import data, models
from .common import out_dir, spearman


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="S1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("stream_fit", a.out)
    tag = f"lr{a.lr:g}_drop{a.dropout:g}_jit{a.jitter:g}"
    torch.manual_seed(0)
    rows = data.subset(data.select(data.read_manifest(), a.fold, ("train",)), 20)
    ev = data.collate([data.EvalCrops(rows, 518, 1)[i] for i in range(20)])
    it = iter(torch.utils.data.DataLoader(data.TrainCrops(rows, 518, seed=1, jitter=a.jitter), batch_size=4,
                                          num_workers=0, collate_fn=data.collate))
    m = models.build("camind_corr", stage1_space="logsigma", stage1_weight=1.0, depth_weight=0.0,
                     dropout=a.dropout).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    cam, gt = ev["cam"], ev["depth"]
    C, s = m.blur_scale(cam).view(-1, 1, 1), cam[:, 2].view(-1, 1, 1)
    t = torch.sqrt(m.sigma0 ** 2 + (C * (s - gt).abs() / gt) ** 2)
    v = torch.isfinite(t)
    tm = torch.stack([t[k][v[k]].median() for k in range(20)])
    curve, losses = [], []
    for i in range(a.steps + 1):
        if i % 100 == 0:
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                b = torch.cat([(models.predict(m, {k: x[j:j + 4] for k, x in ev.items() if torch.is_tensor(x)}),
                                m.last_blur.float().abs().cpu())[1] for j in range(0, 20, 4)])
            m.train()
            pm = torch.stack([b[k][v[k]].median() for k in range(20)])
            curve.append({"step": i, "train_loss": np.mean(losses[-100:]) if losses else np.nan,
                          "pixel_spearman": spearman(b[v], t[v]), "crop_median_spearman": spearman(pm, tm),
                          "pred_median_min": float(pm.min()), "pred_median_max": float(pm.max()),
                          "true_median_min": float(tm.min()), "true_median_max": float(tm.max())})
            print(curve[-1], flush=True)
        if i == a.steps:
            break
        bt = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = models.predict(m, bt)
        loss = m.loss(pred, bt)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    pd.DataFrame(curve).to_csv(out / f"curve_{tag}.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
