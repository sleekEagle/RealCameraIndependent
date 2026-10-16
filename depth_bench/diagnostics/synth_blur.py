"""
Diagnostic only (no reported model is trained on synthetic data): can Camind's stage 1 (dropout 0,
log-blur loss, AdamW lr 1e-3, clipping 1.0) learn SYNTHETIC defocus blur, and which property of the
real data stops it?

Sharp source crops: MODEST views shrunk 3x (area averaging), own blur ~0.3 px. Each crop gets a blur
map applied as spatially varying Gaussian blur (interpolated between 14 blur levels). Test crops come
from other (val) views.
  base: smooth random blur map per crop (4x4 log-uniform field, bicubic), 0.5-10 px
  A:    base + camera processing after the optics: unsharp mask (0.75, 0.85 px; reports/base_blur.md),
        Gaussian noise 0.01, JPEG quality 92
  B:    one uniform blur per crop, 0.5-10 px (no sharp-vs-blurred comparison inside a crop)
  C:    smooth maps but only 0.6-2.5 px (the size of most real blurs)

    python -m depth_bench.diagnostics.synth_blur --variant base --steps 3000
Writes curve_<variant>_s<seed>.csv, bins_<variant>_s<seed>.csv, example_<variant>_s<seed>.png to --out.
"""
import argparse
import math
import random
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import data, models
from .common import gauss_blur, out_dir, spearman

LEVELS = np.geomspace(0.3, 12.0, 14)
RANGES = {"base": (0.5, 10.0), "A": (0.5, 10.0), "B": (0.5, 10.0), "C": (0.6, 2.5)}
BINS = [0, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12]
RESIDUAL = 0.3   # px, own blur of the shrunk crops


def sharp_crops(rows, per_img, rng):
    out = []
    for r in rows:
        img = data.load_image(r)
        small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
        H, W = small.shape[:2]
        for _ in range(per_img):
            y, x = rng.randrange(H - 518), rng.randrange(W - 518)
            out.append(small[y:y + 518, x:x + 518])
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()


def make_batch(src, idx, gen, variant):
    smin, smax = RANGES[variant]
    x = src[idx].cuda().float() / 255
    B = x.shape[0]
    lo = torch.rand(B, 1, 4, 4, generator=gen).cuda()
    if variant == "B":
        lo = lo[:, :, :1, :1].expand(B, 1, 4, 4).contiguous()
    ls = F.interpolate(lo, size=(518, 518), mode="bicubic", align_corners=False).clamp(0, 1)
    sig = torch.exp(math.log(smin) + ls * (math.log(smax) - math.log(smin)))[:, 0]
    stack = torch.stack([gauss_blur(x, float(s)) for s in LEVELS])
    pos = torch.from_numpy(np.interp(np.log(sig.cpu().numpy()), np.log(LEVELS), np.arange(len(LEVELS)))).cuda().float()
    i0 = pos.floor().long().clamp(0, len(LEVELS) - 2)
    w = (pos - i0).unsqueeze(1)
    g0 = torch.gather(stack, 0, i0.unsqueeze(0).unsqueeze(2).expand(1, B, 3, 518, 518))[0]
    g1 = torch.gather(stack, 0, (i0 + 1).unsqueeze(0).unsqueeze(2).expand(1, B, 3, 518, 518))[0]
    img = (g0 * (1 - w) + g1 * w).clamp(0, 1)
    if variant == "A":
        img = (img + 0.75 * (img - gauss_blur(img, 0.85))).clamp(0, 1)
        g = torch.Generator().manual_seed(int(torch.randint(0, 1 << 30, (1,), generator=gen)))
        img = (img + 0.01 * torch.randn(img.shape, generator=g).cuda()).clamp(0, 1)
        u8 = (img * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        dec = [cv2.imdecode(cv2.imencode(".jpg", u, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_UNCHANGED)
               for u in u8]
        img = torch.from_numpy(np.stack(dec)).cuda().permute(0, 3, 1, 2).float() / 255
    return img, torch.sqrt(sig ** 2 + RESIDUAL ** 2), x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(RANGES), default="base")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fold", default="S2")
    ap.add_argument("--seed", type=int, default=0, help="crops, blur maps and initial weights")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir("synth_blur", a.out)
    V, tag = a.variant, f"{a.variant}_s{a.seed}"
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    man = data.read_manifest()
    TR = sharp_crops(rng.sample(data.select(man, a.fold, ("train",)), 80), 6, rng)
    TE = sharp_crops(rng.sample(data.select(man, a.fold, ("val",)), 20), 2, rng)
    print(f"variant {V}: sharp crops train {len(TR)}, test {len(TE)}", flush=True)

    m = models.build("camind").cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    cam = torch.tensor([[0.05, 5.0, 2.0]]).cuda()
    gte = torch.Generator().manual_seed(123)
    test = [make_batch(TE, torch.arange(k, min(k + 8, len(TE))), gte, V) for k in range(0, len(TE), 8)]

    def evaluate():
        m.eval()
        P, T, X = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for img, true, _ in test:
                m(img, None, cam.expand(len(img), 3))
                P.append(m.last_blur.float().abs().cpu())
                T.append(true.cpu())
                X.append(models.CamindW.texture_mask(img, 0.02).cpu())
        m.train()
        return torch.cat(P)[:, ::4, ::4], torch.cat(T)[:, ::4, ::4], torch.cat(X)[:, ::4, ::4]

    gen = torch.Generator().manual_seed(a.seed)
    curve, losses, t0 = [], [], time.time()
    for step in range(a.steps + 1):
        if step % 250 == 0:
            p, t, tex = evaluate()
            curve.append({"step": step, "train_loss": np.mean(losses[-250:]) if losses else np.nan,
                          "spearman_all": spearman(p, t), "spearman_textured": spearman(p[tex], t[tex]),
                          "median_rel_err": float(((p - t).abs() / t).median())})
            print(curve[-1], f"{time.time() - t0:.0f}s", flush=True)
        if step == a.steps:
            break
        img, true, _ = make_batch(TR, torch.randint(0, len(TR), (a.batch,), generator=gen), gen, V)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(img, None, cam.expand(a.batch, 3))
        loss = F.mse_loss(torch.log(m.last_blur.float().abs() + 0.05), torch.log(true + 0.05))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    pd.DataFrame(curve).to_csv(out / f"curve_{tag}.csv", index=False)

    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (t >= lo) & (t < hi)
        if s.sum() >= 100:
            st = s & tex
            rows.append({"bin": f"{lo:g}-{hi:g}", "pixels": int(s.sum()), "true_median": float(t[s].median()),
                         "pred_median": float(p[s].median()),
                         "pred_median_textured": float(p[st].median()) if st.sum() > 50 else np.nan})
    bins = pd.DataFrame(rows)
    bins.to_csv(out / f"bins_{tag}.csv", index=False)
    print(bins.round(2).to_string(index=False))

    img, true, sharp = test[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.eval()
        m(img[:3], None, cam.expand(3, 3))
        pred = m.last_blur.float().abs().cpu()
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for k in range(3):
        for j, (im, title, kw) in enumerate([
                (sharp[k].permute(1, 2, 0).cpu().numpy(), "sharp source", {}),
                (img[k].permute(1, 2, 0).cpu().numpy(), f"synthetic ({V})", {}),
                (true[k].cpu().numpy(), "true blur sigma (px)", {"vmin": 0, "vmax": RANGES[V][1]}),
                (pred[k].numpy(), "stage-1 output (px)", {"vmin": 0, "vmax": RANGES[V][1]})]):
            ax[k, j].imshow(im, **kw)
            ax[k, j].set_title(title)
            ax[k, j].axis("off")
    fig.tight_layout()
    fig.savefig(out / f"example_{tag}.png", dpi=60)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
