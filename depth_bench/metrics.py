"""Loss and evaluation metrics (all on metric depth, valid pixels only)."""
import torch


def silog_loss(pred, gt, lam=0.85, eps=1e-6):
    """Scale-invariant log loss (as in the Depth Anything V2 / BTS metric training), averaged
    over the batch. pred, gt: (B, H, W); gt NaN = invalid."""
    losses = []
    for p, g in zip(pred, gt):
        v = torch.isfinite(g) & (g > 0)
        if v.sum() < 10:
            continue
        d = torch.log(p[v].float() + eps) - torch.log(g[v])
        losses.append(torch.sqrt(torch.clamp((d ** 2).mean() - lam * d.mean() ** 2, min=1e-8)))
    return torch.stack(losses).mean() if losses else pred.sum() * 0


@torch.no_grad()
def depth_metrics(pred, gt):
    """Per-sample metrics. Returns a list of dicts (one per sample)."""
    out = []
    for p, g in zip(pred.float(), gt):
        v = torch.isfinite(g) & (g > 0)
        n = int(v.sum())
        if n < 10:
            out.append({"n_valid": n})
            continue
        p, g = p[v].clamp_min(1e-3), g[v]
        ratio = torch.maximum(p / g, g / p)
        s = torch.median(g / p)                           # median scaling for the scale-invariant score
        out.append({
            "n_valid": n,
            "absrel": float((torch.abs(p - g) / g).mean()),
            "rmse": float(torch.sqrt(((p - g) ** 2).mean())),
            "delta1": float((ratio < 1.25).float().mean()),
            "si_absrel": float((torch.abs(s * p - g) / g).mean()),
            "scale": float(s),                            # gt / pred; 1 = right scale
        })
    return out
