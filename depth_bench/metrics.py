"""Loss and evaluation metrics (all on metric depth, valid pixels only)."""
import numpy as np
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
        dl = torch.log(p) - torch.log(g)
        out.append({
            "n_valid": n,
            "absrel": float((torch.abs(p - g) / g).mean()),
            "rmse": float(torch.sqrt(((p - g) ** 2).mean())),
            "delta1": float((ratio < 1.25).float().mean()),
            "delta2": float((ratio < 1.25 ** 2).float().mean()),
            "delta3": float((ratio < 1.25 ** 3).float().mean()),
            # SILog (Eigen et al.), x100 as usually reported: std of the log error, ignores global scale
            "silog": float(100 * torch.sqrt(torch.clamp((dl ** 2).mean() - dl.mean() ** 2, min=0))),
            "si_absrel": float((torch.abs(s * p - g) / g).mean()),
            "scale": float(s),                            # gt / pred; 1 = right scale
        })
    return out


@torch.no_grad()
def blur_binned_absrel(pred, gt, blur, edges):
    """Per-sample AbsRel within blur bins. blur: (B, H, W) blur diameter in pixels (NaN = unknown).
    Returns per sample a list of (absrel mean or None, pixel count) per bin [edges[k], edges[k+1])."""
    out = []
    for p, g, c in zip(pred.float(), gt, blur):
        v = torch.isfinite(g) & (g > 0) & torch.isfinite(c)
        are = torch.abs(p[v].clamp_min(1e-3) - g[v]) / g[v]
        cv = c[v]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cv >= lo) & (cv < hi)
            n = int(m.sum())
            bins.append((float(are[m].mean()) if n else None, n))
        out.append(bins)
    return out


# ---- Boundary sharpness: scale-invariant boundary F1 from Depth Pro --------------------------------
# Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (ICLR 2025),
# ported from apple/ml-depth-pro src/depth_pro/eval/boundary_metrics.py (boundary_f1, SI_boundary_F1).
# Two neighbouring pixels form an occluding boundary at threshold t when their inverse depths differ by
# a factor > t (4 directions: left, top, right, bottom). Precision and recall compare the predicted
# boundaries with the true ones; F1 is averaged over t = 1.05 ... 1.25 (10 values), weighted by t.
# Only depth ratios are used, so the score ignores the global scale.
# Changes for MODEST:
#  1. Depth Pro used dense synthetic ground truth. Our stereo depth has invalid pixels, so only neighbour
#     pairs where both true depths are valid are counted (for prediction and truth).
#  2. Our true depth edges are spread over ~4 pixels (stereo at 5472 px width; measured: an inverse-depth
#     jump > 1.05 between direct neighbours exists in 44% of test crops, between pixels 4 apart in 62%,
#     and no more at 8 or 16 apart). So both maps are subsampled by `stride` (every 4th pixel) first,
#     which is like evaluating at 1368 px width, close to the image sizes Depth Pro evaluated on.
#     stride=1 gives the original metric (checked equal to Apple's SI_boundary_F1 on dense inputs).

def _fgbg(d, valid_h, valid_v, t):
    """Depth Pro fgbg_depth (on inverse depth), limited to valid neighbour pairs."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((d[:, :-1] / d[:, 1:] > t) & valid_h,   # left
                (d[:-1, :] / d[1:, :] > t) & valid_v,   # top
                (d[:, 1:] / d[:, :-1] > t) & valid_h,   # right
                (d[1:, :] / d[:-1, :] > t) & valid_v)   # bottom


def boundary_f1_si(pred, gt, t_min=1.05, t_max=1.25, n=10, stride=4):
    """pred, gt: (H, W) numpy depth (gt NaN = invalid). Returns weighted boundary F1, precision,
    recall (NaN when the true depth has no boundary at t_min, i.e. the score is undefined)."""
    pred, gt = pred[::stride, ::stride], gt[::stride, ::stride]
    ip = 1.0 / np.clip(pred, 1e-6, None)
    ig = 1.0 / gt
    ok = np.isfinite(ig)
    valid_h, valid_v = ok[:, :-1] & ok[:, 1:], ok[:-1, :] & ok[1:, :]
    ts = np.linspace(t_min, t_max, n)
    w = ts / ts.sum()
    f1s, ps, rs = [], [], []
    for k, t in enumerate(ts):
        P = _fgbg(ip, valid_h, valid_v, t)
        G = _fgbg(ig, valid_h, valid_v, t)
        if k == 0 and not any(g.any() for g in G):
            return {"bf1": np.nan, "bprec": np.nan, "brec": np.nan}
        r = 0.25 * sum(np.count_nonzero(p & g) / max(np.count_nonzero(g), 1) for p, g in zip(P, G))
        p_ = 0.25 * sum(np.count_nonzero(p & g) / max(np.count_nonzero(p), 1) for p, g in zip(P, G))
        f1s.append(0.0 if r + p_ == 0 else 2 * r * p_ / (r + p_))
        ps.append(p_)
        rs.append(r)
    return {"bf1": float(np.dot(f1s, w)), "bprec": float(np.dot(ps, w)), "brec": float(np.dot(rs, w))}
