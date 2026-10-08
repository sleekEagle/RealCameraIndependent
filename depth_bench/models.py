"""
Wrappers that give the four baselines one interface:

    depth = model(image, K)       image (B,3,H,W) RGB in [0,1], K (B,3,3) crop intrinsics
                                  -> depth (B,H,W) in metres, > 0

Each wrapper does its own normalization, padding (to the patch / stride multiple) and the
conversion of its native output to metric depth, so training and evaluation code is the
same for all models. No wrapper resizes the input.

    encoder_blocks()  ordered list of encoder blocks (for unfreezing the last ones)
    head_modules()    decoder / head modules (always trained when fine-tuning)

Models (and the HF / hub checkpoints they load):
    da2      Depth Anything V2 metric, Hypersim (indoor), ViT-S/B/L
    unidepth UniDepth V2, ViT-S/B/L; uses the given intrinsics unless use_intrinsics=False
    metric3d Metric3D v2 ConvNeXt-L (canonical focal 1000 px, label-scale mode)
    da3      Depth Anything 3 metric large (canonical focal 300 px)
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CODE_EXT

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _pad_to(x, mult, value=0.0):
    """Pad right/bottom so H and W are multiples of `mult`; returns padded x and (H, W)."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=value)
    return x, (H, W)


def _flatten_blocks(blocks):
    out = []
    for b in blocks:
        if isinstance(b, nn.ModuleList):  # DINOv2 "BlockChunk"
            out += [m for m in b if not isinstance(m, nn.Identity)]
        else:
            out.append(b)
    return out


class BaseWrapper(nn.Module):
    name = "base"
    uses_intrinsics = False

    def _norm(self, image):
        return (image - IMAGENET_MEAN.to(image)) / IMAGENET_STD.to(image)


class DA2Metric(BaseWrapper):
    name = "da2"
    CFG = {"vits": (64, [48, 96, 192, 384], "Small"), "vitb": (128, [96, 192, 384, 768], "Base"),
           "vitl": (256, [256, 512, 1024, 1024], "Large")}

    def __init__(self, variant="vitb", max_depth=20.0):
        super().__init__()
        sys.path.insert(0, str(CODE_EXT / "Depth-Anything-V2" / "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download
        feats, outc, size = self.CFG[variant]
        self.net = DepthAnythingV2(encoder=variant, features=feats, out_channels=outc, max_depth=max_depth)
        ckpt = hf_hub_download(f"depth-anything/Depth-Anything-V2-Metric-Hypersim-{size}",
                               f"depth_anything_v2_metric_hypersim_{variant}.pth")
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.variant = variant

    def forward(self, image, K):
        x, (H, W) = _pad_to(self._norm(image), 14)
        d = self.net(x)                      # (B, H, W) metres
        return d[:, :H, :W].clamp_min(1e-3)

    def encoder_blocks(self):
        return _flatten_blocks(self.net.pretrained.blocks)

    def head_modules(self):
        return [self.net.depth_head]


class _UniDepthRays:
    """Camera stand-in for UniDepthV2.encode_decode, which only calls camera.get_rays().
    UniDepth's own Pinhole.get_rays builds the pixel grid for a single image, so a batch of
    crops with different intrinsics fails; get_pinhole_rays is the batch-aware version."""

    def __init__(self, K):
        from unidepth.utils.camera import Pinhole
        self.cam = Pinhole(K=K)

    def get_rays(self, shapes, noisy=False):
        return self.cam.get_pinhole_rays(shapes)


class UniDepthV2W(BaseWrapper):
    name = "unidepth"

    def __init__(self, variant="vitb14", use_intrinsics=True):
        super().__init__()
        from unidepth.models import UniDepthV2
        self.net = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{variant}")
        self.uses_intrinsics = use_intrinsics
        self.variant = variant

    def forward(self, image, K):
        x, (H, W) = _pad_to(self._norm(image), 14)
        cam = _UniDepthRays(K.float()) if self.uses_intrinsics else None
        _, out = self.net.encode_decode({"image": x, "camera": cam}, image_metas=[])
        return out["depth"][:, 0, :H, :W].clamp_min(1e-3)

    def encoder_blocks(self):
        return _flatten_blocks(self.net.pixel_encoder.blocks)

    def head_modules(self):
        return [self.net.pixel_decoder]


def _stub_mmcv():
    """Metric3D's mono/utils/comm.py imports mmcv.utils only for environment-info helpers.
    mmcv has no build for this torch on Windows, so register a minimal stand-in; the config
    loader in Metric3D's hubconf then falls back to mmengine (missing Config -> ImportError)."""
    import types
    if "mmcv" in sys.modules:
        return
    from mmengine.utils import get_git_hash
    mmcv, utils = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")
    utils.collect_env = lambda: {}
    utils.get_git_hash = get_git_hash
    mmcv.utils = utils
    sys.modules["mmcv"], sys.modules["mmcv.utils"] = mmcv, utils


class Metric3DW(BaseWrapper):
    name = "metric3d"
    uses_intrinsics = True
    MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __init__(self, variant="convnext_large", canonical_focal=1000.0):
        """canonical_focal: Metric3D predicts depth for a camera with this focal length (px) and
        rescales by fx / canonical_focal. The official value is 1000, but its decoder only outputs
        canonical depths in [0.3, 150]; with native crops (fx ~4500-12000 px) real depths of a few
        metres map below 0.3 (e.g. 3 m at fx 12000 -> 0.25), so they cannot be predicted. For
        fine-tuning on MODEST use a larger value (e.g. 6000) so typical depths land in range."""
        super().__init__()
        repo = str(CODE_EXT / "Metric3D")
        sys.path.insert(0, repo)
        _stub_mmcv()
        self.net = torch.hub.load(repo, f"metric3d_{variant}", pretrain=True, source="local", trust_repo=True)
        self.variant = variant
        self.canonical_focal = float(canonical_focal)

    CANVAS = (544, 1216)  # Metric3D ConvNeXt training input size

    def forward(self, image, K):
        # The ConvNeXt model's output depends strongly on the input canvas: the same 518x518 crop
        # gave 1585 m padded right/bottom to 544x544, 77 m centred in 544x544, ~11 m centred in its
        # training size 544x1216 (true 4.1 m). So the crop is centred in a canvas of at least the
        # training size, filled with the mean colour (0 after normalization); only the crop region
        # is used. The model sees exactly the same pixels as the other baselines.
        x = (image * 255 - self.MEAN.to(image)) / self.STD.to(image)
        B, _, H, W = x.shape
        Hc = max(self.CANVAS[0], -(-H // 32) * 32)
        Wc = max(self.CANVAS[1], -(-W // 32) * 32)
        t, l = (Hc - H) // 2, (Wc - W) // 2
        canvas = x.new_zeros(B, 3, Hc, Wc)
        canvas[:, :, t:t + H, l:l + W] = x
        pred, _, _ = self.net({"input": canvas})   # depth for the canonical camera
        pred = pred[:, 0, t:t + H, l:l + W]
        fx = K[:, 0, 0].view(-1, 1, 1)
        return (pred * fx / self.canonical_focal).clamp_min(1e-3)

    def encoder_blocks(self):
        enc = self.net.depth_model.encoder
        return [blk for stage in enc.stages for blk in stage]

    def head_modules(self):
        return [self.net.depth_model.decoder]


class DA3Metric(BaseWrapper):
    name = "da3"
    uses_intrinsics = True

    def __init__(self, variant="da3metric-large"):
        super().__init__()
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.net = create_object(load_config(MODEL_REGISTRY[variant]))
        sd = load_file(hf_hub_download("depth-anything/DA3METRIC-LARGE", "model.safetensors"))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if unexpected or [m for m in missing if not m.startswith(("cam_", "gs_"))]:
            raise RuntimeError(f"DA3 weights mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.variant = variant

    def forward(self, image, K):
        x, (H, W) = _pad_to(self._norm(image), 14)
        out = self.net(x[:, None])            # (B, N=1, 3, H, W)
        d = out["depth"] if isinstance(out, dict) else out.depth
        d = d.reshape(d.shape[0], -1, *d.shape[-2:])[:, 0, :H, :W]
        fx = K[:, 0, 0].view(-1, 1, 1)
        return (d * fx / 300.0).clamp_min(1e-3)  # DA3 metric: depth = focal * output / 300

    def encoder_blocks(self):
        return _flatten_blocks(self.net.backbone.pretrained.blocks)

    def head_modules(self):
        return [self.net.head]


MODELS = {
    "da2": lambda **kw: DA2Metric(**kw),
    "unidepth": lambda **kw: UniDepthV2W(**kw),
    "unidepth_noK": lambda **kw: UniDepthV2W(use_intrinsics=False, **kw),
    "metric3d": lambda **kw: Metric3DW(**kw),                              # official canonical focal 1000
    "metric3d_c6000": lambda **kw: Metric3DW(canonical_focal=6000.0, **kw),  # for fine-tuning on MODEST
    "da3": lambda **kw: DA3Metric(**kw),
}


def build(name, **kw):
    return MODELS[name](**kw)


def set_trainable(model, unfreeze_frac=0.25):
    """Freeze everything, then unfreeze the head and the last `unfreeze_frac` of encoder blocks.
    Returns (encoder_params, head_params)."""
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = model.encoder_blocks()
    n = int(round(len(blocks) * unfreeze_frac))
    enc = [p for b in blocks[len(blocks) - n:] for p in b.parameters()] if n else []
    head = [p for m in model.head_modules() for p in m.parameters()]
    for p in enc + head:
        p.requires_grad_(True)
    return enc, head
