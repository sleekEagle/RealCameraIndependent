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
    camind   Camind (WACV 2024) network as a plain two-stage CNN, trained from scratch, no camera input
    camind_corr / camind_corr0  the same network with the camera correction of the blur model (with /
             without the base blur sigma0), trained from scratch; uses intrinsics, f, N and focus distance
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
        self._decoder_fp32()

    def _decoder_fp32(self):
        """Under fp16 autocast (GPUs without bf16, e.g. Colab T4) the decoder overflows: activations in
        decoder_mono.upconv_1_1 reach ~34000 (fp16 max 65504) and conv_out_2 gives inf, so the depth is
        NaN. The encoder is fine. So with fp16 the decoder runs in fp32 (autocast off, inputs cast up).
        bf16 has the fp32 range, so nothing changes there."""
        from .train_utils import amp_dtype
        if amp_dtype() != torch.float16:
            return
        dec = self.net.depth_model.decoder
        fwd = dec.forward

        def up(x):
            if torch.is_tensor(x):
                return x.float() if x.is_floating_point() else x
            if isinstance(x, (list, tuple)):
                return type(x)(up(v) for v in x)
            if isinstance(x, dict):
                return {k: up(v) for k, v in x.items()}
            return x

        def fwd32(*args, **kw):
            with torch.autocast("cuda", enabled=False):
                return fwd(*up(args), **up(kw))
        dec.forward = fwd32

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


BLUR_K = 36000.0        # sigma (px) per metre of blur-circle diameter on the sensor (reports/base_blur.md)
SIGMA0_PX = 0.8         # base blur (px) (reports/base_blur.md: 0.62 raw px; ~0.8 px in the rectified images)
X_SCALE = 2.0           # |s - d| / d is divided by this (its 99th percentile on training views is 1.94)


class CamindW(BaseWrapper):
    """Camind: "Camera-Independent Single Image Depth Estimation from Defocus Blur" (WACV 2024),
    network in camind_net.py (dofNet_arch4.AENet, 16 base filters). Trained from scratch.

    Stage 1 maps the RGB crop to a one-channel map (the intermediate "blur" output). Stage 2 maps that
    map, after the optional camera correction, plus a second input channel to metric depth.

    correction=False ("camind"): a plain two-stage CNN with no camera information. Stage 2 gets the
    stage-1 map as it is, its second channel is the constant FOC_CONST, and the loss is depth MSE only.

    correction=True ("camind_corr", "camind_corr0"): camera correction from the blur model
        sigma(d)^2 = sigma0^2 + (C |s - d| / d)^2,   C = k f^2 / (N (s - f))
    (f from the calibrated image distance v and s, 1/f = 1/v + 1/s). Read as sigma(d), the stage-1 output
    goes through sqrt(max(sigma^2 - sigma0^2, 0)) = C |s - d| / d inside AENet and is multiplied by the
    scalar camparam = 1 / (C x_scale), so stage 2 gets |s - d| / d scaled to about 0-1 (divided by
    x_scale); its second channel is the focus distance s (m). Stage 1 is supervised on that quantity:
    loss = depth_weight * depth MSE + stage1_weight * stage-1 loss, where the stage-1 loss is
      stage1_space="x":     MSE(stage-2 input, true |s - d| / d / x_scale). A blur error of e px costs
                            (e / (C x_scale))^2, so images with a small C (high F-number, short focal
                            length) dominate; there the defocus is mostly below sigma0.
      stage1_space="sigma": MSE(stage-1 output, true blur sqrt(sigma0^2 + (C |s - d| / d)^2)) in px:
                            the same weight for a blur error in every image.
    depth_weight=0 trains stage 1 alone (a probe: can the blur be read from the image at all?).
    camind_corr0 is the ablation that ignores the base blur (sigma0 = 0: stage-2 input = stage-1
    output / (C x_scale), the original Camind camind=True path)."""
    name = "camind"
    pretrained = False
    FOC_CONST = 1.0
    INIT_BLUR = 2.0  # px: start value of the stage-1 output (output-layer bias), above sigma0, so the
                     # base-blur removal does not block the gradient at the start

    def __init__(self, num_filter=16, correction=False, sigma0=SIGMA0_PX, blur_k=BLUR_K, stage1_weight=10.0,
                 x_scale=X_SCALE, stage1_space="x", depth_weight=1.0):
        super().__init__()
        from .camind_net import AENet
        self.net = AENet(3, 1, num_filter, flag_step2=True)
        self.correction, self.sigma0, self.blur_k, self.x_scale = correction, sigma0, blur_k, x_scale
        self.stage1_weight = stage1_weight if correction else 0.0
        self.stage1_space, self.depth_weight = stage1_space, depth_weight
        self.needs_camera = correction
        if correction:
            nn.init.constant_(self.net.conv_out_blur[0].bias, self.INIT_BLUR)
        self.last_blur = None   # stage-1 output of the last forward pass
        self.last_mul = None    # stage-2 input of the last forward pass (corrected: |s - d| / d estimate)

    def blur_scale(self, cam):
        """C = k f^2 / (N (s - f)): defocus blur sigma (px) per unit of |s - d| / d."""
        v, N, s = cam[:, 0], cam[:, 1], cam[:, 2]
        f = v * s / (v + s)   # cam holds the calibrated image distance v; thin lens 1/f = 1/v + 1/s
        return self.blur_k * f ** 2 / (N * (s - f))

    def forward(self, image, K=None, cam=None):
        H, W = image.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16           # 4 poolings
        x = F.pad(image, (0, pw, 0, ph), mode="reflect") if ph or pw else image
        B, _, Hp, Wp = x.shape
        if self.correction:
            camparam = 1.0 / (self.blur_scale(cam.float()) * self.x_scale).view(B, 1, 1, 1)
            foc_dist = cam[:, 2].float().view(B, 1, 1, 1).expand(B, 1, Hp, Wp).to(x.dtype)
            depth, blur, mul = self.net(x, camind=True, camparam=camparam, foc_dist=foc_dist,
                                        sigma0=self.sigma0 if self.sigma0 > 0 else None)
        else:
            foc_dist = x.new_full((B, 1, Hp, Wp), self.FOC_CONST)
            depth, blur, mul = self.net(x, camind=False, camparam=None, foc_dist=foc_dist)
        self.last_blur, self.last_mul = blur[:, 0, :H, :W], mul[:, 0, :H, :W]
        return depth[:, 0, :H, :W]

    def loss(self, pred, batch):
        gt = batch["depth"].cuda(non_blocking=True)
        v = torch.isfinite(gt) & (gt > 0)
        loss = self.depth_weight * F.mse_loss(pred.float()[v], gt[v])
        if self.stage1_weight > 0:
            cam = batch["cam"].cuda(non_blocking=True).float()
            x = (cam[:, 2].view(-1, 1, 1) - gt).abs() / gt
            if self.stage1_space == "sigma":   # stage-1 output vs the blur the blur model gives (px)
                C = self.blur_scale(cam).view(-1, 1, 1)
                target = torch.sqrt(self.sigma0 ** 2 + (C * x) ** 2)
                loss = loss + self.stage1_weight * F.mse_loss(self.last_blur.float()[v], target[v])
            else:   # stage 1, after the correction, should give the scaled true |s - d| / d
                loss = loss + self.stage1_weight * F.mse_loss(self.last_mul.float()[v], x[v] / self.x_scale)
        return loss

    def encoder_blocks(self):
        return []

    def head_modules(self):
        return [self.net]   # trained from scratch: everything is trainable


def predict(model, batch):
    """Run a wrapper on a collated batch (passes the lens settings to models that need them)."""
    img, K = batch["image"].cuda(non_blocking=True), batch["K"].cuda(non_blocking=True)
    if getattr(model, "needs_camera", False):
        return model(img, K, batch["cam"].cuda(non_blocking=True))
    return model(img, K)


MODELS = {
    "da2": lambda **kw: DA2Metric(**kw),
    "unidepth": lambda **kw: UniDepthV2W(**kw),
    "unidepth_noK": lambda **kw: UniDepthV2W(use_intrinsics=False, **kw),
    "metric3d": lambda **kw: Metric3DW(**kw),                              # official canonical focal 1000
    "metric3d_c6000": lambda **kw: Metric3DW(canonical_focal=6000.0, **kw),  # for fine-tuning on MODEST
    "da3": lambda **kw: DA3Metric(**kw),
    "camind": lambda **kw: CamindW(**kw),   # from scratch, no camera input
    "camind_corr": lambda **kw: CamindW(correction=True, **kw),     # camera correction with sigma0
    "camind_corr0": lambda sigma0=None, **kw: CamindW(correction=True, sigma0=0.0, **kw),  # ablation: sigma0 ignored
}


def build(name, **kw):
    return MODELS[name](**kw)


def add_model_args(ap):
    """Options of the camera-corrected Camind variants (ignored by the other models)."""
    ap.add_argument("--sigma0", type=float, default=SIGMA0_PX, help="camind_corr: base blur in crop px")
    ap.add_argument("--blur_k", type=float, default=BLUR_K, help="camind_corr*: blur constant k (px per m)")
    ap.add_argument("--x_scale", type=float, default=X_SCALE,
                    help="camind_corr*: |s-d|/d is divided by this (stage-2 input and stage-1 target)")
    ap.add_argument("--stage1_weight", type=float, default=10.0,
                    help="camind_corr*: weight of the stage-1 loss, MSE(corrected stage-1 output, true |s-d|/d)")
    ap.add_argument("--stage1_space", choices=["x", "sigma"], default="x",
                    help="camind_corr*: stage-1 loss on |s-d|/d (x) or on the blur in px (sigma)")
    ap.add_argument("--depth_weight", type=float, default=1.0,
                    help="camind_corr*: weight of the depth loss (0: train stage 1 alone, a blur probe)")


def model_kwargs(name, a):
    if name in ("camind_corr", "camind_corr0"):
        return {"sigma0": a.sigma0, "blur_k": a.blur_k, "stage1_weight": a.stage1_weight, "x_scale": a.x_scale,
                "stage1_space": a.stage1_space, "depth_weight": a.depth_weight}
    return {}


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
