"""Mixed precision and checkpoint helpers shared by finetune and evaluate."""
import contextlib
import os

import torch


def amp_dtype():
    """bf16 where the GPU supports it, else fp16 (with loss scaling, e.g. Colab T4).
    DEPTH_BENCH_AMP=fp16 forces fp16 (to test the T4 path on another GPU)."""
    if os.environ.get("DEPTH_BENCH_AMP") == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_ctx():
    return torch.autocast("cuda", dtype=amp_dtype())


def trainable_module_names(model):
    """Names of modules whose weights are trained (head modules + unfrozen encoder blocks)."""
    names = []
    for name, m in model.named_modules():
        ps = list(m.parameters(recurse=False))
        if ps and all(p.requires_grad for p in ps):
            names.append(name)
    return names


def trainable_state(model):
    """State dict (params and buffers, e.g. BatchNorm statistics) of the trained modules only."""
    mods = set(trainable_module_names(model))
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if k.rsplit(".", 1)[0] in mods}


def load_trainable(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    missing = [k for k in sd if k not in model.state_dict()]
    if missing:
        raise KeyError(f"checkpoint keys not in model: {missing[:5]}")
    model.load_state_dict(sd, strict=False)
    return ck


@contextlib.contextmanager
def nullctx():
    yield
