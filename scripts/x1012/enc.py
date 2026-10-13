"""Frozen DINOv3-L/16 encoder access for fresh observations (same preprocessing as the O24 cache)."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import lab

MDIR = lab.ASSETS / "demo4_cache/models/dinov3-vitl16-timm"


@lru_cache(1)
def model():
    import timm
    from safetensors.torch import load_file
    cfg = json.loads((MDIR / "config.json").read_text())
    m = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
    m.load_state_dict(load_file(str(MDIR / "model.safetensors")), strict=True)
    return m.to(lab.DEV).eval()


def rgb(binding, role):
    """Official model input image: decoded RGB with the loader's NumPy crop."""
    row = binding["row"]
    with Image.open(lab.ASSETS / row[role + "_path"]) as im:
        x = np.asarray(im.convert("RGB")).copy()
    crop = row.get(role + "_crop")
    if crop is not None:
        x0, y0, x1, y1 = crop
        x = x[y0:y1, x0:x1]
    return Image.fromarray(x)


def transform(size=1024):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def layers(x, indices, norm=True):
    """Patch tokens of the requested blocks (1-based, 24 = last), each (N, C) after model.norm."""
    m = model()
    outs = m.forward_intermediates(x[None].to(lab.DEV), indices=[i - 1 for i in indices], norm=norm,
                                   output_fmt="NLC", intermediates_only=True)
    return {i: o[0].float() for i, o in zip(indices, outs)}
