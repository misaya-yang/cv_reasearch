"""Existing DINOv3 wrapper, seed-defined COCO episodes and I/U helpers.

Moved from icx/common.py; no experiment is run or model constructed on import.
DEMO4_CACHE remains the legacy environment variable for the existing asset store.
"""
import json, os, pickle
import numpy as np
import torch
from PIL import Image

CACHE = os.environ.get("DEMO4_CACHE", "/root/demo4_cache")

class TimmDINOv3(torch.nn.Module):
    """timm DINOv3 with the `get_intermediate_layers(x, n=1, reshape=True)` call of the official hub model."""
    def __init__(self, mdir=None):
        super().__init__(); import timm; from safetensors.torch import load_file
        mdir = mdir or f"{CACHE}/models/dinov3-vitl16-timm"; cfg = json.load(open(f"{mdir}/config.json"))
        self.m = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
        print("load", self.m.load_state_dict(load_file(f"{mdir}/model.safetensors"), strict=True))

    def get_intermediate_layers(self, x, n=1, reshape=True):
        return self.m.forward_intermediates(x, indices=n, norm=True, output_fmt="NCHW" if reshape else "NLC", intermediates_only=True)



def coco_episodes(fold, n=1000, shot=1, seed=0):
    """The episode list INSID3's COCO-20i loader draws with seed 0 (same calls to numpy's generator, in the same order)."""
    base = f"{CACHE}/data/COCO2014"; meta = pickle.load(open(f"{base}/splits/val/fold{fold}.pkl", "rb"))
    class_ids = [fold + 4 * v for v in range(20)]; np.random.seed(seed); eps = []
    for _ in range(n):
        c = np.random.choice(class_ids, 1, replace=False)[0]; t = np.random.choice(meta[c], 1, replace=False)[0]; refs = []
        while True:
            r = np.random.choice(meta[c], 1, replace=False)[0]
            if t != r: refs.append(r)
            if len(refs) == shot: break
        eps.append((int(c), str(t), [str(r) for r in refs]))
    return eps, class_ids, base



def coco_load(base, name, c):
    img = Image.open(os.path.join(base, name)).convert("RGB")
    m = torch.from_numpy(np.array(Image.open(os.path.join(base, "annotations", name[:-4] + ".png"))))
    return img, (m == c + 1)



def iu(pred, gt):
    """Foreground intersection and union of two boolean maps."""
    return float((pred & gt).sum()), float((pred | gt).sum())



class Meter:
    """Per-class sums of intersection and union; mIoU = mean over classes of the ratio (the few-shot segmentation convention)."""
    def __init__(self): self.d = {}
    def add(self, name, c, i, u):
        a = self.d.setdefault(name, {}).setdefault(c, [0.0, 0.0]); a[0] += i; a[1] += u
    def miou(self, name): return 100 * float(np.mean([i / max(u, 1.0) for i, u in self.d[name].values()]))
    def names(self): return list(self.d)

