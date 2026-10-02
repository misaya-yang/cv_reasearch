"""Shared pieces for the demo4 experiments (training-free in-context segmentation on top of INSID3).

INSID3's own code (github.com/visinf/INSID3, cloned next to this package) is imported unchanged; the encoder is the
ungated timm copy of the DINOv3 ViT-L/16 weights wrapped to expose the one method INSID3 calls.
"""
import json, math, os, pickle, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"); CACHE = os.environ.get("DEMO4_CACHE", "/root/demo4_cache")
sys.path.insert(0, f"{ROOT}/INSID3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda": torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DEMO4_GPU_FRAC", 0.3)))


class TimmDINOv3(torch.nn.Module):
    """timm DINOv3 with the `get_intermediate_layers(x, n=1, reshape=True)` call of the official hub model."""
    def __init__(self, mdir=None):
        super().__init__(); import timm; from safetensors.torch import load_file
        mdir = mdir or f"{CACHE}/models/dinov3-vitl16-timm"; cfg = json.load(open(f"{mdir}/config.json"))
        self.m = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
        print("load", self.m.load_state_dict(load_file(f"{mdir}/model.safetensors"), strict=True))

    def get_intermediate_layers(self, x, n=1, reshape=True):
        return self.m.forward_intermediates(x, indices=n, norm=True, output_fmt="NCHW" if reshape else "NLC", intermediates_only=True)


def build_model(image_size=1024, svd=500, tau=0.6, merge=0.2):
    from models.insid3 import INSID3
    m = INSID3(encoder=TimmDINOv3().eval(), image_size=image_size, svd_components=svd, tau=tau, merge_threshold=merge,
               mask_refiner="bilinear", resize_to_orig_size=False, device=DEV)
    for p in m.parameters(): p.requires_grad = False
    return m.to(DEV).eval()


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
