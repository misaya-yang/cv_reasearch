"""Shared pieces for demo7: data listing, the frozen backbone, and a faithful re-implementation of the
SuperADD feature pipeline (github.com/LukasRoom/SuperADD, commit 44cf251, CC BY-NC 4.0).

Pipeline of the baseline, kept identical here:
  image in [0,1] -> x * gain, clip to [0,1] -> bicubic resize by 0.625 (antialias) -> ImageNet normalisation
  -> overlapping 640x640 tiles (128 px overlap) -> DINOv3 ViT-H+/16 blocks 7/15/23/31, no final norm
  -> per layer: Euclidean distance to the nearest memory-bank vector, divided by the feature dimension
  -> mean over layers -> bilinear upsampling to 1/4 of the input size.
"""
import json, math, os, glob
import numpy as np, torch
import torch.nn.functional as F
from itertools import product
from PIL import Image

ROOT = os.environ.get("DEMO7_ROOT", "/root/autodl-tmp/demo7")
CACHE = os.environ.get("DEMO7_CACHE", "/root/demo7_cache")
DATA = f"{CACHE}/data/mvtec_ad_2"
CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
BACKBONES = {"hplus": (f"{CACHE}/models/dinov3-vith16plus-timm", [7, 15, 23, 31]),
             "large": ("/root/demo4_cache/models/dinov3-vitl16-timm", [5, 11, 17, 23])}
RESIZE, TILE, OVERLAP, MP = 0.625, 640, 128, 16
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load_backbone(name="hplus"):
    import timm
    from safetensors.torch import load_file
    mdir, layers = BACKBONES[name]
    cfg = json.load(open(f"{mdir}/config.json"))
    m = timm.create_model(cfg["architecture"], pretrained=False, num_classes=0)
    m.load_state_dict(load_file(f"{mdir}/model.safetensors"), strict=True)
    return m.cuda().eval(), layers


def load_image(path):
    """(3,H,W) float tensor in [0,1], like torchvision ToTensor on an RGB-converted image."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)


def load_mask(path, size=None):
    m = Image.open(path).convert("L")
    return torch.from_numpy(np.asarray(m, dtype=np.uint8) > 127)


def cond_of(path):
    """lighting condition from the file name: 000_shift_2.png -> ('000', 'shift_2')"""
    b = os.path.basename(path)[:-4]
    return b.split("_")[0], "_".join(b.split("_")[1:])


def list_split(cat):
    """dict with train prototypes, train threshold part (SuperADD: every 8th image), validation, test good/bad"""
    tr = sorted(glob.glob(f"{DATA}/{cat}/train/good/*.png"))
    va = sorted(glob.glob(f"{DATA}/{cat}/validation/good/*.png"))
    tg = sorted(glob.glob(f"{DATA}/{cat}/test_public/good/*.png"))
    tb = sorted(glob.glob(f"{DATA}/{cat}/test_public/bad/*.png"))
    return dict(proto=[p for i, p in enumerate(tr) if i % 8 != 0], thr=[p for i, p in enumerate(tr) if i % 8 == 0],
                val=va, good=tg, bad=tb)


def mask_path(img_path):
    d, b = os.path.split(img_path)
    return os.path.join(os.path.dirname(d), "ground_truth", "bad", b[:-4] + "_mask.png")


def axis_split(dim, ps=TILE, ov=OVERLAP, mp=MP):
    """tile layout along one axis, copied from SuperADD's PatchedExecution.axis_patch_split"""
    assert dim >= ps
    dt, ot, pt = dim // mp, ov // mp, ps // mp
    n = math.ceil((dt - pt) / (pt - 2 * ot)) + 1
    fac, div = dt - pt, max(1, n - 1)
    inp = [((i * fac) // div, (i * fac) // div + pt) for i in range(n)]
    pred = []
    for i in range(n):
        s = 0 if i == 0 else math.ceil((inp[i - 1][1] - inp[i][0]) / 2)
        e = pt if i == n - 1 else pt - math.floor((inp[i][1] - inp[i + 1][0]) / 2)
        pred.append((s, e))
    res = [(0, pred[0][1] - pred[0][0])]
    for i in range(1, n):
        res.append((res[i - 1][1], res[i - 1][1] + pred[i][1] - pred[i][0]))
    return [(s * mp, e * mp) for s, e in inp], pred, res


def preprocess(img, gain=1.0):
    """img (3,H,W) in [0,1] on the GPU -> normalised, resized (1,3,h,w)"""
    x = torch.clip(img * gain, 0, 1)[None]
    new = (int(x.shape[-2] * RESIZE), int(x.shape[-1] * RESIZE))
    x = F.interpolate(x, size=new, mode="bicubic", align_corners=False, antialias=True)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.inference_mode()
def extract(model, layers, img, gain=1.0, bs=6):
    """features of one image: list over layers of (ht, wt, C) float32 GPU tensors"""
    x = preprocess(img, gain)
    _, c, h, w = x.shape
    iy, py, ry = axis_split(h)
    ix, px, rx = axis_split(w)
    tiles = torch.stack([x[0, :, a:b, cc:d] for (a, b), (cc, d) in product(iy, ix)])
    outs = [[] for _ in layers]
    for s in range(0, len(tiles), bs):
        o = model.forward_intermediates(tiles[s:s + bs], indices=layers, norm=False, output_fmt="NLC", intermediates_only=True)
        for k, t in enumerate(o):
            outs[k].append(t)
    pt = TILE // MP
    res = []
    for k in range(len(layers)):
        t = torch.cat(outs[k]).reshape(len(tiles), pt, pt, -1)
        full = torch.zeros(h // MP, w // MP, t.shape[-1], device=t.device)
        for i, ((p_y, p_x), (r_y, r_x)) in enumerate(zip(product(py, px), product(ry, rx))):
            full[r_y[0]:r_y[1], r_x[0]:r_x[1]] = t[i, p_y[0]:p_y[1], p_x[0]:p_x[1]]
        res.append(full)
    return res


def nn_dist(q, keys, chunk=4096):
    """Euclidean distance and index of the nearest key for every query row (float32, GEMM-based like the baseline)"""
    ds, ix = [], []
    keys = keys.float()
    for s in range(0, len(q), chunk):
        d = torch.cdist(q[s:s + chunk].float(), keys, compute_mode="use_mm_for_euclid_dist")
        v, i = d.min(1)
        ds.append(v); ix.append(i)
    return torch.cat(ds), torch.cat(ix)


def subsample_superadd(feats, target, rng, iterations=100, knn=100):
    """SuperADD's density-based subsampling (nearest_neighbor.subsampling_distance_based_fast); returns kept row indices.
    feats: (N, C) CPU tensor (any float dtype)."""
    n_all = len(feats)
    keep = np.zeros(n_all, bool)
    size, tgt = int(n_all / iterations), target // iterations
    for _ in range(iterations):
        cand = np.where(~keep)[0]
        idx = rng.choice(cand, size=min(size, len(cand)), replace=False)
        x = feats[torch.from_numpy(idx)].float().cuda()
        d = torch.cdist(x, x, compute_mode="use_mm_for_euclid_dist").topk(min(knn, len(idx)), dim=-1, largest=False).values.cpu().numpy()
        t = float(np.mean(np.float64(d))) / 10
        r = rng.random(len(idx))
        n = tgt + 1
        while n > tgt:
            km = r < 1.0 / ((d < t).sum(-1) + 1)
            n = int(km.sum()); t *= 1.1
        keep[idx] = km
    diff = target - int(keep.sum())
    if diff > 0:
        add = np.where(~keep)[0]; rng.shuffle(add); keep[add[:diff]] = True
    return np.where(keep)[0]
