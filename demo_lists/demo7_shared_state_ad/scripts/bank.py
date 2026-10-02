#!/usr/bin/env python3
"""GPU step: build the memory banks for one category and save, for every query image, the distance tables that all
read-outs (per-patch / per-patch-across-layers / image-shared state) are computed from offline.

Banks (all from the same prototype images = 7/8 of train, as in SuperADD):
  A     no augmentation (gain 1), SuperADD subsampling, K vectors per layer
  B     SuperADD: one random gain per image in [0.8, 1.2], SuperADD subsampling, K vectors per layer
  grid  paired grid: K random patch identities (image, y, x), the same for all layers and all gains;
        every identity is stored once per gain in --gains, with its gain label
Queries: train part held out for thresholds (every 8th image), validation, test_public good and bad (all lighting
conditions), always encoded without augmentation; plus one control where the query is rescaled to the mean
intensity of the prototype images before encoding (bank A).

Usage: bank.py --cat vial [--backbone hplus] [--K 100000] [--gains 0.8,0.87,0.93,1,1.07,1.13,1.2]
"""
import argparse, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE, load_backbone, load_image, load_mask, cond_of, list_split, mask_path, extract, nn_dist, subsample_superadd

ap = argparse.ArgumentParser()
ap.add_argument("--cat", required=True)
ap.add_argument("--backbone", default="hplus")
ap.add_argument("--K", type=int, default=100000)
ap.add_argument("--gains", default="0.8,0.87,0.93,1,1.07,1.13,1.2")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0, help="debug: use only this many prototype / query images")
ap.add_argument("--tag", default="")
ap.add_argument("--nogrid", action="store_true", help="only banks A and B (two passes); the grid keeps its gain-1 slice")
a = ap.parse_args()
gains = [1.0] if a.nogrid else [float(g) for g in a.gains.split(",")]
assert 1.0 in gains
rng = np.random.default_rng(a.seed)
t0 = time.time()
log = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)

model, layers = load_backbone(a.backbone)
sp = list_split(a.cat)
proto = sp["proto"][:a.limit] if a.limit else sp["proto"]
L = len(layers)


def all_feats(gain_of):
    """features of every prototype image, per layer one (N, C) float16 CPU tensor"""
    out = [[] for _ in layers]
    for i, p in enumerate(proto):
        f = extract(model, layers, load_image(p).cuda(), gain_of(i))
        for k in range(L):
            out[k].append(f[k].reshape(-1, f[k].shape[-1]).half().cpu())
        if i % 50 == 0: log(f"  image {i}/{len(proto)}")
    shape = f[0].shape[:2]
    return [torch.cat(o) for o in out], shape


def radii(bank, k=10, chunk=4096):
    """distance / C of every bank vector to its 1st..k-th nearest other bank vector, (K, k) float32 CPU"""
    keys = bank.float(); out = []
    for s0 in range(0, len(keys), chunk):
        d = torch.cdist(keys[s0:s0 + chunk], keys, compute_mode="use_mm_for_euclid_dist")
        d[torch.arange(len(d)), torch.arange(s0, s0 + len(d))] = float("inf")
        out.append(d.topk(k, dim=1, largest=False).values.sort(1).values.cpu())
    return torch.cat(out) / bank.shape[1]


# ---- pass 1: no augmentation -> bank A and the gain-1 slice of the grid
log(f"{a.cat}: {len(proto)} prototype images, pass gain 1")
means = np.array([float(load_image(p).mean()) for p in proto[::5]])
ref_mean = float(means.mean())
feats, (ht, wt) = all_feats(lambda i: 1.0)
N, C = feats[0].shape
per = ht * wt
log(f"tokens per image {ht}x{wt}, N={N}, C={C}, prototype mean intensity {ref_mean:.3f}")
K = min(a.K, N)
ident = np.sort(rng.choice(N, size=K, replace=False))
bankG = torch.zeros(len(gains), L, K, C, dtype=torch.half)
bankG[gains.index(1.0)] = torch.stack([f[torch.from_numpy(ident)] for f in feats])
idxA = [subsample_superadd(f, K, rng) for f in feats]
bankA = [f[torch.from_numpy(ix)].cuda() for f, ix in zip(feats, idxA)]
del feats
rA = [radii(b) for b in bankA]
log("bank A done")

# ---- pass 2: SuperADD augmentation -> bank B
img_gain = rng.uniform(0.8, 1.2, size=len(proto))
feats, _ = all_feats(lambda i: float(img_gain[i]))
idxB = [subsample_superadd(f, K, rng) for f in feats]
bankB = [f[torch.from_numpy(ix)].cuda() for f, ix in zip(feats, idxB)]
gainB = [img_gain[ix // per] for ix in idxB]
del feats
rB = [radii(b) for b in bankB]
log("bank B done")

# ---- passes for the other gains of the paired grid (only the K identities are kept)
by_img = [ident[(ident >= i * per) & (ident < (i + 1) * per)] - i * per for i in range(len(proto))]
for gi, g in enumerate(gains):
    if g == 1.0: continue
    pos = 0
    for i, p in enumerate(proto):
        if len(by_img[i]) == 0: continue
        f = extract(model, layers, load_image(p).cuda(), g)
        sel = torch.from_numpy(by_img[i]).cuda()
        for k in range(L):
            bankG[gi, k, pos:pos + len(sel)] = f[k].reshape(-1, C)[sel].half().cpu()
        pos += len(sel)
    assert pos == K
    log(f"grid gain {g} done")
bankG = bankG.cuda()

# ---- queries
queries = [(p, kind) for kind in ["thr", "val", "good", "bad"] for p in (sp[kind][:a.limit] if a.limit else sp[kind])]
n = len(queries)
dA = torch.zeros(n, L, ht, wt); dB = torch.zeros(n, L, ht, wt); dAn = torch.zeros(n, L, ht, wt)
iA = torch.zeros(n, L, ht, wt, dtype=torch.int32); iB = torch.zeros(n, L, ht, wt, dtype=torch.int32)
D = torch.zeros(n, L, ht, wt, len(gains)); I = torch.zeros(n, L, ht, wt, len(gains), dtype=torch.int32)
meta, gts = [], []
for q, (p, kind) in enumerate(queries):
    img = load_image(p).cuda()
    H, W = img.shape[-2:]
    m = float(img.mean())
    f = extract(model, layers, img, 1.0)
    fn = extract(model, layers, img, ref_mean / m)
    for k in range(L):
        x = f[k].reshape(-1, C)
        d, i = nn_dist(x, bankA[k]); dA[q, k] = (d / C).reshape(ht, wt).cpu(); iA[q, k] = i.reshape(ht, wt).int().cpu()
        d, i = nn_dist(x, bankB[k]); dB[q, k] = (d / C).reshape(ht, wt).cpu(); iB[q, k] = i.reshape(ht, wt).int().cpu()
        d, _ = nn_dist(fn[k].reshape(-1, C), bankA[k]); dAn[q, k] = (d / C).reshape(ht, wt).cpu()
        for gi in range(len(gains)):
            d, i = nn_dist(x, bankG[gi, k]); D[q, k, :, :, gi] = (d / C).reshape(ht, wt).cpu(); I[q, k, :, :, gi] = i.reshape(ht, wt).int().cpu()
    sid, cond = cond_of(p)
    if kind == "bad":
        gt = load_mask(mask_path(p)).float()[None, None]
        gt = F.interpolate(gt, size=(H // 4, W // 4), mode="area")[0, 0] >= 0.5
    else:
        gt = torch.zeros(H // 4, W // 4, dtype=torch.bool)
    gts.append(gt)
    meta.append(dict(path=p, kind=kind, scene=sid, cond=cond if kind in ("good", "bad") else "regular", mean=m, size=(H, W)))
    if q % 25 == 0: log(f"  query {q}/{n}")

out = f"{CACHE}/results/{a.cat}{a.tag}.pt"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(dict(cat=a.cat, backbone=a.backbone, layers=layers, gains=gains, K=K, C=C, grid=(ht, wt), per=per, ref_mean=ref_mean,
                meta=meta, gt=torch.stack(gts), dA=dA, iA=iA, dB=dB, iB=iB, dAn=dAn, D=D, I=I,
                idxA=idxA, idxB=idxB, rA=rA, rB=rB, gainB=gainB, ident=ident, img_gain=img_gain, proto=proto), out)
log("saved", out)
