#!/usr/bin/env python3
"""CPU step: is the nearest-neighbour distance mis-calibrated across normal content?

The expected distance of a NORMAL patch to its nearest bank vector depends on how densely that kind of content is
covered by the bank. Here that expectation is estimated per bank vector (from held-out normal images that matched it)
and per image position, and the raw distance is divided by / reduced by it. Estimation uses one half of the held-out
normals (the train part), the threshold uses the other half (validation), so nothing is fitted on test images.

Read-outs:  raw | proto-ratio d/mu_j | proto-diff d-mu_j | pos-z (d-mu(pos))/sd(pos) | radius-ratio d/r_j (needs rA/rB)
"""
import argparse, os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE
from adx.metrics import seg_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--cats", default="vial"); ap.add_argument("--tag", default=""); ap.add_argument("--bank", default="B")
ap.add_argument("--kappa", type=float, default=2.0)
a = ap.parse_args()
GROUPS = ["regular", "shifted", "all"]
tab = {}
for cat in a.cats.split(","):
    R = torch.load(f"{CACHE}/results/{cat}{a.tag}.pt", weights_only=False)
    meta, gt = R["meta"], R["gt"].numpy(); size = gt.shape[1:]
    d = R["d" + a.bank].float(); ix = R["i" + a.bank].long()        # (n, L, h, w)
    n, L, h, w = d.shape; K = R["K"]
    kind = np.array([m["kind"] for m in meta]); cond = np.array([m["cond"] for m in meta])
    fit = torch.from_numpy(kind == "thr"); cal = kind == "val"; test = np.isin(kind, ["good", "bad"])
    reads = {"raw": d.mean(1)}
    # expected normal distance per bank vector, shrunk towards the layer mean
    mu = torch.zeros(L, K)
    for l in range(L):
        dl, il = d[fit, l].reshape(-1), ix[fit, l].reshape(-1)
        s = torch.zeros(K).index_add_(0, il, dl); c = torch.zeros(K).index_add_(0, il, torch.ones_like(dl))
        mu[l] = (s + a.kappa * dl.mean()) / (c + a.kappa)
    cover = float((c > 0).float().mean())
    mu_q = torch.stack([mu[l][ix[:, l]] for l in range(L)], 1)      # (n, L, h, w)
    reads["proto-ratio"] = (d / mu_q).mean(1)
    reads["proto-diff"] = (d - mu_q).mean(1)
    pm, ps = d[fit].mean(1).mean(0), d[fit].mean(1).std(0) + 1e-6   # per position, layer-mean map
    reads["pos-z"] = (d.mean(1) - pm) / ps
    reads["pos-diff"] = d.mean(1) - pm
    if "r" + a.bank in R:
        r = torch.stack([R["r" + a.bank][l].float() for l in range(L)])             # (L, K, 10)
        for k in (1, 5, 10):
            rq = torch.stack([r[l][:, k - 1][ix[:, l]] for l in range(L)], 1)
            reads[f"radius{k}-ratio"] = (d / (rq + 1e-6)).mean(1)
    res = {}
    for name, m in reads.items():
        s = F.interpolate(m[:, None], size=size, mode="bilinear", align_corners=False)[:, 0].numpy()
        thr = float(np.percentile(s[cal], 95) * 1.421) if name in ("raw", "proto-ratio") or name.startswith("radius") else float(np.percentile(s[cal], 99.5))
        res[name] = {}
        for g in GROUPS:
            sel = test & ((cond == "regular") if g == "regular" else (cond != "regular") if g == "shifted" else True)
            res[name][g] = seg_metrics(list(s[sel]), list(gt[sel]), thr)
    tab[cat] = res
    print(f"\n=== {cat} bank {a.bank}; share of bank vectors with a held-out normal match (last layer): {cover:.2f}")
    for metric in ["ap", "aupro", "f1max", "f1"]:
        print(f"   -- {metric:6s}" + "".join(f"{g:>10s}" for g in GROUPS))
        for name in res:
            print(f"      {name:14s}" + "".join(f"{100 * res[name][g][metric]:10.2f}" for g in GROUPS))
if len(tab) > 1:
    print("\n=== mean over categories")
    for metric in ["ap", "aupro", "f1max", "f1"]:
        print(f"   -- {metric:6s}" + "".join(f"{g:>10s}" for g in GROUPS))
        for name in next(iter(tab.values())):
            print(f"      {name:14s}" + "".join(f"{np.mean([100 * tab[c][name][g][metric] for c in tab if name in tab[c]]):10.2f}" for g in GROUPS))
