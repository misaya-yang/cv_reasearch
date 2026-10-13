"""Extent by query-intrinsic conductance (sweep cut) along a fixed cross-image ranking.

Reference decides the order (which tokens are most target-like); the query's own affinity
graph decides where to stop: the prefix with the smallest conductance.  No label, no minmax.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *
from lab import kmeans, minmax

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows = lab.subset(N)
n = 4096


def knn_graph(x, k=16, sigma=0.1):
    """Symmetric kNN affinity with weights exp((cos-1)/sigma); dense (n, n), zero diagonal."""
    s = x @ x.T
    s.fill_diagonal_(-2)
    v, i = s.topk(k, dim=1)
    W = torch.zeros_like(s).scatter_(1, i, torch.exp((v - 1) / sigma))
    return torch.maximum(W, W.T)


def sweep(W, score):
    """Conductance of every prefix of the descending-score order. Returns (order, phi)."""
    order = torch.argsort(score, descending=True)
    Wp = W[order][:, order]
    deg = Wp.sum(1)
    vol = torch.cumsum(deg, 0)
    inside = torch.cumsum(2 * torch.tril(Wp, -1).sum(1), 0)
    cut = vol - inside
    phi = cut / torch.minimum(vol, vol[-1] - vol).clamp_min(1e-9)
    return order, phi


def pick(order, phi, lo=8, hi=int(0.7 * n)):
    k = lo + int(phi[lo:hi].argmin())
    m = torch.zeros(n, dtype=torch.bool, device=DEV)
    m[order[: k + 1]] = True
    return m


M = {}
t0 = time.time()
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    muR = r.mean(0)
    s_proto = lab.unit(q - muR) @ lab.unit((r * w[:, None]).sum(0) / w.sum() - muR)
    mu = 0.5 * (muR + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    C = kmeans(qc, 64, iters=6)
    assign = (qc @ C.T).argmax(1)
    P = torch.zeros(n, 64, device=DEV).scatter_(1, assign[:, None], 1.0)
    fwd = torch.softmax((qc @ rc.T) / 0.05, dim=1) @ y
    s_reg = ((P.T @ fwd) / P.sum(0).clamp_min(1))[assign] + 1e-3 * fwd
    for sname, s in (("proto", s_proto), ("fwdreg", s_reg)):
        S = up(s.reshape(64, 64))
        M.setdefault(f"{sname}|minmax.5", Meter()).add(b, *iu(minmax(S) > 0.5, gt))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault(f"{sname}|oracle", Meter()).add(b, i, u)
        # token-level oracle prefix (what a perfect stop on the 64-grid order would give after upsampling)
        for gname, (k, sig) in (("k16s.1", (16, 0.1)), ("k32s.2", (32, 0.2))):
            W = knn_graph(qc, k, sig)
            order, phi = sweep(W, s)
            m = pick(order, phi)
            M.setdefault(f"{sname}|sweep_{gname}", Meter()).add(b, *iu(up(m.float().reshape(64, 64)) > 0.5, gt))
    lab.free()
    if kk % 100 == 0:
        print(kk, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU), n={N}/domain, no CRF")
print(f"{'ranking|extent':24s}" + "".join(f"{d:>16s}" for d in DOMAINS))
for name in M:
    r = M[name].row()
    print(f"{name:24s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in DOMAINS))
