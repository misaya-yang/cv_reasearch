"""Nuisance-whitened matching (MNF/MAF-style): metric from spatial-difference covariance.

Hypothesis: token differences between spatial neighbours estimate task-irrelevant nuisance
(position, local detail). Whitening that covariance (NOT the total covariance) should improve
cross-image matching without flattening semantics. Compared against raw and positional projection.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = lab.subset(N, offset=1)


def diff_cov(x):
    """Second moment of horizontal+vertical neighbour differences of a (4096, d) token grid."""
    g = x.reshape(64, 64, -1)
    dh = (g[:, 1:] - g[:, :-1]).reshape(-1, g.shape[-1])
    dv = (g[1:] - g[:-1]).reshape(-1, g.shape[-1])
    d = torch.cat([dh, dv]).cpu().double()
    return d.T @ d / d.shape[0]


def inv_sqrt(S, eps):
    ev, V = torch.linalg.eigh(S)
    ev = ev.clamp_min(0)
    w = (ev + eps * ev.mean()).rsqrt()
    return ((V * w) @ V.T).float().to(DEV)


def topk_mean(sim, k):
    return sim.topk(max(1, min(k, sim.shape[1])), dim=1).values.mean(1)


def readouts(r, q, cr, tag, out, zero):
    """r, q: centred+transformed unit features."""
    fg, bg = cr > 0.5, cr < 0.5
    if not fg.any():
        fg = cr >= cr.max()
    w = cr if cr.sum() > 1e-6 else fg.float()
    muF = lab.unit((r * w[:, None]).sum(0))
    out[f"{tag}|proto"] = q @ muF
    sim = q @ r.T
    f = topk_mean(sim[:, fg], 3)
    b = topk_mean(sim[:, bg], 3) if bg.any() else torch.zeros_like(f)
    out[f"{tag}|tok3"] = f - b
    zero.add(f"{tag}|tok3")
    out[f"{tag}|tok3F"] = f


M, A, Z = {}, {}, {}
t0 = time.time()
for k, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    cr = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    R, Q = lab.unit(lab.feats(b, "reference")), lab.unit(lab.feats(b, "query"))
    out, zero = {}, set()
    mu = 0.5 * (R.mean(0) + Q.mean(0))
    readouts(lab.unit(R - mu), lab.unit(Q - mu), cr, "raw", out, zero)
    rd, qd = lab.debias(R), lab.debias(Q)
    mud = 0.5 * (rd.mean(0) + qd.mean(0))
    readouts(lab.unit(rd - mud), lab.unit(qd - mud), cr, "apd", out, zero)
    S = 0.5 * (diff_cov(R) + diff_cov(Q))
    for eps in (0.03, 0.3, 3.0):
        W = inv_sqrt(S, eps)
        readouts(lab.unit((R - mu) @ W), lab.unit((Q - mu) @ W), cr, f"mnf{eps}", out, zero)
    Sd = 0.5 * (diff_cov(rd) + diff_cov(qd))
    W = inv_sqrt(Sd, 0.3)
    readouts(lab.unit((rd - mud) @ W), lab.unit((qd - mud) @ W), cr, "apd+mnf0.3", out, zero)
    for name, s in out.items():
        s = up(s.reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(s, gt)
        M.setdefault(name, Meter()).add(b, i, u)
        A.setdefault(name, lab.defaultdict(list))[d].append(lab.auroc(s, gt))
        if name in zero:
            Z.setdefault(name, Meter()).add(b, *iu(s > 0, gt))
    lab.free()
    if k % 50 == 0:
        print(k, round(time.time() - t0, 1), flush=True)

print(f"\noracle-threshold mIoU | AUROC | mIoU at zero, n={N}/domain")
print(f"{'features|readout':20s}" + "".join(f"{d:>22s}" for d in DOMAINS))
for name in M:
    r = M[name].row()
    zr = Z[name].row() if name in Z else None
    print(f"{name:20s}" + "".join(f"{r[d][0]:8.2f} {np.nanmean(A[name][d]):.3f} " + (f"{zr[d][0]:6.2f}" if zr else "   -  ") for d in DOMAINS))
