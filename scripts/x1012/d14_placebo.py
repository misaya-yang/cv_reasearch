"""Placebo-calibrated matching: extent as multiple testing.

For every query token, its similarity to the target prototype is standardised against its own
similarities to K placebo prototypes (reference background concepts):
    z_j = (s_F(j) - mean_b s_b(j)) / std_b s_b(j)
Each token is its own control, so token/image-specific offsets cancel and z has a null scale.
Extent rules: fixed z level, Benjamini-Hochberg on Gaussian p-values, empirical-rank p-values.
"""
import sys, time, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *
from lab import kmeans, minmax

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows = lab.subset(N)
n = 4096


def bh(p, q):
    """Benjamini-Hochberg rejection set at level q."""
    order = torch.argsort(p)
    ok = p[order] <= q * torch.arange(1, p.numel() + 1, device=p.device) / p.numel()
    k = int(ok.nonzero().max()) + 1 if ok.any() else 0
    m = torch.zeros_like(p, dtype=torch.bool)
    m[order[:k]] = True
    return m


def gauss_p(z):
    return 0.5 * torch.erfc(z / math.sqrt(2))


M = {}
t0 = time.time()
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    mu = r.mean(0)
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    proto = lab.unit((r * w[:, None]).sum(0) / w.sum() - mu)
    sF = qc @ proto
    C = kmeans(qc, 64, iters=6)
    assign = (qc @ C.T).argmax(1)
    P = torch.zeros(n, 64, device=DEV).scatter_(1, assign[:, None], 1.0)
    cnt = P.sum(0).clamp_min(1)
    out = {}
    for KB in (32, 96):
        PB = kmeans(rc, KB, iters=6, w=(1 - y).clamp(0, 1) * (y < 0.5))
        sB = qc @ PB.T                                           # (n, KB) placebo similarities
        z = (sF - sB.mean(1)) / sB.std(1).clamp_min(1e-6)
        rank_p = (1 + (sB >= sF[:, None]).sum(1).float()) / (KB + 1)
        zr = ((P.T @ z) / cnt)[assign]                           # region-averaged z
        out[f"z{KB}"] = z
        out[f"z{KB}reg"] = zr
        for name, zz in ((f"z{KB}", z), (f"z{KB}reg", zr)):
            Z = up(zz.reshape(64, 64))
            for c in (1.0, 2.0, 3.0, 4.0):
                M.setdefault(f"{name}>{c}", Meter()).add(b, *iu(Z > c, gt))
            for qq in (0.01, 0.1):
                m = bh(gauss_p(zz), qq)
                M.setdefault(f"{name}|BH{qq}", Meter()).add(b, *iu(up(m.float().reshape(64, 64)) > 0.5, gt))
            i, u, _ = lab.best_threshold_iou(Z, gt)
            M.setdefault(f"{name}|oracle", Meter()).add(b, i, u)
        M.setdefault(f"z{KB}|margin>0", Meter()).add(b, *iu(up((sF - sB.max(1).values).reshape(64, 64)) > 0, gt))
        M.setdefault(f"z{KB}|rankBH0.1", Meter()).add(b, *iu(up(bh(rank_p, 0.1).float().reshape(64, 64)) > 0.5, gt))
    M.setdefault("FoRIS_crf_output", Meter()).add(b, *iu(lab.foris_cli(b).to(DEV), gt))
    lab.free()
    if kk % 100 == 0:
        print(kk, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU), n={N}/domain, no CRF")
print(f"{'rule':20s}" + "".join(f"{d:>16s}" for d in DOMAINS) + "    mean")
for name in M:
    r = M[name].row()
    print(f"{name:20s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in DOMAINS) + f"{np.mean([r[d][0] for d in DOMAINS]):8.1f}")
