"""Does the reference tell the *relative* level of the task boundary?

alpha_R: IoU-optimal level of the reference's own (cross-fitted) score map against M_R, expressed
as a fraction of that map's dynamic range.  Rule: cut the query map at the same fraction of its
own range.  Controls: best constant fraction, oracle.  Also reports corr(alpha_R, alpha*_Q).
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows = lab.subset(N)


def best_level(s, y):
    """Soft-IoU-optimal threshold of token scores s against soft labels y (64-grid)."""
    order = torch.argsort(s, descending=True)
    inter = torch.cumsum(y[order], 0)
    union = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32) + y.sum() - inter
    k = int((inter / union).argmax())
    return float(s[order[k]]), float((inter / union)[k])


def rng(s, kind):
    if kind == "minmax":
        return float(s.min()), float(s.max())
    return float(s.median()), float(torch.quantile(s, 0.995))


M, stats = {}, lab.defaultdict(list)
g = torch.Generator().manual_seed(0)
for kk, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    muR = r.mean(0)
    rc, qc = lab.unit(r - muR), lab.unit(q - muR)
    proto = lab.unit((r * w[:, None]).sum(0) / w.sum() - muR)
    sq = qc @ proto
    # cross-fitted reference self-scores: two random halves of the foreground mass
    half = (torch.rand(4096, generator=g) < 0.5).to(DEV).float()
    pa = lab.unit((r * (w * half)[:, None]).sum(0) / (w * half).sum().clamp_min(1e-6) - muR)
    pb = lab.unit((r * (w * (1 - half))[:, None]).sum(0) / (w * (1 - half)).sum().clamp_min(1e-6) - muR)
    sr_cf = torch.where(half > 0.5, rc @ pb, rc @ pa)
    sr_in = rc @ proto
    SQ = up(sq.reshape(64, 64))
    i, u, tq = lab.best_threshold_iou(SQ, gt)
    M.setdefault("oracle", Meter()).add(b, i, u)
    for kind in ("minmax", "robust"):
        lo, hi = rng(sq, kind)
        aq = (tq - lo) / max(hi - lo, 1e-6)
        for const in (0.5, 0.6, 0.7):
            M.setdefault(f"{kind}|const{const}", Meter()).add(b, *iu(SQ > lo + const * (hi - lo), gt))
        for tag, sr in (("cf", sr_cf), ("in", sr_in)):
            tr, _ = best_level(sr, y)
            lr, hr = rng(sr, kind)
            ar = (tr - lr) / max(hr - lr, 1e-6)
            M.setdefault(f"{kind}|alphaR_{tag}", Meter()).add(b, *iu(SQ > lo + ar * (hi - lo), gt))
            M.setdefault(f"{kind}|alphaR_{tag}_avg.6", Meter()).add(b, *iu(SQ > lo + 0.5 * (ar + 0.6) * (hi - lo), gt))
            if tag == "cf":
                stats[(d, kind)].append((ar, aq, float(y.mean()), float(lab.coverage64(gt).mean())))
    lab.free()

print(f"class-pooled mIoU (per-episode mean), ranking = R-centred prototype, n={N}/domain, no CRF")
print(f"{'extent rule':26s}" + "".join(f"{d:>16s}" for d in DOMAINS))
for name in M:
    r = M[name].row()
    print(f"{name:26s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in DOMAINS))
print("\ncorr(alpha_R, alpha*_Q) | mean alpha_R | mean alpha*_Q | corr(alpha*_Q, log query area) | corr(alpha_R, log ref area)")
for kind in ("minmax", "robust"):
    for d in DOMAINS:
        a = np.array(stats[(d, kind)])
        c = lambda u, v: np.corrcoef(u, v)[0, 1]
        print(f"{kind:7s}{d:12s} {c(a[:,0], a[:,1]):+.3f} | {a[:,0].mean():.3f} | {a[:,1].mean():.3f} | {c(a[:,1], np.log(a[:,3]+1e-4)):+.3f} | {c(a[:,0], np.log(a[:,2]+1e-4)):+.3f}")
