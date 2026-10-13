"""Ranking benchmark for token score functions (AUROC + GT-optimal-threshold IoU).

Diagnostic only: the oracle threshold isolates ranking quality from extent selection.
"""
import sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rows = lab.subset(N, offset=3)


def auroc(score, gt):
    s, g = score.flatten(), gt.flatten()
    idx = torch.randperm(s.numel(), device=s.device, generator=None)[:60000]
    s, g = s[idx], g[idx]
    if g.all() or not g.any():
        return float("nan")
    r = torch.argsort(torch.argsort(s)).float()
    n1, n0 = g.sum().float(), (~g).sum().float()
    return float((r[g].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def eig(x):
    """Eigen-decomposition of the token covariance of x (n, d), on CPU float64."""
    xc = (x - x.mean(0)).cpu().double()
    ev, V = torch.linalg.eigh(xc.T @ xc / x.shape[0])
    return ev.float().to(DEV).clamp_min(0), V.float().to(DEV)


def mf(q, target, mu, ev, V, alpha, ace=False):
    """Regularised matched filter of signature `target` against background (mu, Sigma)."""
    lam = alpha * ev.mean()
    qc, tc = (q - mu) @ V, (target - mu) @ V
    a = 1.0 / (ev + lam)
    s = (qc * a) @ tc
    if ace:
        s = s / ((qc * qc * a).sum(1).sqrt() * (tc * tc * a).sum().sqrt()).clamp_min(1e-12)
    return s


def scores(r, q, cr):
    w = cr.clone()
    if w.sum() < 1e-6:
        w[cr.argmax()] = 1
    muF = (r * w[:, None]).sum(0) / w.sum()
    muB = (r * (1 - w)[:, None]).sum(0) / (1 - w).sum().clamp_min(1e-6)
    out = {"proto": q @ lab.unit(muF)}
    muQ, muR = q.mean(0), r.mean(0)
    out["proto_centQ"] = lab.unit(q - muQ) @ lab.unit(muF - muQ)
    out["proto_centR"] = lab.unit(q - muR) @ lab.unit(muF - muR)
    out["fg_minus_bg"] = q @ (lab.unit(muF) - lab.unit(muB))
    evQ, VQ = eig(q)
    evR, VR = eig(r)
    for a in (0.03, 0.1, 0.3, 1.0, 3.0):
        out[f"mfQ_{a}"] = mf(q, muF, muQ, evQ, VQ, a)
        out[f"aceQ_{a}"] = mf(q, muF, muQ, evQ, VQ, a, ace=True)
    for a in (0.1, 1.0):
        out[f"ridgeR_{a}"] = mf(q, muF, muB, evR, VR, a)  # reference-side Fisher/ridge control
        out[f"mfR_{a}"] = mf(q, muF, muR, evR, VR, a)
    return out


M, A = {}, {}
t0 = time.time()
for k, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    cr = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    for name, s in scores(r, q, cr).items():
        s = up(s.reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(s, gt, n=48)
        M.setdefault(name, Meter()).add(b, i, u)
        A.setdefault(name, lab.defaultdict(list))[d].append(auroc(s, gt))
    if k % 100 == 0:
        print(k, round(time.time() - t0, 1), flush=True)

print(f"\noracle-threshold class-pooled mIoU | AUROC, n={N} per domain")
print(f"{'score':16s}" + "".join(f"{d:>16s}" for d in DOMAINS))
for name in M:
    r = M[name].row()
    print(f"{name:16s}" + "".join(f"{r[d][0]:8.2f} {np.nanmean(A[name][d]):.4f}" for d in DOMAINS))
