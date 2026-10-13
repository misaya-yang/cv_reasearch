"""Label transfer by (unbalanced) entropic transport between query and reference tokens.

Query marginal is kept exact (every query token must be explained); the reference marginal is
relaxed with exponent lam in [0, 1]: lam = 0 is the usual row-softmax label transfer, lam = 1
is balanced transport (hard capacity per reference token => explaining-away between query
tokens that compete for the same reference foreground).
Decision statistics: P(F|j) at 0.5, lift P(F|j) > P(F), and oracle level / AUROC.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = lab.subset(N, offset=1)


def transport(sim, eps, lam, iters=60):
    """Return the plan diag(u) K diag(v) with exact row marginal and lam-relaxed column marginal."""
    K = torch.exp((sim - sim.max()) / eps)
    n, m = K.shape
    a, b = torch.full((n,), 1.0 / n, device=DEV), torch.full((m,), 1.0 / m, device=DEV)
    v = torch.ones(m, device=DEV)
    for _ in range(iters):
        u = a / (K @ v).clamp_min(1e-30)
        v = (b / (K.T @ u).clamp_min(1e-30)) ** lam
    return u[:, None] * K * v[None, :]


M = {}
A = {}
t0 = time.time()
for kk, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    for eps in (0.05,):
        for lam in (0.0, 0.5, 0.9, 1.0):
            Pi = transport(sim, eps, lam)
            pf = (Pi @ y) / Pi.sum(1).clamp_min(1e-30)
            prior = float((Pi @ y).sum() / Pi.sum())
            S = up(pf.reshape(64, 64))
            name = f"lam={lam}"
            i, u, _ = lab.best_threshold_iou(S, gt)
            M.setdefault(name + " | oracle", Meter()).add(b, i, u)
            M.setdefault(name + " | P(F|j)>0.5", Meter()).add(b, *iu(S > 0.5, gt))
            M.setdefault(name + " | lift>1", Meter()).add(b, *iu(S > prior, gt))
            M.setdefault(name + " | lift>2", Meter()).add(b, *iu(S > min(2 * prior, 0.95), gt))
            M.setdefault(name + " | minmax 0.5", Meter()).add(b, *iu(lab.minmax(S) > 0.5, gt))
            A.setdefault(name, lab.defaultdict(list))[d].append(lab.auroc(S, gt))
    M.setdefault("FoRIS_crf_output", Meter()).add(b, *iu(lab.foris_cli(b).to(DEV), gt))
    lab.free()
    if kk % 60 == 0:
        print(kk, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU), n={N}/domain, no CRF")
print(f"{'transport | rule':28s}" + "".join(f"{d:>16s}" for d in DOMAINS) + "    mean")
for name in M:
    r = M[name].row()
    extra = "".join(f" {np.nanmean(A[name.split(' | ')[0]][d]):.3f}" for d in DOMAINS) if name.endswith("oracle") else ""
    print(f"{name:28s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in DOMAINS) + f"{np.mean([r[d][0] for d in DOMAINS]):8.1f} {extra}")
