"""How far do global-constant extent rules go, for several normalisations of one ranking?

Scores: jointly/R-centred prototype contrast and region-averaged forward label transfer.
Normalisations: minmax, (s-median)/(q995-median), z-score, raw cosine level, raw posterior level.
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
GRID = {
    "minmax": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "robust": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "zscore": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "raw": None,
}
RAW = {"proto": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "fwdreg": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

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
    s_reg = ((P.T @ fwd) / P.sum(0).clamp_min(1))[assign]
    for sname, s in (("proto", s_proto), ("fwdreg", s_reg)):
        S = up(s.reshape(64, 64))
        f = s
        norms = {
            "minmax": (S - f.min()) / (f.max() - f.min()).clamp_min(1e-6),
            "robust": (S - f.median()) / (torch.quantile(f, 0.995) - f.median()).clamp_min(1e-6),
            "zscore": (S - f.mean()) / f.std().clamp_min(1e-6),
            "raw": S,
        }
        for nname, Sn in norms.items():
            for a in (GRID[nname] or RAW[sname]):
                M.setdefault((sname, nname, a), Meter()).add(b, *iu(Sn > a, gt))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault((sname, "oracle", 0), Meter()).add(b, i, u)
    lab.free()

print(f"class-pooled mIoU, n={N}/domain, no CRF")
print(f"{'score norm level':26s}" + "".join(f"{d:>12s}" for d in DOMAINS) + "   mean")
for key, m in M.items():
    r = m.row()
    v = [r[d][0] for d in DOMAINS]
    print(f"{key[0]:8s}{key[1]:8s}{key[2]:<10}" + "".join(f"{x:12.1f}" for x in v) + f"{np.mean(v):7.1f}")
