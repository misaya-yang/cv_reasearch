"""Dump bidirectional attention fields between query and reference tokens (64-grid).

fwdF[tau]  : sum_i P(i|j) y_i          query token's attention mass on reference foreground
backF[tau] : sum_i y_i P(j|i) / sum y  reference foreground's attention mass on query token j
backB[tau] : same for reference background
Features: positional projection, centred by the mean of both images, unit norm.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

N, OUT = int(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for kk, b in enumerate(lab.subset(N)):
    f = OUT / f"{b['index']:06d}.npz"
    if f.exists():
        continue
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T                  # (nq, nr)
    out = {}
    for tau in (0.02, 0.05):
        Pq = torch.softmax(sim / tau, dim=1)                     # query -> reference
        Pr = torch.softmax(sim / tau, dim=0)                     # reference -> query (columns sum to 1)
        out[f"fwdF{tau}"] = (Pq @ y).cpu().numpy()
        out[f"backF{tau}"] = (Pr @ y / y.sum().clamp_min(1e-6)).cpu().numpy()
        out[f"backB{tau}"] = (Pr @ (1 - y) / (1 - y).sum().clamp_min(1e-6)).cpu().numpy()
    np.savez(f, **out)
    lab.free()
print("done", time.time() - t0)
