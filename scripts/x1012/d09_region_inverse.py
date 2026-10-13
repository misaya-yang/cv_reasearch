"""Region-level inverse label transport.

Unknown: one label z_k in [0, 1] per query region. Measurement: the reference mask y, read by
every reference token through its attention over query regions:  y ~ (A P) z.
Solve the class-balanced box-constrained least squares; output z at the natural 0.5 level.
Forward control: region mean of the usual query->reference label transfer.
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


def box_ls(G, h, eps, iters=300):
    """min z'Gz - 2h'z + eps|z|^2 over [0,1]^K by projected gradient (fixed step 1/L)."""
    K = G.shape[0]
    G = G + eps * torch.eye(K, device=G.device)
    L = torch.linalg.eigvalsh(G.cpu().double()).max().float().to(G.device)
    z = torch.full((K,), 0.0, device=G.device)
    for _ in range(iters):
        z = (z - (G @ z - h) / L).clamp(0, 1)
    return z


def solve(rc, qc, y, K, tau):
    C = kmeans(qc, K, iters=6)
    assign = (qc @ C.T).argmax(1)
    P = torch.zeros(qc.shape[0], K, device=DEV).scatter_(1, assign[:, None], 1.0)
    sim = rc @ qc.T
    AP = torch.softmax(sim / tau, dim=1) @ P                     # (n_ref, K) attention mass per region
    w = y / y.sum().clamp_min(1e-6) + (1 - y) / (1 - y).sum().clamp_min(1e-6)
    G = AP.T @ (AP * w[:, None])
    h = AP.T @ (w * y)
    eps = 1e-3 * torch.diagonal(G).mean()
    z_inv = box_ls(G, h, eps)
    fwd_tok = torch.softmax(sim.T / tau, dim=1) @ y              # query->reference label transfer
    z_fwd = (P.T @ fwd_tok) / P.sum(0).clamp_min(1)
    return z_inv[assign], z_fwd[assign], fwd_tok


M, Zr = {}, {}
t0 = time.time()
for k, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    out = {}
    for K in (64, 160):
        for tau in (0.02, 0.05):
            zi, zf, ft = solve(rc, qc, y, K, tau)
            out[f"inv_K{K}_t{tau}"] = zi
            out[f"fwd_K{K}_t{tau}"] = zf
            if K == 64:
                out[f"fwdtok_t{tau}"] = ft
    for name, s in out.items():
        S = up(s.reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault(name, Meter()).add(b, i, u)
        Zr.setdefault(name, Meter()).add(b, *iu(S > 0.5, gt))
    Zr.setdefault("FoRIS_crf_output", Meter()).add(b, *iu(lab.foris_cli(b).to(DEV), gt))
    lab.free()
    if k % 100 == 0:
        print(k, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU): natural 0.5 || oracle threshold, n={N}/domain, no CRF")
print(f"{'method':18s}" + "".join(f"{d:>26s}" for d in DOMAINS))
for name in list(M) + ["FoRIS_crf_output"]:
    z = Zr[name].row()
    r = M[name].row() if name in M else None
    print(f"{name:18s}" + "".join(f"  {z[d][0]:5.1f} ({z[d][1]:5.1f}) ||" + (f" {r[d][0]:5.1f}   " if r else "    -    ") for d in DOMAINS))
