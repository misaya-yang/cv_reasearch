"""Layer scan: per-block task visibility under one reference (fresh encodes, score maps saved).

For each block l we save two 64x64 query score maps computed from that block alone:
  cR : cos(q - mu_R, mu_F - mu_R)                       (contrast prototype)
  nn : mean top-3 sim to reference FG - mean top-3 sim to reference BG (local competition)
No positional projection is applied here (the native basis exists for block 24 only).
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab, enc
from lab import *

N = int(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
LAYERS = [4, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24]
rows = lab.subset(N, offset=1)
tf = enc.transform()
t0 = time.time()
for k, b in enumerate(rows):
    f = OUT / f"{b['index']:06d}.npz"
    if f.exists():
        continue
    cr = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    R = enc.layers(tf(enc.rgb(b, "reference")), LAYERS)
    Q = enc.layers(tf(enc.rgb(b, "query")), LAYERS)
    fg, bg = cr > 0.5, cr < 0.5
    if not fg.any():
        fg = cr >= cr.max()
    w = cr if cr.sum() > 1e-6 else fg.float()
    out = {}
    for l in LAYERS:
        r, q = lab.unit(R[l]), lab.unit(Q[l])
        mu = r.mean(0)
        muF = (r * w[:, None]).sum(0) / w.sum()
        out[f"cR/{l}"] = (lab.unit(q - mu) @ lab.unit(muF - mu)).reshape(64, 64).cpu().numpy()
        mj = 0.5 * (mu + q.mean(0))
        rc, qc = lab.unit(r - mj), lab.unit(q - mj)
        sim = qc @ rc.T
        sf = sim[:, fg].topk(min(3, int(fg.sum())), dim=1).values.mean(1)
        sb = sim[:, bg].topk(min(3, int(bg.sum())), dim=1).values.mean(1) if bg.any() else torch.zeros_like(sf)
        out[f"nn/{l}"] = (sf - sb).reshape(64, 64).cpu().numpy()
        out[f"nnF/{l}"] = sf.reshape(64, 64).cpu().numpy()
    np.savez(f, **out)
    if k % 20 == 0:
        print(k, len(rows), round(time.time() - t0, 1), flush=True)
print("done", time.time() - t0)
