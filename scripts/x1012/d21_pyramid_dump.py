"""Whole-image pyramid dump: zoom-OUT observations (never tested in this project).

Reference and query are encoded as whole images at 256, 512, 768 px (1024 is cached).
For every (reference scale a, query scale b) we save the contrast-prototype score map of the
query grid at scale b:  cos(q_b - mu_a, muF_a - mu_a), with the positional projection applied.
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab, enc
from lab import *

N, OUT = int(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SCALES = (256, 512, 768, 1024)
rows = lab.subset(N, offset=1) if len(sys.argv) < 4 else lab.ext_subset(N, tuple(sys.argv[3].split(",")))
tfs = {s: enc.transform(s) for s in SCALES[:-1]}
t0 = time.time()
for kk, b in enumerate(rows):
    f = OUT / f"{b['index'] if isinstance(b['index'], str) else format(b['index'], '06d')}.npz"
    if f.exists():
        continue
    mr = lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)
    imr, imq = enc.rgb(b, "reference"), enc.rgb(b, "query")
    R, Q, Y = {}, {}, {}
    for s in SCALES:
        g = s // 16
        if s == 1024:
            R[s], Q[s] = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
        else:
            R[s] = lab.debias(enc.layers(tfs[s](imr), [24])[24])
            Q[s] = lab.debias(enc.layers(tfs[s](imq), [24])[24])
        Y[s] = F.adaptive_avg_pool2d(mr[None, None].float().cpu(), g)[0, 0].flatten().to(DEV)
    out = {}
    for a in SCALES:
        y = Y[a]
        w = y if y.sum() > 1e-6 else (y >= y.max()).float()
        mu = R[a].mean(0)
        proto = lab.unit((R[a] * w[:, None]).sum(0) / w.sum() - mu)
        for bq in SCALES:
            out[f"r{a}_q{bq}"] = (lab.unit(Q[bq] - mu) @ proto).cpu().numpy().astype(np.float32)
    np.savez(f, **out)
    lab.free()
    if kk % 30 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)
print("done", time.time() - t0)
