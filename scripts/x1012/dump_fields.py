"""Dump per-episode 64-grid fields for fast offline extent studies (no GPU needed afterwards)."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *
from lab import kmeans

N, OUT = int(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rows = lab.subset(N)
g = torch.Generator().manual_seed(0)
t0 = time.time()
for kk, b in enumerate(rows):
    f = OUT / f"{b['index']:06d}.npz"
    half = (torch.rand(4096, generator=g) < 0.5).to(DEV).float()
    if f.exists():
        continue
    gt = lab.mask1024(b["row"]["query_mask_path"])
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    muR = r.mean(0)
    rc, qc = lab.unit(r - muR), lab.unit(q - muR)
    proto = lab.unit((r * w[:, None]).sum(0) / w.sum() - muR)
    pa = lab.unit((r * (w * half)[:, None]).sum(0) / (w * half).sum().clamp_min(1e-6) - muR)
    pb = lab.unit((r * (w * (1 - half))[:, None]).sum(0) / (w * (1 - half)).sum().clamp_min(1e-6) - muR)
    mu = 0.5 * (muR + q.mean(0))
    rj, qj = lab.unit(r - mu), lab.unit(q - mu)
    C = kmeans(qj, 64, iters=6)
    sim = qj @ rj.T
    np.savez(f,
             sq=(qc @ proto).cpu().numpy(), sr=(rc @ proto).cpu().numpy(),
             sr_cf=torch.where(half > 0.5, rc @ pb, rc @ pa).cpu().numpy(),
             fwd=(torch.softmax(sim / 0.05, dim=1) @ y).cpu().numpy(),
             assign=(qj @ C.T).argmax(1).cpu().numpy().astype(np.int16),
             y=y.cpu().numpy(), gt=np.packbits(gt.numpy()),
             foris_score=np.load(b["score64"]["path"])["score"].astype(np.float32),
             foris_iu=np.array(lab.iu(lab.foris_cli(b), gt)))
    lab.free()
    if kk % 100 == 0:
        print(kk, round(time.time() - t0, 1), flush=True)
print("done", time.time() - t0)
