"""Spatial-arrangement cues of the dense correspondence (the one channel no earlier cue uses).

Per query token j with forward nearest neighbour i*(j) in the reference:
  vote      label of i*(j)
  coh       fraction of the 8 neighbours whose matches stay within 3 reference tokens of i*(j)
  coh_lab   fraction of the 8 neighbours whose matches carry the same label as i*(j)
  mnn       1 if j is the backward nearest neighbour of i*(j)
  hitF/hitB number of reference FG / BG tokens whose backward nearest neighbour is j
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

OUT = lab.ASSETS / "a/x1012/geo300"
OUT.mkdir(parents=True, exist_ok=True)
F300 = lab.ASSETS / "a/x1012/fields300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
yy, xx = torch.meshgrid(torch.arange(64, device=DEV), torch.arange(64, device=DEV), indexing="ij")
pos = torch.stack([yy.flatten(), xx.flatten()], 1).float()
t0 = time.time()
for kk, b in enumerate(rows):
    f = OUT / f"{b['index']:06d}.npz"
    if f.exists():
        continue
    y = (lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten() > 0.5).float()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    fs, fi = sim.max(1)
    bs, bi = sim.max(0)
    u = pos[fi].reshape(64, 64, 2).permute(2, 0, 1)[None]
    lab_j = y[fi].reshape(1, 1, 64, 64)
    up_ = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lp = F.pad(lab_j, (1, 1, 1, 1), mode="replicate")
    coh = torch.zeros(64, 64, device=DEV)
    cl = torch.zeros(64, 64, device=DEV)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            d = (up_[0, :, dy:dy + 64, dx:dx + 64] - u[0]).abs().amax(0)
            coh += (d <= 3).float() / 8
            cl += (lp[0, 0, dy:dy + 64, dx:dx + 64] == lab_j[0, 0]).float() / 8
    mnn = (bi[fi] == torch.arange(4096, device=DEV)).float()
    hitF = torch.zeros(4096, device=DEV).index_add_(0, bi, y)
    hitB = torch.zeros(4096, device=DEV).index_add_(0, bi, 1 - y)
    np.savez(f, vote=y[fi].cpu().numpy().astype(np.float32), coh=coh.flatten().cpu().numpy(), coh_lab=cl.flatten().cpu().numpy(),
             mnn=mnn.cpu().numpy(), hitF=hitF.cpu().numpy(), hitB=hitB.cpu().numpy(), fs=fs.cpu().numpy(), fi=fi.cpu().numpy().astype(np.int16))
    lab.free()
    if kk % 200 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)
print("done", round(time.time() - t0, 1))
