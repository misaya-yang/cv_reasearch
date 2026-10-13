"""Dump 4-neighbour token dissimilarities (query and reference) for boundary-contrast studies. CPU."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(6)
N, OUT = int(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for b in lab.subset(N):
    f = OUT / f"{b['index']:06d}.npz"
    if f.exists():
        continue
    out = {}
    for role, tag in (("query", "q"), ("reference", "r")):
        g = lab.debias(lab.feats(b, role)).reshape(64, 64, -1)
        out[tag + "h"] = (1 - (g[:, 1:] * g[:, :-1]).sum(-1)).numpy().astype(np.float32)   # (64, 63)
        out[tag + "v"] = (1 - (g[1:] * g[:-1]).sum(-1)).numpy().astype(np.float32)         # (63, 64)
    np.savez(f, **out)
print("done")
