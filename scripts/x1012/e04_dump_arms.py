"""Dump complete 1024 masks of the candidate voters for main and extended dev episodes.

  G     MEAN (locked repo method) -- recomputed only where no archived mask exists (extended sets)
  T     coverage transfer, bilinear, min-max > 0.5          (token level)
  Treg  the same averaged over 64 k-means query regions
plus reference-mask statistics that need no query information (share of FG area lying in impure tokens).
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab, repo_methods as rm
from lab import DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = lab.ASSETS / "a/x1012/arms"
OUT.mkdir(parents=True, exist_ok=True)
rows = (lab.subset(N) if N > 0 else lab.bindings()) + lab.ext_subset(None)
key = lambda b: b["index"] if isinstance(b["index"], str) else format(b["index"], "06d")
t0 = time.time()
for kk, b in enumerate(rows):
    f = OUT / f"{key(b)}.npz"
    if f.exists():
        continue
    mr = lab.mask1024(b["row"]["reference_mask_path"])
    y = lab.coverage64(mr.to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    c = torch.softmax((lab.unit(q - mu) @ lab.unit(r - mu).T) / 0.05, dim=1) @ y
    C = lab.kmeans(lab.unit(q), 64, iters=6)
    oh = torch.nn.functional.one_hot((lab.unit(q) @ C.T).argmax(1), 64).float()
    creg = oh @ ((oh.T @ c) / oh.sum(0).clamp_min(1))
    out = {"T": np.packbits((lab.minmax(lab.up(c.reshape(64, 64))) > 0.5).cpu().numpy()),
           "Treg": np.packbits((lab.minmax(lab.up(creg.reshape(64, 64))) > 0.5).cpu().numpy()),
           "c": c.cpu().numpy().astype(np.float32), "creg": creg.cpu().numpy().astype(np.float32)}
    pure = y >= 0.9
    out["ref_stats"] = np.array([float(y.mean()), float(pure.sum()), float(1 - (y * pure).sum() / y.sum().clamp_min(1e-9))], np.float32)
    if "G" not in b["old_arms"]:
        z = rm.mean_field(b)
        out["G"] = np.packbits(rm.field_mask(z).numpy())
        out["mean64"] = z
    np.savez_compressed(f, **out)
    lab.free()
    if kk % 200 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)
print("done", round(time.time() - t0, 1))
