"""Offline report for the layer scan: per-block ranking quality by domain."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

OUT = Path(sys.argv[1])
LAYERS = [4, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24]
rows = [b for b in lab.subset(int(sys.argv[2]), offset=1) if (OUT / f"{b['index']:06d}.npz").exists()]
print("episodes", len(rows))


def auroc(s, g):
    s, g = s.flatten()[::7], g.flatten()[::7]
    if g.all() or not g.any():
        return float("nan")
    r = torch.argsort(torch.argsort(s)).float()
    n1, n0 = g.sum().float(), (~g).sum().float()
    return float((r[g].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def z(x):
    return (x - x.mean()) / x.std().clamp_min(1e-6)


combos = {"z24+20": (24, 20), "z24+18": (24, 18), "z24+16": (24, 16), "z24+12": (24, 12),
          "z24+22+20+18": (24, 22, 20, 18), "z24+20+16+12": (24, 20, 16, 12), "z20+16+12": (20, 16, 12)}
M, A = {}, {}
for b in rows:
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    zf = np.load(OUT / f"{b['index']:06d}.npz")
    maps = {k: torch.from_numpy(zf[k]).to(DEV) for k in zf.files}
    for kind in ("cR", "nn"):
        for name, ls in combos.items():
            maps[f"{kind}/{name}"] = sum(z(maps[f"{kind}/{l}"]) for l in ls)
    for name, s in maps.items():
        if name.startswith("nnF"):
            continue
        s = up(s)
        i, u, _ = lab.best_threshold_iou(s, gt, n=40)
        M.setdefault(name, Meter()).add(b, i, u)
        A.setdefault(name, lab.defaultdict(list))[d].append(auroc(s, gt))
        if name.startswith("nn/") and name[3:].isdigit():
            M.setdefault("zero:" + name, Meter()).add(b, *iu(s > 0, gt))

print(f"oracle-threshold mIoU | AUROC   ({len(rows)} episodes)")
print(f"{'score':20s}" + "".join(f"{d:>16s}" for d in DOMAINS))
for name in sorted(M, key=lambda n: (n.split('/')[0], n)):
    r = M[name].row()
    a = A.get(name)
    print(f"{name:20s}" + "".join(f"{r[d][0]:8.2f} " + (f"{np.nanmean(a[d]):.4f}" if a else "   -   ") for d in DOMAINS))
