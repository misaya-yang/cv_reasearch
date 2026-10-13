"""Extended sets: how far does a fixed level get, and in which direction does the natural level err?

Scores: region-averaged coverage transfer (see e01) and FoRIS's cached pre-binarisation score.
Rules: min-max level grid, absolute level grid (transfer only), and the area bias of minmax 0.5.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

DOMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else lab.EXT_DOMAINS
rows = lab.ext_subset(None, DOMS)
G = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
M, bias = {}, lab.defaultdict(list)
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b)
    ig = None if ig is None else ig.to(DEV)
    g = gt if ig is None else gt & ~ig
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    c = torch.softmax((lab.unit(q - mu) @ lab.unit(r - mu).T) / 0.05, dim=1) @ y
    C = lab.kmeans(lab.unit(q), 64, iters=6)
    oh = torch.nn.functional.one_hot((lab.unit(q) @ C.T).argmax(1), 64).float()
    S = {"region": lab.up((oh @ ((oh.T @ c) / oh.sum(0).clamp_min(1))).reshape(64, 64)), "token": lab.up(c.reshape(64, 64))}
    fs = b.get("score64")
    for name, s in S.items():
        if ig is not None:
            s = torch.where(ig, s.min(), s)
        sn = lab.minmax(s)
        for a in G:
            M.setdefault(f"{name} | minmax > {a}", lab.Meter()).add(b, *lab.iu(sn > a, g))
            M.setdefault(f"{name} | absolute > {a}", lab.Meter()).add(b, *lab.iu(s > a, g))
        i, u, _ = lab.best_threshold_iou(s, g)
        M.setdefault(f"{name} | GT-optimal level", lab.Meter()).add(b, i, u)
        bias[(b["dataset"], name)].append(np.log2(max(float((sn > 0.5).sum()), 1) / max(float(g.sum()), 1)))
    p = lab.foris_cli(b).to(DEV)
    if ig is not None:
        p = p & ~ig
    M.setdefault("FoRIS complete (CRF)", lab.Meter()).add(b, *lab.iu(p, g))
    bias[(b["dataset"], "foris")].append(np.log2(max(float(p.sum()), 1) / max(float(g.sum()), 1)))
    bias[(b["dataset"], "refarea")].append(np.log2(max(float(y.mean()) * 1024 ** 2, 1) / max(float(g.sum()), 1)))
    lab.free()

print("class-pooled mIoU; columns " + " ".join(DOMS))
for name in sorted(M, key=lambda n: (n.split(" | ")[0], n)):
    r = M[name].row(DOMS)
    print(f"{name:30s}" + "".join(f"{r[d][0]:8.1f}" for d in DOMS))
print("\nlog2(predicted area / true area): median, and median absolute")
for d in DOMS:
    print(f"{d:14s}" + "   ".join(f"{k}: {np.median(bias[(d, k)]):+.2f} / {np.median(np.abs(bias[(d, k)])):.2f}" for k in ("foris", "region", "token", "refarea")))
