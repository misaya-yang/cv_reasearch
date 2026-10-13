"""One uniform table on all ten benchmark dev sets: cached FoRIS against three fixed simple readouts.

No per-domain choice anywhere.  Features: cached O24, positional projection, joint centring.
  proto     cos(q - mu, mean_FG(r) - mu), bilinear, min-max, > 0.7
  transfer  c_j = sum_i softmax(<q_j, r_i> / 0.05) y_i  (y = soft reference coverage), bilinear, min-max, > 0.5
  region    transfer averaged over 64 spherical k-means regions of the query, bilinear, min-max, > 0.5
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = lab.subset(N) + lab.ext_subset(None)
DOMS = tuple(dict.fromkeys(b["dataset"] for b in rows))
M, cnt = {}, lab.defaultdict(int)
t0 = time.time()
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b)
    ig = None if ig is None else ig.to(DEV)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    proto = qc @ lab.unit((r * w[:, None]).sum(0) / w.sum() - mu)
    c = torch.softmax((qc @ rc.T) / 0.05, dim=1) @ y
    C = lab.kmeans(lab.unit(q), 64, iters=6)
    a = (lab.unit(q) @ C.T).argmax(1)
    oh = torch.nn.functional.one_hot(a, 64).float()
    creg = (oh @ ((oh.T @ c) / oh.sum(0).clamp_min(1)))

    def add(name, pred):
        g = gt
        if ig is not None:
            pred, g = pred & ~ig, gt & ~ig
        M.setdefault(name, lab.Meter()).add(b, *lab.iu(pred, g))

    add("FoRIS complete (cached, CRF)", lab.foris_cli(b).to(DEV))
    add("proto, minmax > 0.7", lab.minmax(lab.up(proto.reshape(64, 64))) > 0.7)
    add("transfer, minmax > 0.5", lab.minmax(lab.up(c.reshape(64, 64))) > 0.5)
    add("region transfer, minmax > 0.5", lab.minmax(lab.up(creg.reshape(64, 64))) > 0.5)
    for name, S in (("transfer", c), ("region transfer", creg)):
        S = lab.up(S.reshape(64, 64))
        g = gt
        if ig is not None:
            S, g = torch.where(ig, S.min(), S), gt & ~ig
        i, u, _ = lab.best_threshold_iou(S, g)
        M.setdefault(name + ", GT-optimal level", lab.Meter()).add(b, i, u)
    cnt[b["dataset"]] += 1
    lab.free()
    if kk % 300 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)

print("\nclass-pooled mIoU (per-episode mean IoU), CLI-1024; episodes: " + ", ".join(f"{d}={cnt[d]}" for d in DOMS))
print(f"{'method':34s}" + "".join(f"{d[:9]:>15s}" for d in DOMS))
for name, m in M.items():
    r = m.row(DOMS)
    print(f"{name:34s}" + "".join(f"{r[d][0]:7.1f} ({r[d][1]:5.1f})" for d in DOMS))
