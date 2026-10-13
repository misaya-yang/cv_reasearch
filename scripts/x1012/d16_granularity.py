"""Does the reference mask tell the right query granularity?

Query tokens are grouped by average-link cosine agglomeration cut at merge similarity tau.
Region-mean evidence (R-centred prototype contrast) is cut at half of its own range.
Rules for tau: fixed, reference-selected (the cut whose reference regions align best with M_R),
and the per-episode oracle (diagnostic).  CPU only.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(6)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = lab.subset(N, offset=1)
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def tree(x):
    """Average-link tree on a 64-d PCA of unit tokens (cosine distance)."""
    xc = x - x.mean(0)
    _, _, V = torch.pca_lowrank(xc, q=64, center=False, niter=2)
    z = torch.nn.functional.normalize(x @ V, dim=1).numpy().astype(np.float64)
    return linkage(z, method="average", metric="cosine")


def region_mean(labels, v):
    k = labels.max() + 1
    s = np.bincount(labels, weights=v, minlength=k)
    c = np.bincount(labels, minlength=k)
    return (s / np.maximum(c, 1))[labels], k


M, chosen = {}, lab.defaultdict(list)
t0 = time.time()
for kk, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"])
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"])).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = y if y.sum() > 1e-6 else (y >= y.max()).float()
    mu = r.mean(0)
    s = (lab.unit(q - mu) @ lab.unit((r * w[:, None]).sum(0) / w.sum() - mu)).numpy().astype(np.float64)
    ZQ, ZR = tree(q), tree(r)
    yn = y.numpy().astype(np.float64)
    res, align = {}, {}
    for tau in TAUS:
        lq = fcluster(ZQ, t=1 - tau, criterion="distance") - 1
        rm, kq = region_mean(lq, s)
        f = torch.from_numpy(rm + 1e-3 * s).float()
        S = lab.up(f.reshape(64, 64))
        res[tau] = lab.iu(lab.minmax(S) > 0.5, gt)
        lr = fcluster(ZR, t=1 - tau, criterion="distance") - 1
        frac, kr = region_mean(lr, yn)
        pred = (frac > 0.5).astype(np.float64)
        inter = (pred * yn).sum()
        align[tau] = inter / max(pred.sum() + yn.sum() - inter, 1e-9)
        M.setdefault(f"tau={tau}", lab.Meter()).add(b, *res[tau])
    best_align = max(align.values())
    t_ref = min(t for t in TAUS if align[t] >= best_align - 0.02)      # coarsest cut that keeps the mask
    M.setdefault("tau=ref-selected", lab.Meter()).add(b, *res[t_ref])
    t_or = max(TAUS, key=lambda t: res[t][0] / max(res[t][1], 1))
    M.setdefault("tau=oracle", lab.Meter()).add(b, *res[t_or])
    chosen[d].append((t_ref, t_or, best_align))
    i, u, _ = lab.best_threshold_iou(lab.up(torch.from_numpy(s).float().reshape(64, 64)), gt)
    M.setdefault("token score|oracle level", lab.Meter()).add(b, i, u)
    M.setdefault("FoRIS_crf_output", lab.Meter()).add(b, *lab.iu(lab.foris_cli(b), gt))
    if kk % 30 == 0:
        print(kk, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU), region-mean evidence at half range, n={N}/domain, no CRF")
print(f"{'granularity':26s}" + "".join(f"{d:>16s}" for d in lab.DOMAINS) + "    mean")
for name in M:
    r = M[name].row()
    print(f"{name:26s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in lab.DOMAINS) + f"{np.mean([r[d][0] for d in lab.DOMAINS]):8.1f}")
print("\nmean reference-selected tau | mean oracle tau | agreement | mean best reference alignment")
for d in lab.DOMAINS:
    a = np.array(chosen[d])
    print(f"{d:12s} {a[:,0].mean():.2f} | {a[:,1].mean():.2f} | {np.mean(a[:,0]==a[:,1]):.2f} | {a[:,2].mean():.3f}")
