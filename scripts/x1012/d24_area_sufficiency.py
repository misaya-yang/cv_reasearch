"""Is the query target AREA a sufficient statistic for extent?

For each ranking (centred prototype, FoRIS score) cut the top-k pixels with
  k = true query area (ceiling of any area predictor), k = reference area (same-scale prior),
and compare with the level oracle and the fixed minmax level.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
F300 = lab.ASSETS / "a/x1012/fields300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()][::2]
M, C = {}, lab.defaultdict(list)


def topk(S, k):
    k = int(min(max(k, 1), S.numel()))
    thr = torch.kthvalue(S.flatten(), S.numel() - k + 1).values
    return S >= thr


for b in rows:
    z = np.load(F300 / f"{b['index']:06d}.npz")
    gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
    aq, ar = int(gt.sum()), float(z["y"].mean()) * 1024 * 1024
    C[b["dataset"]].append((np.log(max(aq, 1) / 1024 ** 2), np.log(max(ar, 1) / 1024 ** 2)))
    for name, s in (("proto", z["sq"]), ("foris", z["foris_score"].reshape(-1))):
        S = lab.up(torch.from_numpy(s.astype(np.float32)).reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault(f"{name} | level oracle", lab.Meter()).add(b, i, u)
        M.setdefault(f"{name} | top-k, k = TRUE area", lab.Meter()).add(b, *lab.iu(topk(S, aq), gt))
        for g in (0.5, 0.7, 1.4, 2.0):
            M.setdefault(f"{name} | top-k, k = {g} x TRUE area", lab.Meter()).add(b, *lab.iu(topk(S, g * aq), gt))
        M.setdefault(f"{name} | top-k, k = REF area", lab.Meter()).add(b, *lab.iu(topk(S, ar), gt))
        M.setdefault(f"{name} | minmax 0.5", lab.Meter()).add(b, *lab.iu(lab.minmax(S) > 0.5, gt))
    M.setdefault("FoRIS complete (CRF)", lab.Meter()).add(b, int(z["foris_iu"][0]), int(z["foris_iu"][1]))

print(f"class-pooled mIoU, n={len(rows)}")
print(f"{'ranking | rule':38s}" + "".join(f"{d[:9]:>10s}" for d in lab.DOMAINS) + "    mean")
for name in M:
    r = M[name].row(lab.DOMAINS)
    print(f"{name:38s}" + "".join(f"{r[d][0]:10.1f}" for d in lab.DOMAINS) + f"{np.mean([r[d][0] for d in lab.DOMAINS]):8.1f}")
print("\nlog query area vs log reference area: Pearson r, and median |log2 ratio|")
for d in lab.DOMAINS:
    a = np.array(C[d])
    print(f"{d:12s} r={np.corrcoef(a[:,0], a[:,1])[0,1]:+.2f}   median |log2(Aq/Ar)|={np.median(np.abs(a[:,0]-a[:,1]))/np.log(2):.2f}")
