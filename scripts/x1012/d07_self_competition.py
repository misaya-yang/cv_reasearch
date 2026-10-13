"""Complete-output test: centred prototype contrast (identify) + query-own competition (extent).

P0  s0 = cos(q - mu, muF - mu) for several centrings mu; output at minmax 0.5 and at oracle.
P1  seeds from s0 (minmax 0.5) -> query-own FG / BG multi-prototypes -> s1 = max_F - max_B,
    iterated; output at the natural zero and at oracle.  No reference negatives after init.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N)


from lab import kmeans, minmax


def self_compete(qc, s0, KF, KB, iters):
    """Query-own two-sided prototype competition initialised by s0 (seed: minmax > .5)."""
    fg = minmax(s0) > 0.5
    s = s0
    for _ in range(iters):
        if fg.sum() < 2 or (~fg).sum() < 2:
            break
        PF, PB = kmeans(qc, KF, w=fg.float()), kmeans(qc, KB, w=(~fg).float())
        s = (qc @ PF.T).max(1).values - (qc @ PB.T).max(1).values
        fg = s > 0
    return s


M, Zr = {}, {}
t0 = time.time()
for k, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    cr = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    w = cr if cr.sum() > 1e-6 else (cr >= cr.max()).float()
    muF = (r * w[:, None]).sum(0) / w.sum()
    muR, muQ = r.mean(0), q.mean(0)
    cents = {"none": None, "R": muR, "Q": muQ, "J": 0.5 * (muR + muQ)}
    out, nat = {}, {}
    for name, mu in cents.items():
        s = (q @ lab.unit(muF)) if mu is None else lab.unit(q - mu) @ lab.unit(muF - mu)
        out[f"P0_{name}"] = s
        nat[f"P0_{name}"] = minmax(s) - 0.5
    out["P0_own"] = lab.unit(q - muQ) @ lab.unit(muF - muR)
    nat["P0_own"] = minmax(out["P0_own"]) - 0.5
    qc = lab.unit(q - cents["J"])
    for KF, KB, it in ((1, 1, 1), (2, 8, 1), (4, 16, 1), (4, 16, 3), (8, 32, 3)):
        s = self_compete(qc, out["P0_J"], KF, KB, it)
        out[f"P1_F{KF}B{KB}i{it}"] = s
        nat[f"P1_F{KF}B{KB}i{it}"] = s
    fs = torch.from_numpy(np.load(b["score64"]["path"])["score"]).to(DEV).flatten()
    out["FoRIS_score"] = fs
    nat["FoRIS_score"] = minmax(fs) - 0.5
    for name, s in out.items():
        i, u, _ = lab.best_threshold_iou(up(s.reshape(64, 64)), gt)
        M.setdefault(name, Meter()).add(b, i, u)
        Zr.setdefault(name, Meter()).add(b, *iu(up(nat[name].reshape(64, 64)) > 0, gt))
    Zr.setdefault("FoRIS_crf_output", Meter()).add(b, *iu(lab.foris_cli(b).to(DEV), gt))
    lab.free()
    if k % 150 == 0:
        print(k, round(time.time() - t0, 1), flush=True)

print(f"\nclass-pooled mIoU (per-episode mean IoU): natural threshold || oracle threshold, n={N}/domain, no CRF")
print(f"{'method':18s}" + "".join(f"{d:>26s}" for d in DOMAINS))
for name in list(M) + ["FoRIS_crf_output"]:
    z = Zr[name].row()
    r = M[name].row() if name in M else None
    print(f"{name:18s}" + "".join(f"  {z[d][0]:5.1f} ({z[d][1]:5.1f}) ||" + (f" {r[d][0]:5.1f}   " if r else "    -    ") for d in DOMAINS))
