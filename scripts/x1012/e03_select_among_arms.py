"""Can a reference-consistency score pick the better complete mask among existing arms, per episode?

Candidates: cached complete masks of FoRIS (B), whole ridge (R), MEAN (G) [and the consensus arms C, N for reference].
Selector  : bidirectional-transfer surrogate J(S) = a c / (a + c - a c) on the 64-grid coverage of each mask,
            a = forward mass on reference FG averaged over S, c = backward mass from reference FG captured by S.
No query label is used by the selector.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
A_ = lab.ASSETS / "a/x1012"
rows = [b for b in lab.subset(300) if (A_ / "bidir300" / f"{b['index']:06d}.npz").exists()]
ARMS = ("B", "R", "G", "C", "N")
M = lab.defaultdict(lab.Meter)
pick = lab.defaultdict(lambda: lab.defaultdict(int))
for b in rows:
    bd = np.load(A_ / "bidir300" / f"{b['index']:06d}.npz")
    gt = lab.mask1024(b["row"]["query_mask_path"])
    masks = {a: lab.arm_cli(b, a) for a in ARMS}
    res = {a: lab.iu(m, gt) for a, m in masks.items()}
    for a in ARMS:
        M[a].add(b, *res[a])
    cov = {a: lab.coverage64(m).flatten().numpy().astype(np.float64) for a, m in masks.items()}
    for t in ("0.02", "0.05"):
        f, bF = bd["fwdF" + t].astype(np.float64), bd["backF" + t].astype(np.float64)
        bF = bF / max(bF.sum(), 1e-30)
        J = {}
        for a in ARMS:
            s = cov[a]
            pa, pc = (s * f).sum() / max(s.sum(), 1e-9), (s * bF).sum()
            J[a] = pa * pc / max(pa + pc - pa * pc, 1e-12)
        for cand in (("B", "R", "G"), ("B", "G"), ("B", "R"), ("B", "N"), ("B", "R", "G", "C", "N")):
            best = max(cand, key=lambda a: J[a])
            M[f"J-select t{t} among {'/'.join(cand)}"].add(b, *res[best])
            if t == "0.05" and cand == ("B", "R", "G"):
                pick[b["dataset"]][best] += 1
    j = lambda x: x[0] / max(x[1], 1)
    M["GT pick among B/R/G"].add(b, *max((res[a] for a in ("B", "R", "G")), key=j))

print(f"class-pooled mIoU, n={len(rows)}; delta vs FoRIS in brackets")
print(f"{'arm / selector':40s}" + "".join(f"{d[:11]:>15s}" for d in lab.DOMAINS))
base = M["B"].row(lab.DOMAINS)
names = {"B": "B  FoRIS complete", "R": "R  whole ridge", "G": "G  MEAN", "C": "C  3-way vote B,R,G", "N": "N  supported_mean"}
for a, m in M.items():
    r = m.row(lab.DOMAINS)
    print(f"{names.get(a, a):40s}" + "".join(f"{r[d][0]:8.2f} [{r[d][0] - base[d][0]:+5.2f}]" for d in lab.DOMAINS))
print("picks (t0.05, among B/R/G): " + "; ".join(f"{d}: " + ",".join(f"{a}={n}" for a, n in sorted(pick[d].items())) for d in lab.DOMAINS))
