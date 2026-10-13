"""Offline: bidirectional attention as evidence AND extent.

f_j = forward mass on reference FG (query -> reference), g_j = backward mass received from
reference FG (reference -> query).  J(S) = a c / (a + c - a c) with a = mean_S f, c = sum_S g.
For each ranking we report: oracle level, best fitted constant, and the parameter-free J-optimal
prefix.  Rankings: prototype contrast, f, g, f*g, FoRIS score.
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
F300 = lab.ASSETS / "a/x1012/fields300"
B300 = lab.ASSETS / "a/x1012/bidir300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
dom = np.array([b["dataset"] for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    idx = np.where(dom == d)[0]
    half[idx[::2]] = True
TAU = sys.argv[1] if len(sys.argv) > 1 else "0.05"
CONST = [0.3, 0.4, 0.5, 0.6, 0.7]


def jprefix(rank_score, f, g):
    """Level of rank_score whose prefix maximises J; returns the threshold value."""
    order = np.argsort(-rank_score)
    a = np.cumsum(f[order]) / np.arange(1, len(order) + 1)
    c = np.cumsum(g[order])
    J = a * c / np.maximum(a + c - a * c, 1e-9)
    k = int(np.argmax(J[4:])) + 4
    return rank_score[order[k]], J[k]


M = {}
for e, b in enumerate(rows):
    z = np.load(F300 / f"{b['index']:06d}.npz")
    bd = np.load(B300 / f"{b['index']:06d}.npz")
    gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
    f, g = bd[f"fwdF{TAU}"].astype(np.float64), bd[f"backF{TAU}"].astype(np.float64)
    ranks = {"proto": z["sq"].astype(np.float64), "f": f, "g": g, "f*g": f * g,
             "foris": z["foris_score"].reshape(-1).astype(np.float64)}
    tag = "A" if half[e] else "B"
    for name, s in ranks.items():
        S = lab.up(torch.from_numpy(s).float().reshape(64, 64))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M.setdefault((name, "oracle level", tag), lab.Meter()).add(b, i, u)
        lo, hi = float(s.min()), float(s.max())
        for cst in CONST:
            M.setdefault((name, f"const {cst}", tag), lab.Meter()).add(b, *lab.iu(S > lo + cst * (hi - lo), gt))
        t, _ = jprefix(s, f, g)
        M.setdefault((name, "J-optimal prefix", tag), lab.Meter()).add(b, *lab.iu(S >= t, gt))
    M.setdefault(("FoRIS", "complete output (CRF)", tag), lab.Meter()).add(b, int(z["foris_iu"][0]), int(z["foris_iu"][1]))

print(f"class-pooled mIoU on report half; constants chosen on the other half; tau={TAU}; no CRF")
print(f"{'ranking | extent':34s}" + "".join(f"{d[:7]:>8s}" for d in lab.DOMAINS) + "    mean")
names = []
for (name, rule, tag) in M:
    if (name, rule) not in names:
        names.append((name, rule))
for name in dict.fromkeys(n for n, _ in names):
    rules = [r for n, r in names if n == name]
    consts = [r for r in rules if r.startswith("const")]
    if consts:
        bestc = max(consts, key=lambda r: np.mean([M[(name, r, "A")].row()[d][0] for d in lab.DOMAINS]))
        rules = [r for r in rules if not r.startswith("const")] + [bestc]
    for r in rules:
        v = [M[(name, r, "B")].row()[d][0] for d in lab.DOMAINS]
        print(f"{name + ' | ' + r:34s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}")
