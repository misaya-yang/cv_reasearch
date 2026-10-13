"""Offline: is extent a self-consistent function of area?  alpha = a - b * log A(alpha).

Per episode we tabulate IoU and area fraction for a grid of relative levels alpha of several
64-grid score maps (minmax range). Constants are fit on half A of the episodes (all domains
pooled, one rule) and reported on the disjoint half B.
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
OUT = Path(sys.argv[1])
rows = [b for b in lab.subset(int(sys.argv[2])) if (OUT / f"{b['index']:06d}.npz").exists()]
ALPHA = np.arange(0.04, 0.97, 0.02)
CACHE = OUT / "_tab.npz"


def tabulate():
    T = {k: [] for k in ("I", "U", "A", "oI", "oU", "fI", "fU", "gtA", "refA")}
    for b in rows:
        z = np.load(OUT / f"{b['index']:06d}.npz")
        gt = torch.from_numpy(np.unpackbits(z["gt"])[: 1024 * 1024].reshape(1024, 1024).astype(bool))
        P = np.zeros((64, 4096), np.float32)
        P[z["assign"], np.arange(4096)] = 1
        reg = (P @ z["fwd"]) / np.maximum(P.sum(1), 1)
        maps = {"proto": z["sq"], "fwdreg": reg[z["assign"]] + 1e-3 * z["fwd"], "foris": z["foris_score"].reshape(-1)}
        I, U, A, oI, oU = [], [], [], [], []
        for name, s in maps.items():
            s = torch.from_numpy(s).float()
            sn = (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)
            S = lab.up(sn.reshape(64, 64))
            ii, uu = zip(*[lab.iu(S > a, gt) for a in ALPHA])
            I.append(ii); U.append(uu)
            A.append([(sn > a).float().mean().item() for a in ALPHA])
            bi, bu, _ = lab.best_threshold_iou(S, gt)
            oI.append(bi); oU.append(bu)
        for k, v in zip(("I", "U", "A", "oI", "oU"), (I, U, A, oI, oU)):
            T[k].append(v)
        T["fI"].append(z["foris_iu"][0]); T["fU"].append(z["foris_iu"][1])
        T["gtA"].append(gt.float().mean().item()); T["refA"].append(float(z["y"].mean()))
    T = {k: np.array(v) for k, v in T.items()}
    np.savez(CACHE, **T)
    return T


T = dict(np.load(CACHE)) if CACHE.exists() else tabulate()
dom = np.array([b["dataset"] for b in rows])
cls = np.array([f"{b['row']['fold']}_{b['row']['global_class_id']}" for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    idx = np.where(dom == d)[0]
    half[idx[::2]] = True                                   # A = fit, ~A = report
NAMES = ["proto", "fwdreg", "foris"]


def miou(I, U, sel):
    out = []
    for d in lab.DOMAINS:
        m = sel & (dom == d)
        out.append(np.mean([100 * I[m & (cls == c)].sum() / max(U[m & (cls == c)].sum(), 1) for c in np.unique(cls[m])]))
    return np.array(out)


def pick(k, alpha_idx):
    n = np.arange(len(rows))
    return T["I"][n, k, alpha_idx], T["U"][n, k, alpha_idx]


def self_consistent(k, a, b):
    """Level index solving alpha = a - b*log A(alpha): first crossing from the top."""
    g = ALPHA[None, :] - (a - b * np.log(np.maximum(T["A"][:, k, :], 1e-4)))
    # g increases with alpha on the scale that matters; take the level with the smallest |g|
    return np.abs(g).argmin(1)


def show(tag, v):
    print(f"{tag:40s}" + "".join(f"{x:8.1f}" for x in v) + f"{v.mean():8.1f}")


print(f"class-pooled mIoU on report half (n={int((~half).sum())}); constants fit on the other half; no CRF")
print(f"{'':40s}" + "".join(f"{d[:7]:>8s}" for d in lab.DOMAINS) + "    mean")
show("FoRIS complete output (CRF)", miou(T["fI"], T["fU"], ~half))
for k, name in enumerate(NAMES):
    show(f"{name} | oracle level", miou(T["oI"][:, k], T["oU"][:, k], ~half))
    best = max(range(len(ALPHA)), key=lambda j: miou(*pick(k, np.full(len(rows), j)), half).mean())
    show(f"{name} | const alpha={ALPHA[best]:.2f}", miou(*pick(k, np.full(len(rows), best)), ~half))
    show(f"{name} | const alpha=0.50", miou(*pick(k, np.full(len(rows), 23)), ~half))
    # oracle-area law: alpha = a - b log(true area)
    grid = [(a, b) for a in np.arange(0.1, 0.9, 0.05) for b in np.arange(0.0, 0.25, 0.02)]
    def law_true(a, b):
        al = np.clip(a - b * np.log(np.maximum(T["gtA"], 1e-4)), ALPHA[0], ALPHA[-1])
        return np.abs(ALPHA[None] - al[:, None]).argmin(1)
    a, b = max(grid, key=lambda ab: miou(*pick(k, law_true(*ab)), half).mean())
    show(f"{name} | law, TRUE area a={a:.2f} b={b:.2f}", miou(*pick(k, law_true(a, b)), ~half))
    a, b = max(grid, key=lambda ab: miou(*pick(k, self_consistent(k, *ab)), half).mean())
    show(f"{name} | self-consistent a={a:.2f} b={b:.2f}", miou(*pick(k, self_consistent(k, a, b)), ~half))
    def law_ref(a, b):
        al = np.clip(a - b * np.log(np.maximum(T["refA"], 1e-4)), ALPHA[0], ALPHA[-1])
        return np.abs(ALPHA[None] - al[:, None]).argmin(1)
    a, b = max(grid, key=lambda ab: miou(*pick(k, law_ref(*ab)), half).mean())
    show(f"{name} | law, REF area a={a:.2f} b={b:.2f}", miou(*pick(k, law_ref(a, b)), ~half))
