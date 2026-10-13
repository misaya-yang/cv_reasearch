"""Offline: does query-internal boundary contrast along level sets locate the extent?

B(alpha) = mean token dissimilarity over the grid edges crossed by the level-alpha contour of a
score map.  Peaks of B are levels whose contour sits on a real feature discontinuity.
Diagnostics: IoU at argmax B; oracle among the top-3 peaks; peak nearest to a fitted prior level.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lab

F300 = lab.ASSETS / "a/x1012/fields300"
E300 = lab.ASSETS / "a/x1012/edges300"
rows = [b for b in lab.subset(300) if (F300 / f"{b['index']:06d}.npz").exists()]
T = dict(np.load(F300 / "_tab.npz"))
ALPHA = np.arange(0.04, 0.97, 0.02)
dom = np.array([b["dataset"] for b in rows])
cls = np.array([f"{b['row']['fold']}_{b['row']['global_class_id']}" for b in rows])
half = np.zeros(len(rows), bool)
for d in lab.DOMAINS:
    idx = np.where(dom == d)[0]
    half[idx[::2]] = True


def profile(sn, dh, dv):
    """Mean edge dissimilarity and edge count of the level-alpha contour, for all ALPHA."""
    g = sn.reshape(64, 64)
    lo = np.concatenate([np.minimum(g[:, 1:], g[:, :-1]).ravel(), np.minimum(g[1:], g[:-1]).ravel()])
    hi = np.concatenate([np.maximum(g[:, 1:], g[:, :-1]).ravel(), np.maximum(g[1:], g[:-1]).ravel()])
    d = np.concatenate([dh.ravel(), dv.ravel()])
    act = (lo[None, :] <= ALPHA[:, None]) & (ALPHA[:, None] < hi[None, :])
    cnt = act.sum(1)
    return (act * d[None, :]).sum(1) / np.maximum(cnt, 1), cnt


def peaks(B, k=3, lo=6, hi=44):
    """Indices of the k strongest local maxima of the smoothed profile within [lo, hi)."""
    s = np.convolve(B, np.ones(3) / 3, mode="same")
    idx = [i for i in range(lo, hi) if s[i] >= s[i - 1] and s[i] >= s[i + 1]]
    idx = sorted(idx, key=lambda i: -s[i])[:k]
    return idx or [int(np.argmax(s[lo:hi])) + lo]


def miou(I, U, sel):
    out = []
    for d in lab.DOMAINS:
        m = sel & (dom == d)
        out.append(np.mean([100 * I[m & (cls == c)].sum() / max(U[m & (cls == c)].sum(), 1) for c in np.unique(cls[m])]))
    return np.array(out)


def show(tag, v):
    print(f"{tag:44s}" + "".join(f"{x:8.1f}" for x in v) + f"{v.mean():8.1f}")


n = np.arange(len(rows))
print(f"class-pooled mIoU on report half (n={int((~half).sum())}); no CRF")
print(f"{'':44s}" + "".join(f"{d[:7]:>8s}" for d in lab.DOMAINS) + "    mean")
show("FoRIS complete output (CRF)", miou(T["fI"], T["fU"], ~half))
for k, name in ((0, "proto"), (2, "foris")):
    PK, AM = [], []
    for b in rows:
        z = np.load(F300 / f"{b['index']:06d}.npz")
        e = np.load(E300 / f"{b['index']:06d}.npz")
        s = z["sq"] if name == "proto" else z["foris_score"].reshape(-1)
        sn = (s - s.min()) / max(s.max() - s.min(), 1e-6)
        B, cnt = profile(sn, e["qh"], e["qv"])
        PK.append(peaks(B))
        AM.append(int(np.argmax(np.convolve(B, np.ones(3) / 3, mode="same")[6:44])) + 6)
    AM = np.array(AM)
    iou = T["I"][:, k, :] / np.maximum(T["U"][:, k, :], 1)
    show(f"{name} | oracle level", miou(T["oI"][:, k], T["oU"][:, k], ~half))
    best = max(range(len(ALPHA)), key=lambda j: miou(T["I"][n, k, j], T["U"][n, k, j], half).mean())
    show(f"{name} | const alpha={ALPHA[best]:.2f}", miou(T["I"][n, k, best], T["U"][n, k, best], ~half))
    show(f"{name} | argmax boundary contrast", miou(T["I"][n, k, AM], T["U"][n, k, AM], ~half))
    for kk in (1, 2, 3):
        pick = np.array([max(p[:kk], key=lambda i: iou[j, i]) for j, p in enumerate(PK)])
        show(f"{name} | ORACLE among top-{kk} contrast peaks", miou(T["I"][n, k, pick], T["U"][n, k, pick], ~half))
    pick = np.array([min(p, key=lambda i: abs(i - best)) for p in PK])
    show(f"{name} | contrast peak nearest const level", miou(T["I"][n, k, pick], T["U"][n, k, pick], ~half))
    # snap: move from the constant level to the nearest peak only if within w levels
    for w in (3, 6):
        pick = np.array([min(p, key=lambda i: abs(i - best)) if min(abs(i - best) for i in p) <= w else best for p in PK])
        show(f"{name} | snap const level to peak within {w*0.02:.2f}", miou(T["I"][n, k, pick], T["U"][n, k, pick], ~half))
