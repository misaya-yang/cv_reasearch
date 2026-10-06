"""Is a known object size worth anything, and do stored label-free size estimates deliver it? fresh600 token record.

Cut level chosen per quartile of a size estimate (levels fitted on the other folds), token-level class mIoU against the
fixed 0.5 cut of the RCG field. Run on the server: PYTHONPATH=/root/demo4_cache/env python scripts/score_size_threshold.py
"""
import json, numpy as np
from scipy import ndimage
D = "/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/claude_order_fresh600/"
Z = np.load(D + "tokens.npz"); rows = json.load(open(D + "rows.json")); fold = np.array([r["fold"] for r in rows]); N = len(rows)
f = lambda k: Z[k].astype(np.float32)
t, r, afg, abg, knn, ref, s2 = f("truth"), f("rcg"), f("a_fg"), f("a_bg"), f("knn_fg"), f("ref"), f("s2"); bf = Z["back_fg"]
L = lambda x: np.log(np.maximum(x, .5) / 4096)
y = L(t.sum(1))
def peak_cc(i, lv):
    m = (r[i] > lv).reshape(64, 64); lab, _ = ndimage.label(m); p = lab.reshape(-1)[r[i].argmax()]
    return (lab == p).sum() if p else 0
P = {"area rcg>0.5": L((r > .5).sum(1)), "area rcg>0.8": L((r > .8).sum(1)), "peak component rcg>0.5": L(np.array([peak_cc(i, .5) for i in range(N)])),
     "peak component rcg>0.7": L(np.array([peak_cc(i, .7) for i in range(N)])), "10-NN vote majority count": L((knn > .5).sum(1)),
     "reverse hits from ref object": L(bf.sum(1)), "tokens closer to ref object than ref background": L((afg > abg).sum(1)),
     "reference object size": L(ref.sum(1)), "area s2>0.5": L((s2 > .5).sum(1))}
def held(X):
    out = np.zeros(N)
    for k in np.unique(fold):
        a, b = fold != k, fold == k; A = np.c_[X[a], np.ones(a.sum())]; w = np.linalg.lstsq(A, y[a], rcond=None)[0]; out[b] = np.c_[X[b], np.ones(b.sum())] @ w
    return out

cls = np.array([r["c"] for r in rows]); C = np.unique(cls); lev = np.linspace(.2, .9, 29); k0 = int(np.argmin(abs(lev - .5))); tb = t > .5
I = np.stack([((r > l) & tb).sum(1) for l in lev], 1).astype(float); U = np.stack([((r > l) | tb).sum(1) for l in lev], 1).astype(float); a = np.arange(N)
miou = lambda i, u: np.mean([i[cls == c].sum() / max(u[cls == c].sum(), 1e-9) for c in C]) * 100
base = miou(I[:, k0], U[:, k0]); J = base / 100; V = (I - J * U) - (I[:, [k0]] - J * U[:, [k0]])
def gain(est, nb=4):
    pick = np.full(N, k0)
    for k in np.unique(fold):
        fit, hd = fold != k, fold == k; edges = np.quantile(est[fit], np.linspace(0, 1, nb + 1)[1:-1]); b = np.digitize(est, edges)
        for q in range(nb):
            if (fit & (b == q)).sum() > 15: pick[hd & (b == q)] = V[fit & (b == q)].sum(0).argmax()
    return miou(I[a, pick], U[a, pick]) - base, [round(float(lev[pick[np.digitize(est, np.quantile(est, [.25, .5, .75])) == q]].mean()), 2) for q in range(4)]
print(f"token-level mIoU at 0.5: {base:.2f}, n={N}")
g, lv = gain(y); print(f"{'TRUE size (diagnostic)':50s} {g:+.2f}  mean level by size quartile {lv}")
P["all nine, held-fold linear fit"] = held(np.stack(list(P.values()), 1))
for k, v in P.items():
    g, lv = gain(v); print(f"{k:50s} {g:+.2f}  mean level by quartile {lv}")
