"""Bound of the cut-level route: how much of the per-episode best level can a rule read from the field itself?

A flexible regressor is fitted on the other folds (episodes sharing a photograph with the held fold removed) to the
value of every level of an episode, from label-free descriptors of the field's level sets; the held fold takes the
level with the highest predicted value. The fit uses base-fold labels, so this is an upper reference for label-free
cut rules built on the same descriptors, not a method. Scored from stored per-level counts: no model, no GPU.

    python score_cut_predictability.py outputs/claude_cut_levels4000 rcg
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor

D, NAME = Path(sys.argv[1]), sys.argv[2]
Z = np.load(D / "counts.npz"); rows = json.loads((D / "rows.json").read_text())
lev = Z["levels"]; F = Z["rcg_field" if NAME == "rcg" else "foris_score"].astype(np.float32); I, U = Z[f"I_{NAME}"].astype(float), Z[f"U_{NAME}"].astype(float)
N = len(rows); cls = np.array([r["c"] for r in rows]); fold = np.array([r["fold"] for r in rows]); area = Z["truth"].astype(float)
k0 = int(np.argmin(np.abs(lev - .5))); KS = np.where((lev >= .2) & (lev <= .9))[0]

parent = {}                                         # photograph groups: episodes linked by any shared photograph
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, r in enumerate(rows):
    for p in (r["support"], r["query"]):
        parent[find(("p", p))] = find(("e", i))
grp = np.unique([str(find(("e", i))) for i in range(N)], return_inverse=True)[1]; G = grp.max() + 1
C = np.unique(cls); oh = (cls[:, None] == C[None]).astype(float)


def miou(I, U, ix=None):
    ix = np.ones(N, bool) if ix is None else ix
    return float(np.mean([I[ix & (cls == c)].sum() / max(U[ix & (cls == c)].sum(), 1e-9) for c in np.unique(cls[ix])]) * 100)


def boot(a, b):
    rs = np.random.RandomState(0); out = []
    gs = lambda x: (lambda o: (np.add.at(o, grp, oh[:, :, None] * np.stack(x, 1)[:, None, :]), o)[1])(np.zeros((G, len(C), 2)))
    X, Y = gs(a), gs(b)
    for _ in range(2000):
        s = rs.randint(0, G, G); x, y = X[s].sum(0), Y[s].sum(0); ok = y[:, 1] > 0; out.append((np.mean(x[ok, 0] / np.maximum(x[ok, 1], 1e-9)) - np.mean(y[ok, 0] / y[ok, 1])) * 100)
    return miou(*a) - miou(*b), np.percentile(out, [2.5, 97.5])


def at(pick):
    return I[np.arange(N), pick], U[np.arange(N), pick]


S8 = np.ones((3, 3), bool); GL = [int(np.argmin(np.abs(lev - t))) for t in (.2, .3, .4, .5, .6, .7, .8, .9)]
HIST = ["level", "area", "log_area", "area_over_area_at_half", "stability"] + [f"area@{lev[k]:.1f}" for k in GL] + ["field_mean", "field_std", "field_q50", "field_q90", "field_q99"]
SPAT = ["components", "largest_share", "peak_share", "perimeter_per_area", "edge_gradient", "mean_inside", "mean_outside", "components_at_half", "peak_share_at_half"]
X = np.zeros((N, len(KS), len(HIST) + len(SPAT)), np.float32)
for i in range(N):
    f = F[i]; gy, gx = np.gradient(f); g = np.hypot(gx, gy); peak = np.unravel_index(f.argmax(), f.shape); a = np.array([(f > lev[k]).mean() for k in range(len(lev))])
    glob = [a[k] for k in GL] + [f.mean(), f.std(), *np.quantile(f, [.5, .9, .99])]; sp = {}
    for k in list(KS) + [k0]:
        m = f > lev[k]; n = m.sum()
        if n == 0:
            sp[k] = [0] * 7; continue
        lab, nc = ndimage.label(m, S8); sz = np.bincount(lab.ravel())[1:]; edge = m & ~ndimage.binary_erosion(m)
        per = (m[:, 1:] != m[:, :-1]).sum() + (m[1:] != m[:-1]).sum()
        sp[k] = [nc, sz.max() / n, sz[lab[peak] - 1] / n if lab[peak] else 0, per / n, g[edge].mean(), f[m].mean(), f[~m].mean() if n < f.size else 0]
    for j, k in enumerate(KS):
        stab = (a[max(k - 1, 0)] - a[min(k + 1, len(lev) - 1)]) / max(a[k], 1 / f.size)
        X[i, j] = [lev[k], a[k], np.log(max(a[k], 1 / f.size)), a[k] / max(a[k0], 1 / f.size), stab] + glob + sp[k] + [sp[k0][0], sp[k0][2]]

base = at(np.full(N, k0)); J = miou(*base) / 100
Y = ((I[:, KS] - J * U[:, KS]) - (I[:, [k0]] - J * U[:, [k0]])) / 1024 ** 2      # value of a level against the half level, at the class ratio J


def nested(cols):
    pick = np.full(N, k0)
    for f in np.unique(fold):
        held = fold == f; fit = ~held & ~np.isin(grp, grp[held])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=.06, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1., random_state=0)
        m.fit(X[fit][:, :, cols].reshape(-1, len(cols)), Y[fit].reshape(-1))
        pick[held] = KS[m.predict(X[held][:, :, cols].reshape(-1, len(cols))).reshape(held.sum(), len(KS)).argmax(1)]
    return pick


def binned():                                       # control: one level per bin of the mask area at the half level, fitted on the other folds
    pick = np.full(N, k0); a0 = X[:, 0, HIST.index("area@0.5")]; edges = np.quantile(a0, np.linspace(0, 1, 7)[1:-1]); b = np.digitize(a0, edges)
    for f in np.unique(fold):
        held = fold == f; fit = ~held & ~np.isin(grp, grp[held])
        for q in range(6):
            pick[held & (b == q)] = KS[Y[fit & (b == q)].sum(0).argmax()]
    return pick


def oracle():                                       # class-optimal level per episode: each episode maximises I - q U at its class ratio
    pick = np.full(N, k0)
    for c in C:
        ix = np.where(cls == c)[0]; q = J
        for _ in range(30):
            p = (I[ix] - q * U[ix]).argmax(1); q = I[ix, p].sum() / U[ix, p].sum()
        pick[ix] = p
    return pick


out = [f"## how much of the best level per episode the field itself predicts: {NAME} field, {N} episodes, class mIoU from stored per-level counts", "| rule | uses | mIoU | vs fixed 0.5 | folds |", "|---|---|---:|---|---|"]
RULES = [("fixed 0.5 (control)", "nothing", np.full(N, k0)), ("level per bin of the mask area at 0.5", "base-fold labels, 6 numbers", binned()),
         ("boosted trees on histogram descriptors", "base-fold labels", nested(list(range(len(HIST))))), ("boosted trees on histogram and spatial descriptors", "base-fold labels", nested(list(range(X.shape[2])))),
         ("best IoU level per episode", "query truth", (I / np.maximum(U, 1)).argmax(1)), ("class-optimal level per episode", "query truth", oracle())]
for name, uses, pick in RULES:
    d, ci = boot(at(pick), base)
    out.append(f"| {name} | {uses} | {miou(*at(pick)):.2f} | {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | " + " / ".join(f"{miou(*at(pick), fold == f) - miou(*base, fold == f):+.2f}" for f in np.unique(fold)) + " |")
pick = RULES[3][2]; bins = [0, .02, .1, .3, 1.01]; share = area / 1024 ** 2
out += ["", "boosted trees on all descriptors, by true object share: " + "; ".join(f"{lo}-{hi}: n={int(ix.sum())}, mean level {lev[pick[ix]].mean():.2f}, gain {miou(*at(pick), ix) - miou(*base, ix):+.2f}" for lo, hi in zip(bins[:-1], bins[1:]) for ix in [(share >= lo) & (share < hi)] if ix.any())]
(D / f"predictability_{NAME}.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
