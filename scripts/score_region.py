#!/usr/bin/env python3
"""Score run_region.py: ordering by layer, and native against enlarged tokens inside the same window (CPU, reads truth).

  python scripts/score_region.py outputs/claude_region_a outputs/claude_region_b
"""
import json
import sys

import numpy as np

rows, T = [], {}
for d in sys.argv[1:]:
    rows += json.load(open(d + "/rows.json")); Z = np.load(d + "/tokens.npz")
    for k in Z.files:
        T.setdefault(k, []).append(Z[k].astype(np.float32))
T = {k: np.concatenate(v) for k, v in T.items()}; N = len(rows); cls = np.array([r["c"] for r in rows]); stat = T.pop("stat")
miou = lambda I, U, ix=slice(None): float(np.mean([I[ix][cls[ix] == c].sum() / max(U[ix][cls[ix] == c].sum(), 1e-9) for c in np.unique(cls[ix])]) * 100)


def best(f, t):
    o = np.argsort(-f, 1); tt = np.take_along_axis(t, o, 1); ci = np.cumsum(tt, 1); cu = t.sum(1, keepdims=True) + np.cumsum(1 - tt, 1); k = (ci / np.maximum(cu, 1e-9)).argmax(1); n = np.arange(N); m = np.zeros(f.shape, bool)
    for i in range(N):
        m[i, o[i, :k[i] + 1]] = True
    return ci[n, k], cu[n, k], m


def auc(p, q):
    r = np.argsort(np.argsort(np.concatenate([p, q]))); return (r[:len(p)].sum() - len(p) * (len(p) - 1) / 2) / (len(p) * len(q))


t = T["truth"]; B = {k: best(T[k], t) for k in T if not k.startswith("win:") and k != "truth"}; e = B["rcg"][0] / np.maximum(B["rcg"][1], 1e-9); chosen = B["rcg"][2]
groups = (("all", e >= 0), ("RCG best-cut IoU < 0.2", e < .2), ("0.2-0.5", (e >= .2) & (e < .5)), ("0.5-0.8", (e >= .5) & (e < .8)), (">= 0.8", e >= .8))
print("## ordering score by field, %d episodes; and AUC of true object tokens against the tokens RCG wrongly keeps" % N)
print("| field | ordering score | " + " | ".join("AUC, %s" % g for g, _ in groups[1:]) + " |"); print("|---|---:|" + "---:|" * 4)
for k in sorted(B, key=lambda k: miou(*B[k][:2])):
    cells = []
    for _, ix in groups[1:]:
        v = [auc(T[k][i][t[i] >= .9], T[k][i][(t[i] <= .1) & chosen[i]]) for i in np.flatnonzero(ix) if (t[i] >= .9).sum() > 2 and ((t[i] <= .1) & chosen[i]).sum() > 2]; cells.append("%.2f (n=%d)" % (np.mean(v), len(v)))
    print("| %s | %.2f | " % (k, miou(*B[k][:2])) + " | ".join(cells) + " |")
ok = stat[:, 7] > 0; wt = T["win:truth"]; Wf, Wz = best(T["win:full"], wt), best(T["win:zoom"], wt)
print("\n## inside the window around the true object: native tokens against tokens of the enlarged window")
print("| episodes | n | window side / image | cos(ref signature, object signature) native | enlarged | window mIoU native tokens | enlarged tokens | mean episode IoU native | enlarged |"); print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
zoomed = ok & (stat[:, 2] < .75)
for g, ix in groups + (("window < 0.75 of the image", zoomed), ("window < 0.75 and RCG IoU < 0.5", zoomed & (e < .5)), ("window >= 0.75", ok & ~zoomed)):
    ix = ix & ok
    if ix.sum():
        print("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (g, ix.sum(), stat[ix, 2].mean(), stat[ix, 3].mean(), stat[ix, 4].mean(), miou(Wf[0], Wf[1], ix), miou(Wz[0], Wz[1], ix), stat[ix, 5].mean(), stat[ix, 6].mean()))
