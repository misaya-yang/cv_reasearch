#!/usr/bin/env python3
"""Ceiling of a query-only region hierarchy against level sets of FoRIS / RCG (CPU, token level, reads truth).

  python scripts/score_candidates.py outputs/claude_candidates_a outputs/claude_candidates_b
"""
import json
import os
import sys

import numpy as np

rows, T = [], {}
for j, d in enumerate(sys.argv[1:]):
    rows += [dict(r, c=r["c"] + 1000 * j * int(os.environ.get("SEPARATE", "0"))) for r in json.load(open(d + "/rows.json"))]; Z = np.load(d + "/tokens.npz")
    for k in Z.files:
        T.setdefault(k, []).append(Z[k])
T = {k: np.concatenate(v) for k, v in T.items()}; N = len(rows); cls = np.array([r["c"] for r in rows]); t = T["truth"].astype(np.float64); tot = t.sum(1)
miou = lambda I, U: float(np.mean([I[cls == c].sum() / max(U[cls == c].sum(), 1e-9) for c in np.unique(cls)]) * 100)


def level(f, cut=None):
    f = f.astype(np.float64)
    if cut is not None:
        m = f > cut; return (m * t).sum(1), tot + (m * (1 - t)).sum(1)
    o = np.argsort(-f, 1); tt = np.take_along_axis(t, o, 1); ci = np.cumsum(tt, 1); cu = tot[:, None] + np.cumsum(1 - tt, 1); k = (ci / np.maximum(cu, 1e-9)).argmax(1); n = np.arange(N); return ci[n, k], cu[n, k]


print("## candidates from the query alone against level sets of a reference score: token-level class mIoU, %d episodes" % N); print("| candidate set | chosen how | mIoU |"); print("|---|---|---:|")
sn = (T["score"] - T["score"].min(1, keepdims=True)) / np.maximum(T["score"].max(1, keepdims=True) - T["score"].min(1, keepdims=True), 1e-9)
print("| level sets of the FoRIS score | fixed 0.5 (FoRIS before its CRF) | %.2f |" % miou(*level(sn, .5))); print("| level sets of the RCG field | fixed 0.5 | %.2f |" % miou(*level(T["rcg"], .5)))
print("| level sets of the FoRIS score | best level per episode (truth) | %.2f |" % miou(*level(T["score"]))); print("| level sets of the RCG field | best level per episode (truth) | %.2f |" % miou(*level(T["rcg"])))
for l in [k[5:] for k in T if k.startswith("node:")]:
    S = T["node:" + l].astype(np.float64); kids = T["kids:" + l].astype(np.int64); size, tr = S[:, :, 0], S[:, :, 1]; iou = tr / (tot[:, None] + size - tr)
    b = iou.argmax(1); n = np.arange(N); I1, U1 = tr[n, b], tot + size[n, b] - tr[n, b]; print("| nodes of the %s hierarchy | best single node (truth) | %.2f |" % (l, miou(I1, U1)))
    anc = np.zeros((N, 8191), np.int64)                                 # parent of every node, to keep unions disjoint
    for i in range(N):
        anc[i, kids[i][:, 0]] = np.arange(4096, 8191); anc[i, kids[i][:, 1]] = np.arange(4096, 8191); anc[i, -1] = -1
    Ig, Fg = I1.copy(), (size[n, b] - tr[n, b]).copy(); banned = np.zeros((N, 8191), bool)

    def ban(i, k):                                                      # the node, its ancestors and its descendants
        j = k
        while j >= 0:
            banned[i, j] = True; j = anc[i, j]
        stack = [k]
        while stack:
            j = stack.pop(); banned[i, j] = True
            if j >= 4096:
                stack += list(kids[i][j - 4096])
    for i in range(N):
        ban(i, b[i])
    for extra in (2, 3):
        for i in range(N):
            cand = (Ig[i] + tr[i]) / (tot[i] + Fg[i] + size[i] - tr[i]); cand[banned[i]] = -1; k = int(cand.argmax())
            if cand[k] > Ig[i] / max(tot[i] + Fg[i], 1e-9):
                Ig[i] += tr[i, k]; Fg[i] += size[i, k] - tr[i, k]; ban(i, k)
        print("| nodes of the %s hierarchy | best union of up to %d disjoint nodes (truth, greedy) | %.2f |" % (l, extra, miou(Ig, tot + Fg)))
    for name, col in (("reference cosine", 2), ("RCG field", 3)):       # truth-free: the node that best splits the field (largest between-class variance)
        s = S[:, :, col]; tot_s = s[:, -1:]; inside = s / size; outside = (tot_s - s) / np.maximum(4096 - size, 1); bcv = size * (4096 - size) * (inside - outside) ** 2 * (inside > outside); c = bcv.argmax(1)
        print("| nodes of the %s hierarchy | node that best splits the %s (no truth) | %.2f |" % (l, name, miou(tr[n, c], tot + size[n, c] - tr[n, c])))
