#!/usr/bin/env python3
"""Ordering score (token-level class mIoU under the best cut of each episode) of every stored field (CPU, reads truth).

  python scripts/score_transfer.py outputs/claude_transfer_a outputs/claude_transfer_b
"""
import json
import sys

import numpy as np

rows, T = [], {}
for d in sys.argv[1:]:
    rows += json.load(open(d + "/rows.json")); Z = np.load(d + "/tokens.npz")
    for k in Z.files:
        T.setdefault(k, []).append(Z[k].astype(np.float32))
T = {k: np.concatenate(v) for k, v in T.items()}; N = len(rows); cls = np.array([r["c"] for r in rows]); t = T.pop("truth"); stat = T.pop("stat")
miou = lambda I, U, ix=slice(None): float(np.mean([I[ix][cls[ix] == c].sum() / max(U[ix][cls[ix] == c].sum(), 1e-9) for c in np.unique(cls[ix])]) * 100)


def best(f):
    o = np.argsort(-f, 1); tt = np.take_along_axis(t, o, 1); ci = np.cumsum(tt, 1); cu = t.sum(1, keepdims=True) + np.cumsum(1 - tt, 1); k = (ci / np.maximum(cu, 1e-9)).argmax(1); n = np.arange(N)
    return ci[n, k], cu[n, k]


B = {k: best(v) for k, v in T.items()}; B["token ceiling"] = best(t); share = stat[:, 0]
bins = ((0, .02), (.02, .1), (.1, .3), (.3, 1.01)); order = sorted(B, key=lambda k: miou(*B[k]))
print("## ordering score: token-level class mIoU under the best cut of each episode, %d episodes" % N)
print("| field | all | " + " | ".join("object share %g-%g (n=%d)" % (lo, hi, ((share >= lo) & (share < hi)).sum()) for lo, hi in bins) + " | episodes with IoU < 0.2 |"); print("|---|---:|" + "---:|" * (len(bins) + 1))
for k in order:
    I, U = B[k]; print("| %s | %.2f | " % (k, miou(I, U)) + " | ".join("%.2f" % miou(I, U, (share >= lo) & (share < hi)) for lo, hi in bins) + " | %d |" % int((I / np.maximum(U, 1e-9) < .2).sum()))
e = lambda k: B[k][0] / np.maximum(B[k][1], 1e-9)
print("\n## per episode: FoRIS final score against the ceilings")
for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
    ix = (e("score") >= lo) & (e("score") < hi)
    print("FoRIS best-cut IoU %g-%g: n=%d; mean IoU cos %.2f, cos_oracle %.2f, lin_oracle %.2f, rcg %.2f; cos(ref object mean, query object mean) %.2f, cos(ref object mean, query background mean) %.2f, cos(query object mean, query background mean) %.2f, object share %.3f"
          % (lo, hi, ix.sum(), e("cos")[ix].mean(), e("cos_oracle")[ix].mean(), e("lin_oracle")[ix].mean(), e("rcg")[ix].mean(), stat[ix, 1].mean(), stat[ix, 2].mean(), stat[ix, 3].mean(), share[ix].mean()))
