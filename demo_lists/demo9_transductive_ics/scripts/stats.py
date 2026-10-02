#!/usr/bin/env python3
"""CPU: tables and paired intervals from run_episodes.py / run_blackbox.py outputs (one JSON per fold).

mIoU = mean over classes of sum(intersection) / sum(union), per fold, then averaged over folds. The interval of a difference
resamples EPISODES with replacement inside every fold (both methods see the same resample), B times.
  python scripts/stats.py results/e1_f*.json --base 1shot [--vs agree:tophalf]
"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--base", default="1shot"); ap.add_argument("--vs", default=""); ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args(); R = [json.load(open(f)) for f in sorted(a.files)]
keys = [k for k in R[0]["records"][0]["iu"] if all(k in x["iu"] for r in R for x in r["records"])]      # rows present in every episode
def fold_arrays(r):
    c = np.array([x["c"] for x in r["records"]]); cid = np.unique(c, return_inverse=True)[1]
    return cid, {k: np.array([x["iu"][k] for x in r["records"]]) for k in keys}
FA = [fold_arrays(r) for r in R]
def miou(cid, iu, sel=None):
    if sel is not None: cid, iu = cid[sel], iu[sel]
    n = cid.max() + 1; i = np.bincount(cid, iu[:, 0], n); u = np.bincount(cid, iu[:, 1], n); ok = u > 0
    return 100 * float(np.mean(i[ok] / u[ok]))
rng = np.random.default_rng(0); sels = [[rng.integers(0, len(cid), len(cid)) for _ in range(a.B)] for cid, _ in FA]
def diff_ci(k, base):
    d = np.array([np.mean([miou(cid, A[k], s[b]) - miou(cid, A[base], s[b]) for (cid, A), s in zip(FA, sels)]) for b in range(a.B)])
    return np.percentile(d, 2.5), np.percentile(d, 97.5)
print(f"{len(R)} folds, episodes per fold: {[len(r['records']) for r in R]}")
print(f"{'method':24s}" + "".join(f"{'fold' + str(i):>8s}" for i in range(len(R))) + f"{'mean':>8s}   diff vs {a.base} [95% interval]" + (f"   diff vs {a.vs}" if a.vs else ""))
for k in keys:
    v = [miou(cid, A[k]) for cid, A in FA]; line = f"{k:24s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}"
    if k != a.base:
        lo, hi = diff_ci(k, a.base); line += f"   {np.mean(v) - np.mean([miou(cid, A[a.base]) for cid, A in FA]):+5.1f} [{lo:+.1f}, {hi:+.1f}]"
    if a.vs and k not in (a.vs, a.base):
        lo, hi = diff_ci(k, a.vs); line += f"   {np.mean(v) - np.mean([miou(cid, A[a.vs]) for cid, A in FA]):+5.1f} [{lo:+.1f}, {hi:+.1f}]"
    print(line)
