#!/usr/bin/env python3
"""Does FoRIS lose accuracy when the reference object and the query object differ in scale? CPU only, saved results.

  python scripts/analyze_scale_mismatch.py --run results/extent_v1/run --out results/self_support_v0/scale_mismatch.json

Scale ratio = sqrt(query target area / reference object area), both as shares of their own image. Uses query
labels: this sizes a cause, it is not a method. Exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_extent import load, miou


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    cls = np.array([r["c"] for r in recs])
    ref = np.array([float(np.load(a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))["cov"].mean()) for r in recs])
    q = np.array([r["area"] for r in recs])
    arms = {k: np.array([r["original_iu"][k] for r in recs], float) for k in ("native", "zoom_query", "zoom_pair", "zoom_oracle", "oracle_cut")}
    N = arms["native"]
    iou = 100 * N[:, 0] / np.maximum(N[:, 1], 1)
    L = np.array([np.stack([r["levels"]["I"], r["levels"]["U"]], 1) for r in recs], float)
    T = q * 1024 * 1024
    over = (L[:, 24, 1] - T + L[:, 24, 0]) / T  # FoRIS area over true area, before refinement
    s = np.log2(np.sqrt(q / np.maximum(ref, 1e-6)))
    gain = lambda k, m: float(miou(arms[k][m, 0], arms[k][m, 1], cls[m])[0] - miou(N[m, 0], N[m, 1], cls[m])[0])
    out = dict(state="ANALYSED", episodes=n, scope="original resolution; uses query labels; exploratory", by_scale_ratio={}, cells={})
    b = np.digitize(s, [-1.5, -0.5, 0.5, 1.5])
    for i, name in enumerate(("query_below_0p35x", "0p35_to_0p7x", "0p7_to_1p4x", "1p4_to_2p8x", "query_above_2p8x")):
        m = b == i
        out["by_scale_ratio"][name] = dict(episodes=int(m.sum()), class_miou=float(miou(N[m, 0], N[m, 1], cls[m])[0]),
                                           share_area_above_1p5x=float((over[m] > 1.5).mean()), share_area_below_0p67x=float((over[m] < 0.67).mean()),
                                           gain_zoom_first_pass_box=gain("zoom_pair", m), gain_zoom_truth_box=gain("zoom_oracle", m), gain_oracle_cut=gain("oracle_cut", m))
    qb, rb = np.digitize(q, [0.03, 0.1]), np.digitize(ref, [0.03, 0.1])
    names = ("below_3pct", "3_to_10pct", "above_10pct")
    for i in range(3):
        for j in range(3):
            m = (qb == i) & (rb == j)
            out["cells"]["query_%s__reference_%s" % (names[i], names[j])] = dict(episodes=int(m.sum()), mean_iou=float(iou[m].mean()) if m.any() else None)
    X = np.stack([np.log(q), np.abs(s), np.ones(n)], 1)
    rng = np.random.default_rng(0)
    coef = np.linalg.lstsq(X, iou, rcond=None)[0]
    boot = np.array([np.linalg.lstsq(X[k], iou[k], rcond=None)[0] for k in (rng.integers(0, n, n) for _ in range(2000))])
    out["regression_iou_on_ln_query_area_and_abs_log2_scale_ratio"] = dict(
        per_e_fold_of_query_area=float(coef[0]), per_doubling_of_scale_mismatch=float(coef[1]),
        mismatch_ci95=[float(x) for x in np.percentile(boot[:, 1], [2.5, 97.5])])
    out["mismatch_at_least_2x"] = dict(episodes=int((np.abs(s) >= 1).sum()), mean_iou=float(iou[np.abs(s) >= 1].mean()),
                                       others_mean_iou=float(iou[np.abs(s) < 1].mean()))
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
