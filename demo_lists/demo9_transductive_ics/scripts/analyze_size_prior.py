#!/usr/bin/env python3
"""How much of the cut gap is the target's size in the query, and how accurate must a size estimate be? CPU only.

  python scripts/analyze_size_prior.py --run results/extent_v1/run --out results/self_support_v0/size_prior.json

Level family of the FoRIS score before refinement (57 thresholds, model size). `true_area` picks the level whose
area is closest to the labelled target area: an upper bound for any size estimate, not a method. The estimators are
label-free: FoRIS's own area, a scale from mutual patch matches, a boosted regression fitted on three folds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingRegressor

from analyze_extent import load, miou
from evidence_audit import normalise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    L = np.array([np.stack([r["levels"]["I"], r["levels"]["U"]], 1) for r in recs], float)
    T = np.array([r["area"] for r in recs]) * 1024 * 1024
    A = L[:, :, 1] - T[:, None] + L[:, :, 0]  # predicted area at every level
    mid = int(np.argmin(np.abs(0.2 + 0.0125 * np.arange(L.shape[1]) - 0.5)))
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)

    def row(pick):
        t, b = L[np.arange(n), pick], L[:, mid]
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(b[:, 0], b[:, 1], cls, w)
        return dict(miou=float(miou(t[:, 0], t[:, 1], cls)[0]), gain=float(miou(t[:, 0], t[:, 1], cls)[0] - miou(b[:, 0], b[:, 1], cls)[0]),
                    ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])
    by_area = lambda est: np.abs(np.log(np.maximum(A, 1)) - est[:, None]).argmin(1)
    y = np.log(T)
    out = dict(state="ANALYSED", episodes=n, native=float(miou(L[:, mid, 0], L[:, mid, 1], cls)[0]),
               scope="level family before refinement, model size; exploratory",
               oracle_level=row((L[:, :, 0] / np.maximum(L[:, :, 1], 1)).argmax(1)), true_area=row(by_area(y)))
    rng = np.random.default_rng(1)
    out["true_area_with_lognormal_error"] = {str(s): row(by_area(y + rng.normal(0, s, n))) for s in (0.2, 0.35, 0.5, 0.7)}
    out["true_area_scaled"] = {str(f): row(by_area(y + np.log(f))) for f in (0.5, 0.7, 1.4, 2.0)}
    ratio = A[:, mid] / T
    out["foris_area_over_true"] = dict(quartiles=[float(x) for x in np.quantile(ratio, [0.25, 0.5, 0.75])],
                                       share_above_1p5=float((ratio > 1.5).mean()), share_below_0p67=float((ratio < 0.67).mean()),
                                       log_error_std=float(np.std(np.log(np.maximum(ratio, 1e-3)))),
                                       log_error_median_abs=float(np.median(np.abs(np.log(np.maximum(ratio, 1e-3))))))
    yy, xx = np.divmod(np.arange(4096), 64)
    F = []
    for r in recs:
        z = np.load(a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))
        sn = normalise(z["score"].astype(np.float32)).ravel()
        cov, fwd, back = z["cov"].ravel(), z["fwd_idx"].astype(int), z["back_idx"].astype(int)
        refn = max(int((cov >= 0.5).sum()), 1)
        j = np.nonzero(cov >= 0.5)[0]
        i = back[j]
        ok = np.hypot(yy[fwd[i]] - yy[j], xx[fwd[i]] - xx[j]) <= 1  # the reference patch finds itself again
        mi, mj = i[ok], j[ok]
        geo = np.nan
        if len(mi) >= 4:
            dq = np.hypot(yy[mi][:, None] - yy[mi][None], xx[mi][:, None] - xx[mi][None])
            dr = np.hypot(yy[mj][:, None] - yy[mj][None], xx[mj][:, None] - xx[mj][None])
            m = dr >= 2
            if m.sum() >= 6:
                s = np.median(dq[m] / dr[m])
                geo = np.log(s * s * refn * 256) if s > 0 else np.nan
        lab, k = ndi.label((sn > 0.5).reshape(64, 64))
        big = np.bincount(lab.ravel())[1:].max() if k else 0
        area = lambda t: np.log(max(int((sn > t).sum()), 1) * 256)
        F.append([np.log(refn * 256), area(0.35), area(0.5), area(0.65), area(0.8), np.log(max(int((cov[fwd] >= 0.5).sum()), 1) * 256),
                  np.log(max(len(mi), 1)), k, np.log(max(big, 1) * 256), float(z["score"].max()), float(z["score"].max() - z["score"].min()), geo])
    F = np.array(F)
    est = np.zeros(n)
    for f in range(4):
        model = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05, random_state=0).fit(F[folds != f], y[folds != f])
        est[folds == f] = model.predict(F[folds == f])
    err = lambda e: dict(log_error_std=float(np.nanstd(e - y)), log_error_bias=float(np.nanmean(e - y)), available=float(np.isfinite(e).mean()))
    geo = F[:, -1]
    out["estimators"] = dict(match_geometry=dict(err(geo), **row(by_area(np.where(np.isfinite(geo), geo, F[:, 2])))),
                             held_out_regression=dict(err(est), **row(by_area(est))))
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
