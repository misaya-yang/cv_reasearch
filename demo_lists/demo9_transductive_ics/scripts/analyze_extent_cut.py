"""Extent ledger and cut-rule pilot on saved complete-FoRIS traces (CPU only, no encoder).

Reads results/native_membership_v1/causal_v3/experiment/trace_*.npz (40 reused development tasks,
official COCO-20i seed 0, 10 per fold, arm foris_stateful). One task per class, so the mean over
tasks is the class mIoU of this cohort. Not a method score.

  python scripts/analyze_extent_cut.py [--traces DIR] [--out FILE]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

S8 = np.ones((3, 3), bool)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "results", "native_membership_v1", "causal_v3")


def grid_iou(pred64, t64, tsum):
    """IoU at pixel level of a patch-constant prediction; t64 holds true pixels per patch."""
    if not pred64.any():
        return 0.0
    return float(t64[pred64].sum() / (tsum + (256 - t64[pred64]).sum()))


def contrast_cut(sn, ring=2, ts=np.linspace(0.3, 0.9, 61)):
    """Level set with the largest mean score gap between the region and a ring around it."""
    best, bv = sn > 0.5, -1.0
    for t in ts:
        m = sn > t
        if m.sum() < 4 or m.sum() > 0.9 * m.size:
            continue
        r = ndimage.binary_dilation(m, iterations=ring) & ~m
        if r.any() and sn[m].mean() - sn[r].mean() > bv:
            bv, best = sn[m].mean() - sn[r].mean(), m
    return best


def otsu(sn):
    x = np.sort(sn.ravel())
    n, c, k = len(x), np.cumsum(x), np.arange(1, len(x))
    var = k * (n - k) * (c[k - 1] / k - (c[-1] - c[k - 1]) / (n - k)) ** 2
    return sn > x[var.argmax()]


def stable_cut(sn, d=0.05, ts=np.linspace(0.3, 0.9, 61)):
    a = np.array([(sn > t).sum() for t in ts])
    q = (np.array([(sn > t - d).sum() for t in ts]) - np.array([(sn > t + d).sum() for t in ts])) / np.maximum(a, 1)
    q[a < 4] = 1e9
    return sn > ts[q.argmin()]


def component_tree(sn, ts=np.linspace(0.3, 0.9, 31)):
    nodes, prev, prevlab = [], None, None
    for t in ts:
        lab, n = ndimage.label(sn > t, S8)
        ids = []
        for c in range(1, n + 1):
            m = lab == c
            if m.sum() < 2:
                ids.append(None)
                continue
            r = ndimage.binary_dilation(m, iterations=2) & ~m
            par = None
            if prevlab is not None and prevlab[m][0] > 0:
                par = prev[prevlab[m][0] - 1]
            nodes.append(dict(m=m, a=int(m.sum()), c=sn[m].mean() - (sn[r].mean() if r.any() else sn[m].mean()),
                              par=par, ch=[]))
            ids.append(len(nodes) - 1)
            if par is not None:
                nodes[par]["ch"].append(len(nodes) - 1)
        prev, prevlab = ids, lab
    return nodes


def union(nodes, sel):
    m = np.zeros((64, 64), bool)
    for k in sel:
        m |= nodes[k]["m"]
    return m


def tree_contrast_dp(nodes):
    """Keep a node unless its children's area-weighted contrast is larger. Tried and failed."""
    best = {}
    for i in range(len(nodes) - 1, -1, -1):
        kids = sum((best[c] for c in nodes[i]["ch"]), [])
        if not kids:
            best[i] = [i]
            continue
        a = sum(nodes[k]["a"] for k in kids)
        best[i] = [i] if nodes[i]["c"] >= sum(nodes[k]["a"] * nodes[k]["c"] for k in kids) / a else kids
    return union(nodes, sum((best[i] for i, nd in enumerate(nodes) if nd["par"] is None), []))


def tree_oracle(nodes, t64, tsum):
    """Best set of disjoint components by Dinkelbach; uses ground truth."""
    lam = 0.0
    for _ in range(30):
        val, pick = {}, {}
        for i in range(len(nodes) - 1, -1, -1):
            m = nodes[i]["m"]
            own = t64[m].sum() - lam * (256 - t64[m]).sum()
            kv = sum(val[c] for c in nodes[i]["ch"])
            if own >= kv and own > 0:
                val[i], pick[i] = own, [i]
            elif kv > 0:
                val[i], pick[i] = kv, sum((pick[c] for c in nodes[i]["ch"]), [])
            else:
                val[i], pick[i] = 0.0, []
        sel = sum((pick[i] for i, nd in enumerate(nodes) if nd["par"] is None), [])
        if not sel:
            return 0.0
        new = grid_iou(union(nodes, sel), t64, tsum)
        if abs(new - lam) < 1e-6:
            break
        lam = new
    return lam


def paired(a, b, rng, n=5000):
    d = 100 * (np.asarray(a) - np.asarray(b))
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(diff_pp=float(d.mean()), ci95_pp=[float(lo), float(hi)],
                up=int((d > 1).sum()), down=int((d < -1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(DEFAULT, "experiment"))
    ap.add_argument("--out", default=os.path.join(DEFAULT, "extent_cut_ledger.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(args.traces, "trace_*.npz")))
    led, cut, area = {}, {}, []
    add = lambda d, k, v: d.setdefault(k, []).append(float(v))
    px = dict(fn_ext=0, fn_miss=0, fp_ext=0, fp_iso=0)
    for f in files:
        z = np.load(f)
        T, P = z["truth_model"], z["post_refinement"]
        I, U, TP = (T & P).sum(), (T | P).sum(), T & P
        lt, nt = ndimage.label(T, S8)
        lp, npd = ndimage.label(P, S8)
        found, anch = np.zeros(nt + 1, bool), np.zeros(npd + 1, bool)
        found[np.unique(lt[TP])] = True
        anch[np.unique(lp[TP])] = True
        found[0] = anch[0] = False
        fne, fnm = (T & ~P & found[lt]).sum(), (T & ~P & ~found[lt]).sum()
        fpe, fpi = (P & ~T & anch[lp]).sum(), (P & ~T & ~anch[lp]).sum()
        for k, v in zip(px, (fne, fnm, fpe, fpi)):
            px[k] += int(v)
        add(led, "foris_public", I / U)
        add(led, "fill_misses_inside_located_objects", (I + fne) / U)
        add(led, "fill_wholly_missed_components", (I + fnm) / U)
        add(led, "remove_fp_attached_to_correct_region", I / (U - fpe))
        add(led, "remove_fp_in_wrong_components", I / (U - fpi))
        area.append(float(T.mean()))
        t64, tsum = T.reshape(64, 16, 64, 16).sum((1, 3)).astype(float), T.sum()
        for key in ("part4_score", "part3_score", "part2_score"):
            s = z[key].astype(float)
            sn = (s - s.min()) / (s.max() - s.min())
            add(cut, key + "/midpoint", grid_iou(sn > 0.5, t64, tsum))
            add(cut, key + "/contrast", grid_iou(contrast_cut(sn), t64, tsum))
            add(cut, key + "/oracle_threshold", max(grid_iou(sn > t, t64, tsum) for t in np.linspace(0.02, 0.98, 97)))
            if key != "part4_score":
                continue
            for t in (0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75):
                add(cut, "part4_score/fixed_%.2f" % t, grid_iou(sn > t, t64, tsum))
            add(cut, key + "/otsu", grid_iou(otsu(sn), t64, tsum))
            add(cut, key + "/most_stable", grid_iou(stable_cut(sn), t64, tsum))
            nodes = component_tree(sn)
            add(cut, key + "/tree_contrast_dp", grid_iou(tree_contrast_dp(nodes), t64, tsum))
            add(cut, key + "/tree_oracle", tree_oracle(nodes, t64, tsum))
            mid = sn > 0.5
            add(cut, "area_ratio_midpoint", mid.sum() * 256 / tsum)
        cm = z["cosine_margin"].astype(float)
        cn = (cm - cm.min()) / (cm.max() - cm.min())
        add(cut, "cosine_margin/oracle_threshold", max(grid_iou(cn > t, t64, tsum) for t in np.linspace(0.02, 0.98, 97)))
        add(cut, "grid_ceiling", grid_iou(t64 > 128, t64, tsum))
    area = np.array(area)
    small = area < 0.03
    row = lambda v: dict(all=float(100 * np.mean(v)), small=float(100 * np.mean(np.array(v)[small])),
                         large=float(100 * np.mean(np.array(v)[~small])))
    err = sum(px.values())
    out = dict(
        scope="40 reused development tasks, official COCO-20i seed 0, 10 per fold, complete public FoRIS traces; "
              "exploratory, not a method score",
        tasks=len(files), small_tasks=int(small.sum()), small_definition="target area below 3% of the 1024 frame",
        ledger_model_size={k: row(v) for k, v in led.items()},
        error_pixel_share={k: v / err for k, v in px.items()},
        cut_patch_constant={k: row(v) for k, v in cut.items() if k != "area_ratio_midpoint"},
        area_ratio_midpoint_median=dict(small=float(np.median(np.array(cut["area_ratio_midpoint"])[small])),
                                        large=float(np.median(np.array(cut["area_ratio_midpoint"])[~small]))),
        paired_vs_midpoint={k: paired(cut[k + "/contrast"], cut[k + "/midpoint"], rng)
                            for k in ("part4_score", "part3_score", "part2_score")},
        paired_tree_dp_vs_midpoint=paired(cut["part4_score/tree_contrast_dp"], cut["part4_score/midpoint"], rng),
        script="scripts/analyze_extent_cut.py", interval="bootstrap over tasks, 5000 draws, seed 0",
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
