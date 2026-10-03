#!/usr/bin/env python3
"""Does the evidence FoRIS already computed contain more than its own cut uses? A held-out learned read-out. CPU only.

  python scripts/probe_extent_evidence.py --run results/extent_v1/run --out results/extent_v1/evidence_probe.json

A per-patch gradient-boosted classifier is fitted on three COCO-20i folds and scored on the fourth, whose classes it
never saw. Three sets of label-free inputs: the final score with its spatial context; everything the packets hold
(stage scores, best foreground and background similarity, nearest-neighbour label, round trip, adjacency, signature);
and the raw correspondence evidence without any FoRIS score. This is a measuring device for information, not a method:
about 180 training episodes, patch level (64 x 64), no refinement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from analyze_extent import load, miou

SETS = dict(
    score=("sn", "rank", "sn3", "sn7", "sn15", "snmax7", "sn_rel15", "comp_area", "comp_peak", "depth", "dist", "mid_area"),
    correspondence=("fg", "bg", "margin", "nn_label", "nn_sim", "hits_fg", "hits_bg", "cycle", "margin3", "margin7",
                    "nn_label3", "nn_label7", "ref_area"),
    extra=("s2", "s3", "s2_3", "s3_3", "aff_mean", "aff_min", "sig", "raw_max", "raw_range"))
ARMS = dict(score_only=("score",), all_evidence=("score", "correspondence", "extra"), correspondence_only=("correspondence",))


def norm(x):
    x = x - x.min()
    return x / max(float(x.max()), 1e-6)


def per_edge(right, down, fn):
    """Fold the two edge maps [h, w-1], [h-1, w] to one value per patch over its four edges."""
    h, w = right.shape[0], down.shape[1]
    stack = np.full((4, h, w), np.nan)
    stack[0, :, :-1], stack[1, :, 1:], stack[2, :-1], stack[3, 1:] = right, right, down, down
    return fn(stack, 0)


def features(z, h=64, w=64):
    f = {}
    sn = norm(z["score"].astype(float))
    box = lambda x, k: ndi.uniform_filter(x, k, mode="nearest")
    f["sn"] = sn
    f["rank"] = (np.argsort(np.argsort(sn.ravel())) / (h * w - 1)).reshape(h, w)
    f["sn3"], f["sn7"], f["sn15"] = box(sn, 3), box(sn, 7), box(sn, 15)
    f["snmax7"] = ndi.maximum_filter(sn, 7, mode="nearest")
    f["sn_rel15"] = sn - f["sn15"]
    m = sn > 0.5
    lab, k = ndi.label(m)
    area = np.bincount(lab.ravel(), minlength=k + 1) / (h * w)
    peak = ndi.maximum(sn, lab, np.arange(k + 1)) if k else np.zeros(1)
    f["comp_area"] = np.where(m, area[lab], 0.0)
    f["comp_peak"] = np.where(m, np.asarray(peak)[lab], 0.0)
    f["depth"] = ndi.distance_transform_edt(m) if m.any() else np.zeros((h, w))
    f["dist"] = ndi.distance_transform_edt(~m) if m.any() else np.full((h, w), 64.0)
    f["mid_area"] = np.full((h, w), m.mean())
    cov = z["cov"].astype(float).ravel()
    fg, bg = z["fg_max"].astype(float).reshape(h, w), z["bg_max"].astype(float).reshape(h, w)
    fwd, back = z["fwd_idx"].astype(int), z["back_idx"].astype(int)
    f["fg"], f["bg"], f["margin"] = fg, bg, fg - bg
    f["nn_label"] = cov[fwd].reshape(h, w)
    f["nn_sim"] = z["fwd_sim"].astype(float).reshape(h, w)
    f["hits_fg"] = np.bincount(back, weights=cov, minlength=h * w).reshape(h, w)
    f["hits_bg"] = np.bincount(back, weights=1 - cov, minlength=h * w).reshape(h, w)
    there = back[fwd]  # where the nearest reference patch points back to
    far = np.hypot(there // w - np.arange(h * w) // w, there % w - np.arange(h * w) % w)
    f["cycle"] = (cov[fwd] * (far <= 2)).reshape(h, w)
    f["margin3"], f["margin7"] = box(f["margin"], 3), box(f["margin"], 7)
    f["nn_label3"], f["nn_label7"] = box(f["nn_label"], 3), box(f["nn_label"], 7)
    f["ref_area"] = np.full((h, w), cov.mean())
    f["s2"], f["s3"] = norm(z["s2"].astype(float)), norm(z["s3"].astype(float))
    f["s2_3"], f["s3_3"] = box(f["s2"], 3), box(f["s3"], 3)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f["aff_mean"] = per_edge(z["aff_r"], z["aff_d"], np.nanmean)
            f["aff_min"] = per_edge(z["aff_r"], z["aff_d"], np.nanmin)
            f["sig"] = per_edge(np.maximum(z["sig_right"], z["sig_left"]), np.maximum(z["sig_down"], z["sig_up"]), np.nanmax)
    f["raw_max"] = np.full((h, w), float(z["score"].max()))
    f["raw_range"] = np.full((h, w), float(z["score"].max() - z["score"].min()))
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--train-rows", type=int, default=400000)
    a = p.parse_args()
    recs = load(a.run)
    X, T = [], []
    for r in recs:
        z = np.load(a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))
        f = features(z)
        X.append(f)
        truth = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3))
        T.append(truth)
    n = len(recs)
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    tf = np.stack(T).reshape(n, -1)  # truth fraction per patch

    def iu(pred):  # [n, 4096] bool -> [n, 2] in pixels
        i = (tf * pred).sum(1) * 256
        return np.stack([i, tf.sum(1) * 256 + pred.sum(1) * 256 - i], 1)
    cols = lambda names: np.stack([np.stack([x[k].ravel() for k in names], 1) for x in X])  # [n, 4096, d]
    base = iu(np.stack([x["sn"].ravel() > 0.5 for x in X]))
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    score = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])

    def paired(t):
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
        return dict(miou=score(t), gain=score(t) - score(base), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                    by_fold={int(f): score(t, folds == f) - score(base, folds == f) for f in np.unique(folds)})
    levels = np.stack([iu(np.stack([x["sn"].ravel() > 0.2 + 0.0125 * k for x in X])) for k in range(57)], 1)
    best = (levels[:, :, 0] / np.maximum(levels[:, :, 1], 1)).argmax(1)
    out = dict(state="PROBED", episodes=n, scope="patch level, no refinement; classes of the scored fold are unseen in fitting; "
               "exploratory measuring device, not a method score", native_patch_level=score(base),
               oracle_level=paired(levels[np.arange(n), best]), grid_ceiling=paired(iu(tf > 0.5)), arms={})
    rng = np.random.default_rng(0)
    for arm, sets in ARMS.items():
        names = [k for s in sets for k in SETS[s]]
        D = cols(names)
        pred = np.zeros((n, 4096), bool)
        for f in np.unique(folds):
            held = {x for r in recs if r["fold"] == f for x in (r["support"], r["query"])}
            tr = np.array([j for j, r in enumerate(recs) if r["fold"] != f and r["support"] not in held and r["query"] not in held])
            xs, ys = D[tr].reshape(-1, D.shape[-1]), (tf[tr] > 0.5).ravel()
            pick = rng.choice(len(ys), min(a.train_rows, len(ys)), replace=False)
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
            model.fit(xs[pick], ys[pick])
            te = np.nonzero(folds == f)[0]
            pred[te] = model.predict_proba(D[te].reshape(-1, D.shape[-1]))[:, 1].reshape(len(te), -1) > 0.5
        out["arms"][arm] = dict(paired(iu(pred)), inputs=len(names))
        print(arm, json.dumps(out["arms"][arm]), flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
