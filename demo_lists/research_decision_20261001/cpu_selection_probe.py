# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "scikit-learn"]
# ///
"""CPU-only feasibility probe on existing INSID3 diagnostic caches.

This is NOT a standard benchmark result: caches contain validation episodes.
Leave-one-class-fold-out fitting, with both support and query image exclusion,
tests whether low-dimensional evidence predicts region foreground occupancy.
The predictor receives no query labels, category identities, or feature vectors.
All reported IoUs use the stored 64x64 fractional ground truth, not CRF masks.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)

FEATURES = [
    "log_area_fraction", "backward_foreground_fraction", "foreground_similarity",
    "background_similarity", "fg_bg_margin", "original_fg_bg_margin",
    "cross_similarity", "seed_similarity", "candidate_fraction",
    "mean_similarity_to_other_clusters", "max_similarity_to_other_clusters",
]


def episode_images(base, fold, count):
    with (base / "splits" / "val" / f"fold{fold}.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    rng = np.random.RandomState(0)
    ids = []
    for _ in range(count):
        cls = int(rng.choice([fold + 4 * v for v in range(20)], 1, replace=False)[0])
        query = str(rng.choice(meta[cls], 1, replace=False)[0])
        while True:
            ref = str(rng.choice(meta[cls], 1, replace=False)[0])
            if ref != query:
                break
        ids.append((cls, query, ref))
    return ids


def extract(t):
    get = lambda key: t[key].float().numpy()
    area = get("area")
    vectors = get("Po")
    sim = np.clip(vectors @ vectors.T, -1, 1)
    np.fill_diagonal(sim, -1)
    fg, bg = get("fgmax"), get("bgmax")
    x = np.column_stack([
        np.log(np.maximum(area / area.sum(), 1e-8)), get("back"), fg, bg, fg - bg,
        get("fgmax_o") - get("bgmax_o"), get("cross"), get("intra"), get("aw"),
        np.sum((sim + np.eye(len(area))) * (area / area.sum())[None], axis=1),
        sim.max(axis=1),
    ])
    return x.astype(np.float64)


def plugin_iou_mask(prob, area):
    """Exact maximizer of ratio-of-expected-counts IoU for disjoint regions.

    This is a plug-in ratio; it is not the expectation of a random IoU.
    """
    prob = np.clip(prob, 0, 1)
    order = np.argsort(-prob, kind="stable")
    tp = np.cumsum(area[order] * prob[order])
    denom = np.cumsum(area[order]) + float(area @ prob) - tp
    score = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    selected = np.zeros(len(prob), dtype=bool)
    if score.max(initial=0) > 0:
        selected[order[: int(np.argmax(score)) + 1]] = True
    return selected


def counts(t, selected):
    lab = t["lab"].numpy().astype(np.int64)
    pred = selected[lab]
    gt = t["g"].numpy().astype(np.float64) / 255
    inter = float((pred * gt).sum())
    union = float(pred.sum() + gt.sum() - inter)
    return inter, union


def miou(rows, name):
    classes = sorted({r["class"] for r in rows})
    scores = []
    for cls in classes:
        a = np.array([r["counts"][name] for r in rows if r["class"] == cls])
        scores.append(a[:, 0].sum() / max(a[:, 1].sum(), 1e-9))
    return 100 * float(np.mean(scores))


def paired_ci(rows, name, baseline="insid3", reps=1000):
    rng = np.random.default_rng(20261001)
    groups = [[r for r in rows if r["class"] == cls] for cls in sorted({r["class"] for r in rows})]
    diffs = np.zeros(reps)
    for group in groups:
        a = np.array([r["counts"][name] for r in group])
        b = np.array([r["counts"][baseline] for r in group])
        ix = rng.integers(len(group), size=(reps, len(group)))
        aa, bb = a[ix].sum(axis=1), b[ix].sum(axis=1)
        diffs += aa[:, 0] / np.maximum(aa[:, 1], 1e-9) - bb[:, 0] / np.maximum(bb[:, 1], 1e-9)
    return (100 * np.quantile(diffs / len(groups), [0.025, 0.975])).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = []
    for fold in range(4):
        path = args.cache / f"budget_coco_f{fold}.tables.pt"
        records = torch.load(path, map_location="cpu", weights_only=False)
        ids = episode_images(args.data, fold, len(records))
        for t, (cls, query, ref) in zip(records, ids, strict=True):
            assert t["c"] == cls
            t.update(fold=fold, images=(query, ref), x=extract(t))
            data.append(t)
    all_rows, fold_reports = [], []
    for fold in range(4):
        test = [t for t in data if t["fold"] == fold]
        blocked = {image for t in test for image in t["images"]}
        train = [t for t in data if t["fold"] != fold and not blocked.intersection(t["images"])]
        x = np.concatenate([t["x"] for t in train])
        y = np.concatenate([t["pur"].float().numpy() for t in train])
        weights = np.concatenate([(t["area"] / t["area"].sum()).numpy() for t in train])
        weights *= len(weights) / weights.sum()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10))
        ridge.fit(x, y, ridge__sample_weight=weights)
        boost = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, l2_regularization=10,
            min_samples_leaf=30, learning_rate=0.08, early_stopping=False, random_state=0,
        )
        boost.fit(x, y, sample_weight=weights)
        rows = []
        for t in test:
            area = t["area"].numpy()
            back = t["back"].float().numpy()
            methods = {
                "insid3": t["combined"].float().numpy() > 0.2,
                "backward_majority": back > 0.5,
                "margin_positive": t["x"][:, 4] > 0,
                "backward_plugin_iou": plugin_iou_mask(back, area),
                "oracle_majority": t["pur"].float().numpy() > 0.5,
            }
            for name, model in (("ridge", ridge), ("boost", boost)):
                pred = np.clip(model.predict(t["x"]), 0, 1)
                methods[name + "_half"] = pred > 0.5
                methods[name + "_plugin_iou"] = plugin_iou_mask(pred, area)
            row = {"fold": fold, "episode": t["e"], "class": t["c"], "query": t["images"][0],
                   "reference": t["images"][1], "counts": {k: counts(t, v) for k, v in methods.items()}}
            rows.append(row)
        report = {"fold": fold, "train_episodes_after_image_exclusion": len(train),
                  "test_episodes": len(test), "miou": {k: miou(rows, k) for k in rows[0]["counts"]}}
        print(json.dumps(report), flush=True)
        all_rows.extend(rows)
        fold_reports.append(report)
    summary = {k: {"miou": miou(all_rows, k), "paired_episode_bootstrap_delta_ci": paired_ci(all_rows, k)}
               for k in all_rows[0]["counts"]}
    result = {"status": "development feasibility diagnostic; not final held-out benchmark",
              "metric": "class mean of summed fractional 64x64 intersections / unions",
              "fit": "leave class fold out; exclude any train support/query image present in test",
              "features": FEATURES, "cpu_threads": 2, "gpu_used": False,
              "folds": fold_reports, "summary": summary, "rows": all_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
