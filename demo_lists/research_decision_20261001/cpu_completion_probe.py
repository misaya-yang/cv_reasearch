# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "scipy"]
# ///
"""A bounded CPU test of support-calibrated graph completion.

Calibration hides foreground components of the annotated reference, never query
labels. Query ground truth is used only after all masks have been constructed.
This is a development probe, not an established method or benchmark claim.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

torch.set_num_threads(2)
LAMBDAS = np.array([0.0, 0.25, 1.0, 4.0, 16.0, 64.0])


def region_graph(children, heights, leaf_gt=None, cut=0.4):
    n = len(children) + 1
    children = children.astype(np.int64)
    birth = np.r_[np.zeros(n), heights.astype(float)]
    parent = np.full(2 * n - 1, -1, dtype=int)
    parent[children[:, 0]] = n + np.arange(n - 1)
    parent[children[:, 1]] = n + np.arange(n - 1)
    death = np.full(2 * n - 1, np.inf)
    death[parent >= 0] = birth[parent[parent >= 0]]
    roots = np.flatnonzero((birth < cut) & (death >= cut))
    if len(roots) == 0:
        roots = np.array([2 * n - 2])
    labels = np.empty(n, dtype=int)
    groups = [[] for _ in range(2 * n - 1)]
    for k, root in enumerate(roots):
        groups[root] = [k]
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if node < n:
                labels[node] = k
            else:
                stack.extend(children[node - n])
    distance = np.zeros((len(roots), len(roots)))
    root_set = set(roots.tolist())
    for k, (left, right) in enumerate(children):
        node = n + k
        if node in root_set:
            continue
        a, b = groups[left], groups[right]
        if a and b:
            distance[np.ix_(a, b)] = birth[node]
            distance[np.ix_(b, a)] = birth[node]
        groups[node] = a + b
    area = np.bincount(labels, minlength=len(roots)).astype(float)
    kernel = np.exp(-distance / 0.1)
    np.fill_diagonal(kernel, 0)
    degrees = np.maximum(kernel.sum(axis=1), 1e-12)
    weights = kernel / np.sqrt(degrees[:, None] * degrees[None, :])
    weights *= np.sqrt(area[:, None] * area[None, :])
    truth = None
    if leaf_gt is not None:
        truth = np.bincount(labels, weights=leaf_gt.astype(float), minlength=len(roots)) / area
    return labels, area, weights, truth


def cut_mask(observed, area, weights, strength):
    n = len(area)
    cap = np.zeros((n + 2, n + 2), dtype=float)
    # Source side means foreground. Observed positive regions are retained.
    cap[:n, :n] = strength * weights
    cap[n, :n] = area * observed
    cap[:n, n + 1] = area * (1 - observed)
    cap[n, :n][observed > 0.5] += 1e6
    cap = csr_matrix(np.rint(cap * 1000).astype(np.int64))
    flow = maximum_flow(cap, n, n + 1).flow
    residual = cap - flow
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    selected = np.zeros(n + 2, dtype=bool)
    selected[reachable] = True
    return selected[:n]


def iou(pred, truth, area):
    inter = float(np.sum(pred * truth * area))
    union = float(np.sum((pred + truth - pred * truth) * area))
    return inter / max(union, 1e-9)


def choose_strength(area, weights, truth):
    positives = np.flatnonzero(truth >= 0.9)
    if len(positives) == 0:
        return 0, np.zeros(len(LAMBDAS)), 0
    # At most five seeds, chosen by reference area ranks, independent of query GT.
    positives = positives[np.argsort(area[positives])]
    ix = np.unique(np.linspace(0, len(positives) - 1, min(5, len(positives))).astype(int))
    seeds = positives[ix]
    risk = np.zeros(len(LAMBDAS))
    for seed in seeds:
        observed = np.zeros(len(area))
        observed[seed] = 1
        for k, strength in enumerate(LAMBDAS):
            mask = cut_mask(observed, area, weights, strength)
            risk[k] += 1 - iou(mask, truth, area)
    risk /= len(seeds)
    return int(np.argmin(risk)), risk, len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    records = torch.load(args.cache, map_location="cpu", weights_only=False)
    totals, rows = {}, []
    for ix, t in enumerate(records):
        _, ra, rw, ry = region_graph(t["r_children"], t["r_dist"], t["r_g"])
        choice, calibration_risk, seeds = choose_strength(ra, rw, ry)
        labels, area, weights, _ = region_graph(t["children"], t["dist"])
        base = t["insid3"].astype(float)
        observations = np.bincount(labels, weights=base, minlength=len(area)) / area
        # Cache heights have FP16 rounding. Use majority if the reconstructed cut differs.
        observations = (observations > 0.5).astype(float)
        predictions = {f"fixed_{strength:g}": cut_mask(observations, area, weights, strength)[labels]
                       for strength in LAMBDAS}
        predictions["support_calibrated"] = predictions[f"fixed_{LAMBDAS[choice]:g}"]
        predictions["insid3"] = base
        # The following block is the only use of query ground truth.
        gt = t["g"].astype(float)
        counts = {}
        for name, mask in predictions.items():
            inter = float(mask @ gt)
            union = float(mask.sum() + gt.sum() - inter)
            counts[name] = [inter, union]
            total = totals.setdefault(name, {}).setdefault(int(t["c"]), [0.0, 0.0])
            total[0] += inter
            total[1] += union
        rows.append({"episode": int(t["e"]), "class": int(t["c"]), "chosen_lambda": float(LAMBDAS[choice]),
                     "calibration_risk": calibration_risk.tolist(), "reference_seeds": seeds,
                     "cut_reconstruction_difference": int(np.count_nonzero(observations[labels] != base)),
                     "counts": counts})
        if (ix + 1) % 50 == 0:
            print("episodes", ix + 1, flush=True)
    summary = {name: 100 * float(np.mean([i / max(u, 1e-9) for i, u in cls.values()]))
               for name, cls in totals.items()}
    result = {"status": "development probe", "metric": "fractional64grid", "n": len(rows),
              "summary": summary, "lambda_histogram": {str(x): int(sum(r["chosen_lambda"] == x for r in rows)) for x in LAMBDAS},
              "gpu_used": False, "rows": rows}
    args.out.parent.mkdir(exist_ok=True, parents=True)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
