#!/usr/bin/env python3
"""Exhaust every existing-mask add/delete recipe from a sealed DEV histogram."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def zeta(hist, superset):
    out = hist.copy()
    step = 1
    while step < out.shape[1]:
        block = out.reshape(out.shape[0], -1, 2, step, 2)
        if superset:
            block[:, :, 0] += block[:, :, 1]
        else:
            block[:, :, 1] += block[:, :, 0]
        step *= 2
    return out


def solve(source, destination):
    started = time.monotonic()
    arms = json.loads((source / "arms.json").read_text())
    counts = np.load(source / "class_patterns.npy")
    k = len(arms)
    if counts.shape != (80, 1 << k, 2):
        raise ValueError("Expected complete80-class histogram with all producers")
    count = 1 << (k - 1)
    local = np.arange(count, dtype=np.int64)
    population = np.array([int(x).bit_count() for x in local], np.uint8)
    costs = 1 + population[np.bitwise_or(local[:, None], local[None, :])]
    truth = counts[:, :, 1].sum(axis=1)
    results = []
    for origin in range(k):
        lower = (1 << origin) - 1
        expanded = (local & lower) | ((local >> origin) << (origin + 1))
        outside = zeta(counts[:, expanded, :], False)
        additions = outside[:, -1:, :] - outside[:, (count - 1) ^ local, :]
        retained = zeta(counts[:, expanded | (1 << origin), :], True)
        scores = np.zeros((count, count), np.float64)
        for c in range(len(truth)):
            numerator = additions[c, :, 1, None] + retained[c, None, :, 1]
            denominator = truth[c] + additions[c, :, 0, None] + retained[c, None, :, 0]
            scores += np.divide(numerator, denominator, out=np.zeros_like(scores), where=denominator > 0)
        scores *= 100 / len(truth)
        maximum = float(scores.max())
        winning = scores == maximum
        minimum_cost = int(costs[winning].min())
        a, b = np.argwhere(winning & (costs == minimum_cost))[0].tolist()
        remaining = [i for i in range(k) if i != origin]
        recipe = dict(origin=arms[origin]["id"],
                      additions=[arms[i]["id"] for bit, i in enumerate(remaining) if a & (1 << bit)],
                      deletions=[arms[i]["id"] for bit, i in enumerate(remaining) if b & (1 << bit)],
                      origin_index=origin, a=a, b=b)
        row = dict(score=maximum, distinct_producers=minimum_cost, recipe=recipe,
                   recipes_examined=count * count, seconds=time.monotonic() - started)
        results.append(row)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "progress.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    best = sorted(results, key=lambda x: (-x["score"], x["distinct_producers"]))[0]
    report = dict(state="ALL_LIBRARY_RECIPES_ENUMERATED", K=k,
                  recipes_examined=k * count * count, best=best, per_origin=results,
                  histogram_sha256=hashlib.sha256((source / "class_patterns.npy").read_bytes()).hexdigest(),
                  interpretation="GT-labelled global DEV optimum; fixed recipe, no per-query GT routing; no selection-adjusted confidence interval", 
                  cost="distinct producer count only, not measured runtime",
                  seconds=time.monotonic() - started)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    solve(args.hist, args.out)
