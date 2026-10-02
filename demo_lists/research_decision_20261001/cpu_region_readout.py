# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Fit foreground/background contrast from one reference's region vectors.

No new encoder runs: evaluate only episodes whose reference ALSO exists as a
cached query of the same class. This deterministic cache-availability subset is
a development diagnostic, not the full COCO benchmark. No query GT enters fitting.
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

torch.set_num_threads(2)


def ids(base, fold, count):
    with (base / "splits" / "val" / f"fold{fold}.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    rng = np.random.RandomState(0)
    result = []
    for _ in range(count):
        c = int(rng.choice([fold + 4 * v for v in range(20)], 1, replace=False)[0])
        query = str(rng.choice(meta[c], 1, replace=False)[0])
        while True:
            ref = str(rng.choice(meta[c], 1, replace=False)[0])
            if ref != query:
                break
        result.append((c, query, ref))
    return result


def regression(kr, kq, y, w, regularizer):
    mean = kr @ w
    grand = float(w @ mean)
    centered = kr - mean[:, None] - mean[None, :] + grand
    query_centered = kq - (kq @ w)[:, None] - mean[None, :] + grand
    root_w = np.sqrt(w)
    inverse = np.linalg.inv(centered * root_w[:, None] * root_w[None, :] + regularizer * np.eye(len(y)))
    operator = root_w[:, None] * inverse * root_w[None, :]
    hat = np.ones((len(y), 1)) @ w[None, :] + centered @ operator
    mean_y = float(w @ y)
    prediction = mean_y + query_centered @ operator @ (y - mean_y)
    fitted = hat @ y
    diagonal = np.diag(hat)
    loo = (fitted - diagonal * y) / np.maximum(1 - diagonal, 1e-5)
    return prediction, float(w @ (loo - y) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    records, lookup = [], {}
    for fold in range(4):
        tables = torch.load(args.cache / f"budget_coco_f{fold}.tables.pt", map_location="cpu", weights_only=False)
        names = ids(args.data, fold, len(tables))
        for t, (c, q, r) in zip(tables, names, strict=True):
            assert c == t["c"]
            t.update(fold=fold, query=q, reference=r)
            records.append(t)
            lookup[(c, q)] = t
    totals, rows = {}, []
    for t in records:
        ref = lookup.get((t["c"], t["reference"]))
        if ref is None:
            continue
        y = ref["pur"].float().numpy().astype(float)
        a = ref["area"].numpy().astype(float)
        fg = float(a @ y)
        bg = float(a @ (1 - y))
        if fg < 1 or bg < 1:
            continue
        w = 0.5 * a * (y / fg + (1 - y) / bg)
        w /= w.sum()
        predictions = {"insid3": t["combined"].float().numpy() > 0.2}
        for space in ("Pd", "Po"):
            r = ref[space].float().numpy().astype(float)
            q = t[space].float().numpy().astype(float)
            r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
            sr, sq = r @ r.T, q @ r.T
            if (y > 0.5).any() and (y <= 0.5).any():
                predictions[space + "_nearest"] = sq[:, y > 0.5].max(1) > sq[:, y <= 0.5].max(1)
            else:
                predictions[space + "_nearest"] = np.zeros(len(q), dtype=bool)
            risks, values = [], []
            for name, k, reg in (("linear_01", 0, .01), ("linear_1", 0, .1),
                                 ("rbf10_01", 10, .01), ("rbf30_01", 30, .01)):
                kr = sr if k == 0 else np.exp(k * (sr - 1))
                kq = sq if k == 0 else np.exp(k * (sq - 1))
                pred, risk = regression(kr, kq, y, w, reg)
                predictions[space + "_" + name] = pred > 0.5
                risks.append(risk)
                values.append(pred)
            predictions[space + "_source_loo"] = values[int(np.argmin(risks))] > 0.5
        # Query GT is accessed only for evaluation after fitting is complete.
        area = t["area"].numpy().astype(float)
        truth = t["pur"].float().numpy().astype(float)
        mass = area * truth
        counts = {}
        for name, mask in predictions.items():
            inter = float(mask @ mass)
            union = float(mask @ area + mass.sum() - inter)
            counts[name] = [inter, union]
            value = totals.setdefault(name, {}).setdefault(int(t["c"]), [0., 0.])
            value[0] += inter
            value[1] += union
        rows.append({"fold": int(t["fold"]), "episode": int(t["e"]), "class": int(t["c"]), "counts": counts})
    summary = {name: 100 * float(np.mean([i / max(u, 1e-9) for i, u in per_class.values()]))
               for name, per_class in totals.items()}
    result = {"status": "cache availability subset diagnostic", "n": len(rows),
              "summary": summary, "per_class_counts": totals, "rows": rows, "gpu_used": False}
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "per_class_counts")}, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
