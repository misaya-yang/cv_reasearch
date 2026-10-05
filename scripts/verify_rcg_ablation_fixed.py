#!/usr/bin/env python3
"""Verify only the three fixed RCG ablations on the retained historical600 cache.

Inference indexes only q/r/cov/score and previously sealed RCG outputs. All
predictions are sealed before evaluation indexes query truth/pre/native. Dense
cross/query similarities and the reciprocal graph are shared across the arms.

python scripts/verify_rcg_ablation_fixed.py all --root outputs/fresh600_root \
  --run outputs/recheck_fresh600_v1 --out outputs/rcg_ablation_verified600_v1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = {"rerank_only": (0.5, 0.0), "smooth_only": (0.0, 16.0), "rcg": (0.5, 16.0)}
CONFIG = dict(
    arms={k: dict(alpha=a, graph_lambda=l) for k, (a, l) in ARMS.items()},
    cross_image_k=10, query_k=20, source_purity=0.9, confidence_floor=0.1,
    cg_rtol=1e-7, cg_atol=1e-9, cg_maxiter=300,
    finalizer="FP64 solve to FP32, bilinear1024 align_corners=False, >0.5; no re-minmax or CRF",
    query_gt_in_inference=False, exposure="historical training-pool isolated600; not never-seen confirmation",
    required_parity="returned FP32 RCG field and rendered/stored RCG masks exactly equal",
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def one(job):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.experiment import load_inputs, render, sha, unpack
    from ics.methods.rcg import minmax, rank

    root, run, out, row, expected_field_sha, expected_mask_sha = job
    torch.set_num_threads(1)
    begin = time.perf_counter()
    (q, r, cov, score), inputs = load_inputs(root, row)
    q, r = torch.as_tensor(q, device="cpu").float(), torch.as_tensor(r, device="cpu").float()
    cov, score = np.asarray(cov), np.asarray(score)
    if q.shape != (4096, 1024) or r.shape != q.shape or cov.shape != (64, 64) or score.shape != cov.shape:
        raise ValueError("Require the original aligned 4096x1024 cache and 64x64 fields")
    if not bool(torch.isfinite(q).all() and torch.isfinite(r).all()) or not np.isfinite(cov).all() or not np.isfinite(score).all():
        raise ValueError("Nonfinite inference input")
    if cov.min() < 0 or cov.max() > 1 or bool((q.norm(dim=1) == 0).any() or (r.norm(dim=1) == 0).any()):
        raise ValueError("Invalid coverage or zero-norm tokens")
    q, r = F.normalize(q, dim=1), F.normalize(r, dim=1)
    fi = np.flatnonzero(cov.ravel() >= 0.9)
    if not len(fi):
        fi = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T
    dq, dr = sim.topk(10, dim=1).values.mean(1), sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, fi] - dr[fi][None, :]).max(1).values - dq).numpy()
    del sim, dq, dr, r
    s = minmax(score).ravel()
    correction = rank(guide) - rank(s)
    sim = q @ q.T
    sim.fill_diagonal_(-2)
    values, indices = sim.topk(20, dim=1)
    del sim, q
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), indices.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    a = 0.1 + np.abs(2 * s - 1)
    a = (a / a.mean()).astype(np.float64)
    matrix = sparse.diags(a) + 16 * (sparse.diags(degree) - w)

    key, run, out = row["key"], Path(run), Path(out)
    source_field_path, source_mask_path = run / "fields" / (key + ".npz"), run / "predictions" / (key + ".npz")
    source_field_sha, source_mask_sha = sha(source_field_path), sha(source_mask_path)
    if source_field_sha != expected_field_sha or source_mask_sha != expected_mask_sha:
        raise ValueError("Source RCG output changed: " + key)
    with np.load(source_field_path, allow_pickle=False) as z:
        sealed_field = z["rcg"].copy()
    with np.load(source_mask_path, allow_pickle=False) as z:
        sealed_mask = unpack(z["RCG"])
    if sealed_field.shape != (64, 64) or sealed_field.dtype != np.float32 or not np.isfinite(sealed_field).all():
        raise ValueError("Expected finite sealed FP32 RCG field")
    fields, masks, audits = {}, {}, {}
    for name, (alpha, lam) in ARMS.items():
        y = (s + alpha * correction).astype(np.float64)
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        rhs = a * y
        if lam:
            solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
            residual = float(np.linalg.norm(matrix @ solved - rhs))
        else:
            solved, status, residual = y, 0, 0.0
        finite = bool(np.isfinite(solved).all())
        if not finite:
            raise RuntimeError("Nonfinite fixed-arm solve: " + key + "/" + name)
        field = solved.reshape(64, 64).astype(np.float32)
        mask = render(field)
        fields[name], masks[name] = field, np.packbits(mask)
        norm_rhs = float(np.linalg.norm(rhs))
        audits[name] = dict(cg_status=int(status), cg_iterations=iterations[0], absolute_residual=residual,
                            relative_residual=residual / max(norm_rhs, 1e-12),
                            residual_tolerance=max(1e-9, 1e-7 * norm_rhs), direct_solve=not bool(lam),
                            fp64_to_fp32_maxdiff=float(np.abs(solved.reshape(64, 64) - field).max()),
                            weighted_mean_error=float(abs(np.dot(a, solved - y))),
                            maximum_principle_error=float(max(0, solved.max() - y.max(), y.min() - solved.min())))
        if name == "rcg":
            audits[name].update(
                fp64_vs_sealed_field_maxdiff=float(np.abs(solved.reshape(64, 64) - sealed_field).max()),
                fp32_vs_sealed_field_maxdiff=float(np.abs(field - sealed_field).max()),
                fp32_field_mismatched_entries=int(np.count_nonzero(field != sealed_field)),
                rendered_vs_sealed_field_mask_mismatched_pixels=int(np.count_nonzero(mask != render(sealed_field))),
                rendered_vs_stored_rcg_mask_mismatched_pixels=int(np.count_nonzero(mask != sealed_mask)),
                sealed_field_vs_stored_rcg_mask_mismatched_pixels=int(np.count_nonzero(render(sealed_field) != sealed_mask)),
            )
    np.savez_compressed(out / "fields" / (key + ".npz"), **fields)
    np.savez_compressed(out / "predictions" / (key + ".npz"), **masks)
    return key, dict(inputs=inputs, source_field_sha256=source_field_sha, source_predictions_sha256=source_mask_sha,
                     source_pure_tokens=int(len(fi)), graph_undirected_edges=int(w.nnz // 2), arms=audits,
                     elapsed_seconds=time.perf_counter() - begin)


def infer(args):
    import multiprocessing as mp
    import numpy as np
    import scipy
    import torch
    from ics.experiment import load_rows, sha

    if args.out.exists():
        raise FileExistsError("Use a fresh verification output directory")
    source_seal = json.loads((args.run / "sealed.json").read_text())
    if source_seal.get("state") != "ALL_PREDICTIONS_SEALED" or sha(args.run / "manifest.json") != source_seal["manifest_sha256"]:
        raise ValueError("Unsealed source manifest")
    rows = load_rows(args.run / "manifest.json")
    keys = {r["key"] for r in rows}
    if len(rows) != 600 or keys != set(source_seal["predictions"]) or keys != set(source_seal["fields"]):
        raise ValueError("The paired full historical600 source is required")
    args.out.mkdir(parents=True)
    (args.out / "fields").mkdir()
    (args.out / "predictions").mkdir()
    (args.out / "manifest.json").write_bytes((args.run / "manifest.json").read_bytes())
    write_json(args.out / "config.json", CONFIG)
    begin, audit = time.perf_counter(), {}
    jobs = [(str(args.root), str(args.run), str(args.out), r, source_seal["fields"][r["key"]], source_seal["predictions"][r["key"]]) for r in rows]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, (key, info) in enumerate(pool.imap_unordered(one, jobs, chunksize=1), 1):
            audit[key] = info
            if n % 50 == 0 or n == len(rows):
                print(json.dumps(dict(inferred=n, total=len(rows), elapsed_seconds=time.perf_counter() - begin)), flush=True)
    write_json(args.out / "audits.json", audit)
    failures = []
    for key, info in audit.items():
        for arm, d in info["arms"].items():
            if d["cg_status"] != 0 or d["absolute_residual"] > d["residual_tolerance"] * 1.01:
                failures.append(dict(key=key, arm=arm, reason="CG convergence/residual", audit=d))
        d = info["arms"]["rcg"]
        if any(d[k] for k in ("fp32_field_mismatched_entries", "rendered_vs_sealed_field_mask_mismatched_pixels", "rendered_vs_stored_rcg_mask_mismatched_pixels", "sealed_field_vs_stored_rcg_mask_mismatched_pixels")):
            failures.append(dict(key=key, arm="rcg", reason="Exact field/mask parity", audit=d))
    files = [Path(__file__), Path(__file__).resolve().parents[1] / "src/ics/methods/rcg.py", Path(__file__).resolve().parents[1] / "src/ics/experiment.py"]
    receipt = dict(source_root=str(args.root.resolve()), source_run=str(args.run.resolve()),
                   source_seal_sha256=sha(args.run / "sealed.json"), manifest_sha256=sha(args.out / "manifest.json"),
                   source_code_sha256={str(p): sha(p) for p in files}, source_astra_hashes=source_seal.get("astra_sources", {}),
                   versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
                   workers=args.workers, inference_seconds=time.perf_counter() - begin, failures=failures)
    write_json(args.out / "receipt.json", receipt)
    if failures:
        raise RuntimeError(f"Fixed verification failed on {len(failures)} checks; audit retained, no scoring")
    seal = dict(state="ALL_PREDICTIONS_SEALED", arms=list(ARMS), manifest_sha256=sha(args.out / "manifest.json"),
                config_sha256=sha(args.out / "config.json"), audits_sha256=sha(args.out / "audits.json"),
                receipt_sha256=sha(args.out / "receipt.json"), inputs={k: v["inputs"] for k, v in audit.items()},
                predictions={k: sha(args.out / "predictions" / (k + ".npz")) for k in keys},
                fields={k: sha(args.out / "fields" / (k + ".npz")) for k in keys}, query_gt_in_inference=False)
    write_json(args.out / "sealed.json", seal)
    print("ALL_PREDICTIONS_SEALED", flush=True)


def evaluate(args):
    import numpy as np
    from ics.experiment import load_rows, metric, packet, photo_groups, render, sha, summarize, unpack
    from ics.methods.rcg import minmax

    seal = json.loads((args.out / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("Unsealed predictions")
    for filename, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256"), ("audits.json", "audits_sha256"), ("receipt.json", "receipt_sha256")):
        if sha(args.out / filename) != seal[key]:
            raise ValueError("Changed verification artifact: " + filename)
    rows = load_rows(args.out / "manifest.json")
    arrays = {k: [] for k in ("native", "foris_pre.control", *ARMS)}
    details, corrections = [], {k: [] for k in arrays}
    for row in rows:
        key = row["key"]
        for directory in ("predictions", "fields"):
            if sha(args.out / directory / (key + ".npz")) != seal[directory][key]:
                raise ValueError("Changed fixed output: " + key)
        pp = packet(args.root, row)
        if sha(pp) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Packet changed after inference: " + key)
        with np.load(pp, allow_pickle=False) as z:
            truth, native, pre = (unpack(z[k]) for k in ("truth", "native", "pre"))
            score = z["score"].copy()
        if np.any(render(minmax(score)) != pre):
            raise ValueError("FoRIS pre renderer mismatch: " + key)
        with np.load(args.out / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            masks = {k: unpack(z[k]) for k in ARMS}
        masks.update(native=native, **{"foris_pre.control": pre})
        record = dict(row, iu={}, edits_vs_pre={})
        for arm, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            arrays[arm].append(iu)
            record["iu"][arm] = iu
            add, delete = mask & ~pre, pre & ~mask
            record["edits_vs_pre"][arm] = dict(add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                                               delete_FP=int((delete & ~truth).sum()), delete_TP=int((delete & truth).sum()))
            add, delete = mask & ~native, native & ~mask
            corrections[arm].append(dict(key=key, c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")),
                                         add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                                         delete_FP=int((delete & ~truth).sum()), delete_TP=int((delete & truth).sum())))
        details.append(record)
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in arrays.items()}
    result, draws = summarize(rows, arrays, corrections)
    groups, classes = photo_groups(rows), np.array([r["c"] for r in rows])
    weights = np.stack([np.bincount(d, minlength=int(groups.max()) + 1) for d in draws])[:, groups]
    samples = {k: np.array([metric(arrays[k], classes, w) for w in weights]) for k in ("foris_pre.control", *ARMS)}
    v = result["scores"]
    result["fixed_stage_contrasts"] = {}
    for name, arm, base in (("rerank_without_graph", "rerank_only", "foris_pre.control"),
                            ("graph_without_rerank", "smooth_only", "foris_pre.control"),
                            ("rerank_with_graph", "rcg", "smooth_only"),
                            ("graph_with_rerank", "rcg", "rerank_only")):
        result["fixed_stage_contrasts"][name] = dict(arm=arm, base=base, gain=v[arm] - v[base],
                                                       ci95=np.percentile(samples[arm] - samples[base], [2.5, 97.5]).tolist())
    interaction = samples["rcg"] - samples["rerank_only"] - samples["smooth_only"] + samples["foris_pre.control"]
    result["factorial_interaction"] = dict(definition="both - rerank_only - smooth_only + pre",
                                           gain=v["rcg"] - v["rerank_only"] - v["smooth_only"] + v["foris_pre.control"],
                                           ci95=np.percentile(interaction, [2.5, 97.5]).tolist())
    result.update(exposure=CONFIG["exposure"], config=CONFIG, prediction_seal_sha256=sha(args.out / "sealed.json"),
                  pre_mismatched_pixels=0, solver_and_parity_audits_sha256=sha(args.out / "audits.json"))
    np.savez_compressed(args.out / "IU.npz", **arrays)
    np.save(args.out / "bootstrap_photo_draws.npy", draws)
    (args.out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    write_json(args.out / "report.json", result)
    print(json.dumps(dict(n=result["n"], scores=v, fixed_stage_contrasts=result["fixed_stage_contrasts"],
                          factorial_interaction=result["factorial_interaction"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("infer", "evaluate", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four bounded CPU workers")
    if args.stage in ("infer", "all"):
        infer(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
