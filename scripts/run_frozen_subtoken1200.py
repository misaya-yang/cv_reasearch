#!/usr/bin/env python3
"""Frozen DEV1200 subtoken readout with the existing lambda64 strong control.

Stages are separate: CPU prepare seals lambda16 replay/lambda64 fields and
provider masks; CUDA infer streams four query shifts and seals all1200 fine
masks; CPU score alone opens GT and checks prior native/RCG/MEAN I/U.
No refitting, parameter grid, CRF, new image/reference, or fine-feature archive.

python scripts/run_frozen_subtoken1200.py prepare \
    --manifest outputs/confirm1200_conditional_v1/inputs/manifest.json \
    --prepared outputs/subtoken1200_rcg64_prepared_v1
python scripts/run_frozen_subtoken1200.py infer \
    --prepared outputs/subtoken1200_rcg64_prepared_v1 --out outputs/frozen_subtoken1200_v1
python scripts/run_frozen_subtoken1200.py score \
    --out outputs/frozen_subtoken1200_v1 \
    --prior-episodes outputs/composed_full1200_dev_v1/episodes.jsonl
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import metric, photo_groups, render, sha, summarize, unpack

FINE16 = "fine.rcg16.control"
FINE64 = "fine.rcg64"
ARMS = ("native", "rcg", "mean.control", "rcg64.control", FINE16, FINE64)


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest(path):
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data["episodes"]
    if len(rows) != 1200 or len({r["key"] for r in rows}) != 1200:
        raise ValueError("Require all1200 original draw keys; never deduplicate identities or subset")
    for row in rows:
        for name in ("fold", "c", "e"):
            row[name] = int(row[name])
        if not all(Path(row[k]).is_absolute() for k in ("feature_export", "packet_export", "recheck_run")):
            raise ValueError("Provider manifest must contain absolute cache/packet/run paths")
    if any(sum(r["fold"] == f for r in rows) != 300 for f in range(4)):
        raise ValueError("Require original300/fold cohort")
    return rows


def parameters(path):
    report = json.loads(Path(path).read_text())
    selection = report["selection"]["feat.nested"]
    if report["n"] != 600 or set(selection) != {"0", "1", "2", "3"}:
        raise ValueError("Require recorded full600 fine-feature fold selection")
    expected = {0: (1.25, .07), 1: (1.25, .15), 2: (1.25, .15), 3: (1.25, .07)}
    result = {}
    for fold, arm in selection.items():
        match = re.fullmatch(r"feat\[s([0-9.]+),t([0-9.]+)\]", arm)
        if match is None or tuple(map(float, match.groups())) != expected[int(fold)]:
            raise ValueError("Recorded600 frozen parameters changed")
        result[fold] = dict(arm=arm, sigma=expected[int(fold)][0], tau=expected[int(fold)][1])
    return result


def verify_code(config):
    for path, digest in config["source_code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Changed immutable stage implementation: " + path)


def prepare_one(job):
    """CPU-only; share similarities/graph, then use two genuinely different matrices."""
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods.rcg import minmax, rank

    torch.set_num_threads(1)
    row, destination, expected_field, expected_mask = job
    begin = time.monotonic()
    key, source_key = row["key"], row.get("source_key", row["key"])
    run = Path(row["recheck_run"])
    field_path, mask_path = run / "fields" / (source_key + ".npz"), run / "predictions" / (source_key + ".npz")
    inputs = dict(feature_sha256=sha(row["feature_export"]), packet_sha256=sha(row["packet_export"]),
                  source_field_sha256=sha(field_path), source_predictions_sha256=sha(mask_path))
    if inputs["source_field_sha256"] != expected_field or inputs["source_predictions_sha256"] != expected_mask:
        raise ValueError("Changed provider field/masks: " + key)
    cached = torch.load(row["feature_export"], map_location="cpu", weights_only=True)
    q, r = (cached[k].float() for k in ("q", "r"))
    if any(cached[k].dtype != torch.float16 for k in ("q", "r")) or q.shape != (4096, 1024) or r.shape != q.shape:
        raise ValueError("Require the retained4096x1024 FP16 q/r cache")
    with np.load(row["packet_export"], allow_pickle=False) as z:
        # Supplied-reference coverage, source score, and prediction only. No truth.
        cov, score, native = z["cov"].copy(), z["score"].copy(), z["native"].copy()
    with np.load(field_path, allow_pickle=False) as z:
        sealed_field = z["rcg"].copy()
    with np.load(mask_path, allow_pickle=False) as z:
        original_rcg, mean = z["RCG"].copy(), z["MEAN_CONTROL"].copy()
    if cov.shape != (64, 64) or score.shape != cov.shape or sealed_field.shape != cov.shape:
        raise ValueError("Invalid aligned source grids")
    if sealed_field.dtype != np.float32 or not all(np.isfinite(v).all() for v in (cov, score, sealed_field)):
        raise ValueError("Invalid sealed/source float fields")
    if cov.min() < 0 or cov.max() > 1 or not bool(torch.isfinite(q).all() and torch.isfinite(r).all()):
        raise ValueError("Invalid q/r or reference coverage")
    if bool((q.norm(dim=1) == 0).any() or (r.norm(dim=1) == 0).any()):
        raise ValueError("Zero-norm cached token")
    for packed in (native, original_rcg, mean):
        unpack(packed)
    q, r = F.normalize(q, dim=1), F.normalize(r, dim=1)
    pure = np.flatnonzero(cov.ravel() >= .9)
    if not len(pure):
        pure = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T
    dq, dr = sim.topk(10, dim=1).values.mean(1), sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, pure] - dr[pure][None, :]).max(1).values - dq).numpy()
    del sim, dq, dr, r
    s = minmax(score).ravel()
    y = (s + .5 * (rank(guide) - rank(s))).astype(np.float64)
    sim = q @ q.T
    sim.fill_diagonal_(-2)
    values, ids = sim.topk(20, dim=1)
    del sim, q
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), ids.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    laplacian = sparse.diags(degree) - w
    confidence = .1 + np.abs(2 * s - 1)
    confidence = (confidence / confidence.mean()).astype(np.float64)
    anchor = sparse.diags(confidence)
    rhs = confidence * y
    fields, audits = {}, {}
    for lam in (16, 64):
        matrix = anchor + lam * laplacian
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        solved, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
        residual = float(np.linalg.norm(matrix @ solved - rhs))
        tolerance = max(1e-9, 1e-7 * float(np.linalg.norm(rhs)))
        if status != 0 or not np.isfinite(solved).all() or residual > 1.01 * tolerance:
            raise RuntimeError("Fixed lambda%d CG convergence failed: %s" % (lam, key))
        field = solved.reshape(64, 64).astype(np.float32)
        fields["rcg%d" % lam] = field
        audits[str(lam)] = dict(graph_lambda=lam, cg_status=int(status), cg_iterations=iterations[0],
                               absolute_residual=residual, residual_tolerance=tolerance,
                               weighted_mean_error=float(abs(np.dot(confidence, solved - y))))
        if lam == 16:
            replay = render(field)
            mismatch_field = int(np.count_nonzero(field != sealed_field))
            mismatch_mask = int(np.count_nonzero(replay != unpack(original_rcg)))
            audits["16"].update(fp32_field_mismatched_entries=mismatch_field,
                                  stored_rcg_mask_mismatched_pixels=mismatch_mask,
                                  maximum_field_difference=float(np.abs(field - sealed_field).max()))
            if mismatch_field or mismatch_mask:
                raise RuntimeError("Exact lambda16 field/mask replay failed before lambda64: " + key)
    masks = {"native": native, "rcg": original_rcg, "mean.control": mean,
             "rcg64.control": np.packbits(render(fields["rcg64"]))}
    out = Path(destination)
    fp, pp = out / "fields" / (key + ".npz"), out / "predictions" / (key + ".npz")
    np.savez_compressed(fp, **fields)
    np.savez_compressed(pp, **masks)
    inputs["debiased"] = bool(cached["debiased"])
    return key, sha(fp), sha(pp), dict(inputs=inputs, solver=audits,
                                      source_pure_tokens=int(len(pure)), seconds=time.monotonic() - begin)


def prepare(a):
    rows = manifest(a.manifest)
    frozen = parameters(a.parameter_report)
    if a.prepared.exists():
        raise FileExistsError("Use a fresh prepared directory; never remove retained inputs")
    providers, jobs = {}, []
    for row in rows:
        run = Path(row["recheck_run"])
        if str(run) not in providers:
            seal = json.loads((run / "sealed.json").read_text())
            if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(run / "manifest.json") != seal["manifest_sha256"]:
                raise ValueError("Provider manifest is not sealed")
            data = json.loads((run / "manifest.json").read_text())
            data = data if isinstance(data, list) else data["episodes"]
            providers[str(run)] = dict(seal=seal, rows={r["key"]: r for r in data},
                                       seal_sha256=sha(run / "sealed.json"))
        source = providers[str(run)]
        key = row.get("source_key", row["key"])
        original = source["rows"][key]
        if int(original["c"]) != row["c"] or int(original["fold"]) != row["fold"] or any(
                Path(original[k]).name != Path(row[k]).name for k in ("support", "query")):
            raise ValueError("Provider photo/class/fold identity changed: " + row["key"])
        jobs.append((row, str(a.prepared), source["seal"]["fields"][key], source["seal"]["predictions"][key]))
    a.prepared.mkdir(parents=True)
    (a.prepared / "fields").mkdir()
    (a.prepared / "predictions").mkdir()
    write(a.prepared / "manifest.json", rows)
    code = [Path(__file__).resolve(), REPO / "src/ics/experiment.py", REPO / "src/ics/methods/rcg.py"]
    config = dict(stage="CPU_ONLY_FIXED_RCG64_PREPARATION", n=1200,
                  exposure="all1200 previously exposed DEV draws; no independent confirmation or SOTA claim",
                  source_manifest=str(a.manifest.resolve()), source_manifest_sha256=sha(a.manifest),
                  parameter_report=str(a.parameter_report.resolve()), parameter_report_sha256=sha(a.parameter_report),
                  parameters=frozen, source_code_sha256={str(p): sha(p) for p in code},
                  providers={name: dict(seal_sha256=v["seal_sha256"]) for name, v in providers.items()},
                  graph_lambdas=[16, 64], alpha=.5, cross_k=10, query_k=20, purity=.9,
                  confidence_floor=.1, cg_rtol=1e-7, cg_atol=1e-9, cg_maxiter=300,
                  lambda16_parity="exact FP32 source field and complete stored mask before lambda64 solve",
                  workers=a.workers, query_gt_in_inference=False, encoder_forwards=0, images_opened=0,
                  legacy_provider_inputs="not assumed; fresh packet/cache hashes captured and prior I/U checked at score")
    write(a.prepared / "config.json", config)
    begin, records = time.monotonic(), {}
    write(a.prepared / "state.json", dict(state="RUNNING_CPU_PREPARATION", pid=os.getpid(), completed=0, n=1200))
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_one, job) for job in jobs]
        for future in as_completed(futures):
            key, field_hash, prediction_hash, audit = future.result()
            records[key] = dict(fields_sha256=field_hash, predictions_sha256=prediction_hash, **audit)
            if len(records) % 25 == 0:
                state = dict(state="RUNNING_CPU_PREPARATION", pid=os.getpid(), completed=len(records), n=1200,
                             seconds=time.monotonic() - begin)
                write(a.prepared / "state.json", state)
                print(json.dumps(state), flush=True)
    for name, info in providers.items():
        if sha(Path(name) / "sealed.json") != info["seal_sha256"]:
            raise ValueError("Provider seal changed during CPU preparation")
    write(a.prepared / "audits.json", records)
    seal = dict(state="ALL1200_RCG64_FIELDS_AND_PROVIDER_MASKS_SEALED", n=1200,
                manifest_sha256=sha(a.prepared / "manifest.json"), config_sha256=sha(a.prepared / "config.json"),
                audits_sha256=sha(a.prepared / "audits.json"),
                fields={k: v["fields_sha256"] for k, v in records.items()},
                predictions={k: v["predictions_sha256"] for k, v in records.items()},
                inputs={k: v["inputs"] for k, v in records.items()},
                seconds=time.monotonic() - begin, query_truth_opened=False)
    write(a.prepared / "sealed.json", seal)
    write(a.prepared / "state.json", dict(state=seal["state"], pid=os.getpid(), completed=1200, n=1200,
                                         seconds=seal["seconds"]))
    print("SUBTOKEN1200_CPU_PREPARATION_SEALED", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "infer", "score"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/confirm1200_conditional_v1/inputs/manifest.json"))
    p.add_argument("--parameter-report", type=Path, default=Path("outputs/claude_subtoken_fresh600/report.json"))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--prior-episodes", type=Path)
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--readers", type=int, default=6)
    p.add_argument("--writers", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=12)
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args()
    if any(v < 1 for v in (a.workers, a.readers, a.writers, a.prefetch, a.chunk)):
        p.error("execution sizes must be positive")
    if a.stage in ("prepare", "infer") and a.prepared is None:
        p.error("prepare/infer require --prepared")
    if a.stage in ("infer", "score") and a.out is None:
        p.error("infer/score require --out")
    if a.stage == "score" and a.prior_episodes is None:
        p.error("score requires the existing --prior-episodes baseline I/U source")
    if a.stage == "prepare" and a.prepared.exists():
        p.error("prepare requires a fresh directory")
    if a.stage == "infer" and a.out.exists():
        p.error("infer requires a fresh directory")
    try:
        if a.stage == "prepare":
            prepare(a)
        else:
            raise RuntimeError("GPU infer/CPU score stage is being prepared; CPU prepare is ready")
    except Exception as error:
        destination = a.prepared if a.stage == "prepare" else a.out
        if destination is not None and (destination / "config.json").exists():
            write(destination / ("score_state.json" if a.stage == "score" else "state.json"),
                  dict(state="FAILED", stage=a.stage, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
