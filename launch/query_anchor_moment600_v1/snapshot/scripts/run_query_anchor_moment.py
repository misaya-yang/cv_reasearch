#!/usr/bin/env python3
"""Fixed query-anchor moment family; CPU masks seal before native CUDA CRF.

Only q/r and reference cov enter candidate inference. Original packet pre is
copied solely for native finalizer identity. No FoRIS score, query annotation or
query-truth packet member is read; no encoder or native Part3/Part4 is called.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import textwrap
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.experiment import load_rows, packet, sha, unpack
import run_part2_evidence as shared


def get_binarizer():
    if "binarizer" in shared.WORKER:
        return shared.WORKER["binarizer"]
    config = shared.WORKER["config"]
    if shared.native_sources(config["foris_root"]) != config["native_source"]:
        raise ValueError("Native source changed after snapshot")
    sys.path.insert(0, config["foris_root"])
    import models.foris as native
    if Path(native.__file__).resolve() != (Path(config["foris_root"]) / "models/foris.py").resolve():
        raise ValueError("Wrong models.foris import")
    function = native.FoRIS._binarize_response
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if any(isinstance(node, ast.Name) and node.id == "self"
           for statement in tree.body[0].body for node in ast.walk(statement)):
        raise ValueError("Native binarizer now requires a host")
    shared.WORKER["binarizer"] = function
    return function


def process_memory():
    values = dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    path = Path("/proc/self/smaps_rollup")
    if path.exists():
        values.update({line.split(":", 1)[0]: int(line.split()[1]) * 1024
                       for line in path.read_text().splitlines()
                       if line.startswith(("Pss:", "Private_Clean:", "Private_Dirty:"))})
    return values


def infer_episode(row):
    import torch
    from ics.methods.query_anchor_moment import ARMS, predict
    config, key = shared.WORKER["config"], row["key"]
    out, root = Path(config["out"]), Path(config["root"])
    started = time.monotonic()
    feature = root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt")
    target = Path(config["data_root"]) / row["query"]
    cached_packet = packet(root, row)
    inputs = dict(feature_sha256=sha(feature), packet_sha256=sha(cached_packet), query_rgb_sha256=sha(target))
    cache = torch.load(feature, map_location="cpu", weights_only=True)
    q, r = cache["q"], cache["r"]
    if any(tuple(value.shape) != (4096, 1024) or not torch.isfinite(value).all() for value in (q, r)):
        raise ValueError("Require finite retained [4096,1024] q/r")
    # Mixed native debias decisions are legal post-Part1 outputs and are retained.
    cache_receipt = dict(q_dtype=str(q.dtype), r_dtype=str(r.dtype), debiased=bool(cache["debiased"]),
                         normalization="FP32 unit normalize q/r; no Part1 replay")
    with np.load(cached_packet, allow_pickle=False) as saved:
        cov, native_pre = saved["cov"].astype(np.float32), saved["pre"].copy()
    unpack(native_pre)  # Validate packed shape/dtype without reading native/truth.
    if cov.shape != (64, 64) or not np.isfinite(cov).all():
        raise ValueError("Require finite source foreground coverage [64,64]")
    with torch.inference_mode():
        fields, receipt = predict(q, r, cov)
        binarizer = get_binarizer()
        prefinal = {ARMS[0]: native_pre}
        for arm in ARMS[1:]:
            pre = binarizer(None, torch.from_numpy(np.ascontiguousarray(fields[arm])), target_hw=(1024, 1024))
            prefinal[arm] = np.packbits(pre.numpy())
    if prefinal[ARMS[0]].tobytes() != native_pre.tobytes():
        raise ValueError("Original native pre identity changed")
    fields["reference_cov"] = cov
    for folder, values in (("prefinal", prefinal), ("fields", fields)):
        shared.write_npz(out / folder / f"{key}.npz", values)
    return dict(key=key, inputs=inputs, cache=cache_receipt, method=receipt,
                elapsed_seconds=time.monotonic() - started, worker_pid=os.getpid(), memory=process_memory(),
                native_prefinal_identity=True,
                **{folder: sha(out / folder / f"{key}.npz") for folder in ("prefinal", "fields")})


def infer(args):
    from ics.methods import query_anchor_moment as method
    manifest = json.loads(args.manifest.read_text())
    rows = load_rows(args.manifest)
    if args.keys:
        keys = args.keys.split(",")
        lookup = {row["key"]: row for row in rows}
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError("Smoke keys must be unique manifest members")
        rows = [lookup[key] for key in keys]
    if not rows or len(rows) != args.expected:
        raise ValueError("Empty cohort or unexpected episode count")
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory")
    allowed = ("fold", "e", "c", "key", "support", "query", "batch", "role", "collection_fold",
               "feature_export", "packet_export")
    rows = [{name: row[name] for name in allowed if name in row} for row in rows]
    foris_root = str(Path(manifest["foris_root"]).resolve())
    sources = shared.native_sources(foris_root)
    project = Path(__file__).resolve().parents[1]
    source_files = dict(runner=Path(__file__).resolve(), method=Path(method.__file__).resolve(),
        covariance=project / "src/ics/methods/query_covariance.py",
        graph=project / "src/ics/methods/mean_graph.py", graph_utilities=project / "src/ics/methods/rcg.py",
        shared_runner=Path(shared.__file__).resolve(), experiment=project / "src/ics/experiment.py",
        native_basis=project / "src/ics/native_basis.py")
    hashes = {name: sha(path) for name, path in source_files.items()}
    config = dict(method.CONFIG, root=str(args.root.resolve()), out=str(args.out.resolve()),
        foris_root=foris_root, data_root=manifest["data_root"], annotation_root=manifest["annotation_root"],
        projection_basis=manifest["projection_basis"], projection_basis_sha256=sha(manifest["projection_basis"]),
        source_manifest_sha256=sha(args.manifest), native_source=sources, method_sources=hashes,
        arms=list(method.ARMS), primary=method.ARMS[1], native_arm=method.ARMS[0],
        primary_baseline=method.ARMS[0], official_baseline="native",
        cpu_phase="query_anchor_moment_unary_and_fixed_graph", native_host_class=True,
        gpu_phase="native_CUDA_CRF_only", decoder_mode="native",
        workers=args.workers, threads=args.threads, memory_budget_pss_bytes=4 * 1024**3,
        feature_space="retained native conditional Part1 q/r; FP16 cache unit normalized in FP32",
        parameter_provenance="fixed query moment estimator and existing LW/RCG graph constants; no tuning or labels",
        finalizer="source native minmax/binarizer and identical native CUDA CRF for each arm",
        native_control="Original packet pre copied bit for bit; native score/final mask never read in infer",
        approved_packet_members=["cov", "pre"], native_part3_or_part4_called=False)
    for folder in ("predictions", "prefinal", "fields", "native_source/models", "native_source/utils", "method_source"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for name in shared.NATIVE_FILES:
        shutil.copyfile(Path(foris_root) / name, args.out / "native_source" / name)
    if shared.native_sources(args.out / "native_source") != sources:
        raise ValueError("Native source changed during snapshot")
    for name, path in source_files.items():
        shutil.copyfile(path, args.out / "method_source" / f"{name}.py")
        if sha(args.out / "method_source" / f"{name}.py") != hashes[name]:
            raise ValueError("Method source changed during snapshot")
    shared.write_json(args.out / "native_source_digest.json", sources)
    shared.write_json(args.out / "manifest.json", rows)
    shared.write_json(args.out / "config.json", config)
    seal = dict(state="INFERRING", manifest_sha256=sha(args.out / "manifest.json"),
        config_sha256=sha(args.out / "config.json"), native_source_snapshot_sha256=sources["sha256"],
        predictions={}, prefinal={}, fields={}, inputs={}, query_labels_opened=False, arms=list(method.ARMS))
    shared.write_json(args.out / "sealed.json", seal)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    audit, started = {}, time.monotonic()
    try:
        with mp.get_context("spawn").Pool(args.workers, initializer=shared.start_worker, initargs=(config,)) as pool:
            for completed, receipt in enumerate(pool.imap_unordered(infer_episode, rows), 1):
                key = receipt["key"]
                for group in ("prefinal", "fields", "inputs"):
                    seal[group][key] = receipt[group]
                audit[key] = receipt
                print(json.dumps(dict(completed=completed, total=len(rows), key=key,
                    seconds=receipt["elapsed_seconds"], memory=receipt["memory"],
                    degenerate_moment_fallback=receipt["method"]["degenerate_moment_fallback"])), flush=True)
        if shared.native_sources(foris_root) != sources:
            raise ValueError("Native source changed during inference")
        if {name: sha(path) for name, path in source_files.items()} != hashes:
            raise ValueError("Frozen method source changed during inference")
        if sha(manifest["projection_basis"]) != config["projection_basis_sha256"]:
            raise ValueError("Native basis changed during inference")
        shared.write_json(args.out / "audit.json", dict(episodes=audit, elapsed_seconds=time.monotonic() - started,
            native_prefinal_identity_episodes=sum(v["native_prefinal_identity"] for v in audit.values()),
            degenerate_moment_fallback_episodes=sum(v["method"]["degenerate_moment_fallback"] for v in audit.values()),
            maximum_coverage_fallback_episodes=sum(v["method"]["maximum_coverage_reference_fallback"] for v in audit.values()),
            numerical_floor_episodes=sum(v["method"]["covariance"].get("numerical_floor_applied", False) for v in audit.values()),
            debiased_episodes=sum(v["cache"]["debiased"] for v in audit.values()),
            native_no_debias_episodes=sum(not v["cache"]["debiased"] for v in audit.values()),
            query_labels_opened=False, encoder_forwards=0, native_part3_or_part4_called=False,
            complete_method_result=False))
        seal.update(state="ALL_PREFINALS_SEALED", audit_sha256=sha(args.out / "audit.json"))
    except BaseException as error:
        seal.update(state="INFERENCE_FAILED", error=f"{type(error).__name__}: {error}")
        shared.write_json(args.out / "sealed.json", seal)
        raise
    shared.write_json(args.out / "sealed.json", seal)
    shared.write_json(args.out / "prefinal_sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("infer")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--workers", type=int, choices=(1, 2, 3, 4, 5, 6), default=6)
    replay.add_argument("--threads", type=int, choices=(1,), default=1)
    replay.add_argument("--keys")
    replay.add_argument("--expected", type=int, required=True)
    finalizer = sub.add_parser("finalize")
    finalizer.add_argument("--out", type=Path, required=True)
    finalizer.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    {"infer": infer, "finalize": shared.finalize}[args.command](args)


if __name__ == "__main__":
    main()
