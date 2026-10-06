#!/usr/bin/env python3
"""CPU closed-form query covariance inference, then shared native CUDA CRF.

The inference phase never opens query masks or truth/native packet members.
Every CPU mask and field is sealed before the separate CRF and scoring phases.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.experiment import load_rows, packet, sha, unpack
import run_part2_evidence as shared


def infer_episode(row):
    import torch
    from PIL import Image
    from ics.methods.query_covariance import ARMS, prefinal_arms
    config, key = shared.WORKER["config"], row["key"]
    out, root = Path(config["out"]), Path(config["root"])
    started = time.monotonic()
    feature = root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt")
    reference = Path(config["data_root"]) / row["support"]
    target = Path(config["data_root"]) / row["query"]
    annotation = Path(config["annotation_root"]) / Path(row["support"]).with_suffix(".png")
    cached_packet = packet(root, row)
    inputs = dict(feature_sha256=sha(feature), packet_sha256=sha(cached_packet),
                  reference_rgb_sha256=sha(reference), query_rgb_sha256=sha(target),
                  reference_mask_sha256=sha(annotation))
    cache = torch.load(feature, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or not all(name in cache for name in ("q", "r", "debiased")):
        raise ValueError("Require retained q/r/debiased post-Part1 cache")
    q, r = cache["q"], cache["r"]
    if any(tuple(value.shape) != (4096, 1024) or not torch.isfinite(value).all() for value in (q, r)):
        raise ValueError("Require finite retained [4096,1024] q/r tokens")
    # Source Part1 conditionally skips positional debiasing. Its retained output
    # is still the correct post-Part1 input; preserve and report that decision.
    cache_receipt = dict(q_dtype=str(q.dtype), r_dtype=str(r.dtype), debiased=bool(cache["debiased"]),
                         conversion="FP32 cast only; no token renormalization or encoder/Part1 replay")
    fmaps = torch.stack((r.float().T.reshape(1024, 64, 64),
                         q.float().T.reshape(1024, 64, 64)))[None].contiguous()
    host = shared.get_host()
    import models.foris as native
    # Only the reference annotation is decoded. Query annotations never enter this function.
    with Image.open(annotation) as image:
        mask = torch.from_numpy((np.asarray(image) == row["c"] + 1).copy())
    try:
        with Image.open(reference) as image:
            host.set_reference(image.convert("RGB"), mask)
        with Image.open(target) as image:
            host.set_target(image.convert("RGB"))
        with torch.no_grad():
            masks, values, receipts, timings = prefinal_arms(host, fmaps, native)
        prefinal = {arm: np.packbits(value.cpu().numpy()) for arm, value in masks.items()}
        fields = {name: value.cpu().numpy().astype(np.float32) for name, value in values.items()}
        # np.load is lazy; only score, s2 and pre are decoded. No s3/native/truth member is read.
        drift = {}
        with np.load(cached_packet, allow_pickle=False) as cached:
            for name in ("s2", "score"):
                difference = np.abs(fields[f"{ARMS[0]}.{name}"] - cached[name].astype(np.float32))
                drift[name] = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()))
            drift["pre"] = dict(changed_pixels=int((masks[ARMS[0]].cpu().numpy() != unpack(cached["pre"])).sum()))
        for folder, data in (("prefinal", prefinal), ("fields", fields)):
            shared.write_npz(out / folder / f"{key}.npz", data)
        return dict(key=key, inputs=inputs, cache=cache_receipt, drift=drift, arm_receipts=receipts,
                    timings=timings, elapsed_seconds=time.monotonic() - started,
                    worker_pid=os.getpid(), peak_worker_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    basis=shared.WORKER["basis"],
                    **{folder: sha(out / folder / f"{key}.npz") for folder in ("prefinal", "fields")})
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def infer(args):
    from ics.methods import query_covariance as method
    manifest = json.loads(args.manifest.read_text())
    rows = load_rows(args.manifest)
    if args.keys:
        keys = args.keys.split(",")
        lookup = {row["key"]: row for row in rows}
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError("Explicit smoke keys must be unique manifest members")
        rows = [lookup[key] for key in keys]
    if not rows or (args.expected is not None and len(rows) != args.expected):
        raise ValueError("Empty cohort or unexpected episode count")
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory; preserve existing artifacts")
    allowed = ("fold", "e", "c", "key", "support", "query", "batch", "role", "collection_fold",
               "feature_export", "packet_export")
    rows = [{name: row[name] for name in allowed if name in row} for row in rows]
    foris_root = str(Path(manifest["foris_root"]).resolve())
    sources = shared.native_sources(foris_root)
    project = Path(__file__).resolve().parents[1]
    source_files = dict(runner=Path(__file__).resolve(), method=Path(method.__file__).resolve(),
                        shared_runner=Path(shared.__file__).resolve(),
                        experiment=project / "src/ics/experiment.py", native_basis=project / "src/ics/native_basis.py")
    source_hashes = {name: sha(path) for name, path in source_files.items()}
    config = dict(method.CONFIG, root=str(args.root.resolve()), out=str(args.out.resolve()),
                  foris_root=foris_root, data_root=manifest["data_root"], annotation_root=manifest["annotation_root"],
                  projection_basis=manifest["projection_basis"], native_source=sources,
                  source_manifest_sha256=sha(args.manifest),
                  projection_basis_sha256=sha(manifest["projection_basis"]),
                  method_sources=source_hashes, workers=args.workers, threads=args.threads,
                  arms=list(method.ARMS), primary="query_covariance", native_arm=method.ARMS[0],
                  primary_baseline=method.ARMS[0], official_baseline="native",
                  cpu_phase="query_covariance_native_tail", native_host_class=True,
                  gpu_phase="native_CUDA_CRF_only", decoder_mode="native",
                  feature_space="retained normalized post-Part1 q/r; native conditional debias decision preserved per episode; FP16 may drift from original native",
                  parameter_provenance="closed-form query-only covariance; no tuned shrinkage, labels or fitting",
                  source_prior="unchanged raw Part3 geometry, original normalized foreground anchor and gated Part4 geometry",
                  finalizer="source FoRIS Part3/Part4/binarizer followed by identical native CUDA CRF",
                  approved_packet_members=["s2", "score", "pre"],
                  memory_budget_bytes=args.memory_budget_gib * 1024**3)
    for folder in ("predictions", "prefinal", "fields", "native_source/models", "native_source/utils", "method_source"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for name in shared.NATIVE_FILES:
        shutil.copyfile(Path(foris_root) / name, args.out / "native_source" / name)
    if shared.native_sources(args.out / "native_source") != sources:
        raise ValueError("Native source changed during snapshot")
    for name, path in source_files.items():
        shutil.copyfile(path, args.out / "method_source" / f"{name}.py")
        if sha(args.out / "method_source" / f"{name}.py") != source_hashes[name]:
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
                                      seconds=receipt["elapsed_seconds"], native_drift=receipt["drift"],
                                      covariance=receipt["arm_receipts"]["query_covariance"],
                                      peak_worker_rss_bytes=receipt["peak_worker_rss_bytes"])), flush=True)
        if shared.native_sources(foris_root) != sources:
            raise ValueError("Native source changed during inference")
        if {name: sha(path) for name, path in source_files.items()} != source_hashes:
            raise ValueError("Frozen method/runner source changed during inference")
        if sha(manifest["projection_basis"]) != config["projection_basis_sha256"]:
            raise ValueError("Frozen native positional basis changed")
        worst = {name: max(value["drift"][name]["max_abs"] for value in audit.values()) for name in ("s2", "score")}
        drift_pixels = sum(value["drift"]["pre"]["changed_pixels"] for value in audit.values())
        cov_receipts = [value["arm_receipts"]["query_covariance"] for value in audit.values()]
        summary = dict(episodes=audit, elapsed_seconds=time.monotonic() - started,
                       parity_max_abs=worst, native_pre_changed_pixels=drift_pixels,
                       covariance_fallback_episodes=sum(value["native_fallback"] for value in cov_receipts),
                       numerical_floor_episodes=sum(value.get("numerical_floor_applied", False) for value in cov_receipts),
                       debiased_episodes=sum(value["cache"]["debiased"] for value in audit.values()),
                       native_no_debias_episodes=sum(not value["cache"]["debiased"] for value in audit.values()),
                       query_labels_opened=False, encoder_forwards=0, complete_method_result=False,
                       interpretation="Compare paired same-cache native and isotropic full pipeline after shared CUDA CRF; original native remains separate")
        shared.write_json(args.out / "audit.json", summary)
        seal.update(state="ALL_PREFINALS_SEALED", audit_sha256=sha(args.out / "audit.json"))
    except BaseException as error:
        seal.update(state="INFERENCE_FAILED", error=f"{type(error).__name__}: {error}")
        shared.write_json(args.out / "sealed.json", seal)
        raise
    shared.write_json(args.out / "sealed.json", seal)
    shared.write_json(args.out / "prefinal_sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows), parity_max_abs=worst,
                         native_pre_changed_pixels=drift_pixels)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("infer")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--workers", type=int, choices=tuple(range(1, 9)), default=3)
    replay.add_argument("--threads", type=int, choices=(1,), default=1)
    replay.add_argument("--keys", help="Comma-separated manifest keys for four-fold execution smoke")
    replay.add_argument("--expected", type=int, required=True)
    replay.add_argument("--memory-budget-gib", type=int, choices=(3, 6), default=3,
                        help="Record the authorized job memory budget; not a per-process virtual-memory limit")
    finalizer = sub.add_parser("finalize", help="Shared native CUDA CRF; root owns GPU dispatch")
    finalizer.add_argument("--out", type=Path, required=True)
    finalizer.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    {"infer": infer, "finalize": shared.finalize}[args.command](args)


if __name__ == "__main__":
    main()
