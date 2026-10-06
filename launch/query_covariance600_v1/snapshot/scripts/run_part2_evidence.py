#!/usr/bin/env python3
"""CPU Part2 cache replay followed by a separate native CUDA CRF finalizer.

infer never opens query labels. Native cached packets are opened lazily for only
s2/s3/score/pre drift; cached native is a separate official comparator.
finalize applies only the original CUDA CRF to sealed pre masks and query RGB.
Separate scoring is permitted only after ALL_PREDICTIONS_SEALED.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.experiment import load_rows, packet, sha, unpack

WORKER = {}
NATIVE_FILES = ("models/foris.py", "utils/clustering.py", "utils/data.py", "utils/refinement.py")


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_npz(path, values):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def native_sources(root):
    files = {name: sha(Path(root) / name) for name in NATIVE_FILES}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=files, sha256=digest)


def start_worker(config):
    import torch
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    WORKER.update(config=config, host=None)


def get_host(device="cpu"):
    import torch
    from ics.native_basis import reuse_native_basis
    config = WORKER["config"]
    if WORKER["host"] is not None:
        return WORKER["host"]
    if native_sources(config["foris_root"]) != config["native_source"]:
        raise RuntimeError("Native source changed after snapshot")
    sys.path.insert(0, config["foris_root"])
    import models.foris as native
    if Path(native.__file__).resolve() != (Path(config["foris_root"]) / "models/foris.py").resolve():
        raise RuntimeError("Imported the wrong models.foris module")
    for name, relative in (("agglomerative_clustering", "utils/clustering.py"),
                           ("compute_cluster_prototypes", "utils/clustering.py"),
                           ("downsample_mask", "utils/data.py"), ("crf_refine", "utils/refinement.py")):
        if Path(inspect.getsourcefile(getattr(native, name))).resolve() != (Path(config["foris_root"]) / relative).resolve():
            raise RuntimeError(f"Imported the wrong native utility: {name}")
    if config.get("cpu_phase") == "cached_native_tail" or config.get("native_host_class"):
        cls = native.FoRIS
    else:
        from ics.methods.part2_evidence import intervention_class
        cls = intervention_class(native.FoRIS, native)
    try:
        with reuse_native_basis(cls, config["projection_basis"]) as basis:
            host = cls(encoder=torch.nn.Identity(), image_size=1024, svd_components=500,
                       tau=0.6, mask_refiner="crf", resize_to_orig_size=False, device=device,
                       cluster_logsumexp_temp=0.07, dino_bg_weight=0.55).eval().requires_grad_(False)
    except Exception as error:
        raise RuntimeError(f"FoRIS/CRF construction on {device} failed; encoder and no-CRF fallbacks are forbidden") from error
    # Prevent even an accidental future encoder call; constructor used the frozen basis.
    def forbidden(*args, **kwargs):
        raise RuntimeError("Encoder invocation is forbidden in cache replay")
    host.encoder.forward = forbidden
    host._extract_features = forbidden
    WORKER.update(host=host, basis=basis)
    return host


def infer_episode(row):
    import torch
    from PIL import Image
    from ics.methods.part2_evidence import ARMS, prefinal_tail
    config, key = WORKER["config"], row["key"]
    out, root = Path(config["out"]), Path(config["root"])
    start = time.monotonic()
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
        raise ValueError("Expected retained q/r/debiased post-Part1 cache")
    q, r = cache["q"], cache["r"]
    if any(tuple(x.shape) != (4096, 1024) or not torch.isfinite(x).all() for x in (q, r)):
        raise ValueError("Expected finite cached [4096,1024] q/r tokens")
    cache_receipt = dict(q_dtype=str(q.dtype), r_dtype=str(r.dtype), debiased=bool(cache["debiased"]),
                         conversion="FP32 cast only; no renormalization and no Part1 replay")
    fmaps = torch.stack([r.float().T.reshape(1024, 64, 64), q.float().T.reshape(1024, 64, 64)])[None].contiguous()
    host = get_host()
    # Only the reference annotation is opened, with the episode's labeled class.
    with Image.open(annotation) as image:
        mask = torch.from_numpy((np.asarray(image) == row["c"] + 1).copy())
    try:
        with Image.open(reference) as image:
            host.set_reference(image.convert("RGB"), mask)
        with Image.open(target) as image:
            host.set_target(image.convert("RGB"))
        prefinal, fields, arm_receipts, timings, drift = {}, {}, {}, {}, {}
        for arm, aggregation in zip(ARMS, ("native", "lse", "max")):
            begin = time.monotonic()
            pre, stages, receipt, stage_times = prefinal_tail(host, fmaps, aggregation)
            prefinal[arm] = np.packbits(pre.cpu().numpy())
            for name, value in stages.items():
                fields[f"{arm}.{name}"] = value.cpu().numpy().astype(np.float32)
            timings[arm] = dict(total_seconds=time.monotonic() - begin, stages_seconds=stage_times)
            arm_receipts[arm] = receipt
            if aggregation == "native":
                # np.load is lazy. No truth member is accessed during inference.
                with np.load(cached_packet, allow_pickle=False) as cached:
                    for name in ("s2", "s3", "score"):
                        difference = np.abs(fields[f"{arm}.{name}"] - cached[name].astype(np.float32))
                        drift[name] = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()))
                    drift["pre"] = dict(changed_pixels=int((pre.cpu().numpy() != unpack(cached["pre"])).sum()))
        for arm in ARMS[1:]:
            for name in ("sf", "mu_fg", "cand_soft", "seed_prior", "gated_target_norm"):
                if not np.array_equal(fields[f"{ARMS[0]}.{name}"], fields[f"{arm}.{name}"]):
                    raise RuntimeError(f"Intervention changed a retained native auxiliary: {arm}.{name}")
        if arm_receipts[ARMS[1]]["bg_prototypes_sha256"] != arm_receipts[ARMS[2]]["bg_prototypes_sha256"]:
            raise RuntimeError("LSE and max controls did not use identical BG prototypes")
        for folder, values in (("prefinal", prefinal), ("fields", fields)):
            write_npz(out / folder / f"{key}.npz", values)
        return dict(key=key, inputs=inputs, cache=cache_receipt, drift=drift, arm_receipts=arm_receipts,
                    timings=timings, elapsed_seconds=time.monotonic() - start,
                    basis=WORKER["basis"], **{folder: sha(out / folder / f"{key}.npz")
                    for folder in ("prefinal", "fields")})
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def infer(args):
    from ics.methods.part2_evidence import ARMS, CONFIG
    manifest = json.loads(args.manifest.read_text())
    rows = load_rows(args.manifest)
    if args.keys:
        keys = args.keys.split(",")
        lookup = {row["key"]: row for row in rows}
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError("Explicit keys must be unique manifest members")
        rows = [lookup[key] for key in keys]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or (args.expected is not None and len(rows) != args.expected):
        raise ValueError("Empty cohort or unexpected episode count")
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory; existing runs are never replaced")
    root = str((args.foris_root or Path(manifest["foris_root"])).resolve())
    sources = native_sources(root)
    allowed = ("fold", "e", "c", "key", "support", "query", "batch", "role", "collection_fold",
               "feature_export", "packet_export")
    rows = [{name: row[name] for name in allowed if name in row} for row in rows]
    config = dict(CONFIG, root=str(args.root.resolve()), out=str(args.out.resolve()), foris_root=root,
                  data_root=manifest["data_root"], annotation_root=manifest["annotation_root"],
                  projection_basis=manifest["projection_basis"], threads=args.threads, workers=args.workers,
                  native_source=sources, source_manifest_sha256=sha(args.manifest),
                  projection_basis_sha256=sha(manifest["projection_basis"]),
                  primary_baseline="native_cache", official_baseline="native",
                  cpu_phase="Part2->Part3->Part4->native_binarize", gpu_phase="native_CUDA_CRF_only",
                  feature_space="retained post-Part1 normalized/debiased q/r; FP16 may drift from original",
                  method_sources={"runner": sha(__file__), "method": sha(Path(__file__).resolve().parents[1] /
                                "src/ics/methods/part2_evidence.py")})
    for folder in ("predictions", "prefinal", "fields", "native_source/models", "native_source/utils"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for name in NATIVE_FILES:
        shutil.copyfile(Path(root) / name, args.out / "native_source" / name)
    if native_sources(args.out / "native_source") != sources:
        raise RuntimeError("Source changed while snapshotting")
    write_json(args.out / "native_source_digest.json", sources)
    write_json(args.out / "manifest.json", rows)
    write_json(args.out / "config.json", config)
    seal = dict(state="INFERRING", manifest_sha256=sha(args.out / "manifest.json"),
                config_sha256=sha(args.out / "config.json"), native_source_snapshot_sha256=sources["sha256"],
                predictions={}, prefinal={}, fields={}, inputs={}, query_labels_opened=False, arms=list(ARMS))
    write_json(args.out / "sealed.json", seal)
    audit, start = {}, time.monotonic()
    # Bound CPU BLAS/torch workers to avoid starving peer GPU feeders.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    try:
        with mp.get_context("spawn").Pool(args.workers, initializer=start_worker, initargs=(config,)) as pool:
            for completed, receipt in enumerate(pool.imap_unordered(infer_episode, rows), 1):
                key = receipt["key"]
                for group in ("prefinal", "fields", "inputs"):
                    seal[group][key] = receipt[group]
                audit[key] = receipt
                print(json.dumps(dict(completed=completed, total=len(rows), key=key,
                                      seconds=receipt["elapsed_seconds"], drift=receipt["drift"])), flush=True)
        if native_sources(root) != sources or sha(manifest["projection_basis"]) != config["projection_basis_sha256"]:
            raise RuntimeError("Native source or positional basis changed during inference")
        worst = {name: max(audit[key]["drift"][name]["max_abs"] for key in audit) for name in ("s2", "s3", "score")}
        mask_drift = {"pre": sum(audit[key]["drift"]["pre"]["changed_pixels"] for key in audit)}
        write_json(args.out / "audit.json", dict(episodes=audit, elapsed_seconds=time.monotonic() - start,
                   parity_max_abs=worst, drift_pixels=mask_drift,
                   interpretation="Compare candidate with paired same-cache native first; official native stays separate"))
        seal.update(state="ALL_PREFINALS_SEALED", audit_sha256=sha(args.out / "audit.json"))
    except BaseException as error:
        seal.update(state="INFERENCE_FAILED", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "sealed.json", seal)
        raise
    write_json(args.out / "sealed.json", seal)
    write_json(args.out / "prefinal_sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows), parity_max_abs=worst, drift_pixels=mask_drift)), flush=True)


def verify_prefinal(out, seal):
    """Verify the immutable CPU artifact without opening any annotation."""
    config = json.loads((out / "config.json").read_text())
    for name in ("manifest", "config", "audit"):
        if sha(out / f"{name}.json") != seal[f"{name}_sha256"]:
            raise ValueError(f"Sealed {name} changed")
    if native_sources(out / "native_source")["sha256"] != seal["native_source_snapshot_sha256"]:
        raise ValueError("Native source snapshot changed")
    for group in ("prefinal", "fields"):
        rows = json.loads((out / "manifest.json").read_text())
        if set(seal[group]) != {row["key"] for row in rows}:
            raise ValueError(f"Incomplete {group} seal")
        for key, digest in seal[group].items():
            if sha(out / group / f"{key}.npz") != digest:
                raise ValueError(f"Sealed {group} changed: {key}")
    return config


def finalize(args):
    """One CUDA host, all arms, actual source-transformed query RGB; no encoder."""
    import torch
    from PIL import Image
    seal = json.loads((args.out / "sealed.json").read_text())
    ARMS = tuple(seal["arms"])
    if seal.get("state") != "ALL_PREFINALS_SEALED":
        raise ValueError("Finalize requires a complete sealed CPU prefinal run")
    config = verify_prefinal(args.out, seal)
    if not (args.out / "prefinal_sealed.json").exists() or sha(args.out / "prefinal_sealed.json") != sha(args.out / "sealed.json"):
        raise ValueError("CPU prefinal seal changed before finalization")
    if native_sources(config["foris_root"]) != config["native_source"]:
        raise ValueError("Current source differs from the frozen CPU source")
    if sha(config["projection_basis"]) != config["projection_basis_sha256"]:
        raise ValueError("Frozen native positional basis changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Native CRF requires CUDA; no CPU or alternate finalizer fallback")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    WORKER.update(config=config, host=None)
    host = get_host("cuda")
    rows = json.loads((args.out / "manifest.json").read_text())
    audit, start = {}, time.monotonic()
    seal.update(state="FINALIZING", prefinal_seal_sha256=sha(args.out / "prefinal_sealed.json"))
    write_json(args.out / "sealed.json", seal)
    try:
        for done, row in enumerate(rows, 1):
            key = row["key"]
            target = Path(config["data_root"]) / row["query"]
            cached_packet = packet(config["root"], row)
            if sha(target) != seal["inputs"][key]["query_rgb_sha256"]:
                raise ValueError(f"Query RGB changed: {key}")
            if sha(cached_packet) != seal["inputs"][key]["packet_sha256"]:
                raise ValueError(f"Native comparison packet changed: {key}")
            with Image.open(target) as image:
                host.set_target(image.convert("RGB"))
            predictions, times = {}, {}
            with np.load(args.out / "prefinal" / f"{key}.npz", allow_pickle=False) as saved:
                if set(saved.files) != set(ARMS):
                    raise ValueError("Prefinal arm set changed")
                for arm in ARMS:
                    pre = torch.from_numpy(unpack(saved[arm])).to("cuda")
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    with torch.no_grad():
                        final = host._finalize_mask(pre, host._tgt_image.unsqueeze(0))
                    torch.cuda.synchronize()
                    times[arm] = time.monotonic() - begin
                    if tuple(final.shape) != (1024, 1024) or final.dtype != torch.bool:
                        raise RuntimeError("Native CRF did not return a bool 1024x1024 mask")
                    predictions[arm] = np.packbits(final.cpu().numpy())
            with np.load(cached_packet, allow_pickle=False) as cached:
                native_arm = config.get("native_arm", ARMS[0])
                native_drift = int((unpack(predictions[native_arm]) != unpack(cached["native"])).sum())
            write_npz(args.out / "predictions" / f"{key}.npz", predictions)
            seal["predictions"][key] = sha(args.out / "predictions" / f"{key}.npz")
            audit[key] = dict(crf_seconds=times, native_changed_pixels=native_drift)
            host._tgt_image = host._orig_tgt_size = None
            print(json.dumps(dict(completed=done, total=len(rows), key=key,
                                  crf_seconds=times, native_changed_pixels=native_drift)), flush=True)
        if native_sources(config["foris_root"]) != config["native_source"]:
            raise ValueError("Native source changed during finalization")
        write_json(args.out / "finalize_audit.json", dict(episodes=audit, device="cuda",
                   elapsed_seconds=time.monotonic() - start, encoder_forwards=0,
                   query_labels_opened=False,
                   native_changed_pixels=sum(value["native_changed_pixels"] for value in audit.values())))
        seal.update(state="ALL_PREDICTIONS_SEALED", finalize_audit_sha256=sha(args.out / "finalize_audit.json"))
    except BaseException as error:
        seal.update(state="FINALIZATION_FAILED", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "sealed.json", seal)
        raise
    write_json(args.out / "sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("infer")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--root", type=Path, required=True, help="Cache/packet root, e.g. outputs/fresh600_root")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--foris-root", type=Path)
    replay.add_argument("--workers", type=int, choices=(1, 2), default=2)
    replay.add_argument("--threads", type=int, choices=(1, 2), default=1)
    replay.add_argument("--limit", type=int)
    replay.add_argument("--keys", help="Comma-separated manifest keys in explicit smoke order")
    replay.add_argument("--expected", type=int)
    finalizer = sub.add_parser("finalize", help="Native CUDA CRF only; requires sealed CPU pre masks")
    finalizer.add_argument("--out", type=Path, required=True)
    finalizer.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.command == "infer" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    {"infer": infer, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
