#!/usr/bin/env python3
"""Frozen middle/end RCG placement using the original cached native tail.

infer is CPU only and reads q/r plus packet s2, score, cov and pre. It replaces
only the scalar score entering the fixed additive native tail, never regenerates
foreground/background evidence, and never opens query annotations. finalize is
the shared native CUDA CRF phase; complete predictions must seal before scoring.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.experiment import load_rows, packet, sha, unpack
import run_part2_evidence as shared

ARMS = ("native.cache.control", "rcg.middle", "rcg.end.control", "mean.middle.control")
EPSILON = 1e-6


def identity(row):
    return tuple(row[name] for name in ("fold", "e", "c", "support", "query"))


def span_info(field):
    low, high = float(field.min()), float(field.max())
    raw_span = float(field.max() - field.min())
    return dict(min=low, max=high, raw_span=raw_span,
                effective_span=max(raw_span, EPSILON), epsilon=EPSILON)


def restore_units(z, info):
    """The prespecified positive affine mapping, without clipping or tuning."""
    return (np.float32(info["min"]) + np.float32(info["effective_span"]) *
            np.asarray(z, dtype=np.float32)).astype(np.float32)


def middle_score(native_score, native_s2, replacement_s2):
    """Keep cached tail corrections; use g + (new_s2 - s2), in this order."""
    delta = np.asarray(replacement_s2 - native_s2, dtype=np.float32)
    if not np.any(delta):
        return native_score.copy()  # Preserve every bit, including signed zeros.
    return np.asarray(native_score + delta, dtype=np.float32)


def get_binarizer():
    """Import and call the original unbound method; never construct a host."""
    if "binarizer" in shared.WORKER:
        return shared.WORKER["binarizer"]
    config = shared.WORKER["config"]
    if shared.native_sources(config["foris_root"]) != config["native_source"]:
        raise ValueError("Native source changed after freezing")
    sys.path.insert(0, config["foris_root"])
    import models.foris as native
    if Path(native.__file__).resolve() != (Path(config["foris_root"]) / "models/foris.py").resolve():
        raise ValueError("Wrong models.foris import")
    function = native.FoRIS._binarize_response
    if tuple(inspect.signature(function).parameters) != ("self", "score_hw", "target_hw"):
        raise ValueError("Native binarizer signature changed")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if any(isinstance(node, ast.Name) and node.id == "self"
           for statement in tree.body[0].body for node in ast.walk(statement)):
        raise ValueError("Native binarizer now accesses self; constructor-free call is invalid")
    shared.WORKER["binarizer"] = function
    return function


def render(field):
    import torch
    value = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))
    return get_binarizer()(None, value, target_hw=(1024, 1024)).numpy()


def verify_end_run(path, rows):
    if path is None:
        return None
    seal = json.loads((path / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("End field provider must be a complete sealed run")
    if sha(path / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("End field provider manifest changed")
    source_rows = load_rows(path / "manifest.json")
    source = {row["key"]: row for row in source_rows}
    for row in rows:
        if row["key"] not in source or identity(source[row["key"]]) != identity(row):
            raise ValueError(f"End field episode mismatch: {row['key']}")
    return dict(path=str(path.resolve()), manifest_sha256=seal["manifest_sha256"],
                seal_sha256=sha(path / "sealed.json"), fields=seal.get("fields", {}),
                inputs=seal.get("inputs", {}),
                input_binding=("provider per-episode hashes verified when present; otherwise identity only; "
                               "current cache/packet hashes recorded, no historical input hash invented"))


def infer_episode(row):
    import torch
    from ics.methods import rcg, mean_graph
    config, key = shared.WORKER["config"], row["key"]
    root, out = Path(config["root"]), Path(config["out"])
    started = time.monotonic()
    feature = root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt")
    cached_packet = packet(root, row)
    target = Path(config["data_root"]) / row["query"]
    inputs = dict(feature_sha256=sha(feature), packet_sha256=sha(cached_packet),
                  query_rgb_sha256=sha(target))
    cache = torch.load(feature, map_location="cpu", weights_only=True)
    q, r = cache["q"], cache["r"]
    if any(tuple(value.shape) != (4096, 1024) or not torch.isfinite(value).all() for value in (q, r)):
        raise ValueError("Require finite retained [4096,1024] post-Part1 q/r")
    # np.load is lazy. Only these four approved packet members are accessed.
    with np.load(cached_packet, allow_pickle=False) as saved:
        s2, g, cov = (saved[name].astype(np.float32) for name in ("s2", "score", "cov"))
        cached_pre = saved["pre"].copy()
    if any(field.shape != (64, 64) or not np.isfinite(field).all() for field in (s2, g, cov)):
        raise ValueError("Require finite native 64x64 s2/score/cov")
    native_mask = unpack(cached_pre)
    rendered_native = render(g)
    parity = int((native_mask != rendered_native).sum())
    if parity:
        raise ValueError(f"Original native score/pre binarizer parity failed: {key}, {parity} pixels")
    if middle_score(g, s2, s2).tobytes() != g.tobytes():
        raise ValueError("Zero-delta intervention changed cached native score bits")
    spans = dict(s2=span_info(s2), score=span_info(g))
    timings, metadata = {}, {}
    begin = time.monotonic()
    z_middle, metadata[ARMS[1]] = rcg.predict(q, r, cov, s2, device="cpu")
    timings[ARMS[1]] = time.monotonic() - begin
    begin = time.monotonic()
    z_mean, metadata[ARMS[3]] = mean_graph.predict(q, r, cov, s2, device="cpu")
    timings[ARMS[3]] = time.monotonic() - begin
    begin = time.monotonic()
    provider = config["end_provider"]
    if provider is not None and key in provider["fields"]:
        source = Path(provider["path"]) / "fields" / f"{key}.npz"
        digest = sha(source)
        if digest != provider["fields"][key]:
            raise ValueError(f"Sealed external RCG field changed: {key}")
        provider_inputs = provider["inputs"].get(key, {})
        for name in ("feature_sha256", "packet_sha256"):
            if name in provider_inputs and provider_inputs[name] != inputs[name]:
                raise ValueError(f"End field used different {name}: {key}")
        with np.load(source, allow_pickle=False) as saved:
            z_end = saved["rcg"].astype(np.float32)
        inputs["end_field_sha256"] = digest
        metadata[ARMS[2]] = dict(mode="sealed_existing_rcg_field", field=str(source),
                                 provider_manifest_sha256=provider["manifest_sha256"],
                                 provider_seal_sha256=provider["seal_sha256"],
                                 input_binding=provider["input_binding"])
    else:
        z_end, details = rcg.predict(q, r, cov, g, device="cpu")
        metadata[ARMS[2]] = dict(mode="canonical_locked_rcg_on_cached_final_score", solver=details)
    timings[ARMS[2]] = time.monotonic() - begin
    if any(field.shape != (64, 64) or not np.isfinite(field).all() for field in (z_middle, z_mean, z_end)):
        raise ValueError("Graph readout must be a finite 64x64 field")
    mapped_middle = restore_units(z_middle, spans["s2"])
    mapped_mean = restore_units(z_mean, spans["s2"])
    mapped_end = restore_units(z_end, spans["score"])
    scores = {ARMS[0]: g.copy(), ARMS[1]: middle_score(g, s2, mapped_middle),
              ARMS[2]: z_end.copy(),
              ARMS[3]: middle_score(g, s2, mapped_mean)}
    prefinal = {ARMS[0]: cached_pre}
    for arm in ARMS[1:]:
        begin = time.monotonic()
        prefinal[arm] = np.packbits(render(scores[arm]))
        timings[f"{arm}.binarize"] = time.monotonic() - begin
    # Audit finite-precision deviation from positive-affine invariance at the end.
    end_affine_drift = int((render(mapped_end) != unpack(prefinal[ARMS[2]])).sum())
    fields = dict(scores)
    fields.update({"native.s2": s2, "native.tail_correction": g - s2,
                   "rcg.middle.z": z_middle, "rcg.middle.mapped_s2": mapped_middle,
                   "mean.middle.control.z": z_mean, "mean.middle.control.mapped_s2": mapped_mean,
                   "rcg.end.control.z": z_end, "rcg.end.control.mapped_score": mapped_end})
    for folder, values in (("prefinal", prefinal), ("fields", fields)):
        shared.write_npz(out / folder / f"{key}.npz", values)
    return dict(key=key, inputs=inputs, spans=spans, graph_metadata=metadata, timings=timings,
                elapsed_seconds=time.monotonic() - started,
                cache=dict(q_dtype=str(q.dtype), r_dtype=str(r.dtype),
                           debiased=bool(cache["debiased"]) if "debiased" in cache else "unrecorded"),
                native_pre_changed_pixels=parity, zero_delta_score_bitexact=True,
                end_positive_affine_binarize_changed_pixels=end_affine_drift,
                **{folder: sha(out / folder / f"{key}.npz") for folder in ("prefinal", "fields")})


def infer(args):
    from ics.methods import rcg, mean_graph
    manifest = json.loads(args.manifest.read_text())
    rows = load_rows(args.manifest)
    if args.keys:
        keys = args.keys.split(",")
        lookup = {row["key"]: row for row in rows}
        if len(keys) != len(set(keys)) or any(key not in lookup for key in keys):
            raise ValueError("Explicit smoke keys must be unique manifest members")
        rows = [lookup[key] for key in keys]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or (args.expected is not None and len(rows) != args.expected):
        raise ValueError("Empty cohort or unexpected episode count")
    if args.out.exists():
        raise FileExistsError("Use a fresh run directory")
    foris_root = str((args.foris_root or Path(manifest["foris_root"])).resolve())
    sources = shared.native_sources(foris_root)
    provider = verify_end_run(args.end_run, rows)
    allowed = ("fold", "e", "c", "key", "support", "query", "batch", "role", "collection_fold",
               "feature_export", "packet_export")
    rows = [{name: row[name] for name in allowed if name in row} for row in rows]
    method_files = dict(runner=__file__, rcg=rcg.__file__, mean_graph=mean_graph.__file__)
    method_hashes = {name: sha(path) for name, path in method_files.items()}
    config = dict(root=str(args.root.resolve()), out=str(args.out.resolve()), foris_root=foris_root,
                  data_root=manifest["data_root"], annotation_root=manifest["annotation_root"],
                  projection_basis=manifest["projection_basis"], native_source=sources,
                  projection_basis_sha256=sha(manifest["projection_basis"]),
                  source_manifest_sha256=sha(args.manifest), workers=args.workers, threads=args.threads,
                  arms=list(ARMS), native_arm=ARMS[0], primary="rcg.middle", primary_baseline=ARMS[0],
                  official_baseline="native", cpu_phase="cached_native_tail", gpu_phase="native_CUDA_CRF_only",
                  source_prior="score-only middle refinement; not evidence replacement",
                  tail_rule="g_cached + (min(s2) + max(span(s2),1e-6)*z_middle - s2); native tail evidence unchanged",
                  end_rule="Native binarizer receives z_end directly; explicit min(g)+max(span(g),1e-6)*z_end mapping saved for unit symmetry",
                  finalizer="source FoRIS binarizer + native CUDA CRF for every arm",
                  decoder_mode="native", epsilon=EPSILON, rcg_config=dict(rcg.CONFIG),
                  mean_graph_config=dict(mean_graph.CONFIG), end_provider=provider,
                  method_sources=method_hashes, encoder_forwards=0, query_labels_in_inference=False,
                  feature_space="retained post-Part1 q/r; canonical graphs perform their own token normalization",
                  parameter_provenance="locked RCG and MEAN constants; placement/remapping fixed before scoring",
                  floating_point_note="g+(mapped_s2-s2) preserves zero delta; subtraction/readdition of the cached tail is avoided")
    for folder in ("predictions", "prefinal", "fields", "native_source/models", "native_source/utils"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for name in shared.NATIVE_FILES:
        shutil.copyfile(Path(foris_root) / name, args.out / "native_source" / name)
    if shared.native_sources(args.out / "native_source") != sources:
        raise ValueError("Source changed while copying snapshot")
    shared.write_json(args.out / "native_source_digest.json", sources)
    shared.write_json(args.out / "manifest.json", rows)
    shared.write_json(args.out / "config.json", config)
    seal = dict(state="INFERRING", manifest_sha256=sha(args.out / "manifest.json"),
                config_sha256=sha(args.out / "config.json"), native_source_snapshot_sha256=sources["sha256"],
                predictions={}, prefinal={}, fields={}, inputs={}, query_labels_opened=False, arms=list(ARMS))
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
                                      seconds=receipt["elapsed_seconds"], native_pre_changed_pixels=receipt["native_pre_changed_pixels"])), flush=True)
        if shared.native_sources(foris_root) != sources:
            raise ValueError("Native source changed during inference")
        if {name: sha(path) for name, path in method_files.items()} != method_hashes:
            raise ValueError("Locked method/runner source changed during inference")
        if sha(manifest["projection_basis"]) != config["projection_basis_sha256"]:
            raise ValueError("Frozen positional basis changed")
        if provider is not None and sha(Path(provider["path"]) / "sealed.json") != provider["seal_sha256"]:
            raise ValueError("End provider seal changed during inference")
        shared.write_json(args.out / "audit.json", dict(episodes=audit, elapsed_seconds=time.monotonic() - started,
                          native_pre_changed_pixels=sum(value["native_pre_changed_pixels"] for value in audit.values()),
                          source_prior=config["source_prior"], complete_method_result=False,
                          query_labels_opened=False))
        seal.update(state="ALL_PREFINALS_SEALED", audit_sha256=sha(args.out / "audit.json"))
    except BaseException as error:
        seal.update(state="INFERENCE_FAILED", error=f"{type(error).__name__}: {error}")
        shared.write_json(args.out / "sealed.json", seal)
        raise
    shared.write_json(args.out / "sealed.json", seal)
    shared.write_json(args.out / "prefinal_sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows), arms=list(ARMS))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("infer")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--end-run", type=Path, help="Reuse its sealed rcg fields after manifest/hash checks")
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--foris-root", type=Path)
    replay.add_argument("--workers", type=int, choices=(1, 2), default=2)
    replay.add_argument("--threads", type=int, choices=(1, 2), default=1)
    replay.add_argument("--keys", help="Comma-separated explicit manifest keys in smoke order")
    replay.add_argument("--limit", type=int)
    replay.add_argument("--expected", type=int)
    finalizer = sub.add_parser("finalize", help="Shared native CUDA CRF phase; no encoder")
    finalizer.add_argument("--out", type=Path, required=True)
    finalizer.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.command == "infer" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    (infer if args.command == "infer" else shared.finalize)(args)


if __name__ == "__main__":
    main()
