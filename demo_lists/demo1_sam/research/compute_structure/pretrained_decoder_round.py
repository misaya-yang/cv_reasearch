#!/usr/bin/env python3
"""Replay exported pretrained encoded inputs: fixed gate and optional pure decoder timing.

Input schema is shared with quality_mechanisms/run_real_sam --save-encoded-inputs.
No encoder, downloads, training, GT-based prompt selection, or quality scoring.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sam_shared_decoder/execution_baselines"))
from baselines import PositionCache, official_modules, official_predict
import experimental_methods as experimental
from benchmark import measure, synchronize, reset_memory, memory_result

DEFAULT_VARIANTS = "cached_merged_phase:explicit,cached_merged_phase:sdpa,factor_merged_phase:explicit,factor_implicit_phase:explicit"


def variants(value):
    result = []
    for entry in value.split(","):
        parts = entry.split(":")
        if len(parts) != 2 or parts[0] not in experimental.METHODS or parts[1] not in ("explicit", "sdpa"):
            raise argparse.ArgumentTypeError(f"invalid registered method:attention variant: {entry}")
        result.append(tuple(parts))
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("duplicate variants are not independent trials")
    return result


def load_decoder(checkpoint, device):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {"transformer_dim": 256, "num_multimask_outputs": 3, "depth": 2, "num_heads": 8, "mlp_dim": 2048}
    if bundle.get("architecture") != expected:
        raise ValueError("decoder checkpoint architecture does not match official SAM vit-b decoder")
    transformer, decoder, _common, provenance = official_modules()
    model = decoder.MaskDecoder(transformer_dim=256,
        transformer=transformer.TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048),
        num_multimask_outputs=3).eval()
    model.load_state_dict(bundle["state_dict"], strict=True)
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in bundle["state_dict"].values()):
        raise ValueError("checkpoint is not FP32")
    return model.to(device=device, dtype=torch.float32), {k: v for k, v in bundle.items() if k != "state_dict"}, provenance


def load_inputs(path, device):
    with np.load(path, allow_pickle=False) as arrays:
        required = ("image_embeddings", "image_pe", "dense_nomask", "original_size", "input_size", "image_id", "mask_threshold")
        if any(k not in arrays for k in required):
            raise ValueError("encoded NPZ missing required replay fields")
        tensors = {key: torch.from_numpy(arrays[key].copy()).to(device) for key in required[:3]}
        sparse = {key.removeprefix("sparse_"): torch.from_numpy(arrays[key].copy()).to(device)
                  for key in arrays.files if key.startswith("sparse_")}
        metadata = {"original_size": arrays["original_size"].astype(int).tolist(),
                    "input_size": arrays["input_size"].astype(int).tolist(),
                    "image_id": int(arrays["image_id"]), "mask_threshold": float(arrays["mask_threshold"]),
                    "annotation_ids": arrays["annotation_ids"].astype(int).tolist() if "annotation_ids" in arrays else None}
    image, pe, dense = [tensors[key] for key in required[:3]]
    if image.ndim != 4 or image.shape[0:2] != (1, 256) or pe.shape != image.shape or dense.shape != image.shape:
        raise ValueError("one shared D256 image/PE/dense prompt required")
    for name, value in {**tensors, **sparse}.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"nonfinite or non-FP32 input: {name}")
    for name, value in sparse.items():
        if value.ndim != 3 or value.shape[0] < 1 or value.shape[-1] != 256:
            raise ValueError(f"invalid sparse prompt shape: {name}")
    if not sparse:
        raise ValueError("no exported prompt regimes")
    return image, pe, dense, sparse, metadata


def full_masks(logits, info):
    # Same resize -> crop -> resize -> threshold as official Sam.postprocess_masks.
    # Stream at most four prompts here; never move this diagnostic into timing.
    masks = F.interpolate(logits, (1024, 1024), mode="bilinear", align_corners=False)
    masks = masks[..., :info["input_size"][0], :info["input_size"][1]]
    return F.interpolate(masks, info["original_size"], mode="bilinear", align_corners=False) > info["mask_threshold"]


def check_outputs(reference, candidate, info):
    errors = {"masks": float((reference[0] - candidate[0]).abs().max()),
              "iou": float((reference[1] - candidate[1]).abs().max())}
    finite = all(bool(torch.isfinite(t).all()) for t in reference + candidate)
    if candidate[0].shape[:2] != (candidate[1].shape[0], 4) or candidate[1].shape[1:] != (4,):
        raise ValueError("candidate did not preserve four masks and four IoU predictions")
    flips = []
    for start in range(0, reference[0].shape[0], 4):
        a, b = full_masks(reference[0][start:start+4], info), full_masks(candidate[0][start:start+4], info)
        flips.extend((a != b).sum((-1, -2)).cpu().tolist())
        del a, b
    return {"passed": finite and max(errors.values()) <= 5e-5, "finite": finite, "max_abs": errors,
            "full_resolution_binary_flips_per_prompt_per_mask": flips,
            "all_four_iou_argmax_changes": int((reference[1].argmax(-1) != candidate[1].argmax(-1)).sum()),
            "multimask_iou_argmax_changes": int((reference[1][:, 1:].argmax(-1) != candidate[1][:, 1:].argmax(-1)).sum())}


def replay(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no GPU execution")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    if args.timings and args.mode == "compile" and args.compiler_backend != "inductor":
        raise ValueError("backend=eager cannot be used for performance evidence")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "report.json"
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
              "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "precision": "float32", "tf32": False, "tolerance": 5e-5, "torch": torch.__version__,
              "input_scope": "exported encoded real inputs; image and prompt encoders not executed in this replay",
              "timed_scope": "complete four-mask+IoU low-resolution decoder; no diagnostics, final resize, GT metrics, I/O, or input slicing inside timing",
              "quality_scope": "implementation numerical checks only; no GT accuracy or SOTA claim", "records": []}
    def save():
        temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(report, indent=2) + "\n"); temp.replace(path)
    save()
    try:
        model, state_metadata, sources = load_decoder(args.decoder_state, device)
        image, pe, dense, all_sparse, info = load_inputs(args.encoded_inputs, device)
        report.update(decoder_state_metadata=state_metadata, official_source_provenance=sources, encoded_metadata=info)
        regimes = list(all_sparse) if args.regimes == "all" else args.regimes.split(",")
        with torch.inference_mode():
            for regime in regimes:
                sparse = all_sparse[regime]
                if args.prompt_limit:
                    sparse = sparse[:args.prompt_limit]
                mb = min(args.microbatch, len(sparse))
                chunks = tuple(sparse[start:start+mb] for start in range(0, len(sparse), mb))
                for method, attention in args.variants:
                    plans = experimental.plans_for(method, model, image.shape[-2]*image.shape[-1], sparse.shape[1]+5, "auto", "auto")
                    factory = experimental.CacheFactory(method, plans)
                    def build_cache():
                        return factory.build(model, image, dense, PositionCache.build(model, pe, plans))
                    cache = build_cache()
                    def eager(sp, ca):
                        return experimental.forward(method, model, ca, sp, attention, plans)
                    forward = torch.compile(eager, backend=args.compiler_backend, fullgraph=True, dynamic=False) if args.mode == "compile" else eager
                    def batch(ca):
                        result = None
                        for sp in chunks:
                            result = None
                            result = forward(sp, ca)
                        return result
                    synchronize(device); start_mem = reset_memory(device); started = time.perf_counter()
                    first = batch(cache); synchronize(device)
                    first_ms = (time.perf_counter() - started)*1000
                    first_memory = memory_result(device, start_mem); del first
                    checks = []
                    for sp in chunks:
                        reference = official_predict(model, image, pe, dense, sp)
                        candidate = forward(sp, cache)
                        checks.append(check_outputs(reference, candidate, info))
                        del reference, candidate
                    passed = all(check["passed"] for check in checks)
                    record = {"regime": regime, "method": method, "attention": attention,
                              "attention_coverage": experimental.coverage(method, attention),
                              "prompt_count": len(sparse), "microbatch": mb, "requested_microbatch": args.microbatch,
                              "numeric_status": "PASSED" if passed else "FAILED", "checks": checks,
                              "compile_status": "optimizing_inductor_fullgraph" if args.mode == "compile" and args.compiler_backend == "inductor" else "diagnostic_backend" if args.mode == "compile" else "eager",
                              "first_complete_decode_ms_excluded": first_ms, "first_complete_decode_memory": first_memory}
                    if args.timings and (passed or args.diagnostic_on_failure):
                        for _ in range(args.warmup):
                            result = batch(cache); del result
                        record["timing_status"] = "PASSED_NUMERIC" if passed else "FAILED_DIAGNOSTIC_TIMING_ONLY"
                        record["warm_complete_prompt_batch"] = measure(lambda: batch(cache), device, args.repetitions)
                        record["warm_prompts_per_second"] = len(sparse)*1000/record["warm_complete_prompt_batch"]["median_ms"]
                        # Only one live cache in cold peak measurement.
                        cache = None; gc.collect()
                        cold_first = batch(build_cache()); del cold_first
                        record["cache_build_including_PE"] = measure(build_cache, device, args.repetitions)
                        record["cold_complete_prompt_batch"] = measure(lambda: batch(build_cache()), device, args.repetitions)
                    else:
                        record["timing_status"] = "CHECK_ONLY" if not args.timings else "SKIPPED_FIXED_NUMERIC_GATE"
                    report["records"].append(record); save()
                    print(json.dumps({"regime": regime, "method": method, "attention": attention,
                                      "numeric_status": record["numeric_status"], "timing_status": record["timing_status"]}), flush=True)
                    del forward, cache
                    gc.collect()
        report["all_numeric_passed"] = all(r["numeric_status"] == "PASSED" for r in report["records"])
        report["execution_completed"] = True
        report["status"] = "PASSED" if report["all_numeric_passed"] else "FAILED_DIAGNOSTIC_TIMING_ONLY" if args.timings and args.diagnostic_on_failure else "FAILED_FIXED_NUMERIC_GATE"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); save()
        return 0 if report["all_numeric_passed"] else 2
    except BaseException as exc:
        report.update(status="ERROR", execution_completed=False, error={"type": type(exc).__name__, "message": str(exc)})
        save(); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoded-inputs", type=Path, required=True)
    p.add_argument("--decoder-state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--variants", type=variants, default=variants(DEFAULT_VARIANTS))
    p.add_argument("--regimes", default="all")
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--prompt-limit", type=int, default=0)
    p.add_argument("--mode", choices=("eager", "compile"), default="eager")
    p.add_argument("--compiler-backend", choices=("inductor", "eager"), default="inductor")
    p.add_argument("--timings", action="store_true")
    p.add_argument("--diagnostic-on-failure", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if min(args.microbatch, args.warmup, args.repetitions, args.threads) < 1 or args.prompt_limit < 0:
        p.error("positive batches/warmup/repetitions/threads and nonnegative prompt limit required")
    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
