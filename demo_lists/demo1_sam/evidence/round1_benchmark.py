#!/usr/bin/env python3
"""Benchmark-ready SAM baseline harness. CPU runs only validate this harness.

Explicit device/dtype/batching, independent cold/warm timings, synchronized
measurement, compilation warm-up excluded, no hooks or diagnostic reductions in
any timed forward. No GPU was run for this deliverable. No speedup is presumed.
"""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from baselines import (ImageCache, PositionCache, cached_predict, make_fixture,
                       official_predict, shape_plan, tensor_bytes)
from factor_candidate import FactorCache, factor_predict
from official_sdpa import official_sdpa_predict

HERE = Path(__file__).resolve().parent


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def reset_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)
    return None


def memory_result(device, start):
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        return {"start_allocated_bytes": start, "peak_allocated_bytes": peak,
                "peak_increment_bytes": peak - start,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return {"peak_allocated_bytes": None,
            "reason": "PyTorch allocator peak accounting is implemented here only for CUDA; CPU/MPS peak is not measured."}


def measure(function, device, repetitions):
    samples = []
    gc.collect()
    synchronize(device)
    start_mem = reset_memory(device)
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter()
        value = function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        # Release each result before next iteration. Timed forwards always finish
        # before synchronization; no item(), traces, comparisons, or hooks here.
        del value
    return {"milliseconds": samples, "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "memory": memory_result(device, start_mem)}


def compare(ref, got, tolerance):
    metrics = {}
    for name, a, b in zip(("masks", "iou"), ref, got):
        delta = (a - b).abs()
        metrics[name] = {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    metrics["passed"] = all(x["finite"] and x["max_abs"] <= tolerance for x in metrics.values())
    return metrics


def profile_forward(function, device, path):
    """Separate untimed execution; retain operators AND actual device events."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, record_shapes=True,
                                profile_memory=True) as prof:
        value = function()
        synchronize(device)
        del value
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    rows = [{"name": e.key, "count": e.count,
             "self_cpu_us": e.self_cpu_time_total,
             "self_device_us": e.self_device_time_total,
             "self_cpu_memory_bytes": e.self_cpu_memory_usage,
             "self_device_memory_bytes": e.self_device_memory_usage,
             "input_shapes": e.input_shapes}
            for e in prof.key_averages(group_by_input_shape=True)]
    kernels = sorted({e.name for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA})
    return {"trace": str(path), "operators": rows, "observed_cuda_events": kernels,
            "sdpa_note": "Dispatch must be read from these events; API usage alone is not FlashAttention evidence."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True, help="Explicit cpu, cuda:0, or mps; never auto-selects GPU")
    p.add_argument("--dtype", required=True, choices=("float32", "float64", "float16", "bfloat16"))
    p.add_argument("--prompt-batch", required=True, type=int)
    p.add_argument("--microbatch", required=True, type=int)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--tokens", type=int, default=7)
    p.add_argument("--method", choices=("all", "official", "official_sdpa", "cached", "dense_assoc", "factor_projected"), default="all")
    p.add_argument("--attention", choices=("explicit", "sdpa"), default="explicit")
    p.add_argument("--order", choices=("auto", "projected", "associated"), default="auto")
    p.add_argument("--write-output", choices=("auto", "dense", "sparse"), default="auto")
    p.add_argument("--mode", choices=("eager", "compile"), default="eager")
    p.add_argument("--compiler-backend", default="inductor", choices=("inductor", "aot_eager", "eager"))
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=5e-5,
                   help="Gate applied before timing. 5e-5 is the validated FP32 tolerance; lower precision is not yet validated")
    p.add_argument("--tf32", action="store_true", help="Explicitly enable TF32; off by default")
    p.add_argument("--diagnostic-on-failure", action="store_true",
                   help="Retain FAILED numeric status but continue diagnostic timing; never relax tolerance")
    p.add_argument("--profile-dir", type=Path, help="Separate untimed profiler pass after timings")
    p.add_argument("--output", type=Path, default=HERE / "benchmark_results.json")
    args = p.parse_args()
    if args.dtype == "float32" and args.tolerance > 5e-5:
        p.error("FP32 tolerance cannot exceed the original 5e-5 gate")
    if min(args.prompt_batch, args.microbatch, args.repetitions, args.warmup) < 1:
        p.error("batch, microbatch, warmup and repetitions must be positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda", "mps"): p.error("Only cpu, cuda or mps are supported")
    if device.type == "cuda" and not torch.cuda.is_available(): p.error("Requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available(): p.error("Requested MPS device is unavailable")
    if device.type != "cuda" and args.tf32: p.error("TF32 is CUDA-only")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    dtype = getattr(torch, args.dtype)
    model, image, pe, dense, sparse, metadata = make_fixture(args.seed, args.grid,
        args.tokens, args.prompt_batch, device, dtype)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "options": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "fixture": metadata, "outputs": "All four low-resolution masks plus all four IoU; no output-token pruning",
        "timed_scope": "Complete mask decoder; no encoder, prompt encoder or full-resolution predictor postprocessing. Prompt batches share one image and identical dense embedding. Output copying/concatenation across microbatches excluded equally; final microbatch outputs returned.",
        "cpu_warning": "CPU smoke measurements test harness plumbing only; they are not a GPU ranking or speedup prediction." if device.type == "cpu" else None,
        "compile_warning": "eager/aot_eager compiler backends are diagnostic; only an optimizing backend such as inductor is a performance-compilation trial. Fullgraph mode raises on unsupported graph breaks. Actual CUDA SDPA backend dispatch is not inferred from the API name.",
        "precision_warning": "Random-weight same-dtype numerical gate does not establish pretrained, binary decision, TF32 or lower-precision fidelity.",
        "methods": []}
    methods = ("official", "cached", "dense_assoc", "factor_projected") if args.method == "all" else (args.method,)
    # Build views once, outside every timed path (including tail microbatches).
    chunks = tuple(sparse[lo:lo+args.microbatch]
                   for lo in range(0, args.prompt_batch, args.microbatch))
    with torch.inference_mode():
        # Numerical checks are streamed at the requested microbatch size, then
        # released. Never materialize a full prompt-batch decoder output merely
        # to gate a microbatched benchmark, or retain one in memory measurements.
        for method in methods:
            uncached = method in ("official", "official_sdpa")
            plans = None if uncached or method == "factor_projected" else shape_plan(model, args.grid*args.grid,
                args.tokens, method, args.order, args.write_output)
            position = None if uncached else PositionCache.build(model, pe, plans)
            cache_type = FactorCache if method == "factor_projected" else ImageCache
            cache = None if uncached else cache_type.build(model, image, dense, position)
            if method == "official":
                def eager(sp, ca): return official_predict(model, image, pe, dense, sp)
            elif method == "official_sdpa":
                def eager(sp, ca): return official_sdpa_predict(model, image, pe, dense, sp)
            elif method == "factor_projected":
                def eager(sp, ca): return factor_predict(model, ca, sp, args.attention)
            else:
                def eager(sp, ca): return cached_predict(model, ca, sp, method, args.attention,
                                                       args.order, args.write_output, plans=plans)
            forward = eager
            if args.mode == "compile":
                forward = torch.compile(eager, backend=args.compiler_backend,
                    mode=args.compile_mode, fullgraph=True, dynamic=False)
            def batch_forward(ca):
                # No accumulating/repeated concatenation of results in timing.
                result = None
                for sp in chunks:
                    result = None  # Free previous chunk before the next forward.
                    result = forward(sp, ca)
                return result
            # Isolate first invocation / optimizing compilation from numerical
            # comparisons. This is preparation, never steady-state acceleration.
            synchronize(device)
            first_memory_start = reset_memory(device)
            first_start = time.perf_counter()
            first_value = batch_forward(cache)
            synchronize(device)
            first_forward_ms = (time.perf_counter() - first_start) * 1000
            first_forward_memory = memory_result(device, first_memory_start)
            del first_value
            synchronize(device)
            init_start = time.perf_counter()
            checks = {"passed": True, "microbatch_checks": []}
            for lo in range(0, args.prompt_batch, args.microbatch):
                sp = sparse[lo:lo+args.microbatch]
                reference = official_predict(model, image, pe, dense, sp)
                candidate = forward(sp, cache)
                check = compare(reference, candidate, args.tolerance)
                checks["passed"] = checks["passed"] and check["passed"]
                checks["microbatch_checks"].append({"offset": lo, "size": sp.shape[0], **check})
                del reference, candidate, sp
            synchronize(device)
            prepare_ms = (time.perf_counter() - init_start) * 1000
            if not checks["passed"] and not args.diagnostic_on_failure:
                report["methods"].append({"method": method, "verification": checks,
                    "timing_skipped": "Fixed before-timing numerical gate failed; tolerance was not relaxed"})
                report["completed"] = False
                report["failure"] = "Numerical equivalence gate"
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                raise RuntimeError(f"Before-timing equivalence gate failed for {method}: {checks}")
            for _ in range(args.warmup):
                result = batch_forward(cache)
                del result
            synchronize(device)
            warm = measure(lambda: batch_forward(cache), device, args.repetitions)
            item = {"method": method, "effective_attention": "original explicit official" if method == "official" else "uncached SDPA adapter" if method == "official_sdpa" else args.attention,
                "verification": checks, "verification_ms_excluded": prepare_ms,
                "first_complete_prompt_batch_ms_excluded": first_forward_ms,
                "first_complete_prompt_batch_memory": first_forward_memory,
                "numeric_status": "passed_random_fixture_only" if checks["passed"] else "FAILED_DIAGNOSTIC_TIMING_ONLY",
                "compile_status": "optimizing_inductor_fullgraph" if args.mode == "compile" and args.compiler_backend == "inductor" else "diagnostic_backend" if args.mode == "compile" else "eager",
                "attention_coverage": "Original explicit throughout" if method == "official" else "SDPA throughout the uncached official adapter" if method == "official_sdpa" else "Candidate writes remain explicit to form factors; SDPA applies to reads and token self-attention" if method == "factor_projected" else "SDPA applies only to eligible routes; associated sparse writes stay explicit",
                "warm_complete_prompt_batch": warm,
                "warm_prompts_per_second": args.prompt_batch * 1000 / warm["median_ms"],
                "warm_ms_per_prompt": warm["median_ms"] / args.prompt_batch}
            if uncached:
                item["cache_build"] = "None: original official decoder; shared encoder inputs already supplied"
                item["cold_image_including_cache_and_prompt_batch"] = measure(lambda: batch_forward(None), device, args.repetitions)
            else:
                item["execution_plan"] = plans
                item["fixed_PE_build"] = measure(lambda: PositionCache.build(model, pe, plans), device, args.repetitions)
                item["image_cache_build_fixed_PE_warm"] = measure(
                    lambda: cache_type.build(model, image, dense, position), device, args.repetitions)
                def build_all():
                    return cache_type.build(model, image, dense, PositionCache.build(model, pe, plans))
                item["cold_cache_build_including_fixed_PE"] = measure(build_all, device, args.repetitions)
                # Release the old caches so the cold-memory peak doesn't include
                # an unrelated retained warm cache in addition to the cold one.
                item["logical_persistent_cache_bytes"] = tensor_bytes(cache)
                if args.profile_dir:
                    item["profile"] = profile_forward(lambda: batch_forward(cache), device,
                        args.profile_dir / f"{method}.json")
                cache = None; position = None; gc.collect()
                def cold_full():
                    return batch_forward(build_all())
                # Exercise fresh cache objects outside measurement. Compilation
                # may specialize on nested objects; do not charge that first
                # fresh-cache graph preparation as steady-state cold latency.
                synchronize(device)
                cold_prepare_start = time.perf_counter()
                cold_result = cold_full()
                synchronize(device)
                item["fresh_cache_first_forward_ms_excluded"] = (time.perf_counter() - cold_prepare_start)*1000
                del cold_result
                item["cold_image_including_cache_and_prompt_batch"] = measure(cold_full, device, args.repetitions)
            if uncached and args.profile_dir:
                item["profile"] = profile_forward(lambda: batch_forward(None), device,
                    args.profile_dir / f"{method}.json")
            report["methods"].append(item)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"method": method, "verified": checks["passed"],
                "warm_ms": warm["median_ms"], "device": str(device),
                "warning": report["cpu_warning"]}), flush=True)
            del forward, eager, cache, position
            gc.collect()
        report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (HERE / "baselines.py", HERE / "factor_candidate.py", HERE / "official_sdpa.py", Path(__file__))}
        report["completed"] = True
        report["all_numerical_checks_passed"] = all(m["verification"]["passed"] for m in report["methods"])
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
