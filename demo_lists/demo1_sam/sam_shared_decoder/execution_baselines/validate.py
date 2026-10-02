#!/usr/bin/env python3
"""Bounded FP32 CPU tests: output, normalized-state and decision fidelity."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from baselines import (ImageCache, PositionCache, cached_predict, make_fixture,
                       official_predict, shape_plan)

HERE = Path(__file__).resolve().parent
TOLERANCE = 5e-5


def error(ref, got):
    delta = (ref - got).abs()
    return {"shape": list(ref.shape), "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "finite": bool(torch.isfinite(ref).all() and torch.isfinite(got).all())}


def decisions(ref, got):
    rm, ri = ref; gm, gi = got
    flip = (rm > 0) != (gm > 0)
    top = ri.sort(dim=-1, descending=True).values
    return {"mask_threshold": 0.0, "binary_disagreements": int(flip.sum()),
        "mask_logits_count": rm.numel(), "minimum_abs_mask_logit": rm.abs().min().item(),
        "reference_logits_within_error_of_zero": int((rm.abs() <= (rm-gm).abs().max()).sum()),
        "all_four_iou_argmax_disagreements": int((ri.argmax(-1) != gi.argmax(-1)).sum()),
        "multimask_iou_argmax_disagreements": int((ri[:, 1:].argmax(-1) != gi[:, 1:].argmax(-1)).sum()),
        "minimum_all_four_iou_top_two_margin": (top[:, 0] - top[:, 1]).min().item(),
        "decoder_selection": "Single returns mask token 0; multimask returns tokens 1:4. Both slices checked; argmax is an additional downstream diagnostic."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cases = [(s, 8, 7, False) for s in (0, 1, 2)]
    cases += [(0, 8, t, False) for t in (5, 9, 17)]
    cases += [(0, 8, 7, True), (0, 64, 7, False), (1, 64, 9, False)]
    if args.quick: cases = cases[:1]
    variants = [
        ("cached_explicit", "cached", "explicit", "auto", "auto"),
        ("cached_sdpa", "cached", "sdpa", "auto", "auto"),
        ("dense_assoc_auto_explicit", "dense_assoc", "explicit", "auto", "auto"),
        ("dense_assoc_auto_sdpa", "dense_assoc", "sdpa", "auto", "auto"),
        ("dense_assoc_forced_explicit", "dense_assoc", "explicit", "associated", "sparse"),
        ("dense_assoc_forced_sdpa", "dense_assoc", "sdpa", "associated", "sparse"),
    ]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "python": platform.python_version(),
        "device": "cpu", "dtype": "float32", "absolute_tolerance": TOLERANCE,
        "scope": "Random weights, official SAM dimensions D256/H8/depth2/MLP2048, shared image and dense prompt. Complete four-mask logits and IoU, normalized dense/sparse states, both public output slices. No pretrained, prompt encoder, predictor resizing, SAM2 or GPU.",
        "cases": []}
    out = HERE / ("quick_results.json" if args.quick else "validation_results.json")
    start = time.perf_counter()
    with torch.inference_mode():
        for seed, grid, t, spatial in cases:
            model, image, pe, dense, sparse, metadata = make_fixture(seed, grid, t)
            if spatial: dense = torch.randn_like(image)
            reference_trace = {}
            hooks = []
            for i, block in enumerate(model.transformer.layers):
                def capture(_module, _args, output, i=i):
                    reference_trace[f"layer{i}.sparse"] = output[0].detach()
                    reference_trace[f"layer{i}.dense"] = output[1].detach()
                hooks.append(block.register_forward_hook(capture))
            hooks.append(model.transformer.norm_final_attn.register_forward_hook(
                lambda _m, _a, out: reference_trace.__setitem__("final.sparse", out.detach())))
            ref = official_predict(model, image, pe, dense, sparse)
            for hook in hooks: hook.remove()
            native_modes = {}
            for multi in (False, True):
                native_modes[multi] = model(image_embeddings=image, image_pe=pe,
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense.expand(2, -1, -1, -1),
                    multimask_output=multi)
                select = slice(1, None) if multi else slice(0, 1)
                assert all(torch.equal(a, b[:, select]) for a,b in zip(native_modes[multi], ref))
            pc = PositionCache.build(model, pe)
            cache = ImageCache.build(model, image, dense, pc)
            case = {**metadata, "grid": grid, "batch": 2,
                    "dense_prompt": "shared spatial random" if spatial else "shared constant vector",
                    "variants": []}
            for name, method, backend, order, write_output in variants:
                trace = {}
                result = cached_predict(model, cache, sparse, method, backend, order, write_output, trace)
                checks = {k: error(v, trace[k]) for k,v in reference_trace.items()}
                checks.update(masks=error(ref[0], result[0]), iou=error(ref[1], result[1]))
                for multi in (False, True):
                    select = slice(1, None) if multi else slice(0, 1)
                    checks[f"public_{multi}_masks"] = error(native_modes[multi][0], result[0][:, select])
                    checks[f"public_{multi}_iou"] = error(native_modes[multi][1], result[1][:, select])
                failed = [k for k,v in checks.items() if not v["finite"] or v["max_abs"] > TOLERANCE]
                entry = {"name": name, "checks": checks, "passed": not failed, "failed": failed,
                    "decisions": decisions(ref, result),
                    "execution_plan": shape_plan(model, grid*grid, t, method, order, write_output)}
                case["variants"].append(entry)
                print(json.dumps({"seed": seed, "N": grid*grid, "T": t, "method": name,
                    "passed": not failed, "max_state_abs": max(v["max_abs"] for v in checks.values()),
                    "max_mask_abs": checks["masks"]["max_abs"], "max_iou_abs": checks["iou"]["max_abs"],
                    "binary_flips": entry["decisions"]["binary_disagreements"]}), flush=True)
                if failed: raise AssertionError((name, failed))
            # Same prompts via microbatch 1 and 2; concatenate every output.
            full = cached_predict(model, cache, sparse, "dense_assoc")
            split = [cached_predict(model, cache, sparse[j:j+1], "dense_assoc") for j in range(2)]
            chunked = tuple(torch.cat([r[k] for r in split]) for k in range(2))
            case["microbatch_1_vs_2"] = {"masks": error(full[0], chunked[0]), "iou": error(full[1], chunked[1])}
            assert all(v["max_abs"] <= TOLERANCE for v in case["microbatch_1_vs_2"].values())
            try:
                ImageCache.build(model, image, dense.expand(2, -1, -1, -1), pc)
            except ValueError:
                case["rejects_per_prompt_dense_input"] = True
            else: raise AssertionError("Shared-cache contract guard failed")
            report["cases"].append(case)
            out.write_text(json.dumps(report, indent=2) + "\n")
    report["all_passed"] = True
    report["case_count"] = len(cases)
    report["variant_case_count"] = len(cases) * len(variants)
    report["run_budget_seconds_not_a_benchmark"] = time.perf_counter() - start
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE / "baselines.py", Path(__file__))}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {report['variant_case_count']} variants; {out}", flush=True)


if __name__ == "__main__":
    main()
