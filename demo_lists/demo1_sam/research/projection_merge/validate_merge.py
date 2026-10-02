#!/usr/bin/env python3
"""Fixed-threshold CPU validation, never a performance benchmark."""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
HERE = Path(__file__).resolve().parent
BASELINES = HERE.parents[1] / "sam_shared_decoder/execution_baselines"
sys.path.insert(0, str(BASELINES))
from baselines import (ImageCache, PositionCache, cached_predict, make_fixture,
                       official_predict, shape_plan)
from factor_candidate import FactorCache, FactorState, factor_predict
from projection_merge import MergeCache, cached_merge_predict, factor_merge_predict
from validate import TOLERANCE, decisions, error


def group_algebra(model, image, pe, dense, sparse):
    """Check every block, including PE/bias placement, away from the decoder."""
    position = PositionCache.build(model, pe)
    cache = MergeCache.build(model, image, dense, position)
    old = FactorCache.build(model, image, dense, position)
    state = FactorState(cache, sparse.shape[0])
    # Exercise nonunit position scales and nontrivial arbitrary low-rank terms.
    state.s = torch.randn_like(state.s)
    state.u = image.new_empty(sparse.shape[0], cache.first.base.shape[1], 13).normal_()
    state.v = image.new_empty(sparse.shape[0], 13, image.shape[1]).normal_()
    result = []
    for index, group in enumerate(cache.groups, 1):
        read = (model.transformer.layers[index].cross_attn_token_to_image
                if index == 1 else model.transformer.final_attn_token_to_image)
        expected = [state.project(read.k_proj, old.read_base[index][0], position.read_k[index]),
                    state.project(read.v_proj, old.read_base[index][1])]
        if index == 1:
            linear = model.transformer.layers[1].cross_attn_image_to_token.q_proj
            expected.append(state.project(linear, old.write_base[index], position.write_q[index]))
        else:
            expected.append(state.s * old.conv_base + state.u @ (state.v @ old.conv_matrix) + old.conv_bias)
        result.append({"names": group.names, "weight_shape": list(group.weight.shape),
                       "base_shape": list(group.base.shape), "checks": {
                           name: error(ref, got) for name, ref, got in
                           zip(group.names, expected, group.factor_project(state))}})
    return result


def run_case(seed, grid, tokens, spatial, dtype):
    model, image, pe, dense, sparse, metadata = make_fixture(seed, grid, tokens, dtype=dtype)
    if spatial:
        dense = torch.randn_like(image)
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
    for hook in hooks:
        hook.remove()
    assert list(ref[0].shape) == [sparse.shape[0], 4, 4*grid, 4*grid]
    assert list(ref[1].shape) == [sparse.shape[0], 4]
    case = {**metadata, "grid": grid, "batch": sparse.shape[0], "dtype": str(dtype),
            "dense_prompt": "shared spatial random" if spatial else "shared constant vector", "variants": []}
    routes = [("factor", "auto", "auto"), ("cached", "auto", "auto"),
              ("dense_assoc", "auto", "auto"),
              ("dense_assoc", "projected", "sparse"),
              ("dense_assoc", "associated", "sparse")]
    for method, order, write_output in routes:
        plans = None if method == "factor" else shape_plan(model, grid*grid, tokens, method, order, write_output)
        position = PositionCache.build(model, pe, plans)
        merged_cache = MergeCache.build(model, image, dense, position, method, plans)
        original_cache = (FactorCache.build(model, image, dense, position) if method == "factor"
                          else ImageCache.build(model, image, dense, position))
        predict = factor_merge_predict if method == "factor" else cached_merge_predict
        for backend in ("explicit", "sdpa"):
            original = (factor_predict(model, original_cache, sparse, backend) if method == "factor"
                        else cached_predict(model, original_cache, sparse, method, backend,
                                            order, write_output, plans=plans))
            trace = {}
            output = predict(model, merged_cache, sparse, backend, trace)
            checks = {name: error(value, trace[name]) for name, value in reference_trace.items()}
            checks.update(masks=error(ref[0], output[0]), iou=error(ref[1], output[1]),
                          merge_vs_original_masks=error(original[0], output[0]),
                          merge_vs_original_iou=error(original[1], output[1]))
            for multi in (False, True):
                select = slice(1, None) if multi else slice(0, 1)
                checks[f"public_{multi}_masks"] = error(ref[0][:, select], output[0][:, select])
                checks[f"public_{multi}_iou"] = error(ref[1][:, select], output[1][:, select])
            split = [predict(model, merged_cache, sparse[j:j+1], backend) for j in range(sparse.shape[0])]
            checks.update({f"microbatch_{name}": error(output[k], torch.cat([x[k] for x in split]))
                           for k, name in enumerate(("masks", "iou"))})
            failed = [name for name, check in checks.items() if not check["finite"] or check["max_abs"] > TOLERANCE]
            name = f"{method}_{order}_{write_output}_{backend}"
            entry = {"name": name, "passed": not failed, "failed": failed, "checks": checks,
                     "decisions": decisions(ref, output), "execution_plan": plans,
                     "groups": [None if g is None else {"names": g.names, "weight_shape": list(g.weight.shape)}
                                for g in merged_cache.groups]}
            case["variants"].append(entry)
            print(json.dumps({"dtype": str(dtype), "seed": seed, "N": grid*grid, "T": tokens,
                              "name": name, "passed": not failed,
                              "max_mask_abs": checks["masks"]["max_abs"],
                              "worst_check": max(c["max_abs"] for c in checks.values()),
                              "binary_flips": entry["decisions"]["binary_disagreements"]}), flush=True)
    return case


def expansion_counts():
    """Separate CPU operator trace; no profiler exists on inference timing paths."""
    model, image, pe, dense, sparse, _ = make_fixture(0, 8, 7)
    position = PositionCache.build(model, pe)
    original = FactorCache.build(model, image, dense, position)
    merged = MergeCache.build(model, image, dense, position)
    ranks = [8*(7-1)+2, 2*(8*(7-1)+1)+1]
    result = {}
    for name, cache, predict in (("original", original, factor_predict),
                                  ("merged", merged, factor_merge_predict)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
            predict(model, cache, sparse)
        expansions = [event.input_shapes for event in prof.events()
                      if event.name == "aten::bmm" and event.input_shapes
                      and event.input_shapes[0] in [[2, 64, rank] for rank in ranks]]
        result[name] = {"factor_expansion_bmm_count": len(expansions), "input_shapes": expansions}
    assert result["original"]["factor_expansion_bmm_count"] == 6
    assert result["merged"]["factor_expansion_bmm_count"] == 2
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = [(s, 8, 7, False) for s in (0, 1, 2)]
    cases += [(0, 8, t, False) for t in (5, 9, 17)]
    cases += [(0, 8, 7, True), (0, 64, 7, False), (1, 64, 9, False)]
    if args.quick:
        cases = cases[:1]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__,
              "absolute_tolerance": TOLERANCE, "tf32_matmul": False, "tf32_cudnn": False,
              "scope": "Random weights and synthetic encoded images; complete four-mask/IoU outputs. CPU evidence only, no GPU speed or pretrained accuracy claim.",
              "cases": [], "algebra": []}
    output_path = args.output or HERE / ("quick_results.json" if args.quick else "cpu_results.json")
    started = time.perf_counter()
    with torch.inference_mode():
        # FP64 first, then FP32 at actual SAM channel/head/depth dimensions.
        for dtype in (torch.float64, torch.float32):
            fixture = make_fixture(0, 2, 7, dtype=dtype)
            report["algebra"].append({"dtype": str(dtype), "groups": group_algebra(*fixture[:5])})
            report["cases"].append(run_case(0, 2, 7, False, dtype))
        for args_case in cases:
            report["cases"].append(run_case(*args_case, torch.float32))
            output_path.write_text(json.dumps(report, indent=2) + "\n")
        report["cpu_profiler_factor_expansion_counts"] = expansion_counts()
    algebra_checks = [check for entry in report["algebra"] for group in entry["groups"]
                      for check in group["checks"].values()]
    report["algebra_passed"] = all(c["finite"] and c["max_abs"] <= TOLERANCE for c in algebra_checks)
    report["all_passed"] = report["algebra_passed"] and all(v["passed"] for c in report["cases"] for v in c["variants"])
    report["variant_case_count"] = sum(len(c["variants"]) for c in report["cases"])
    report["elapsed_seconds_not_a_benchmark"] = time.perf_counter() - started
    report["source_sha256"] = {str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (HERE / "projection_merge.py", Path(__file__), BASELINES / "baselines.py", BASELINES / "factor_candidate.py")}
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {report['variant_case_count']} variant-cases; passed={report['all_passed']}; {output_path}", flush=True)
    if not report["all_passed"]:
        raise AssertionError("Original fixed tolerance failed; full evidence retained")


if __name__ == "__main__":
    main()
