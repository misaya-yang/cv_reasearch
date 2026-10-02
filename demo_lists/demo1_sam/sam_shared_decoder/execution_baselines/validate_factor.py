#!/usr/bin/env python3
"""Add inference-only candidate checks without modifying original baseline tests."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from baselines import PositionCache, make_fixture, official_predict
from factor_candidate import FactorCache, factor_predict
from validate import TOLERANCE, decisions, error

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    baseline_report = json.loads((HERE / "validation_results.json").read_text())
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "python": platform.python_version(), "absolute_tolerance": TOLERANCE,
        "scope": "Inference-only projected factors with dense LN companion, native-order first-image caches. Same frozen random fixtures as baseline suite. No trained weights, GPU or implicit attention schedule.",
        "frozen_baseline_results_sha256": hashlib.sha256((HERE / "validation_results.json").read_bytes()).hexdigest(),
        "cases": []}
    with torch.inference_mode():
        for fixture in baseline_report["cases"]:
            seed, grid, tokens, batch = fixture["seed"], fixture["grid"], fixture["T"], fixture["batch"]
            model, image, pe, dense, sparse, metadata = make_fixture(seed, grid, tokens, batch)
            assert metadata["model_fp32_state_sha256"] == fixture["model_fp32_state_sha256"]
            if fixture["dense_prompt"] == "shared spatial random": dense = torch.randn_like(image)
            reference_trace = {}
            handles = []
            for i, block in enumerate(model.transformer.layers):
                def capture(_m, _a, out, i=i):
                    reference_trace[f"layer{i}.sparse"] = out[0].detach()
                    reference_trace[f"layer{i}.dense"] = out[1].detach()
                handles.append(block.register_forward_hook(capture))
            handles.append(model.transformer.norm_final_attn.register_forward_hook(
                lambda _m, _a, out: reference_trace.__setitem__("final.sparse", out.detach())))
            ref = official_predict(model, image, pe, dense, sparse)
            for handle in handles: handle.remove()
            native_modes = {}
            for mode in (False, True):
                native_modes[mode] = model(image_embeddings=image, image_pe=pe,
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense.expand(batch, -1, -1, -1),
                    multimask_output=mode)
            pc = PositionCache.build(model, pe)
            assert pc.read_k[0] is None and pc.write_q[0] is None
            cache = FactorCache.build(model, image, dense, pc)
            assert cache.read_base[0] is None and cache.write_base[0] is None
            case = {**metadata, "grid": grid, "batch": batch, "dense_prompt": fixture["dense_prompt"], "variants": []}
            for backend in ("explicit", "sdpa"):
                trace = {}
                output = factor_predict(model, cache, sparse, backend, trace)
                errors = {k: error(v, trace[k]) for k, v in reference_trace.items()}
                errors.update(masks=error(ref[0], output[0]), iou=error(ref[1], output[1]))
                for mode in (False, True):
                    select = slice(1, None) if mode else slice(0, 1)
                    errors[f"public_{mode}_masks"] = error(native_modes[mode][0], output[0][:, select])
                    errors[f"public_{mode}_iou"] = error(native_modes[mode][1], output[1][:, select])
                expected = [8*(tokens-1)+2, 2*(8*(tokens-1)+1)+1]
                ranks = [trace[f"layer{i}.rank"] for i in range(2)]
                assert ranks == expected
                chunks = [factor_predict(model, cache, sparse[i:i+1], backend) for i in range(batch)]
                micro = {name: error(output[k], torch.cat([chunk[k] for chunk in chunks]))
                         for k, name in enumerate(("masks", "iou"))}
                failed = [name for name, e in errors.items() if not e["finite"] or e["max_abs"] > TOLERANCE]
                failed += [f"microbatch.{name}" for name,e in micro.items() if not e["finite"] or e["max_abs"] > TOLERANCE]
                entry = {"backend": backend, "errors": errors, "ranks": ranks,
                    "microbatch_1_vs_2": micro, "decisions": decisions(ref, output),
                    "passed": not failed, "failed": failed}
                case["variants"].append(entry)
                print(json.dumps({"seed": seed, "N": grid*grid, "T": tokens, "backend": backend,
                    "passed": not failed, "max_mask_abs": errors["masks"]["max_abs"],
                    "max_iou_abs": errors["iou"]["max_abs"], "worst_state": max(e["max_abs"] for e in errors.values()),
                    "binary_flips": entry["decisions"]["binary_disagreements"]}), flush=True)
            result["cases"].append(case)
            (HERE / "factor_validation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["all_passed"] = all(v["passed"] for c in result["cases"] for v in c["variants"])
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (HERE / "baselines.py", HERE / "factor_candidate.py", Path(__file__))}
    (HERE / "factor_validation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_passed"]: raise AssertionError("Fixed tolerance failed; results retained")


if __name__ == "__main__":
    main()
