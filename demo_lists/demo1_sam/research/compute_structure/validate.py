#!/usr/bin/env python3
"""Reproducible CPU identity tests; original absolute gate remains 5e-5."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import torch

from prototype import ResearchCache, DensePhaseCache, dense_phase_predict, predict, EXEC
from baselines import PositionCache, make_fixture, official_predict
from factor_candidate import factor_predict


def error(ref, out):
    diff = (ref - out).abs()
    return {"finite": bool(torch.isfinite(out).all()), "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).with_name("cpu_results.json")))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases = [(0, 4, 7, 2, True), (3, 8, 5, 3, True), (11, 8, 11, 2, True), (17, 16, 7, 2, True)]
    if args.full:
        source = json.loads((EXEC / "validation_results.json").read_text())
        cases = [(c["seed"], c["grid"], c["T"], c["batch"], c["dense_prompt"] == "shared spatial random")
                 for c in source["cases"]]
    report = {"scope": "CPU random-weight complete decoder, synthetic encoded input. No GPU or trained accuracy claim.",
              "torch": torch.__version__, "python": platform.python_version(), "absolute_tolerance": 5e-5,
              "cases": [], "variants": {
                  "phase": [False, False, True], "implicit": [True, False, False],
                  "statistics": [False, True, False], "all": [True, True, True]}}
    with torch.inference_mode():
        for dtype in (torch.float64, torch.float32):
            for seed, grid, tokens, batch, spatial in cases:
                model, image, pe, dense, sparse, metadata = make_fixture(seed, grid, tokens, batch, dtype=dtype)
                # Preserve the original FP32 RNG stream and dense-prompt mode.
                if spatial:
                    dense = torch.randn_like(image)
                cache = ResearchCache.build(model, image, dense, PositionCache.build(model, pe))
                ref = official_predict(model, image, pe, dense, sparse)
                for name, switches in report["variants"].items():
                    started = time.perf_counter()
                    out = predict(model, cache, sparse, implicit_attention=switches[0],
                                  statistic_ln=switches[1], phase_layout=switches[2])
                    errors = {"masks": error(ref[0], out[0]), "iou": error(ref[1], out[1])}
                    entry = {"seed": seed, "grid": grid, "T": tokens, "batch": batch,
                             "dtype": str(dtype), "variant": name, "errors": errors,
                             "dense_prompt": "shared spatial random" if spatial else "shared constant vector",
                             "model_fp32_state_sha256": metadata["model_fp32_state_sha256"],
                             "binary_flips": int(((ref[0] > 0) != (out[0] > 0)).sum()),
                             "mask_shape": list(out[0].shape), "iou_shape": list(out[1].shape),
                             "passed": all(e["finite"] and e["max_abs"] <= 5e-5 for e in errors.values()),
                             "diagnostic_wall_s": time.perf_counter() - started}
                    report["cases"].append(entry)
                    print(json.dumps(entry), flush=True)
                for order, write_output in (("auto", "auto"), ("projected", "dense"), ("associated", "sparse")):
                    out = dense_phase_predict(model, DensePhaseCache.build(model, image, dense, PositionCache.build(model, pe)),
                                              sparse, order=order, write_output=write_output)
                    errors = {"masks": error(ref[0], out[0]), "iou": error(ref[1], out[1])}
                    entry = {"seed": seed, "grid": grid, "T": tokens, "batch": batch,
                             "dtype": str(dtype), "variant": f"dense_phase_{order}_{write_output}",
                             "dense_prompt": "shared spatial random" if spatial else "shared constant vector",
                             "model_fp32_state_sha256": metadata["model_fp32_state_sha256"],
                             "errors": errors, "binary_flips": int(((ref[0] > 0) != (out[0] > 0)).sum()),
                             "mask_shape": list(out[0].shape), "iou_shape": list(out[1].shape),
                             "passed": all(e["finite"] and e["max_abs"] <= 5e-5 for e in errors.values())}
                    report["cases"].append(entry)
                    print(json.dumps(entry), flush=True)
    # A cancellation stress case proves real equivalence is not float stability:
    # x = s*b + u*v = b + (-b + delta), with high dynamic range.
    cancellation = []
    for scale in (1., 100., 10000.):
        torch.manual_seed(41)
        b = torch.randn(1, 256) * scale
        v = -b + torch.randn_like(b) * .01
        bc, vc = b - b.mean(-1, keepdim=True), v - v.mean(-1, keepdim=True)
        native = (b + v - (b + v).mean(-1, keepdim=True)).square().mean(-1)
        moment = bc.square().mean(-1) + 2*(bc*vc).mean(-1) + vc.square().mean(-1)
        cancellation.append({"scale": scale, "direct_variance": float(native[0]),
                             "statistic_variance": float(moment[0]),
                             "statistic_finite_rstd": bool(torch.isfinite(torch.rsqrt(moment + 1e-5)).all())})
    report["cancellation_stress"] = cancellation
    report["all_random_passed"] = all(c["passed"] for c in report["cases"])
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(__file__).with_name("prototype.py"))}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_random_passed"]:
        raise SystemExit("Original numerical gate failed; diagnostic records retained")


if __name__ == "__main__":
    main()
