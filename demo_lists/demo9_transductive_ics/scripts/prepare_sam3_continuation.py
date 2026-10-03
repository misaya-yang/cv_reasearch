#!/usr/bin/env python3
"""Prepare an annotation-free finite continuation from an already scored smoke.

Only metadata and existing compact packed predictions are checked. Original v2
prediction/scoring identities and errors remain intact. GPU/score use the exact
frozen v2 file; merge/cleanup use the canonical-shard CPU repair. No CUDA, model
construction, downloads, fitting, boot, or provider shutdown is performed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from sam3_handover_pipeline import check_paired_manifest, completed, read_json, sha, write_json


GPU_SHA = "276a53fd8c5fd1fa45ce64317b695c22feef247ee25f3bade14b7587cedbc07d"
WEIGHT_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def relocate(value, old, new):
    if isinstance(value, str): return value.replace(str(old), str(new))
    if isinstance(value, list): return [relocate(v, old, new) for v in value]
    if isinstance(value, dict): return {k: relocate(v, old, new) for k, v in value.items()}
    return value


def replace_option(argv, option, value):
    argv[argv.index(option) + 1] = str(value)


def prepare(a):
    root, previous, out = (p.resolve() for p in (a.root, a.previous, a.out))
    if previous == out or previous in out.parents: raise ValueError("Fresh independent continuation directory required")
    if out.exists() and any(out.iterdir()): raise ValueError("Preserve previous preparation; use a fresh output directory")
    if shutil.disk_usage(root).free < 5 * 1024**3: raise ValueError("5GiB data-disk reserve required")
    if not 300 <= a.budget_seconds <= 3000: raise ValueError("Explicit finite continuation budget 300..3000 required")
    if not 0 <= a.provision_reserve_seconds <= 540: raise ValueError("Provision reserve cannot exceed nine minutes")
    hard_cap = int(.9 * a.budget_seconds)
    total_cap = a.previous_guard_seconds + a.provision_reserve_seconds + hard_cap
    if a.previous_guard_seconds < 0 or total_cap > 4500:
        raise ValueError("Previous guard + provision reserve + continuation hard cap exceeds total75min")
    old_plan = read_json(previous / "pipeline_plan.json")
    if old_plan.get("state") != "CPU_PREPARED": raise ValueError("Prepared original A/B plan required")
    if old_plan["order"] != ["A_smoke", "B_smoke", "A_dev", "A_confirm", "B_confirm", "B_dev"]:
        raise ValueError("Only the finite existing A/B stage order can continue")
    gpu_file = root / "scripts/sam3_stitch_gpu_v2.py"
    cpu_file = root / "scripts/sam3_stitch.py"
    old_gpu_path = str(cpu_file)
    if sha(gpu_file) != GPU_SHA or old_plan["source_hashes"].get(old_gpu_path) != GPU_SHA:
        raise ValueError("Frozen GPU file must be byte-identical to the real v2 predictor/scorer")
    hashes = {path: digest for path, digest in old_plan["source_hashes"].items() if path != old_gpu_path}
    for path, digest in hashes.items():
        if sha(path) != digest: raise ValueError("Unchanged v2 dependency drift: " + path)
    hashes.update({str(gpu_file): GPU_SHA, str(cpu_file): sha(cpu_file), str(Path(__file__).resolve()): sha(__file__)})
    for path in hashes: compile(Path(path).read_text(), path, "exec")
    smoke = previous / "A_smoke"
    report_path = smoke / "report_shard0.json"
    freeze_path = smoke / "prediction_freeze_shard0.json"
    merged_path = smoke / "report.json"
    predictions = smoke / "predictions_shard0.jsonl"
    scored = smoke / "episodes_shard0.jsonl"
    report, freeze, merged = map(read_json, (report_path, freeze_path, merged_path))
    if (report.get("state") != "COMPLETED" or report.get("episodes") != 10 or report.get("scored_episodes") != 10 or
        report.get("source_sha256") != GPU_SHA or report.get("checkpoint_sha256") != WEIGHT_SHA or
        report.get("scored_jsonl_sha256") != sha(scored) or report.get("prediction_freeze_sha256") != sha(freeze_path)):
        raise ValueError("Original complete/scored smoke identity must remain intact")
    if (freeze.get("state") != "PREDICTIONS_FROZEN" or freeze.get("episodes") != 10 or
        freeze.get("query_annotation_opened") is not False or freeze.get("source_sha256") != GPU_SHA or
        freeze.get("protocol") != report.get("protocol") or freeze.get("predictions_jsonl_sha256") != sha(predictions)):
        raise ValueError("Original no-query-GT prediction freeze required")
    if (merged.get("state") != "COMPLETED" or merged.get("episodes") != 10 or
        merged.get("prediction_source_sha256") != [GPU_SHA] or merged.get("aggregation_source_sha256") != sha(cpu_file) or
        merged.get("input_scored_jsonl_sha256") != {scored.name: sha(scored)}):
        raise ValueError("Canonical CPU merge must report original GPU and repaired aggregation provenance")
    assets_checked = 0
    for asset in freeze["prediction_files"]:
        path = smoke / asset["path"]
        if path.stat().st_size != asset["bytes"] or sha(path) != asset["sha256"]:
            raise ValueError("Unchanged packed smoke masks/candidates required for prefix reuse")
        assets_checked += 1
    n, inference, load = int(report["new_inferred_cases"]), float(report["inference_s"]), float(report["model_load_s"])
    if n != 10 or inference <= 0 or load <= 0:
        raise ValueError("Use real v2 case/load timing; a metadata reuse is not zero-cost inference")
    plan = relocate(deepcopy(old_plan), previous, out)
    paired = {}
    for name, expected in (("dev", 241), ("confirm", 600)):
        stage = old_plan["stages"]["A_" + name]
        argv = stage["commands"][0]["argv"]
        manifest = Path(argv[argv.index("--manifest") + 1])
        merge_args = stage["commands"][2]["argv"]
        baseline = Path(merge_args[merge_args.index("--foris") + 1])
        paired[name] = check_paired_manifest(manifest, baseline)
        if paired[name]["episodes"] != expected or paired[name]["manifest_sha256"] != stage["identity"]["manifest_sha256"]:
            raise ValueError("Original complete paired cohort changed")
        if name == "dev" and freeze["manifest_sha256"] != paired[name]["manifest_sha256"]:
            raise ValueError("Smoke no longer belongs to the exact DEV cohort")
    out.mkdir(parents=True)
    # B's accepted metadata/constants are reused byte-for-byte; there is no refit.
    for split in ("dev", "confirm"):
        shutil.copyfile(previous / ("B_" + split + "_prepared.json"), out / ("B_" + split + "_prepared.json"))
    for name, stage in plan["stages"].items():
        stage["identity"]["source_hashes"] = hashes
        stage["identity"]["GPU_source_sha256"] = GPU_SHA
        stage["identity"]["CPU_aggregation_source_sha256"] = sha(cpu_file)
        for cmd in stage["commands"]:
            argv = cmd["argv"]
            if argv[1] == str(cpu_file):
                argv[1] = str(cpu_file if "--merge" in argv or "--cleanup-candidates" in argv else gpu_file)
            if name == "A_dev" and "--reuse-predictions-from" in argv:
                replace_option(argv, "--reuse-predictions-from", smoke)
            if name == "A_dev" and "--cleanup-candidates" in argv:
                replace_option(argv, "--out", smoke)
                cmd["success_marker"]["path"] = str(smoke / "candidate_cleanup_after_reuse_shard0.json")
    alias = out / "A_smoke"; alias.mkdir()
    shutil.copyfile(merged_path, alias / "report.json")
    timing = {**report, "new_predictions": n, "prediction_elapsed_s": load + inference,
              "timing_source_report": str(report_path), "timing_source_report_sha256": sha(report_path),
              "derivation": "original model_load_s + inference_s; excludes CPU scoring/merge; not a speed benchmark",
              "reused_prediction_cases": n, "new_GPU_calls_in_continuation_smoke": 0}
    write_json(alias / "timing_from_v2.json", timing)
    reused = plan["stages"]["A_smoke"]
    reused["commands"] = []
    reused["timing_report"] = str(alias / "timing_from_v2.json")
    reused["identity"]["kind"] = "REUSED_CANONICAL_SCORED_V2_SMOKE"
    reused["identity"]["original_GPU_stage_identity"] = old_plan["stages"]["A_smoke"]["identity"]
    reused["identity"]["original_prediction_freeze_sha256"] = sha(freeze_path)
    reused["identity"]["canonical_merge_sha256"] = sha(merged_path)
    write_json(alias / "pipeline_identity.json", reused["identity"])
    write_json(alias / "pipeline_completion.json", {"stage_identity": reused["identity"],
                                                   "report_sha256": sha(alias / "report.json")})
    if not completed(reused) or reused["commands"]:
        raise RuntimeError("Reused complete smoke must pass the actual controller contract with zero commands")
    plan.update(source_hashes=hashes, approved_budget_seconds=a.budget_seconds, budget_safety_fraction=.9,
                hard_cap_seconds=hard_cap, GPU_source_sha256=GPU_SHA, CPU_aggregation_source_sha256=sha(cpu_file),
                previous_guard_seconds=a.previous_guard_seconds, provision_reserve_seconds=a.provision_reserve_seconds,
                cumulative_reserved_hard_cap_seconds=total_cap,
                scope="Reuse canonical scored v2 smoke; unchanged finite A/B continuation only, no F4 or fresh160",
                smoke_policy="A10 original predictions/scoring and actual timing reused; new B10 precedes measured-cost gate",
                autostart=False, gpu_execution_authorized=False)
    write_json(out / "pipeline_plan.json", plan)
    originals = [previous / "pipeline_plan.json", report_path, freeze_path, merged_path, predictions, scored,
                 out / "B_dev_prepared.json", out / "B_confirm_prepared.json", alias / "timing_from_v2.json"]
    checks = [{"path": str(path), "sha256": sha(path)} for path in originals]
    final = {"path": str(out / "pipeline_status.json"), "json_equals": {"state": "COMPLETED"}}
    env = {"DEMO9_CUDA_GUARD": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "PYTHONPATH": "/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions"}
    guard = {"platform": "autodl", "cuda_python": a.python, "stages": [{
        "name": "reused_SAM3_smoke_finite_A_B_continuation", "kind": "gpu", "cwd": str(root),
        "timeout_seconds": hard_cap,
        "argv": [a.python, str(root / "scripts/sam3_handover_pipeline.py"), "run-body",
                 "--plan", str(out / "pipeline_plan.json"), "--out", str(out)],
        "requires": [{"path": str(out / "pipeline_plan.json"), "json_equals": {"state": "CPU_PREPARED"}}],
        "cpu_artifacts": checks, "code_files": list(hashes), "env": env,
        "produces": [final], "success_checks": [final]}]}
    write_json(out / "guard_plan.json", guard)
    receipt = {"state": "CPU_CONTINUATION_PREPARED_NOT_GPU_EXECUTED", "paired": paired,
               "GPU_source_sha256": GPU_SHA, "CPU_aggregation_source_sha256": sha(cpu_file),
               "old_smoke_prediction_cases_reused": 10, "new_smoke_GPU_calls": 0,
               "original_packed_assets_checked": assets_checked, "original_model_load_s": load,
               "original_per_case_inference_s": inference/n, "original_v2_identity_changed": False,
               "previous_errors_preserved": True, "budget_seconds": a.budget_seconds, "hard_cap_seconds": hard_cap,
               "cumulative_reserved_hard_cap_seconds": total_cap, "source_hashes": hashes,
               "query_GT_pixels_read": False, "CUDA_initialized": False, "new_training": False,
               "F4_restarted": False, "stage_order": plan["order"], "not_a_task_quality_result": True}
    write_json(out / "preparation_status.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("state", "old_smoke_prediction_cases_reused", "new_smoke_GPU_calls",
          "original_per_case_inference_s", "hard_cap_seconds", "cumulative_reserved_hard_cap_seconds", "CUDA_initialized")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--previous", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--budget-seconds", type=int, default=1800)
    parser.add_argument("--previous-guard-seconds", type=float, required=True)
    parser.add_argument("--provision-reserve-seconds", type=float, default=540)
    prepare(parser.parse_args())


if __name__ == "__main__": main()
