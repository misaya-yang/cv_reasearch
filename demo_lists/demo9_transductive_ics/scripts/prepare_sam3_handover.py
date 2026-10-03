#!/usr/bin/env python3
"""CPU-only preparation of the current PLAN A/B/C/D, no GPU or data downloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from sam3_handover_pipeline import check_paired_manifest, read_json, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--data-owner", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--suite", type=Path, default=Path("/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1"))
    p.add_argument("--sam3-assets", type=Path, default=Path("/root/autodl-tmp/sam3_preparation"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--python", default="/root/miniconda3/bin/python")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--budget-seconds", type=int, default=1800)
    a = p.parse_args()
    if a.workers != 1:
        raise SystemExit("This prepared first pipeline uses one worker; no unmeasured six-worker estimate")
    root, owner, suite, assets, out = (x.resolve() for x in (a.root, a.data_owner, a.suite, a.sam3_assets, a.out))
    if not 300 <= a.budget_seconds <= 43200:
        raise SystemExit("Finite budget required")
    if shutil.disk_usage(root).free < 5 * 1024**3:
        raise SystemExit("5GiB disk reserve required")
    weight = read_json(assets / "download_status.json")
    import_receipt = read_json(assets / "import_check.json")
    checkpoint = assets / "checkpoints/sam3.pt"
    if (weight.get("state") != "WEIGHT_VERIFIED" or
        weight.get("actual_sha256") != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" or
        checkpoint.stat().st_size != 3450062241 or
        import_receipt.get("state") != "CPU_IMPORT_CHECKED" or import_receipt.get("CUDA_initialized") is not False):
        raise SystemExit("Verified SAM3 asset and bounded CPU import receipts required")
    formula = owner / "results/decision_v1/fit/models/formula_relations.pt"
    fit_model = owner / "results/decision_v1/fit/models/pixel_relations.pt"
    if not formula.exists() or not fit_model.exists():
        raise SystemExit("Frozen formula/provenance checkpoints missing; never refit here")
    out.mkdir(parents=True, exist_ok=True)
    manifests, baselines, paired = {}, {}, {}
    for name, expected in (("dev", 241), ("confirm", 600)):
        manifests[name] = suite / f"{name}_episodes.json"
        baselines[name] = owner / f"results/decision_v1/infer_1_{name}/episodes.jsonl"
        paired[name] = check_paired_manifest(manifests[name], baselines[name])
        if paired[name]["episodes"] != expected:
            raise SystemExit(f"Unexpected {name} count")
    write_json(out / "paired_preflight.json", {"state": "COMPLETE_BASELINE_PAIRS_CHECKED", **paired})
    env_b = "/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions"
    env_a = str(assets / "python") + ":" + str(assets / "code") + ":/root/demo4_cache/env"
    common = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DEMO9_CUDA_GUARD": "1",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    prepared_b = {}
    # This validates existing constants/metadata. It opens no query mask pixels.
    for name, expected in (("dev", 241), ("confirm", 600)):
        prepared_b[name] = out / f"B_{name}_prepared.json"
        args = [a.python, str(root / "scripts/formula_pipeline.py"), "prepare", "--manifest", str(manifests[name]),
                "--models", str(formula), "--fit-models", str(fit_model), "--cache", str(owner / "cache/decision_v1"),
                "--baseline-records", str(baselines[name]), "--train-manifest", str(suite / "train_episodes.json"),
                "--out", str(prepared_b[name]), "--expected-count", str(expected), "--workers", "1"]
        if name == "dev":
            args += ["--packets", str(owner / "results/extent_v1/run/packets")]
        env = dict(os.environ, PYTHONPATH=env_b, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run(args, cwd=root, env=env, check=True, timeout=120)
    source_files = [root / x for x in (
        "scripts/prepare_sam3_handover.py", "scripts/sam3_handover_pipeline.py", "scripts/sam3_stitch.py",
        "scripts/formula_pipeline.py", "scripts/experiment_resource_guard.py", "scripts/decision_infer.py", "scripts/decision_cache.py",
        "scripts/extent_experiment.py", "scripts/analyze_extent.py", "tics/decision_heads.py",
        "tics/relations.py", "tics/extent_cut.py", "tics/native_assets.py")]
    source_files += [assets / "code/sam3/model_builder.py", assets / "code/sam3/model/sam3_image_processor.py"]
    for file in source_files:
        compile(file.read_text(), str(file), "exec")
    hashes = {str(file): sha(file) for file in source_files}
    def command(argv, env, marker, states, timeout=1620):
        return {"argv": argv, "cwd": str(root), "env": {**common, "PYTHONPATH": env},
                "timeout_seconds": timeout, "success_marker": {"path": str(marker), "states": states}}
    stages = {}
    for name, split, limit, reuse in (("A_smoke", "dev", 10, None), ("A_dev", "dev", None, "A_smoke"), ("A_confirm", "confirm", None, None)):
        target = out / name
        shared = ["--manifest", str(manifests[split]), "--out", str(target)] + (["--limit", str(limit)] if limit else [])
        gpu = [a.python, str(root / "scripts/sam3_stitch.py"), *shared, "--sam3", str(assets / "code"),
               "--checkpoint", str(checkpoint), "--checkpoint-status", str(assets / "download_status.json"),
               "--memory-fraction", ".3", "--resume"]
        if reuse:
            gpu += ["--reuse-predictions-from", str(out / reuse)]
        score = [a.python, str(root / "scripts/sam3_stitch.py"), "--score", *shared, "--resume"]
        if name == "A_smoke":
            score += ["--keep-candidates"]
        merge = [a.python, str(root / "scripts/sam3_stitch.py"), "--merge", *shared, "--foris", str(baselines[split])]
        stages[name] = {"commands": [command(gpu, env_a, target / "prediction_freeze_shard0.json", ["PREDICTIONS_FROZEN"]),
                        command(score, env_a, target / "report_shard0.json", ["COMPLETED"], 60),
                        command(merge, env_a, target / "report.json", ["COMPLETED"], 60)],
                        "completion": str(target / "report.json"), "timing_report": str(target / "report_shard0.json"),
                        "identity": {"source_hashes": hashes, "manifest_sha256": paired[split]["manifest_sha256"],
                                     "baseline_sha256": paired[split]["baseline_sha256"], "limit": limit,
                                     "weight_sha256": weight["actual_sha256"], "kind": name}}
        if name == "A_dev":
            smoke = out / "A_smoke"
            stages[name]["commands"].append(command(
                [a.python, str(root / "scripts/sam3_stitch.py"), "--cleanup-candidates", "--out", str(smoke)],
                env_a, smoke / "candidate_cleanup_after_reuse_shard0.json", ["SCORING_TEMPORARIES_REMOVED"], 60))
    for name, split, limit, reuse in (("B_smoke", "confirm", 10, None), ("B_confirm", "confirm", None, "B_smoke"), ("B_dev", "dev", None, None)):
        target = out / name
        gpu = [a.python, str(root / "scripts/formula_pipeline.py"), "formula-freeze", "--prepared", str(prepared_b[split]),
               "--out", str(target), "--workers", "1", "--resume"]
        if limit:
            gpu += ["--limit", str(limit)]
        if reuse:
            gpu += ["--reuse-from", str(out / reuse)]
        freeze_state = "SMOKE_PREDICTIONS_FROZEN" if limit else "PREDICTIONS_FROZEN"
        commands = [command(gpu, env_b, target / "freeze_report.json", [freeze_state, "COMPLETED"])]
        if not limit:
            score = [a.python, str(root / "scripts/formula_pipeline.py"), "score-frozen", "--prepared", str(prepared_b[split]), "--out", str(target)]
            commands.append(command(score, env_b, target / "report.json", ["COMPLETED"], 60))
        stages[name] = {"commands": commands, "completion": str(target / ("freeze_report.json" if limit else "report.json")),
                        "completion_states": [freeze_state] if limit else ["COMPLETED"],
                        "timing_report": str(target / "freeze_report.json"),
                        "identity": {"source_hashes": hashes, "prepared_sha256": sha(prepared_b[split]), "limit": limit,
                                     "formula_sha256": sha(formula), "workers": 1, "kind": name}}
    plan = {"state": "CPU_PREPARED", "autostart": False, "gpu_execution_authorized": False,
            "order": ["A_smoke", "B_smoke", "A_dev", "A_confirm", "B_confirm", "B_dev"], "stages": stages,
            "source_hashes": hashes, "approved_budget_seconds": a.budget_seconds,
            "budget_safety_fraction": .9, "hard_cap_seconds": int(a.budget_seconds * .9),
            "A_total_episodes": 841, "B_total_episodes": 841, "publication_goal_complete": False,
            "smoke_policy": "Both same-configuration timings before long stages; prefixes reused, B smoke opens no GT",
            "scope": "Current PLAN handover11 A/B/C/D. Other methods/tables require review, never auto-start"}
    write_json(out / "pipeline_plan.json", plan)
    made = {"path": str(out / "pipeline_status.json"), "json_equals": {"state": "COMPLETED"}}
    guard = {"platform": "autodl", "cuda_python": a.python, "stages": [{"name": "sam3_formula_finite_handover", "kind": "gpu",
        "cwd": str(root), "timeout_seconds": plan["hard_cap_seconds"],
        "argv": [a.python, str(root / "scripts/sam3_handover_pipeline.py"), "run-body", "--plan", str(out / "pipeline_plan.json"), "--out", str(out)],
        "requires": [{"path": str(out / "pipeline_plan.json"), "json_equals": {"state": "CPU_PREPARED"}},
                     {"path": str(assets / "download_status.json"), "json_equals": {"state": "WEIGHT_VERIFIED"}},
                     {"path": str(assets / "import_check.json"), "json_equals": {"state": "CPU_IMPORT_CHECKED"}}],
        "code_files": [str(file) for file in source_files], "env": {**common, "PYTHONPATH": env_b},
        "produces": [made], "success_checks": [made]}]}
    write_json(out / "guard_plan.json", guard)
    write_json(out / "preparation_status.json", {"state": "CPU_PREPARED_NOT_GPU_EXECUTED", "paired": paired,
        "stages": plan["order"], "source_hashes": hashes, "GPU_used": False, "new_training": False,
        "hard_cap_seconds": plan["hard_cap_seconds"], "unmeasured_runtime": True})
    print(json.dumps({"state": "CPU_PREPARED_NOT_GPU_EXECUTED", "out": str(out), "stages": plan["order"]}))


if __name__ == "__main__":
    main()
