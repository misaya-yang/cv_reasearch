#!/usr/bin/env python3
"""One finite outer-guard batch: F4 DEV, repaired SAM3/formula, conditional F4.

Preparation and contract checks use only the standard library. This wrapper
never boots a GPU, calls shutdown, fits a model, or changes accepted sources.
The resource guard owns its process group and the provider lifecycle.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ARMS = ("native", "anchor_only", "full001", "perFG002", "matchedGlobal", "Part4cal", "querycore")
NAIVE = {"anchor_only": ["querycore"], "full001": ["anchor_only", "querycore"],
         "perFG002": ["full001", "matchedGlobal", "querycore"], "Part4cal": ["querycore"]}
# The simplest surviving intervention wins; do not re-select on confirmation.
PRIMARY_ORDER = ("anchor_only", "Part4cal", "full001", "perFG002")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def key(row):
    return tuple(int(row[name]) for name in ("fold", "e", "c"))


def require_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected: raise RuntimeError("Frozen source/input changed: " + path)


def combine_hashes(*collections):
    result = {}
    for collection in collections:
        for path, value in collection.items():
            if path in result and result[path] != value: raise ValueError("Conflicting source proof: " + path)
            result[path] = value
    return result


def check_f4_report(report, manifest, manifest_sha, count):
    """An eight-case COMPLETED report can never certify 241 cases."""
    rows = report.get("records", [])
    if (report.get("state") != "COMPLETED" or report.get("episodes") != count or
        report.get("expected_episodes") != count or len(rows) != count or
        report.get("manifest_sha256") != manifest_sha):
        raise RuntimeError("F4 count/state/manifest completion gate failed")
    if [key(item["row"]) for item in rows] != [key(row) for row in manifest["episodes"][:count]]:
        raise RuntimeError("F4 completed cases are not the frozen requested prefix")
    for item in rows:
        if (item.get("predictions_frozen_before_query_GT") is not True or
            item.get("full_native_exact") is not True or
            item.get("actual_encoder_B2_calls") != 1 or item.get("extra_encoder_B2_calls") != 0):
            raise RuntimeError("F4 native/freeze/one-encoding contract failed")
        if manifest.get("design_DEV") and (item.get("baseline_original_IU_exact") is not True or
                                         item.get("baseline_working_mask_bit_exact") is not True):
            raise RuntimeError("F4 current DEV native gate failed")
    return rows


def comparison_pass(analysis, arm, base, minimum=0.0, positive_folds=0):
    pair = analysis.get("comparisons", {}).get(arm + "__minus__" + base)
    if not pair: return False
    folds = pair.get("folds", {})
    return (float(pair["delta_pp"]) >= minimum and
            float(pair["photo_connected_paired_CI95"][0]) > 0 and
            set(folds) == {"0", "1", "2", "3"} and
            sum(float(value) > 0 for value in folds.values()) >= positive_folds)


def select_primary(analysis, dev_sha, analysis_sha, fresh_sha, hashes):
    if (analysis.get("state") != "CPU_F4_SEVEN_ANALYZED" or analysis.get("episodes") != 241 or
        analysis.get("expected_episodes") != 241 or analysis.get("all_four_folds_present") is not True):
        raise RuntimeError("Full all-fold F4 DEV analysis required for selection")
    eligible = []
    for arm in PRIMARY_ORDER:
        if (comparison_pass(analysis, arm, "native", minimum=2.0, positive_folds=3) and
            all(comparison_pass(analysis, arm, base) for base in NAIVE[arm])):
            eligible.append(arm)
    primary = eligible[0] if eligible else None
    return {"state": "FROZEN_SINGLE_PRIMARY" if primary else "REGISTERED_NO_EXPANSION",
            "primary": primary, "naive": NAIVE.get(primary, []), "eligible_DEV_arms": eligible,
            "dev_analysis_sha256": analysis_sha, "dev_manifest_sha256": dev_sha,
            "confirmation_manifest_sha256": fresh_sha, "source_hashes": hashes,
            "selection_rule_version": "simplest_v1_gain2_CIlower0_threefold_and_matched_naive",
            "confirmation_may_select_primary": False, "publication_goal_complete": False,
            "reason": "Frozen simplest supported intervention" if primary else
                      "No new arm passes the registered task/paired/naive-control gate; no fresh expansion"}


def prepare(a):
    root, out = a.root.resolve(), a.out.resolve()
    if not 300 <= a.budget_seconds <= 4500: raise ValueError("Finite total cap must be 300..4500 seconds")
    if shutil.disk_usage(root).free < 5 * 1024**3: raise ValueError("5GiB reserve required")
    dev, cpu, ab = read(a.dev_manifest), read(a.cpu_receipt), read(a.ab_plan)
    if (dev.get("schema") != "f4_seven_arms_v1" or dev.get("design_DEV") is not True or
        len(dev.get("episodes", [])) != 241 or len(dev.get("baseline_rows", {})) != 241 or
        [row["fold"] for row in dev["episodes"][:8]].count(0) != 2 or
        any(sum(row["fold"] == fold for row in dev["episodes"][:8]) != 2 for fold in range(4))):
        raise ValueError("Complete 241-case DEV with two smoke cases per fold required")
    if (cpu.get("state") != "CPU_F4_SEVEN_SOURCE_PASSED" or cpu.get("cases") != 10 or
        cpu.get("CUDA_initialized") is not False or cpu.get("no_query_GT_in_decisions") is not True):
        raise ValueError("Accepted real-source ten-case CPU proof required")
    if (ab.get("state") != "CPU_PREPARED" or
        ab.get("order") != ["A_smoke", "B_smoke", "A_dev", "A_confirm", "B_confirm", "B_dev"] or
        int(ab.get("approved_budget_seconds", 0)) > 1800 or
        int(ab.get("hard_cap_seconds", 0)) > 1620 or
        ab.get("budget_safety_fraction") != .9):
        raise ValueError("A/B must retain its frozen 30-minute budget and .9 safety margin")
    proof = read(a.cpu_check_receipt)
    if (proof.get("state") != "CPU_FINITE_BATCH_CONTRACTS_PASSED" or
        proof.get("source_sha256") != sha(__file__) or proof.get("GPU_used") is not False):
        raise ValueError("This wrapper needs its own server CPU-only contract receipt")
    sources = combine_hashes(dev["source_hashes"], cpu["source_hashes"], ab["source_hashes"],
                             {str(Path(__file__).resolve()): sha(__file__)})
    require_hashes(sources)
    for item in dev["assets"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError("Prepared F4 asset changed: " + item["path"])
    inputs = {str(path.resolve()): sha(path) for path in
              (a.dev_manifest, a.cpu_receipt, a.ab_plan, a.cpu_check_receipt)}
    confirmation = None
    if a.fresh_manifest or a.fresh_helper:
        if not (a.fresh_manifest and a.fresh_helper): raise ValueError("Fresh helper and manifest must be provided together")
        fresh = read(a.fresh_manifest)
        if (fresh.get("schema") != "f4_seven_arms_v1" or fresh.get("design_DEV") is not False or
            fresh.get("scope") != "registered_photo_disjoint_confirmation" or
            len(fresh.get("episodes", [])) != 160 or fresh.get("baseline_rows")):
            raise ValueError("Transparent registered-scope 160-case confirmation contract required")
        if fresh.get("query_GT_previously_unopened") is True:
            raise ValueError("Do not turn registered exposure checks into a global unseen-GT claim")
        require_hashes(fresh["source_hashes"])
        sources = combine_hashes(sources, fresh["source_hashes"],
                                 {str(a.fresh_helper.resolve()): sha(a.fresh_helper)})
        inputs[str(a.fresh_manifest.resolve())] = sha(a.fresh_manifest)
        if fresh.get("unseen_receipt"):
            audit = Path(fresh["unseen_receipt"]).resolve()
            if sha(audit) != fresh.get("unseen_receipt_sha256"): raise ValueError("Exposure audit drift")
            inputs[str(audit)] = sha(audit)
        confirmation = {"manifest": str(a.fresh_manifest.resolve()), "manifest_sha256": sha(a.fresh_manifest),
                        "helper": str(a.fresh_helper.resolve()), "out": str(a.fresh_out.resolve())}
    plan = {"state": "CPU_PREPARED", "root": str(root), "python": a.python,
            "source_hashes": sources, "input_hashes": inputs,
            "hard_cap_seconds": a.budget_seconds,
            "dev_manifest": str(a.dev_manifest.resolve()), "cpu_receipt": str(a.cpu_receipt.resolve()),
            "dev_out": str(a.dev_out.resolve()), "selection": str(a.selection.resolve()),
            "ab_plan": str(a.ab_plan.resolve()), "ab_out": str(a.ab_plan.resolve().parent),
            "confirmation": confirmation,
            "scope": "F4 DEV seven arms, independent A/B budget, registered conditional single-primary confirmation",
            "GPU_used": False, "autoboot": False, "shutdown_owner": "outer experiment_resource_guard only",
            "pending_confirmation": not bool(confirmation), "publication_goal_complete": False}
    out.mkdir(parents=True, exist_ok=True); write(out / "batch_plan.json", plan)
    final = {"path": str(out / "batch_status.json"), "json_equals": {"state": "COMPLETED"}}
    guard = {"platform": "autodl", "cuda_python": a.python, "stages": [{
        "name": "F4_then_repaired_SAM3_finite_batch", "kind": "gpu", "cwd": str(root),
        "timeout_seconds": a.budget_seconds,
        "argv": [a.python, str(Path(__file__).resolve()), "--run-body", "--plan", str(out / "batch_plan.json"), "--out", str(out)],
        "env": {"PYTHONPATH": "/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions",
                "DEMO9_CUDA_GUARD": "1", "DEMO9_F4_ROOT_LAUNCH": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
        "requires": [{"path": str(out / "batch_plan.json"), "json_equals": {"state": "CPU_PREPARED"}}],
        "cpu_artifacts": [{"path": path, "sha256": digest} for path, digest in inputs.items()],
        "code_files": list(sources), "produces": [final], "success_checks": [final]}]}
    write(out / "guard_plan.json", guard)
    print(json.dumps({"state": plan["state"], "GPU_used": False,
                      "hard_cap_seconds": a.budget_seconds, "confirmation_prepared": bool(confirmation)}))


def run_body(a):
    if os.environ.get("DEMO9_CUDA_GUARD") != "1": raise RuntimeError("Only the outer resource guard may launch")
    plan = read(a.plan); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if plan.get("state") != "CPU_PREPARED": raise RuntimeError("Prepared finite batch required")
    with (out / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic(); events = []; background = None; background_log = None
        def remaining(): return plan["hard_cap_seconds"] - (time.monotonic() - started)
        def emit(state, **fields):
            event = {"state": state, "elapsed_seconds": time.monotonic()-started, **fields}; events.append(event)
            write(out / "batch_status.json", {"state": state, "events": events, "plan_sha256": sha(a.plan),
                                             "publication_goal_complete": False})
            print(json.dumps(event), flush=True)
        def command(argv, name, maximum=None):
            seconds = remaining() if maximum is None else min(maximum, remaining())
            if seconds <= 0: raise subprocess.TimeoutExpired(argv, 0)
            emit("RUNNING", stage=name)
            with (out / (name + ".log")).open("a") as log:
                proc = subprocess.Popen(argv, cwd=plan["root"], stdout=log, stderr=log)
                try: rc = proc.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
                    raise
            if rc: raise RuntimeError(name + " returned " + str(rc))
        def f4_args():
            return [plan["python"], str(Path(plan["root"]) / "scripts/f4_experiment.py"), "--run", "--allow-gpu",
                    "--cpu-receipt", plan["cpu_receipt"], "--manifest", plan["dev_manifest"],
                    "--out", plan["dev_out"], "--resume", "--budget-seconds", str(max(1, int(remaining())))]
        try:
            require_hashes(plan["source_hashes"]); require_hashes(plan["input_hashes"])
            dev = read(plan["dev_manifest"]); man_sha = sha(plan["dev_manifest"])
            report_path = Path(plan["dev_out"]) / "report.json"; smoke = out / "F4_smoke_immutable.json"
            if smoke.exists():
                proof = read(smoke)
                if proof["manifest_sha256"] != man_sha or proof["source_hashes"] != dev["source_hashes"]:
                    raise RuntimeError("Immutable F4 smoke proof identity changed")
                require_hashes(proof["frozen_case_hashes"])
                emit("REUSED_SMOKE8", cases=8)
            else:
                command(f4_args() + ["--max-pairs", "8"], "F4_smoke8")
                records = check_f4_report(read(report_path), dev, man_sha, 8)
                frozen_hashes = {}
                for item in records:
                    for path in (item["prediction_path"], item["frozen_receipt_path"]): frozen_hashes[path] = sha(path)
                write(smoke, {"state": "F4_SMOKE8_NATIVE_EXACT", "episodes": 8, "manifest_sha256": man_sha,
                              "source_hashes": dev["source_hashes"], "frozen_case_hashes": frozen_hashes,
                              "captured_report_sha256": sha(report_path), "prefix_keys": [key(r["row"]) for r in records]})
                emit("SMOKE8_COMPLETED_NATIVE_EXACT", cases=8)
            complete = False
            if report_path.exists():
                try: check_f4_report(read(report_path), dev, man_sha, 241); complete = True
                except RuntimeError: pass
            if not complete: command(f4_args(), "F4_DEV241")
            records = check_f4_report(read(report_path), dev, man_sha, 241)
            write(out / "F4_DEV241_completion.json", {"state": "F4_COMPLETE241_NATIVE_EXACT", "episodes": len(records),
                                                       "report_sha256": sha(report_path), "manifest_sha256": man_sha})
            emit("F4_DEV_COMPLETED", cases=241)
            # CPU bootstrap runs while the already-prepared SAM3 job uses CUDA.
            analyze = [plan["python"], str(Path(plan["root"]) / "scripts/f4_experiment.py"),
                       "--analyze", "--manifest", plan["dev_manifest"], "--out", plan["dev_out"]]
            background_log = (out / "F4_DEV_analysis.log").open("a")
            background = subprocess.Popen(analyze, cwd=plan["root"], stdout=background_log, stderr=background_log)
            emit("F4_CPU_ANALYSIS_BACKGROUND", pid=background.pid, GPU_next="SAM3_A_B")
            # No nested guard and no raised A/B safety fraction or budget.
            command([plan["python"], str(Path(plan["root"]) / "scripts/sam3_handover_pipeline.py"),
                     "run-body", "--plan", plan["ab_plan"], "--out", plan["ab_out"]], "SAM3_A_B")
            try: analysis_rc = background.wait(timeout=max(.01, min(60, remaining())))
            except subprocess.TimeoutExpired:
                background.terminate(); analysis_rc = None
            if analysis_rc != 0:
                write(out / "F4_analysis_failure.json", {"state": "CPU_ANALYSIS_FAILED_OR_TIMED_OUT", "return_code": analysis_rc,
                                                         "A_B_healthy_stages_not_interrupted": True})
                emit("F4_ANALYSIS_HOLD", return_code=analysis_rc)
            else:
                analysis_path = Path(plan["dev_out"]) / "analysis.json"
                confirmation = plan.get("confirmation")
                selection = select_primary(read(analysis_path), man_sha, sha(analysis_path),
                                           confirmation["manifest_sha256"] if confirmation else None, dev["source_hashes"])
                selection["dev_analysis_path"] = str(analysis_path.resolve())
                selection_path = Path(plan["selection"])
                if selection_path.exists() and read(selection_path) != selection: raise RuntimeError("Frozen primary selection drift")
                write(selection_path, selection)
                emit(selection["state"], primary=selection["primary"], selection_sha256=sha(selection_path))
                if selection["primary"] and confirmation:
                    # The registered helper checks the single-primary freeze. Its
                    # output is evaluated against that primary, never re-selected.
                    command([plan["python"], confirmation["helper"], "--run", "--allow-gpu",
                             "--manifest", confirmation["manifest"], "--selection", str(selection_path),
                             "--cpu-receipt", plan["cpu_receipt"], "--out", confirmation["out"],
                             "--budget-seconds", str(max(1, int(remaining()))), "--resume"], "F4_registered_confirmation160")
                elif selection["primary"]:
                    write(out / "confirmation_hold.json", {"state": "HOLD_REGISTERED_CONFIRMATION_NOT_PREPARED",
                                                            "primary": selection["primary"], "no_fake_unseen_flags": True})
                    emit("HOLD_REGISTERED_CONFIRMATION_NOT_PREPARED", primary=selection["primary"])
            emit("COMPLETED", automatic_new_stages=False, shutdown_owner="outer_guard",
                 confirmation_status="Conditional registered stop or frozen single-primary result; see selection/hold artifacts")
            return 0
        except BaseException as exc:
            emit("RESOURCE_INCOMPLETE" if isinstance(exc, subprocess.TimeoutExpired) else "BATCH_ERROR",
                 error_type=type(exc).__name__, detail=str(exc), failed_artifacts_preserved=True)
            return 2
        finally:
            if background is not None and background.poll() is None:
                background.terminate()
                try: background.wait(timeout=5)
                except subprocess.TimeoutExpired: background.kill(); background.wait(timeout=5)
            if background_log is not None: background_log.close()


def cpu_check(a):
    """Scheduling/completion arithmetic only; no torch, labels, CUDA or shutdown."""
    rows = [{"fold": i % 4, "e": i, "c": i % 4} for i in range(241)]
    recs = [{"row": row, "predictions_frozen_before_query_GT": True, "full_native_exact": True,
             "baseline_original_IU_exact": True, "baseline_working_mask_bit_exact": True,
             "actual_encoder_B2_calls": 1, "extra_encoder_B2_calls": 0} for row in rows[:8]]
    report = {"state": "COMPLETED", "episodes": 8, "expected_episodes": 8, "records": recs, "manifest_sha256": "fixture"}
    man = {"episodes": rows, "design_DEV": True}
    assert len(check_f4_report(report, man, "fixture", 8)) == 8
    try: check_f4_report(report, man, "fixture", 241); raise AssertionError("Smoke became full completion")
    except RuntimeError: pass
    report["records"][0]["baseline_working_mask_bit_exact"] = False
    try: check_f4_report(report, man, "fixture", 8); raise AssertionError("Native drift accepted")
    except RuntimeError: pass
    comparisons = {arm + "__minus__" + base: {"delta_pp": 3, "photo_connected_paired_CI95": [1, 4],
                                            "folds": {str(f): 1 for f in range(4)}}
                   for arm in PRIMARY_ORDER for base in ["native", *NAIVE[arm]]}
    analysis = {"state": "CPU_F4_SEVEN_ANALYZED", "episodes": 241, "expected_episodes": 241,
                "all_four_folds_present": True, "comparisons": comparisons}
    assert select_primary(analysis, "DEV", "ANALYSIS", "FRESH", {})["primary"] == "anchor_only"
    comparisons["anchor_only__minus__querycore"]["photo_connected_paired_CI95"][0] = -.1
    assert select_primary(analysis, "DEV", "ANALYSIS", "FRESH", {})["primary"] == "Part4cal"
    for arm in PRIMARY_ORDER: comparisons[arm + "__minus__native"]["delta_pp"] = 1.99
    assert select_primary(analysis, "DEV", "ANALYSIS", "FRESH", {})["state"] == "REGISTERED_NO_EXPANSION"
    with tempfile.TemporaryDirectory(prefix="f4_batch_cpu_") as folder:
        path = Path(folder) / "immutable.json"; write(path, {"state": "F4_SMOKE8_NATIVE_EXACT"})
        digest = sha(path); require_hashes({str(path): digest})
        write(path, {"state": "changed"})
        try: require_hashes({str(path): digest}); raise AssertionError("Mutable proof accepted")
        except RuntimeError: pass
    result = {"state": "CPU_FINITE_BATCH_CONTRACTS_PASSED", "checks": 7, "GPU_used": False,
              "source_sha256": sha(__file__), "imports_torch": False, "shutdown_called": False,
              "not_a_task_quality_result": True}
    write(a.out, result); print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true"); mode.add_argument("--run-body", action="store_true")
    mode.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--plan", type=Path)
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--budget-seconds", type=int, default=4500)
    parser.add_argument("--dev-manifest", type=Path); parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--dev-out", type=Path); parser.add_argument("--ab-plan", type=Path)
    parser.add_argument("--cpu-check-receipt", type=Path); parser.add_argument("--selection", type=Path)
    parser.add_argument("--fresh-manifest", type=Path); parser.add_argument("--fresh-helper", type=Path)
    parser.add_argument("--fresh-out", type=Path)
    args = parser.parse_args()
    if args.cpu_check: cpu_check(args); return 0
    if args.run_body:
        if args.plan is None: parser.error("--run-body requires --plan")
        return run_body(args)
    required = (args.dev_manifest, args.cpu_receipt, args.dev_out, args.ab_plan, args.cpu_check_receipt, args.selection)
    if any(value is None for value in required): parser.error("--prepare needs DEV, CPU proof, A/B plan, wrapper CPU proof and selection paths")
    if args.fresh_manifest and args.fresh_out is None: parser.error("Fresh output path required")
    prepare(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
