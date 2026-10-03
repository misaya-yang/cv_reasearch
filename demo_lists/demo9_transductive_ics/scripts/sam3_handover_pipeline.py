#!/usr/bin/env python3
"""Finite A/B handover; default preparation never probes a GPU or powers off.

The active PLAN handover specifies SAM3 visual/text diagnostics, the fixed
supervised formula, then a written decision. Completed/retired directions are
not executable stages. Runtime is entered only by experiment_resource_guard.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def photo_uid(value):
    name = Path(str(value)).stem
    hit = re.search(r"(\d+)$", name)
    return str(int(hit.group(1))) if hit else name


def key(row):
    return tuple(int(row[x]) for x in ("fold", "e", "c"))


def check_paired_manifest(manifest, baseline_file):
    """No target PNG pixels: verify complete baseline coverage and both photos."""
    man = read_json(manifest)
    if man.get("state") != "PREPARED":
        raise ValueError("Manifest is not PREPARED: " + str(manifest))
    rows = man["episodes"]
    expected = {}
    for row in rows:
        k = key(row)
        if k in expected:
            raise ValueError("Duplicate manifest key: " + str(k))
        expected[k] = row
    base = {}
    with Path(baseline_file).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            k = key(row)
            if k in base:
                raise ValueError("Duplicate baseline key: " + str(k))
            base[k] = row
    missing = set(expected) - set(base)
    if missing:
        raise ValueError("Incomplete FoRIS baseline: " + str(sorted(missing)[:3]))
    for k, row in expected.items():
        for role in ("support", "query"):
            if role not in base[k] or photo_uid(row[role]) != photo_uid(base[k][role]):
                raise ValueError("Paired photo mismatch: " + str((k, role)))
    return {"episodes": len(rows), "baseline_coverage": len(expected),
            "manifest_sha256": sha(manifest), "baseline_sha256": sha(baseline_file),
            "target_GT_pixels_read": False,
            "exposure": "Previously scored DEV/confirmation cohort; not fresh independent confirmation"}


def decision(a_report, b_report, success_iou=0.5):
    """PLAN C only writes next decisions. It cannot launch a new experiment."""
    paired = a_report["against_foris"]
    gain = float(paired["gain"])
    lo, hi = map(float, paired["ci95"])
    n = int(paired["episodes"])
    if n != int(a_report["episodes"]) or n <= 0:
        raise ValueError("A needs the complete paired cohort")
    counts = paired["outcome_counts"]
    exclusive = int(counts["only_foris"]) + int(counts["only_sam3"])
    if sum(int(counts[k]) for k in ("only_foris", "only_sam3", "both", "neither")) != n:
        raise ValueError("Outcome counts do not cover the cohort")
    if gain >= 3.0 and lo > 0:
        a_state = "SAM3_STRONGER_ON_PAIRED_EXPOSED_COHORT"
        next_a = "Review causal perfect-input ledger; proposal selection is not a removable-error proof"
    elif gain >= 3.0:
        a_state = "SAM3_GAIN_UNCERTAIN_REVIEW"
        next_a = "Paired interval crosses zero; no host or method claim is established"
    elif gain < -3.0:
        a_state = "SAM3_BELOW_FORIS_PROTOCOL_REVIEW"
        next_a = "Check the legal component-box adaptation against the public instance-box protocol"
    elif exclusive / n > 0.25:
        a_state = "COMPLEMENTARY_FAILURES_REVIEW"
        next_a = "A new legal cross-host prompting experiment needs its own frozen card"
    else:
        a_state = "SAM3_NO_MATERIAL_ADVANTAGE_ON_THIS_COHORT"
        next_a = "Retain DINO evidence; no automatic main-table or new-method expansion"
    row = b_report["rows"]["readout"]
    b_gain = float(row["gain"])
    b_folds = [float(v) for v in row["per_fold"]]
    keep = b_gain >= 1.5 and len(b_folds) == 4 and all(v > 0 for v in b_folds)
    return {"state": "DECISION_WRITTEN_FOR_USER_REVIEW", "A": a_state,
            "A_gain_pp": gain, "A_ci95_pp": [lo, hi], "A_next": next_a,
            "exclusive_success_fraction_all_episodes": exclusive / n,
            "success_definition": f"original-resolution IoU > {success_iou}",
            "B": "RETAIN_SUPERVISED_FIXED_COMPONENT" if keep else "DROP_THIS_FIXED_FORMULA",
            "B_gain_pp": b_gain, "B_per_fold_pp": b_folds,
            "B_ci95_pp": row["ci95"],
            "B_lose_more_than_10": row.get("lose_more_than_10"),
            "B_provenance": "Constants averaged from four supervised fold models; pooled-class calibration, not training-free or class-held-out FSS",
            "automatic_new_stages": False, "publication_goal_complete": False}


def estimate(smoke_a, smoke_b, total_a, total_b):
    """Measured same-worker smoke only. An estimate is not a timing claim."""
    values = []
    for report, count in ((smoke_a, total_a), (smoke_b, total_b)):
        n = int(report.get("new_predictions", report.get("episodes", 0)))
        elapsed = float(report.get("prediction_elapsed_s", report.get("elapsed_s", 0)))
        load = float(report.get("model_load_s", 0))
        if n <= 0 or elapsed <= 0:
            raise ValueError("Both same-configuration smoke timings are required")
        values.append(max(0.0, elapsed - load) / n * max(0, count - n) + 2 * load)
    return {"remaining_estimate_seconds": sum(values), "A_remaining_s": values[0],
            "B_remaining_s": values[1], "kind": "Scheduling estimate from smoke, not an exclusive speed result",
            "same_worker_count_required": True}


class Halt(RuntimeError):
    def __init__(self, state, detail):
        self.state, self.detail = state, detail
        super().__init__(state + ": " + detail)


def run_command(command, log, timeout):
    env = os.environ.copy()
    env.update(command.get("env", {}))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    # Inherit the outer guard's process group; it can stop only its own workers.
    with Path(log).open("a") as stream:
        return subprocess.run(command["argv"], cwd=command["cwd"], env=env,
                              stdout=stream, stderr=stream, timeout=timeout).returncode


def completed(stage):
    path = Path(stage["completion"])
    receipt = path.parent / "pipeline_completion.json"
    if not path.exists() or not receipt.exists():
        return False
    data = read_json(receipt)
    return (read_json(path).get("state") in stage.get("completion_states", ["COMPLETED"])
            and data.get("stage_identity") == stage["identity"]
            and data.get("report_sha256") == sha(path))


def bind_stage_identity(stage):
    parent = Path(stage["completion"]).parent
    path = parent / "pipeline_identity.json"
    if path.exists():
        if read_json(path) != stage["identity"]:
            raise Halt("HOLD_STAGE_IDENTITY_DRIFT", str(parent))
    else:
        if parent.exists() and any(parent.iterdir()):
            raise Halt("HOLD_UNOWNED_STAGE_OUTPUTS", str(parent))
        write_json(path, stage["identity"])


def run_body(plan_path, out):
    if os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("Runtime must be owned by experiment_resource_guard")
    plan = read_json(plan_path)
    if plan.get("state") != "CPU_PREPARED":
        raise SystemExit("Prepared contract required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This pipeline already has an owner")
        start = time.monotonic()
        events = []
        def emit(state, **extra):
            events.append({"state": state, "elapsed_s": time.monotonic() - start, **extra})
            write_json(out / "pipeline_status.json", {"state": state, "events": events,
                "plan_sha256": sha(plan_path), "publication_goal_complete": False})
        try:
            for file, expected in plan["source_hashes"].items():
                if sha(file) != expected:
                    raise Halt("HOLD_SOURCE_DRIFT", file)
            for name in plan["order"]:
                stage = plan["stages"][name]
                bind_stage_identity(stage)
                if completed(stage):
                    emit("REUSED_COMPLETED_STAGE", stage=name)
                else:
                    emit("RUNNING_STAGE", stage=name)
                    for index, command in enumerate(stage["commands"]):
                        marker = command.get("success_marker")
                        if marker and Path(marker["path"]).exists():
                            data = read_json(marker["path"])
                            if data.get("state") in marker["states"]:
                                continue
                        remaining = plan["hard_cap_seconds"] - (time.monotonic() - start)
                        if remaining <= 0:
                            raise Halt("RESOURCE_INCOMPLETE", "Finite total cap reached")
                        rc = run_command(command, out / f"{name}_{index}.log",
                                         min(float(command["timeout_seconds"]), remaining))
                        if rc:
                            raise Halt("STAGE_ERROR", f"{name} command{index} returned{rc}")
                        if marker and (not Path(marker["path"]).exists() or
                                       read_json(marker["path"]).get("state") not in marker["states"]):
                            raise Halt("STAGE_CONTRACT_ERROR", f"{name} missing success artifact")
                    report = Path(stage["completion"])
                    if not report.exists() or read_json(report).get("state") not in stage.get("completion_states", ["COMPLETED"]):
                        raise Halt("STAGE_CONTRACT_ERROR", name + " did not complete")
                    write_json(report.parent / "pipeline_completion.json", {
                        "stage_identity": stage["identity"], "report_sha256": sha(report)})
                    emit("STAGE_COMPLETED", stage=name)
                if name == "B_smoke":
                    a = read_json(plan["stages"]["A_smoke"]["timing_report"])
                    b = read_json(stage["timing_report"])
                    budget = estimate(a, b, plan["A_total_episodes"], plan["B_total_episodes"])
                    write_json(out / "measured_budget.json", budget)
                    emit("MEASURED_BUDGET", **budget)
                    if budget["remaining_estimate_seconds"] + time.monotonic() - start > plan["approved_budget_seconds"] * plan.get("budget_safety_fraction", .9):
                        raise Halt("HOLD_MEASURED_COST", "Smoke estimate exceeds the frozen30-minute budget")
                if name == "A_dev":
                    miou = float(read_json(stage["completion"])["class_miou"]["visual"])
                    if miou < 60:
                        raise Halt("HOLD_PROTOCOL_REVIEW", "DEV visual below60; adaptation/source must be checked")
                    if miou > 70:
                        raise Halt("HOLD_STANDARD_PROTOCOL_REVIEW", "DEV above70; no automatic headline claim")
            a = read_json(plan["stages"]["A_confirm"]["completion"])
            b = read_json(plan["stages"]["B_confirm"]["completion"])
            write_json(out / "decision.json", decision(a, b))
            emit("COMPLETED", next_stage="USER_REVIEW_OF_A_B", automatic_new_stages=False)
        except (Halt, subprocess.TimeoutExpired) as exc:
            state = exc.state if isinstance(exc, Halt) else "RESOURCE_INCOMPLETE"
            emit(state, detail=str(exc), failed_artifacts_preserved=True)
            return 75 if state.startswith("HOLD_") else 2
    return 0


def cpu_checks():
    """Synthetic contracts, no torch, filesystem experiment, CUDA or shutdown."""
    a = {"episodes": 100, "against_foris": {"gain": 3.5, "ci95": [1, 5], "episodes": 100,
        "outcome_counts": {"only_foris": 10, "only_sam3": 20, "both": 50, "neither": 20}}}
    b = {"rows": {"readout": {"gain": 2, "per_fold": [1, 2, 3, 2], "ci95": [1, 3]}}}
    assert decision(a, b)["A"] == "SAM3_STRONGER_ON_PAIRED_EXPOSED_COHORT"
    a["against_foris"]["ci95"] = [-1, 5]
    assert decision(a, b)["A"] == "SAM3_GAIN_UNCERTAIN_REVIEW"
    a["against_foris"]["ci95"] = [1, 5]
    a["against_foris"]["gain"] = 0
    assert decision(a, b)["A"] == "COMPLEMENTARY_FAILURES_REVIEW"
    b["rows"]["readout"]["per_fold"][0] = -0.1
    assert decision(a, b)["B"] == "DROP_THIS_FIXED_FORMULA"
    a["against_foris"]["episodes"] = 99
    try:
        decision(a, b)
        raise AssertionError("Partial pairing accepted")
    except ValueError:
        pass
    assert photo_uid("COCO_val2014_000000123456.jpg") == photo_uid("123456.jpg")
    budget = estimate({"episodes": 10, "elapsed_s": 20}, {"episodes": 10, "elapsed_s": 10}, 841, 841)
    assert budget["remaining_estimate_seconds"] == 2493
    return {"state": "CPU_PIPELINE_CONTRACTS_PASSED", "checks": 7,
            "GPU_used": False, "shutdown_called": False, "not_a_task_quality_result": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("cpu-check")
    check.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-body")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "cpu-check":
        result = cpu_checks()
        write_json(args.out, result)
        print(json.dumps(result))
        return 0
    return run_body(args.plan, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
