#!/usr/bin/env python3
"""Bounded layout/implicit GPU ablation; dry-run never imports torch or launches jobs."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "sam_shared_decoder/execution_baselines/benchmark.py"
WRAPPER = ROOT / "tools/with_data_disk.sh"

# (method, attention, contraction order, write output). All controls are measured
# in this round; no arithmetic-cost rule selects the strongest dense route.
CONTROLS = (
    ("cached_merged", "explicit", "auto", "auto"),
    ("cached_merged", "sdpa", "auto", "auto"),
    ("factor_merged", "explicit", "auto", "auto"),
    ("dense_assoc", "explicit", "associated", "sparse"),
    ("dense_assoc", "explicit", "projected", "dense"),
    ("factor_projected", "explicit", "auto", "auto"),
)
PHASE = (
    ("dense_phase", "explicit", "auto", "auto"),
    ("dense_phase", "explicit", "associated", "sparse"),
    ("dense_phase", "explicit", "projected", "dense"),
    ("cached_phase", "sdpa", "auto", "auto"),
    ("factor_phase", "explicit", "auto", "auto"),
)
IMPLICIT = (
    ("factor_implicit", "explicit", "auto", "auto"),
    ("factor_implicit_phase", "explicit", "auto", "auto"),
)


def positive_list(value):
    try:
        values = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positive comma-separated integers required") from exc
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("positive distinct comma-separated integers required")
    return values


def coverage(method, attention):
    if "implicit" in method:
        return "explicit implicit read/write after first layer; SDPA only first read/self-attention when requested"
    if method.startswith("factor"):
        return "explicit factor writes; eligible reads/self-attention use " + attention
    return "eligible dense routes use " + attention + "; associated sparse writes remain explicit"


def registry_check():
    tree = ast.parse((BENCH.parent / "experimental_methods.py").read_text())
    methods = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "METHODS" for t in n.targets))
    needed = {v[0] for v in CONTROLS + PHASE + IMPLICIT} - {"dense_assoc", "factor_projected"}
    if not needed.issubset(methods):
        raise RuntimeError(f"Experimental registry missing methods: {needed - set(methods)}")


def jobs(args):
    batches = [(p, min(p, mb)) for p in args.prompts for mb in args.microbatches]
    if len(set(batches)) != len(batches):
        raise ValueError("Requested microbatches collapse to duplicate effective groups")
    selected = PHASE if args.stage in ("phase", "all") else IMPLICIT
    sections = [("controls_and_phase" if args.stage != "implicit" else "controls_and_implicit", CONTROLS + selected)]
    if args.stage == "all":
        # Complete both public-phase groups first, then append the two implicit
        # arms using the same round's controls. No repeated control matrix.
        sections.append(("implicit", IMPLICIT))
    for section, variants in sections:
        for prompts, mb in batches:
            for method, attention, order, write in variants:
                label = f"{args.mode}_p{prompts}_mb{mb}_{method}_{attention}_{order}_{write}"
                output = args.output_dir / f"{label}.json"
                command = ["bash", str(WRAPPER), sys.executable, str(BENCH),
                           "--device", args.device, "--dtype", "float32", "--mode", args.mode,
                           "--compiler-backend", "inductor", "--prompt-batch", str(prompts),
                           "--microbatch", str(mb), "--grid", "64", "--tokens", "7", "--seed", "0",
                           "--method", method, "--attention", attention, "--order", order,
                           "--write-output", write, "--warmup", "10", "--repetitions", "20",
                           "--tolerance", "0.00005", "--output", str(output)]
                yield {"label": label, "section": section, "prompts": prompts, "microbatch": mb,
                       "method": method, "attention": attention, "order": order, "write_output": write,
                       "attention_coverage": coverage(method, attention), "command": command,
                       "output": str(output), "log": str(output.with_suffix(".log"))}


def stop_process_group(proc):
    # A compile timeout must not leave this job's owned worker subprocesses.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_job(job, timeout):
    started = time.monotonic()
    output, log_path = Path(job["output"]), Path(job["log"])
    if output.exists() or log_path.exists():
        raise FileExistsError(f"Refusing raw result overwrite: {output}")
    with log_path.open("x") as log:
        proc = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_group(proc)
            return {**job, "status": "TIMEOUT", "wall_seconds": time.monotonic() - started,
                    "timeout_seconds": timeout, "winner_eligible": False}
        except BaseException:
            stop_process_group(proc)
            raise
    text = log_path.read_text(errors="replace")
    record = {**job, "returncode": returncode, "wall_seconds": time.monotonic() - started,
              "winner_eligible": False}
    if returncode and any(v in text for v in ("CUDA out of memory", "torch.OutOfMemoryError")):
        record["status"] = "OOM"
        return record
    if not output.exists():
        record["status"] = "FAILED_NO_JSON"
        return record
    try:
        report = json.loads(output.read_text())
    except (ValueError, OSError) as exc:
        record.update(status="FAILED_INVALID_JSON", parse_error=str(exc))
        return record
    record["fixture"] = report.get("fixture")
    record["numeric_gate_passed"] = report.get("all_numerical_checks_passed", False)
    numerical_failure = report.get("failure") == "Numerical equivalence gate" or any(
        not m.get("verification", {}).get("passed", False) for m in report.get("methods", []))
    if numerical_failure:
        record["status"] = "NUMERIC_GATE_FAILED"
    elif returncode or not report.get("completed"):
        record["status"] = "FAILED"
    else:
        options = report.get("options", {})
        expected = {"method": job["method"], "attention": job["attention"],
                    "order": job["order"], "write_output": job["write_output"],
                    "prompt_batch": job["prompts"], "microbatch": job["microbatch"],
                    "grid": 64, "tokens": 7, "seed": 0, "dtype": "float32",
                    "tf32": False, "tolerance": 5e-5, "warmup": 10, "repetitions": 20,
                    "compiler_backend": "inductor", "mode": job["label"].split("_", 1)[0]}
        mismatches = {key: {"expected": value, "actual": options.get(key)}
                      for key, value in expected.items() if options.get(key) != value}
        methods = report.get("methods", [])
        expected_compile = "optimizing_inductor_fullgraph" if expected["mode"] == "compile" else "eager"
        method_ok = (len(methods) == 1 and methods[0].get("method") == job["method"]
                     and methods[0].get("compile_status") == expected_compile)
        model_hash = report.get("fixture", {}).get("model_fp32_state_sha256")
        hash_ok = isinstance(model_hash, str) and len(model_hash) == 64 and all(c in "0123456789abcdef" for c in model_hash)
        if mismatches or not method_ok or not hash_ok:
            record.update(status="PROTOCOL_MISMATCH", protocol_mismatches=mismatches,
                          method_contract_valid=method_ok, model_hash_valid=hash_ok)
        else:
            record["status"] = "DONE"
            record["winner_eligible"] = bool(record["numeric_gate_passed"])
    return record


def save_manifest(path, manifest):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)  # Only the manifest owned by this fresh run is updated.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("phase", "implicit", "all"), default="all")
    parser.add_argument("--mode", choices=("compile", "eager"), default="compile")
    parser.add_argument("--prompts", type=positive_list, default=[128])
    parser.add_argument("--microbatches", type=positive_list, default=[8, 128])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gpu_layout_implicit")
    parser.add_argument("--timeout-per-job", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.device.startswith("cuda") or args.timeout_per_job < 1:
        parser.error("CUDA device and positive per-job timeout required")
    registry_check()
    plan = list(jobs(args))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
                "mode": args.mode, "dry_run": args.dry_run, "job_count": len(plan),
                "protocol": {"dtype": "float32", "tf32": False, "grid": 64, "T": 7, "seed": 0,
                             "warmup": 10, "repetitions": 20, "absolute_gate": 5e-5,
                             "compiler": "inductor/fullgraph/dynamic=False",
                             "input_scope": "same random weights and deterministic synthetic encoded input; four masks+IoU",
                             "batch_failure": "No silent batch change. An OOM or failed arm makes its group incomplete; rerun all arms at a common smaller batch in a fresh directory.",
                             "speed_scope": "complete low-resolution decoder only; no image/prompt encoder or final resize"},
                "planned_jobs": plan, "attempts": []}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.output_dir.exists():
        parser.error(f"Refusing existing output directory: {args.output_dir}; choose a fresh directory")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --dry-run in CPU/no-card mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.json"
    save_manifest(manifest_path, manifest)
    telemetry = None
    with (args.output_dir / "gpu_telemetry.csv").open("x") as log:
        try:
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.current.sm,clocks.current.memory",
                                          "--format=csv", "--loop-ms=1000"], stdout=log, stderr=subprocess.STDOUT)
            for i, job in enumerate(plan, 1):
                print(json.dumps({"progress": f"{i}/{len(plan)}", "label": job["label"]}), flush=True)
                record = run_job(job, args.timeout_per_job)
                manifest["attempts"].append(record)
                save_manifest(manifest_path, manifest)
                print(json.dumps({"label": job["label"], "status": record["status"]}), flush=True)
        finally:
            if telemetry is not None and telemetry.poll() is None:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait()
    groups = {}
    for p, mb in sorted({(j["prompts"], j["microbatch"]) for j in plan}):
        runs = [r for r in manifest["attempts"] if r["prompts"] == p and r["microbatch"] == mb]
        expected = sum(j["prompts"] == p and j["microbatch"] == mb for j in plan)
        hashes = {r.get("fixture", {}).get("model_fp32_state_sha256") for r in runs if r.get("fixture")}
        complete = len(runs) == expected and all(r["winner_eligible"] for r in runs) and len(hashes) == 1 and None not in hashes
        groups[f"p{p}_mb{mb}"] = {"complete_and_numeric_passed": complete, "expected_jobs": expected,
                                  "actual_jobs": len(runs), "model_hashes": sorted(hashes)}
    manifest["groups"] = groups
    manifest["completed"] = all(g["complete_and_numeric_passed"] for g in groups.values())
    save_manifest(manifest_path, manifest)
    if not manifest["completed"]:
        raise SystemExit("Incomplete/failed groups retained; they cannot establish a winner")


if __name__ == "__main__":
    main()
