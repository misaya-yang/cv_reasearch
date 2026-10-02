#!/usr/bin/env python3
"""Bounded four-decoder trials; launch only after switching to GPU mode."""
import argparse
import atexit
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "sam_shared_decoder/execution_baselines/benchmark.py"


def integers(value):
    values = [int(x) for x in value.split(",")]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("positive comma-separated integers required")
    return values


def variants():
    # Original official stays untouched/explicit. Alternate dense schedules
    # are measured, rather than trusting auto selection based on FLOPs.
    result = [("official", "explicit", "auto", "auto"),
              ("official_sdpa", "sdpa", "auto", "auto")]
    for attention in ("explicit", "sdpa"):
        result += [("cached", attention, "auto", "auto"),
                   ("dense_assoc", attention, "auto", "auto"),
                   ("dense_assoc", attention, "projected", "dense"),
                   ("dense_assoc", attention, "associated", "sparse"),
                   ("factor_projected", attention, "auto", "auto")]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("smoke", "eager", "compile"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompts", type=integers, default=[1, 8, 32, 64, 128])
    p.add_argument("--microbatches", type=integers, default=[8, 32])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--tokens", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--timeout-per-run", type=int, default=600)
    p.add_argument("--diagnostic-on-failure", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Emit commands; no GPU, import, download or execution")
    a = p.parse_args()
    if not a.device.startswith("cuda"):
        p.error("This runner is CUDA-only; use benchmark.py directly for CPU plumbing")
    if not a.dry_run:
        import torch
        if not torch.cuda.is_available():
            p.error("CUDA unavailable: do not run this in no-card mode")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        telemetry_log = (a.output_dir / "gpu_telemetry.csv").open("w")
        telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.current.sm,clocks.current.memory",
                                      "--format=csv", "--loop-ms=1000"],
                                     stdout=telemetry_log, stderr=subprocess.STDOUT)
        def stop_telemetry():
            if telemetry.poll() is None:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait()
            telemetry_log.close()
        atexit.register(stop_telemetry)
    if a.phase == "smoke":
        counts, requested, cases = [3], [2], [(m, "explicit", "auto", "auto")
            for m in ("official", "cached", "dense_assoc", "factor_projected")]
        grid, warmup, repetitions = 4, 2, 3
    else:
        counts, requested, cases = a.prompts, a.microbatches, variants()
        grid, warmup, repetitions = a.grid, a.warmup, a.repetitions
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "phase": a.phase, "dry_run": a.dry_run,
                "telemetry": None if a.dry_run else str(a.output_dir / "gpu_telemetry.csv"),
                "attempts": []}
    manifest_path = a.output_dir / "manifest.json"
    for prompts in counts:
        for requested_mb in sorted({min(prompts, b) for b in requested}):
            mb = requested_mb
            while True:
                oom = False
                for method, attention, order, write in cases:
                    label = f"p{prompts}_requested{requested_mb}_mb{mb}_{method}_{attention}_{order}_{write}"
                    output = a.output_dir / f"{label}.json"
                    cmd = [sys.executable, str(BENCH), "--device", a.device, "--dtype", "float32",
                           "--prompt-batch", str(prompts), "--microbatch", str(mb),
                           "--grid", str(grid), "--tokens", str(a.tokens), "--seed", str(a.seed),
                           "--method", method, "--attention", attention, "--order", order,
                           "--write-output", write, "--warmup", str(warmup),
                           "--repetitions", str(repetitions), "--output", str(output),
                           "--mode", "compile" if a.phase == "compile" else "eager",
                           "--compiler-backend", "inductor"]
                    if a.diagnostic_on_failure:
                        cmd.append("--diagnostic-on-failure")
                    record = {"label": label, "prompts": prompts, "requested_microbatch": requested_mb,
                              "effective_microbatch": mb, "command": cmd, "output": str(output)}
                    print(json.dumps(record), flush=True)
                    if a.dry_run:
                        record["status"] = "PLANNED_ONLY"
                    else:
                        if output.exists() or output.with_suffix(".log").exists():
                            raise FileExistsError(f"Refusing to overwrite raw run: {output}; use a new directory")
                        try:
                            with output.with_suffix(".log").open("w") as log:
                                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                         timeout=a.timeout_per_run)
                            log_text = output.with_suffix(".log").read_text()
                            # Only explicit CUDA allocator failures trigger common batching fallback.
                            oom = (process.returncode != 0 and any(s in log_text for s in
                                   ("CUDA out of memory", "torch.OutOfMemoryError")))
                            record["returncode"] = process.returncode
                            record["status"] = "OOM" if oom else "DONE" if process.returncode == 0 else "FAILED"
                        except subprocess.TimeoutExpired:
                            record["status"] = "TIMEOUT"
                    manifest["attempts"].append(record)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    if oom:
                        break
                if not oom:
                    break
                # Discard no raw evidence; rerun ALL arms at the same smaller
                # microbatch. Summaries must reject earlier partial groups.
                if mb == 1:
                    manifest["failure"] = f"OOM at microbatch=1, prompts={prompts}"
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    raise RuntimeError(manifest["failure"])
                mb = max(1, mb // 2)
    manifest["completed"] = not a.dry_run and all(
        r["status"] in ("DONE", "OOM") for r in manifest["attempts"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not a.dry_run and not manifest["completed"]:
        raise RuntimeError("Some trials failed/timed out. Read retained logs; incomplete groups cannot establish a winner.")


if __name__ == "__main__":
    main()
