"""Execute the prepared GPU stages continuously, with durable logs and status."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def plan(output):
    real = output/"real_original"
    return [
        ("real_24_images_export", [sys.executable, "research/quality_mechanisms/run_real_sam.py",
            "--subset-dir", "assets/coco_quality_seed2027_v1", "--checkpoint", "assets/checkpoints/sam_vit_b_01ec64.pth",
            "--output-dir", str(real), "--device", "cuda:0", "--limit-images", "24", "--microbatch", "4",
            "--diagnostic-on-failure", "--defer-quality", "--npz-compression", "none", "--save-encoded-inputs"], 600),
        ("fair_compiled_phase_comparison", [sys.executable, "tools/takeover_benchmark.py",
            "--device", "cuda:0", "--mode", "compile", "--output-dir", str(output/"fair_compile"),
            "--rounds", "2", "--microbatches", "8,128"], 900),
        ("real_new_arms_replay", [sys.executable, "tools/takeover_replay.py", "--run-dir", str(real),
            "--output-dir", str(output/"real_new_arms"), "--device", "cuda:0", "--microbatch", "4"], 600),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    output = args.output_dir.resolve()
    jobs = plan(output)
    if args.dry_run:
        print(json.dumps(jobs, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    lock = (ROOT/"runtime/takeover_gpu.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no stage started")
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError("Unexpected allocator use before queue")
    # The queue records evidence and progress independently of the chat UI.
    report = {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "owner": "01a0f783-6584-7e60-b6d3-5bf227e7e7bc", "pid": os.getpid(),
              "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
              "stages": []}
    path = output/"queue_status.json"
    save(path, report)
    telemetry_log = (output/"gpu_telemetry.csv").open("x")
    telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw",
                                  "--format=csv", "--loop-ms=1000"], stdout=telemetry_log, stderr=subprocess.STDOUT)
    try:
        for name, command, timeout in jobs:
            record = {"name": name, "status": "RUNNING", "command": command,
                      "started_at_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": timeout}
            report["stages"].append(record); save(path, report)
            started = time.monotonic()
            print("START "+name, flush=True)
            with (output/(name+".log")).open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = proc.pid; save(path, report)
                try:
                    code = proc.wait(timeout=timeout)
                except BaseException:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    record["status"] = "INTERRUPTED_OR_TIMEOUT"; save(path, report)
                    raise
            record.update(returncode=code, wall_seconds=time.monotonic()-started,
                          status="DONE" if code==0 else "FAILED",
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
            save(path, report)
            print("DONE "+name+" rc="+str(code), flush=True)
            if code:
                raise RuntimeError(f"{name} failed; raw log retained, no silent fallback")
        report["status"] = "GPU_STAGES_COMPLETE"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = str(error)
        raise
    finally:
        save(path, report)
        telemetry.terminate(); telemetry.wait(timeout=5); telemetry_log.close()


if __name__ == "__main__":
    main()
