#!/usr/bin/env python3
"""Fixed native-CRF successor with shared VRAM and five-second resource records.

It never encodes images, kills a foreign job, retries a failed stage or shuts down.
Only the declared sealed runs and scorer commands are executed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def resources():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if gpu.returncode:
        return dict(known=False, error=gpu.stderr.strip())
    util, used, total = map(float, gpu.stdout.strip().splitlines()[0].split(","))
    stats = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return dict(known=True, gpu_util=util, gpu_used_mib=used, gpu_total_mib=total,
                cpu_usage_usec=int(stats["usage_usec"]),
                host_memory_bytes=int(Path("/sys/fs/cgroup/memory.current").read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.state.exists() or args.log.exists():
        raise FileExistsError("Fresh controller files required; no implicit restart")
    state = dict(state="INITIAL", controller_pid=os.getpid(), owned_processes=[], encoder_forwards=0,
                 gpu_overlap="Only declared native CRF, after VRAM headroom checks")
    previous = None
    with args.log.open("x") as log:
        def record(name, **data):
            nonlocal previous
            now, sample = time.time(), resources()
            if previous and sample["known"] and previous[1]["known"]:
                sample["cpu_cores_used"] = (sample["cpu_usage_usec"] - previous[1]["cpu_usage_usec"]) / max((now - previous[0]) * 1e6, 1)
            previous = now, sample
            state.update(state=name, last_event=dict(time=now, resources=sample, **data))
            write_json(args.state, state)
            log.write(json.dumps(dict(state=name, **state["last_event"])) + "\n")
            log.flush()
            return sample
        for index, stage in enumerate(plan["stages"]):
            for required in stage.get("requires", []):
                value = json.loads(Path(required["path"]).read_text())
                if any(value.get(k) != v for k, v in required.get("equals", {}).items()):
                    raise ValueError("Required sealed artifact condition failed")
            if stage["kind"] == "gpu":
                while True:
                    sample = record("WAITING_SHARED_VRAM", stage=stage["name"])
                    if sample["known"] and sample["gpu_total_mib"] - sample["gpu_used_mib"] >= plan["required_free_mib"]:
                        break
                    time.sleep(5)
            environment = os.environ.copy()
            environment.update(plan["env"])
            if stage["kind"] == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            output = args.log.parent / f"stage-{index}.log"
            with output.open("xb") as stream:
                child = subprocess.Popen(stage["argv"], cwd=plan["cwd"], env=environment,
                                         stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                stat = Path("/proc", str(child.pid), "stat").read_text().rsplit(")", 1)[1].split()
                state["owned_processes"].append(dict(pid=child.pid, start_ticks=int(stat[19]), stage=stage["name"]))
                while child.poll() is None:
                    record("STAGE_RUNNING", stage=stage["name"], pid=child.pid)
                    time.sleep(5)
            if child.returncode:
                record("STAGE_FAILED", stage=stage["name"], returncode=child.returncode)
                return child.returncode
            for check in stage.get("checks", []):
                value = json.loads(Path(check["path"]).read_text())
                if any(value.get(k) != v for k, v in check.get("equals", {}).items()):
                    record("STAGE_ARTIFACT_FAILED", stage=stage["name"])
                    return 3
            record("STAGE_COMPLETED", stage=stage["name"])
        record("QUEUE_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
