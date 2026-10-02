#!/usr/bin/env python3
"""Read-only host/GPU snapshot. Does not install, set clocks, or start work."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(argv):
    if shutil.which(argv[0]) is None:
        return {"argv": argv, "available": False}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"argv": argv, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "error": "timeout after 20 seconds"}


def collect():
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version,
              "python_executable": sys.executable,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_modification": "Vendor/model name alone cannot verify VRAM modification; record physical configuration separately.",
              "commands": [command(["nvidia-smi"]), command(["nvidia-smi", "-q", "-x"]),
                           command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]),
                           command(["nvcc", "--version"])]}
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cudnn": torch.backends.cudnn.version(),
                           "cuda_available": torch.cuda.is_available(),
                           "build_config": torch.__config__.show(), "devices": []}
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            report["torch"]["devices"].append({"index": i, "name": prop.name,
                "total_memory_bytes": prop.total_memory, "runtime_total_bytes": total,
                "free_memory_bytes": free, "capability": [prop.major, prop.minor],
                "multiprocessors": prop.multi_processor_count})
    except ImportError:
        report["torch"] = {"available": False}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modification-note", default="Not yet supplied or physically verified")
    a = p.parse_args()
    report = collect()
    report["operator_modification_note"] = a.modification_note
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
