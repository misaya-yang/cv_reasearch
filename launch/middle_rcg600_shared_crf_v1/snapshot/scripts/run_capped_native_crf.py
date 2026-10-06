#!/usr/bin/env python3
"""Run only a sealed native CRF phase with a bounded CUDA allocator."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--memory-gib", type=float, default=8.0)
    args = parser.parse_args()
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * 2**30 / total, 1.0))
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(args.executor_dir.resolve()))
    shared = importlib.import_module("run_part2_evidence")
    started = time.monotonic()
    result = dict(state="RUNNING", encoder_forwards=0, memory_cap_gib=args.memory_gib,
                  executor_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest())
    try:
        shared.finalize(argparse.Namespace(out=args.out, threads=1))
        result["state"] = "NATIVE_CRF_COMPLETED"
    except BaseException as error:
        result.update(state="NATIVE_CRF_FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(seconds=time.monotonic() - started,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        args.profile.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
