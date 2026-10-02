#!/usr/bin/env python3
"""Untimed traces of measured wins/losses; no simultaneous GPU jobs."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "sam_shared_decoder/execution_baselines/benchmark.py"
OUT = ROOT / "results/gpu_profiles"
CASES = [
    ("small_base", 1, 1, "compile", "cached", "sdpa", "auto", "auto"),
    ("small_factor", 1, 1, "compile", "factor_projected", "sdpa", "auto", "auto"),
    ("large_base", 128, 128, "compile", "dense_assoc", "explicit", "associated", "sparse"),
    ("large_factor", 128, 128, "compile", "factor_projected", "explicit", "auto", "auto"),
    ("eager_loss_base", 128, 32, "eager", "dense_assoc", "explicit", "auto", "auto"),
    ("eager_loss_factor", 128, 32, "eager", "factor_projected", "explicit", "auto", "auto"),
    ("wide_sdpa", 128, 32, "eager", "dense_assoc", "sdpa", "associated", "sparse"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for label, prompts, mb, mode, method, attention, order, write in CASES:
        output = OUT / f"{label}.json"
        if output.exists():
            raise FileExistsError(output)
        cmd = [sys.executable, str(BENCH), "--device", "cuda:0", "--dtype", "float32",
               "--grid", "64", "--tokens", "7", "--prompt-batch", str(prompts),
               "--microbatch", str(mb), "--mode", mode, "--compiler-backend", "inductor",
               "--method", method, "--attention", attention, "--order", order,
               "--write-output", write, "--warmup", "3", "--repetitions", "3",
               "--profile-dir", str(OUT / label), "--output", str(output)]
        with output.with_suffix(".log").open("w") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        records.append({"label": label, "command": cmd, "returncode": p.returncode})
        (OUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print(label, p.returncode, flush=True)


if __name__ == "__main__":
    main()
