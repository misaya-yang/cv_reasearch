#!/usr/bin/env python3
"""Discriminative merge study with common-baseline optimization and old controls."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "sam_shared_decoder/execution_baselines/benchmark.py"
OUT = ROOT / "results/gpu_projection_merge"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    # Old best configurations plus both new eligible merged variants. All
    # methods see exactly the same prompts/batch. Associated strong routes stay.
    for mode in ("eager", "compile"):
        for prompts, mb in ((1, 1), (128, 8), (128, 128)):
            strong = ("cached", "sdpa", "auto", "auto") if mb == 1 else ("dense_assoc", "explicit", "associated", "sparse")
            original_factor = ("factor_projected", "sdpa" if mb == 1 else "explicit", "auto", "auto")
            cases = [strong, original_factor]
            cases += [(m, att, "auto", "auto") for m in ("cached_merged", "factor_merged")
                      for att in ("explicit", "sdpa")]
            for method, attention, order, write in cases:
                label = f"{mode}_p{prompts}_mb{mb}_{method}_{attention}"
                output = OUT / f"{label}.json"
                if output.exists() or output.with_suffix(".log").exists():
                    raise FileExistsError(output)
                cmd = [sys.executable, str(BENCH), "--device", "cuda:0", "--dtype", "float32",
                       "--grid", "64", "--tokens", "7", "--prompt-batch", str(prompts),
                       "--microbatch", str(mb), "--mode", mode, "--compiler-backend", "inductor",
                       "--method", method, "--attention", attention, "--order", order,
                       "--write-output", write, "--warmup", "10", "--repetitions", "20",
                       "--output", str(output)]
                with output.with_suffix(".log").open("w") as log:
                    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                records.append({"label": label, "command": cmd, "returncode": p.returncode})
                (OUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
                print(label, p.returncode, flush=True)


if __name__ == "__main__":
    main()
