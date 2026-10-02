#!/usr/bin/env python3
"""Summarize complete, same-batch CUDA comparisons; retain numerical failures."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_short_gpu import variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    groups = defaultdict(list)
    rejected = []
    for directory in a.directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["dry_run"]:
            rejected.append({"directory": str(directory), "reason": "dry run"})
            continue
        for attempt in manifest["attempts"]:
            if attempt["status"] != "DONE":
                rejected.append({"label": attempt["label"], "reason": attempt["status"]})
                continue
            path = Path(attempt["output"])
            # Output references may come from a remote host; support moved folders.
            if not path.exists():
                path = directory / path.name
            report = json.loads(path.read_text())
            if not report.get("completed") or not report["device"].startswith("cuda"):
                rejected.append({"path": str(path), "reason": "incomplete or not CUDA"})
                continue
            opts = report["options"]
            key = (str(directory), manifest["phase"], attempt["requested_microbatch"],
                   opts["prompt_batch"], opts["microbatch"], opts["grid"], opts["tokens"],
                   opts["dtype"], opts["tf32"], report["torch"], report["hardware"],
                   report["fixture"]["model_fp32_state_sha256"], opts["seed"])
            groups[key].append((path, report["methods"][0], opts))
    rows = []
    for key, runs in groups.items():
        directory, phase, requested, prompts, mb, grid, tokens, dtype, tf32, torch_version, hardware, model_hash, seed = key
        expected = 4 if phase == "smoke" else len(variants())
        identities = {(r[1]["method"], r[2]["attention"], r[2]["order"], r[2]["write_output"]) for r in runs}
        if len(runs) != expected or len(identities) != expected:
            rejected.append({"directory": directory, "prompts": prompts, "microbatch": mb,
                             "reason": f"incomplete/duplicate group: {len(identities)}/{expected}"})
            continue
        def hot(run):
            return run[1]["warm_complete_prompt_batch"]["median_ms"]
        def cold(run):
            return run[1]["cold_image_including_cache_and_prompt_batch"]["median_ms"]
        baselines = [r for r in runs if r[1]["method"] != "factor_projected"]
        candidates = [r for r in runs if r[1]["method"] == "factor_projected"]
        baseline, candidate = min(baselines, key=hot), min(candidates, key=hot)
        cold_baseline, cold_candidate = min(baselines, key=cold), min(candidates, key=cold)
        def describe(run):
            path, item, opts = run
            return {"raw_result": str(path), "method": item["method"],
                    "attention": opts["attention"], "order": opts["order"],
                    "write_output": opts["write_output"], "numeric_status": item["numeric_status"],
                    "hot_median_ms": hot(run), "cold_median_ms": cold(run),
                    "hot_memory": item["warm_complete_prompt_batch"]["memory"],
                    "cold_memory": item["cold_image_including_cache_and_prompt_batch"]["memory"],
                    "logical_cache_bytes": item.get("logical_persistent_cache_bytes", 0)}
        all_passed = all(r[1]["verification"]["passed"] for r in runs)
        rows.append({"directory": directory, "phase": phase, "prompts": prompts,
                     "requested_microbatch": requested, "microbatch": mb,
                     "grid": grid, "tokens": tokens, "dtype": dtype, "tf32": tf32,
                     "torch": torch_version, "hardware": hardware, "model_hash": model_hash, "seed": seed,
                     "all_random_numeric_checks_passed": all_passed,
                     "status": "random_fixture_performance_only" if all_passed else "FAILED_NUMERICS_DIAGNOSTIC_ONLY",
                     "fastest_measured_baseline": describe(baseline),
                     "fastest_measured_candidate": describe(candidate),
                     "hot_baseline_over_candidate": hot(baseline) / hot(candidate),
                     "fastest_cold_baseline": describe(cold_baseline),
                     "fastest_cold_candidate": describe(cold_candidate),
                     "cold_baseline_over_candidate": cold(cold_baseline) / cold(cold_candidate)})
    summary = {"scope": "Complete synthetic encoded-input CUDA decoder; ratios are measured medians, not FLOPs. >1 favors candidate. No real-image or statistical-significance claim.",
               "groups": rows, "rejected_or_incomplete": rejected}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
