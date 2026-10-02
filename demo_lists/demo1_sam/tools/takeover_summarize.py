"""Summarize raw receipts without changing gates or selecting away failed arms."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir/"fair_compile/report.json").read_text())
    groups = collections.defaultdict(list)
    for record in report["records"]:
        groups[(record["microbatch"],record["arm"])].append(record)
    rows = []
    for (mb,arm), records in sorted(groups.items()):
        passed = len(records)==report["options"]["rounds"] and all(r["numeric_status"]=="PASSED_RANDOM_FIXTURE_ONLY" for r in records)
        row = {"microbatch": mb, "arm": arm, "rounds": len(records), "numeric_passed_and_complete": passed}
        if passed:
            medians = [r["warm_complete_prompt_batch"]["median_ms"] for r in records]
            row.update(warm_median_ms_per_round=medians, mean_of_round_medians_ms=statistics.mean(medians),
                       cold_mean_of_round_medians_ms=statistics.mean(r["cold_image_including_cache_and_prompt_batch"]["median_ms"] for r in records),
                       peak_allocated_bytes=max(r["warm_complete_prompt_batch"]["memory"]["peak_allocated_bytes"] for r in records),
                       persistent_cache_bytes=max(r["logical_persistent_cache_bytes"] for r in records))
        rows.append(row)
    complete = report["status"]=="COMPLETED" and len(rows)==len(report["arms"])*len(report["options"]["microbatches"]) and all(r["numeric_passed_and_complete"] for r in rows)
    result = {"status": "COMPLETE" if complete else "INCOMPLETE_NO_WINNER", "performance_scope": report["scope"],
              "round_contract": report["round_contract"], "results": rows, "comparisons": []}
    if complete:
        for mb in report["options"]["microbatches"]:
            same = [r for r in rows if r["microbatch"]==mb]
            dense = min((r for r in same if not r["arm"].startswith("factor")),key=lambda r:r["mean_of_round_medians_ms"])
            factor = min((r for r in same if r["arm"].startswith("factor")),key=lambda r:r["mean_of_round_medians_ms"])
            result["comparisons"].append({"microbatch": mb, "fastest_measured_dense": dense, "fastest_measured_factor": factor,
                                          "factor_latency_reduction_percent":100*(1-factor["mean_of_round_medians_ms"]/dense["mean_of_round_medians_ms"]),
                                          "per_round_latency_reduction_percent":[100*(1-f/d) for f,d in zip(factor["warm_median_ms_per_round"],dense["warm_median_ms_per_round"])]})
    result["real_numeric"] = {}
    for name in ("real_original","real_new_arms"):
        path = args.run_dir/name/"report.json"
        if not path.exists(): continue
        d=json.loads(path.read_text()); records=d["records"]
        chunks=[c for r in records for c in r.get("numeric_chunks",[]) if c["status"]!="REFERENCE"]
        result["real_numeric"][name]={"status": d["status"], "numeric_status":d["numeric_status"], "images":len(d["images"]),
                                     "records":len(records), "failed_records":sum(r["numeric_status"]=="FAILED" for r in records),
                                     "max_mask_logit_error":max((c["mask_logit_max_abs"] for c in chunks),default=None),
                                     "max_iou_prediction_error":max((c["iou_prediction_max_abs"] for c in chunks),default=None),
                                     "total_binary_pixel_flips":sum(sum(sum(x) for x in c.get("full_resolution_binary_flips_per_prompt_per_mask",[])) for c in chunks),
                                     "argmax_changes":sum(sum(c.get("multimask_iou_argmax_changed_per_prompt",[])) for c in chunks)}
    (args.run_dir/"analysis_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
