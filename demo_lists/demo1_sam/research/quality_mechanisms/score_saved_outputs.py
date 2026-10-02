"""Offline NumPy-only scoring of NPZ outputs from run_real_sam.py.

Does not load PyTorch, a model/checkpoint, CUDA, or the original COCO archives.
Leaves the generation report and its numeric failure status untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from diagnose_oracle_gap import diagnose
from quality_summary import attach_prompt_metadata, summarize_quality


def score_run(run_dir, output_dir, boundary_ratio=0.02):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if run_dir == output_dir:
        raise ValueError("offline score output-dir must differ from raw generation run-dir")
    if (output_dir / "score_report.json").exists():
        raise FileExistsError("score report already exists; choose a fresh output-dir")
    source_bytes = (run_dir / "report.json").read_bytes()
    source = json.loads(source_bytes)
    status = source["status"]
    numeric_status = source.get("numeric_status", "FAILED" if status.startswith("FAILED") else ("PASSED" if status.startswith("PASSED") else status))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "RUNNING_CPU_QUALITY_SCORING",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_run_dir": str(run_dir), "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "source_run_status": status, "source_numeric_status": numeric_status,
              "numeric_status": numeric_status, "boundary_ratio": boundary_ratio,
              "subset_manifest_sha256": source.get("subset_manifest_sha256"),
              "contract": "CPU quality scoring only; preserves source numeric states; GT oracles are diagnostic; no throughput/SOTA claim",
              "records": []}
    def save():
        report["quality_summary"] = summarize_quality(report["records"])
        temporary = output_dir / "score_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(output_dir / "score_report.json")
    start = time.perf_counter()
    try:
        for original in source["records"]:
            npz_path = (run_dir / original["npz"]).resolve()
            if not npz_path.is_relative_to(run_dir):
                raise ValueError("source NPZ path escapes run-dir")
            with np.load(npz_path, allow_pickle=False) as arrays:
                annotation_ids = arrays["annotation_ids"]
                try:
                    quality = diagnose(arrays, boundary_ratio)
                except ValueError as error:
                    quality = {"status": "INVALID_QUALITY_INPUT", "error": str(error), "rows": []}
                attach_prompt_metadata(quality, original, annotation_ids)
            image_id = int(original["image_id"])
            folder = output_dir / f"image_{image_id:012d}"
            folder.mkdir(exist_ok=True)
            quality_path = folder / f"{original['regime']}_{original['method']}_quality.json"
            quality_path.write_text(json.dumps(quality, indent=2)+"\n")
            record = {"image_id": image_id, "regime": original["regime"], "method": original["method"],
                      "numeric_status": original["numeric_status"], "numeric_chunks": original.get("numeric_chunks", []),
                      "source_npz": original["npz"], "quality_file": str(quality_path.relative_to(output_dir)), "quality": quality}
            report["records"].append(record)
            save()
        invalid = sum(record["quality"]["status"] == "INVALID_QUALITY_INPUT" for record in report["records"])
        report["status"] = "CPU_QUALITY_COMPLETED_WITH_INVALID_OUTPUTS" if invalid else "CPU_QUALITY_COMPLETED"
        report["source_records_scored"] = len(report["records"])
        report["invalid_quality_records"] = invalid
        report["cpu_scoring_wall_seconds"] = time.perf_counter()-start
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return report
    except Exception as error:
        report["status"] = "ERROR_CPU_QUALITY_SCORING"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-ratio", type=float, default=0.02)
    args = parser.parse_args()
    result = score_run(args.run_dir, args.output_dir, args.boundary_ratio)
    print(json.dumps({key: result[key] for key in ("status", "source_run_status", "source_numeric_status", "source_records_scored", "quality_summary")}, indent=2))
