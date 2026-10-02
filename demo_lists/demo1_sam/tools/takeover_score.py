"""CPU-only scoring for original and newly named decoder arms."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/quality_mechanisms"))
import quality_summary
from score_saved_outputs import score_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.run_dir/"report.json").read_text())
    quality_summary.METHODS = tuple(dict.fromkeys(r["method"] for r in source["records"]))
    result = score_run(args.run_dir, args.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "source_records_scored", "source_numeric_status")}), flush=True)


if __name__ == "__main__":
    main()
