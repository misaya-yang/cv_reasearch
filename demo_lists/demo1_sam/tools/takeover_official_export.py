"""Official-only real-image outputs, avoiding a redundant implementation matrix."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/quality_mechanisms"))
import run_real_sam

run_real_sam.METHODS=("official",)
raise SystemExit(run_real_sam.main())
