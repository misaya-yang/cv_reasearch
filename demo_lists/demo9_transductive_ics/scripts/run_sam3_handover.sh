#!/usr/bin/env bash
# Default: no-card preparation only. execute never rents or boots an instance.
set -euo pipefail
ROOT=${DEMO9_ROOT:-/root/autodl-tmp/demo9_transductive_ics}
PY=${DEMO9_PYTHON:-/root/miniconda3/bin/python}
OUT=${DEMO9_AB_OUT:-$ROOT/results/sam3_handover_v1}
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT"
case "${1:-prepare}" in
  prepare)
    CUDA_VISIBLE_DEVICES= "$PY" scripts/prepare_sam3_handover.py --root "$ROOT" --out "$OUT"
    CUDA_VISIBLE_DEVICES= "$PY" scripts/experiment_resource_guard.py \
      --plan "$OUT/guard_plan.json" --state-file "$OUT/guard.json" --preflight-only
    ;;
  execute)
    # One outer process-group owner. No nested guards, retry loops or deletion.
    "$PY" scripts/experiment_resource_guard.py --plan "$OUT/guard_plan.json" \
      --state-file "$OUT/guard.json" --run --allow-shutdown
    ;;
  *) printf '%s\n' 'Usage: run_sam3_handover.sh [prepare|execute]' >&2; exit 2 ;;
esac
