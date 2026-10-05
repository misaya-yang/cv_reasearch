#!/usr/bin/env bash
# Prepared only. Invoke explicitly in an authorized compute session.
# Usage: bash scripts/launch_astra_dev241.sh PYTHON CACHE_ROOT NEW_OUTPUT_DIR
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PYTHON CACHE_ROOT NEW_OUTPUT_DIR" >&2
  exit 2
fi
astra_python=$1
astra_cache=$2
astra_output=$3
astra_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -d "$astra_cache" || -e "$astra_output" ]]; then
  echo "Cache root must exist and output path must be new." >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$astra_python" "$astra_repo/evidence/local/research_20261005/astra_portable/rerun.py" all \
  --root "$astra_cache" \
  --manifest "$astra_repo/evidence/local/research_20261005/dev241.json" \
  --out "$astra_output" --expected 241 --bg-evidence packet
