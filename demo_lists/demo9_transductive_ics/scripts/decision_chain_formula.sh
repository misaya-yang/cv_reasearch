#!/bin/bash
# Retired B-only chain delegates to the prepared A -> B -> C handover.
# Default is CPU preparation: no old queue, deletion, retry, training or CUDA.
# Formula constants are pooled-class label-supervised calibration; freezing
# them does not make their derivation training-free or held-class FSS.
set -euo pipefail
FORMULA_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FORMULA_ROOT_WRAPPER="$FORMULA_SCRIPT_DIR/run_sam3_handover.sh"
if [[ ! -f "$FORMULA_ROOT_WRAPPER" ]]; then
  echo "Unified handover missing; the historical formula queue will not run." >&2
  exit 2
fi
case "${1:-prepare}" in
  prepare|execute) exec bash "$FORMULA_ROOT_WRAPPER" "${1:-prepare}" ;;
  *) echo "Usage: decision_chain_formula.sh [prepare|execute] (unified handover only)" >&2; exit 2 ;;
esac
