#!/usr/bin/env bash
# OPTIONAL FALLBACK ONLY. Existing remote torch/NumPy were found, so the initial
# prefetch was stopped. Do not run unless the existing GPU env proves unusable.
# Downloads binary wheels ONLY; no installation/test/compilation.
set -euo pipefail
task_root="${1:-/root/autodl-tmp/demo1_sam}"
task_pip="${2:-/root/miniconda3/bin/pip}"
mkdir -p "$task_root/downloads/wheels"
"$task_pip" download --only-binary=:all: --dest "$task_root/downloads/wheels" \
  --index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0'
"$task_pip" download --only-binary=:all: --dest "$task_root/downloads/wheels" \
  --index-url https://pypi.org/simple 'numpy==2.2.6'
date -u +%FT%TZ > "$task_root/downloads/PREFETCH_COMPLETE"
