#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/astra_granularity_20261008
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/demo4_cache/env
/root/miniconda3/bin/python -u cache_features.py > cache.log 2>&1
/root/miniconda3/bin/python -u run_candidate.py > candidate.log 2>&1
/root/miniconda3/bin/python -u evaluate_transfer_if_pass.py > transfer.log 2>&1
date -u > pipeline_complete.txt
