#!/bin/bash
# CPU checks before the scale-alignment run takes the GPU. Run from the direction's root on the server.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
FORIS=/root/autodl-tmp/demo8_local_verification/foris_source
R=results/scale_align_v0
mkdir -p $R && rm -rf $R/fixture && : > $R/cpu_checks.log
step() { name=$1; shift; "$@" > $R/$name.log 2>&1; rc=$?; echo "$name rc=$rc" >> $R/cpu_checks.log; [ $rc -eq 0 ] || { echo "FAILED $name" >> $R/cpu_checks.log; exit 1; }; }
step fixture $PY scripts/scale_align_experiment.py --fixture $R/fixture --foris-root $FORIS
step fixture_analysis $PY scripts/analyze_scale_align.py --run $R/fixture/scale_run --out $R/fixture/analysis.json
step plan $PY scripts/prepare_scale_align_queue.py --manifest results/extent_v1/episodes.json --packets results/extent_v1/run/packets \
  --fixture-report $R/fixture/scale_run/report.json --out $R/run --plan $R/plan.json
step preflight $PY scripts/experiment_resource_guard.py --plan $R/plan.json --state-file $R/guard.json --preflight-only
echo ALL_DONE >> $R/cpu_checks.log
