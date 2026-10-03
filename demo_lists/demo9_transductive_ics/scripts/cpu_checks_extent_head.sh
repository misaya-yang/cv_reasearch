#!/bin/bash
# CPU checks before the extent head takes the GPU. Run from the direction's root on the server.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
FORIS=/root/autodl-tmp/demo8_local_verification/foris_source
R=results/extent_head_v0
mkdir -p $R && rm -rf $R/fixture && : > $R/cpu_checks.log
step() { name=$1; shift; "$@" > $R/$name.log 2>&1; rc=$?; echo "$name rc=$rc" >> $R/cpu_checks.log; [ $rc -eq 0 ] || { echo "FAILED $name" >> $R/cpu_checks.log; exit 1; }; }
step selfcheck $PY scripts/extent_head.py --selfcheck --out $R/selfcheck.json
step fixture_extent $PY scripts/extent_experiment.py --fixture $R/fixture --foris-root $FORIS
step fixture_evidence $PY scripts/evidence_cache.py --fixture $R/fixture --foris-root $FORIS --pool-images 8 --pool-patches 64
step fixture_inputs $PY scripts/extent_train_cache.py --fixture $R/fixture --foris-root $FORIS
step fixture_replay $PY scripts/extent_train_cache.py --replay --fixture $R/fixture
step fixture_fit $PY scripts/extent_head.py --cache $R/fixture/head --run $R/fixture/run --out $R/fixture/head/fit.json --epochs 2 --seeds 1
step prepare $PY scripts/extent_train_cache.py --prepare --test-manifest results/extent_v1/episodes.json --per-fold 600 --manifest $R/train_episodes.json
step plan $PY scripts/prepare_extent_head_queue.py --train-manifest $R/train_episodes.json --test-manifest results/extent_v1/episodes.json \
  --extent-run results/extent_v1/run --evidence-cache cache/evidence_v1 --selfcheck $R/selfcheck.json \
  --fixture-report $R/fixture/head/fit.json --cache cache/extent_head_v0 --out $R --plan $R/plan.json
step preflight $PY scripts/experiment_resource_guard.py --plan $R/plan.json --state-file $R/guard.json --preflight-only
echo ALL_DONE >> $R/cpu_checks.log
