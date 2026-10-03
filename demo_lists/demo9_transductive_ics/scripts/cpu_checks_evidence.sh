#!/bin/bash
# CPU checks before the evidence audit takes the GPU. Run from the direction's root on the server.
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
FORIS=/root/autodl-tmp/demo8_local_verification/foris_source
R=results/evidence_v1
mkdir -p $R && rm -rf $R/fixture && : > $R/cpu_checks.log
step() { name=$1; shift; "$@" > $R/$name.log 2>&1; rc=$?; echo "$name rc=$rc" >> $R/cpu_checks.log; [ $rc -eq 0 ] || { echo "FAILED $name" >> $R/cpu_checks.log; exit 1; }; }
step selfcheck $PY scripts/evidence_audit.py --selfcheck --out $R/selfcheck.json
step fixture_extent $PY scripts/extent_experiment.py --fixture $R/fixture --foris-root $FORIS
step fixture_cache $PY scripts/evidence_cache.py --fixture $R/fixture --foris-root $FORIS --pool-images 8 --pool-patches 64
step fixture_maps $PY scripts/evidence_audit.py --cache $R/fixture/cache --run $R/fixture/run --out $R/fixture/audit --device cpu
step fixture_analysis $PY scripts/evidence_audit.py --analyse --run $R/fixture/run --out $R/fixture/audit
step plan $PY scripts/prepare_evidence_queue.py --manifest results/extent_v1/episodes.json --extent-run results/extent_v1/run \
  --selfcheck $R/selfcheck.json --fixture-report $R/fixture/audit/maps_report.json --cache cache/evidence_v1 --out $R --plan $R/plan.json
step preflight $PY scripts/experiment_resource_guard.py --plan $R/plan.json --state-file $R/guard.json --preflight-only
echo ALL_DONE >> $R/cpu_checks.log
