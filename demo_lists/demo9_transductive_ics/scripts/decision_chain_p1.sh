#!/bin/bash
# Stage P1, started only by the rule fixed before the P0 readings: when tonight's queue has ended and the chosen
# read-out, inside the pipeline at original resolution on the confirmation episodes, gains at least +2.5 with the
# interval above 0 and at least 3 folds positive. Then: training cache for the main table (image-disjoint from the
# 4000 evaluation episodes), one fit of the frozen arm, one reading on the standard 4 x 1000 episodes.
cd /root/autodl-tmp/demo9_extent
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
R=results/decision_v1
M=results/decision_main
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
say() { echo "$(date +%H:%M:%S) $*"; }
guard() {
  for TRY in $(seq 1 30); do
    CUDA_VISIBLE_DEVICES="" $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --preflight-only > ${2%.json}.preflight.log 2>&1 || { say "preflight failed: $1"; exit 1; }
    $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --run > ${2%.json}.log 2>&1
    RC=$?
    [ $RC -ne 4 ] && break
    say "the GPU is held by another job; attempt $TRY for $1 again in 60 s"
    sleep 60
  done
  say "guard finished rc=$RC for $1"
}
state() { $PY -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" $1 ${2:-state} 2>/dev/null; }
until grep -q "chain finished\|not completed\|preflight failed" $R/chain.log; do sleep 30; done
FIT=fit; TAG=1
[ "$(state $R/fit2/report.json)" = COMPLETED ] && [ "$(state $R/infer_2_confirm/report.json)" = COMPLETED ] && { FIT=fit2; TAG=2; }
GO=$($PY - <<PYEOF
import json
try:
    r = json.load(open("$R/infer_${TAG}_confirm/report.json"))["rows"]["readout"]
    print("GO" if r["gain"] >= 2.5 and r["ci95"][0] > 0 and sum(x > 0 for x in r["per_fold"]) >= 3 else "NO %+.2f" % r["gain"])
except Exception as err:
    print("NO", repr(err))
PYEOF
)
say "rule for P1 on $FIT: $GO"
[ "$GO" = GO ] || exit 0
ARM=$(state $R/$FIT/selection.json chosen)
LEAN="--lean"; case $ARM in *:features) LEAN="";; esac
$PY scripts/prepare_decision_queue.py cache --suite $SUITE --manifest $M/train_episodes.json --extent-run results/extent_v1/run \
    --fixture-report $R/fixture/decision/report.json --cache cache/decision_v1 --plan $M/plan_cache.json --workers 6 --timeout 9000 $LEAN || exit 1
guard $M/plan_cache.json $M/guard_cache.json
[ "$(state cache/decision_v1/report_train_episodes.json)" = COMPLETED ] || { say "main-table training cache not completed"; exit 1; }
$PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $M/fit --plan $M/plan_fit.json --timeout 7200 -- \
    --train-manifest $M/train_episodes.json --arms $ARM,conv:score --curve 2250,4500 --curve-arms $ARM || exit 1
guard $M/plan_fit.json $M/guard_fit.json
[ "$(state $M/fit/report.json)" = COMPLETED ] || { say "main-table fit not completed"; exit 1; }
$PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models $M/fit/models/${ARM/:/_}.pt --episodes $M/eval_episodes.json \
    --packets results/extent_v1/run/packets --out $M/infer --plan $M/plan_infer.json --workers 6 --timeout 9000 || exit 1
guard $M/plan_infer.json $M/guard_infer.json
say "main table: $(state $M/infer/report.json)"
say "P1 finished"
