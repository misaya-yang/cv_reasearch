#!/bin/bash
# After the decision cache: fit; the chosen read-out inside the complete pipeline at original resolution (development and
# confirmation); if any arm gains at least +1 on development, 4800 more training episodes, a second fit and its
# reading at original resolution; then scale alignment. Every step runs under the guard; stops at the first failure.
cd /root/autodl-tmp/demo9_extent
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
R=results/decision_v1
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
say() { echo "$(date +%H:%M:%S) $*"; }
guard() {  # plan, state file; when another job holds the GPU the guard returns 4 without starting: wait and try again
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
infer() {  # fit folder, tag
  ARM=$(state $R/$1/selection.json chosen)
  M=$R/$1/models/${ARM/:/_}.pt
  for SET in dev confirm; do
    EXTRA=""; [ $SET = dev ] && EXTRA="--packets results/extent_v1/run/packets"
    $PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models $M --episodes $SUITE/${SET}_episodes.json $EXTRA \
        --out $R/infer_$2_$SET --plan $R/plan_infer_$2_$SET.json --workers 6 --timeout 2400 || exit 1
    guard $R/plan_infer_$2_$SET.json $R/guard_infer_$2_$SET.json
    say "original resolution, $1 $ARM on $SET: $(state $R/infer_$2_$SET/report.json)"
  done
}
while pgrep -f "experiment_resource_guard.py --plan $R/plan_cache.json" > /dev/null; do sleep 10; done
[ "$(state cache/decision_v1/report.json)" = COMPLETED ] || { say "cache not completed"; exit 1; }
say "cache completed"
$PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $R/fit --plan $R/plan_fit.json --timeout 5400 || exit 1
guard $R/plan_fit.json $R/guard_fit.json
[ "$(state $R/fit/report.json)" = COMPLETED ] || { say "fit not completed"; exit 1; }
BEST=$($PY -c "import json;d=json.load(open('$R/fit/report.json'));print(max(v['over_foris'] for k,v in d['arms'].items() if not k.endswith(':score')))")
CHOSEN=$(state $R/fit/selection.json chosen)
say "fit completed: chosen $CHOSEN, best development gain $BEST"
infer fit 1
if $PY -c "import sys;sys.exit(0 if float('$BEST') >= 1.0 else 1)"; then
  $PY scripts/prepare_decision_queue.py cache --suite $SUITE --manifest $R/train_more_episodes.json --extent-run results/extent_v1/run \
      --fixture-report $R/fixture/decision/report.json --cache cache/decision_v1 --plan $R/plan_more.json --workers 6 --timeout 4500 || exit 1
  guard $R/plan_more.json $R/guard_more.json
  [ "$(state cache/decision_v1/report_train_more_episodes.json)" = COMPLETED ] || { say "more training episodes not completed"; exit 1; }
  ARMS=$($PY -c "print(','.join(dict.fromkeys(['$CHOSEN','mlp:relations','conv:score'])))")
  $PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $R/fit2 --plan $R/plan_fit2.json --timeout 5400 -- --arms $ARMS --curve 1800,3600 --curve-arms $ARMS || exit 1
  guard $R/plan_fit2.json $R/guard_fit2.json
  say "second fit: $(state $R/fit2/report.json)"
  [ "$(state $R/fit2/report.json)" = COMPLETED ] && infer fit2 2
else
  say "no arm reached +1 on development: more training episodes are not built"
fi
guard results/scale_align_v0/plan.json results/scale_align_v0/guard.json
say "scale alignment: $(state results/scale_align_v0/run/report.json)"
say "chain finished"
