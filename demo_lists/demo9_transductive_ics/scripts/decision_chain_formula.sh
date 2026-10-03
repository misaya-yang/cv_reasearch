#!/bin/bash
# The one fixed formula (17 constants, the same for every fold; results/decision_v1/fit/models/formula_relations.pt,
# written by scripts/decision_rule_probe.py --export) inside the complete FoRIS pipeline at original resolution:
# confirmation first, then development. About 6 workers, about 110 episodes a minute when the GPU is free.
cd /root/autodl-tmp/demo9_extent
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
R=results/decision_v1
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
say() { echo "$(date +%H:%M:%S) $*"; }
state() { $PY -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" $1 ${2:-state} 2>/dev/null; }
guard() {
  for TRY in $(seq 1 60); do
    CUDA_VISIBLE_DEVICES="" $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --preflight-only > ${2%.json}.preflight.log 2>&1 || { say "preflight failed: $1"; exit 1; }
    $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --run > ${2%.json}.log 2>&1
    RC=$?
    [ "$(state $2)" = DEFER_FOREIGN_GPU ] || break
    sleep 60
  done
  say "guard finished rc=$RC for $1"
}
[ -f $R/fit/models/formula_relations.pt ] || { say "formula file missing"; exit 1; }
for SET in confirm dev; do
  [ "$(state $R/infer_formula_$SET/report.json)" = COMPLETED ] && continue
  EXTRA=""; [ $SET = dev ] && EXTRA="--packets results/extent_v1/run/packets"
  rm -rf $R/infer_formula_$SET
  $PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models $R/fit/models/formula_relations.pt --episodes $SUITE/${SET}_episodes.json $EXTRA \
      --out $R/infer_formula_$SET --plan $R/plan_infer_formula_$SET.json --workers 6 --timeout 2400 || exit 1
  guard $R/plan_infer_formula_$SET.json $R/guard_infer_formula_$SET.json
  say "formula at original resolution on $SET: $(state $R/infer_formula_$SET/report.json)"
  $PY -c "
import json;d=json.load(open('$R/infer_formula_$SET/report.json'));v=d['rows']['readout']
print('FoRIS %.2f  formula %+.2f [%+.2f, %+.2f]  folds %s  lose>10: %d' % (d['class_miou']['native'], v['gain'], *v['ci95'], ' '.join('%+.1f' % x for x in v['per_fold']), v['lose_more_than_10']))" 2>&1
done
say "formula chain finished"
