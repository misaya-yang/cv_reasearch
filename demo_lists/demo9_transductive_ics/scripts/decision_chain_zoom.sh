#!/bin/bash
# Only the second look at matched scale with the first fit's chosen read-out, on development then confirmation
# (about 20 minutes of GPU). Each stage under the guard, asked again every 60 s while another job holds the GPU.
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
CHOSEN=$(state $R/fit/selection.json chosen)
for SET in dev confirm; do
  [ "$(state $R/zoom_$SET/report.json)" = COMPLETED ] && continue
  rm -rf $R/zoom_$SET
  $PY scripts/prepare_decision_queue.py zoom --cache cache/decision_v1 --models $R/fit/models/${CHOSEN/:/_}.pt --episodes $SUITE/${SET}_episodes.json \
      --out $R/zoom_$SET --plan $R/plan_zoom_$SET.json --workers 6 --timeout 3600 || exit 1
  guard $R/plan_zoom_$SET.json $R/guard_zoom_$SET.json
  say "second look, $CHOSEN on $SET: $(state $R/zoom_$SET/report.json)"
done
say "zoom finished"
