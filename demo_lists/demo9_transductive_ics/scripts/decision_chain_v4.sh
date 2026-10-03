#!/bin/bash
# The whole decision queue from wherever it stands (a stage whose report is COMPLETED is skipped); every stage under the
# guard, which is asked again every 60 s while another job holds the GPU. Order: the chosen read-out inside the
# pipeline (development, confirmation); 4800 more training episodes, second fit, its reading inside the pipeline;
# scale alignment; the host-mask-only fit; the second look at matched scale; then, by the rule fixed before the
# readings (confirmation at original resolution: gain at least +2.5, interval above 0, at least 3 folds positive),
# the main table: training cache disjoint from the 4000 evaluation episodes, one fit of the frozen arm, one reading.
#   bash scripts/decision_chain_v4.sh [PID of a guard that is still running a stage]
cd /root/autodl-tmp/demo9_extent
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
R=results/decision_v1
M=results/decision_main
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
say() { echo "$(date +%H:%M:%S) $*"; }
state() { $PY -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" $1 ${2:-state} 2>/dev/null; }
done_() { [ "$(state $1)" = COMPLETED ]; }
guard() {  # plan, state file
  for TRY in $(seq 1 240); do
    CUDA_VISIBLE_DEVICES="" $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --preflight-only > ${2%.json}.preflight.log 2>&1 || { say "preflight failed: $1"; exit 1; }
    $PY scripts/experiment_resource_guard.py --plan $1 --state-file $2 --run > ${2%.json}.log 2>&1
    RC=$?
    [ "$(state $2)" = DEFER_FOREIGN_GPU ] || break
    [ $((TRY % 10)) -eq 1 ] && say "another job holds the GPU; asking again every 60 s for $1"
    sleep 60
  done
  say "guard finished rc=$RC for $1"
}
infer() {  # fit folder, tag
  ARM=$(state $R/$1/selection.json chosen)
  for SET in dev confirm; do
    done_ $R/infer_$2_$SET/report.json && continue
    EXTRA=""; [ $SET = dev ] && EXTRA="--packets results/extent_v1/run/packets"
    rm -rf $R/infer_$2_$SET
    $PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models $R/$1/models/${ARM/:/_}.pt --episodes $SUITE/${SET}_episodes.json $EXTRA \
        --out $R/infer_$2_$SET --plan $R/plan_infer_$2_$SET.json --workers 6 --timeout 2400 || exit 1
    guard $R/plan_infer_$2_$SET.json $R/guard_infer_$2_$SET.json
    say "original resolution, $1 $ARM on $SET: $(state $R/infer_$2_$SET/report.json)"
  done
}
[ -n "$1" ] && while kill -0 $1 2>/dev/null; do sleep 10; done
done_ $R/fit/report.json || { say "the first fit is not completed"; exit 1; }
CHOSEN=$(state $R/fit/selection.json chosen)
infer fit 1
if ! done_ $R/fit2/report.json; then
  if ! done_ cache/decision_v1/report_train_more_episodes.json; then
    $PY scripts/prepare_decision_queue.py cache --suite $SUITE --manifest $R/train_more_episodes.json --extent-run results/extent_v1/run \
        --fixture-report $R/fixture/decision/report.json --cache cache/decision_v1 --plan $R/plan_more.json --workers 6 --timeout 4500 || exit 1
    guard $R/plan_more.json $R/guard_more.json
    done_ cache/decision_v1/report_train_more_episodes.json || { say "more training episodes not completed"; exit 1; }
  fi
  ARMS=$($PY -c "print(','.join(dict.fromkeys(['$CHOSEN','unroll:layers','mlp:relations','conv:score'])))")
  rm -rf $R/fit2
  $PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $R/fit2 --plan $R/plan_fit2.json --timeout 5400 -- --arms $ARMS --curve 1800,3600 --curve-arms $CHOSEN,mlp:relations || exit 1
  guard $R/plan_fit2.json $R/guard_fit2.json
  say "second fit: $(state $R/fit2/report.json)"
fi
done_ $R/fit2/report.json && infer fit2 2
if ! done_ results/scale_align_v0/run/report.json; then
  guard results/scale_align_v0/plan.json results/scale_align_v0/guard.json
  say "scale alignment: $(state results/scale_align_v0/run/report.json)"
fi
if ! done_ $R/fit_agnostic/report.json; then
  rm -rf $R/fit_agnostic
  $PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $R/fit_agnostic --plan $R/plan_fit_agnostic.json --timeout 5400 -- \
      --arms convctx:agnostic,unroll:agnostic --curve "" || exit 1
  guard $R/plan_fit_agnostic.json $R/guard_fit_agnostic.json
  say "host-mask-only fit: $(state $R/fit_agnostic/report.json)"
fi
for SET in dev confirm; do
  done_ $R/zoom_$SET/report.json && continue
  rm -rf $R/zoom_$SET
  $PY scripts/prepare_decision_queue.py zoom --cache cache/decision_v1 --models $R/fit/models/${CHOSEN/:/_}.pt --episodes $SUITE/${SET}_episodes.json \
      --out $R/zoom_$SET --plan $R/plan_zoom_$SET.json --workers 6 --timeout 3600 || exit 1
  guard $R/plan_zoom_$SET.json $R/guard_zoom_$SET.json
  say "second look, $CHOSEN on $SET: $(state $R/zoom_$SET/report.json)"
done
GO=$($PY - <<PYEOF
import json
try:
    r = json.load(open("$R/infer_1_confirm/report.json"))["rows"]["readout"]
    print("GO" if r["gain"] >= 2.5 and r["ci95"][0] > 0 and sum(x > 0 for x in r["per_fold"]) >= 3 else "NO %+.2f" % r["gain"])
except Exception as err:
    print("NO", repr(err))
PYEOF
)
say "rule for the main table (first fit's arm $CHOSEN): $GO"
[ "$GO" = GO ] || { say "queue finished without the main table"; exit 0; }
LEAN="--lean"; case $CHOSEN in *:features) LEAN="";; esac
if ! done_ cache/decision_v1/report_train_episodes.json; then
  $PY scripts/prepare_decision_queue.py cache --suite $SUITE --manifest $M/train_episodes.json --extent-run results/extent_v1/run \
      --fixture-report $R/fixture/decision/report.json --cache cache/decision_v1 --plan $M/plan_cache.json --workers 6 --timeout 9000 $LEAN || exit 1
  guard $M/plan_cache.json $M/guard_cache.json
  done_ cache/decision_v1/report_train_episodes.json || { say "main-table training cache not completed"; exit 1; }
fi
if ! done_ $M/fit/report.json; then
  rm -rf $M/fit
  $PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out $M/fit --plan $M/plan_fit.json --timeout 7200 -- \
      --train-manifest $M/train_episodes.json --arms $CHOSEN,conv:score --curve 1125,2250 --curve-arms $CHOSEN || exit 1
  guard $M/plan_fit.json $M/guard_fit.json
  done_ $M/fit/report.json || { say "main-table fit not completed"; exit 1; }
fi
if ! done_ $M/infer/report.json; then
  rm -rf $M/infer
  $PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models $M/fit/models/${CHOSEN/:/_}.pt --episodes $M/eval_episodes.json \
      --packets results/extent_v1/run/packets --out $M/infer --plan $M/plan_infer.json --workers 6 --timeout 9000 || exit 1
  guard $M/plan_infer.json $M/guard_infer.json
fi
say "main table: $(state $M/infer/report.json)"
say "queue finished"
