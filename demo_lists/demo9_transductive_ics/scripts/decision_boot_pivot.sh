#!/bin/bash
# One GPU session, about one hour, two readings that decide what the paper is (PLAN.md, first section, "Pivot boot"):
# (D1 runs first; the packs T2 needs are completed on the CPU meanwhile.)
#   T2  the read-out fitted on COCO base classes, read WITHOUT refitting on packs of LVIS-92i, PASCAL-Part, PACO-Part,
#       SUIM and lung X-ray (300 episodes each) inside the complete FoRIS pipeline at original resolution;
#   D1  the read-out fitted ONLY on constructed pairs (no annotation): two constructions, three fits, read on DEV241,
#       and, if the written rule passes, once on CONFIRM600 inside the pipeline.
# No main table. A failed stage does not stop the other reading. The provider poweroff is requested at the very end
# (POWEROFF=0 keeps the instance on). Run on the instance in GPU mode:
#     cd /root/autodl-tmp/demo9_extent && nohup bash scripts/decision_boot_pivot.sh > results/decision_pivot_v1.log 2>&1 &
set -u
cd /root/autodl-tmp/demo9_extent || exit 1
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
SELF=/root/autodl-tmp/demo9_extent/selfmade_v2
PACKS=/root/autodl-tmp/demo9_extent/transfer_v1
R=results/decision_pivot_v1
LABEL=results/decision_v1/fit/models
WORKERS=${WORKERS:-6}
EPISODES=${EPISODES:-300}
say() { echo "$(date '+%F %H:%M:%S') $*"; }
state() { $PY -c "import json,sys,os;print(json.load(open(sys.argv[1])).get(sys.argv[2]) if os.path.exists(sys.argv[1]) else 'MISSING')" "$1" "${2:-state}" 2>/dev/null; }
guard_run() {  # plan, state file, allow-shutdown (1/0); returns the guard's status instead of leaving the script
  local SHUT=""; [ "${3:-0}" = "1" ] && SHUT="--allow-shutdown"
  local RC=1
  for TRY in $(seq 1 30); do
    CUDA_VISIBLE_DEVICES="" $PY scripts/experiment_resource_guard.py --plan "$1" --state-file "$2" --preflight-only > "${2%.json}.preflight.log" 2>&1 \
      || { say "preflight failed: $1 ($(tail -1 "${2%.json}.preflight.log" | cut -c1-200))"; return 1; }
    $PY scripts/experiment_resource_guard.py --plan "$1" --state-file "$2" --run $SHUT > "${2%.json}.log" 2>&1
    RC=$?
    [ "$(state "$2")" = DEFER_FOREIGN_GPU ] || [ $RC -eq 4 ] || break
    say "the GPU is held by a foreign job; retry $1 in 60 s"; sleep 60
  done
  say "guard rc=$RC: $1"
  return $RC
}
infer() {  # models file, manifest, output folder name
  [ "$(state "$R/$3/report.json")" = COMPLETED ] && return 0
  rm -rf "$R/$3"
  $PY scripts/prepare_decision_queue.py infer --cache cache/decision_v1 --models "$1" --episodes "$2" --out "$R/$3" \
      --plan "$R/plan_$3.json" --workers "$WORKERS" --timeout 2400 || return 1
  guard_run "$R/plan_$3.json" "$R/guard_$3.json" 0
  [ "$(state "$R/$3/report.json")" = COMPLETED ]
}

mkdir -p "$R" "$PACKS"
say "GPU before the boot: $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"
START=$(date +%s)

# --- T2: transfer without refitting ---------------------------------------------------------------------------
t2() {
  for D in lvis pascal_part paco_part suim lung; do
    [ "$(state "$PACKS/$D/pack_receipt.json")" = COMPLETED ] || { say "pack $D is missing: $(tail -1 "$R/pack_$D.log" 2>/dev/null | cut -c1-200)"; continue; }
    say "pack $D: $(state "$PACKS/$D/pack_receipt.json" episodes) episodes"
    infer "$LABEL/convctx_layers.pt" "$PACKS/$D/episodes.json" "transfer_$D" || say "transfer $D did not complete"
  done
}

# --- D1: fit on constructed pairs only ------------------------------------------------------------------------
d1() {
  if [ "$(state "$SELF/build_receipt.json")" != COMPLETED ]; then
    rm -rf "$SELF"
    $PY - <<PYEOF
import json
done = dict(path="$SELF/build_receipt.json", json_equals=dict(state="COMPLETED"))
plan = dict(platform="autodl", cuda_python="$PY", protocol="one pass; the card is in the head of decision_selfmade_pairs.py",
  stages=[dict(name="selfmade_pairs", kind="gpu", cwd="/root/autodl-tmp/demo9_extent", timeout_seconds=2400,
    argv=["$PY", "scripts/decision_selfmade_pairs.py", "--train-manifest", "$SUITE/train_episodes.json", "--out", "$SELF",
          "--images", "300", "--pairs-per-image", "4", "--mode", "mixed", "--seed", "0"],
    requires=[dict(path="$SUITE/train_episodes.json", json_equals=dict(state="PREPARED"))],
    code_files=["scripts/decision_selfmade_pairs.py"],
    env=dict(PYTHONPATH="$PYTHONPATH", DEMO9_CUDA_GUARD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    produces=[done], success_checks=[done])])
open("$R/plan_selfmade.json", "w").write(json.dumps(plan, indent=1))
PYEOF
    guard_run "$R/plan_selfmade.json" "$R/guard_selfmade.json" 0 || return 1
  fi
  say "constructed pairs: $(state "$SELF/build_receipt.json" pairs) ($(state "$SELF/build_receipt.json" made))"
  if [ "$(state cache/decision_v1/report_selfmade_episodes.json)" != COMPLETED ]; then
    rm -f cache/decision_v1/report_selfmade_episodes.json
    $PY scripts/prepare_decision_queue.py cache --suite "$SUITE" --manifest "$SELF/selfmade_episodes.json" \
        --extent-run results/extent_v1/run --fixture-report results/decision_v1/fixture/decision/report.json \
        --cache cache/decision_v1 --plan "$R/plan_cache.json" --workers "$WORKERS" --timeout 3600 --lean || return 1
    guard_run "$R/plan_cache.json" "$R/guard_cache.json" 0 || return 1
  fi
  for S in same paste all; do
    [ "$(state "$R/fit_$S/report.json")" = COMPLETED ] && continue
    MAN="$SELF/selfmade_$S.json"; [ $S = all ] && MAN="$SELF/selfmade_episodes.json"
    rm -rf "$R/fit_$S"
    $PY scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out "$R/fit_$S" --plan "$R/plan_fit_$S.json" \
        --workers 2 --timeout 3600 -- --train-manifest "$MAN" --arms pixel:relations,convctx:layers --curve "" --no-confirm || return 1
    guard_run "$R/plan_fit_$S.json" "$R/guard_fit_$S.json" 0 || say "fit on $S did not complete"
    [ -e "$R/fit_$S/models/pixel_relations.pt" ] && CUDA_VISIBLE_DEVICES="" $PY scripts/decision_selfmade_compare.py --label "$LABEL/pixel_relations.pt" \
        --made "$R/fit_$S/models/pixel_relations.pt" --out "$R/compare_$S.json" > "$R/compare_$S.log" 2>&1
  done
  $PY scripts/decision_pivot_verdict.py --root "$R" --stage d1
  if [ -e "$R/d1_choice.json" ]; then
    infer "$(state "$R/d1_choice.json" model)" "$SUITE/confirm_episodes.json" infer_d1_confirm || say "the confirmation reading did not complete"
  else
    say "D1: no fit passed the written rule; no confirmation reading"
  fi
}

# packs that the no-card preparation did not finish are written now, on the CPU, side by side, while D1 uses the GPU
EPISODES=$EPISODES bash scripts/build_packs_cpu.sh parallel > "$R/packs_boot.log" 2>&1 &
PACK_PID=$!
d1; say "D1 finished after $(( ($(date +%s) - START) / 60 )) min"
wait $PACK_PID; say "packs: $(tail -1 "$R/packs_boot.log")"
t2; say "T2 finished after $(( ($(date +%s) - START) / 60 )) min"
$PY scripts/decision_pivot_verdict.py --root "$R" --stage all | tee "$R/pivot.txt"

# --- close the session ----------------------------------------------------------------------------------------
if [ "${POWEROFF:-1}" = "1" ]; then
  $PY - <<PYEOF
import json
plan = dict(platform="autodl", cuda_python="$PY", protocol="session end",
  stages=[dict(name="session_end", kind="cpu", role="finalize", cwd="/root/autodl-tmp/demo9_extent", timeout_seconds=30,
    argv=["/bin/sh", "-c", "echo session_end \$(date '+%F %T') > $R/session_end.txt"],
    produces=[dict(path="$R/session_end.txt")], success_checks=[dict(path="$R/session_end.txt")])])
open("$R/plan_end.json", "w").write(json.dumps(plan, indent=1))
PYEOF
  guard_run "$R/plan_end.json" "$R/guard_end.json" 1
  say "boot finished; the provider shutdown was requested (foreign jobs are protected by the guard)"
else
  say "boot finished; POWEROFF=0, the instance stays on"
fi
