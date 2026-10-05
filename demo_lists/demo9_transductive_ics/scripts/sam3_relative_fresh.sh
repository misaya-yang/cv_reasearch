#!/bin/bash
# Everything after sam3_chain.sh has done DEV241 and CONFIRM600 (docs/codex_plan/10_04.md, additions 1 to 3).
# Order, most valuable first, so that an interrupted session loses the least:
#   1. rest shard 0: SAM3 proposals; if the CONFIRM600 reading warrants it, scored and read once with the rule frozen
#      on DEV241, on its fresh part (no image shared with DEV241 or CONFIRM600).
#   2. NEW METHOD, measured: the backward check of query proposals on DEV241 (10-episode smoke first) and CONFIRM600,
#      and its AUC table (scripts/sam3_backward_check.py). No query label in the GPU stage.
#   3. rest shards 1..: proposals only; their query labels stay unopened (later fresh confirmations, scored on the CPU).
#   4. the backward check on rest shard 0, only if step 2 shows that it adds information on DEV241.
# rest = the standard seed-0 list minus the episodes of DEV241 and CONFIRM600, in SHARDS interleaved shards.
# No shutdown here: the caller powers off. PLAN_ONLY=1 writes the manifests and guard plans and runs nothing.
set -euo pipefail
ROOT=${ROOT:-/root/autodl-tmp/demo9_transductive_ics}
PREP=${PREP:-/root/autodl-tmp/sam3_preparation}
PY=${PY:-/root/miniconda3/bin/python}
SRC=${SRC:-$PREP/code}
CKPT=${CKPT:-$PREP/checkpoints/sam3.pt}
STATUS=${STATUS:-$PREP/download_status.json}
SUITE=${SUITE:-$ROOT/results/extent_head_t1_isolated_v1}
EVAL=${EVAL:-/root/autodl-tmp/demo9_extent/results/decision_main/eval_episodes.json}
READ=${READ:-$ROOT/results/sam3_relative_v1}
BANK=${BANK:-$ROOT/results/sam3_keep_v1}
OUT=$BANK/rest
SHARDS=${SHARDS:-3}
K=${K:-5}
MIN_FREE_GB=${MIN_FREE_GB:-8}
ENVP=$PREP/python:$SRC:/root/demo4_cache/env
cd "$ROOT"
say() { echo "$(date +%H:%M:%S) $*"; }
cpu() { env PYTHONPATH="$ENVP" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$@"; }
state() { "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("state"))' "$1" 2>/dev/null || true; }
guard() {  # plan, state file
  if [ "${PLAN_ONLY:-0}" = 1 ]; then say "plan only: $1"; return 0; fi
  cpu scripts/experiment_resource_guard.py --plan "$1" --state-file "$2" --preflight-only
  "$PY" scripts/experiment_resource_guard.py --plan "$1" --state-file "$2" --run
}
plan() {  # plan file, stage name, file whose state must become $4, that state, then the argv after the python executable
  local FILE=$1 NAME=$2 DONE=$3 WANT=$4; shift 4
  "$PY" - "$FILE" "$NAME" "$DONE" "$WANT" "$ROOT" "$PY" "$SRC" "$CKPT" "$ENVP" "$@" <<'PY'
import json, sys
from pathlib import Path
file, name, done, want, root, py, source, checkpoint, env, *argv = sys.argv[1:]
check = dict(path=done, json_equals={"state": want})
stage = dict(name=name, kind="gpu", cwd=root, timeout_seconds=3600, argv=[py] + argv,
             requires=[dict(path=checkpoint), dict(path=str(Path(source) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"))],
             code_files=[str(Path(root) / "scripts/sam3_stitch.py"), str(Path(root) / "scripts/sam3_backward_check.py")],
             env=dict(PYTHONPATH=env, DEMO9_CUDA_GUARD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
             produces=[check], success_checks=[check])
Path(file).write_text(json.dumps(dict(platform="autodl", cuda_python=py, stages=[stage], protocol="docs/codex_plan/10_04.md"), indent=1))
PY
}
predict() {  # shard index of the rest
  local S; S=$(state "$OUT/report_shard$1.json")
  if [ "$S" = PREDICTIONS_FROZEN ] || [ "$S" = COMPLETED ]; then say "rest shard $1 already predicted"; return 0; fi
  # explicit returns: inside `if` and `||` the shell does not stop a function at a failing command
  plan "$OUT/plan_rest_$1.json" "sam3_rest_$1" "$OUT/report_shard$1.json" PREDICTIONS_FROZEN "$ROOT/scripts/sam3_stitch.py" \
    --manifest "$OUT/rest_episodes.json" --sam3 "$SRC" --checkpoint "$CKPT" --checkpoint-status "$STATUS" --out "$OUT" \
    --memory-fraction 0.3 --resume --shard "$1/$SHARDS" || return 1
  guard "$OUT/plan_rest_$1.json" "$OUT/guard_rest_$1.json" || return 1
  [ "${PLAN_ONLY:-0}" = 1 ] && return 0
  say "rest shard $1 predicted: $(state "$OUT/report_shard$1.json")"
  [ "$(state "$OUT/report_shard$1.json")" = PREDICTIONS_FROZEN ]
}
backward() {  # name, run folder, manifest, then extra arguments of sam3_backward_check.py
  local NAME=$1 RUN=$2 MAN=$3; shift 3
  if [ "$(state "$READ/backward_$NAME.report.json")" = COMPLETED ]; then say "backward $NAME already done"; return 0; fi
  plan "$READ/plan_backward_$NAME.json" "backward_$NAME" "$READ/backward_$NAME.report.json" COMPLETED "$ROOT/scripts/sam3_backward_check.py" \
    --run "$RUN" --manifest "$MAN" --sam3 "$SRC" --checkpoint "$CKPT" --out "$READ/backward_$NAME.jsonl" --k "$K" "$@" || return 1
  guard "$READ/plan_backward_$NAME.json" "$READ/guard_backward_$NAME.json" || return 1
  [ "${PLAN_ONLY:-0}" = 1 ] && return 0
  say "backward $NAME: $(cat "$READ/backward_$NAME.report.json" 2>/dev/null | tr -d '\n' | cut -c1-200)"
  [ "$(state "$READ/backward_$NAME.report.json")" = COMPLETED ]
}

# 0. inputs and disk
[ -e "$EVAL" ] || { say "BLOCKER: the standard episode list is missing: $EVAL"; exit 3; }
mkdir -p "$OUT" "$READ"
FREE=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1048576)}')
[ "$FREE" -ge "$MIN_FREE_GB" ] || { say "BLOCKER: $FREE GB free under $OUT, $MIN_FREE_GB GB needed for the proposal bitmaps"; exit 3; }
"$PY" - "$EVAL" "$SUITE/dev_episodes.json" "$SUITE/confirm_episodes.json" "$OUT/rest_episodes.json" <<'PY'
import json, sys
ev, dev, conf, out = sys.argv[1:]
E = json.load(open(ev))
old = [r for p in (dev, conf) for r in json.load(open(p))["episodes"]]
seen = {(r["fold"], r["e"], r["c"]) for r in old}
images = {r[k] for r in old for k in ("support", "query")}
rows = [r for r in E["episodes"] if (r["fold"], r["e"], r["c"]) not in seen]
fresh = [r for r in rows if r["support"] not in images and r["query"] not in images]
if not rows:
    raise SystemExit("nothing left of the standard list")
print("standard list %d; in DEV241 or CONFIRM600 %d; rest %d, of which fresh (no shared image) %d; rest per fold %s" % (
    len(E["episodes"]), len(E["episodes"]) - len(rows), len(rows), len(fresh), [sum(r["fold"] == f for r in rows) for f in range(4)]))
E["episodes"] = rows
json.dump(E, open(out, "w"))
PY

# 1. rest shard 0, and its one reading when the CONFIRM600 reading warrants it
predict 0 || say "rest shard 0 did not complete"
if [ "$(state "$OUT/report_shard0.json")" = PREDICTIONS_FROZEN ] && "$PY" - "$READ/confirm.json" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))["reading"]
ok = v["ci95"][0] > 0 and v["gain_over_control"]["gain"] > 0
print("CONFIRM600: %s %+.2f [%+.2f, %+.2f] over absolute_0.5; %+.2f over the best fixed level (%s)" % (
    v["rule"], v["gain"], v["ci95"][0], v["ci95"][1], v["gain_over_control"]["gain"], v["control"]))
sys.exit(0 if ok else 1)
PY
then
  cpu scripts/sam3_stitch.py --score --manifest "$OUT/rest_episodes.json" --out "$OUT" --shard "0/$SHARDS" --resume --keep-candidates
  cpu scripts/sam3_relative_decision.py --run "$OUT" --manifest "$OUT/rest_episodes.json" --frozen-from "$READ/dev.json" \
      --fresh-from "$SUITE/dev_episodes.json" "$SUITE/confirm_episodes.json" --out "$READ/fresh.json"
  say "fresh confirmation written: $READ/fresh.json"
else
  say "no fresh reading (rest shard 0 has no predictions, or the CONFIRM600 reading does not warrant it); rest shard 0 stays unscored"
fi

# 2. the backward check: smoke on 10 episodes, then DEV241 and CONFIRM600, then the AUC tables (CPU)
MORE=0
if backward smoke "$BANK/dev" "$SUITE/dev_episodes.json" --limit 10 && backward dev "$BANK/dev" "$SUITE/dev_episodes.json"; then
  backward confirm "$BANK/confirm" "$SUITE/confirm_episodes.json" || say "backward confirm did not complete"
  if [ "${PLAN_ONLY:-0}" != 1 ]; then
    for SET in dev confirm; do
      [ -e "$READ/backward_$SET.jsonl" ] && { say "backward signal on $SET:"; cpu scripts/sam3_backward_check.py --report --run "$BANK/$SET" --out "$READ/backward_$SET.jsonl" || true; }
    done
    if "$PY" - "$READ/backward_dev.auc.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["proposals kept by relative_0.7"]["auc"]
best = max(v for k, v in a.items() if k not in ("forward_score", "presence") and v is not None)
print("DEV241, proposals kept by relative_0.7: forward score AUC %.3f, best backward signal AUC %.3f" % (a["forward_score"], best))
sys.exit(0 if best >= a["forward_score"] + 0.05 else 1)
PY
    then MORE=1; fi
  fi
else
  say "backward check did not complete on the smoke or on DEV241; the session continues with the proposal bank"
fi

# 3. the other rest shards: proposals only, labels unopened
I=1
while [ "$I" -lt "$SHARDS" ]; do predict "$I" || say "rest shard $I did not complete"; I=$((I + 1)); done

# 4. the backward check on rest shard 0, when it added information on DEV241
if [ "$MORE" = 1 ]; then
  backward rest0 "$OUT" "$OUT/rest_episodes.json" --shards 0 || say "backward rest0 did not complete"
else
  say "backward check on rest shard 0 skipped"
fi
say "proposal bank: $(du -sh "$BANK" 2>/dev/null | cut -f1) under $BANK; $(ls "$OUT/candidates" 2>/dev/null | wc -l) rest episodes with bitmaps"
