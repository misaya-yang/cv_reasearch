#!/bin/bash
# A ONLY. Preparation is default; no GPU boot or shutdown. Primary A+B plan prevails.
set -euo pipefail
ROOT=${ROOT:-/root/autodl-tmp/demo9_transductive_ics}
PREP=${PREP:-/root/autodl-tmp/sam3_preparation}
PY=${PY:-/root/miniconda3/bin/python}
SRC=${SRC:-$PREP/code}
CKPT=${CKPT:-$PREP/checkpoints/sam3.pt}
STATUS=${STATUS:-$PREP/download_status.json}
SUITE=${SUITE:-$ROOT/results/extent_head_t1_isolated_v1}
FORIS_ROOT=${FORIS_ROOT:-/root/autodl-tmp/demo9_extent/results/decision_v1}
R=${R:-$ROOT/results/sam3_v0}
ENVP=$PREP/python:$SRC:/root/demo4_cache/env
MODE=${1:---prepare}
APPROVED_LONG=${2:-}
cd "$ROOT"
mkdir -p "$R"
say() { echo "$(date +%H:%M:%S) $*"; }
state() { "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))' "$1" "${2:-state}" 2>/dev/null || true; }
cpu() { env PYTHONPATH="$ENVP" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$@"; }
prepare() {
  "$PY" - "$CKPT" "$STATUS" <<'PY'
import json,sys
from pathlib import Path
path,status=Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())
assert status['state']=='WEIGHT_VERIFIED'
assert status['actual_sha256']=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
assert path.stat().st_size==status['actual_bytes']==3450062241
assert path.resolve()==Path(status['path']).resolve()
PY
  for SET in smoke dev confirm; do
    MANIFEST="$SUITE/${SET}_episodes.json"
    FORIS="$FORIS_ROOT/infer_1_$SET/episodes.jsonl"
    EXTRA=()
    if [ "$SET" = smoke ]; then MANIFEST="$SUITE/dev_episodes.json"; FORIS="$FORIS_ROOT/infer_1_dev/episodes.jsonl"; EXTRA=(--limit 10); fi
    CHECK="$R/preflight_$SET.json"
    if [ ! -e "$CHECK" ]; then
      cpu scripts/sam3_stitch.py --preflight --manifest "$MANIFEST" --foris "$FORIS" --out "$CHECK" "${EXTRA[@]}"
    fi
    "$PY" - "$CHECK" "$MANIFEST" "$FORIS" "$ROOT/scripts/sam3_stitch.py" <<'PY'
import json,sys,hashlib
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['state']=='CPU_PROTOCOL_CHECKED'
for field,path in zip(('manifest_sha256','foris_sha256','source_sha256'),sys.argv[2:]):
 assert r[field]==hashlib.sha256(Path(path).read_bytes()).hexdigest(),'oldpreflight identity: preserve and HOLD'
PY
    "$PY" - "$ROOT" "$PY" "$SRC" "$CKPT" "$STATUS" "$MANIFEST" "$R" "$SET" "$ENVP" "$CHECK" "${EXTRA[@]}" <<'PY'
import json,sys
from pathlib import Path
root,py,source,checkpoint,status,manifest,result,name,env,check,*extra=sys.argv[1:]
out=str(Path(result)/name)
freeze=dict(path=str(Path(out)/'report_shard0.json'),json_equals={'state':'PREDICTIONS_FROZEN'})
argv=[py,str(Path(root)/'scripts/sam3_stitch.py'),'--manifest',manifest,'--sam3',source,
 '--checkpoint',checkpoint,'--checkpoint-status',status,'--out',out,'--memory-fraction','0.3','--resume']+extra
if name=='dev': argv+=['--reuse-predictions-from',str(Path(result)/'smoke')]
stage=dict(name='sam3_'+name,kind='gpu',cwd=root,timeout_seconds=900 if name=='smoke' else 3000,
 argv=argv,requires=[dict(path=checkpoint),dict(path=manifest,json_equals={'state':'PREPARED'}),
 dict(path=check,json_equals={'state':'CPU_PROTOCOL_CHECKED'}),dict(path=str(Path(source)/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))],
 code_files=[str(Path(root)/'scripts/sam3_stitch.py')],
 env=dict(PYTHONPATH=env,DEMO9_CUDA_GUARD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),produces=[freeze],success_checks=[freeze])
plan=Path(result)/('plan_sam3_'+name+'.json')
if plan.exists():
 old=json.loads(plan.read_text())
 if old.get('stages')!=[stage]: raise RuntimeError('Existing plan identity changed: preserve and HOLD')
else:
 plan.write_text(json.dumps(dict(platform='autodl',cuda_python=py,stages=[stage],
 protocol='A only: full predict-freeze, separate CPUscore; no boot/shutdown; known paired adaptation'),indent=1))
PY
  done
  say "A finite plans prepared; no GPU stage started"
}
case "$MODE" in
 --prepare) prepare; exit 0 ;;
 --run) prepare ;;
 *) say "usage: sam3_chain.sh --prepare | --run [--approved-long-budget]"; exit 2 ;;
esac
for SET in smoke dev confirm; do
 MANIFEST="$SUITE/${SET}_episodes.json"; FORIS="$FORIS_ROOT/infer_1_$SET/episodes.jsonl"; EXTRA=()
 if [ "$SET" = smoke ]; then MANIFEST="$SUITE/dev_episodes.json"; FORIS="$FORIS_ROOT/infer_1_dev/episodes.jsonl"; EXTRA=(--limit 10); fi
 S=$(state "$R/$SET/report_shard0.json")
 if [ "$S" != PREDICTIONS_FROZEN ] && [ "$S" != COMPLETED ]; then
   cpu scripts/experiment_resource_guard.py --plan "$R/plan_sam3_$SET.json" --state-file "$R/guard_$SET.json" --preflight-only
   # Foreign work causes HOLD; do not wait on paid GPU for an hour.
   "$PY" scripts/experiment_resource_guard.py --plan "$R/plan_sam3_$SET.json" --state-file "$R/guard_$SET.json" --run
   [ "$(state "$R/$SET/report_shard0.json")" = PREDICTIONS_FROZEN ] || { say "$SET did not freeze; preserve partials"; exit 1; }
 fi
 SCORE_EXTRA=(--resume)
 if [ "$SET" = smoke ]; then SCORE_EXTRA+=(--keep-candidates); fi
 cpu scripts/sam3_stitch.py --score --manifest "$MANIFEST" --out "$R/$SET" "${EXTRA[@]}" "${SCORE_EXTRA[@]}"
 cpu scripts/sam3_stitch.py --merge --manifest "$MANIFEST" --out "$R/$SET" --foris "$FORIS" "${EXTRA[@]}" > "$R/$SET/table.log"
 [ "$(state "$R/$SET/report.json")" = COMPLETED ] || exit 1
 if [ "$SET" = smoke ]; then
  cpu - "$R" "$SUITE" "$APPROVED_LONG" <<'PY'
import json,sys
from pathlib import Path
out,suite,approval=sys.argv[1:];out=Path(out);suite=Path(suite)
r=json.loads((out/'smoke/report_shard0.json').read_text())
remaining=sum(len(json.loads((suite/(n+'_episodes.json')).read_text())['episodes']) for n in ('dev','confirm'))-r['episodes']
estimate=r['elapsed_s']+remaining*r['mean_episode_s']+2*r['model_load_s']
receipt=dict(state='BUDGET_ESTIMATED',estimated_A_seconds=estimate,measurement='actual smoke10 FP32 two-arm time',
 extra_B_seconds='parent measures one-worker prefix; historical six-worker8min is inapplicable',
 smoke_predictions_reused_in_DEV=True,not_a_measured_total=True)
if estimate>1800 and approval!='--approved-long-budget':
 receipt['state']='HOLD_BUDGET_REVIEW';(out/'budget_gate.json').write_text(json.dumps(receipt,indent=1));print(json.dumps(receipt));raise SystemExit(77)
(out/'budget_gate.json').write_text(json.dumps(receipt,indent=1));print(json.dumps(receipt))
PY
 fi
 if [ "$SET" = dev ]; then
  cpu scripts/sam3_stitch.py --cleanup-candidates --out "$R/smoke"
  cpu - "$R/dev/report.json" "$R/protocol_gate.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());score=r['class_miou']['visual']
gate=dict(state='PASS' if score>=60 else 'HOLD_PROTOCOL_REVIEW',visual_DEV=score,
 if_below60='compare3 identical episodes with public evaluator, not default implementation-error attribution',
 if_above70='standard public episode protocol required before claims; no automatic stage')
Path(sys.argv[2]).write_text(json.dumps(gate,indent=1));print(json.dumps(gate))
if score<60: raise SystemExit(78)
PY
 fi
done
say "A completed; decision review belongs to primary A+B orchestrator"
