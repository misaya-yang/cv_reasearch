#!/bin/bash
# Frozen SAM3 as FSS-SAM3 runs it, on our episodes: 10 episodes for speed, DEV241, CONFIRM600, then the tables on CPU.
# Each GPU stage under the guard, asked again every 60 s while another job holds the GPU. One process, FP32.
# The checkpoint is the one Codex downloads to /root/autodl-tmp/sam3_preparation (read only here; never written).
cd /root/autodl-tmp/demo9_extent
ENVP=/root/autodl-tmp/demo9_extent/sam3_env:/root/demo4_cache/env
PY=/root/miniconda3/bin/python
R=results/sam3_v0
SRC=/root/autodl-tmp/demo9_extent/sam3_source
CKPT=${CKPT:-/root/autodl-tmp/sam3_preparation/checkpoints/sam3.pt}
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
mkdir -p $R
say() { echo "$(date +%H:%M:%S) $*"; }
state() { $PY -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" $1 ${2:-state} 2>/dev/null; }
[ "$(stat -c %s $CKPT 2>/dev/null)" = 3450062241 ] || { say "checkpoint missing or incomplete: $CKPT"; exit 1; }
plan() {  # name, manifest, out folder, extra arguments
  $PY - "$@" <<PYEOF
import json, sys
name, manifest, out, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
root = "/root/autodl-tmp/demo9_extent"
done = {"path": "%s/%s/report_shard0.json" % (root, out), "json_equals": {"state": "COMPLETED"}}
stage = dict(name=name, kind="gpu", cwd=root, timeout_seconds=3000,
             argv=["$PY", root + "/scripts/sam3_stitch.py", "--manifest", manifest, "--sam3", "$SRC", "--checkpoint", "$CKPT", "--out", root + "/" + out] + extra,
             requires=[{"path": "$CKPT"}, {"path": manifest, "json_equals": {"state": "PREPARED"}}, {"path": "$SRC/sam3/assets/bpe_simple_vocab_16e6.txt.gz"}],
             code_files=[root + "/scripts/sam3_stitch.py"],
             env={"PYTHONPATH": "$ENVP", "DEMO9_CUDA_GUARD": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"},
             produces=[done], success_checks=[done])
json.dump(dict(platform="autodl", cuda_python="$PY", stages=[stage], protocol="one pass, no retries; the card is in scripts/sam3_stitch.py"),
          open("$R/plan_%s.json" % name, "w"), indent=1)
PYEOF
}
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
for SET in smoke dev confirm; do
  [ "$(state $R/$SET/report_shard0.json)" = COMPLETED ] && continue
  rm -rf $R/$SET
  if [ $SET = smoke ]; then plan sam3_smoke $SUITE/dev_episodes.json $R/smoke --limit 10; else plan sam3_$SET $SUITE/${SET}_episodes.json $R/$SET; fi
  guard $R/plan_sam3_$SET.json $R/guard_$SET.json
  S=$(state $R/$SET/report_shard0.json)
  say "$SET: $S, $(state $R/$SET/report_shard0.json episodes) episodes in $(state $R/$SET/report_shard0.json elapsed_s) s"
  [ "$S" = COMPLETED ] || { say "stopped: $SET did not complete"; exit 1; }
  FORIS=results/decision_v1/infer_1_$SET/episodes.jsonl
  if [ $SET = smoke ]; then PYTHONPATH=$ENVP CUDA_VISIBLE_DEVICES="" $PY scripts/sam3_stitch.py --merge --manifest $SUITE/dev_episodes.json --limit 10 --out $R/smoke > $R/smoke/table.log 2>&1
  else PYTHONPATH=$ENVP CUDA_VISIBLE_DEVICES="" $PY scripts/sam3_stitch.py --merge --manifest $SUITE/${SET}_episodes.json --out $R/$SET --foris $FORIS > $R/$SET/table.log 2>&1; fi
  say "$SET table: $($PY -c "import json;d=json.load(open('$R/$SET/report.json'));print(d['class_miou'], d.get('against_foris',{}).get('foris'))" 2>&1)"
done
say "sam3 chain finished"
