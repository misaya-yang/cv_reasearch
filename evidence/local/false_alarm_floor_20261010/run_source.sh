#!/bin/bash
# One source through the in-mask instrument: fields in parallel shards, then the sealed evaluation.
#   bash evidence/local/false_alarm_floor_20261010/run_source.sh 6 --cohort a/coco_role_competition200_20261010 --name coco
#   bash evidence/local/false_alarm_floor_20261010/run_source.sh 6 --cohort a/lvis_mean200_20261008 --foris a/lvis_foris1400_score_reuse_20261009 --name lvis
#   bash evidence/local/false_alarm_floor_20261010/run_source.sh 6 --run /path/to/run_m4_baselines_output --dataset paco --name paco
# The first argument is the number of processes (each uses --threads, default 2). Set PYTHON to the environment's python.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
python="${PYTHON:-python3}"
shards="$1"; shift
logs="$here/inmask/logs"; mkdir -p "$logs"
stamp="$(date +%m%d_%H%M%S)"
# The first episode alone first: a wrong path or key stops here, not at three in the morning.
"$python" "$here/inmask_evidence.py" fields "$@" --shard 0/1000000007 > "$logs/$stamp.first.log" 2>&1 \
  || { echo "first episode failed, see $logs/$stamp.first.log"; tail -5 "$logs/$stamp.first.log"; exit 1; }
for k in $(seq 0 $((shards - 1))); do
  "$python" "$here/inmask_evidence.py" fields "$@" --shard "$k/$shards" > "$logs/$stamp.shard$k.log" 2>&1 &
done
wait
"$python" "$here/inmask_evidence.py" evaluate "$@" 2>&1 | tee "$logs/$stamp.evaluate.log"
