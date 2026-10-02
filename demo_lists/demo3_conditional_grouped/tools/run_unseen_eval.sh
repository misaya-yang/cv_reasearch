#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_gate_controls.sh' >/dev/null; do sleep 5; done
test -f results/gate_controls/neighborhood_time_gate/report.json || exit 1
wait_memory() {
 while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt "$1" ]; do sleep 5; done
}
data=assets/data/coco_unseen_v1/manifest.json
if [ ! -f assets/data/coco_unseen_v1/text_cache/contract.json ]; then
 for attempt in 1 2 3; do
  wait_memory 6000
  if /root/miniconda3/bin/python tools/cache_text.py --data "$data" --batch 2 > "logs/unseen_text_retry${attempt}.log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "logs/unseen_text_retry${attempt}.log"; then exit 1; fi
 done
 test -f assets/data/coco_unseen_v1/text_cache/contract.json || exit 1
fi
for spec in static:neighborhood:context_gate_v1/neighborhood semantic:neighborhood_gated:context_gate_lr/neighborhood_gated gain:neighborhood_static_gate:gate_controls/neighborhood_static_gate time:neighborhood_time_gate:gate_controls/neighborhood_time_gate static_seed2028:neighborhood:seed_probe/neighborhood semantic_seed2028:neighborhood_gated:seed_probe/neighborhood_gated; do
 IFS=: read -r label mode branch <<< "$spec"
 out="results/unseen_v1/$label.json"
 if [ -f "$out" ]; then continue; fi
 for attempt in 1 2 3; do
  wait_memory 12000
  if /root/miniconda3/bin/python tools/evaluate_mse.py --mode "$mode" --adapter "results/$branch/resume.pt" --data "$data" --out "$out" > "logs/unseen_${label}_retry${attempt}.log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "logs/unseen_${label}_retry${attempt}.log"; then exit 1; fi
 done
 test -f "$out" || exit 1
done
if [ ! -f results/rollout_gate_lr/report.json ]; then
 for attempt in 1 2 3; do
  wait_memory 15000
  if /root/miniconda3/bin/python tools/sample_paired.py --only neighborhood_gated --gate-adapter results/context_gate_lr/neighborhood_gated/resume.pt --images 4 --out results/rollout_gate_lr > "logs/rollout_gate_lr_retry${attempt}.log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "logs/rollout_gate_lr_retry${attempt}.log"; then exit 1; fi
 done
 test -f results/rollout_gate_lr/report.json || exit 1
fi
