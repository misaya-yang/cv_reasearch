#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_gate_placement.sh' >/dev/null; do sleep 5; done
test -f results/gate_placement/image_mean.json || exit 1
cp staging/native/evaluate_mse.py tools/evaluate_mse.py
/root/miniconda3/bin/python tools/prepare_native_probe.py > logs/native_prepare.log 2>&1
for spec in static:neighborhood:context_gate_v1/neighborhood semantic:neighborhood_gated:context_gate_lr/neighborhood_gated time:neighborhood_time_gate:gate_controls/neighborhood_time_gate static_seed2028:neighborhood:seed_probe/neighborhood semantic_seed2028:neighborhood_gated:seed_probe/neighborhood_gated; do
 IFS=: read -r label mode branch <<< "$spec"
 out="results/native_probe_v1/$label.json"
 if [ -f "$out" ]; then continue; fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 12000 ]; do sleep 5; done
  log="logs/native_${label}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/evaluate_mse.py --mode "$mode" --adapter "results/$branch/resume.pt" --data assets/data/native_probe_v1/manifest.json --native-crop --save-reconstructions 2 --out "$out" > "$log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
 done
 test -f "$out" || exit 1
done
