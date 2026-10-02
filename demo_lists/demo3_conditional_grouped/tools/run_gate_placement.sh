#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_unseen_eval.sh' >/dev/null; do sleep 5; done
test -f results/rollout_gate_lr/report.json || exit 1
cp staging/placement/model.py cgd/model.py
cp staging/placement/evaluate_mse.py tools/evaluate_mse.py
# Two posthoc interventions, no retraining. Existing aligned scores are the control.
for placement in shifted image_mean; do
 out="results/gate_placement/$placement.json"
 if [ -f "$out" ]; then continue; fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 12000 ]; do sleep 5; done
  log="logs/placement_${placement}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/evaluate_mse.py --mode neighborhood_gated --adapter results/context_gate_lr/neighborhood_gated/resume.pt --data assets/data/coco_unseen_v1/manifest.json --gate-placement "$placement" --out "$out" > "$log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
 done
 test -f "$out" || exit 1
done
