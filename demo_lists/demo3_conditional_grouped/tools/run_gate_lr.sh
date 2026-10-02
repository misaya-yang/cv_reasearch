#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=results/context_gate_lr/neighborhood_gated
test ! -f "$out/report.json" || exit 0
for attempt in 1 2 3; do
 while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 14000 ]; do sleep 5; done
 log="logs/context_gate_lr_retry${attempt}.log"
 if /root/miniconda3/bin/python tools/pilot.py --mode neighborhood_gated --phase train --steps 256 --time-sampling stratified --gate-lr 0.001 --warmstart results/time_probe/neighborhood-stratified/resume.pt --out "$out" > "$log" 2>&1; then exit 0; fi
 if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
 sleep 5
done
exit 1
