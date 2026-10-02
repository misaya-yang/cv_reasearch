#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Do not change imported-source receipts while the preceding controller is active.
while pgrep -f '[b]ash tools/run_gate_lr.sh' >/dev/null; do sleep 5; done
test -f results/context_gate_lr/neighborhood_gated/report.json || exit 1
cp staging/combo/model.py cgd/model.py
cp staging/combo/pilot.py tools/pilot.py
out=results/context_stem_v1/neighborhood_full_stem
test ! -f "$out/report.json" || exit 0
for attempt in 1 2 3; do
 while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 15000 ]; do sleep 5; done
 log="logs/context_stem_retry${attempt}.log"
 if /root/miniconda3/bin/python tools/pilot.py --mode neighborhood_full_stem --phase train --steps 256 --time-sampling stratified --warmstart results/time_probe/neighborhood-stratified/resume.pt --out "$out" > "$log" 2>&1; then exit 0; fi
 if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
 sleep 5
done
exit 1
