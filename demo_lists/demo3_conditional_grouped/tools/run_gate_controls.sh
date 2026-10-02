#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_seed_probe.sh' >/dev/null; do sleep 5; done
test -f results/seed_probe/neighborhood_gated/report.json || exit 1
cp staging/gate_controls/model.py cgd/model.py
# Freeze this code before either control. Only 512 new updates total.
for mode in neighborhood_static_gate neighborhood_time_gate; do
 out="results/gate_controls/$mode"
 if [ -f "$out/report.json" ]; then continue; fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 14000 ]; do sleep 5; done
  available=$(/root/miniconda3/bin/python -c 'import os; s=os.statvfs(".");print(s.f_bavail*s.f_frsize)')
  if [ "$available" -lt 6500000000 ]; then echo 'Insufficient data-disk reserve for a checkpoint';exit 1;fi
  log="logs/control_${mode}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/pilot.py --mode "$mode" --phase train --steps 256 --time-sampling stratified --seed 2027 --gate-lr 0.001 --warmstart results/time_probe/neighborhood-stratified/resume.pt --out "$out" > "$log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
  sleep 5
 done
 test -f "$out/report.json" || exit 1
done
