#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_combo.sh' >/dev/null; do sleep 5; done
test -f results/context_stem_v1/neighborhood_full_stem/report.json || exit 1
cp staging/seed/pilot.py tools/pilot.py
# Both arms share the same pretrained+256-step field; seed 2028 varies this continuation.
for mode in neighborhood neighborhood_gated; do
 out="results/seed_probe/$mode"
 if [ -f "$out/report.json" ]; then continue; fi
 extra=()
 if [ "$mode" = neighborhood_gated ]; then extra=(--gate-lr 0.001); fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 14000 ]; do sleep 5; done
  log="logs/seed_${mode}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/pilot.py --mode "$mode" --phase train --steps 256 --time-sampling stratified --seed 2028 "${extra[@]}" --warmstart results/time_probe/neighborhood-stratified/resume.pt --out "$out" > "$log" 2>&1; then break; fi
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1; fi
  sleep 5
 done
 test -f "$out/report.json" || exit 1
done
