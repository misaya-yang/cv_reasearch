#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_native_probe.sh' >/dev/null; do sleep 5; done
test -f results/native_probe_v1/semantic_seed2028.json || exit 1
/root/miniconda3/bin/python tools/prepare_native_adapt.py > logs/native_adapt_prepare.log 2>&1
# Same static512-step checkpoint, reset AdamW both arms, seed2029,256 updates each.
# No automatic retraining after a mid-training failure: keep the512-update budget reviewable.
for mode in neighborhood neighborhood_gated; do
 out="results/native_adapt_v1/$mode"
 if [ -f "$out/report.json" ]; then continue; fi
 while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 14000 ]; do sleep 5; done
 available=$(/root/miniconda3/bin/python -c 'import os;s=os.statvfs(".");print(s.f_bavail*s.f_frsize)')
 if [ "$available" -lt 6500000000 ]; then echo 'Insufficient checkpoint reserve';exit 1;fi
 extra=()
 if [ "$mode" = neighborhood_gated ]; then extra=(--gate-lr 0.001);fi
 /root/miniconda3/bin/python tools/pilot.py --mode "$mode" --phase train --steps 256 --time-sampling stratified --seed 2029 "${extra[@]}" --warmstart results/context_gate_v1/neighborhood/resume.pt --data assets/data/native_adapt_v1/manifest.json --out "$out" > "logs/native_adapt_$mode.log" 2>&1
done
