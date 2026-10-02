#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_envelope_probe.sh' >/dev/null; do sleep 5; done
test -f results/envelope_probe_v1/reliability.json || exit 1
cp staging/cfg/model.py cgd/model.py
cp staging/cfg/sample_paired.py tools/sample_paired.py
# Index0 is a fixed ordinary case; index4 selected for overshoot diagnosis.
# Seeds stay31000 and31004, matching existing full-CFG4 records.
for spec in shared4:shared_uncond:4 independent3:independent:3; do
 IFS=: read -r label route cfg <<< "$spec"
 out="results/cfg_probe_v1/$label"
 if [ -f "$out/report.json" ]; then continue;fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 15000 ]; do sleep 5;done
  log="logs/cfg_${label}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/sample_paired.py --only neighborhood_gated --images 8 --indices 0 4 --guidance "$cfg" --cfg-gate "$route" --manifest assets/data/coco_unseen_v1/manifest.json --gate-adapter results/native_adapt_v1/neighborhood_gated/resume.pt --out "$out" > "$log" 2>&1; then break;fi
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1;fi
 done
 test -f "$out/report.json" || exit 1
done
