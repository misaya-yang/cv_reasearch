#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[b]ash tools/run_native_generations.sh' >/dev/null; do sleep 5; done
test -f results/native_generations_v1/report.json || exit 1
cp staging/envelope/model.py cgd/model.py
cp staging/envelope/evaluate_mse.py tools/evaluate_mse.py
for envelope in sigma reliability; do
 out="results/envelope_probe_v1/$envelope.json"
 if [ -f "$out" ]; then continue; fi
 for attempt in 1 2 3; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 12500 ]; do sleep 5; done
  log="logs/envelope_${envelope}_retry${attempt}.log"
  if /root/miniconda3/bin/python tools/evaluate_mse.py --mode neighborhood_gated --adapter results/native_adapt_v1/neighborhood_gated/resume.pt --data assets/data/native_adapt_v1/manifest.json --gate-envelope "$envelope" --save-reconstructions 2 --out "$out" > "$log" 2>&1; then exit_status=0;break;fi
  exit_status=1
  if ! grep -q 'OutOfMemoryError' "$log"; then exit 1;fi
 done
 test "$exit_status" -eq 0 || exit 1
done
