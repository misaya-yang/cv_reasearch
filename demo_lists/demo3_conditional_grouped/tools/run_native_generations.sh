#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -f results/native_adapt_v1/neighborhood_gated/report.json || exit 1
test ! -f results/native_generations_v1/report.json || exit 0
for attempt in 1 2 3; do
 while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 15000 ]; do sleep 5; done
 log="logs/native_generation_retry${attempt}.log"
 if /root/miniconda3/bin/python tools/sample_paired.py --pair --images 8 --manifest assets/data/coco_unseen_v1/manifest.json --static-adapter results/native_adapt_v1/neighborhood/resume.pt --gate-adapter results/native_adapt_v1/neighborhood_gated/resume.pt --out results/native_generations_v1 > "$log" 2>&1; then exit 0;fi
 if ! grep -q 'OutOfMemoryError' "$log"; then exit 1;fi
done
exit 1
