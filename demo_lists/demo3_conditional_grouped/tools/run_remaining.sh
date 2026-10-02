#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" TRITON_CACHE_DIR="$PWD/cache/triton"
python=/root/miniconda3/bin/python
for mode in original fixed neighborhood dynamic full_stem; do
 if [ -f "results/pilot_v1/$mode/report.json" ]; then continue; fi
 # Other projects currently occupy this GPU; one new full model at a time.
 "$python" tools/pilot.py --mode "$mode" --phase train --steps 128 --out "results/pilot_v1/$mode" > "logs/train_$mode.log" 2>&1
done
"$python" tools/summarize.py --root results/pilot_v1 > logs/summary.log 2>&1
