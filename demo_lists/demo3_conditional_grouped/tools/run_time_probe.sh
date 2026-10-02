#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD" TMPDIR="$PWD/tmp" TRITON_CACHE_DIR="$PWD/cache/triton"
python=/root/miniconda3/bin/python
# From identical 128-step states; new 128 images/noise seeds match across forks.
for spec in conditional:native conditional:stratified fixed:stratified neighborhood:stratified original:stratified; do
 mode=${spec%:*};time_mode=${spec#*:};out="results/time_probe/$mode-$time_mode"
 if [ -f "$out/report.json" ]; then continue; fi
 backend=eager
 if [ "$mode" = conditional ] || [ "$mode" = fixed ]; then backend=triton; fi
 "$python" tools/pilot.py --mode "$mode" --phase train --steps 128 --time-sampling "$time_mode" --butterfly-backend "$backend" --resume "results/pilot_v1/$mode/resume.pt" --out "$out" > "logs/time_${mode}_${time_mode}.log" 2>&1
done
