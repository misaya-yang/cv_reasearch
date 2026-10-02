#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo3_conditional_grouped
if [ -f GPU_PAUSED ]; then echo "GPU_PAUSED: user requested CPU-only work" >&2; exit 2; fi
export PYTHONPATH="$PWD/runtime/python_packages:$PWD/assets/source/PixNerd:$PWD"
export HF_HOME="$PWD/cache/huggingface" TMPDIR="$PWD/tmp" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton" CUDA_CACHE_PATH="$PWD/cache/cuda"
mkdir -p logs results/resource_v1 results/pilot_v1 "$HF_HOME" "$TMPDIR"
python=/root/miniconda3/bin/python
while ! "$python" -c 'import pathlib; p=pathlib.Path("assets/checkpoints"); assert all((p/f).exists() for f in ["pixnerd/model.ckpt","qwen3/model-00001-of-00002.safetensors","qwen3/model-00002-of-00002.safetensors","qwen3/model.safetensors.index.json"]); assert pathlib.Path("assets/data/coco_pilot_v1/manifest.json").exists()' >/dev/null 2>&1; do sleep 20; done
"$python" tools/cache_text.py > logs/text_cache.log 2>&1
for mode in original conditional fixed group32 neighborhood dynamic full_stem; do
 "$python" tools/pilot.py --mode "$mode" --phase resource --out "results/resource_v1/$mode" > "logs/resource_$mode.log" 2>&1
done
if [ "${1:-}" != "--train" ]; then
 printf 'Resource probes complete. Matched training can continue with --train.\n'
 exit 0
fi
# Resource probe decides memory safety; concurrency is capped at two new jobs.
concurrency=$("$python" - <<'PY'
import json,pathlib,subprocess
peaks=[json.loads(p.read_text())['peak_reserved_bytes'] for p in pathlib.Path('results/resource_v1').glob('*/report.json')]
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
print(2 if 2*max(peaks)<.8*free else 1)
PY
)
printf 'pilot concurrency=%s; identical 128 steps / branch\n' "$concurrency"
# Stable paired inputs/noise/time are generated independently inside each process.
active=0
pids=()
for mode in original conditional fixed group32 neighborhood dynamic full_stem; do
 "$python" tools/pilot.py --mode "$mode" --phase train --steps 128 --out "results/pilot_v1/$mode" > "logs/train_$mode.log" 2>&1 &
 pids+=("$!")
 active=$((active+1))
 if [ "$active" -ge "$concurrency" ]; then
  for pid in "${pids[@]}"; do wait "$pid"; done
  active=0; pids=()
 fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$python" tools/summarize.py --root results/pilot_v1 > logs/summary.log 2>&1
