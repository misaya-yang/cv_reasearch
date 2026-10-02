#!/usr/bin/env bash
# Run only AFTER the instance is switched to GPU mode. Reuse existing packages.
set -euo pipefail
task_root="${1:-/root/autodl-tmp/demo1_sam}"
nvidia-smi
task_python="${2:-/root/miniconda3/bin/python}"
"$task_python" -c 'import torch, numpy; print("torch", torch.__version__, "CUDA build", torch.version.cuda, "numpy", numpy.__version__); assert torch.cuda.is_available(), "CUDA unavailable; inspect driver/runtime before experiments"'
"$task_python" "$task_root/tools/collect_environment.py" --output "$task_root/results/gpu_environment.json"
# No installs here. If a specific dependency/driver failure is found, repair only
# that issue after reading the captured environment; do not replace the base env.
