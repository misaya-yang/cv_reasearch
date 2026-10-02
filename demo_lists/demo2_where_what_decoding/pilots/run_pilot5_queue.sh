#!/bin/sh
# Runs pilot 5 on each model as soon as its weights are complete (hf_fetch removes .part files on success).
cd /root/autodl-tmp/demo2_pilot
P=/root/miniconda3/bin/python
run() {  # family dir short
  until { [ -f models/$2/model.safetensors ] || [ -f models/$2/pytorch_model.bin ]; } && ! ls models/$2/.*.part* >/dev/null 2>&1; do sleep 10; done
  PYTHONDONTWRITEBYTECODE=1 ITERS=4 $P pilot5_models.py $1 models/$2 $3 > pilot5_$2.log 2>&1
}
run segformer segformer-b5-ade 640
run mask2former m2f-swin-tiny-ade 512
run mask2former m2f-swin-large-ade 640
run upernet upernet-convnext-large 640
run segformer segformer-b2-ade 512
