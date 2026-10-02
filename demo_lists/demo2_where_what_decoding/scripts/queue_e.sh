#!/bin/sh
# H4: does a different / stronger frozen encoder make a better recogniser? DINOv2-L vs SigLIP2-so400m (vs both, once the joint recogniser exists).
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
export DEMO2_GPU_FRAC=0.35
run() { name=$1; shift; [ -f results/ade20k/$name.json ] || $PY scripts/eval_fusion.py "$@" --out results/ade20k/$name.json > logs/$name.log 2>&1; }
ENC="--encoder models/dinov2-large:dinov2l --encoder models/siglip2-so400m-p16-512:siglip2:512"
while [ ! -f "/root/demo2_cache/rec/dinov2l+siglip2_gt.pt" ]; do sleep 30; done
REC="--rec dino=dinov2l:rec/dinov2l_gt.pt --rec sig=siglip2:rec/siglip2_gt.pt --rec both=dinov2l+siglip2:rec/dinov2l+siglip2_gt.pt"
run h4_eomt-large --seg eomt:models/eomt-large-ade:512 $ENC $REC
run h4_m2f-swin-tiny --seg mask2former:models/m2f-swin-tiny-ade:512 $ENC $REC
run h4_m2f-swin-large --seg mask2former:models/m2f-swin-large-ade:640 $ENC $REC
echo QUEUE_E_DONE
