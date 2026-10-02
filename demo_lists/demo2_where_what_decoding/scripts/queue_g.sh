#!/bin/sh
# Consolidated runs with per-image statistics, for bootstrap intervals on the headline comparisons. $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
export DEMO2_GPU_FRAC=0.35
while kill -0 $1 2>/dev/null; do sleep 10; done
run() { name=$1; shift; [ -f results/ade20k/$name.json ] || $PY scripts/eval_fusion.py "$@" --out results/ade20k/$name.json > logs/$name.log 2>&1; }
ALL="--pix lin=dinov2l:pix/dinov2l_mlp_cls.pt --encoder models/dinov2-large:dinov2l --encoder models/siglip2-so400m-p16-512:siglip2:512 --rec ext=dinov2l:regclf_dinov2l.pt --rec both=dinov2l+siglip2:rec/dinov2l+siglip2_gt.pt --views flip"
run final_eomt-large --seg eomt:models/eomt-large-ade:512 $ALL
run final_m2f-swin-tiny --seg mask2former:models/m2f-swin-tiny-ade:512 $ALL
echo QUEUE_G_DONE
