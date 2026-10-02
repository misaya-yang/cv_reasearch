#!/bin/sh
# Evaluation queue (ADE20K val, 2000 images): H1 control and diagnostics on every remaining segmenter. $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding; E="--encoder models/dinov2-large:dinov2l"; EXT="--rec ext=dinov2l:regclf_dinov2l.pt"
PIX="--pix lin=dinov2l:pix/dinov2l_mlp_cls.pt"
export DEMO2_GPU_FRAC=0.35
while kill -0 $1 2>/dev/null; do sleep 10; done
run() { name=$1; shift; [ -f results/ade20k/$name.json ] || $PY scripts/eval_fusion.py "$@" --out results/ade20k/$name.json > logs/$name.log 2>&1; }
run ens_segformer-b5 --seg segformer:models/segformer-b5-ade:640 --seg2 upernet=upernet:models/upernet-convnext-large:640 --views flip,0.75,1.5 $PIX $E $EXT
run h1_eomt-dinov3-large --seg eomt_dinov3:models/eomt-dinov3-large-ade:512 $PIX $E $EXT
run h1_upernet-convnext-large --seg upernet:models/upernet-convnext-large:640 $PIX $E $EXT
run h1_m2f-swin-base-in21k --seg mask2former:models/m2f-swin-base-in21k-ade:640 $PIX $E $EXT
run h1_m2f-swin-small --seg mask2former:models/m2f-swin-small-ade:512 $PIX $E $EXT
run h1_segformer-b2 --seg segformer:models/segformer-b2-ade:512 $PIX $E $EXT
run h1_segformer-b0 --seg segformer:models/segformer-b0-ade:512 $PIX $E $EXT
echo QUEUE_A7_DONE
