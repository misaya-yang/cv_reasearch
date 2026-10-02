#!/bin/sh
# H2 with the full training set: recognisers on the segmenter's OWN backbone features vs DINOv2-L features, trained on
# ground-truth regions and on predicted regions of the training images (the segmenter has seen these images: see EXPERIMENTS.md A2). $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
while kill -0 $1 2>/dev/null; do sleep 10; done
T=m2f-swin-tiny-ade
$PY scripts/extract_regions.py --split train --fast --seg mask2former:models/$T:512 --encoder models/dinov2-large:dinov2l > logs/extract_${T}_train.log 2>&1
for src in own dinov2l; do for kinds in gt gt,pred; do k=$(echo $kinds | tr -d ,)
  $PY scripts/train_recogniser.py --train rf/ade20k_train_${T}_${src}.pt --val rf/ade20k_val_${T}_${src}.pt --kinds $kinds --out rec/${T}_${src}_train_${k}.pt > logs/rec_${T}_${src}_train_${k}.log 2>&1
done; done
export DEMO2_GPU_FRAC=0.35
$PY scripts/eval_fusion.py --seg mask2former:models/$T:512 --encoder models/dinov2-large:dinov2l --rec ext=dinov2l:regclf_dinov2l.pt \
    --rec own_gt=own:rec/${T}_own_train_gt.pt --rec own_gtpred=own:rec/${T}_own_train_gtpred.pt \
    --rec dino_gt=dinov2l:rec/${T}_dinov2l_train_gt.pt --rec dino_gtpred=dinov2l:rec/${T}_dinov2l_train_gtpred.pt \
    --out results/ade20k/h2_m2f-swin-tiny.json > logs/h2_m2f-swin-tiny.log 2>&1
echo QUEUE_B3_DONE
