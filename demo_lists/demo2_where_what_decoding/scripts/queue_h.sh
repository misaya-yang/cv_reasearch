#!/bin/sh
# Recognisers trained on the segmenter's own predicted regions of the training set, on the strongest segmenter (EoMT-L).
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
T=eomt-large-ade
[ -f /root/demo2_cache/rf/ade20k_train_${T}_own.pt ] || $PY scripts/extract_regions.py --split train --fast --seg eomt:models/$T:512 --encoder models/dinov2-large:dinov2l > logs/extract_${T}_train.log 2>&1
for src in own dinov2l; do for kinds in gt gt,pred; do k=$(echo $kinds | tr -d ,)
  $PY scripts/train_recogniser.py --train rf/ade20k_train_${T}_${src}.pt --val rf/ade20k_val_${T}_${src}.pt --kinds $kinds --out rec/${T}_${src}_train_${k}.pt > logs/rec_${T}_${src}_train_${k}.log 2>&1
done; done
export DEMO2_GPU_FRAC=0.35
$PY scripts/eval_fusion.py --seg eomt:models/$T:512 --encoder models/dinov2-large:dinov2l --rec ext=dinov2l:regclf_dinov2l.pt \
    --rec own_gt=own:rec/${T}_own_train_gt.pt --rec own_gtpred=own:rec/${T}_own_train_gtpred.pt \
    --rec dino_gt=dinov2l:rec/${T}_dinov2l_train_gt.pt --rec dino_gtpred=dinov2l:rec/${T}_dinov2l_train_gtpred.pt \
    --out results/ade20k/h2_eomt-large.json > logs/h2_eomt-large.log 2>&1
echo QUEUE_H_DONE
