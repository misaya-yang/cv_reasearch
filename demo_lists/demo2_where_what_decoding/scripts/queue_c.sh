#!/bin/sh
# Recogniser strength (H4): ground-truth-region features of several frozen encoders, then recognisers. $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
while kill -0 $1 2>/dev/null; do sleep 10; done
for enc in "models/dinov2-large:dinov2l" "models/siglip2-so400m-p16-512:siglip2:512" "models/dinov2-large:dinov2l742:742:ratio"; do
  tag=$(echo $enc | cut -d: -f2)
  for split in val train; do $PY scripts/extract_regions.py --split $split --encoder $enc > logs/extract_noseg_${tag}_${split}.log 2>&1; done
  $PY scripts/train_recogniser.py --train rf/ade20k_train_noseg_${tag}.pt --val rf/ade20k_val_noseg_${tag}.pt --kinds gt --out rec/${tag}_gt.pt > logs/rec_${tag}_gt.log 2>&1
done
$PY scripts/train_recogniser.py --train rf/ade20k_train_noseg_dinov2l.pt --concat rf/ade20k_train_noseg_siglip2.pt --val rf/ade20k_val_noseg_dinov2l.pt \
    --val-concat rf/ade20k_val_noseg_siglip2.pt --kinds gt --out rec/dinov2l+siglip2_gt.pt > logs/rec_dinov2l+siglip2_gt.log 2>&1
echo QUEUE_C_DONE
