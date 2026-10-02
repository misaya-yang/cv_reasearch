#!/bin/sh
# Validation region features for cross-fitted recognisers. $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
while kill -0 $1 2>/dev/null; do sleep 10; done
for spec in "eomt:models/eomt-large-ade:512" "segformer:models/segformer-b5-ade:640" "mask2former:models/m2f-swin-large-ade:640"; do
  tag=$(basename $(echo $spec | cut -d: -f2))
  $PY scripts/extract_regions.py --split val --seg $spec --encoder models/dinov2-large:dinov2l > logs/extract_${tag}_val.log 2>&1
  for src in own dinov2l; do for kinds in gt gt,pred; do k=$(echo $kinds | tr -d ,)
    $PY scripts/train_recogniser.py --train rf/ade20k_val_${tag}_${src}.pt --kinds $kinds --crossfit --out rec/${tag}_${src}_cf_${k}.pt > logs/rec_${tag}_${src}_cf_${k}.log 2>&1
  done; done
done
echo QUEUE_B1_DONE
