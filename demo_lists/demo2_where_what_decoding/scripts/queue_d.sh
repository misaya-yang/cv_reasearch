#!/bin/sh
# H5: a second dataset. COCO panoptic collapsed to 133 semantic classes (val 5000; recogniser trained on a fixed 10k subset of train). $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding
while kill -0 $1 2>/dev/null; do sleep 10; done
for split in val train; do $PY scripts/extract_regions.py --dataset coco --split $split --encoder models/dinov2-large:dinov2l > logs/extract_coco_noseg_dinov2l_${split}.log 2>&1; done
$PY scripts/train_recogniser.py --C 133 --train rf/coco_train_noseg_dinov2l.pt --val rf/coco_val_noseg_dinov2l.pt --kinds gt --out rec/coco_dinov2l_gt.pt > logs/rec_coco_dinov2l_gt.log 2>&1
export DEMO2_GPU_FRAC=0.35; mkdir -p results/coco
run() { name=$1; shift; [ -f results/coco/$name.json ] || $PY scripts/eval_fusion.py --dataset coco "$@" --out results/coco/$name.json > logs/coco_$name.log 2>&1; }
run ens_m2f-swin-large --seg mask2former:models/m2f-swin-large-coco:800 --views flip --encoder models/dinov2-large:dinov2l --rec ext=dinov2l:rec/coco_dinov2l_gt.pt
run ens_m2f-swin-tiny --seg mask2former:models/m2f-swin-tiny-coco:800 --views flip --seg2 m2fl=mask2former:models/m2f-swin-large-coco:800 --encoder models/dinov2-large:dinov2l --rec ext=dinov2l:rec/coco_dinov2l_gt.pt
echo QUEUE_D_DONE
