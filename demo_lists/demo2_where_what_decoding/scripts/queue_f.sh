#!/bin/sh
# Re-runs of the two Swin-L evaluations that ran out of GPU memory on the 1600x1600 validation image. $1 = pid to wait for.
PY=/root/miniconda3/bin/python; cd /root/autodl-tmp/demo2_where_what_decoding; E="--encoder models/dinov2-large:dinov2l"; EXT="--rec ext=dinov2l:regclf_dinov2l.pt"
PIX="--pix lin=dinov2l:pix/dinov2l_mlp_cls.pt"
export DEMO2_GPU_FRAC=0.35
while kill -0 $1 2>/dev/null; do sleep 10; done
run() { name=$1; shift; [ -f results/ade20k/$name.json ] || $PY scripts/eval_fusion.py "$@" --out results/ade20k/$name.json > logs/$name.log 2>&1; }
run ens_m2f-swin-large --seg mask2former:models/m2f-swin-large-ade:640 --seg2 eomt=eomt:models/eomt-large-ade:512 --views flip,0.75,1.25 $PIX $E $EXT
ENC="--encoder models/dinov2-large:dinov2l --encoder models/siglip2-so400m-p16-512:siglip2:512"
run h4_m2f-swin-large --seg mask2former:models/m2f-swin-large-ade:640 $ENC --rec dino=dinov2l:rec/dinov2l_gt.pt --rec sig=siglip2:rec/siglip2_gt.pt --rec both=dinov2l+siglip2:rec/dinov2l+siglip2_gt.pt
echo QUEUE_F_DONE
