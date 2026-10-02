#!/bin/bash
# usage: run_cats.sh <backbone> <tag> <extra args or ""> cat1 cat2 ...   (sequential, one GPU)
cd /root/autodl-tmp/demo7; mkdir -p logs
BB=$1; TAG=$2; EXTRA=$3; shift 3
for c in "$@"; do
  PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python scripts/bank.py --cat $c --backbone $BB --tag $TAG $EXTRA > logs/bank_${c}${TAG}.log 2>&1
done
echo ALLDONE > logs/done${TAG}.flag
