#!/bin/bash
# CPU only: the five transfer packs, one after the other (no model is loaded). Safe in no-card mode; a pack that
# does not finish here is rebuilt by scripts/decision_boot_pivot.sh.
cd /root/autodl-tmp/demo9_extent
export PYTHONPATH=/root/demo4_cache/env CUDA_VISIBLE_DEVICES=""
SUITE=/root/autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1
mkdir -p transfer_v1 results/decision_pivot_v1
one() {
  [ -e transfer_v1/$1/pack_receipt.json ] && return 0
  rm -rf transfer_v1/$1
  echo "$(date +%T) start $1"
  /root/miniconda3/bin/python scripts/decision_transfer_pack.py --dataset $1 --like $SUITE/dev_episodes.json --out transfer_v1/$1 --episodes ${EPISODES:-300} > results/decision_pivot_v1/pack_$1.log 2>&1
  echo "$(date +%T) $1 rc=$? $(tail -1 results/decision_pivot_v1/pack_$1.log | cut -c1-200)"
}
if [ "${1:-}" = parallel ]; then  # with real cores (GPU mode): the missing packs side by side
  for D in pascal_part lung lvis paco_part suim; do one $D & done
  wait
else
  for D in pascal_part lung lvis paco_part suim; do one $D; done
fi
echo "$(date +%T) packs finished"
