#!/bin/bash
# GPU session: the second pass inside the query (scripts/sam3_second_pass.py), then region features of rest shards 1, 2.
# Smoke on 10 episodes first; every later stage runs even if one fails; the caller powers off.
cd /root/autodl-tmp/demo9_transductive_ics
S=results/extent_head_t1_isolated_v1 R=results/sam3_keep_v1 X=results/sam3_relative_v1 PREP=/root/autodl-tmp/sam3_preparation
export PYTHONPATH=$PREP/python:$PREP/code:/root/demo4_cache/env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
P=/root/miniconda3/bin/python
A="--sam3 $PREP/code --checkpoint $PREP/checkpoints/sam3.pt"
$P scripts/sam3_second_pass.py --run $R/dev --manifest $S/dev_episodes.json $A --out $R/smoke_second --limit 10 > $X/second_smoke.log 2>&1 || { echo "smoke failed"; tail -n 5 $X/second_smoke.log; exit 1; }
for n in dev confirm; do $P scripts/sam3_second_pass.py --run $R/$n --manifest $S/${n}_episodes.json $A --out $R/$n/second > $X/second_$n.log 2>&1; done
$P scripts/sam3_second_pass.py --run $R/rest --shards 0 --manifest $R/rest/rest_episodes.json $A --out $R/rest/second > $X/second_rest0.log 2>&1
$P scripts/sam3_region_features.py --run $R/rest --manifest $R/rest/rest_episodes.json --shards 1 2 --out $X/regions_rest12.npz > $X/regions_rest12.log 2>&1
tail -n 1 $X/second_*.log $X/regions_rest12.log
