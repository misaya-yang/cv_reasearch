#!/bin/bash
# The public COCO-20i list (seed 0, 1000 draws per fold) in batches of 150 draws per fold: per batch, complete FoRIS
# and the feature export (GPU), the delivered Astra modules (CPU pool), the counts (GPU), then the features are
# removed and the cumulative table is rewritten. Fixed arms and pairs frozen beforehand only; nothing is selected here.
set -e
W=${W:-/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9}; cd "$W"
PY=/root/miniconda3/bin/python; BASE=outputs/claude_official; mkdir -p $BASE
HOST=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PYTHONPATH=/root/demo4_cache/env $PY - <<'E'
import json, sys
sys.path.insert(0, "/root/autodl-tmp/demo9_extent"); sys.path.insert(0, "/root/autodl-tmp/demo9_extent/scripts")
from extent_experiment import coco_episodes
ref = json.load(open("/root/autodl-tmp/demo9_extent/results/extent_v1/episodes.json"))
common = {k: ref[k] for k in ("data_root", "annotation_root", "foris_root", "projection_basis")}
draws = {f: coco_episodes(ref["data_root"], f, 1000) for f in range(4)}
for r in ref["episodes"]:                                       # the frozen DEV draws must reappear at their indices
    if r["e"] < 1000: assert (r["c"], r["query"], r["support"]) == draws[r["fold"]][r["e"]], r
for b, lo in enumerate(range(0, 1000, 150)):
    rows = [dict(fold=f, e=e, c=draws[f][e][0], query=draws[f][e][1], support=draws[f][e][2], batch=f"official{b}") for f in range(4) for e in range(lo, min(lo + 150, 1000))]
    json.dump(dict(state="PREPARED", episodes=rows, protocol=dict(name="public COCO-20i list, seed 0, draws %d to %d of each fold" % (lo, min(lo + 150, 1000) - 1)), **common), open(f"outputs/claude_official/batch{b}.json", "w"))
    print(b, len(rows))
E
RUNS=""
for b in 0 1 2 3 4 5 6; do
  M=$BASE/batch$b.json; N=$($PY -c "import json;print(len(json.load(open('$M'))['episodes']))")
  if [ ! -f $BASE/run$b/sweep_counts_wide.npz ]; then
    rm -rf $BASE/root$b $BASE/run$b
    PYTHONPATH=$HOST $PY scripts/export_confirm_cache.py --manifest $M --out $BASE/root$b
    PYTHONPATH=/root/demo4_cache/env $PY scripts/run_recheck_sweep.py infer --root $BASE/root$b --manifest $M --astra external/astra_emd/external_mean_delete600_modules --out $BASE/run$b --workers 11 --expected $N
    PYTHONPATH=/root/demo4_cache/env $PY scripts/run_recheck_sweep.py sweep --root $BASE/root$b --out $BASE/run$b --grid wide --suffix _wide > $BASE/run$b.sweep.log 2>&1
    rm -rf $BASE/root$b/cache
  fi
  RUNS="$RUNS $BASE/run$b"
  PYTHONPATH=/root/demo4_cache/env $PY scripts/run_recheck_sweep.py aggregate --runs $RUNS --frozen outputs/recheck600_v1_frozen.json --out $BASE/cumulative.json
  echo BATCH_DONE $b
done
echo QUEUE_COMPLETED
