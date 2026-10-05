set -e
cd /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9
export PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
PY=/root/miniconda3/bin/python
M=launch/fresh600_v1/episodes.json
$PY scripts/export_confirm_cache.py --manifest $M --out outputs/fresh600_root
export PYTHONPATH=/root/demo4_cache/env
$PY scripts/run_recheck_sweep.py infer --root outputs/fresh600_root --manifest $M --astra external/astra_emd/external_mean_delete600_modules --out outputs/recheck_fresh600_v1 --workers 11 --expected 600
$PY scripts/run_recheck_sweep.py sweep --root outputs/fresh600_root --out outputs/recheck_fresh600_v1 --grid wide --suffix _wide
echo QUEUE_COMPLETED
