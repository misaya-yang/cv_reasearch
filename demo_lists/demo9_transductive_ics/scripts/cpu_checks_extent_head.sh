#!/bin/bash
# Verify completed no-card preparation; never refit or reopen confirmation.
set -euo pipefail
cd /root/autodl-tmp/demo9_transductive_ics
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_py=/root/miniconda3/bin/python
"$task_py" - <<'PY'
import hashlib,json
from pathlib import Path
r=Path("results/extent_head_t1_preparation")
head=json.loads((r/"head_contract.json").read_text())
assert head["state"]=="PASSED" and head["channel_count"]==32 and head["relation_channels"]==16
assert head["checks"]["all_relation_maps_preserved"] and head["checks"]["reject_truncated_schema"]
assert head["head_source_sha256"]==hashlib.sha256(Path("scripts/extent_head.py").read_bytes()).hexdigest()
assert json.loads((r/"pca_contract_selfcheck.json").read_text())["state"]=="PASSED"
assert json.loads((r/"confirmation_contract.json").read_text())["state"]=="PASSED"
fixture=json.loads((r/"fixture_fit/confirmation/report.json").read_text())
assert fixture["state"]=="COMPLETED" and fixture["channel_count"]==32
assert fixture["exposure_registry"]["queryGT_opened_after_prediction_commit"]
assert fixture["exposure_registry"]["no_confirmation_fit"]
sampling=json.loads(Path("results/extent_head_t1_isolated_v1/sampling_receipt.json").read_text())
assert sampling["groups"]["training"]["episodes"]==2400 and sampling["groups"]["confirmation"]["episodes"]==600
assert sampling["UID_intersections_all_zero"]
readout=json.loads(Path("results/reference_removal_241_preparation.cpu.json").read_text())
assert readout["state"]=="CPU_REFERENCE_REMOVAL_CHECKED" and readout["tests"]==9
assert readout["source_sha256"]==hashlib.sha256(Path("scripts/reference_removal_241.py").read_bytes()).hexdigest()
print(json.dumps({"state":"CPU_PREPARATION_ARTIFACTS_VERIFIED","training_run_started":False,"CUDA_started":False}))
PY
"$task_py" scripts/experiment_resource_guard.py --plan results/extent_head_t1_preparation/plan.json --state-file results/extent_head_t1_preparation/guard.json --preflight-only
"$task_py" scripts/experiment_resource_guard.py --plan results/reference_removal_241_preparation.queue.json --state-file results/reference_removal_241_preparation.guard.json --preflight-only
