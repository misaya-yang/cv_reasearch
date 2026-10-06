#!/usr/bin/env python3
"""CPU-only actual remote preflight and successor-plan creation. Never launches GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

R=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9')
L=R/'launch/frozen_family_confirm1200_v1';S=L/'snapshot'
P='/root/miniconda3/bin/python';D=S/'scripts/run_frozen_family_confirm1200.py'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(S/'scripts'));sys.path.insert(0,str(S/'src'))
from ics.experiment import sha
from run_frozen_family_confirm1200 import MANIFEST_SHA,FREEZE_SHA,RECEIPT_SHA,VERIFY_SHA,EXPOSURE_SHA,FROZEN_PARAMS,ARMS,PRIMARY,storage

source=json.loads((L/'source_manifest.json').read_text())
for name,digest in source['files'].items():
    if sha(S/name)!=digest:raise ValueError('New immutable source changed: '+name)
assets=json.loads((L/'expected_assets.json').read_text())
for path,item in assets.items():
    if sha(path)!=item['sha256']:raise ValueError('Existing asset changed or absent: '+path)
env=dict(PYTHONPATH=str(S/'src')+':/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions',
    DEMO4_ROOT='/root/autodl-tmp/demo4',DEMO4_CACHE='/root/demo4_cache',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
checkenv=dict(os.environ,**env,CUDA_VISIBLE_DEVICES='')
preflight=R/'outputs/frozen_family_confirm1200_v1_prepared'
subprocess.run([P,str(D),'preflight','--out',str(preflight)],cwd=R,env=checkenv,check=True)
pre=json.loads((preflight/'preflight.json').read_text())
common=[dict(path=str(S/name),sha256=digest) for name,digest in source['files'].items()]
common += [dict(path=path,sha256=item['sha256']) for path,item in assets.items()]
common += [dict(path=str(R/path),sha256=digest) for path,digest in (
    ('outputs/photo_disjoint_confirm1200_v1/manifest.json',MANIFEST_SHA),
    ('outputs/photo_disjoint_confirm1200_v1/receipt.json',RECEIPT_SHA),
    ('outputs/photo_disjoint_confirm1200_v1/verification.json',VERIFY_SHA),
    ('outputs/photo_disjoint_confirm1200_v1/exposure.json',EXPOSURE_SHA),
    ('outputs/strict_family_confirm1200_v1_prepared/freeze.json',FREEZE_SHA))]
for directory in ('outputs/frozen_subtoken1200_v1','outputs/frozen_subtoken4000_v1'):
    for filename in ('manifest.json','config.json','sealed.json'):
        path=R/directory/filename;common.append(dict(path=str(path),sha256=sha(path)))
common.append(dict(path=str(preflight/'preflight.json'),sha256=sha(preflight/'preflight.json')))
ready=L/'dependency_ready.json';smoke=R/'outputs/frozen_family_confirm1200_smoke_v1';out=R/'outputs/frozen_family_confirm1200_v1'
ready_check=dict(path=str(ready),json_equals=dict(state='CONFIRM1200_PREDECESSOR_SEALED_READY'))
smoke_check=dict(path=str(smoke/'sealed.json'),json_equals=dict(state='SMOKE_PREDICTIONS_SEALED',n=3))
full_check=dict(path=str(out/'sealed.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=1200,primary=PRIMARY))
stages=[
    dict(name='wait_DirectMEAN4000_complete_seal',kind='cpu',cwd=str(R),env=env,
         argv=[P,str(D),'wait-inputs','--out',str(ready)],requires=common,produces=[ready_check]),
    dict(name='OLD3_exact_fullnative_qr_coarse_fine_operation_parity',kind='gpu',cwd=str(R),env=env,
         argv=[P,str(D),'smoke','--out',str(smoke)],requires=common+[ready_check],produces=[smoke_check],
         success_checks=[dict(path=str(smoke/'state.json'),json_equals=dict(state='SMOKE_COMPLETE',n=3)),
                         dict(path=str(smoke/'parity.json'),json_equals=dict(state='OLD3_OPERATION_PARITY_EXACT',n=3))]),
    dict(name='locked_photo_disjoint_CONFIRM1200_full_cold_inference',kind='gpu',cwd=str(R),env=env,
         argv=[P,str(D),'infer','--out',str(out)],requires=common+[ready_check,smoke_check],produces=[full_check],
         success_checks=[dict(path=str(out/'state.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=1200))]),
    dict(name='CPU6_ALL1200_complete_paired_GT_score_only_after_full_seal',kind='cpu',cwd=str(R),env=env,
         argv=[P,str(D),'score','--out',str(out),'--workers','6'],requires=common+[full_check],
         produces=[dict(path=str(out/'report.json')),dict(path=str(out/'report.md')),dict(path=str(out/'episodes.jsonl')),
                   dict(path=str(out/'score_state.json'),json_equals=dict(state='CPU_SCORE_COMPLETE',n=1200,primary=PRIMARY))])]
plan=dict(name='frozen_family_confirm1200_v1',stages=stages,primary=PRIMARY,arms=list(ARMS),
    comparison='One globally frozen training-free complete recipe versus all frozen strong simple same-information controls on one immutable photograph-disjoint cohort',
    manifest_sha256=MANIFEST_SHA,freeze_sha256=FREEZE_SHA,source_manifest_sha256=sha(L/'source_manifest.json'),
    fixed_recipe='C=(fine16 & scalar_graft) | (~fine16 & fine64)',parameters=FROZEN_PARAMS,
    cohort=dict(n=1200,per_fold=300,classes=74,absent_classes=[32,35,68,70,78,79],unique_photos=2249,
                photo_overlap_with17741_registered_exposed=0,official_public80class_leaderboard=False,
                no_resampling_rebalancing=True),
    production='actual complete full FoRIS same-shot paired FP32 forward/native CRF/final score/refcov/q/r; FP16 source tokens; same4 +/-4 query shifts; one shared FP32 cosine/kernel for all3 fine fields',
    CPU='6 concurrent source-identical RCG16/64 and original components.mean_control graph solves; bounded6 inflight pairs;12 RGB prefetch;2 writers',
    storage=storage(),query_GT='No query annotation pixels before all1200 complete arm outputs sealed; score only',
    statistics='class-summed I/U over74 observed classes; paired2000 RandomState(0) connected-photo95% intervals versus every control, folds/batches and addTP/addFP/deleteTP/deleteFP versus native',
    runtime='complete cold GPU runtime unmeasured; old3 smoke checks operations only',
    predecessor='DirectMEAN4000 complete mask/cosine seal; a declared failure stops, no retries',
    peer_policy='Default wait for every GPU PID; root alone may explicitly supply read-only coordinated live peer identity/state',
    automatic_shutdown=False,automatic_retry=False,GPU_launch_authority='root explicitly launches this concrete plan',
    terminal_states=dict(smoke='SMOKE_COMPLETE',inference='ALL_PREDICTIONS_SEALED',score='CPU_SCORE_COMPLETE'))
(L/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
valid=subprocess.run([P,str(S/'scripts/experiment_pipeline.py'),'--plan',str(L/'plan.json'),
    '--state-file',str(L/'state.json'),'--log-file',str(L/'events.jsonl')],cwd=R,env=checkenv,check=True,capture_output=True,text=True)
(L/'plan_validation.json').write_text(valid.stdout)
receipt=dict(state='ACTUAL_REMOTE_FROZEN_FAMILY_CONFIRM1200_PLAN_VALIDATED_NO_GPU',
    plan_sha256=sha(L/'plan.json'),source_manifest_sha256=sha(L/'source_manifest.json'),
    preflight_sha256=sha(preflight/'preflight.json'),manifest_sha256=MANIFEST_SHA,freeze_sha256=FREEZE_SHA,
    source_files=len(source['files']),existing_asset_files=len(assets),storage=storage(),free_bytes=pre['free_bytes'],
    pending_predecessor_seal=not (R/'outputs/direct_mean_fine4000_v1/sealed.json').is_file(),
    GPU_executed=False,query_truth_opened=False,
    remaining='root dispatch, GPU old3 exact parity, GPU all1200 cold inference, CPU post-seal score')
(L/'remote_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
