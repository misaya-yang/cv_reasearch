#!/usr/bin/env python3
"""Actual remote read-only input validation + owned immutable plan construction.
No model creation, CUDA operations, query-GT indexing or GPU stage execution.
"""
import hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
R=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9')
L=R/'launch/uniform_tau15_sizecut4000_v2';S=L/'snapshot'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(S/'scripts'));sys.path.insert(0,str(S/'src'))
from run_uniform_tau15_sizecut4000 import metadata,frozen_contract,FROZEN_SHA,MATCH_MANIFEST_SHA,MATCH_SEAL_SHA
from ics.experiment import sha
source=json.loads((L/'source_manifest.json').read_text())
for name,h in source['files'].items():
    if sha(S/name)!=h:raise ValueError('Snapshot checksum changed: '+name)
a=SimpleNamespace(base4000=R/'outputs/frozen_subtoken4000_v1',base4000_scored=R/'outputs/frozen_subtoken4000_scored_v1',base1200=R/'outputs/frozen_subtoken1200_v1',matching_run=R/'outputs/size_cut_composition_cached4000_v2')
rows,bs,bc,cached,c12,s12,matching,prior=metadata(a)
frozen=R/'outputs/claude_rcg2_frozen.json';frozen_contract(frozen)
prior_rows=[json.loads(x) for x in prior.read_text().splitlines()]
if len(prior_rows)!=4000 or {x['key'] for x in prior_rows}!={r['key'] for r in rows}:raise ValueError('Prior is not exact full4000 per-draw source')
expected_assets=json.loads((L/'expected_assets.json').read_text())
assets=[]
for path,item in expected_assets.items():
    p=Path(path)
    if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Actual source asset missing/changed: '+path)
    assets.append(dict(path=path,sha256=item['sha256']))
for row in rows:
    paths=[a.base4000/'fields'/(row['key']+'.npz'),a.base4000/'predictions'/(row['key']+'.npz'),Path(row['packet_export'])]
    if row['key'] in cached and row['fold'] in (0,3):paths.append(Path(cached[row['key']]['feature_export']))
    if row['key'] in matching:paths.extend(Path(matching[row['key']][k]) for k in ('prediction','field_source'))
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
selected=[next(r for r in rows if r['fold']==0 and r['key'] in cached),next(r for r in rows if r['fold']==1),next(r for r in rows if r['fold']==0 and r['key'] not in cached)]
import numpy as np
import torch
torch.set_num_threads(1)
headers=[]
for row in selected:
    key=row['key'];fp=a.base4000/'fields'/(key+'.npz');pp=a.base4000/'predictions'/(key+'.npz')
    if sha(fp)!=bs['fields'][key] or sha(pp)!=bs['predictions'][key] or sha(row['packet_export'])!=bs['inputs'][key]['packet_sha256']:raise ValueError('Smoke source receipt changed: '+key)
    with np.load(fp,allow_pickle=False) as z:
        if z['rcg'].shape!=(64,64) or z['rcg'].dtype!=np.float32 or z['fine.rcg16.control'].shape!=(128,128) or z['fine.rcg16.control'].dtype!=np.float32:raise ValueError('Smoke fields header changed')
    with np.load(row['packet_export'],allow_pickle=False) as z:
        cov=z['cov'];score=z['score']
        if cov.shape!=(64,64) or score.shape!=(64,64):raise ValueError('Packet allowed cov/score headers changed')
    record=dict(key=key,fold=row['fold'],cached=key in cached,query_truth_indexed=False)
    if key in cached:
        cr=cached[key];p=Path(cr['feature_export'])
        if sha(p)!=s12['inputs'][cr['key']]['feature_sha256']:raise ValueError('Cache hash changed')
        t=torch.load(p,map_location='cpu',weights_only=True)
        if any(t[k].shape!=(4096,1024) or t[k].dtype!=torch.float16 or not bool(torch.isfinite(t[k]).all()) for k in ('q','r')):raise ValueError('Cached q/r headers changed')
        record['feature_sha256']=sha(p)
    headers.append(record)
mean=R/'outputs/fine_mean1200_v1'
common=[dict(path=str(S/name),sha256=h) for name,h in source['files'].items()]+assets
for directory,names in ((a.base4000,('manifest.json','config.json','sealed.json')),(a.base1200,('manifest.json','config.json','sealed.json')),(a.matching_run,('primary_matching2000_manifest.json','sealed.json','source_receipt.json'))):
    for name in names:common.append(dict(path=str(directory/name),sha256=sha(directory/name)))
common.extend([dict(path=str(prior),sha256=sha(prior)),dict(path=str(frozen),sha256=FROZEN_SHA)])
mean_metadata=[dict(path=str(mean/n),sha256=sha(mean/n)) for n in ('config.json','manifest.json')]
P='/root/miniconda3/bin/python';D=str(S/'scripts/run_uniform_tau15_sizecut4000.py');O=R/'outputs/uniform_tau15_sizecut4000_v2';M=R/'outputs/uniform_tau15_sizecut4000_smoke_v2';ready=L/'original_mean_dependency_ready.json'
env=dict(PYTHONPATH=str(S/'src')+':/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions',DEMO4_ROOT='/root/autodl-tmp/demo4',DEMO4_CACHE='/root/demo4_cache',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
opts=['--base4000',str(a.base4000),'--base4000-scored',str(a.base4000_scored),'--base1200',str(a.base1200),'--matching-run',str(a.matching_run),'--existing-mean-run',str(mean),'--frozen',str(frozen),'--host-manifest',str(R/'outputs/claude_official/batch0.json'),'--workers','6','--readers','6','--writers','2','--prefetch','12','--inflight','6','--chunk','512']
ready_check=dict(path=str(ready),json_equals=dict(state='ORIGINAL_MEAN1200_SEALED_READY',n=1200))
smoke_check=dict(path=str(M/'sealed.json'),json_equals=dict(state='SMOKE_PREDICTIONS_SEALED',n=3))
full_check=dict(path=str(O/'sealed.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=4000))
stages=[dict(name='wait_original_mean1200_sealed',kind='cpu',cwd=str(R),env=env,argv=[P,D,'wait-mean','--out',str(ready),'--existing-mean-run',str(mean)],requires=common+mean_metadata,produces=[ready_check]),dict(name='three_case_producer_parity_smoke',kind='gpu',cwd=str(R),env=env,argv=[P,D,'smoke','--out',str(M)]+opts,requires=common+[ready_check],produces=[smoke_check],success_checks=[dict(path=str(M/'state.json'),json_equals=dict(state='SMOKE_COMPLETE',n=3))]),dict(name='fixed_uniform_tau15_sizecut_complete4000',kind='gpu',cwd=str(R),env=env,argv=[P,D,'infer','--out',str(O)]+opts,requires=common+[ready_check,smoke_check],produces=[full_check],success_checks=[dict(path=str(O/'state.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=4000))]),dict(name='score_all4000_baseline_parity',kind='cpu',cwd=str(R),env=env,argv=[P,D,'score','--out',str(O),'--workers','6'],requires=common+[full_check],produces=[dict(path=str(O/'report.json')),dict(path=str(O/'episodes.jsonl')),dict(path=str(O/'score_state.json'),json_equals=dict(state='CPU_SCORE_COMPLETE',n=4000))])]
plan=dict(name='uniform_tau15_sizecut4000_v2',stages=stages,scientific_scope='same global fresh600 frozen operator; six original complete controls; no fitting or changed threshold/readout; reused DEV4000 draws including repeated identities',requested_mean_policy='Wait for healthy original Mean1200 seal; no duplicated forward, output mutation, process adoption or signal',runtime_expectation='2000 four-shift query readouts and1400 Part1 prefix pairs; matching2000 reused; measured smoke will establish throughput; cold full producer cost separate',resources=dict(cpu_workers=6,reader_threads=6,writer_threads=2,gpu_peak_allocation_budget_bytes=10<<30,cosine_cache_bytes=2000*16384*25*4,remaining_disk_headroom_min_bytes=5<<30),peer_policy='Default runner waits all GPU PIDs. Root may explicitly opt in only to read-only exact PID/start_ticks/argv/cwd verified shared_peer_processes; no automatic signals or peer ownership',terminal_states=dict(smoke='SMOKE_COMPLETE',inference='ALL_PREDICTIONS_SEALED',score='CPU_SCORE_COMPLETE'),query_gt='no indexing during wait/inference/smoke; only CPU score after all4000 masks seal')
(L/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
receipt=dict(state='ACTUAL_REMOTE_STATIC_DEPENDENCIES_VERIFIED_NO_GPU_EXECUTION',source_files=len(source['files']),source_manifest_sha256=sha(L/'source_manifest.json'),plan_sha256=sha(L/'plan.json'),draws=4000,cached_missing_queries=600,prefix_missing_queries=1400,matching_reused=2000,query_four_shift=2000,prior_episodes=str(prior),prior_sha256=sha(prior),smoke_headers=headers,assets=assets,mean_dependency_sealed=(mean/'sealed.json').is_file(),CUDA_VISIBLE_DEVICES='',query_truth_indexed=False)
(L/'remote_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('assets','smoke_headers')}),flush=True)
