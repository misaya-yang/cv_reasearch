#!/usr/bin/env python3
"""Validate static existing inputs and prepare future full4000 control plan.
No GPU/model/GT access. Future Mean/cos artifacts remain explicit dependencies.
"""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
R=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9');L=R/'launch/direct_mean_fine4000_v1';S=L/'snapshot'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(S/'scripts'));sys.path.insert(0,str(S/'src'))
from run_direct_mean_fine4000 import metadata,identity,GSEAL,ARMS
from ics.experiment import sha,render,unpack
source=json.loads((L/'source_manifest.json').read_text())
for name,h in source['files'].items():
    if sha(S/name)!=h:raise ValueError('New snapshot source checksum changed: '+name)
a=SimpleNamespace(base4000=R/'outputs/frozen_subtoken4000_v1',base4000_scored=R/'outputs/frozen_subtoken4000_scored_v1',base1200=R/'outputs/frozen_subtoken1200_v1',graft_run=R/'outputs/mean_fine_residual_transfer4000_v1')
rows,bc,bs,old,c12,s12,gc,gs,prior,_=metadata(a,dependencies=False)
if {f:(p['sigma'],p['tau']) for f,p in c12['parameters'].items()}!={str(f):(1.25,.07 if f in (0,3) else .15) for f in range(4)}:raise ValueError('Recorded original per-fold policy changed')
mean=R/'outputs/fine_mean1200_v1';uniform=R/'outputs/uniform_tau15_sizecut4000_v2'
mean_config=json.loads((mean/'config.json').read_text())
if mean_config['parameters']!=c12['parameters']:raise ValueError('Active original Mean1200 policy differs')
if len({r['key'] for r in rows if r['key'] not in old and r['fold'] in (0,3)})!=1400 or len({r['key'] for r in rows if r['key'] not in old and r['fold'] in (1,2)})!=1400:raise ValueError('Required reuse counts changed')
prior_rows=[json.loads(s) for s in prior.read_text().splitlines()]
if len(prior_rows)!=4000 or {r['key'] for r in prior_rows}!={r['key'] for r in rows}:raise ValueError('All4000 original prior cohort required')
selected=[next(r for r in rows if r['key'] in old and r['fold']==0),next(r for r in rows if r['key'] not in old and r['fold']==0),next(r for r in rows if r['key'] not in old and r['fold']==1)]
import numpy as np
checks=[]
for row in selected:
    key=row['key'];gp=a.graft_run/'fields'/(key+'.npz');fp=a.base4000/'fields'/(key+'.npz');pp=a.base4000/'predictions'/(key+'.npz')
    for p,h in ((gp,gs['fields'][key]),(fp,bs['fields'][key]),(pp,bs['predictions'][key]),(Path(row['packet_export']),bs['inputs'][key]['packet_sha256'])):
        if sha(p)!=h:raise ValueError('Three-case existing source receipt changed: '+str(p))
    with np.load(gp,allow_pickle=False) as z:G=z['mean.control'].copy()
    with np.load(fp,allow_pickle=False) as z:R64=z['rcg'];Y=z['fine.rcg16.control']
    with np.load(pp,allow_pickle=False) as z:M=z['mean.control'].copy()
    if G.shape!=(64,64) or G.dtype!=np.float32 or not np.isfinite(G).all() or R64.shape!=(64,64) or Y.shape!=(128,128):raise ValueError('Existing allowed field header changed')
    if np.count_nonzero(render(G)!=unpack(M)):raise ValueError('Actual G original MEAN mask replay failed')
    checks.append(dict(key=key,fold=row['fold'],original1200=key in old,G64_mean_mask_mismatches=0,query_truth_indexed=False))
assets=[]
for path,item in json.loads((L/'expected_assets.json').read_text()).items():
    if not Path(path).is_file() or sha(path)!=item['sha256']:raise ValueError('Existing model/basis/source asset changed or missing: '+path)
    assets.append(dict(path=path,sha256=item['sha256']))
common=[dict(path=str(S/name),sha256=h) for name,h in source['files'].items()]+assets
for directory in (a.base4000,a.base1200,a.graft_run):
    for name in ('manifest.json','config.json','sealed.json'):common.append(dict(path=str(directory/name),sha256=sha(directory/name)))
common.append(dict(path=str(prior),sha256=sha(prior)))
for name in ('config.json','manifest.json'):common.append(dict(path=str(mean/name),sha256=sha(mean/name)))
O=R/'outputs/direct_mean_fine4000_v1';M=R/'outputs/direct_mean_fine4000_smoke_v1';ready=L/'dependency_ready.json';P='/root/miniconda3/bin/python';D=str(S/'scripts/run_direct_mean_fine4000.py')
env=dict(PYTHONPATH=str(S/'src')+':/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions',DEMO4_ROOT='/root/autodl-tmp/demo4',DEMO4_CACHE='/root/demo4_cache',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
opts=['--base4000',str(a.base4000),'--base4000-scored',str(a.base4000_scored),'--base1200',str(a.base1200),'--mean1200',str(mean),'--uniform-run',str(uniform),'--graft-run',str(a.graft_run),'--uniform-pipeline-state',str(R/'launch/uniform_tau15_sizecut4000_v2/state.json'),'--host-manifest',str(R/'outputs/claude_official/batch0.json'),'--workers','6','--readers','6','--writers','2','--prefetch','12','--inflight','6','--chunk','512']
ready_check=dict(path=str(ready),json_equals=dict(state='DIRECT_MEAN4000_DEPENDENCIES_SEALED_READY'))
smoke_check=dict(path=str(M/'sealed.json'),json_equals=dict(state='SMOKE_PREDICTIONS_SEALED',n=3))
full_check=dict(path=str(O/'sealed.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=4000))
stages=[dict(name='wait_mean1200_and_uniform4000_cos_sealed',kind='cpu',cwd=str(R),env=env,argv=[P,D,'wait-inputs','--out',str(ready)]+opts,requires=common,produces=[ready_check]),dict(name='three_path_exact_producer_parity',kind='gpu',cwd=str(R),env=env,argv=[P,D,'smoke','--out',str(M)]+opts,requires=common+[ready_check],produces=[smoke_check],success_checks=[dict(path=str(M/'state.json'),json_equals=dict(state='SMOKE_COMPLETE',n=3))]),dict(name='exact_direct_mean_fine_all4000',kind='gpu',cwd=str(R),env=env,argv=[P,D,'infer','--out',str(O)]+opts,requires=common+[ready_check,smoke_check],produces=[full_check],success_checks=[dict(path=str(O/'state.json'),json_equals=dict(state='ALL_PREDICTIONS_SEALED',n=4000))]),dict(name='score4000_six_baseline_exact_parity',kind='cpu',cwd=str(R),env=env,argv=[P,D,'score','--out',str(O),'--workers','6'],requires=common+[full_check],produces=[dict(path=str(O/'report.json')),dict(path=str(O/'episodes.jsonl')),dict(path=str(O/'score_state.json'),json_equals=dict(state='CPU_SCORE_COMPLETE',n=4000,new_control='fine.mean16.control'))])]
plan=dict(name='direct_mean_fine4000_v1',stages=stages,comparison='Full4000 strong simple same-information control, exact extension of original1200 fine.mean16.control; original fine64 primary unchanged',parameters=c12['parameters'],G64='Existing all4000 original MEAN_a.25/lambda16, never graft C128; each source G render checked against original complete MEAN',reuse=dict(original1200=1200,existing_fold03_cos=1400,new_fold12_prefix_fourshift=1400),cache=dict(new_cos_bytes=1400*16384*25*4,all_existing_assets_retained=True,minimum_free_bytes_before_full=4<<30),resource=dict(cpu_workers=6,reader_threads=6,writer_threads=2,bounded_inflight=6,gpu_peak_allocation_budget_bytes=10<<30),runtime='Actual3path smoke and measured full timings pending; do not claim incremental cache runtime as complete cold producer',dependency_policy='CPU waits for healthy original Mean1200 and uniform4000 complete cos seal; explicit predecessor failure stops; no GPU launch/signals until root execution',peer_policy='Default waits all GPU PIDs; root may explicitly opt-in read-only exact PID/start_ticks/argv/cwd shared peers; ownership none, no signals',query_GT='First indexed only CPUscore after ALL4000_PREDICTIONS_SEALED',terminal_states=dict(smoke='SMOKE_COMPLETE',inference='ALL_PREDICTIONS_SEALED',score='CPU_SCORE_COMPLETE'))
(L/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
receipt=dict(state='ACTUAL_REMOTE_STATIC_DIRECT_MEAN4000_INPUTS_VERIFIED_NO_GPU_EXECUTION',source_files=len(source['files']),source_manifest_sha256=sha(L/'source_manifest.json'),plan_sha256=sha(L/'plan.json'),G64_seal_sha256=GSEAL,G64_key='mean.control',prior_episodes=str(prior),prior_sha256=sha(prior),three_case_G64_header_mean_render=checks,assets=assets,old1200_reuse=1200,existing_cos_reuse=1400,new_queries=1400,new_prefix_pairs=1400,new_cos_bytes=1400*16384*25*4,pending_external_seals=[str(d/'sealed.json') for d in (mean,uniform) if not (d/'sealed.json').is_file()],GPU_executed=False,query_truth_indexed=False)
(L/'remote_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('assets','three_case_G64_header_mean_render')}),flush=True)
