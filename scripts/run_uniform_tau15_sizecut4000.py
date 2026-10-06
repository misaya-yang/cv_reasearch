#!/usr/bin/env python3
"""Successor v2: fixed supplied tau15+sizecut4000, preserving active Mean1200.

Reuse region's actual sealed2000 tau15 masks/fields; only missing fold0/3
2000 queries get four source-identical shifts.600 use protected FP16 cache;
1400 use actual Part1-only paired FP32 prefix plus exact lambda16 replay.
Retain missing2000 FP32 cosine grids (~3.28GB). Never modify/duplicate the
active DirectMEAN run or peer. Root launches the explicitly coordinated plan.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from ics.experiment import sha,unpack,summarize
from run_frozen_subtoken1200 import ARMS,FINE16,verify_code,write
from run_frozen_subtoken4000 import make_host,fine_grid
from benchmark_part1_producer import part1_only
from run_fine_mean1200 import base_metadata
from run_fused_tau15_sizecut4000 import frozen_contract,geometry,fixed_masks,verify_prefix,score_one,TAU15,METHOD,FROZEN_SHA

MATCH_MANIFEST_SHA='953dbd9408da8d3c451fb267025a80da1377166a8b88939285d79c831f259981'
MATCH_SEAL_SHA='a5f1deb8e92086419f7b1f33507aa2cf8f2a99841027d97ca4167223d8c98f2c'
MATCH_PRIMARY='provided_uniform_t15_size_cut.primary'


def mean_receipt(directory):
    seal_path=directory/'sealed.json'
    seal=json.loads(seal_path.read_text())
    if seal['state']!='ALL_PREDICTIONS_AND_KERNELS_SEALED' or seal['n']!=1200:raise ValueError('Original requested MEAN must finish first; never duplicate it')
    for name,key in (('config.json','config_sha256'),('manifest.json','manifest_sha256')):
        if sha(directory/name)!=seal[key]:raise ValueError('Original requested MEAN metadata changed')
    return dict(run=str(directory.resolve()),seal_sha256=sha(seal_path),config_sha256=seal['config_sha256'],manifest_sha256=seal['manifest_sha256'],n=1200)


def wait_mean(a):
    # Only poll the owned dependency. Never inspect GT, load models or signal jobs.
    while not (a.existing_mean_run/'sealed.json').is_file():
        state_path=a.existing_mean_run/'state.json'
        if state_path.is_file() and json.loads(state_path.read_text()).get('state')=='FAILED':raise RuntimeError('Original MEAN dependency failed; no replacement execution')
        time.sleep(2)
    receipt=mean_receipt(a.existing_mean_run)
    write(a.out,dict(state='ORIGINAL_MEAN1200_SEALED_READY',**receipt))
    print(json.dumps(receipt),flush=True)


def metadata(a):
    bs=json.loads((a.base4000/'sealed.json').read_text());bc=json.loads((a.base4000/'config.json').read_text());rows=json.loads((a.base4000/'manifest.json').read_text())
    if bs['state']!='ALL_PREDICTIONS_SEALED' or bs['n']!=4000 or len(rows)!=4000 or len({r['key'] for r in rows})!=4000:raise ValueError('Require all4000 preserved draws')
    for file,key in (('manifest.json','manifest_sha256'),('config.json','config_sha256')):
        if sha(a.base4000/file)!=bs[key]:raise ValueError('4000 metadata changed')
    verify_code(bc)
    r12,c12,s12=base_metadata(a.base1200);cached={'public%d:%s'%(r['public_batch'],r.get('source_key',r['key'])):r for r in r12}
    if len(cached)!=1200 or len(set(cached)&{r['key'] for r in rows})!=1200:raise ValueError('Protected original1200 mapping changed')
    mp=a.matching_run/'primary_matching2000_manifest.json'
    if sha(mp)!=MATCH_MANIFEST_SHA or sha(a.matching_run/'sealed.json')!=MATCH_SEAL_SHA:raise ValueError('Actual region matching2000 changed')
    matching={r['key']:r for r in json.loads(mp.read_text())}
    if len(matching)!=2000 or set(matching)!={r['key'] for r in rows if r['fold'] in (1,2)}:raise ValueError('Matching tau15 draw set changed')
    prior=a.base4000_scored/'episodes.jsonl'
    if not prior.is_file():raise FileNotFoundError(prior)
    return rows,bs,bc,cached,c12,s12,matching,prior


def infer(a,smoke=False):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frozen=frozen_contract(a.frozen);rows,bs,bc,cached,c12,s12,matching,prior=metadata(a)
    mean_dependency=mean_receipt(a.existing_mean_run)
    if smoke:rows=[next(r for r in rows if r['fold']==0 and r['key'] in cached),next(r for r in rows if r['fold']==1),next(r for r in rows if r['fold']==0 and r['key'] not in cached)]
    ancestor=a.out.parent.resolve()
    while not ancestor.exists():ancestor=ancestor.parent
    if not smoke and shutil.disk_usage(ancestor).free < 5<<30:raise OSError('Require5GiB for cos2000+fields/masks/headroom')
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('fields','predictions','cosines'):(a.out/name).mkdir()
    write(a.out/'manifest.json',rows)
    host,producer,man=make_host(a)
    if sha(producer.__file__)!=c12['host_builder_sha256'] or sha(a.host_manifest)!=c12['host_manifest_sha256']:raise ValueError('Original source producer changed')
    files=[Path(__file__).resolve(),REPO/'scripts/run_fused_tau15_sizecut4000.py',REPO/'scripts/run_frozen_subtoken4000.py',REPO/'scripts/run_frozen_subtoken1200.py',REPO/'scripts/run_fine_mean1200.py',REPO/'scripts/benchmark_part1_producer.py',REPO/'src/ics/experiment.py',REPO/'src/ics/methods/rcg.py']
    config=dict(n=len(rows),primary=METHOD,frozen=frozen,frozen_sha256=FROZEN_SHA,source_code_sha256={str(p):sha(p) for p in files},
        base4000=str(a.base4000.resolve()),base4000_scored=str(a.base4000_scored.resolve()),baseline_prior_episodes_sha256=sha(prior),
        matching_run=str(a.matching_run.resolve()),matching_manifest_sha256=MATCH_MANIFEST_SHA,matching_seal_sha256=MATCH_SEAL_SHA,
        existing_mean_run=str(a.existing_mean_run.resolve()),existing_mean_receipt=mean_dependency,direct_mean_execution='none; original active immutable run preserved',
        numerical_contract='fresh fitting unit(FP16 q/r.float), FP32 four separate shift forwards, lambda16/alpha.5,sigma1.25/window2,uniformtau.15; unnormalized peer stream not substituted',
        expected=dict(reused_tau15=2000,four_shift_queries=2000,paired_prefix_queries=1400,cached_queries=600,shift_forwards=8000),
        cache='FP32 cos[16384,25] only missing2000; shared neighbors/valid/spatial; no q/r/fine archives or deletions',
        exposure='all4000 original draws including repeated identities; exposed benchmark reuse, not fresh confirmation/SOTA',
        peak_gpu_budget_bytes=10<<30,query_truth_opened=False,workers=a.workers,readers=a.readers,writers=a.writers,prefetch=a.prefetch,inflight=a.inflight,chunk=a.chunk)
    write(a.out/'config.json',config)
    nb,valid,spatial=geometry('cuda')
    np.save(a.out/'neighbors.npy',nb.cpu().numpy().astype(np.uint16));np.save(a.out/'valid_neighbors.npy',valid.cpu().numpy());np.save(a.out/'spatial.npy',spatial.cpu().numpy())
    data=Path(man['data_root'])
    def load(row):
        key=row['key'];fp=a.base4000/'fields'/(key+'.npz');pp=a.base4000/'predictions'/(key+'.npz')
        if sha(fp)!=bs['fields'][key] or sha(pp)!=bs['predictions'][key]:raise ValueError('4000 source changed')
        receipt=dict(packet_export=row['packet_export'],packet_sha256=sha(row['packet_export']),base_fields_sha256=bs['fields'][key],base_prediction_sha256=bs['predictions'][key])
        if receipt['packet_sha256']!=bs['inputs'][key]['packet_sha256']:raise ValueError('Source packet changed')
        with np.load(pp,allow_pickle=False) as z:masks={name:z[name].copy() for name in ARMS}
        with np.load(fp,allow_pickle=False) as z:coarse=z['rcg'].copy();old_fine=z[FINE16].copy()
        if key in matching:
            m=matching[key]
            if m['source_tau']!=.15 or m['c']!=row['c'] or any(Path(m[k]).name!=Path(row[k]).name for k in ('support','query')):raise ValueError('Matching identity/tau differs')
            if sha(m['prediction'])!=m['prediction_sha256'] or sha(m['field_source'])!=m['field_sha256']:raise ValueError('Actual matching2000 asset changed')
            with np.load(m['prediction'],allow_pickle=False) as z:combined=z[MATCH_PRIMARY].copy()
            receipt.update(mode='actual_sealed_tau15_and_sizecut_reuse',matching_prediction_sha256=m['prediction_sha256'])
            return None,None,coarse,old_fine,masks,None,None,receipt,combined
        with np.load(row['packet_export'],allow_pickle=False) as z:cov,score=z['cov'].copy(),z['score'].copy()
        with Image.open(data/row['query']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        tensor=None
        if key in cached:
            old=cached[key]
            if old['c']!=row['c'] or any(Path(old[k]).name!=Path(row[k]).name for k in ('support','query')):raise ValueError('Protected cache identity differs')
            if sha(old['feature_export'])!=s12['inputs'][old['key']]['feature_sha256']:raise ValueError('Protected FP16 q/r changed')
            tensor=torch.load(old['feature_export'],map_location='cpu',weights_only=True)
            if any(tensor[k].shape!=(4096,1024) or tensor[k].dtype!=torch.float16 or not bool(torch.isfinite(tensor[k]).all()) for k in ('q','r')):raise ValueError('Protected original finite FP16 q/r headers changed')
            receipt['feature_export']=old['feature_export'];receipt['feature_sha256']=s12['inputs'][old['key']]['feature_sha256']
        receipt['query_image_sha256']=sha(data/row['query'])
        return tensor,rgb,coarse,old_fine,masks,cov,score,receipt,None
    def save(row,field,masks,receipt,cos):
        key=row['key'];fp=a.out/'fields'/(key+'.npz');pp=a.out/'predictions'/(key+'.npz')
        np.savez_compressed(fp,**{TAU15:field});np.savez_compressed(pp,**masks)
        record=dict(key=key,fields_sha256=sha(fp),predictions_sha256=sha(pp),inputs=receipt)
        if cos is not None:
            cp=a.out/'cosines'/(key+'.npy');np.save(cp,cos);record['cosine_sha256']=sha(cp)
        return record
    records=[];pending=[];writes=[];encoded=prefixes=reused=0;started=time.monotonic()
    def finish(item):
        row,q16,fine_array,coarse,expected,masks,receipt,future,producer_seconds=item
        if future is not None:receipt['actual_prefix_lambda16']=future.result()
        tick=time.monotonic();q=F.normalize(torch.from_numpy(q16).to('cuda').float(),dim=1);fine=torch.from_numpy(fine_array).to('cuda');parts=[]
        for start in range(0,16384,a.chunk):
            stop=min(start+a.chunk,16384);parts.append(torch.einsum('pc,pkc->pk',fine[start:stop],q[nb[start:stop]]))
        cos=torch.cat(parts);z=torch.from_numpy(coarse).to('cuda').flatten();new_parts=[];old_parts=[];tau_old=c12['parameters'][str(row['fold'])]['tau']
        for start in range(0,16384,a.chunk):
            stop=min(start+a.chunk,16384);ids=nb[start:stop]
            w=spatial[start:stop]*torch.exp((cos[start:stop]-1)/.15)*valid[start:stop];new_parts.append((w*z[ids]).sum(1)/w.sum(1).clamp_min(1e-12))
            old_w=spatial[start:stop]*torch.exp((cos[start:stop]-1)/tau_old)*valid[start:stop];old_parts.append((old_w*z[ids]).sum(1)/old_w.sum(1).clamp_min(1e-12))
        field=torch.cat(new_parts).reshape(128,128);replay=torch.cat(old_parts).reshape(128,128)
        nf=int(np.count_nonzero(replay.cpu().numpy()!=expected));rmask=(F.interpolate(replay[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy();nm=int(np.count_nonzero(rmask!=unpack(masks[FINE16])))
        receipt.update(original_fine16_field_mismatches=nf,original_fine16_mask_mismatches=nm)
        if nf or nm:raise RuntimeError('Actual normalized producer did not exactly replay old fine16')
        masks[TAU15],masks[METHOD],cut=fixed_masks(field,frozen);receipt.update(fixed_sizecut=cut,four_shift_seconds=producer_seconds,readout_seconds=time.monotonic()-tick)
        return field.cpu().numpy(),masks,receipt,cos.cpu().numpy()
    with torch.inference_mode(),ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool,ThreadPoolExecutor(a.readers) as readers,ThreadPoolExecutor(a.writers) as writer:
        loaded={i:readers.submit(load,rows[i]) for i in range(min(a.prefetch,len(rows)))}
        for n,row in enumerate(rows):
            tensor,rgb,coarse,old_fine,masks,cov,score,receipt,combined=loaded.pop(n).result();later=n+a.prefetch
            if later<len(rows):loaded[later]=readers.submit(load,rows[later])
            if row['key'] in matching:
                masks[TAU15]=masks[FINE16].copy();masks[METHOD]=combined;writes.append(writer.submit(save,row,old_fine,masks,receipt,None));reused+=1
            else:
                torch.cuda.synchronize();tick=time.monotonic();future=None
                if tensor is not None:
                    q16=tensor['q'].numpy();flag=bool(tensor['debiased']);target=host._transform(Image.fromarray(rgb)).to('cuda');receipt['mode']='protected_FP16_unit_cache'
                else:
                    with Image.open(data/row['support']) as im:support=im.convert('RGB')
                    with Image.open(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')) as im:gold=torch.from_numpy((np.asarray(im)==row['c']+1).copy())
                    prefix=part1_only(host,support,gold,Image.fromarray(rgb));q16=prefix['q'];flag=prefix['debiased'];target=prefix['target']
                    if np.count_nonzero(prefix['cov']!=cov):raise RuntimeError('Source reference coverage drift')
                    future=pool.submit(verify_prefix,(q16,prefix['r'],cov,score,coarse,masks['rcg']));prefixes+=1;receipt['mode']='actual_Part1_FP32_half_then_unit'
                fine=fine_grid(host,target,flag);torch.cuda.synchronize();seconds=time.monotonic()-tick;encoded+=1
                pending.append((row,q16,fine,coarse,old_fine,masks,receipt,future,seconds));del target,rgb
                if len(pending)>=a.inflight:
                    item=pending.pop(0);writes.append(writer.submit(save,item[0],*finish(item)))
            while pending and (pending[0][-2] is None or pending[0][-2].done()):
                item=pending.pop(0);writes.append(writer.submit(save,item[0],*finish(item)))
            while len(writes)>=2*a.writers:records.append(writes.pop(0).result())
            if n==0 or (n+1)%10==0:
                state=dict(state='FIXED_UNIFORM_TAU15_INFERENCE',visited=n+1,n=len(rows),encoded=encoded,prefixes=prefixes,reused=reused,seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),query_truth_opened=False)
                write(a.out/'state.json',state);print(json.dumps(state),flush=True)
                if state['gpu_peak_bytes']>=10<<30:raise RuntimeError('Measured allocation exceeded10GiB shared budget')
        for item in pending:writes.append(writer.submit(save,item[0],*finish(item)))
        records.extend(job.result() for job in writes)
    if not smoke and (encoded,prefixes,reused)!=(2000,1400,2000):raise RuntimeError('Requested successor reuse/accounting changed')
    seal=dict(state='SMOKE_PREDICTIONS_SEALED' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),primary=METHOD,manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),predictions={r['key']:r['predictions_sha256'] for r in records},fields={r['key']:r['fields_sha256'] for r in records},inputs={r['key']:r['inputs'] for r in records},cosines={r['key']:r['cosine_sha256'] for r in records if 'cosine_sha256' in r},neighbors_sha256=sha(a.out/'neighbors.npy'),valid_neighbors_sha256=sha(a.out/'valid_neighbors.npy'),spatial_sha256=sha(a.out/'spatial.npy'),encoded_queries=encoded,prefix_pairs=prefixes,reused_tau15=reused,seconds=time.monotonic()-started,query_truth_opened=False)
    write(a.out/'sealed.json',seal);write(a.out/'state.json',dict(state='SMOKE_COMPLETE' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),query_truth_opened=False));print('UNIFORM_TAU15_SIZECUT_SEALED',flush=True)


def score(a):
    rows=json.loads((a.out/'manifest.json').read_text());seal=json.loads((a.out/'sealed.json').read_text());config=json.loads((a.out/'config.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or seal['n']!=4000:raise ValueError('All4000 masks must seal beforeGT')
    if sha(a.out/'manifest.json')!=seal['manifest_sha256'] or sha(a.out/'config.json')!=seal['config_sha256']:raise ValueError('Sealed scoring metadata changed')
    verify_code(config);prior_path=Path(config['base4000_scored'])/'episodes.jsonl'
    if sha(prior_path)!=config['baseline_prior_episodes_sha256']:raise ValueError('Complete six-arm prior changed')
    prior={r['key']:r for r in [json.loads(s) for s in prior_path.read_text().splitlines()]}
    for row in rows:
        key=row['key']
        if sha(a.out/'predictions'/(key+'.npz'))!=seal['predictions'][key] or sha(row['packet_export'])!=seal['inputs'][key]['packet_sha256']:raise ValueError('Changed scoring inputs')
    started=time.monotonic()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:episodes=list(pool.map(score_one,[(r,str(a.out),prior[r['key']]['iu']) for r in rows],chunksize=4))
    arrays={name:np.asarray([r['iu'][name] for r in episodes],np.int64) for name in (*ARMS,TAU15,METHOD)};report,draws=summarize(rows,arrays,{})
    for key in ('corrections_vs_native','corrections_by_class','corrections_by_batch'):report.pop(key,None)
    report.update(primary=METHOD,frozen_sha256=FROZEN_SHA,original_six_arm_IU_parity=4000,parameter_updates=False,exposure=config['exposure'],raw_origin_edits='not exported; missing',INSID3='original DEV241 only; complete4000 absent',existing_mean_run=config['existing_mean_run'],existing_mean_execution='untouched independent original run',runtime=dict(inference_seconds=seal['seconds'],cpu_score_seconds=time.monotonic()-started))
    write(a.out/'report.json',report);np.save(a.out/'bootstrap_photo_draws.npy',draws);(a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes));write(a.out/'score_state.json',dict(state='CPU_SCORE_COMPLETE',n=4000,primary=METHOD));print(json.dumps(report['scores']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('wait-mean','smoke','infer','score'));p.add_argument('--out',type=Path,required=True)
    for name,default in (('base4000','outputs/frozen_subtoken4000_v1'),('base4000-scored','outputs/frozen_subtoken4000_scored_v1'),('base1200','outputs/frozen_subtoken1200_v1'),('existing-mean-run','outputs/fine_mean1200_v1'),('matching-run','outputs/size_cut_composition_cached4000_v2'),('frozen','outputs/claude_rcg2_frozen.json')):p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--host-manifest',type=Path,default=Path('outputs/claude_official/batch0.json'));p.add_argument('--host-root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4');p.add_argument('--basis',type=Path,default=Path('/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt'))
    p.add_argument('--workers',type=int,default=6);p.add_argument('--readers',type=int,default=6);p.add_argument('--writers',type=int,default=2);p.add_argument('--prefetch',type=int,default=12);p.add_argument('--inflight',type=int,default=6);p.add_argument('--chunk',type=int,default=512)
    a=p.parse_args()
    if any(x<1 for x in (a.workers,a.readers,a.writers,a.prefetch,a.inflight,a.chunk)):p.error('Positive execution sizes required')
    try:
        if a.stage=='wait-mean':wait_mean(a)
        elif a.stage=='score':score(a)
        else:infer(a,a.stage=='smoke')
    except Exception as error:
        if a.out.exists():write(a.out/'state.json',dict(state='FAILED',stage=a.stage,error=repr(error),query_truth_opened=False))
        raise


if __name__=='__main__':main()
