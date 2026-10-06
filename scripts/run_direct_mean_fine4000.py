#!/usr/bin/env python3
"""Frozen full4000 DirectMEAN fine control; root owns GPU execution.

Exact extension of the original1200 fine.mean16.control. Reuse its1200
sealed outputs, reuse tau15 successor cosines on1400 non1200 fold0/3 draws,
and encode only1400 non1200 fold1/2 draws. Use original per-fold tau07/15,
sigma1.25/window2, raw multiply/sum/div and unchanged CUDA >.5 finalizer.
No new cue, fitted parameter, q/r archive, cleanup or query GT during infer.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json,multiprocessing as mp,os
from pathlib import Path
import shutil,sys,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from ics.experiment import render,sha,summarize,unpack
from run_frozen_subtoken1200 import ARMS,FINE16,FINE64,verify_code,write
from run_frozen_subtoken4000 import make_host,fine_grid
from run_fine_mean1200 import MEAN_FINE,base_metadata
from benchmark_part1_producer import part1_only
from run_fused_tau15_sizecut4000 import geometry,verify_prefix
GSEAL='82f9f208f2254cd25aff5a16f0bcd1e90f6af250c45c6efc7ff40275abc5ea8a'


def sealed(directory,state,n):
    s=json.loads((directory/'sealed.json').read_text());c=json.loads((directory/'config.json').read_text());rows=json.loads((directory/'manifest.json').read_text())
    if s['state']!=state or s['n']!=n or len(rows)!=n or len({r['key'] for r in rows})!=n:raise ValueError('Required complete seal/unique draw keys missing: '+str(directory))
    for name,key in (('manifest.json','manifest_sha256'),('config.json','config_sha256')):
        if sha(directory/name)!=s[key]:raise ValueError('Changed sealed metadata: '+str(directory/name))
    verify_code(c)
    return rows,c,s


def identity(row):return row['c'],row['fold'],str(row['support']),str(row['query'])


def metadata(a,dependencies=True):
    rows,bc,bs=sealed(a.base4000,'ALL_PREDICTIONS_SEALED',4000)
    r12,c12,s12=base_metadata(a.base1200);old={'public%d:%s'%(r['public_batch'],r.get('source_key',r['key'])):r for r in r12}
    if len(old)!=1200 or len(set(old)&{r['key'] for r in rows})!=1200:raise ValueError('Original1200 draw mapping changed')
    grow,gc,gs=sealed(a.graft_run,'ALL_PREDICTIONS_SEALED',4000)
    if sha(a.graft_run/'sealed.json')!=GSEAL:raise ValueError('Original all4000 G provider changed')
    gm={r['key']:r for r in grow}
    if set(gm)!={r['key'] for r in rows} or any(identity(gm[r['key']])!=identity(r) for r in rows):raise ValueError('Original all4000 G provider identity differs')
    if gc['base_seal_sha256']!=sha(a.base4000/'sealed.json') or gc['mean_field_key']!='mean.control':raise ValueError('Require actual original mean.control G64 provider')
    prior=a.base4000_scored/'episodes.jsonl'
    if not prior.is_file():raise FileNotFoundError(prior)
    extra=None
    if dependencies:
        mrows,mc,ms=sealed(a.mean1200,'ALL_PREDICTIONS_AND_KERNELS_SEALED',1200)
        if mrows!=r12 or mc['parameters']!=c12['parameters']:raise ValueError('Original DirectMEAN1200 changed draw/parameter policy')
        urows,uc,us=sealed(a.uniform_run,'ALL_PREDICTIONS_SEALED',4000)
        if urows!=rows or set(us['cosines'])!={r['key'] for r in rows if r['fold'] in (0,3)}:raise ValueError('Uniform successor exact missing2000 cosines unavailable')
        extra=(mc,ms,uc,us)
    return rows,bc,bs,old,c12,s12,gc,gs,prior,extra


def wait_inputs(a):
    if a.out.exists():raise FileExistsError('Fresh readiness receipt required')
    for path in (a.mean1200,a.uniform_run):
        while not (path/'sealed.json').is_file():
            st=path/'state.json'
            if st.is_file() and json.loads(st.read_text()).get('state')=='FAILED':raise RuntimeError('Required existing run failed; never replace it: '+str(path))
            if path==a.uniform_run and a.uniform_pipeline_state.is_file() and json.loads(a.uniform_pipeline_state.read_text()).get('state') in ('STAGE_FAILED','REQUIRES_FAILED','SUPERVISOR_ERROR'):
                raise RuntimeError('Uniform predecessor pipeline failed; no replacement or retry')
            time.sleep(2)
    metadata(a)
    write(a.out,dict(state='DIRECT_MEAN4000_DEPENDENCIES_SEALED_READY',mean1200_seal_sha256=sha(a.mean1200/'sealed.json'),uniform_seal_sha256=sha(a.uniform_run/'sealed.json'),G64_seal_sha256=GSEAL,query_truth_opened=False))
    print('DIRECT_MEAN4000_DEPENDENCIES_READY',flush=True)


def infer(a,smoke=False):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,bc,bs,old,c12,s12,gc,gs,prior,extra=metadata(a);mc,ms,uc,us=extra
    if smoke:rows=[next(r for r in rows if r['key'] in old and r['fold']==0),next(r for r in rows if r['key'] not in old and r['fold']==0),next(r for r in rows if r['key'] not in old and r['fold']==1)]
    ancestor=a.out.parent.resolve()
    while not ancestor.exists():ancestor=ancestor.parent
    if not smoke and shutil.disk_usage(ancestor).free<(4<<30):raise OSError('Need4GiB for newcos1400+allfields/masks/headroom; no assets deleted')
    a.out.mkdir(parents=True,exist_ok=False)
    for folder in ('fields','predictions','cosines'):(a.out/folder).mkdir()
    write(a.out/'manifest.json',rows)
    files=[Path(__file__).resolve()]+[REPO/'scripts'/name for name in ('run_frozen_subtoken1200.py','run_frozen_subtoken4000.py','run_fine_mean1200.py','benchmark_part1_producer.py','run_fused_tau15_sizecut4000.py')]+[REPO/'src/ics/experiment.py',REPO/'src/ics/methods/rcg.py']
    config=dict(n=len(rows),primary=FINE64,new_control=MEAN_FINE,arms=[*ARMS,MEAN_FINE],parameters=c12['parameters'],source_code_sha256={str(p):sha(p) for p in files},
        base4000=str(a.base4000.resolve()),base4000_scored=str(a.base4000_scored.resolve()),baseline_prior_episodes_sha256=sha(prior),
        mean1200=str(a.mean1200.resolve()),mean1200_seal_sha256=sha(a.mean1200/'sealed.json'),uniform_run=str(a.uniform_run.resolve()),uniform_seal_sha256=sha(a.uniform_run/'sealed.json'),graft_run=str(a.graft_run.resolve()),G64_seal_sha256=GSEAL,G64_key='mean.control',
        role='strong simple same-information complete4000 control; exact extension of original1200 arm, no new method',
        contract='original normalized FP16 guide/FP32 four separate shifts; sigma1.25/window2; fold0/3tau.07 and fold1/2tau.15; lambda16 original MEAN_a.25 G; legacy unnormalized w multiply/sum/div; CUDA128->1024 alignFalse >.5',
        expected=dict(original1200_reused=1200,existing_cos_reused=1400,new_encoded=1400,new_prefix=1400,new_shift_forwards=5600),
        cache='retain all existing assets; new FP32cos[16384,25] only newly encoded1400 (~2.294GB); sharedgeometry; bounded RAM q/r/fine lifecycle',
        exposure='all4000 original exposed DEV draws including repeats; no fresh confirmation/SOTA',query_truth_opened=False,
        readers=a.readers,writers=a.writers,prefetch=a.prefetch,workers=a.workers,inflight=a.inflight,chunk=a.chunk,peak_gpu_budget_bytes=10<<30)
    write(a.out/'config.json',config)
    nb,valid,spatial=geometry('cuda')
    for name,value in (('neighbors',nb.cpu().numpy().astype(np.uint16)),('valid_neighbors',valid.cpu().numpy()),('spatial',spatial.cpu().numpy())):
        up=a.uniform_run/(name+'.npy')
        if sha(up)!=us[name+'_sha256'] or not np.array_equal(value,np.load(up,allow_pickle=False)):raise ValueError('Exact retained cosine neighbor/spatial contract changed')
        np.save(a.out/(name+'.npy'),value)
    man=json.loads(a.host_manifest.read_text());data=Path(man['data_root']);host=None
    def load(row):
        key=row['key'];fp=a.base4000/'fields'/(key+'.npz');pp=a.base4000/'predictions'/(key+'.npz');gp=a.graft_run/'fields'/(key+'.npz')
        if sha(fp)!=bs['fields'][key] or sha(pp)!=bs['predictions'][key] or sha(gp)!=gs['fields'][key]:raise ValueError('Sealed original field/mask/G source changed')
        receipt=dict(packet_export=row['packet_export'],packet_sha256=sha(row['packet_export']),base_field_sha256=bs['fields'][key],base_prediction_sha256=bs['predictions'][key],G64_field_sha256=gs['fields'][key])
        if receipt['packet_sha256']!=bs['inputs'][key]['packet_sha256'] or receipt['packet_sha256']!=gs['inputs'][key]['packet_sha256']:raise ValueError('Source packet identity changed')
        with np.load(fp,allow_pickle=False) as z:R=z['rcg'].copy();Y=z[FINE16].copy()
        with np.load(pp,allow_pickle=False) as z:masks={name:z[name].copy() for name in ARMS}
        with np.load(gp,allow_pickle=False) as z:G=z['mean.control'].copy()
        if G.shape!=(64,64) or G.dtype!=np.float32 or not np.isfinite(G).all() or np.count_nonzero(render(G)!=unpack(masks['mean.control'])):raise ValueError('Original G64 header/complete MEAN mask parity failed')
        receipt['G64_original_mean_render_mismatches']=0
        if key in old:
            natural=old[key]['key'];df=a.mean1200/'fields'/(natural+'.npz');dp=a.mean1200/'predictions'/(natural+'.npz');prep=Path(mc['prepared'])/'fields'/(natural+'.npz')
            if identity(old[key])!=identity(row) or sha(df)!=ms['fields'][natural] or sha(dp)!=ms['predictions'][natural]:raise ValueError('Original1200 result identity/hash changed')
            if sha(prep)!=ms['inputs'][natural]['prepared_field_sha256']:raise ValueError('Original1200 G64 source changed')
            with np.load(prep,allow_pickle=False) as z:
                if not np.array_equal(G,z['mean.control']):raise ValueError('G64 provider differs from original1200 exact G')
            with np.load(df,allow_pickle=False) as z:field=z[MEAN_FINE].copy();replay=z[FINE16].copy()
            with np.load(dp,allow_pickle=False) as z:
                if any(not np.array_equal(z[name],masks[name]) for name in ARMS):raise ValueError('Original1200 six mask sequences differ')
                masks[MEAN_FINE]=z[MEAN_FINE].copy()
            if not np.array_equal(replay,Y):raise ValueError('Original1200 R fine field differs from4000 baseline')
            receipt.update(mode='exact_original1200_field_mask_reuse',old_mean_key=natural,old_mean_fields_sha256=ms['fields'][natural],old_mean_prediction_sha256=ms['predictions'][natural])
            return R,Y,G,masks,receipt,field,None,None,None,None
        if row['fold'] in (0,3):
            cp=a.uniform_run/'cosines'/(key+'.npy')
            if sha(cp)!=us['cosines'][key]:raise ValueError('Retained actual FP32 cosine changed')
            cos=np.load(cp,allow_pickle=False)
            if cos.shape!=(16384,25) or cos.dtype!=np.float32 or not np.isfinite(cos).all():raise ValueError('Retained cosine header changed')
            receipt.update(mode='exact_original_tau07_readout_from_saved_FP32_cos',uniform_cosine_sha256=us['cosines'][key])
            return R,Y,G,masks,receipt,None,cos,None,None,None
        with np.load(row['packet_export'],allow_pickle=False) as z:cov,score=z['cov'].copy(),z['score'].copy()
        image_path=data/row['query']
        with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB')).copy()
        receipt.update(mode='actual_Part1_FP32_half_unit_plus_four_shift_tau15',query_image_sha256=sha(image_path))
        return R,Y,G,masks,receipt,None,None,rgb,cov,score
    def save(row,field,masks,receipt,newcos):
        key=row['key'];fp=a.out/'fields'/(key+'.npz');pp=a.out/'predictions'/(key+'.npz')
        if field.shape!=(128,128) or field.dtype!=np.float32 or not np.isfinite(field).all():raise ValueError('Finite F32 DirectMEAN fine field required')
        np.savez_compressed(fp,**{MEAN_FINE:field});np.savez_compressed(pp,**masks)
        record=dict(key=key,field_sha256=sha(fp),prediction_sha256=sha(pp),inputs=receipt)
        if newcos is not None:
            cp=a.out/'cosines'/(key+'.npy');np.save(cp,newcos);record['cosine_sha256']=sha(cp)
        return record
    def finish(item):
        row,R,Y,G,masks,receipt,cosarray,q16,finearray,future,encoder_seconds=item
        if future is not None:receipt['actual_prefix_lambda16']=future.result()
        torch.cuda.synchronize();tick=time.monotonic();newcos=None
        if cosarray is None:
            q=F.normalize(torch.from_numpy(q16).to('cuda').float(),dim=1);fine=torch.from_numpy(finearray).to('cuda')
            parts=[torch.einsum('pc,pkc->pk',fine[start:min(start+a.chunk,16384)],q[nb[start:min(start+a.chunk,16384)]]) for start in range(0,16384,a.chunk)]
            cosine=torch.cat(parts);newcos=cosine.cpu().numpy()
        else:cosine=torch.from_numpy(cosarray).to('cuda')
        r=torch.from_numpy(R).to('cuda').flatten();g=torch.from_numpy(G).to('cuda').flatten();rp=[];gp=[];tau=c12['parameters'][str(row['fold'])]['tau']
        for start in range(0,16384,a.chunk):
            stop=min(start+a.chunk,16384);ids=nb[start:stop];w=spatial[start:stop]*torch.exp((cosine[start:stop]-1)/tau)*valid[start:stop];den=w.sum(1).clamp_min(1e-12)
            rp.append((w*r[ids]).sum(1)/den);gp.append((w*g[ids]).sum(1)/den)
        replay=torch.cat(rp).reshape(128,128);field=torch.cat(gp).reshape(128,128)
        nf=int(np.count_nonzero(replay.cpu().numpy()!=Y));rm=(F.interpolate(replay[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy();nm=int(np.count_nonzero(rm!=unpack(masks[FINE16])))
        receipt.update(original_fine16_field_mismatches=nf,original_fine16_mask_mismatches=nm,encoder_seconds=encoder_seconds)
        if nf or nm:raise RuntimeError('Original foldtau R fine field/mask exact producer replay failed')
        masks[MEAN_FINE]=np.packbits((F.interpolate(field[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy())
        torch.cuda.synchronize();receipt['readout_seconds']=time.monotonic()-tick
        return field.cpu().numpy(),masks,receipt,newcos
    records=[];pending=[];writes=[];reused=cos_reused=encoded=0;started=time.monotonic();encoder_total=0.
    with torch.inference_mode(),ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool,ThreadPoolExecutor(a.readers) as reader,ThreadPoolExecutor(a.writers) as writer:
        loaded={i:reader.submit(load,rows[i]) for i in range(min(a.prefetch,len(rows)))}
        for n,row in enumerate(rows):
            R,Y,G,masks,receipt,field,cos,rgb,cov,score=loaded.pop(n).result();later=n+a.prefetch
            if later<len(rows):loaded[later]=reader.submit(load,rows[later])
            if field is not None:writes.append(writer.submit(save,row,field,masks,receipt,None));reused+=1
            elif cos is not None:
                item=(row,R,Y,G,masks,receipt,cos,None,None,None,0.);writes.append(writer.submit(save,row,*finish(item)));cos_reused+=1
            else:
                if host is None:
                    host,producer,_=make_host(a)
                    if sha(producer.__file__)!=c12['host_builder_sha256'] or sha(a.host_manifest)!=c12['host_manifest_sha256']:raise ValueError('Original model/host numerical producer changed')
                torch.cuda.synchronize();tick=time.monotonic()
                with Image.open(data/row['support']) as im:support=im.convert('RGB')
                with Image.open(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')) as im:refmask=torch.from_numpy((np.asarray(im)==row['c']+1).copy())
                prefix=part1_only(host,support,refmask,Image.fromarray(rgb))
                if np.count_nonzero(prefix['cov']!=cov):raise RuntimeError('Reference coverage changed')
                future=pool.submit(verify_prefix,(prefix['q'],prefix['r'],cov,score,R,masks['rcg']))
                fine=fine_grid(host,prefix['target'],prefix['debiased']);torch.cuda.synchronize();seconds=time.monotonic()-tick;encoder_total+=seconds
                pending.append((row,R,Y,G,masks,receipt,None,prefix['q'],fine,future,seconds));encoded+=1;del prefix,rgb
                if len(pending)>=a.inflight:
                    item=pending.pop(0);writes.append(writer.submit(save,item[0],*finish(item)))
            while pending and pending[0][-2].done():
                item=pending.pop(0);writes.append(writer.submit(save,item[0],*finish(item)))
            while len(writes)>=2*a.writers:records.append(writes.pop(0).result())
            peak=torch.cuda.max_memory_allocated()
            if peak>=(10<<30):raise RuntimeError('Measured own CUDA allocation exceeded10GiB shared budget')
            if n==0 or (n+1)%10==0:
                state=dict(state='DIRECT_MEAN4000_INFERENCE',visited=n+1,n=len(rows),old1200_reused=reused,cos_reused=cos_reused,encoded=encoded,seconds=time.monotonic()-started,encoder_seconds=encoder_total,cuda_peak_bytes=peak,query_truth_opened=False)
                write(a.out/'state.json',state);print(json.dumps(state),flush=True)
        for item in pending:writes.append(writer.submit(save,item[0],*finish(item)))
        records.extend(job.result() for job in writes)
    if not smoke and (reused,cos_reused,encoded)!=(1200,1400,1400):raise RuntimeError('Requested exact4000 reuse/accounting changed')
    peak=torch.cuda.max_memory_allocated()
    if peak>=(10<<30):raise RuntimeError('Final own CUDA allocation exceeded10GiB shared budget')
    seal=dict(state='SMOKE_PREDICTIONS_SEALED' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),primary=FINE64,new_control=MEAN_FINE,manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),fields={r['key']:r['field_sha256'] for r in records},predictions={r['key']:r['prediction_sha256'] for r in records},inputs={r['key']:r['inputs'] for r in records},cosines={r['key']:r['cosine_sha256'] for r in records if 'cosine_sha256' in r},neighbors_sha256=sha(a.out/'neighbors.npy'),valid_neighbors_sha256=sha(a.out/'valid_neighbors.npy'),spatial_sha256=sha(a.out/'spatial.npy'),old1200_reused=reused,existing_cos_reused=cos_reused,new_encoded=encoded,seconds=time.monotonic()-started,encoder_seconds=encoder_total,cuda_peak_bytes=peak,query_truth_opened=False)
    write(a.out/'sealed.json',seal);write(a.out/'state.json',dict(state='SMOKE_COMPLETE' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),cuda_peak_bytes=peak,query_truth_opened=False));print('DIRECT_MEAN4000_SEALED',flush=True)


def score_one(job):
    row,out,expected=job
    with np.load(row['packet_export'],allow_pickle=False) as z:truth=unpack(z['truth'])
    with np.load(Path(out)/'predictions'/(row['key']+'.npz'),allow_pickle=False) as z:masks={arm:unpack(z[arm]) for arm in (*ARMS,MEAN_FINE)}
    iu={arm:[int((mask&truth).sum()),int((mask|truth).sum())] for arm,mask in masks.items()}
    if any(iu[arm]!=expected[arm] for arm in ARMS):raise ValueError('All4000 original six baseline I/U parity failed')
    return dict(row,iu=iu)


def score(a):
    rows,c,s=sealed(a.out,'ALL_PREDICTIONS_SEALED',4000);prior_path=Path(c['base4000_scored'])/'episodes.jsonl'
    if sha(prior_path)!=c['baseline_prior_episodes_sha256']:raise ValueError('Original scored4000 prior changed')
    prior={r['key']:r for r in [json.loads(x) for x in prior_path.read_text().splitlines()]}
    if set(prior)!={r['key'] for r in rows}:raise ValueError('All4000 scored prior identities required')
    for row in rows:
        key=row['key']
        if sha(a.out/'predictions'/(key+'.npz'))!=s['predictions'][key] or sha(row['packet_export'])!=s['inputs'][key]['packet_sha256']:raise ValueError('Sealed scoring input changed')
    started=time.monotonic()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:episodes=list(pool.map(score_one,[(r,str(a.out),prior[r['key']]['iu']) for r in rows],chunksize=4))
    arrays={arm:np.asarray([r['iu'][arm] for r in episodes],np.int64) for arm in (*ARMS,MEAN_FINE)};report,draws=summarize(rows,arrays,{})
    for k in ('corrections_vs_native','corrections_by_class','corrections_by_batch'):report.pop(k,None)
    report.update(primary=FINE64,new_strong_simple_control=MEAN_FINE,original_six_arm_IU_parity=4000,parameter_updates=False,exposure=c['exposure'],raw_origin_edits='not exported; missing, not substituted',INSID3='original DEV241 only; full4000 absent',runtime=dict(inference_seconds=s['seconds'],encoder_seconds=s['encoder_seconds'],cpu_score_seconds=time.monotonic()-started,cold_complete_producer_seconds=None,cold_scope='old outputs/G/cos reused; complete cold producer time not measured'),reuse=dict(original1200=s['old1200_reused'],existing_cos=s['existing_cos_reused'],new_queries=s['new_encoded']))
    write(a.out/'report.json',report);np.save(a.out/'bootstrap_photo_draws.npy',draws);np.savez_compressed(a.out/'counts.npz',**{'iu:'+arm:value for arm,value in arrays.items()});(a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes));write(a.out/'score_state.json',dict(state='CPU_SCORE_COMPLETE',n=4000,new_control=MEAN_FINE));print(json.dumps(report['scores']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('wait-inputs','smoke','infer','score'));p.add_argument('--out',type=Path,required=True)
    for name,default in (('base4000','outputs/frozen_subtoken4000_v1'),('base4000-scored','outputs/frozen_subtoken4000_scored_v1'),('base1200','outputs/frozen_subtoken1200_v1'),('mean1200','outputs/fine_mean1200_v1'),('uniform-run','outputs/uniform_tau15_sizecut4000_v2'),('graft-run','outputs/mean_fine_residual_transfer4000_v1')):p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--uniform-pipeline-state',type=Path,default=Path('launch/uniform_tau15_sizecut4000_v2/state.json'))
    p.add_argument('--host-manifest',type=Path,default=Path('outputs/claude_official/batch0.json'));p.add_argument('--host-root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4');p.add_argument('--basis',type=Path,default=Path('/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt'))
    p.add_argument('--workers',type=int,default=6);p.add_argument('--readers',type=int,default=6);p.add_argument('--writers',type=int,default=2);p.add_argument('--prefetch',type=int,default=12);p.add_argument('--inflight',type=int,default=6);p.add_argument('--chunk',type=int,default=512)
    a=p.parse_args()
    if any(v<1 for v in (a.workers,a.readers,a.writers,a.prefetch,a.inflight,a.chunk)):p.error('Positive execution sizes required')
    try:
        if a.stage=='wait-inputs':wait_inputs(a)
        elif a.stage=='score':score(a)
        else:infer(a,a.stage=='smoke')
    except Exception as error:
        if a.out.is_dir():write(a.out/'state.json',dict(state='FAILED',stage=a.stage,error=repr(error),query_truth_opened=False))
        raise


if __name__=='__main__':main()
