#!/usr/bin/env python3
"""Fixed fresh600 tau15/sizecut4000 + unchanged DirectMEAN1200, one encoder.

Reuse matching fold1/2 tau15 fields on2000 draws. Encode missing fold0/3
2000 plus DirectMEAN fold1/2 600 once:2600 four-shift queries,1400 Part1
prefix pairs. Cache FP32 cosines only on missing2000; retain DirectMEAN raw
weights/rowden1200. Neither requested arm is refit or numerically replaced.
Root owns GPU launch, explicit peer coordination and old-waiter cancellation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from ics.experiment import render,sha,summarize,unpack
from run_frozen_subtoken1200 import ARMS,FINE16,FINE64,write,verify_code
from run_frozen_subtoken4000 import make_host,fine_grid
from benchmark_part1_producer import part1_only
from run_fine_mean1200 import MEAN_FINE,base_metadata,score as score_mean1200

FROZEN_SHA='38956334d5e11a0199ea25f659aae0f550aa677f5a7459fa3ed8cf990867c499'
TAU15='provided.tau15.control'
METHOD='provided.rcg_tau15_sizecut'


def frozen_contract(path):
    if sha(path)!=FROZEN_SHA:raise ValueError('Fresh600 frozen JSON changed')
    doc=json.loads(Path(path).read_text())
    if (doc['readout']['sigma'],doc['readout']['tau'],doc['readout']['window'])!=(1.25,.15,2):raise ValueError('Changed supplied readout')
    if doc['size_cut']['mask_area_at_half_edges']!=[0,.01,.03,.08,.2,.4,1.01] or doc['size_cut']['levels']!=[.5,.5625,.5875,.5625,.45,.375]:raise ValueError('Changed six fixed cuts')
    return doc


def load_metadata(a):
    seal=json.loads((a.base4000/'sealed.json').read_text());config=json.loads((a.base4000/'config.json').read_text());rows=json.loads((a.base4000/'manifest.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or seal['n']!=4000 or len(rows)!=4000 or len({r['key'] for r in rows})!=4000:raise ValueError('Require every4000 preserved draw')
    for f,k in (('manifest.json','manifest_sha256'),('config.json','config_sha256')):
        if sha(a.base4000/f)!=seal[k]:raise ValueError('Changed sealed4000 metadata')
    verify_code(config)
    r12,c12,s12=base_metadata(a.base1200)
    mapping={'public%d:%s'%(r['public_batch'],r.get('source_key',r['key'])):r for r in r12}
    if len(mapping)!=1200 or len(set(mapping)&{r['key'] for r in rows})!=1200:raise ValueError('Original1200 draw mapping changed')
    gs=json.loads((a.mean_prepared/'sealed.json').read_text());gc=json.loads((a.mean_prepared/'config.json').read_text())
    if gs['state']!='ALL1200_ORIGINAL_MEAN_FIELDS_PARITY_SEALED' or gs['n']!=1200:raise ValueError('Original1200 G not sealed')
    if sha(a.mean_prepared/'config.json')!=gs['config_sha256']:raise ValueError('G config changed')
    verify_code(gc)
    return rows,seal,config,r12,mapping,c12,s12,gs


def verify_prefix(job):
    import torch
    from ics.methods import rcg
    torch.set_num_threads(1)
    q,r,cov,score,expected,packed=job
    field,info=rcg.predict(q,r,cov,score,device='cpu')
    nf=int(np.count_nonzero(field!=expected));nm=int(np.count_nonzero(render(field)!=unpack(packed)))
    if nf or nm:raise RuntimeError('Actual Part1/unit-FP16 lambda16 source parity failed')
    return dict(field_mismatches=nf,mask_mismatches=nm,solver=info)


def geometry(device):
    import torch
    fi=torch.arange(128,device=device);base=((fi*8+4-8).float()/16).round().long();off=torch.arange(-2,3,device=device)
    ti=base[:,None]+off[None];distance=((ti*16+8)-(fi*8+4)[:,None]).float()/16;ok=(ti>=0)&(ti<64);ti=ti.clamp(0,63)
    ny=ti[:,None,:,None].expand(128,128,5,5);nx=ti[None,:,None,:].expand(128,128,5,5);nb=(ny*64+nx).reshape(16384,25)
    d2=(distance[:,None,:,None].square()+distance[None,:,None,:].square()).reshape(16384,25)
    valid=(ok[:,None,:,None]&ok[None,:,None,:]).reshape(16384,25)
    spatial=torch.exp(-d2/(2*1.25**2))
    return nb,valid,spatial


def fixed_masks(field,frozen):
    import torch
    import torch.nn.functional as F
    up=F.interpolate(field[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]
    fine=up>.5;area=float(fine.float().mean())
    index=int(np.searchsorted(np.asarray(frozen['size_cut']['mask_area_at_half_edges']),area,side='right')-1)
    level=frozen['size_cut']['levels'][index]
    return np.packbits(fine.cpu().numpy()),np.packbits((up>level).cpu().numpy()),dict(area_at_half=area,bin=index,level=level)


def infer(a,smoke=False):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frozen=frozen_contract(a.frozen)
    rows,bs,bc,r12,mapping,c12,s12,gs=load_metadata(a)
    if smoke:
        selected=[next(r for r in rows if r['key'] in mapping and r['fold']==0),next(r for r in rows if r['key'] in mapping and r['fold']==1),next(r for r in rows if r['key'] not in mapping and r['fold']==0)]
        rows=selected
    prior_path=a.base4000_scored/'episodes.jsonl'
    if not prior_path.is_file():raise FileNotFoundError('Actual scored4000 prior missing: '+str(prior_path))
    if a.out.exists():raise FileExistsError('Fresh successor directory required; existing queues/assets untouched')
    ancestor=a.out.parent.resolve()
    while not ancestor.exists():ancestor=ancestor.parent
    if not smoke and shutil.disk_usage(ancestor).free < (7<<30):raise OSError('Require7GiB for cos2000+exact kernels1200+fields/headroom')
    a.out.mkdir(parents=True)
    for name in ('predictions','fields','cosines'): (a.out/name).mkdir()
    mean_out=a.out/'mean1200'
    mean_out.mkdir()
    for name in ('predictions','fields','kernels'):(mean_out/name).mkdir()
    write(a.out/'manifest.json',rows);write(mean_out/'manifest.json',r12 if not smoke else [mapping[r['key']] for r in rows if r['key'] in mapping])
    host,producer,man=make_host(a)
    if sha(producer.__file__)!=c12['host_builder_sha256'] or sha(a.host_manifest)!=c12['host_manifest_sha256']:raise ValueError('Original numerical producer changed')
    code=[Path(__file__).resolve(),REPO/'scripts/run_frozen_subtoken4000.py',REPO/'scripts/run_frozen_subtoken1200.py',REPO/'scripts/benchmark_part1_producer.py',REPO/'scripts/run_fine_mean1200.py',REPO/'src/ics/experiment.py',REPO/'src/ics/methods/rcg.py']
    source_code={str(p):sha(p) for p in code}
    config=dict(n=len(rows),primary=METHOD,frozen=frozen,frozen_sha256=FROZEN_SHA,source_code_sha256=source_code,
        base4000=str(a.base4000.resolve()),base4000_seal_sha256=sha(a.base4000/'sealed.json'),base1200=str(a.base1200.resolve()),
        base4000_scored=str(a.base4000_scored.resolve()),baseline_prior_episodes_sha256=sha(prior_path),
        mean_prepared=str(a.mean_prepared.resolve()),query_gt_in_inference=False,
        exposure='all4000 benchmark draws including repeats; exposed reuse, not fresh confirmation/SOTA',
        numerical_contract='fresh-fitting unit(FP16 q/r float32), source FP32 four single-shift forwards, sigma1.25/window2/lambda16, uniformtau.15; peer stream unnormalized producer is not reused',
        reuse='fold1/2 existing2000 matching lambda16 fine tau15; encode missing2000 plus directMEAN other600 once',
        cache='cos FP32[16384,25] only missing2000; raw_w/rowden1200 for exact original per-fold DirectMEAN',
        expected=dict(query_four_shift=2600,prefix_pairs=1400,shift_forwards=10400,existing_tau15_reuse=2000),
        reader_threads=a.readers,writer_threads=a.writers,cpu_verifiers=a.workers,chunk=a.chunk,inflight=a.inflight,
        peak_gpu_budget_bytes=10<<30,peer_policy='root-coordinated read-only identity; no signaling or ownership of peer')
    write(a.out/'config.json',config)
    mean_config=dict(n=1200,primary=FINE64,new_control=MEAN_FINE,arms=[*ARMS,MEAN_FINE],parameters=c12['parameters'],
        base=str(a.base1200.resolve()),prepared=str(a.mean_prepared.resolve()),source_code_sha256=source_code,
        exposure='existing exposed DEV1200, no confirmation/SOTA',query_truth_opened=False,
        fusion_source=str(a.out.resolve()),runtime_scope='shared producer attribution; not standalone complete cold runtime',
        kernel_cache='raw FP32 weights[16384,25]+original clamped rowden[16384]; exact multiply/sum/div',renderer='unchanged CUDA128->1024 align_corners=False >.5')
    write(mean_out/'config.json',mean_config)
    nb,valid,spatial=geometry('cuda')
    for directory in (a.out,mean_out):
        np.save(directory/'neighbors.npy',nb.cpu().numpy().astype(np.uint16));np.save(directory/'valid_neighbors.npy',valid.cpu().numpy())
    data=Path(man['data_root'])
    def load(row):
        key=row['key'];fp=a.base4000/'fields'/(key+'.npz');pp=a.base4000/'predictions'/(key+'.npz')
        if sha(fp)!=bs['fields'][key] or sha(pp)!=bs['predictions'][key]:raise ValueError('Changed4000 baseline')
        receipt=dict(packet_export=row['packet_export'],packet_sha256=sha(row['packet_export']),base_fields_sha256=bs['fields'][key],base_prediction_sha256=bs['predictions'][key])
        if receipt['packet_sha256']!=bs['inputs'][key]['packet_sha256']:raise ValueError('4000 packet changed')
        with np.load(fp,allow_pickle=False) as z:coarse=z['rcg'].copy();matching=z[FINE16].copy()
        with np.load(pp,allow_pickle=False) as z:masks={name:z[name].copy() for name in ARMS}
        with np.load(row['packet_export'],allow_pickle=False) as z:cov,score=z['cov'].copy(),z['score'].copy()
        with Image.open(data/row['query']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        receipt['query_image_sha256']=sha(data/row['query'])
        cached=None;g=None;expected=None;oldmasks=None
        if key in mapping:
            old=mapping[key];natural=old['key']
            if old['c']!=row['c'] or any(Path(old[k]).name!=Path(row[k]).name for k in ('support','query')):raise ValueError('DirectMEAN source identity changed')
            if sha(old['feature_export'])!=gs['inputs'][natural]['feature_sha256']:raise ValueError('Protected q/r cache changed')
            cached=torch.load(old['feature_export'],map_location='cpu',weights_only=True)
            gp=a.mean_prepared/'fields'/(natural+'.npz');bp=a.base1200/'predictions'/(natural+'.npz');bf=a.base1200/'fields'/(natural+'.npz')
            if sha(gp)!=gs['fields'][natural] or sha(bp)!=s12['predictions'][natural] or sha(bf)!=s12['fields'][natural]:raise ValueError('DirectMEAN checked G/baselines changed')
            with np.load(gp,allow_pickle=False) as z:g=z['mean.control'].copy()
            with np.load(bp,allow_pickle=False) as z:oldmasks={name:z[name].copy() for name in ARMS}
            with np.load(bf,allow_pickle=False) as z:expected=z[FINE16].copy()
            receipt['direct_mean_inputs']=dict(gs['inputs'][natural])
        return cached,rgb,coarse,matching,masks,cov,score,receipt,g,expected,oldmasks
    def save(row,fields,masks,receipt,cosine,direct):
        key=row['key'];fp=a.out/'fields'/(key+'.npz');pp=a.out/'predictions'/(key+'.npz')
        np.savez_compressed(fp,**fields);np.savez_compressed(pp,**masks);record=dict(key=key,field_sha256=sha(fp),prediction_sha256=sha(pp),inputs=receipt)
        if cosine is not None:
            cp=a.out/'cosines'/(key+'.npy');np.save(cp,cosine);record['cosine_sha256']=sha(cp)
        if direct is not None:
            old,dfields,dmasks,kernel,dreceipt=direct;natural=old['key'];df=mean_out/'fields'/(natural+'.npz');dp=mean_out/'predictions'/(natural+'.npz');kp=mean_out/'kernels'/(natural+'.npz')
            np.savez_compressed(df,**dfields);np.savez_compressed(dp,**dmasks);np.savez(kp,**kernel)
            record['direct']=dict(key=natural,field_sha256=sha(df),prediction_sha256=sha(dp),kernel_sha256=sha(kp),inputs=dreceipt)
        return record
    started=time.monotonic();producer_seconds=0.;direct_seconds=0.;records=[];writer_jobs=[];pending=[];encoded=prefixes=reused=0
    def finish(item):
        nonlocal direct_seconds
        row,q16,fine_array,coarse,matching,masks,receipt,g,expected,oldmasks,verify,producer_time=item
        if verify is not None:receipt['prefix_lambda16']=verify.result()
        tick=time.monotonic();q=F.normalize(torch.from_numpy(q16).to('cuda').float(),dim=1);fine=torch.from_numpy(fine_array).to('cuda')
        cos_parts=[]
        for start in range(0,16384,a.chunk):
            stop=min(start+a.chunk,16384);cos_parts.append(torch.einsum('pc,pkc->pk',fine[start:stop],q[nb[start:stop]]))
        cosine=torch.cat(cos_parts);rcg=torch.from_numpy(coarse).to('cuda').flatten();pieces=[]
        for start in range(0,16384,a.chunk):
            stop=min(start+a.chunk,16384);ids=nb[start:stop];w=spatial[start:stop]*valid[start:stop]*torch.exp((cosine[start:stop]-1)/.15)
            pieces.append((w*rcg[ids]).sum(1)/w.sum(1).clamp_min(1e-12))
        f15=torch.cat(pieces).reshape(128,128);f15array=f15.cpu().numpy()
        if row['fold'] in (1,2) and np.count_nonzero(f15array!=matching):raise RuntimeError('Re-encoded normalized tau15 differs from matching existingfield')
        if row['key'] not in mapping:
            old_parts=[];old_tau=c12['parameters'][str(row['fold'])]['tau']
            for start in range(0,16384,a.chunk):
                stop=min(start+a.chunk,16384);ids=nb[start:stop]
                old_w=spatial[start:stop]*torch.exp((cosine[start:stop]-1)/old_tau)*valid[start:stop]
                old_parts.append((old_w*rcg[ids]).sum(1)/old_w.sum(1).clamp_min(1e-12))
            old_field=torch.cat(old_parts).reshape(128,128)
            nf=int(np.count_nonzero(old_field.cpu().numpy()!=matching))
            old_mask=(F.interpolate(old_field[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy()
            nm=int(np.count_nonzero(old_mask!=unpack(masks[FINE16])))
            receipt.update(original_fine16_field_mismatches=nf,original_fine16_mask_mismatches=nm)
            if nf or nm:raise RuntimeError('Actual missing-prefix four-shift producer differs from original sealedfield')
        masks[TAU15],masks[METHOD],cut=fixed_masks(f15,frozen);receipt['fixed_sizecut']=cut
        fields={TAU15:f15array};direct=None
        if row['key'] in mapping:
            old=mapping[row['key']];tau=c12['parameters'][str(row['fold'])]['tau'];mean=torch.from_numpy(g).to('cuda').flatten();rp=[];mp=[];raw=[];den=[]
            for start in range(0,16384,a.chunk):
                stop=min(start+a.chunk,16384);ids=nb[start:stop]
                w=spatial[start:stop]*torch.exp((cosine[start:stop]-1)/tau)*valid[start:stop];d=w.sum(1).clamp_min(1e-12)
                rp.append((w*rcg[ids]).sum(1)/d);mp.append((w*mean[ids]).sum(1)/d);raw.append(w);den.append(d)
            replay=torch.cat(rp).reshape(128,128);newmean=torch.cat(mp).reshape(128,128);ra=replay.cpu().numpy()
            rmask=(F.interpolate(replay[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy()
            nf=int(np.count_nonzero(ra!=expected));nm=int(np.count_nonzero(rmask!=unpack(oldmasks[FINE16])))
            if nf or nm:raise RuntimeError('Original DirectMEAN per-fold producer replay changed')
            oldmasks[MEAN_FINE]=np.packbits((F.interpolate(newmean[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy())
            dreceipt=dict(receipt['direct_mean_inputs'],fine16_field_mismatches=nf,fine16_mask_mismatches=nm,query_image_sha256=receipt['query_image_sha256'])
            direct=(old,{FINE16:ra,MEAN_FINE:newmean.cpu().numpy()},oldmasks,dict(weights=torch.cat(raw).cpu().numpy(),denominator=torch.cat(den).cpu().numpy()),dreceipt)
            direct_seconds+=producer_time+time.monotonic()-tick
        receipt['four_shift_producer_seconds']=producer_time;receipt['readout_seconds']=time.monotonic()-tick
        return fields,masks,receipt,cosine.cpu().numpy() if row['fold'] in (0,3) else None,direct
    with torch.inference_mode(),ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool,ThreadPoolExecutor(a.readers) as readers,ThreadPoolExecutor(a.writers) as writers:
        loaded={i:readers.submit(load,rows[i]) for i in range(min(a.prefetch,len(rows)))}
        for n,row in enumerate(rows):
            cached,rgb,coarse,matching,masks,cov,score,receipt,g,expected,oldmasks=loaded.pop(n).result();later=n+a.prefetch
            if later<len(rows):loaded[later]=readers.submit(load,rows[later])
            needs=row['fold'] in (0,3) or row['key'] in mapping
            if not needs:
                field=torch.from_numpy(matching).to('cuda');masks[TAU15],masks[METHOD],cut=fixed_masks(field,frozen);receipt.update(fixed_sizecut=cut,mode='existing_matching_tau15_reuse')
                writer_jobs.append(writers.submit(save,row,{TAU15:matching},masks,receipt,None,None));reused+=1
            else:
                torch.cuda.synchronize();tick=time.monotonic();verify=None
                if cached is not None:
                    q16=cached['q'].numpy();flag=bool(cached['debiased']);target=host._transform(Image.fromarray(rgb)).to('cuda');receipt['mode']='protected_unit_FP16_cache'
                else:
                    with Image.open(data/row['support']) as im:support=im.convert('RGB')
                    with Image.open(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')) as im:gold=torch.from_numpy((np.asarray(im)==row['c']+1).copy())
                    prefix=part1_only(host,support,gold,Image.fromarray(rgb));q16=prefix['q'];flag=prefix['debiased'];target=prefix['target']
                    if np.count_nonzero(prefix['cov']!=cov):raise RuntimeError('Actual source reference coverage drift')
                    verify=pool.submit(verify_prefix,(q16,prefix['r'],cov,score,coarse,masks['rcg']));prefixes+=1;receipt['mode']='actual_part1_FP32_half_unit_prefix'
                fine_array=fine_grid(host,target,flag);torch.cuda.synchronize();elapsed=time.monotonic()-tick;producer_seconds+=elapsed;encoded+=1
                pending.append((row,q16,fine_array,coarse,matching,masks,receipt,g,expected,oldmasks,verify,elapsed));del target,rgb
                if len(pending)>=a.inflight:
                    item=pending.pop(0);writer_jobs.append(writers.submit(save,item[0],*finish(item)))
            while pending and (pending[0][-2] is None or pending[0][-2].done()):
                item=pending.pop(0);writer_jobs.append(writers.submit(save,item[0],*finish(item)))
            while len(writer_jobs)>=2*a.writers:records.append(writer_jobs.pop(0).result())
            if n==0 or (n+1)%10==0:
                state=dict(state='FUSED_FIXED_INFERENCE',visited=n+1,n=len(rows),encoded=encoded,prefixes=prefixes,reused=reused,completed=len(records),seconds=time.monotonic()-started,producer_seconds=producer_seconds,query_truth_opened=False,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                write(a.out/'state.json',state);print(json.dumps(state),flush=True)
                if state['gpu_peak_bytes']>=10<<30:raise RuntimeError('Measured GPU allocation exceeded10GiB shared budget')
        for item in pending:writer_jobs.append(writers.submit(save,item[0],*finish(item)))
        records.extend(job.result() for job in writer_jobs)
    if not smoke and (encoded,prefixes)!=(2600,1400):raise RuntimeError('Fusion encoding accounting changed')
    direct_records=[r['direct'] for r in records if 'direct' in r]
    if not smoke and len(direct_records)!=1200:raise RuntimeError('DirectMEAN1200 incomplete')
    common=dict(manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),predictions={r['key']:r['prediction_sha256'] for r in records},fields={r['key']:r['field_sha256'] for r in records},inputs={r['key']:r['inputs'] for r in records},query_truth_opened=False,seconds=time.monotonic()-started)
    write(a.out/'sealed.json',dict(state='SMOKE_PREDICTIONS_SEALED' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),primary=METHOD,cosines={r['key']:r['cosine_sha256'] for r in records if 'cosine_sha256' in r},encoded_queries=encoded,prefix_pairs=prefixes,**common))
    write(mean_out/'sealed.json',dict(state='SMOKE_PREDICTIONS_AND_KERNELS_SEALED' if smoke else 'ALL_PREDICTIONS_AND_KERNELS_SEALED',n=len(direct_records),primary=FINE64,new_control=MEAN_FINE,manifest_sha256=sha(mean_out/'manifest.json'),config_sha256=sha(mean_out/'config.json'),predictions={r['key']:r['prediction_sha256'] for r in direct_records},fields={r['key']:r['field_sha256'] for r in direct_records},kernels={r['key']:r['kernel_sha256'] for r in direct_records},inputs={r['key']:r['inputs'] for r in direct_records},neighbors_sha256=sha(mean_out/'neighbors.npy'),valid_neighbors_sha256=sha(mean_out/'valid_neighbors.npy'),seconds=direct_seconds,gpu_seconds=direct_seconds,kernel_bytes=sum((mean_out/'kernels'/(r['key']+'.npz')).stat().st_size for r in direct_records),query_truth_opened=False))
    write(a.out/'state.json',dict(state='SMOKE_COMPLETE' if smoke else 'ALL_PREDICTIONS_SEALED',n=len(rows),query_truth_opened=False));print('FUSED_TAU15_SIZECUT_SEALED',flush=True)


def score_one(job):
    row,out,expected=job
    with np.load(row['packet_export'],allow_pickle=False) as z:truth=unpack(z['truth'])
    with np.load(Path(out)/'predictions'/(row['key']+'.npz'),allow_pickle=False) as z:masks={name:unpack(z[name]) for name in (*ARMS,TAU15,METHOD)}
    iu={name:[int((mask&truth).sum()),int((mask|truth).sum())] for name,mask in masks.items()}
    if any(iu[name]!=expected[name] for name in ARMS):raise ValueError('Original six-arm4000 I/U changed')
    return dict(row,iu=iu)


def score(a):
    rows=json.loads((a.out/'manifest.json').read_text());seal=json.loads((a.out/'sealed.json').read_text());config=json.loads((a.out/'config.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or seal['n']!=4000:raise ValueError('All4000 masks must seal beforeGT')
    verify_code(config)
    prior_path=Path(config['base4000_scored'])/'episodes.jsonl'
    if sha(prior_path)!=config['baseline_prior_episodes_sha256']:raise ValueError('Changed complete scored4000 prior')
    prior={r['key']:r for r in [json.loads(s) for s in prior_path.read_text().splitlines()]}
    for row in rows:
        key=row['key']
        if sha(a.out/'predictions'/(key+'.npz'))!=seal['predictions'][key] or sha(row['packet_export'])!=seal['inputs'][key]['packet_sha256']:raise ValueError('Changed scoring input')
    start=time.monotonic()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:episodes=list(pool.map(score_one,[(r,str(a.out),prior[r['key']]['iu']) for r in rows],chunksize=4))
    arrays={name:np.asarray([r['iu'][name] for r in episodes],np.int64) for name in (*ARMS,TAU15,METHOD)};report,draws=summarize(rows,arrays,{})
    report.update(primary=METHOD,frozen_sha256=FROZEN_SHA,original_six_arm_IU_parity=4000,exposure=config['exposure'],parameter_updates=False,raw_origin_edits='not exported; missing',INSID3='only original DEV241 logic; full4000 absent',cpu_score_seconds=time.monotonic()-start)
    for name in ('corrections_vs_native','corrections_by_class','corrections_by_batch'):report.pop(name,None)
    write(a.out/'report.json',report);np.save(a.out/'bootstrap_photo_draws.npy',draws);(a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes));write(a.out/'score_state.json',dict(state='CPU_SCORE_COMPLETE',n=4000,primary=METHOD));print(json.dumps(report['scores']),flush=True)
    if not a.skip_mean_score:
        from types import SimpleNamespace
        score_mean1200(SimpleNamespace(out=a.out/'mean1200'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('smoke','infer','score'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base4000',type=Path,default=Path('outputs/frozen_subtoken4000_v1'));p.add_argument('--base1200',type=Path,default=Path('outputs/frozen_subtoken1200_v1'));p.add_argument('--mean-prepared',type=Path,default=Path('outputs/fine_mean1200_prepared_v1'));p.add_argument('--frozen',type=Path,default=Path('outputs/claude_rcg2_frozen.json'))
    p.add_argument('--base4000-scored',type=Path,default=Path('outputs/frozen_subtoken4000_scored_v1'))
    p.add_argument('--host-manifest',type=Path,default=Path('outputs/claude_official/batch0.json'));p.add_argument('--host-root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4');p.add_argument('--basis',type=Path,default=Path('/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt'))
    p.add_argument('--workers',type=int,default=6);p.add_argument('--readers',type=int,default=6);p.add_argument('--writers',type=int,default=2);p.add_argument('--prefetch',type=int,default=12);p.add_argument('--inflight',type=int,default=6);p.add_argument('--chunk',type=int,default=512);p.add_argument('--skip-mean-score',action='store_true')
    a=p.parse_args()
    if any(x<1 for x in (a.workers,a.readers,a.writers,a.prefetch,a.inflight,a.chunk)):p.error('Positive execution sizes required')
    try:score(a) if a.stage=='score' else infer(a,a.stage=='smoke')
    except Exception as error:
        if a.out.exists():write(a.out/'state.json',dict(state='FAILED',stage=a.stage,error=repr(error),query_truth_opened=False))
        raise


if __name__=='__main__':main()
