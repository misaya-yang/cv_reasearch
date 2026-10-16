"""W64: reference-selected native key-head graph; sealed predictions before GT."""
from pathlib import Path
import fcntl
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013'),str(ROOT/'scripts/x1012s')]
import common as c
import plan_common as pc
import w59_normalization_causal as io

OUT=c.task_path('W64_refgraph')
DOMAINS=('coco','lvis','pascal_part','paco_part','suim','isic')
HEADS=tuple(f'H{h:02d}' for h in range(17))


def prepare():
    import w31_demonstration as w31
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():print('Retain manifest',io.sha(path));return
    rng=np.random.default_rng(20261015);rows=[]
    original=w31.rows()
    for d in DOMAINS:
        eligible=sorted([b for b in original if b['dataset']==d],key=lambda b:b['row']['episode_id'])
        rows.extend(eligible[i] for i in rng.choice(len(eligible),15,replace=False))
    assert len(rows)==90 and len({str(b['index']) for b in rows})==90 and all(c.half(b)=='S' for b in rows)
    io.write(path,dict(n=90,seed=20261015,domains=DOMAINS,query_GT_reads=0,episodes=rows))
    print('Prepared90 original-S episodes,15 per six domains, metadata only')


def initialize():
    global ENCODER,CAPTURE,THREAD_LIMIT
    import torch
    from threadpoolctl import threadpool_limits
    c.initialize()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    THREAD_LIMIT=threadpool_limits(limits=1)
    spec=importlib.util.spec_from_file_location('w64_frozen_encoder',c.ASSETS/'a/x1013_official/frozen/encoder.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    ENCODER=module.TimmDINOv3(str(c.ASSETS/'demo4_cache/models/dinov3-vitl16-timm')).eval().requires_grad_(False)
    CAPTURE={}
    ENCODER.m.blocks[-1].attn.k_norm.register_forward_hook(lambda mod,inp,out:CAPTURE.__setitem__('key',out))


def encode_pair(b):
    import torch
    from PIL import Image
    began=time.monotonic()
    keys=[];errors=[]
    for role in ('reference','query'):
        with Image.open(c.ASSETS/b['row'][role+'_path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        crop=b['row'].get(role+'_crop')
        if crop is not None:
            x0,y0,x1,y1=crop;rgb=rgb[y0:y1,x0:x1]
        img=Image.fromarray(rgb)
        with torch.inference_mode():
            o=ENCODER.get_intermediate_layers(c.rc.host.transform(img)[None],n=1,reshape=True)[0]
        o=o[0].permute(1,2,0).reshape(4096,1024)
        k=CAPTURE.pop('key')[0,:,5:,:].detach().clone()
        assert k.shape==(16,4096,64) and torch.isfinite(k).all()
        raw=c.rc.raw(b['fit']['reference_raw' if role=='reference' else 'query_raw_identity'],{})
        error=float((o-raw).abs().max());relative=float((o-raw).norm()/raw.norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(o,raw,dim=1).min())
        # Numeric identity of independent CPU replay, not a GT-based acceptance gate.
        assert relative<1e-3 and cosine>.9999,(role,error,relative,cosine)
        errors.append(dict(role=role,max_abs=error,relative_l2=relative,min_token_cosine=cosine))
        keys.append(k)
    return keys,errors,time.monotonic()-began


def graph(x):
    import torch
    z=torch.nn.functional.normalize(x.float(),dim=1)
    sim=z@z.T;sim.diagonal().fill_(-1e9)
    neighbors=sim.topk(20,dim=1).indices.numpy().astype(np.uint16)
    return c.rc.graph(x,neighbors).tocsr()


def average_graph(laps):
    adjacency=sum((sparse.diags(l.diagonal())-l for l in laps),sparse.csr_matrix(laps[0].shape))/len(laps)
    adjacency=adjacency.tocsr();ii=[];jj=[];vv=[]
    for i in range(adjacency.shape[0]):
        sl=slice(adjacency.indptr[i],adjacency.indptr[i+1]);cols=adjacency.indices[sl];vals=adjacency.data[sl]
        positive=vals>0;cols,vals=cols[positive],vals[positive]
        order=np.lexsort((cols,-vals))[:20]
        ii.extend([i]*len(order));jj.extend(cols[order]);vv.extend(vals[order])
    w=sparse.csr_matrix((vv,(ii,jj)),shape=adjacency.shape)
    w=w.multiply(w.T).sqrt().tocsr()
    w=w/max(float(np.asarray(w.sum(1)).ravel().mean()),1e-8)
    return (sparse.diags(np.asarray(w.sum(1)).ravel())-w).tocsr()


def folds(y):
    ii,jj=np.indices((64,64));out=[]
    for a in (0,1):
        for b in (0,1):
            v=((ii<32) if a==0 else (ii>=32))&((jj<32) if b==0 else (jj>=32))
            # Dilate this quadrant one token toward the visible quadrants.
            hidden=((ii<=32) if a==0 else (ii>=31))&((jj<=32) if b==0 else (jj>=31))
            anchor=~hidden;v=v.ravel();anchor=anchor.ravel()
            if y[v].sum()>0 and (1-y[v]).sum()>0 and y[anchor].sum()>0 and (1-y[anchor]).sum()>0:
                out.append((v,anchor))
    return out


def harmonic(lap,y,anchor):
    _,labels=connected_components(lap,directed=False)
    supported=np.isin(labels,np.unique(labels[anchor]))
    unknown=~anchor&supported
    f=np.full(len(y),.5);f[anchor]=y[anchor]
    if not unknown.any():return f,False
    block=lap[unknown][:,unknown].astype(np.float64).tocsr()
    rhs=-(lap[unknown][:,anchor]@y[anchor])
    precondition=sparse.diags(1/np.maximum(block.diagonal(),1e-12))
    sol,status=cg(block,rhs,x0=np.full(len(rhs),.5),M=precondition,rtol=1e-7,atol=1e-9,maxiter=3000)
    failed=bool(status!=0 or not np.isfinite(sol).all() or sol.min()<-1e-5 or sol.max()>1+1e-5)
    if failed:return np.full(len(y),.5),True
    f[unknown]=np.clip(sol,0,1)
    return f,False


def select(laps,y):
    ff=folds(y);risks=[];failures=[]
    if not ff:return 0,[],[],0
    for lap in laps:
        values=[];failed=[]
        for v,anchor in ff:
            pred,bad=harmonic(lap,y,anchor);z=pred[v];t=y[v];s=(z-t)**2
            values.append(float(.5*(t*s).sum()/t.sum()+.5*((1-t)*s).sum()/(1-t).sum()))
            failed.append(bad)
        risks.append(values);failures.append(failed)
    return int(np.argmin(np.mean(risks,axis=1))),risks,failures,len(ff)


def one(b):
    import torch
    import w23_simple_pipeline as sh
    import w42_ordered_cut as capture
    import produce as postprocess
    began=time.monotonic();oi,og=c.rc.independent,c.rc.graph
    with pc.inference_only(b):
        r,q,mask,y,fields=oi(b,{})
        c.rc.independent=lambda *a,**k:(r,q,mask,y,fields)
        try:
            field,baseline,u,m,rho=capture.capture(b)
            masks={'SH':baseline,'H00':baseline}
            meta=dict(selected=0,rho=rho,fallback=rho>.9,valid_folds=0,reference_risks=[],reference_solver_failures=[],query_solver_failures=[],encoder_calls=0,encoder_seconds=0.,encoder_identity=[])
            commonpos=baseline if u is None else (u>.5)&(m>.5)
            commonneg=~baseline if u is None else (u<=.5)&(m<=.5)
            if rho>.9:
                masks.update({a:baseline for a in HEADS+('Mean','Ref','SH+','Mean+','Ref+')})
            else:
                (rk,qk),errors,seconds=encode_pair(b)
                rr=graph(r);rlaps=[rr]+[graph(rk[h]) for h in range(16)]
                chosen,risk,failed,nfold=select(rlaps,y.numpy().astype(np.float64))
                meta.update(selected=chosen,reference_risks=risk,reference_solver_failures=failed,valid_folds=nfold,
                            encoder_identity=errors,encoder_calls=2,encoder_seconds=seconds)
                del rk,rlaps,rr
                qlaps=[og(q,fields['query_neighbors20']).tocsr()]+[graph(qk[h]) for h in range(16)]
                uniform=average_graph(qlaps[1:]);del qk
                for a,lap in list(zip(HEADS[1:],qlaps[1:]))+[('Mean',uniform)]:
                    c.rc.graph=lambda *args,_lap=lap,**kwargs:_lap
                    try:
                        pre,_,_=sh.simple_head(b,crf=False)
                    except (AssertionError,RuntimeError,ValueError) as exc:
                        pre=baseline;meta['query_solver_failures'].append(dict(arm=a,type=type(exc).__name__,message=str(exc)))
                    masks[a]=pre
                masks['Ref']=masks[HEADS[chosen]]
                for a in ('SH','Mean','Ref'):
                    masks[a+'+']=postprocess.crf(b,masks[a])
        finally:c.rc.independent,c.rc.graph=oi,og
    import w31_demonstration as w31
    path=OUT/'predictions'/(c.key(b)+'.npz')
    c.save_npz(path,**{a:c.pack(v) for a,v in masks.items()},common_pos=c.pack(commonpos),common_neg=c.pack(commonneg))
    return dict(w31.meta(b),**meta,prediction=str(path),prediction_sha256=io.sha(path),query_GT_reads=0,seconds=time.monotonic()-began)


def freeze_config(workers):
    paths=[Path(__file__),ROOT/'docs/research/x1012_tasks/W64_reference_head_graph_PLAN.md',ROOT/'scripts/x1012/r5_candidate.py',
           ROOT/'scripts/x1012/w23_simple_pipeline.py',ROOT/'scripts/x1012/w42_ordered_cut.py',c.ASSETS/'a/x1013_official/frozen/encoder.py']
    config=dict(n=90,workers=workers,threads_per_worker=1,device='cpu',manifest_sha256=io.sha(OUT/'manifest.json'),
                sources={str(p):pc.sha(p) for p in paths},weights_sha256=pc.sha(c.ASSETS/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),new_feature_files=0)
    cp=OUT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:io.write(cp,config)
    return config


def smoke():
    # Same first manifest episode, no query GT; retained as part of the planned run.
    initialize();b=json.loads((OUT/'manifest.json').read_text())['episodes'][0]
    r=one(b);assert not (OUT/'episodes.jsonl').exists()
    (OUT/'episodes.jsonl').write_text(json.dumps(r)+'\n')
    io.write(OUT/'smoke.json',dict(index=r['index'],encoder_identity=r['encoder_identity'],query_GT_reads=0,
                                  valid_folds=r['valid_folds'],seconds=r['seconds']))
    print(json.dumps(json.loads((OUT/'smoke.json').read_text()),indent=2))


def produce(workers=2):
    import psutil
    assert workers in (1,2)
    lock=(OUT/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Already running');return
    freeze_config(workers);manifest=json.loads((OUT/'manifest.json').read_text());path=OUT/'episodes.jsonl'
    records=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
    done={r['index'] for r in records};assert len(done)==len(records)
    todo=[b for b in manifest['episodes'] if str(b['index']) not in done]
    io.write(OUT/'execution.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),script=str(Path(__file__)),
                                     status='running',workers=workers,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool,path.open('a') as stream:
            for row in pool.map(one,todo,chunksize=1):
                records.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(len(records),90,row['dataset'],'seconds',round(row['seconds'],2),'selected',row['selected'],flush=True)
        assert len(records)==90 and all(io.sha(r['prediction'])==r['prediction_sha256'] for r in records)
        io.write(OUT/'complete.json',dict(n=90,episodes_sha256=io.sha(path),config_sha256=io.sha(OUT/'config.json'),query_GT_reads=0,utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        io.write(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),records=len(records)));raise
    finally:
        state=json.loads((OUT/'execution.json').read_text());state['status']='complete' if (OUT/'complete.json').exists() else 'partial_or_failed'
        io.write(OUT/'execution.json',state);lock.close()


def score_run():
    import score
    from w59_window_analysis import draws
    seal=json.loads((OUT/'complete.json').read_text());assert seal['n']==90
    assert io.sha(OUT/'episodes.jsonl')==seal['episodes_sha256'] and io.sha(OUT/'config.json')==seal['config_sha256']
    meta={r['index']:r for r in map(json.loads,(OUT/'episodes.jsonl').read_text().splitlines())}
    with gzip.open(ROOT/'evidence/research/w23_simple_pipeline_S.jsonl.gz','rt') as f:old={r['index']:r for r in map(json.loads,f)}
    with gzip.open(ROOT/'evidence/research/x1013_E1_iu.jsonl.gz','rt') as f:external={r['index']:r for r in map(json.loads,f)}
    manifest=json.loads((OUT/'manifest.json').read_text());rows=[]
    arms=HEADS+('SH','SH+','Ref','Ref+','Mean','Mean+')
    for b in manifest['episodes']:
        r=dict(meta[str(b['index'])]);assert io.sha(r['prediction'])==r['prediction_sha256']
        gt,valid=pc.truth(b)
        with np.load(r['prediction'],allow_pickle=False) as z:
            masks={a:c.unpack(z[a])&valid for a in arms}
            for a,m in masks.items():r[a]=c.iu(m,gt,valid)
            assert r['SH']==old[r['index']]['SH'] and r['SH+']==old[r['index']]['SH+']
            baseline=external[r['index']]
            assert all(str(r[k])==str(baseline[k]) for k in ('dataset','fold','cls','photo'))
            r['foris']=baseline['foris']
            best=max(HEADS,key=lambda a:r[a][0]/max(r[a][1],1));r['head_oracle']=r[best];r['oracle_head']=best
            common=(c.unpack(z['common_pos'])|c.unpack(z['common_neg']))&valid
            r['edits']={}
            for a in ('Ref','Mean','Ref+','Mean+'):
                parent=masks['SH+' if a.endswith('+') else 'SH'];changed=masks[a]
                added=changed&~parent;removed=~changed&parent
                value=dict(add_tp=int((added&gt).sum()),add_fp=int((added&~gt).sum()),del_tp=int((removed&gt).sum()),del_fp=int((removed&~gt).sum()))
                if not a.endswith('+'):
                    value.update(common_corrected=int((common&(parent^gt)&~(changed^gt)).sum()),
                                 common_damaged=int((common&~(parent^gt)&(changed^gt)).sum()))
                r['edits'][a]=value
        rows.append(r)
    pairs=[('Ref','SH'),('Ref','Mean'),('Ref+','SH+'),('Ref+','Mean+'),('Ref+','foris'),('SH+','foris'),('Mean','SH'),('head_oracle','SH')]
    scores={};rng=np.random.default_rng(2026101611);allarms=sorted({a for pair in pairs for a in pair})
    points={};samples={}
    for d in DOMAINS:
        sel=[r for r in rows if r['dataset']==d];points[d]={a:score.miou(sel,a) for a in allarms};samples[d]=draws(sel,allarms,rng)
    for a,b in pairs:
        dd=np.mean([samples[d][a]-samples[d][b] for d in DOMAINS],axis=0)
        scores[a+'-'+b]=dict(delta=float(np.mean([points[d][a]-points[d][b] for d in DOMAINS])),ci=np.percentile(dd,[2.5,97.5]).tolist(),
                             by_domain={d:points[d][a]-points[d][b] for d in DOMAINS})
    edits={a:{k:sum(r['edits'][a].get(k,0) for r in rows) for k in rows[0]['edits'][a]} for a in ('Ref','Mean','Ref+','Mean+')}
    numerical=sum(bool(r['query_solver_failures']) or any(any(v) for v in r['reference_solver_failures']) for r in rows)/len(rows)
    gate=(scores['Ref-SH']['delta']>=.5 and scores['Ref-Mean']['delta']>=.5 and min(scores['Ref-SH']['by_domain'].values())>=-1
          and edits['Ref']['common_corrected']>edits['Ref']['common_damaged'] and numerical<=.05)
    selected=[r for r in rows if r['selected']!=0 and r['valid_folds']]
    transfer={'n_changed_head':len(selected)}
    if len(selected)>2:
        from scipy.stats import spearmanr
        source_advantage=[np.mean(r['reference_risks'][0])-np.mean(r['reference_risks'][r['selected']]) for r in selected]
        query_delta=[r['Ref'][0]/max(r['Ref'][1],1)-r['SH'][0]/max(r['SH'][1],1) for r in selected]
        correlation=float(spearmanr(source_advantage,query_delta).statistic)
        transfer.update(mean_source_risk_reduction=float(np.mean(source_advantage)),mean_query_iou_delta=float(np.mean(query_delta)),
                        spearman_source_improvement_query_gain=correlation if np.isfinite(correlation) else None)
    summary=dict(n=90,comparisons=scores,domain_scores=points,edits=edits,continue_gate=gate,numerical_failure_episode_fraction=numerical,
                 reference_to_query=transfer,
                 selected_counts={str(h):sum(r['selected']==h for r in rows) for h in range(17)},fallback=sum(r['fallback'] for r in rows),
                 no_valid_fold=sum(not r['valid_folds'] for r in rows),encoder_calls=sum(r['encoder_calls'] for r in rows),
                 median_episode_seconds=float(np.median([r['seconds'] for r in rows])),
                 median_encoder_seconds=float(np.median([r['encoder_seconds'] for r in rows if r['encoder_calls']])),
                 baseline_identity=90,scope='S development90, six-domain observed-slot macro;500 paired query-photo draws;fixed reference configurations')
    with gzip.open(ROOT/'evidence/research/w64_reference_head_graph_S.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    io.write(OUT/'summary.json',summary);io.write(ROOT/'evidence/research/w64_reference_head_graph_summary.json',summary)
    io.write(OUT/'scored.json',dict(n=90,scorer_sha256=io.sha(Path(__file__)),utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='produce':produce(int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1]=='score':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):score_run()
