"""W61 fixed-query, three-reference stability; prepare/produce/score are separate."""
import copy
import fcntl
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from collections import defaultdict, Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013')]
import common as c
import w59_normalization_causal as run

OUT=c.task_path('W61')
PI=.088
ARMS=tuple(f'R{k}_{a}{post}' for k in range(3) for a in ('SH','Fixed') for post in ('','+'))


def refkey(b):
    r=b['row']
    return (r['reference_path'],r['reference_mask_path'],json.dumps(r.get('reference_crop')))


def slot(b):
    return (b['dataset'],str(b['row']['fold']),str(b['row']['global_class_id']))


def prepare():
    import w31_demonstration as w31
    from w48_reference_specificity import with_reference
    OUT.mkdir(exist_ok=True,parents=True)
    dest=OUT/'manifest.json'
    if dest.exists():
        print('Prepared manifest exists; retained',run.sha(dest));return
    groups=defaultdict(dict)
    for b in c.rows():
        if c.half(b)=='S' and Path(b['fit']['reference_raw']['payload_path']).is_file():
            groups[slot(b)].setdefault(refkey(b),b)
    rng=np.random.default_rng(2026101606)
    selected=[];exclusions=Counter()
    candidates=w31.rows()
    for domain in c.DOMAINS:
        rows=[b for b in candidates if b['dataset']==domain]
        rows=[rows[i] for i in rng.permutation(len(rows))]
        photos=set();count=0
        for b in rows:
            if c.photo(b) in photos:
                exclusions[domain+':duplicate_query_photo']+=1;continue
            eligible=[]
            for key,other in sorted(groups[slot(b)].items()):
                if key==refkey(b):continue
                r=other['row'];q=b['row']
                if r['reference_path']==q['query_path'] or r.get('reference_rgb_hash')==q.get('query_rgb_hash'):
                    continue
                if r.get('reference_photo_id') is not None and r.get('reference_photo_id')==q.get('query_photo_id'):
                    continue
                eligible.append(other)
            if len(eligible)<2:
                exclusions[domain+':fewer_than_two_extra_references']+=1;continue
            others=[eligible[i] for i in rng.choice(len(eligible),size=2,replace=False)]
            variants=[copy.deepcopy(b)]+[with_reference(b,x['row'],x['fit']['reference_raw'],'reference') for x in others]
            assert len({refkey(x) for x in variants})==3 and all(c.half(x)=='S' for x in variants)
            selected.append(dict(index=str(b['index']),dataset=domain,photo=c.photo(b),
                                 reference_catalog_indices=[str(b['index'])]+[str(x['index']) for x in others],bindings=variants))
            photos.add(c.photo(b));count+=1
            if count==40:break
        assert count==40,(domain,count)
    assert len(selected)==400
    run.write(dest,dict(seed=2026101606,n=400,queries_per_domain=40,reference_count=3,pi=PI,
                        selection='metadata only; no query GT or scores',exclusions=dict(exclusions),episodes=selected))
    print('Prepared 400 queries / 1200 one-shot references; no query GT',dict(exclusions),flush=True)


def fixed_fields(r,q,y,fields):
    import torch
    import torch.nn.functional as F
    lab=c.rc.lab
    mu=.5*(r.mean(0)+q.mean(0));sim=lab.unit(q-mu)@lab.unit(r-mu).T
    kernel=torch.exp((sim-sim.max())/.05)
    df,db=kernel@y,kernel@(1-y)
    original=df/(df+db).clamp_min(1e-30)
    err=float(np.max(np.abs(original.numpy()-fields['c32'])))
    assert err<1e-6
    mf,mb=df/y.sum().clamp_min(1e-6),db/(1-y).sum().clamp_min(1e-6)
    cc=PI*mf/(PI*mf+(1-PI)*mb).clamp_min(1e-30)
    labels=torch.from_numpy(fields['regions64'].astype(np.int64));hot=F.one_hot(labels,64).float()
    region=lambda v:hot@((hot.T@v)/hot.sum(0).clamp_min(1))
    return dict(fields,c32=cc.numpy(),ind0_32=region(cc).numpy(),indn_32=region(cc*lab.minmax(sim.max(1).values)).numpy()),err


def one(item):
    if time.time()>=run.DEADLINE:raise TimeoutError('Authorized window ended')
    import w23_simple_pipeline as sh
    started=time.monotonic();masks={};refs=[];oi=c.rc.independent
    for j,b in enumerate(item['bindings']):
        with c.rc.pc.inference_only(b):
            r,q,mask,y,fields=oi(b,{})
            fixed,err=fixed_fields(r,q,y,fields)
            for a,ff in (('SH',fields),('Fixed',fixed)):
                c.rc.independent=lambda *args,_f=ff,**kwargs:(r,q,mask,y,_f)
                try:pre,final,rho=sh.simple_head(b,crf=True)
                finally:c.rc.independent=oi
                masks[f'R{j}_{a}'],masks[f'R{j}_{a}+']=pre,final
            refs.append(dict(reference_index=item['reference_catalog_indices'][j],fraction=float(y.mean()),rho=rho,c_error=err))
    return run.save_episode(OUT,item['bindings'][0],masks,dict(references=refs),time.monotonic()-started)


def produce(workers):
    assert workers in (1,2)
    manifest=json.loads((OUT/'manifest.json').read_text());items=manifest['episodes']
    lock=(OUT/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Already running');return
    config=dict(manifest_sha256=run.sha(OUT/'manifest.json'),n=400,arms=ARMS,workers=workers,
                sources={str(p.relative_to(ROOT)):run.sha(p) for p in (Path(__file__),Path(run.__file__),ROOT/'scripts/x1012/w23_simple_pipeline.py',ROOT/'scripts/x1012/r5_candidate.py')})
    cp=OUT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==json.loads(json.dumps(config))
    else:run.write(cp,config)
    import psutil
    run.write(OUT/'execution.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),script=str(Path(__file__)),
                                       status='running',workers=workers,started_utc=datetime.now(timezone.utc).isoformat()))
    path=OUT/'episodes.jsonl';records=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
    done={r['index'] for r in records};todo=[x for x in items if x['index'] not in done]
    print('todo',len(todo),'queries; two heads x three references; CPU only',flush=True)
    try:
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=run.init) as pool,path.open('a') as f:
            for n,r in enumerate(pool.map(one,todo,chunksize=1),1):
                records.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if n%20==0:print(len(records),400,'seconds/query',r['seconds'],flush=True)
        assert len(records)==400 and len({r['index'] for r in records})==400
        assert all(run.sha(r['prediction'])==r['prediction_sha256'] for r in records)
        run.write(OUT/'complete.json',dict(n=400,episodes_sha256=run.sha(path),config_sha256=run.sha(cp),query_GT_reads=0,utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as e:
        run.write(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),records=len(records)));raise
    finally:
        e=json.loads((OUT/'execution.json').read_text());e['status']='complete' if (OUT/'complete.json').exists() else 'partial_or_failed'
        run.write(OUT/'execution.json',e);lock.close()


def score():
    import score as sc
    import plan_common as pc
    complete=json.loads((OUT/'complete.json').read_text());assert complete['n']==400
    assert run.sha(OUT/'episodes.jsonl')==complete['episodes_sha256'] and run.sha(OUT/'config.json')==complete['config_sha256']
    meta={r['index']:r for r in map(json.loads,(OUT/'episodes.jsonl').read_text().splitlines())}
    manifest=json.loads((OUT/'manifest.json').read_text())
    with gzip.open(ROOT/'evidence/research/w23_simple_pipeline_S.jsonl.gz','rt') as f:old={r['index']:r for r in map(json.loads,f)}
    records=[]
    for item in manifest['episodes']:
        r=dict(meta[item['index']]);assert run.sha(r['prediction'])==r['prediction_sha256']
        gt,valid=pc.truth(item['bindings'][0]);r['gt']=int(gt.sum())
        with np.load(r['prediction'],allow_pickle=False) as z:
            for a in ARMS:r[a]=c.iu(c.unpack(z[a]),gt,valid)
        assert r['R0_SH']==old[r['index']]['SH'] and r['R0_SH+']==old[r['index']]['SH+']
        for a in ('SH','SH+','Fixed','Fixed+'):
            r['pooled_'+a]=np.sum([r[f'R{j}_{a}'] for j in range(3)],axis=0).tolist()
            js=[r[f'R{j}_{a}'][0]/max(r[f'R{j}_{a}'][1],1) for j in range(3)]
            r['stability_'+a]=dict(mean=float(np.mean(js)),minimum=float(min(js)),maximum=float(max(js)),
                                   area_sd=float(np.std([np.log2((r[f'R{j}_{a}'][2]+1)/1048576) for j in range(3)])))
        records.append(r)
    rng=np.random.default_rng(2026101606);summary={'n_queries':400,'n_reference_configurations':1200,'A0_identity':400,'comparisons':{},'stability':{}}
    lines=['400 original S query photos; three independently evaluated one-shot references each; no reference ensemble']
    for post in ('','+'):
        aa,bb='pooled_Fixed'+post,'pooled_SH'+post
        domains={}
        for d in c.DOMAINS:
            sel=[r for r in records if r['dataset']==d];lo,hi=sc.paired_ci(sel,aa,bb,rng)
            domains[d]=dict(delta=sc.miou(sel,aa)-sc.miou(sel,bb),ci=[lo,hi])
        summary['comparisons'][post or 'pre']=dict(mean_delta=float(np.mean([v['delta'] for v in domains.values()])),domains=domains)
        lines.append(f'Fixed - SH{post}: macro {summary["comparisons"][post or "pre"]["mean_delta"]:+.3f}; '+json.dumps(domains))
    for a in ('SH','SH+','Fixed','Fixed+'):
        xs=[];ys=[]
        for r in records:
            x=np.log2(np.array([q['fraction'] for q in r['references']])+1e-4)
            y=np.log2((np.array([r[f'R{j}_{a}'][2] for j in range(3)])+1)/1048576)
            xs.extend(x-x.mean());ys.extend(y-y.mean())
        x,y=np.asarray(xs),np.asarray(ys)
        cat=[r for r in records if r['R0_SH+'][0]/max(r['R0_SH+'][1],1)<.3]
        s=dict(within_query_ref_area_slope=float(x@y/max(x@x,1e-12)),
               mean_area_sd=float(np.mean([r['stability_'+a]['area_sd'] for r in records])),
               mean_iou=float(np.mean([r['stability_'+a]['mean'] for r in records])),
               mean_observed_min_iou=float(np.mean([r['stability_'+a]['minimum'] for r in records])),
               mean_observed_max_iou=float(np.mean([r['stability_'+a]['maximum'] for r in records])),
               original_SHplus_reference_catastrophic=len(cat),
               rescued_by_another_reference=sum(max(r[f'R{j}_{a}'][0]/max(r[f'R{j}_{a}'][1],1) for j in (1,2))>=.5 for r in cat))
        summary['stability'][a]=s;lines.append(a+' '+json.dumps(s))
    with gzip.open(ROOT/'evidence/research/w61_reference_stability_S.jsonl.gz','wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    run.write(OUT/'summary.json',summary);(OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    run.write(OUT/'scored.json',dict(n=400,utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='produce':produce(int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1]=='score':score()
