#!/usr/bin/env python3
"""Exact existing-mask A/B family optimization; GT-labelled DEV search only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ics.experiment import sha,metric,photo_groups,summarize


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def checked(path,digest):
    if sha(path)!=digest:raise ValueError('Changed source '+str(path))


def mask(record):
    with np.load(record['path'],allow_pickle=False) as z:a=z[record['key']].copy()
    if a.dtype!=np.uint8 or a.shape!=(131072,):raise ValueError('Expected packed1024 mask')
    if 'packed_array_sha256' in record and hashlib.sha256(a.tobytes()).hexdigest()!=record['packed_array_sha256']:
        raise ValueError('Packed mask identity changed')
    return a


def library(path,policy):
    d=json.loads(Path(path).read_text());seen=set();arms=[]
    for arm in d['arms']:
        eligible=arm.get('strict_eligible',arm.get('same_information_eligible',False)) if policy=='strict' else arm.get('label_fitted_extended_eligible',arm.get('same_information_eligible',False))
        if eligible and arm['producer_id'] not in seen:
            arms.append(arm);seen.add(arm['producer_id'])
    if not arms or len(arms)>20:raise ValueError('Dense exact histogram requires1..20 distinct producers; no silent library subset')
    rows=d['rows']
    if len(rows)!=d['n'] or len({r['key'] for r in rows})!=len(rows):raise ValueError('All sampled draws required')
    return d,rows,arms


def histogram_one(job):
    row,ids=job;state=np.zeros(1024*1024,dtype=np.uint32)
    for j,arm in enumerate(ids):state|=np.unpackbits(mask(row['masks'][arm])).astype(np.uint32)<<j
    packet=row['packet']
    with np.load(packet['path'],allow_pickle=False) as z:truth=np.unpackbits(z['truth']).astype(np.uint32)
    if truth.shape!=state.shape:raise ValueError('GT size differs')
    counts=np.bincount((2*state+truth).astype(np.int64),minlength=2*(1<<len(ids))).reshape(-1,2).astype(np.uint32)
    if int(counts.sum())!=1024**2:raise ValueError('Nonexhaustive membership')
    return counts


def hist(a):
    started=time.monotonic();d,rows,arms=library(a.manifest,a.policy)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);ids=[x['id'] for x in arms]
    write(a.out/'manifest.json',d);write(a.out/'arms.json',arms)
    write(a.out/'state.json',dict(state='VALIDATING_ALL_LIBRARY_INPUTS',n=len(rows),K=len(arms),GT_read_for_hist=False))
    files={}
    for row in rows:
        for arm in d['arms']:
            rec=row['masks'][arm['id']]
            if rec['path'] in files and files[rec['path']]!=rec['sha256']:raise ValueError('Conflicting file receipt')
            files[rec['path']]=rec['sha256']
        rec=row['packet'];files[rec['path']]=rec['sha256']
    for path,digest in files.items():checked(path,digest)
    write(a.out/'input_receipt.json',dict(library_manifest_sha256=sha(a.manifest),files=files,policy=a.policy,GT_use='explicit global DEV histogram/search; no per-example inference selector'))
    counts=np.lib.format.open_memmap(a.out/'patterns.npy',mode='w+',dtype=np.uint32,shape=(len(rows),1<<len(arms),2))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for i,value in enumerate(pool.map(histogram_one,[(r,ids) for r in rows],chunksize=4)):
            counts[i]=value
            if (i+1)%200==0:
                counts.flush();state=dict(state='BUILDING_MEMBERSHIP_HISTOGRAM',completed=i+1,n=len(rows),K=len(arms),seconds=time.monotonic()-started)
                write(a.out/'state.json',state);print(json.dumps(state),flush=True)
    counts.flush()
    cls=np.array([r['c'] for r in rows]);class_ids=np.unique(cls)
    class_counts=np.stack([counts[cls==c].sum(axis=0,dtype=np.int64) for c in class_ids])
    np.save(a.out/'class_patterns.npy',class_counts)
    patterns=np.arange(1<<len(arms));gt=class_counts[:,:,1].sum(axis=1)
    scores={};baseline_iu={}
    for i,arm in enumerate(arms):
        on=(patterns&(1<<i))!=0
        iu=np.stack((counts[:,on,1].sum(axis=1,dtype=np.int64),counts[:,:,1].sum(axis=1,dtype=np.int64)+counts[:,on,0].sum(axis=1,dtype=np.int64)),axis=1)
        baseline_iu[arm['id']]=iu;scores[arm['id']]=metric(iu,cls)
    np.savez_compressed(a.out/'baseline_counts.npz',**baseline_iu)
    write(a.out/'histogram_complete.json',dict(state='MEMBERSHIP_HISTOGRAM_COMPLETE',n=len(rows),K=len(arms),arms=ids,class_ids=class_ids.tolist(),scores=scores,seconds=time.monotonic()-started,all_inputs_validated_before_GT=True))
    write(a.out/'state.json',dict(state='MEMBERSHIP_HISTOGRAM_COMPLETE',n=len(rows),K=len(arms),seconds=time.monotonic()-started))
    print('MEMBERSHIP_HISTOGRAM_COMPLETE',flush=True)


def zeta(values,subset):
    out=values.copy();n=out.shape[1];step=1
    while step<n:
        view=out.reshape(out.shape[0],-1,2,step,2)
        if subset:view[:,:,1,:,:]+=view[:,:,0,:,:]
        else:view[:,:,0,:,:]+=view[:,:,1,:,:]
        step*=2
    return out


def optimize_origin(job):
    path,p=job;started=time.monotonic();h=np.load(path);L=int(round(np.log2(h.shape[1])));K=L-1;n=1<<K
    small=np.arange(n);low=(1<<p)-1;indices=(small&low)|((small>>p)<<(p+1))
    h0=h[:,indices,:];h1=h[:,indices|(1<<p),:]
    additions=(h0.sum(axis=1)[:,None,:]-zeta(h0,True)[:,(n-1)^small,:]).transpose(1,0,2)
    keep=zeta(h1,False).transpose(1,0,2);truth=h[:,:,1].sum(axis=1)
    pop=np.array([bin(int(x)).count('1') for x in small],dtype=np.uint8)
    best={k:{'score':-1.,'recipes':[]} for k in range(1,L+1)}
    baseline=[]
    # Each baseline is contained: base itself (a=b=0), other Si (a=b=singleton).
    for i in range(L):
        q=0 if i==p else 1<<(i if i<p else i-1)
        score=float(100*np.mean((additions[q,:,1]+keep[q,:,1])/np.maximum(truth+additions[q,:,0]+keep[q,:,0],1)))
        baseline.append(score)
    for start in range(0,n,16):
        aa=np.arange(start,min(start+16,n))
        intersection=additions[aa,None,:,1]+keep[None,:,:,1]
        union=truth[None,None,:]+additions[aa,None,:,0]+keep[None,:,:,0]
        scores=100*np.mean(intersection/np.maximum(union,1),axis=2)
        costs=1+pop[aa[:,None]|small[None,:]]
        for cost in range(1,L+1):
            use=costs==cost
            if not use.any():continue
            maximum=float(scores[use].max());current=best[cost]
            if maximum<current['score']:continue
            positions=np.argwhere(use&(scores==maximum))
            recipes=[dict(origin=p,a=int(aa[i]),b=int(j),cost=cost,score=maximum) for i,j in positions]
            if maximum>current['score']:best[cost]={'score':maximum,'recipes':recipes}
            else:current['recipes'].extend(recipes)
    return dict(origin=p,best_by_cost=best,baseline_containment_scores=baseline,recipes_evaluated=n*n,seconds=time.monotonic()-started)


def optimize(a):
    started=time.monotonic();complete=json.loads((a.out/'histogram_complete.json').read_text());arms=json.loads((a.out/'arms.json').read_text());L=len(arms)
    if complete['state']!='MEMBERSHIP_HISTOGRAM_COMPLETE':raise ValueError('Histogram incomplete')
    best={k:{'score':-1.,'recipes':[]} for k in range(1,L+1)};origins=[]
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(optimize_origin,[(str(a.out/'class_patterns.npy'),p) for p in range(L)]):
            origins.append(result)
            for k in range(1,L+1):
                value=result['best_by_cost'][k];old=best[k]
                if value['score']>old['score']:best[k]=value
                elif value['score']==old['score']:old['recipes'].extend(value['recipes'])
            write(a.out/'state.json',dict(state='EXACT_ALL_A_B_SEARCH',completed_origins=len(origins),origins=L,recipes_evaluated=sum(x['recipes_evaluated'] for x in origins),seconds=time.monotonic()-started))
            print(json.dumps(dict(origin=result['origin'],origin_seconds=result['seconds'],best_score=max(v['score'] for v in result['best_by_cost'].values()))),flush=True)
    baseline=np.array([complete['scores'][arm['id']] for arm in arms])
    certificate_error=float(max(np.max(np.abs(np.array(x['baseline_containment_scores'])-baseline)) for x in origins))
    if certificate_error>1e-10:raise ValueError('Baseline inclusion certificate failed')
    maximum=max(v['score'] for v in best.values());minimum_cost=min(k for k,v in best.items() if v['score']==maximum)
    all_best=[r for v in best.values() if v['score']==maximum for r in v['recipes']]
    selected=best[minimum_cost]['recipes'];pareto=[];highest=-1.
    for k,v in sorted(best.items()):
        if v['score']>highest:pareto.append(dict(cost=k,score=v['score'],recipes=v['recipes']));highest=v['score']
    def names(recipe):
        p=recipe['origin'];others=[i for i in range(L) if i!=p]
        return dict(recipe,origin_method=arms[p]['id'],add_methods=[arms[i]['id'] for j,i in enumerate(others) if recipe['a']&(1<<j)],delete_methods=[arms[i]['id'] for j,i in enumerate(others) if recipe['b']&(1<<j)])
    result=dict(state='EXACT_GLOBAL_DEV_OPTIMUM',n=complete['n'],distinct_producers=L,K_operators_per_side=L-1,recipes_evaluated=L*(1<<(2*(L-1))),maximum_miou=maximum,minimum_cost_at_maximum=minimum_cost,selected_recipes=[names(r) for r in selected],all_maximum_recipes=[names(r) for r in all_best],best_by_cost=best,pareto=[dict(v,recipes=[names(r) for r in v['recipes']]) for v in pareto],baseline_scores=complete['scores'],baseline_inclusion_certificate=dict(max_error_pp=certificate_error,every_prior_is_represented_under_every_origin=True),cost_definition='1+popcount(a|b): distinct complete-mask producer count proxy; not measured seconds or a sparsity theorem',tie_rule='exact float64 score equality, then lowest cost; no epsilon tolerance',selection='Global GT-labelled DEV recipe selection, not per-example GT routing, not untouched test/SOTA',seconds=time.monotonic()-started,per_origin=origins)
    write(a.out/'optimization.json',result);write(a.out/'state.json',dict(state=result['state'],maximum_miou=maximum,cost=minimum_cost,recipes_evaluated=result['recipes_evaluated'],seconds=result['seconds']))
    print(json.dumps(dict(maximum_miou=maximum,cost=minimum_cost,selected_recipes=result['selected_recipes'],seconds=result['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('hist','optimize','finalize','all'))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy',choices=('strict','extended'),default='strict');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.stage in ('hist','all'):hist(a)
    if a.stage in ('optimize','all'):optimize(a)
    if a.stage in ('finalize','all'):finalize(a)


if __name__=='__main__':main()
