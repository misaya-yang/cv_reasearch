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
    allowed=list(range(L)) if a.fixed_origin_id is None else [next(i for i,x in enumerate(arms) if x['id']==a.fixed_origin_id)]
    best={k:{'score':-1.,'recipes':[]} for k in range(1,L+1)};origins=[]
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(optimize_origin,[(str(a.out/'class_patterns.npy'),p) for p in allowed]):
            origins.append(result)
            for k in range(1,L+1):
                value=result['best_by_cost'][k];old=best[k]
                if value['score']>old['score']:best[k]=value
                elif value['score']==old['score']:old['recipes'].extend(value['recipes'])
            write(a.out/'state.json',dict(state='EXACT_ALL_A_B_SEARCH',completed_origins=len(origins),origins=len(allowed),recipes_evaluated=sum(x['recipes_evaluated'] for x in origins),seconds=time.monotonic()-started))
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
    result=dict(state='EXACT_GLOBAL_DEV_OPTIMUM',n=complete['n'],distinct_producers=L,K_operators_per_side=L-1,allowed_origin_ids=[arms[i]['id'] for i in allowed],recipes_evaluated=len(allowed)*(1<<(2*(L-1))),maximum_miou=maximum,minimum_cost_at_maximum=minimum_cost,selected_recipes=[names(r) for r in selected],all_maximum_recipes=[names(r) for r in all_best],best_by_cost=best,pareto=[dict(v,recipes=[names(r) for r in v['recipes']]) for v in pareto],baseline_scores=complete['scores'],baseline_inclusion_certificate=dict(max_error_pp=certificate_error,every_prior_is_represented_under_every_allowed_origin=True),cost_definition='1+popcount(a|b): outer complete-mask producer count proxy; composite rows retain multiple upstream dependencies, not atomic modules or measured seconds',tie_rule='exact float64 score equality, then lowest cost; no epsilon tolerance',selection='Global GT-labelled DEV recipe selection, not per-example GT routing, not untouched test/SOTA',seconds=time.monotonic()-started,per_origin=origins)
    write(a.out/'optimization.json',result);write(a.out/'state.json',dict(state=result['state'],maximum_miou=maximum,cost=minimum_cost,recipes_evaluated=result['recipes_evaluated'],seconds=result['seconds']))
    print(json.dumps(dict(maximum_miou=maximum,cost=minimum_cost,selected_recipes=result['selected_recipes'],seconds=result['seconds'])),flush=True)


def recipe_lut(recipe,L):
    patterns=np.arange(1<<L);p=recipe['origin'];others=[i for i in range(L) if i!=p]
    small=sum(((patterns>>i)&1)<<j for j,i in enumerate(others))
    return np.where((patterns&(1<<p))!=0,(small&recipe['b'])==recipe['b'],(small&recipe['a'])!=0)


def edit_attribution(rows,arrays,corrections):
    cls=np.array([r['c'] for r in rows]);result={}
    for name,records in corrections.items():
        terms=[]
        for c in np.unique(cls):
            ix=cls==c;I0,U0=arrays['native'][ix].sum(axis=0);I1,U1=arrays[name][ix].sum(axis=0)
            counts={k:sum(r[k] for r in records if r['c']==c) for k in ['add_TP','delete_TP','delete_FP','add_FP']}
            J0=I0/max(U0,1)
            parts=np.array([counts['add_TP'],-counts['delete_TP'],J0*counts['delete_FP'],-J0*counts['add_FP']],dtype=float)/max(U1,1)
            if abs(parts.sum()-(I1/max(U1,1)-J0))>1e-10:raise ValueError('Class edit attribution does not reproduce gain')
            terms.append(parts)
        values=100*np.mean(terms,axis=0);gain=metric(arrays[name],cls)-metric(arrays['native'],cls)
        if abs(values.sum()-gain)>1e-10:raise ValueError('Macro edit attribution does not reproduce gain')
        result[name]=dict(zip(['added_TP_pp','deleted_TP_pp','deleted_FP_pp','added_FP_pp'],values.tolist()),sum_pp=float(values.sum()),gain_pp=gain,definition='per-class (addTP-deleteTP+native_J0*(deleteFP-addFP))/final_U1, then mean classes x100; exact additive accounting, not causal attribution')
    return result


def final_mask_one(job):
    row,arms,recipes,directory=job;needed=set()
    for recipe in recipes:
        p=recipe['origin'];others=[i for i in range(len(arms)) if i!=p]
        needed.add(p);needed.update(i for j,i in enumerate(others) if (recipe['a']|recipe['b'])&(1<<j))
    values={}
    for i in needed:
        rec=row['masks'][arms[i]['id']];checked(rec['path'],rec['sha256']);values[i]=mask(rec)
    masks={}
    for k,recipe in enumerate(recipes):
        p=recipe['origin'];origin=values[p];candidate=origin.copy();others=[i for i in range(len(arms)) if i!=p]
        for j,i in enumerate(others):
            if recipe['a']&(1<<j):candidate|=values[i]&~origin
        for j,i in enumerate(others):
            if recipe['b']&(1<<j):candidate&=~(origin&~values[i])
        masks['recipe%d'%k]=candidate
    path=Path(directory)/'finalists'/(row['key']+'.npz');np.savez_compressed(path,**masks)
    return row['key'],sha(path)


def final_count_one(job):
    row,directory,digest,nrecipes=job;path=Path(directory)/'finalists'/(row['key']+'.npz');checked(path,digest)
    checked(row['packet']['path'],row['packet']['sha256'])
    with np.load(row['packet']['path'],allow_pickle=False) as z:truth=np.unpackbits(z['truth']).astype(bool)
    with np.load(path,allow_pickle=False) as z:
        return {('recipe%d'%k):[int((np.unpackbits(z['recipe%d'%k]).astype(bool)&truth).sum()),int((np.unpackbits(z['recipe%d'%k]).astype(bool)|truth).sum())] for k in range(nrecipes)}


def finalize(a):
    started=time.monotonic();d=json.loads((a.out/'manifest.json').read_text());rows=d['rows'];arms=json.loads((a.out/'arms.json').read_text());L=len(arms)
    optimum=json.loads((a.out/'optimization.json').read_text());recipes=[]
    for value in optimum['pareto']:
        for recipe in value['recipes']:
            if not any((recipe['origin'],recipe['a'],recipe['b'])==(x['origin'],x['a'],x['b']) for x in recipes):recipes.append(recipe)
    if (a.out/'finalists_sealed.json').exists():
        seal=json.loads((a.out/'finalists_sealed.json').read_text())
        checked(a.out/'manifest.json',seal['manifest_sha256']);checked(a.out/'final_recipes.json',seal['recipes_sha256'])
        if seal['state']!='ALL_FINALIST_MASKS_SEALED' or seal['n']!=len(rows) or seal['recipes']!=len(recipes):raise ValueError('Cannot resume incomplete finalist seal')
        if json.loads((a.out/'final_recipes.json').read_text())['recipes']!=recipes:raise ValueError('Recipes changed')
        hashes=seal['predictions']
    else:
        (a.out/'finalists').mkdir(exist_ok=False)
        write(a.out/'final_recipes.json',dict(recipes=recipes,selection='all Pareto best-score ties at each nondominated cost; globally selected by exact score then cost',GT_selection='global DEV'))
        hashes={}
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            for key,digest in pool.map(final_mask_one,[(row,arms,recipes,str(a.out)) for row in rows],chunksize=4):hashes[key]=digest
        seal=dict(state='ALL_FINALIST_MASKS_SEALED',n=len(rows),recipes=len(recipes),predictions=hashes,manifest_sha256=sha(a.out/'manifest.json'),recipes_sha256=sha(a.out/'final_recipes.json'),query_GT_read_in_final_mask_construction=False,earlier_GT_used_for_global_DEV_selection=True)
        write(a.out/'finalists_sealed.json',seal)
    # Every actual finalist is sealed before re-opening the GT for final recount.
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        counts=list(pool.map(final_count_one,[(row,str(a.out),hashes[row['key']],len(recipes)) for row in rows],chunksize=4))
    arrays={'recipe%d'%k:np.array([x['recipe%d'%k] for x in counts],dtype=np.int64) for k in range(len(recipes))}
    patterns=np.load(a.out/'patterns.npy',mmap_mode='r');truth=patterns[:,:,1].sum(axis=1,dtype=np.int64)
    for k,recipe in enumerate(recipes):
        lut=recipe_lut(recipe,L)
        expected=np.stack((patterns[:,lut,1].sum(axis=1,dtype=np.int64),truth+patterns[:,lut,0].sum(axis=1,dtype=np.int64)),axis=1)
        if not np.array_equal(expected,arrays['recipe%d'%k]):raise ValueError('Actual finalist masks do not reproduce histogram I/U')
    with np.load(a.out/'baseline_counts.npz',allow_pickle=False) as z:
        for arm in arms:arrays[arm['id']]=z[arm['id']].copy()
    native_ids=[arm['id'] for arm in arms if arm['npz_key']=='native']
    if len(native_ids)!=1:raise ValueError('Exactly one complete native baseline required')
    arrays['native']=arrays[native_ids[0]]
    native_index=next(i for i,arm in enumerate(arms) if arm['id']==native_ids[0])
    native_on=(np.arange(1<<L)&(1<<native_index))!=0;corrections={}
    for k,recipe in enumerate(recipes):
        name='recipe%d'%k;lut=recipe_lut(recipe,L);add=lut&~native_on;delete=~lut&native_on
        edit={key:patterns[:,selection,truth_bit].sum(axis=1,dtype=np.int64) for key,selection,truth_bit in [('add_TP',add,1),('add_FP',add,0),('delete_TP',delete,1),('delete_FP',delete,0)]}
        if not np.array_equal(arrays[name][:,0]-arrays['native'][:,0],edit['add_TP']-edit['delete_TP']):raise ValueError('Intersection edit accounting mismatch')
        if not np.array_equal(arrays[name][:,1]-arrays['native'][:,1],edit['add_FP']-edit['delete_FP']):raise ValueError('Union edit accounting mismatch')
        corrections[name]=[dict(key=row['key'],c=row['c'],fold=row['fold'],batch=row['batch'],**{key:int(value[j]) for key,value in edit.items()}) for j,row in enumerate(rows)]
    report,draws=summarize(rows,arrays,corrections)
    report['batches']=report.pop('batchs')
    report['class_macro_edit_attribution']=edit_attribution(rows,arrays,corrections)
    cls=np.array([r['c'] for r in rows]);groups=photo_groups(rows);g=int(groups.max())+1
    weights=np.stack([np.bincount(draw,minlength=g) for draw in draws])[:,groups]
    selected=[]
    for k,recipe in enumerate(recipes):
        name='recipe%d'%k;point=report['scores'][name]
        if abs(point-recipe['score'])>1e-10:raise ValueError('Final score differs from exact search')
        if recipe['score']==optimum['maximum_miou'] and recipe['cost']==optimum['minimum_cost_at_maximum']:selected.append(name)
        boot=np.array([metric(arrays[name],cls,w) for w in weights]);contrasts={}
        for arm in arms:
            base=arm['id'];samples=np.array([metric(arrays[base],cls,w) for w in weights]);delta=arrays[name][:,0]/np.maximum(arrays[name][:,1],1)-arrays[base][:,0]/np.maximum(arrays[base][:,1],1)
            contrasts[base]=dict(gain=point-report['scores'][base],ci95=np.percentile(boot-samples,[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()))
        report['contrasts'][name]=contrasts
    report.update(selected_global_DEV_recipes=selected,recipes=recipes,optimization=optimum,all_actual_masks_IU_exact=True,finalists_seal_sha256=sha(a.out/'finalists_sealed.json'),selection_warning='Global GT-selected DEV optimum; paired CIs are conditional and NOT selection-adjusted. No unseen-test or SOTA claim.',CPU_finalize_seconds=time.monotonic()-started)
    np.savez_compressed(a.out/'final_counts.npz',**arrays);write(a.out/'report.json',report)
    lines=['# Exact existing A/B family: global DEV optimum','',f"All {optimum['recipes_evaluated']:,} recipes were evaluated. Maximum class-summed mIoU **{optimum['maximum_miou']:.6f}**, outer complete-mask producer count proxy **{optimum['minimum_cost_at_maximum']}**.",'','GT selected this single global recipe; inference uses no query GT. All actual finalist masks sealed before final GT recount; every per-draw I/U matches membership-histogram predictions exactly. Paired confidence intervals are conditional on the selected recipe, not selection-adjusted. This is benchmark DEV reuse, not untouched-test/SOTA.','']
    for name in selected:
        recipe=recipes[int(name.replace('recipe',''))];lines += [f"Selected {name}: P0=`{recipe['origin_method']}`; additions={recipe['add_methods']}; deletions={recipe['delete_methods']}.",'','| Baseline | Gain [95% CI], pp |','|---|---:|']
        for base,result in report['contrasts'][name].items():lines.append(f"| {base} | {result['gain']:+.6f} [{result['ci95'][0]:+.6f}, {result['ci95'][1]:+.6f}] |")
    lines += ['','## Full exact gain-cost Pareto','','| Outer producer-count proxy | mIoU |','|---|---:|']
    lines += [f"| {x['cost']} | {x['score']:.6f} |" for x in optimum['pareto']]
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');write(a.out/'state.json',dict(state='FINAL_FIXED_RECIPES_RECOUNT_COMPLETE',maximum_miou=optimum['maximum_miou'],selected=selected,n=len(rows),CPU_finalize_seconds=time.monotonic()-started))
    print('FINAL_FIXED_RECIPES_RECOUNT_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('hist','optimize','finalize','all'))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy',choices=('strict','extended'),default='strict');p.add_argument('--workers',type=int,default=4);p.add_argument('--fixed-origin-id')
    a=p.parse_args()
    if a.stage in ('hist','all'):hist(a)
    if a.stage in ('optimize','all'):optimize(a)
    if a.stage in ('finalize','all'):finalize(a)


if __name__=='__main__':main()
