#!/usr/bin/env python3
"""Train-three/hold-one recipe selection from a sealed dense mask library."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from run_exact_family_selection import (
    checked, edit_attribution, final_count_one, final_mask_one, metric, optimize_origin,
    photo_groups, recipe_lut, sha, summarize, write,
)


def select(histogram, arms, path, origins, workers):
    np.save(path, histogram)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(optimize_origin,[(str(path),p) for p in origins]))
    maximum=max(v['score'] for r in results for v in r['best_by_cost'].values())
    cost=min(int(k) for r in results for k,v in r['best_by_cost'].items() if v['score']==maximum)
    ties=[v for r in results for k,entry in r['best_by_cost'].items() if int(k)==cost and entry['score']==maximum for v in entry['recipes']]
    ties.sort(key=lambda r:(r['origin'],r['a'],r['b']))
    def named(r):
        p=r['origin'];others=[i for i in range(len(arms)) if i!=p]
        return dict(r,origin_method=arms[p]['id'],add_methods=[arms[i]['id'] for j,i in enumerate(others) if r['a']&(1<<j)],delete_methods=[arms[i]['id'] for j,i in enumerate(others) if r['b']&(1<<j)])
    return dict(selected=named(ties[0]),all_exact_score_and_cost_ties=[named(r) for r in ties],maximum_miou=maximum,minimum_outer_producer_cost=cost,recipes_evaluated=len(origins)*(1<<(2*(len(arms)-1))),tie_rule='exact float64 score; then minimum outer producer proxy; then lexical origin/a/b among ties')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--fixed-origin-id')
    a=p.parse_args();started=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    complete=json.loads((a.source/'histogram_complete.json').read_text())
    if complete['state']!='MEMBERSHIP_HISTOGRAM_COMPLETE':raise ValueError('Complete sealed-library histogram required')
    manifest=json.loads((a.source/'manifest.json').read_text());rows=manifest['rows']
    arms=json.loads((a.source/'arms.json').read_text());L=len(arms)
    patterns=np.load(a.source/'patterns.npy',mmap_mode='r')
    if patterns.shape!=(len(rows),1<<L,2):raise ValueError('Membership shape mismatch')
    a.out.mkdir(parents=True);write(a.out/'manifest.json',manifest);write(a.out/'arms.json',arms)
    cls=np.array([r['c'] for r in rows]);folds=np.array([r['fold'] for r in rows]);classes=np.unique(cls)
    if len(np.unique(folds))!=4:raise ValueError('Train-three/held-one requires four folds')
    origins=list(range(L)) if a.fixed_origin_id is None else [next(i for i,x in enumerate(arms) if x['id']==a.fixed_origin_id)]
    selections={};train_recipe=[]
    for fold in np.unique(folds):
        train=folds!=fold;present=np.unique(cls[train])
        h=np.stack([patterns[train&(cls==c)].sum(axis=0,dtype=np.int64) for c in present])
        result=select(h,arms,a.out/('train_fold%d_patterns.npy'%fold),origins,a.workers)
        result.update(training_n=int(train.sum()),held_n=int((~train).sum()),training_classes=present.tolist(),held_classes=np.unique(cls[~train]).tolist())
        selections[str(fold)]=result;train_recipe.append((int(fold),result['selected']))
        write(a.out/'selection.json',dict(folds=selections,fixed_origin=a.fixed_origin_id,library_sources=L,full_original_library_scope=manifest.get('selection_scope',manifest.get('search_scope',{})),full_library_distinct_count=manifest.get('full_library_distinct_count',L),full187_library_exhaustive_search_claim=manifest.get('full187_library_exhaustive_search_claim',False)))
        state=dict(state='CROSSFOLD_RECIPE_SELECTION',completed_folds=len(selections),n=len(rows),library_sources=L,seconds=time.monotonic()-started)
        write(a.out/'state.json',state);print(json.dumps(dict(state,heldfold=int(fold),selected=result['selected'])),flush=True)
    # This operation uses only selected fixed recipes and complete source masks.
    (a.out/'finalists').mkdir();hashes={}
    jobs=[(row,arms,[selections[str(row['fold'])]['selected']],str(a.out)) for row in rows]
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for key,digest in pool.map(final_mask_one,jobs,chunksize=4):hashes[key]=digest
    seal=dict(state='ALL_HELDFOLD_MASKS_SEALED',n=len(rows),predictions=hashes,manifest_sha256=sha(a.out/'manifest.json'),selection_sha256=sha(a.out/'selection.json'),query_GT_read_in_mask_construction=False,GT_used_for_recipe_selection='three other folds only',source_producer_GT_exposure='retained from source manifest; all600-fitted extended sizecut is NOT label-free heldfold')
    write(a.out/'sealed.json',seal)
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        counts=list(pool.map(final_count_one,[(row,str(a.out),hashes[row['key']],1) for row in rows],chunksize=4))
    selected_iu=np.array([x['recipe0'] for x in counts],dtype=np.int64)
    expected=np.empty_like(selected_iu);edits={key:np.zeros(len(rows),dtype=np.int64) for key in ['add_TP','add_FP','delete_TP','delete_FP']}
    native_index=next(i for i,arm in enumerate(arms) if arm['npz_key']=='native');native_on=(np.arange(1<<L)&(1<<native_index))!=0
    for fold,recipe in train_recipe:
        ix=folds==fold;h=patterns[ix];lut=recipe_lut(recipe,L);truth=h[:,:,1].sum(axis=1,dtype=np.int64)
        expected[ix]=np.stack([h[:,lut,1].sum(axis=1,dtype=np.int64),truth+h[:,lut,0].sum(axis=1,dtype=np.int64)],axis=1)
        for key,selection,t in [('add_TP',lut&~native_on,1),('add_FP',lut&~native_on,0),('delete_TP',~lut&native_on,1),('delete_FP',~lut&native_on,0)]:edits[key][ix]=h[:,selection,t].sum(axis=1,dtype=np.int64)
    if not np.array_equal(selected_iu,expected):raise ValueError('Actual heldfold masks differ from membership I/U')
    arrays={'heldfold_recipe':selected_iu}
    with np.load(a.source/'baseline_counts.npz',allow_pickle=False) as z:
        arrays.update({arm['id']:z[arm['id']].copy() for arm in arms})
    arrays['native']=arrays[arms[native_index]['id']]
    if not np.array_equal(selected_iu[:,0]-arrays['native'][:,0],edits['add_TP']-edits['delete_TP']):raise ValueError('I edit identity failed')
    if not np.array_equal(selected_iu[:,1]-arrays['native'][:,1],edits['add_FP']-edits['delete_FP']):raise ValueError('U edit identity failed')
    corrections={'heldfold_recipe':[dict(key=row['key'],c=row['c'],fold=row['fold'],batch=row['batch'],**{key:int(value[i]) for key,value in edits.items()}) for i,row in enumerate(rows)]}
    report,draws=summarize(rows,arrays,corrections);report['batches']=report.pop('batchs')
    report['class_macro_edit_attribution']=edit_attribution(rows,arrays,corrections)
    groups=photo_groups(rows);g=int(groups.max())+1;weights=np.stack([np.bincount(d,minlength=g) for d in draws])[:,groups]
    boot={name:np.array([metric(value,cls,w) for w in weights]) for name,value in arrays.items()}
    for arm in arms:
        name=arm['id'];delta=selected_iu[:,0]/np.maximum(selected_iu[:,1],1)-arrays[name][:,0]/np.maximum(arrays[name][:,1],1)
        report['contrasts']['heldfold_recipe'][name]=dict(gain=report['scores']['heldfold_recipe']-report['scores'][name],ci95=np.percentile(boot['heldfold_recipe']-boot[name],[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((abs(delta)<=1e-12).sum()))
    report.update(selection=selections,fixed_origin=a.fixed_origin_id,library_sources=L,library_selection_scope=manifest.get('selection_scope',manifest.get('search_scope',{})),full_library_distinct_count=manifest.get('full_library_distinct_count',L),full187_library_exhaustive_search_claim=manifest.get('full187_library_exhaustive_search_claim',False),all_actual_masks_IU_exact=True,edit_identities_exact=True,source_manifest_sha256=sha(a.source/'manifest.json'),selection_warning='Fold-held recipe selection only: source-producer parameter/label exposure remains. Conditional connected-photo CIs do not repeat recipe selection. Reused benchmark development, not fresh confirmation.',CPU_seconds=time.monotonic()-started)
    write(a.out/'report.json',report);np.savez_compressed(a.out/'counts.npz',**arrays)
    lines=['# Fold-held existing A/B family selection','',f"Class-summed mIoU: {report['scores']['heldfold_recipe']:.6f}.",'','Each fold uses one fixed recipe chosen from the other three folds. Source-producer GT exposure is unchanged; supplied all600-fitted sizecut remains label-fitted across held folds. Conditional CIs are not selection-adjusted and this benchmark is development reuse.','', '| Baseline | Gain [95% CI], pp |','|---|---:|']
    for name,value in report['contrasts']['heldfold_recipe'].items():lines.append(f"| {name} | {value['gain']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] |")
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.out/'state.json',dict(state='HELD_FOLD_FIXED_RECIPES_RECOUNT_COMPLETE',n=len(rows),miou=report['scores']['heldfold_recipe'],CPU_seconds=time.monotonic()-started))
    print('HELD_FOLD_FIXED_RECIPES_RECOUNT_COMPLETE',flush=True)


if __name__=='__main__':main()
