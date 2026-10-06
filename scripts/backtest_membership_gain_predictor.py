#!/usr/bin/env python3
"""One Beta(1,1) mask-membership forecast benchmark; labelled research only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ics.experiment import photo_groups,sha,metric


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity(r):return (int(r['fold']),int(r['e']),int(r['c']),Path(r['query']).name,Path(r['support']).name)


def frequencies(job):
    row,ids,extra=job;state=np.zeros(1024**2,dtype=np.uint16);cache={}
    for j,name in enumerate(ids):
        rec=row['masks'][name];path=rec['path']
        if path not in cache:
            if sha(path)!=rec['sha256']:raise ValueError('Source mask file changed')
            with np.load(path,allow_pickle=False) as z:cache[path]={key:z[key].copy() for key in {r['key'] for r in row['masks'].values() if r['path']==path}}
        a=cache[path][rec['key']]
        if a.dtype!=np.uint8 or a.shape!=(131072,) or hashlib.sha256(a.tobytes()).hexdigest()!=rec['packed_array_sha256']:raise ValueError('Invalid source packed mask')
        state|=np.unpackbits(a).astype(np.uint16)<<j
    n=1<<len(ids);total=np.bincount(state,minlength=n).astype(np.uint32);on=None
    if extra:
        path,expected,key=extra
        if sha(path)!=expected:raise ValueError('Extra candidate mask changed')
        with np.load(path,allow_pickle=False) as z:a=z[key].copy()
        if a.dtype!=np.uint8 or a.shape!=(131072,):raise ValueError('Invalid extra packed mask')
        on=np.bincount(state[np.unpackbits(a).astype(bool)],minlength=n).astype(np.uint32)
    return total,on


def rates(iu,classes):
    ids=np.unique(classes);values={str(c):iu[classes==c].sum(axis=0).tolist() for c in ids}
    return float(100*np.mean([i/max(u,1) for i,u in values.values()])),values


def project(counts,on,eta):
    return np.stack([(on*eta[None,:]).sum(1),(counts*eta[None,:]).sum(1)+(on*(1-eta)[None,:]).sum(1)],axis=1)


def rank(scores):return sorted(scores,key=lambda name:(-scores[name],name))


def grade(pred,actual,base):
    names=list(pred);order=rank(pred);real=rank(actual);comparisons={}
    for name in names:
        gain=pred[name]-pred[base];truth=actual[name]-actual[base]
        comparisons[name]=dict(predicted_gain_pp=gain,actual_gain_pp=truth,sign_correct=bool(np.sign(gain)==np.sign(truth)),score_error_pp=pred[name]-actual[name])
    pairs=[bool(np.sign(pred[a]-pred[b])==np.sign(actual[a]-actual[b])) for j,a in enumerate(names) for b in names[j+1:]]
    return dict(predicted_rank=order,actual_rank=real,pairwise_rank_accuracy=float(np.mean(pairs)),predicted_best=order[0],actual_best=real[0],chosen_actual_regret_pp=actual[real[0]]-actual[order[0]],comparisons_vs_base=comparisons,mean_absolute_score_error_pp=float(np.mean([abs(pred[x]-actual[x]) for x in names])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-run',type=Path);p.add_argument('--extra-arm',default='fine.mean16.control');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--frozen-eta-root',type=Path);p.add_argument('--predict-only',action='store_true')
    a=p.parse_args();started=time.monotonic()
    if not 1<=a.workers<=2:raise ValueError('At most two CPU workers')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    manifest=json.loads((a.source/'manifest.json').read_text());rows=manifest['rows'];arms=json.loads((a.source/'arms.json').read_text());ids=[x['id'] for x in arms];L=len(ids)
    if L!=12 or len(rows)!=4000:raise ValueError('Frozen strict12 PUBLIC4000 required')
    folds=np.array([r['fold'] for r in rows]);cls=np.array([r['c'] for r in rows]);groups=photo_groups(rows)
    hist=np.load(a.source/'patterns.npy',mmap_mode='r');models={};training={}
    for fold in range(4):
        held=folds==fold;held_groups=np.unique(groups[held]);train=(folds!=fold)&~np.isin(groups,held_groups)
        if a.frozen_eta_root:
            receipt=json.loads((a.frozen_eta_root/('fold%d'%fold)/'training_receipt.json').read_text())
            frozen=json.loads((a.frozen_eta_root/'eta_frozen.json').read_text())
            src=a.frozen_eta_root/('fold%d'%fold)/'eta.npy'
            if sha(src)!=frozen['fold_eta_sha256'][str(fold)] or sha(src)!=receipt['eta_sha256'] or receipt['source_manifest_sha256']!=sha(a.source/'manifest.json'):raise ValueError('Frozen eta/source changed')
            if receipt['train_draw_keys']!=[r['key'] for i,r in enumerate(rows) if train[i]]:raise ValueError('Frozen photo-disjoint training split changed')
            path=a.out/('fold%d'%fold);path.mkdir();eta=np.load(src,allow_pickle=False);np.save(path/'eta.npy',eta)
            write(path/'training_receipt.json',receipt);models[fold]=eta;training[fold]=receipt
            continue
        h=hist[train].sum(axis=0,dtype=np.int64);N=h.sum(axis=1);fg=float(h[:,1].sum()/h.sum());eta=(h[:,1]+1)/(N+2);eta[N==0]=fg
        path=a.out/('fold%d'%fold);path.mkdir();np.save(path/'eta.npy',eta)
        train_photos={Path(r[k]).name for i,r in enumerate(rows) if train[i] for k in ('query','support')};held_photos={Path(r[k]).name for i,r in enumerate(rows) if held[i] for k in ('query','support')}
        if train_photos&held_photos:raise ValueError('Train/held photograph overlap')
        receipt=dict(train_draw_keys=[r['key'] for i,r in enumerate(rows) if train[i]],train_n=int(train.sum()),held_n=int(held.sum()),train_classes=np.unique(cls[train]).tolist(),held_classes=np.unique(cls[held]).tolist(),photo_overlap=0,connected_photo_groups_excluded=True,seen_patterns=int((N>0).sum()),unseen_pattern_fallback=fg,Beta=[1,1],eta_sha256=sha(path/'eta.npy'),source_manifest_sha256=sha(a.source/'manifest.json'),source_membership_GT_hist_sha256=sha(a.source/'patterns.npy'),source_arm_sequence_ids=[x['producer_id'] for x in arms],held_GT_values_inspected_for_training=False,resource='train query-GT calibration for retrospective research; not a training-free single-reference segmentation method')
        write(path/'training_receipt.json',receipt);models[fold]=eta;training[fold]=receipt
    frozen_record=json.loads((a.frozen_eta_root/'eta_frozen.json').read_text()) if a.frozen_eta_root else dict(state='TRAIN_ONLY_ETA_FROZEN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='one Beta(1,1) per12-mask membership, unseen pattern global train foreground rate; no bins, tuning or network',source_code_sha256=sha(__file__),fold_eta_sha256={str(f):training[f]['eta_sha256'] for f in range(4)},held_GT_inspected=False,main_segmentation_recipe_changed=False)
    write(a.out/'eta_frozen.json',frozen_record)
    extras={};extra_rows=[];extra_score_preexisting=False;extra_receipt={}
    if a.extra_run:
        seal=json.loads((a.extra_run/'sealed.json').read_text());extra_rows=json.loads((a.extra_run/'manifest.json').read_text());extra_rows=extra_rows if isinstance(extra_rows,list) else extra_rows['episodes']
        if not str(seal['state']).startswith('ALL_') or seal['n']!=len(extra_rows):raise ValueError('Extra arm must be complete and sealed')
        if sha(a.extra_run/'manifest.json')!=seal['manifest_sha256']:raise ValueError('Extra manifest changed')
        config=json.loads((a.extra_run/'config.json').read_text())
        if 'config_sha256' in seal and sha(a.extra_run/'config.json')!=seal['config_sha256']:raise ValueError('Extra producer config changed')
        for r in extra_rows:extras[identity(r)]=(str(a.extra_run/'predictions'/(r['key']+'.npz')),seal['predictions'][r['key']],a.extra_arm)
        if not set(extras).issubset({identity(r) for r in rows}):raise ValueError('Extra candidate is not the paired cohort')
        extra_score_preexisting=(a.extra_run/'report.json').exists() or (a.extra_run/'score_state.json').exists()
        extra_receipt=dict(path=str(a.extra_run),seal_sha256=sha(a.extra_run/'sealed.json'),config_sha256=sha(a.extra_run/'config.json'),source_manifest_sha256=sha(a.extra_run/'manifest.json'),arm=a.extra_arm,n=len(extra_rows),parameters=config.get('parameters',config.get('saved_source_parameters')),actual_score_artifact_existed_before_prediction=extra_score_preexisting,interpretation='retrospective' if extra_score_preexisting else 'prediction before observing actual score; automatic scoring was not altered',producer_label_exposure='retained from producer config; supplied uniform-tau15 frozen sizecut uses allfresh600 labels')
    counts=np.lib.format.open_memmap(a.out/'membership_totals.npy',mode='w+',dtype=np.uint32,shape=(4000,1<<L));extra_on=np.zeros_like(counts);extra_ix=np.array([identity(r) in extras for r in rows])
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for i,(total,on) in enumerate(pool.map(frequencies,[(r,ids,extras.get(identity(r))) for r in rows],chunksize=4)):
            counts[i]=total
            if on is not None:extra_on[i]=on
            if (i+1)%400==0:print('UNLABELLED_MASK_MEMBERSHIP',i+1,flush=True)
    counts.flush();np.save(a.out/'extra_mask_membership.npy',extra_on)
    patterns=np.arange(1<<L);luts={name:(patterns&(1<<j))!=0 for j,name in enumerate(ids)}
    v=ids.index('frozen_subtoken4000_v1::fine.rcg16.control');f=ids.index('frozen_subtoken4000_v1::fine.rcg64');g=ids.index('mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1')
    C='strict.family12.frozen';V=ids[v];G=ids[g];M='frozen_subtoken4000_v1::mean.control';luts[C]=np.where((patterns&(1<<v))!=0,(patterns&(1<<g))!=0,(patterns&(1<<f))!=0)
    forecasts={};expected_all={name:np.zeros((4000,2)) for name in luts}
    for fold in range(4):
        held=folds==fold;eta=models[fold];cohorts={}
        for name,lut in luts.items():expected_all[name][held]=project(counts[held],counts[held]*lut[None,:],eta)
        for cohort,ix in [('PUBLIC4000',held),('extra_paired_subset',held&extra_ix)]:
            if not ix.any():continue
            values={name:rates(iu[ix],cls[ix]) for name,iu in expected_all.items()};expected={name:value[0] for name,value in values.items()};class_IU={name:value[1] for name,value in values.items()}
            if cohort=='extra_paired_subset':
                iu=project(counts[ix],extra_on[ix],eta);score,detail=rates(iu,cls[ix]);expected[a.extra_arm]=score;class_IU[a.extra_arm]=detail
            margins={}
            for name in expected:
                parts=[]
                for c in class_IU[name]:
                    I0,U0=class_IU[V][c];I1,U1=class_IU[name][c];J0=I0/max(U0,1);parts.append((I1-I0-J0*(U1-U0))/max(U1,1)*100)
                margins[name]=dict(expected_gain_vs_fine16_pp=float(np.mean(parts)),identity='sum added pixels (p-J*(1-p)) minus sum deleted pixels (p-J*(1-p)), divided by new class U; J is predicted baseline IoU, never held GT J')
            cohorts[cohort]=dict(n=int(ix.sum()),scores=expected,class_E_I_U=class_IU,rank=rank(expected),marginal_net_gain=margins)
        if a.extra_run:
            extra_receipt['actual_score_artifact_exists_at_prediction']=(a.extra_run/'report.json').exists() or (a.extra_run/'score_state.json').exists()
            if extra_receipt['actual_score_artifact_exists_at_prediction']:extra_receipt['interpretation']='retrospective'
        prediction=dict(state='PREDICTION_SAVED_BEFORE_OWN_HELD_GT_EVALUATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fold=fold,cohorts=cohorts,training=training[fold],extra_source=dict(extra_receipt),held_source_manifest_sha256=sha(a.source/'manifest.json'),held_unlabelled_membership_source='recomputed from sealed packed predicted masks; no held GT packet or held labelled histogram totals read',held_GT_inspected_for_prediction=False,all4000_previously_exposed=True,global_C12_was_selected_using_full4000_GT_before_this_diagnostic=True)
        write(a.out/('fold%d'%fold)/'prediction.json',prediction);forecasts[fold]=prediction
    if a.predict_only:
        write(a.out/'prediction_complete.json',dict(state='ALL_FOUR_FOLD_PREDICTIONS_FROZEN_NO_HELD_GT_EVALUATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),extra_source=extra_receipt,source_frozen_eta_sha256=sha(a.out/'eta_frozen.json'),held_GT_inspected=False))
        print('ALL_FOUR_FOLD_PREDICTIONS_FROZEN_NO_HELD_GT_EVALUATION',flush=True);return
    # Predictions for every fold are now durable; only now inspect held GT statistics.
    actual={name:np.zeros((4000,2),dtype=np.int64) for name in luts}
    truth=hist[:,:,1].sum(1,dtype=np.int64)
    for name,lut in luts.items():actual[name]=np.stack([hist[:,lut,1].sum(1,dtype=np.int64),truth+hist[:,lut,0].sum(1,dtype=np.int64)],axis=1)
    extra_actual={}
    if a.extra_run:
        records=[json.loads(line) for line in (a.extra_run/'episodes.jsonl').read_text().splitlines()];extra_actual={identity(r):r['iu'][a.extra_arm] for r in records}
        if set(extra_actual)!=set(extras):raise ValueError('Extra scored cohort differs from sealed candidate')
    evaluations={}
    for fold,pred in forecasts.items():
        held=folds==fold;result={}
        for cohort,forecast in pred['cohorts'].items():
            ix=held if cohort=='PUBLIC4000' else held&extra_ix;observed={name:metric(iu[ix],cls[ix]) for name,iu in actual.items()}
            if cohort=='extra_paired_subset':observed[a.extra_arm]=metric(np.array([extra_actual[identity(r)] for i,r in enumerate(rows) if ix[i]]),cls[ix])
            result[cohort]=grade(forecast['scores'],observed,V);result[cohort]['actual_scores']=observed
            result[cohort]['C12_vs_graft']=dict(predicted_gain=forecast['scores'][C]-forecast['scores'][G],actual_gain=observed[C]-observed[G])
            if cohort=='extra_paired_subset':result[cohort]['extra_vs_graft']=dict(predicted_gain=forecast['scores'][a.extra_arm]-forecast['scores'][G],actual_gain=observed[a.extra_arm]-observed[G])
        evaluations[str(fold)]=result;write(a.out/('fold%d'%fold)/'evaluation.json',result)
    summary=dict(state='ONE_FIXED_MEMBERSHIP_PREDICTOR_BACKTEST_COMPLETE',folds=evaluations,training_n_by_fold={str(f):r['train_n'] for f,r in training.items()},extra_source=extra_receipt,limits='Retrospective class-held and connected-photo-disjoint training; labelled calibration, old4k exposure and globally selected C12 remain. This is a fallible gain forecast, not a new segmentation component or zero-training deployable posterior.',seconds=time.monotonic()-started)
    write(a.out/'report.json',summary);print(json.dumps(dict(state=summary['state'],seconds=summary['seconds'],fold_rank_accuracy={f:v['PUBLIC4000']['pairwise_rank_accuracy'] for f,v in evaluations.items()})),flush=True)


if __name__=='__main__':main()
