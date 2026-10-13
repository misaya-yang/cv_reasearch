#!/usr/bin/env python3
"""Score sealed 2380 supported-MEAN masks with independent same-K lexsort replay."""
from __future__ import annotations
import json,os,sys,time
from collections import Counter
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k,'2')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
DATA=REPO.parent/'cv_data'
ROOT=REPO/'evidence/local/insid3_foris_generalization_20261010'
OUT=DATA/'a/consensus_supported_foreground2380_v2_20261011'
DUAL=DATA/'a/consensus_dual_frame1020_v1_20261011'
LVIS=DATA/'a/consensus_lvis1400_dual_frame_v1_20261011'
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
import sol_supported_foreground2380_v2 as producer
from sol_score_transfer_consensus120 import official
from sol_score_bg_admission900 import connected_intervals,stats,edits
from ics.metrics import summarize
from ics.official_data import array_hash

def read(p):return json.loads(Path(p).read_text())
def sha(p):return producer.sha(p)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def unpack(a,shape):return np.unpackbits(a,count=shape[0]*shape[1]).reshape(shape).astype(bool)
def double_bilinear(a,shape):
    return F.interpolate(torch.from_numpy(np.asarray(a,dtype=np.float64))[None,None],shape,mode='bilinear',align_corners=False)[0,0].numpy()
def single_bilinear(a,shape):
    return F.interpolate(torch.from_numpy(np.asarray(a,dtype=np.float32))[None,None],shape,mode='bilinear',align_corners=False)[0,0].numpy()
def lexsort_control(U,mean64,K):
    shape=U.shape;ids=np.flatnonzero(U.reshape(-1))
    result=np.zeros(shape,bool)
    if K==0:return result
    margin=np.abs(np.float32(2)*single_bilinear(mean64,shape)-np.float32(1)).astype(np.float32).reshape(-1)
    # Independent from producer's stable argsort: primary -margin, secondary flat index.
    order=np.lexsort((ids,-margin[ids].astype(np.float64)))
    result.reshape(-1)[ids[order[:K]]]=True
    return result
def nearest_truth(row,shape):
    with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert array_hash(raw)==row['query_mask_hash']
    y=np.minimum(np.arange(shape[0])*raw.shape[0]//shape[0],raw.shape[0]-1)
    x=np.minimum(np.arange(shape[1])*raw.shape[1]//shape[1],raw.shape[1]-1)
    result=raw[np.ix_(y,x)].astype(bool)
    native=F.interpolate(torch.from_numpy(raw.astype(np.float64))[None,None],shape,mode='nearest')[0,0].numpy()>0
    assert np.array_equal(result,native)
    return result

def main():
    torch.set_num_threads(2);tick=time.monotonic()
    seal,cfg=read(OUT/'sealed.json'),read(OUT/'config.json')
    assert seal['state']=='ALL9520_SUPPORTED_FOREGROUND_MASKS_SEALED' and seal['n']==2380
    assert seal['query_gt_access_attempts']==[] and seal['new_complete_masks']==9520
    for name in ('config','manifest','expected_classes'):assert sha(OUT/(name+'.json'))==seal[name+'_sha256']
    assert sha(REPO/'scripts/sol_supported_foreground2380_v2.py')==cfg['source_sha256']
    assert sha(ROOT/'consensus_supported_foreground_protocol.json')==cfg['protocol_sha256']
    sources,expected=read(OUT/'manifest.json'),read(OUT/'expected_classes.json')
    assert len(sources)==len(seal['records'])==2380
    old={}
    for base in (DUAL,LVIS):
        p=base/'report.json';rep=read(p)
        assert sha(base/'scored_episodes.json')==rep['scored_records_sha256']
        by_frame=read(base/'scored_episodes.json')
        for frame in producer.FRAMES:
            for r in by_frame[frame]:old[(r['episode_id'],frame)]=r
    # LVIS parent overwrites overlapping40 because the new 2380 cohort uses its exact LVIS1400 output.
    rows={frame:[] for frame in producer.FRAMES};checks=0;recon=Counter();affected=Counter()
    extra_names=('region.fast','task.competitive','task.matched','rcg','rcg.fine')
    for i,(s,rec) in enumerate(zip(sources,seal['records'])):
        row=s['row'];eid=row['episode_id'];assert eid==rec['episode_id']
        pp=OUT/'predictions'/rec['file'];assert sha(pp)==rec['sha256']
        assert sha(s['parent_prediction'])==s['parent_prediction_sha256']
        assert sha(s['mean64_path'])==s['mean64_sha256']
        assert sha(s['baseline'])==s['baseline_sha256'] and sha(s['mean'])==s['mean_sha256']
        assert sha(s['evidence_path'])==s['evidence_sha256']
        with np.load(s['mean64_path'],allow_pickle=False) as z:mean64=z['mean'].copy()
        assert mean64.dtype==np.float32 and mean64.shape==(64,64)
        with np.load(s['evidence_path'],allow_pickle=False) as z:e=z[s['evidence_key']].copy()
        if s['kind']=='dual1020':
            cal=s['calibration'];scale=float(cal['separation'])
            h=np.clip((e-float(cal['threshold']))/scale,-1,1) if np.isfinite(scale) and scale>np.finfo(float).eps else np.zeros_like(e)
        else:
            h=e
        cli_mean=single_bilinear(mean64,(1024,1024))>.5
        with np.load(pp,allow_pickle=False) as pred,np.load(s['parent_prediction'],allow_pickle=False) as parent,np.load(s['baseline'],allow_pickle=False) as baseline,np.load(s['mean'],allow_pickle=False) as mean:
            for frame in producer.FRAMES:
                shape=(1024,1024) if frame=='cli' else tuple(row['query_size_hw'])
                B=unpack(baseline[s['baseline_keys'][frame]],shape)
                G=unpack(mean[s['mean_keys'][frame]],shape)
                R=unpack(parent[frame+'/anchor.global'],shape)
                C=unpack(parent[frame+'/shared_edits_complete'],shape)
                old_control=unpack(parent[frame+'/same_edit_budget_source_order'],shape)
                assert np.array_equal(G,cli_mean if frame=='cli' else single_bilinear(cli_mean.astype(np.float32),shape)>.5)
                assert np.array_equal(C,(B&R)|(B&G)|(R&G))
                H=double_bilinear(h,shape);U=(~B)&(~R)&G;A=U&(H>0);K=int(A.sum())
                N=C|A;control=C|lexsort_control(U,mean64,K)
                saved_N=unpack(pred[frame+'/supported_mean'],shape)
                saved_control=unpack(pred[frame+'/own_margin_sameK'],shape)
                for name,actual,expected_mask in (('supported_mean',saved_N,N),('own_margin_sameK',saved_control,control)):
                    diff=int(np.count_nonzero(actual!=expected_mask));recon[frame+'/'+name]+=diff
                    assert diff==0,(eid,frame,name,diff);checks+=1
                assert K==rec['frames'][frame]['K'] and int(U.sum())==rec['frames'][frame]['U']
                assert int((saved_control&~C).sum())==K and not np.any(C&~saved_N)
                assert not np.any((saved_N^C)&B) and np.array_equal(saved_N&~C,A)
                assert np.array_equal(saved_N&B,C&B) and np.array_equal(saved_control&B,C&B)
                if s.get('origin')=='SOURCE_ROLE_DEGENERATE_ZERO_EVIDENCE':
                    assert K==0 and np.array_equal(N,C) and np.array_equal(C,B)
                checks+=6
                gt=nearest_truth(row,shape)
                masks={'foris.crf':B,'mean':G,'anchor.global':R,'shared_edits_complete':C,
                       'same_edit_budget_source_order':old_control,'supported_mean':saved_N,'own_margin_sameK':saved_control}
                prefix=s['baseline_keys'][frame].split('/',1)[0]
                for key in extra_names:
                    full_key=prefix+'/'+key
                    if full_key in baseline.files:
                        masks['task.competitive' if key=='task.matched' else key]=unpack(baseline[full_key],shape)
                metrics={arm:stats(a,gt) for arm,a in masks.items()}
                previous=old[(eid,frame)]
                for arm in ('foris.crf','mean','anchor.global','shared_edits_complete','same_edit_budget_source_order'):
                    assert [metrics[arm][k] for k in ('I','U','FP','FN')]==[previous['metrics'][arm][k] for k in ('I','U','FP','FN')],(eid,frame,arm)
                    checks+=1
                for arm in masks:
                    tab=np.bincount((gt.astype(np.uint8)+2*masks[arm].astype(np.uint8)).ravel(),minlength=4)
                    assert [int(tab[3]),int(tab[1:].sum()),int(tab[2]),int(tab[1])]==[metrics[arm][k] for k in ('I','U','FP','FN')]
                    checks+=1
                at,af,dt,df=edits(saved_N,C,gt)
                assert dt==df==0 and at+af==K
                assert metrics['supported_mean']['I']==metrics['shared_edits_complete']['I']+at
                assert metrics['supported_mean']['U']==metrics['shared_edits_complete']['U']+af
                ct,cf,cdt,cdf=edits(saved_control,C,gt)
                assert cdt==cdf==0 and ct+cf==K
                assert metrics['own_margin_sameK']['I']==metrics['shared_edits_complete']['I']+ct
                assert metrics['own_margin_sameK']['U']==metrics['shared_edits_complete']['U']+cf
                checks+=7
                affected[(row['dataset'],frame)]+=int(K>0)
                rows[frame].append(dict(episode_id=eid,frame=frame,dataset=row['dataset'],fold=row['fold'],
                    class_id=row['loader_class_id'],
                    query_photo_id=row['query_photo_id'] if row.get('query_photo_id') is not None else 'rgb/'+row['query_rgb_hash'],
                    reference_photo_id=row['reference_photo_id'] if row.get('reference_photo_id') is not None else 'rgb/'+row['reference_rgb_hash'],
                    metrics=metrics,iu={arm:[m['I'],m['U']] for arm,m in metrics.items()},
                    additions=dict(K=K,supported_TP=at,supported_FP=af,control_TP=ct,control_FP=cf),
                    parent_origin=s.get('origin')))
        if (i+1)%200==0 or i+1==2380:print(json.dumps(dict(scored=i+1,total=2380,seconds=round(time.monotonic()-tick,2))),flush=True)
    assert len(rows['cli'])==len(rows['original'])==2380 and sum(recon.values())==0
    write(OUT/'scored_episodes.json',rows)
    frame_report={}
    for frame,rr in rows.items():
        domains={}
        for d in producer.EXPECTED_COUNTS:
            sub=[r for r in rr if r['dataset']==d]
            ss=[s for s in sources if s['row']['dataset']==d]
            common=set.intersection(*(set(r['iu']) for r in sub))
            metrics_rows=[dict(r,iu={a:r['iu'][a] for a in common}) for r in sub]
            bases=tuple(a for a in ('foris.crf','mean','anchor.global','shared_edits_complete','same_edit_budget_source_order',
                                     'region.fast','task.competitive','rcg','rcg.fine') if a in common)
            summ=summarize(metrics_rows,baselines=bases,expected_classes=expected[d],repetitions=5000,unit='query_photo')
            for arm,value in summ['miou'].items():assert abs(value-official(sub,arm,expected[d]))<1e-10
            observed={str(f):sorted({r['class_id'] for r in sub if str(r['fold'])==str(f)}) for f in expected[d]}
            domains[d]=dict(summary=summ,official_slots=sum(len(v) for v in expected[d].values()),
                observed_slots=sum(len(v) for v in observed.values()),
                observed_class_only_secondary={a:official(sub,a,observed) for a in common},
                connected_RQ_photo_intervals=connected_intervals(metrics_rows,ss,expected[d],bases,repetitions=2000),
                affected_cases=affected[(d,frame)],control_coverage={a:sum(a in r['iu'] for r in sub) for a in sorted(set.union(*(set(r['iu']) for r in sub)))},
                action_sum={key:int(sum(r['additions'][key] for r in sub)) for key in ('K','supported_TP','supported_FP','control_TP','control_FP')})
        frame_report[frame]=domains
    # Exact LVIS600 source identifiers: compare only the same 600 cases and both exact frames.
    matched_bind=ROOT/'lvis600_matched_strong_control/bindings.json'
    matched_report=ROOT/'lvis600_matched_strong_control/report.json'
    exact=read(matched_bind);prior=read(matched_report);assert len(exact)==prior['n']==600
    match_ids={x['episode_id'] for x in exact};assert len(match_ids)==600
    matched={}
    for frame,rr in rows.items():
        sub=[x for x in rr if x['episode_id'] in match_ids]
        assert len(sub)==600 and all(x['dataset']=='lvis' for x in sub)
        all_slots=expected['lvis']
        observed={str(f):sorted({r['class_id'] for r in sub if str(r['fold'])==str(f)}) for f in all_slots}
        measured={a:official(sub,a,all_slots) for a in ('foris.crf','mean','anchor.global','shared_edits_complete','supported_mean','own_margin_sameK')}
        for a in ('foris.crf','mean','anchor.global','shared_edits_complete'):
            assert abs(measured[a]-prior['frames'][frame]['all_official_slots'][a])<1e-10
        matched[frame]=dict(n=600,all_official_slots=measured,
            observed600_slots={a:official(sub,a,observed) for a in measured},
            external_region_fast_same_subset=prior['frames'][frame]['all_official_slots']['component.query_context'],
            external_region_fast_observed=prior['frames'][frame]['observed600_slots']['component.query_context'],
            prior_report_sha256=sha(matched_report),bindings_sha256=sha(matched_bind))
    report=dict(state='SUPPORTED_FOREGROUND2380_SCORED_AND_INDEPENDENTLY_REPLAYED',n=2380,frames=frame_report,
        matched_LVIS600_strong_control=matched,checks=checks,
        reconstruction_mismatch_pixels=dict(recon),source_producer_sha256=cfg['source_sha256'],
        seal_sha256=sha(OUT/'sealed.json'),scored_records_sha256=sha(OUT/'scored_episodes.json'),
        scorer_sha256=sha(__file__),seconds=time.monotonic()-tick,
        primary_frame='cli',secondary_frame='original',
        exposure='Exposed development cohorts, not unseen confirmation or official full five benchmarks. Fixed class/fold slots retain absent zeros; observed slots secondary. LVIS600 strong control exact subset only.')
    write(OUT/'report.json',report)
    print(json.dumps(dict(state=report['state'],checks=checks,seconds=report['seconds'],
        cli={d:x['summary']['miou'] for d,x in frame_report['cli'].items()})),flush=True)

if __name__=='__main__':main()
