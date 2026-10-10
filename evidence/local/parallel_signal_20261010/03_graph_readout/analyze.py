#!/usr/bin/env python3
"""Read-only source audit and frozen LVIS100 graph replay/GT diagnostics.

All writes stay beside this script. No model, raw features, pool, grid, or MPS.
Replay every mask before opening any query GT in this execution.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = REPO.parent/'cv_data'
SOURCE = DATA/'a/lvis_context_rank_graph100_20261009'
ATOMIC = DATA/'a/lvis_atomic1400_20261009'
OLD = REPO/'evidence/local/research_20261005/pipeline_verified'
CAND, CTRL = 'mean.context_rank_graph', 'mean.context_rank_matched'
ARMS = ['foris.crf', 'mean', 'mean.graph_only', 'mean.rank', CAND, CTRL]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]

def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def point(records, frame, arm):
    groups = defaultdict(lambda: np.zeros(2, np.int64))
    for r in records: groups[(r['fold'], r['class_id'])] += r['frames'][frame]['iu'][arm]
    folds = defaultdict(list)
    for (f, _), (i, u) in groups.items(): folds[f].append(i/max(u, 1))
    return 100*float(np.mean([np.mean(v) for v in folds.values()]))

def exact_contribution(records, frame, candidate, baseline):
    groups = defaultdict(lambda: {'base': np.zeros(2, np.int64), 'new': np.zeros(2, np.int64), 'edits': np.zeros(4, np.int64)})
    for r in records:
        g=groups[(r['fold'], r['class_id'])]
        g['base'] += r['frames'][frame]['iu'][baseline]
        g['new'] += r['frames'][frame]['iu'][candidate]
        g['edits'] += r['frames'][frame]['edits'][candidate][baseline]
    folds = defaultdict(list)
    for (f, _), v in groups.items():
        bi, bu = v['base']; ni, nu = v['new']; at, af, dt, df = v['edits']
        assert ni-bi == at-dt and nu-bu == af-df
        j=bi/max(bu, 1)
        folds[f].append(100*np.array([at, -j*af, -dt, j*df])/max(nu, 1))
    v=np.mean([np.mean(x, axis=0) for x in folds.values()], axis=0)
    expected=point(records, frame, candidate)-point(records, frame, baseline)
    assert abs(v.sum()-expected) < 1e-10
    return dict(zip(['add_TP', 'add_FP', 'delete_TP', 'delete_FP'], v.tolist()))

def old_audit():
    source=OLD/'mean_fine_residual_transfer4000_v1'
    rows=lines(source/'episodes.jsonl'); report=read(source/'report.json')
    cls=np.array([r['c'] for r in rows]); fold=np.array([r['fold'] for r in rows])
    result={'sources': {str(p):sha(p) for p in [source/'counts.npz',source/'episodes.jsonl',source/'report.json']},'n':len(rows),'frame':'CLI1024; 80 class pooled I/U macro', 'recomputed':{},'four_edit_contributions':{},'folds':{}}
    def metric(v, index=None):
        index=np.ones(len(rows), bool) if index is None else index
        return 100*float(np.mean([v[index & (cls==c)].sum(0)[0]/max(v[index & (cls==c)].sum(0)[1],1) for c in np.unique(cls[index])]))
    with np.load(source/'counts.npz') as z:
        for name in report['scores']:
            a=z['iu:'+name];assert np.array_equal(a, np.asarray([r['iu'][name] for r in rows]))
            result['recomputed'][name]=metric(a);assert abs(metric(a)-report['scores'][name]) < 1e-10
        cand='mean_fine_residual_transfer_v1'
        for base in ['native','rcg','mean.control','fine.rcg16.control','fine.rcg64']:
            b=z['iu:'+base];c=z['iu:'+cand];e=z['edits_vs_'+base]
            assert np.array_equal(c[:,0]-b[:,0], e[:,0]-e[:,2])
            assert np.array_equal(c[:,1]-b[:,1], e[:,1]-e[:,3])
            parts=[]
            for label in np.unique(cls):
                bi,bu=b[cls==label].sum(0);ci,cu=c[cls==label].sum(0)
                at,af,dt,df=e[cls==label].sum(0);j=bi/max(bu,1)
                parts.append(100*np.array([at,-j*af,-dt,j*df])/max(cu,1))
            part=np.mean(parts, axis=0);assert abs(part.sum()-metric(c)+metric(b))<1e-10
            result['four_edit_contributions'][base]={'pixels':e.sum(0).tolist(),'pp':part.tolist(),'order':['add_TP','add_FP','delete_TP','delete_FP'], 'positive_cases':int((c[:,0]/np.maximum(c[:,1],1)>b[:,0]/np.maximum(b[:,1],1)+1e-12).sum())}
        for f in range(4): result['folds'][f]={name:metric(z['iu:'+name], fold==f) for name in report['scores']}
    cf=OLD/'exact_family_selection_v1/public4000_v3_crossfold'
    rr=read(cf/'report.json'); arms=read(cf/'arms.json');sel=read(cf/'selection.json')
    with np.load(cf/'counts.npz') as z:
        for name,value in rr['scores'].items():assert abs(metric(z[name])-value)<1e-10
        assert np.array_equal(z['native'], np.asarray([r['iu']['native'] for r in rows]))
        result['crossfold12']={'point':metric(z['heldfold_recipe']), 'library_size':len(arms), 'has_sizecut_arm':any('sizecut' in a['id'] for a in arms),'recipes':{f:v['selected'] for f,v in sel['folds'].items()},'recipes_searched_per_fold':[v['recipes_evaluated'] for v in sel['folds'].values()], 'all_report_scores_recomputed':True}
    audit=REPO/'evidence/local/rcg_transfer_20261007/audit.json'
    d=read(audit);result['cross_dataset_source_sha256']=sha(audit);result['cross_dataset']={}
    for name,v in d['datasets'].items():
        val={'n':v['n'],'classes':v['classes'],'miou':v['miou'],'frame':d['protocol']['metric'],'source_level':'retained per-class IoU summaries; no local original masks/counts', 'contributions':{}}
        for trans,b,a in [('rcg_vs_native','native','rcg'),('fine_vs_rcg','rcg','rcg_fine'),('full_vs_fine','rcg_fine','rcg2')]:
            q=v[trans];pcs=q['per_class']
            assert abs(np.mean([x['base_iou'] for x in pcs])*100-v['miou'][b])<1e-10
            assert abs(np.mean([x['new_iou'] for x in pcs])*100-v['miou'][a])<1e-10
            val['contributions'][trans]={k:x for k,x in q.items() if k!='per_class'}
        val['gt_area_strata']=v['gt_area_strata'];result['cross_dataset'][name]=val
    write(HERE/'historical_recompute.json',result)
    return result

def replay():
    import torch
    from scipy import sparse
    from ics.methods.context_rank_graph import build
    from ics.methods.tail_rank_control import solve_parent
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();config=read(SOURCE/'config.json');seal=read(SOURCE/'sealed.json')
    for file,key in [('config.json','config_sha256'),('manifest.json','manifest_sha256'),('tasks.json','tasks_sha256'),('inference.jsonl','inference_index_sha256')]:assert sha(SOURCE/file)==seal[key]
    for path in [REPO/'src/ics/methods/context_rank_graph.py',REPO/'src/ics/methods/tail_rank_control.py',REPO/'src/ics/methods/rcg.py',REPO/'scripts/run_m4_baselines.py']:
        assert sha(path)==config['source_sha256'][str(path)],str(path)
    manifest=read(SOURCE/'manifest.json'); tasks=read(SOURCE/'tasks.json');index={r['episode_id']:r for r in lines(SOURCE/'inference.jsonl')}
    atomic={r['episode_id']:r for r in lines(ATOMIC/'inference.jsonl')}
    (HERE/'predictions').mkdir(exist_ok=True);(HERE/'fields').mkdir(exist_ok=True)
    result=[]
    for row,task in zip(manifest,tasks):
        assert row['episode_id']==task['episode_id'];rec=index[row['episode_id']]
        fp=SOURCE/'fields'/task['filename'];pp=SOURCE/'predictions'/task['filename'];sp=Path(task['source_field'])
        assert sha(fp)==rec['fields_sha256'] and sha(pp)==rec['prediction_sha256'] and sha(sp)==task['source_field_sha256']
        with np.load(sp) as z:
            y=z['parent_unary'].reshape(-1);a=z['parent_a'].reshape(-1);s=z['source_s'].reshape(-1);g=z['parent_guide'].reshape(-1);parent=z['parent_mean']
            h=sparse.csr_matrix((z['parent_H_data'],z['parent_H_indices'],z['parent_H_indptr']),shape=tuple(z['parent_H_shape']))
        out=build(h,s,g);base,info=solve_parent(y,a,h);assert np.array_equal(base.reshape(64,64),parent)
        fields={'mean':parent,'mean.rank':y.reshape(64,64),'source_s':s.reshape(64,64),'guide':g.reshape(64,64)}
        packed={};case={'episode_id':row['episode_id'],'filename':task['filename'],'source_hashes':{str(p):sha(p) for p in [fp,pp,sp]}, 'graph_mass_error':out['diagnostics']['edge_mass_absolute_error'], 'solver':{'mean':info}}
        with np.load(fp) as z:
            for key,val in out['arrays'].items():assert np.array_equal(val,z[key]),key
            for arm,kind in [(CAND,'candidate'),(CTRL,'matched')]:
                field,inf=solve_parent(y,a,out[kind+'_H']);fields[arm]=field.reshape(64,64);case['solver'][arm]=inf
                assert np.array_equal(fields[arm],z[arm]),arm
        ar=atomic[row['episode_id']];ap=ATOMIC/'predictions'/ar['filename'];assert sha(ap)==ar['prediction_sha256']
        with np.load(pp) as z,np.load(ap) as az:
            for arm in ARMS:
                if arm in fields:
                    mask=mask_from_field(fields[arm]);orig=render(mask,tuple(row['query_size_hw']))
                    for frame,m in [('cli',mask),('original',orig)]:
                        b=np.packbits(m);expected=az[frame+'/'+arm] if arm=='mean.rank' else z[frame+'/'+arm]
                        assert np.array_equal(b,expected),(arm,frame,row['episode_id']);packed[frame+'/'+arm]=b
                else:
                    for frame in ('cli','original'):packed[frame+'/'+arm]=z[frame+'/'+arm].copy()
        p=HERE/'predictions'/task['filename'];f=HERE/'fields'/task['filename']
        np.savez_compressed(p,**packed);np.savez_compressed(f,**fields)
        case.update(predictions_sha256=sha(p),fields_sha256=sha(f),all_fields_and_masks_bit_exact=True)
        result.append(case)
    write(HERE/'replay.json',result);write(HERE/'manifest.json',manifest)
    receipt={'state':'ALL100_REPLAYS_SEALED_BEFORE_QUERY_GT','n':len(result),'source_seal_sha256':sha(SOURCE/'sealed.json'),'manifest_sha256':sha(HERE/'manifest.json'),'replay_sha256':sha(HERE/'replay.json'),'script_sha256':sha(__file__),'encoder_calls':0,'raw_feature_reads':0,'query_GT_opened':False,'new_algorithm_arms':0,'threads':1,'processes':1,'all_saved_graph_arrays_fields_and_masks_bit_exact':True,'seconds':time.monotonic()-start}
    write(HERE/'replay_sealed.json',receipt)
    return receipt

def pool(mask):return mask.reshape(64,16,64,16).sum((1,3),dtype=np.float64).reshape(-1)

def auc(score,positive,negative):
    keep=(positive+negative)>0;s=score.reshape(-1)[keep].astype(np.float64);p=positive[keep];n=negative[keep]
    if p.sum()==0 or n.sum()==0:return None
    order=np.argsort(s,kind='stable');s=s[order];p=p[order];n=n[order]
    start=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];pg=np.add.reduceat(p,start);ng=np.add.reduceat(n,start)
    return float(np.dot(pg,np.cumsum(ng)-.5*ng)/(p.sum()*n.sum()))

def diagnose():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    start=time.monotonic();assert read(HERE/'replay_sealed.json')['query_GT_opened'] is False
    replay_rows=read(HERE/'replay.json');manifest=read(HERE/'manifest.json');prior={r['episode_id']:r for r in lines(SOURCE/'episode_metrics.jsonl')}
    records=[]
    for row,inf in zip(manifest,replay_rows):
        p=HERE/'predictions'/inf['filename'];f=HERE/'fields'/inf['filename'];assert sha(p)==inf['predictions_sha256'] and sha(f)==inf['fields_sha256']
        with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert array_hash(raw)==row['query_mask_hash']
        record={'episode_id':row['episode_id'],'fold':row['fold'],'class_id':row['loader_class_id'],'frames':{}}
        with np.load(p) as z:
            for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                truth=F.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                masks={arm:np.unpackbits(z[frame+'/'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool) for arm in ARMS}
                iu={arm:counts(m,truth) for arm,m in masks.items()}
                for arm in ARMS:
                    if arm!='mean.rank':assert iu[arm]==prior[row['episode_id']]['frames'][frame]['iu'][arm]
                edits={arm:{base:gross_edits(masks[arm],masks[base],truth) for base in ['mean','mean.rank',CTRL,'foris.crf']} for arm in [CAND,CTRL,'mean']}
                record['frames'][frame]={'iu':iu,'edits':edits,'truth_pixels':int(truth.sum())}
                if frame!='cli':continue
                ym=masks['mean.rank'];bm=masks['mean'];cm=masks[CAND];um=masks[CTRL]
                roi={'graph_removed':ym&~bm,'graph_added':~ym&bm,'mean_foreground':bm,'candidate_added':~bm&cm,'candidate_vs_matched_disagreement':cm^um}
                with np.load(f) as ff:
                    signals={'source_s':ff['source_s'],'unary':ff['mean.rank'],'guide':ff['guide'],'mean':ff['mean'], 'candidate_final':ff[CAND], 'matched_final':ff[CTRL], 'candidate_residual':ff[CAND]-ff['mean'],'matched_residual':ff[CTRL]-ff['mean'], 'candidate_minus_matched':ff[CAND]-ff[CTRL]}
                    region_stats={}
                    for name,rr in roi.items():
                        fg=pool(rr&truth);bg=pool(rr&~truth)
                        region_stats[name]={'FG_pixels':int(fg.sum()),'BG_pixels':int(bg.sum()),'weighted_AUC':{key:auc(val,fg,bg) for key,val in signals.items()}}
                conflicts={}
                for arm in [CAND,CTRL]:
                    m=masks[arm]
                    conflicts[arm]={
                        'restored_graph_deleted_TP':int((truth&ym&~bm&m).sum()),
                        'restored_graph_deleted_FP':int((~truth&ym&~bm&m).sum()),
                        'new_TP_outside_positive_unary':int((truth&~ym&~bm&m).sum()),
                        'new_FP_outside_positive_unary':int((~truth&~ym&~bm&m).sum()),
                        'lost_correct_unary_previously_kept_TP':int((truth&ym&bm&~m).sum()),
                        'removed_wrong_unary_previously_kept_FP':int((~truth&ym&bm&~m).sum()),
                        'corrected_graph_added_FP':int((~truth&~ym&bm&~m).sum()),
                        'lost_graph_added_TP':int((truth&~ym&bm&~m).sum())}
                    vals=conflicts[arm];gross=edits[arm]['mean']
                    assert vals['restored_graph_deleted_TP']+vals['new_TP_outside_positive_unary']==gross[0]
                    assert vals['restored_graph_deleted_FP']+vals['new_FP_outside_positive_unary']==gross[1]
                    assert vals['lost_correct_unary_previously_kept_TP']+vals['lost_graph_added_TP']==gross[2]
                    assert vals['removed_wrong_unary_previously_kept_FP']+vals['corrected_graph_added_FP']==gross[3]
                record['regions']=region_stats;record['conflicts']=conflicts
        records.append(record)
    (HERE/'diagnostic_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    summary={'state':'COMPLETE','n':len(records),'exposure':'same previously exposed LVIS100; exploration, no parameter selection here', 'metric_boundaries':'Full masks CLI and original separately; AUC uses pooled actual CLI ROI FG/BG mass and constant64-node score, not pixel-bilinear score or mIoU', 'groups':{}, 'new_algorithm_arms':0,'new_encoder_calls':0,'new_raw_feature_reads':0,'diagnostic_seconds':time.monotonic()-start}
    for group,rs in [('all100',records),('query_lt1pct',[r for r in records if r['frames']['cli']['truth_pixels']<.01*1024**2]),('query_ge1pct',[r for r in records if r['frames']['cli']['truth_pixels']>=.01*1024**2])]:
        v={'n':len(rs),'frames':{},'regions':{},'conflicts':{}}
        for frame in ('cli','original'):
            points={arm:point(rs,frame,arm) for arm in ARMS};contrib={}
            for a,b in [(CAND,CTRL),(CAND,'mean'),(CTRL,'mean'),('mean','mean.rank')]:contrib[a+' vs '+b]=exact_contribution(rs,frame,a,b)
            v['frames'][frame]={'miou':points,'four_edit_pp':contrib,'zero_IU':{arm:sum(r['frames'][frame]['iu'][arm][0]==0 for r in rs) for arm in ARMS}}
        for region in rs[0]['regions']:
            vals=[r['regions'][region] for r in rs];signals=vals[0]['weighted_AUC']
            v['regions'][region]={'FG_pixels':sum(x['FG_pixels'] for x in vals),'BG_pixels':sum(x['BG_pixels'] for x in vals), 'macro_AUC':{s:{'n':sum(x['weighted_AUC'][s] is not None for x in vals),'mean':float(np.mean([x['weighted_AUC'][s] for x in vals if x['weighted_AUC'][s] is not None])) if any(x['weighted_AUC'][s] is not None for x in vals) else None} for s in signals}}
        for arm in [CAND,CTRL]:v['conflicts'][arm]={key:sum(r['conflicts'][arm][key] for r in rs) for key in rs[0]['conflicts'][arm]}
        summary['groups'][group]=v
    summary['source_hashes']={str(p):sha(p) for p in [SOURCE/'sealed.json',SOURCE/'episode_metrics.jsonl',HERE/'replay_sealed.json',HERE/'diagnostic_episodes.jsonl']}
    summary['replay']=read(HERE/'replay_sealed.json')
    write(HERE/'diagnostic_summary.json',summary)
    return summary

def main():
    began=time.monotonic();historical=old_audit();rep=replay();diag=diagnose()
    write(HERE/'COMPLETE.json',{'state':'COMPLETE','n_replay':100,'old4000_IU_checks':True,'seconds':time.monotonic()-began,'script_sha256':sha(__file__),'query_GT_only_after_full100_replay_seal':True,'torch_threads':1,'processes':1,'encoder_calls':0,'raw_feature_reads':0})
    print(json.dumps({'state':'COMPLETE','replay_seconds':rep['seconds'],'total_seconds':time.monotonic()-began,'all100':diag['groups']['all100']},allow_nan=False))

if __name__=='__main__':main()
