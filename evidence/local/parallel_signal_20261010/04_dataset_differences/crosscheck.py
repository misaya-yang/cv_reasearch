"""Metadata/IU crosschecks and late-arriving candidate diagnostics; no predictions."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import analyze as a

OUT=a.OUT;DATA=a.DATA;REPO=a.REPO

def miou(rows,expected=None):
    methods=sorted(set.intersection(*(set(r['iu']) for r in rows)))
    grouped=defaultdict(lambda:defaultdict(lambda:np.zeros(2,dtype=np.int64)))
    for row in rows:
        key=(str(row['fold']),str(row['class_id']))
        for m in methods:grouped[key][m]+=row['iu'][m]
    if expected:
        for fold,classes in expected.items():
            for cls in classes:
                for m in methods:grouped[str(fold),str(cls)][m]+=0
    values={}
    for m in methods:
        folds=defaultdict(list)
        for (fold,cls),v in grouped.items():
            i,u=v[m];folds[fold].append(float(i/u*100) if u else 0.)
        values[m]=float(np.mean([np.mean(v) for v in folds.values()]))
    return dict(n=len(rows),observed_classes=len({(r['fold'],r['class_id']) for r in rows}),
                denominator_slots=len(grouped),score=values)

def corr(x,y):
    v=[(aa,bb) for aa,bb in zip(x,y) if aa is not None and bb is not None and np.isfinite(aa) and np.isfinite(bb)]
    if len(v)<5:return dict(n=len(v),rho=None)
    xx,yy=map(np.array,zip(*v))
    if np.ptp(xx)==0 or np.ptp(yy)==0:return dict(n=len(v),rho=None)
    return dict(n=len(v),rho=float(stats.spearmanr(xx,yy).statistic))

profiles=a.lines(OUT/'mask_profiles.jsonl');main=a.lines(OUT/'main200_features.jsonl')
diags={r['episode_id']:r for r in a.lines(a.MECHANISM/'frontend_fields_episodes.jsonl')}
errs=[]
for r in main:
    errs += [abs(r['legal_observables']['ref_fg_fraction']-diags[r['episode_id']]['reference_fg_fraction']),
             abs(r['legal_observables']['ref_mixed_fg_mass']-diags[r['episode_id']]['mixed_fg_role_mass'])]
assert max(errs)==0

loaders={}
for name in ('paco_part','pascal_part','lvis','coco','suim'):
    p=DATA/f'third_party/foris_official/datasets/{name}.py';loaders[name]=dict(path=str(p),sha256=a.sha(p))
pre=DATA/'third_party/foris_official/utils/data.py'
loaders['preprocess']=dict(path=str(pre),sha256=a.sha(pre))

scores={}
dg=a.lines(DATA/'a/deepglobe_component100_20261009/score/scored_episodes.jsonl')
scores['Deep100_original_native9']=dict(n=len(dg),metric='total I / total U, road foreground',score={
    m:float(sum(r['iu'][m][0] for r in dg)/sum(r['iu'][m][1] for r in dg)*100) for m in dg[0]['iu']})
lvis=a.lines(DATA/'a/lvis_component_completion600_20261009/score/cumulative600_episodes.jsonl')
scores['LVIS600_original_native9']=miou(lvis)
paco_stage=a.lines(DATA/'a/paco_fast9_600_20261009/analysis/stage_attribution/episodes.jsonl')
paco=[dict(r,iu=r['iu']['original']) for r in paco_stage]
expected=a.read(DATA/'a/paco_fast9_600_20261009/expected_classes.json')
scores['PACO600_fast_native9_official_slots']=miou(paco,expected)
coco=a.lines(DATA/'a/coco_role_competition200_20261010/matched_raw/score/scored_episodes.jsonl')
scores['COCO200_whole_only_two_folds']=miou([dict(r,iu=r['iu']['original']) for r in coco],{str(f):list(range(f,80,4)) for f in (0,1)})
old=a.lines(DATA/'a/explore600_20261008_v1/run/episode_metrics.jsonl')
for dataset in ('pascal_part','suim'):
    scores[dataset+'_120_legacy']=miou([r for r in old if r['dataset']==dataset])

parents=a.lines(DATA/'a/paco_fast9_600_20261009/analysis/task_granularity/episodes.jsonl')
parent_stats={}
for cohort,eids in [('PACO600',{r['episode_id'] for r in paco}),('PACO100',{r['episode_id'] for r in main if r['dataset']=='paco_part'})]:
    rows=[r for r in parents if r['episode_id'] in eids]
    pp={}
    for role in ('query','reference'):
        pp[role+'_part_parent_ratio_GT_only']=a.summary([r[role]['part_pixels']/r[role]['parent_pixels'] for r in rows])
    for arm in ('foris.crf','region.fast'):
        c=[r['frames']['original']['arms'][arm] for r in rows]
        pp[arm]={k:sum(r[k] for r in c) for k in ('TP','FP','FN','FP_inside_associated_parent','FP_outside_associated_parent')}
        pp[arm]['FP_inside_fraction']=pp[arm]['FP_inside_associated_parent']/pp[arm]['FP']
    parent_stats[cohort]=dict(n=len(rows),**pp)

# Late-arriving fixed candidate is descriptive only; do not fit or choose any selector here.
candidate=REPO/'evidence/local/difficult_region_signal_20261010/candidate_reference_v1'
cs={r['episode_id']:r for r in a.lines(candidate/'scored_episodes.jsonl')}
cr={r['episode_id']:r for r in a.read(candidate/'sealed.json')['receipts']}
creport=a.read(candidate/'report.json')
candidate_diag={}
extra_scale=[]
candidate_rows=[]
for dataset in ('deepglobe_road','paco_part'):
    rows=[r for r in main if r['dataset']==dataset];ids=[r['episode_id'] for r in rows]
    delta=[cs[e]['iu']['anchor.local4'][0]/cs[e]['iu']['anchor.local4'][1]-cs[e]['iu']['anchor.global'][0]/cs[e]['iu']['anchor.global'][1] for e in ids]
    weights=[cr[e]['local_weight'] for e in ids]
    cd=dict(n=len(rows),local_better_episode_IoU=sum(v>0 for v in delta),whole_better_episode_IoU=sum(v<0 for v in delta),
        local_weight=a.summary(weights),weight_vs_episode_IoU_delta=corr(weights,delta),
        weight_vs_new_ROI_AUC_delta=corr(weights,[r['rois']['new_unary_foreground']['delta_auc'] for r in rows]),
        whole_selected_local_would_improve=sum(w==0 and d>0 for w,d in zip(weights,delta)),
        whole_selected=sum(w==0 for w in weights),
        correlations={f:corr([r['legal_observables'][f] for r in rows],delta) for f in a.FEATURES},
        metric_warning='Per-episode IoU deltas for associations, not class/fold or total-IU aggregate delta; no selector fitted.',
        final_scores=creport['miou'][dataset])
    candidate_diag[dataset]=cd
    for r,d,w in zip(rows,delta,weights):
        candidate_rows.append(dict(episode_id=r['episode_id'],dataset=dataset,anchor_local_minus_whole_episode_IoU=d,local_weight=w,
            reference_threshold=cr[r['episode_id']]['calibration']['threshold'],reference_separation=cr[r['episode_id']]['calibration']['separation']))
    for f in ('reference_input_area','query_input_area'):
        for roi in ('foris_foreground','new_unary_foreground'):
            extra_scale.append(dict(dataset=dataset,feature=f,roi=roi,**corr([r['legal_observables'][f] for r in rows],[r['rois'][roi]['delta_auc'] for r in rows]),
                status='additional descriptive physical input scale check, after initial fixed-family analysis'))

for dataset in ('deepglobe_road','paco_part'):
    rows=[cs[r['episode_id']] for r in main if r['dataset']==dataset]
    calculated=miou(rows)['score'] if dataset=='paco_part' else {m:sum(r['iu'][m][0] for r in rows)/sum(r['iu'][m][1] for r in rows)*100 for m in rows[0]['iu']}
    for m,v in creport['miou'][dataset].items():assert abs(calculated[m]-v)<1e-10,(dataset,m,calculated[m],v)

(OUT/'candidate_association_rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in candidate_rows))
a.write(OUT/'crosscheck.json',dict(loaders=loaders,reference_coverage_vs_prior_max_error=max(errs),full_output_scores=scores,parent_GT_diagnostics=parent_stats,
    candidate_v1=candidate_diag,additional_physical_scale_correlations=extra_scale,source_sha256=a.receipts,
    new_predictions=0,raw_reads=0,encoder_calls=0,query_GT_used_for_selection=False,script_sha256=a.sha(__file__)))
print(json.dumps(dict(full_output_scores=scores,parent_GT_diagnostics=parent_stats,candidate_v1={k:{kk:vv for kk,vv in v.items() if kk not in ('correlations','final_scores')} for k,v in candidate_diag.items()},additional_physical_scale_correlations=extra_scale)),flush=True)
