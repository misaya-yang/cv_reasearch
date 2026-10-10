"""Read-only one-CPU diagnostics from frozen manifests, masks and score fields."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, stats

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
DATA=REPO.parent/'cv_data'
PILOT=DATA/'a/joint_role_pilot200_20261010'
SIGNAL=REPO/'evidence/local/difficult_region_signal_20261010'
MECHANISM=DATA/'a/cache_mechanism_study_20261010'
receipts={}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);receipts[str(p)]=sha(p)
    return json.loads(p.read_text())
def lines(p):
    p=Path(p);receipts[str(p)]=sha(p)
    return [json.loads(v) for v in p.read_text().splitlines() if v]
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def array_sha(x):
    x=np.ascontiguousarray(x)
    h=hashlib.sha256(json.dumps([list(x.shape),x.dtype.str]).encode());h.update(x.tobytes())
    return h.hexdigest()
def summary(x):
    a=np.array([v for v in x if v is not None and np.isfinite(v)],dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)
def rankcorr(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(stats.spearmanr(a,b).statistic)

mask_cache={}
def shape_stats(binary):
    binary=np.asarray(binary,dtype=bool)
    h,w=binary.shape;area=int(binary.sum())
    if not area:return dict(fg_fraction=0.,cc8=0,largest_cc_fraction=0.,bbox_fraction=0.,bbox_fill=0.,boundary_fg_fraction=0.)
    cc,n=ndimage.label(binary,structure=np.ones((3,3),dtype=bool))
    sizes=np.bincount(cc.ravel())[1:]
    ys,xs=np.nonzero(binary)
    boxarea=int((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1))
    # A raster boundary diagnostic, not an object width estimator.
    boundary=binary & ~ndimage.binary_erosion(binary,structure=np.ones((3,3),dtype=bool),border_value=0)
    return dict(fg_fraction=area/(h*w),cc8=int(n),largest_cc_fraction=float(sizes.max()/area),
                bbox_fraction=boxarea/(h*w),bbox_fill=area/boxarea,boundary_fg_fraction=float(boundary.sum()/area))

def mask_stats(row,role):
    p=Path(row[role+'_mask_path']);expected=row[role+'_mask_hash']
    key=(str(p),expected)
    if key in mask_cache:return mask_cache[key]
    with Image.open(p) as im: arr=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert array_sha(arr)==expected,(row['episode_id'],role,'mask identity')
    assert list(arr.shape)==row[role+'_mask_size_hw']
    h,w=arr.shape
    # Exactly the floor source-index mapping of torch interpolate(mode='nearest').
    canon=arr[(np.arange(1024)*h//1024)[:,None],(np.arange(1024)*w//1024)[None,:]]
    c=canon.reshape(64,16,64,16).mean(axis=(1,3)).ravel()
    area=c.sum();mixed=(c>0)&(c<1)
    d=shape_stats(canon)
    d.update(original_h=h,original_w=w,original_fg_fraction=float(arr.mean()),
        fg_mass_purity=float((c*c).sum()/area) if area else None,
        mixed_fg_mass=float(c[mixed].sum()/area) if area else None,
        fg_mass_ge90=float(c[c>=.9].sum()/area) if area else None,
        tokens_ge90=int((c>=.9).sum()),tokens_fg=int((c>0).sum()),
        coverage_max=float(c.max()),coverage_median_positive=float(np.median(c[c>0])) if area else None)
    mask_cache[key]=d
    receipts[str(p)]=sha(p)
    return d

def cohort_rows():
    cohorts={}
    cohorts['Deep100']=read(PILOT/'manifest.json')[:100]
    cohorts['PACO600']=read(DATA/'a/paco_fast9_600_20261009/manifest.json')
    cohorts['LVIS600']=sum([read(DATA/f'a/{n}/manifest.json') for n in ('lvis_fixedwindows200_20261009','lvis_component_replication200_20261009','lvis_component_completion600_20261009')],[])
    cohorts['COCO200']=read(DATA/'a/coco_role_competition200_20261010/manifest.json')
    old=read(DATA/'a/explore600_20261008_v1/manifest.json')
    cohorts['PASCAL120_legacy']=[r for r in old if r['dataset']=='pascal_part']
    cohorts['SUIM120_legacy']=[r for r in old if r['dataset']=='suim']
    return cohorts

def collect():
    begin=time.monotonic();profiles=[];cohorts=cohort_rows()
    for cohort,rows in cohorts.items():
        assert len({r['episode_id'] for r in rows})==len(rows)
        for row in rows:
            d=dict(cohort=cohort,episode_id=row['episode_id'],dataset=row['dataset'],fold=row['fold'],class_id=row['loader_class_id'],
                global_class_id=row['global_class_id'],reference_rgb_hash=row['reference_rgb_hash'],query_rgb_hash=row['query_rgb_hash'],
                reference_mask_hash=row['reference_mask_hash'],query_mask_hash=row['query_mask_hash'],
                reference_size_hw=row['reference_size_hw'],query_size_hw=row['query_size_hw'],
                reference_crop=row['reference_crop'],query_crop=row['query_crop'],
                reference=mask_stats(row,'reference'),query_GT_diagnostic=mask_stats(row,'query'))
            for role in ('reference','query'):
                h,w=row[role+'_size_hw'];d[role+'_input_area']=h*w;d[role+'_aspect_abs']=max(h/w,w/h)
                d[role+'_image_mask_size_match']=row[role+'_size_hw']==row[role+'_mask_size_hw']
            d['reference_query_input_area_ratio']=d['reference_input_area']/d['query_input_area']
            profiles.append(d)
        print(json.dumps(dict(stage='mask_profile',cohort=cohort,n=len(rows),seconds=time.monotonic()-begin)),flush=True)
    (OUT/'mask_profiles.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in profiles))
    # Main200 is an exact indexed subset of the profiled cohorts.
    byid={p['episode_id']:p for p in profiles};manifest=read(PILOT/'manifest.json')
    main=[]
    scores={v['episode_id']:v for v in lines(SIGNAL/'raw/episodes.jsonl')}
    existing={v['episode_id']:v for v in lines(SIGNAL/'existing/episodes.jsonl')}
    seal={v['episode_id']:v for v in read(SIGNAL/'raw/sealed.json')['receipts']}
    infer={v['episode_id']:v for v in lines(PILOT/'inference.jsonl')}
    diags={v['episode_id']:v for v in lines(MECHANISM/'frontend_fields_episodes.jsonl')}
    parents={v['episode_id']:v for v in lines(DATA/'a/paco_fast9_600_20261009/analysis/task_granularity/episodes.jsonl')}
    for row in manifest:
        eid=row['episode_id'];p=byid[eid]
        assert p['reference_mask_hash']==row['reference_mask_hash'] and p['query_mask_hash']==row['query_mask_hash']
        a=scores[eid];s=seal[eid];old=infer[eid]
        fpath=SIGNAL/'raw/fields'/s['filename'];assert sha(fpath)==s['sha256'];receipts[str(fpath)]=s['sha256']
        with np.load(fpath,allow_pickle=False) as z:
            g=z['source_apd.ridge.global'].astype(float);l=z['source_apd.ridge.local4'].astype(float)
        predpath=PILOT/'predictions'/old['filename'];assert sha(predpath)==old['prediction_sha256'];receipts[str(predpath)]=old['prediction_sha256']
        with np.load(predpath,allow_pickle=False) as z:
            base=np.unpackbits(z['cli1024/foris.crf'],count=1024**2).reshape(1024,1024).astype(bool)
            unary=np.unpackbits(z['cli1024/role.unary'],count=1024**2).reshape(1024,1024).astype(bool)
        bst=shape_stats(base)
        ref=p['reference'];qgt=p['query_GT_diagnostic']
        legal={f'ref_{k}':ref[k] for k in ('fg_fraction','fg_mass_purity','mixed_fg_mass','fg_mass_ge90','cc8','largest_cc_fraction','bbox_fill','boundary_fg_fraction')}
        legal.update({k:p[k] for k in ('reference_input_area','query_input_area','reference_aspect_abs','query_aspect_abs','reference_query_input_area_ratio')})
        legal.update(source_semantic_score=s['source_semantic_score'],apd_applied=s['apd_applied'],
            fit_tokens=s['fit']['source_apd']['fit_tokens'],selected_target_max=s['fit']['source_apd']['selected_target_max'],
            view_rank_agreement=rankcorr(g.ravel(),l.ravel()),
            ridge_whole_fgfrac_zero=float((g>0).mean()),ridge_local_fgfrac_zero=float((l>0).mean()),
            ridge_view_sign_disagreement=float(((g>0)!=(l>0)).mean()),
            ridge_std_local_over_whole=float(l.std()/g.std()) if g.std() else None,
            foris_fg_fraction=bst['fg_fraction'],foris_cc8=bst['cc8'],foris_largest_cc_fraction=bst['largest_cc_fraction'],
            foris_fg_over_ref_fg=bst['fg_fraction']/ref['fg_fraction'],
            P0_fg_fraction=float(unary.mean()),new_unary_fg_fraction=float((unary&~base).mean()),
            alpha_fg_mean=diags[eid]['alpha_fg']['mean'],alpha_bg_mean=diags[eid]['alpha_bg']['mean'])
        seam=np.mean(np.r_[np.abs(l[:,64]-l[:,63]),np.abs(l[64,:]-l[63,:])])
        neighbor=np.mean(np.r_[np.abs(l[:,1:]-l[:,:-1]).ravel(),np.abs(l[1:,:]-l[:-1,:]).ravel()])
        legal['local_seam_relative_jump']=float(seam/neighbor) if neighbor else None
        diagnostic={f'query_GT_{k}':qgt[k] for k in ('fg_fraction','fg_mass_purity','fg_mass_ge90','cc8','largest_cc_fraction','bbox_fill','boundary_fg_fraction')}
        diagnostic['query_GT_over_ref_fg']=qgt['fg_fraction']/ref['fg_fraction']
        if eid in parents:
            pp=parents[eid]
            assert pp['query']['part_binary_tensor_sha256']==row['query_mask_hash']
            assert pp['reference']['part_binary_tensor_sha256']==row['reference_mask_hash']
            for role in ('reference','query'):
                ppv=pp[role]
                # Parent segmentation is NOT supplied by this single-reference task.
                diagnostic[role+'_GT_part_over_parent']=ppv['part_pixels']/ppv['parent_pixels']
            cnt=pp['frames']['original']['arms']['foris.crf']
            diagnostic['query_GT_foris_FP_inside_parent_fraction']=cnt['FP_inside_associated_parent']/cnt['FP'] if cnt['FP'] else None
        rois={}
        for roi,values in a['rois'].items():
            gv=values['source_apd.ridge.global'];lv=values['source_apd.ridge.local4']
            evalues=existing[eid]['rois'][roi]
            rois[roi]=dict(nfg=gv['fg'],nbg=gv['bg'],prevalence=gv.get('prevalence'),
                whole_auc=gv['auc'],local_auc=lv['auc'],delta_auc=lv['auc']-gv['auc'] if gv['auc'] is not None and lv['auc'] is not None else None,
                whole_ap=gv['ap'],local_ap=lv['ap'],delta_ap=lv['ap']-gv['ap'] if gv['ap'] is not None and lv['ap'] is not None else None,
                P0_auc=evalues['P0']['auc'],source_auc=evalues['foris_source']['auc'])
        main.append(dict(episode_id=eid,dataset=row['dataset'],fold=row['fold'],class_id=row['loader_class_id'],reference_mask_hash=row['reference_mask_hash'],
                         query_mask_hash=row['query_mask_hash'],legal_observables=legal,GT_diagnostics=diagnostic,rois=rois))
    (OUT/'main200_features.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in main))
    write(OUT/'input_receipt.json',dict(created_unix=time.time(),script_sha256=sha(__file__),threads=1,process_pool=False,
        encoder_calls=0,raw_payload_reads=0,raw_cache_writes=0,unique_mask_path_hash_pairs=len(mask_cache),profile_rows=len(profiles),main200_rows=len(main),
        input_sha256=receipts,seconds=time.monotonic()-begin))

FEATURES=['ref_fg_fraction','ref_fg_mass_purity','ref_mixed_fg_mass','ref_cc8','ref_largest_cc_fraction','ref_bbox_fill',
          'reference_query_input_area_ratio','query_aspect_abs','source_semantic_score','foris_fg_fraction','foris_fg_over_ref_fg',
          'foris_largest_cc_fraction','view_rank_agreement','ridge_std_local_over_whole','ridge_view_sign_disagreement','local_seam_relative_jump']
GT_FEATURES=['query_GT_fg_fraction','query_GT_fg_mass_purity','query_GT_cc8','query_GT_largest_cc_fraction','query_GT_over_ref_fg',
             'query_GT_part_over_parent','query_GT_foris_FP_inside_parent_fraction']

def correlation(rows,key,kind,roi):
    pairs=[(r[kind].get(key),r['rois'][roi]['delta_auc']) for r in rows]
    pairs=[p for p in pairs if all(v is not None and np.isfinite(v) for v in p)]
    if len(pairs)<5:return dict(n=len(pairs),rho=None,p=None)
    x,y=map(np.array,zip(*pairs))
    if np.ptp(x)==0 or np.ptp(y)==0:return dict(n=len(pairs),rho=None,p=None)
    z=stats.spearmanr(x,y)
    return dict(n=len(x),rho=float(z.statistic),p=float(z.pvalue))

def aggregate():
    profiles=lines(OUT/'mask_profiles.jsonl');main=lines(OUT/'main200_features.jsonl')
    cohort_stats={}
    for cohort in sorted({r['cohort'] for r in profiles}):
        rows=[r for r in profiles if r['cohort']==cohort]
        cohort_stats[cohort]=dict(n=len(rows),fold_class_count=len({(r['fold'],r['class_id']) for r in rows}),
            unique_reference_RGB=len({r['reference_rgb_hash'] for r in rows}),unique_query_RGB=len({r['query_rgb_hash'] for r in rows}),
            reference_crops=sum(r['reference_crop'] is not None for r in rows),query_crops=sum(r['query_crop'] is not None for r in rows),
            reference_image_mask_mismatches=sum(not r['reference_image_mask_size_match'] for r in rows),query_image_mask_mismatches=sum(not r['query_image_mask_size_match'] for r in rows),
            reference={k:summary([r['reference'][k] for r in rows]) for k in rows[0]['reference']},
            query_GT_diagnostic={k:summary([r['query_GT_diagnostic'][k] for r in rows]) for k in rows[0]['query_GT_diagnostic']},
            input={k:summary([r[k] for r in rows]) for k in ('reference_input_area','query_input_area','reference_aspect_abs','query_aspect_abs','reference_query_input_area_ratio')})
    groups={name:[r for r in main if name=='pooled' or r['dataset']==name] for name in ('pooled','deepglobe_road','paco_part')}
    grouped={};correlations=[];rng=np.random.default_rng(20261010)
    for name,rows in groups.items():
        grouped[name]=dict(n=len(rows),observables={k:summary([r['legal_observables'][k] for r in rows]) for k in FEATURES},rois={})
        for roi in ('global','foris_foreground','new_unary_foreground'):
            valid=[r['rois'][roi] for r in rows if r['rois'][roi]['delta_auc'] is not None]
            delta=np.array([r['delta_auc'] for r in valid])
            # Diagnostic conditional episode bootstrap; all cohorts already exposed.
            boot=np.mean(rng.choice(delta,(5000,len(delta)),replace=True),axis=1)
            grouped[name]['rois'][roi]=dict(n_valid=len(valid),whole_auc=float(np.mean([r['whole_auc'] for r in valid])),
                local_auc=float(np.mean([r['local_auc'] for r in valid])),delta_mean=float(delta.mean()),delta_median=float(np.median(delta)),
                local_better=int((delta>0).sum()),whole_better=int((delta<0).sum()),ties=int((delta==0).sum()),
                delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
                sign_test_p=float(stats.binomtest(int((delta>0).sum()),int((delta!=0).sum()),.5).pvalue))
            for kind,fs in [('legal_observables',FEATURES),('GT_diagnostics',GT_FEATURES)]:
                for key in fs: correlations.append(dict(group=name,roi=roi,feature=key,kind=kind,**correlation(rows,key,kind,roi)))
    # Report all fixed-list hypotheses, without selecting a deployable formula.
    for name in groups:
        for roi in ('global','foris_foreground','new_unary_foreground'):
            for kind in ('legal_observables','GT_diagnostics'):
                vals=[c for c in correlations if c['group']==name and c['roi']==roi and c['kind']==kind and c['p'] is not None]
                p=np.array([v['p'] for v in vals]);q=stats.false_discovery_control(p,method='bh')
                for c,v in zip(vals,q):c['BH_q_within_fixed_feature_family']=float(v)
    strata=[]
    for feature,limits in [('ref_fg_fraction',[0,.05,.2,1.00001]),('ref_fg_mass_purity',[0,.5,.9,1.00001]),('view_rank_agreement',[-1,.5,.75,1.00001])]:
        for lo,hi in zip(limits[:-1],limits[1:]):
            for group,rows in groups.items():
                selected=[r for r in rows if lo<=r['legal_observables'][feature]<hi]
                for roi in ('foris_foreground','new_unary_foreground'):
                    vals=[r['rois'][roi]['delta_auc'] for r in selected if r['rois'][roi]['delta_auc'] is not None]
                    strata.append(dict(feature=feature,range_left_closed=[lo,hi],group=group,roi=roi,n=len(vals),delta=summary(vals),local_better=sum(v>0 for v in vals)))
    write(OUT/'summary.json',dict(cohort_stats=cohort_stats,main200=grouped,correlations=correlations,strata=strata,
        warning='Exploratory exposed-sample associations. Query GT appears only in GT_diagnostics and outcome; no fitted view selector or threshold. Independent confirmation not claimed.',
        aggregate_script_sha256=sha(__file__)))
    print(json.dumps(dict(stage='aggregate',main200={k:v['rois'] for k,v in grouped.items()})),flush=True)

if __name__=='__main__':
    import sys
    if len(sys.argv)<2 or sys.argv[1]=='collect':collect()
    aggregate()
