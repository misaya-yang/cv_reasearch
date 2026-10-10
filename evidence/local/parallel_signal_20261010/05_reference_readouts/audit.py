"""Read-only audit of sealed reference-readout evidence; no fits or predictions."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
torch.set_num_threads(1)

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
DATA = REPO.parent / 'cv_data'
PILOT = DATA / 'a/joint_role_pilot200_20261010'
STUDY = REPO / 'evidence/local/difficult_region_signal_20261010'
OLD = REPO / 'evidence/local/cpu100_20261006/server'
DRAWS = 4000
SEED = 20261010

def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, x): Path(path).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def qstats(values):
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                p10=float(np.quantile(v,.1)), median=float(np.median(v)), p90=float(np.quantile(v,.9)))

def array_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode()); h.update(x.tobytes())
    return h.hexdigest()

def load_mask(row, role):
    with Image.open(row[role+'_mask_path']) as im: x=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert array_sha(x) == row[role+'_mask_hash']
    return F.interpolate(torch.from_numpy(x.copy()).double()[None,None], (1024,1024), mode='nearest')[0,0].numpy().astype(bool)

def rank_metrics(score, f, b):
    """Independent tie-group weighted rank implementation, descending order."""
    s=np.asarray(score).ravel(); keep=(f+b)>0
    assert np.isfinite(s[keep]).all()
    s,f,b=s[keep],f[keep],b[keep]; nf,nb=float(f.sum()),float(b.sum())
    if not nf or not nb:return dict(auc=None,ap=None,tpr_at_fpr05=None)
    order=np.argsort(-s,kind='stable');s,f,b=s[order],f[order],b[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])]
    ff,bb=np.add.reduceat(f,starts),np.add.reduceat(b,starts)
    tp,fp=np.cumsum(ff),np.cumsum(bb)
    # Pair each positive score with lower-ranked negatives plus half the ties.
    auc=np.sum(ff*(nb-fp+.5*bb))/(nf*nb)
    ap=np.sum(ff*tp/(tp+fp))/nf
    tpr=np.max(np.r_[0.,tp[fp/nb<=.05]/nf])
    return dict(auc=float(auc),ap=float(ap),tpr_at_fpr05=float(tpr))

def components(rows):
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    photos={}
    for i,r in enumerate(rows):
        for role in ('reference','query'):
            p=(r['dataset'],str(r[role+'_photo_id']))
            if p in photos:parent[root(i)]=root(photos[p])
            else:photos[p]=i
    return [root(i) for i in range(len(rows))]

def cluster_ci(values, groups, mode='pooled_episodes'):
    v=np.asarray(values,float); groups=list(groups)
    unique=list(dict.fromkeys(groups));index={g:i for i,g in enumerate(unique)}
    sums=np.zeros(len(unique));counts=np.zeros(len(unique))
    for x,g in zip(v,groups):sums[index[g]]+=x;counts[index[g]]+=1
    means=sums/counts
    rng=np.random.default_rng(SEED)
    ix=rng.integers(0,len(unique),size=(DRAWS,len(unique)))
    draws=means[ix].mean(1) if mode=='equal_clusters' else sums[ix].sum(1)/counts[ix].sum(1)
    return dict(n_clusters=len(unique),ci95=np.quantile(draws,[.025,.975]).tolist(),
                estimate=float(means.mean() if mode=='equal_clusters' else v.mean()),mode=mode)

def pairing(a,b,rows,metric):
    valid=[i for i,(x,y) in enumerate(zip(a,b)) if x[metric] is not None and y[metric] is not None]
    aa=np.array([a[i][metric] for i in valid]);bb=np.array([b[i][metric] for i in valid]);d=aa-bb
    if not len(d):return dict(n=0)
    grp=components(rows)
    class_grp=[(r['fold'],r['loader_class_id']) for r in rows]
    weights=np.array([a[i]['fg']*a[i]['bg'] if metric=='auc' else a[i]['fg'] for i in valid])
    result=dict(n=len(d),left_mean=float(aa.mean()),right_mean=float(bb.mean()),mean_delta=float(d.mean()),
        median_delta=float(np.median(d)),p10_delta=float(np.quantile(d,.1)),p90_delta=float(np.quantile(d,.9)),
        wins=int((d>1e-12).sum()),losses=int((d < -1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),
        paired_episode=cluster_ci(d,range(len(d))),
        paired_photo_components=cluster_ci(d,[grp[i] for i in valid]),
        paired_class=cluster_ci(d,[class_grp[i] for i in valid]),
        equal_class=cluster_ci(d,[class_grp[i] for i in valid],mode='equal_clusters'),
        within_episode_mass_weighted_delta=float(np.average(d,weights=weights)),
        mass_weight='fg*bg' if metric=='auc' else 'fg',
        worst=[dict(episode_id=rows[valid[k]]['episode_id'],delta=float(d[k])) for k in np.argsort(d)[:5]],
        best=[dict(episode_id=rows[valid[k]]['episode_id'],delta=float(d[k])) for k in np.argsort(-d)[:5]])
    return result

def old_audit():
    s=read(OLD/'screen12_native200_v3/methods_summary.json')
    r=read(OLD/'screen12_native200_v3/screen12_native200_v3/score/report.json')
    results={}
    for method,v in r['class_actions_vs_dino_prototype'].items():
        score=100*np.mean([x['final_I']/x['final_U'] for x in v.values()])
        base=100*np.mean([x['baseline_I']/x['baseline_U'] for x in v.values()])
        assert abs(score-r['scores'][method]) < 1e-10
        assert abs(base-s['baseline_score']) < 1e-10
        results[method]=dict(score=float(score),n_classes=len(v),gain_vs_prototype=r['gain_vs_dino_prototype'][method],
                             actions=r['actions_vs_dino_prototype'][method])
    huber=r['class_actions_vs_dino_prototype']['inv_huber_reference_readout']
    ridge=r['class_actions_vs_dino_prototype']['inv_huber_ridge']
    ds=[100*(huber[c]['final_I']/huber[c]['final_U']-ridge[c]['final_I']/ridge[c]['final_U']) for c in huber]
    hist=STUDY/'raw/frozen/historical_invariance_support.py'
    assert sha(hist)==r['method_source_hashes']['src/ics/cpu100/invariance_support.py']
    return dict(n=s['n'],n_classes=s['classes'],n_methods=s['actual_method_count'],n_scored_arms=len(r['scores']),
        exposures=s['exposure'],source_verified=sha(hist),bootstrap=r['bootstrap'],results=results,
        huber_vs_ridge_class_deltas=qstats(ds),huber_vs_ridge_class_wins=int(np.sum(np.array(ds)>0)),
        huber_vs_ridge_class_losses=int(np.sum(np.array(ds)<0)),
        huber_vs_ridge_ci=r['gain_vs_controls']['inv_huber_reference_readout']['inv_huber_ridge'],
        all_methods_vs_strongest=[{k:v for k,v in row.items() if k in ('id','score','best_matched_control_on_this_probe','gain_vs_best_matched_control','ci95_vs_best_control')} for row in s['rows']])

def main():
    start=time.monotonic(); manifest=read(PILOT/'manifest.json')
    raw={x['episode_id']:x for x in lines(STUDY/'raw/episodes.jsonl')}
    old={x['episode_id']:x for x in lines(STUDY/'existing/episodes.jsonl')}
    raw_seal={x['episode_id']:x for x in read(STUDY/'raw/sealed.json')['receipts']}
    old_seal={x['episode_id']:x for x in read(STUDY/'existing/sealed.json')['receipts']}
    infer={x['episode_id']:x for x in lines(PILOT/'inference.jsonl')}
    assert len(manifest)==len(raw)==len(old)==200
    checked=0; max_error=0.; coverage_rows=[]; verify_fields=0
    for i,row in enumerate(manifest):
        eid=row['episode_id'];rr=raw[eid];oo=old[eid];rec=raw_seal[eid]
        assert (rr['dataset'],rr['fold'],rr['class_id'])==(row['dataset'],row['fold'],row['loader_class_id'])
        assert (rr['dataset'],rr['fold'],rr['class_id'])==(oo['dataset'],oo['fold'],oo['class_id'])
        fields={}
        for stage,seal,keys in [('raw',raw_seal,['source_apd.ridge.global','source_apd.ridge.local4']),('existing',old_seal,['P0','foris_source'])]:
            p=STUDY/stage/'fields'/seal[eid]['filename'];assert sha(p)==seal[eid]['sha256'];verify_fields+=1
            with np.load(p,allow_pickle=False) as z:
                for k in keys:fields[k]=z[k].copy()
        pred=PILOT/'predictions'/infer[eid]['filename'];assert sha(pred)==infer[eid]['prediction_sha256']
        with np.load(pred,allow_pickle=False) as z:
            base=np.unpackbits(z['cli1024/foris.crf'],count=1024**2).reshape(1024,1024).astype(bool)
            unary=np.unpackbits(z['cli1024/role.unary'],count=1024**2).reshape(1024,1024).astype(bool)
        truth=load_mask(row,'query')
        rois=dict(global_=np.ones((1024,1024),bool),foris_foreground=base,new_unary_foreground=unary & ~base)
        for roi,region in rois.items():
            if roi=='global_':roi='global'
            f=(region & truth).reshape(128,8,128,8).sum((1,3)).ravel().astype(float)
            b=(region & ~truth).reshape(128,8,128,8).sum((1,3)).ravel().astype(float)
            for sig,score in fields.items():
                got=rank_metrics(score,f,b);want=(rr if sig.startswith('source_apd') else oo)['rois'][roi][sig]
                assert f.sum()==want['fg'] and b.sum()==want['bg']
                for k,x in got.items():
                    if x is None:assert want[k] is None
                    else:
                        error=abs(x-want[k]);max_error=max(max_error,error);assert error<1e-12
                    checked+=1
            for sig,x in rr['rois'][roi].items():
                assert x['fg']==f.sum() and x['bg']==b.sum() and x['excluded_fg']==x['excluded_bg']==0
        c=load_mask(row,'reference').reshape(64,16,64,16).mean((1,3)).ravel()
        assert c.sum()>0 and (1-c).sum()>0
        ids=[]
        for role in (c,1-c):
            cumulative=np.cumsum(role);ids.extend(np.searchsorted(cumulative,(np.arange(128)+.5)*cumulative[-1]/128,side='left'))
        ids=np.unique(ids);held=np.ones(len(c),bool);held[ids]=False
        train=c[ids];test=c[held]
        assert len(ids)==rec['fit']['raw']['fit_tokens']
        assert abs(c.mean()-rec['fit']['raw']['coverage_mean'])<1e-15
        yy=2*train-1;ww=.5*train/train.sum()+.5*(1-train)/(1-train).sum()
        target_role=(train/train.sum()-(1-train)/(1-train).sum())/(train/train.sum()+(1-train)/(1-train).sum())
        cy,cx=np.divmod(np.flatnonzero(held),64);ty,tx=np.divmod(ids,64)
        nearest=np.min(np.maximum(abs(cy[:,None]-ty[None,:]),abs(cx[:,None]-tx[None,:])),axis=1)
        cr=dict(episode_id=eid,dataset=row['dataset'],ref_coverage=float(c.mean()),
            apd_applied=rec['apd_applied'],semantic=rec['source_semantic_score'],
            reference_max_coverage=float(c.max()),fit_n=len(ids),holdout_n=int(held.sum()),
            fit_fg_mass=float(train.sum()),holdout_fg_mass=float(test.sum()),full_fg_mass=float(c.sum()),
            holdout_fg_mass_fraction=float(test.sum()/c.sum()),
            fit_fg_coverage=float(train.mean()),holdout_fg_coverage=float(test.mean()),
            fit_max_coverage=float(train.max()),holdout_max_coverage=float(test.max()),
            full_majority_fg=int((c>.5).sum()),fit_majority_fg=int((train>.5).sum()),holdout_majority_fg=int((test>.5).sum()),
            full_high_fg=int((c>=.9).sum()),fit_high_fg=int((train>=.9).sum()),holdout_high_fg=int((test>=.9).sum()),
            fit_target_mean=float(np.sum(ww*yy)),fit_weight_mass_on_positive_target=float(ww[yy>0].sum()),
            pure_fg_target_mass_fraction=float(c[c==1].sum()/c.sum()),
            majority_fg_target_mass_fraction=float(c[c>.5].sum()/c.sum()),
            mixed_target_mass_fraction=float(c[(c>0)&(c<1)].sum()/c.sum()),
            label_half_hardening_loses_fg_mass=float(c[c<=.5].sum()/c.sum()),
            balanced_binary_equivalent_target_abs_gap=float(np.average(abs(target_role-yy),weights=ww)),
            holdout_fg_mass_within_one_patch_of_train=float(test[nearest<=1].sum()/test.sum()) if test.sum() else None,
            holdout_all_patches_within_one_patch_of_train=float(np.mean(nearest<=1)),
            full_fg_pixels=int(c.sum()*256),half_hard_fg_pixels=int((c>.5).sum()*256),
            zero_output_global=bool((fields['source_apd.ridge.global']<=0).all()),
            zero_output_local=bool((fields['source_apd.ridge.local4']<=0).all()))
        coverage_rows.append(cr)
    comparisons={};duplicates={};coverage={};validity={};boundary={}
    pairs=[]
    for view in ('global','local4'):
        sig=f'source_apd.ridge.{view}'
        pairs.extend([(sig,ctl) for ctl in ('P0','foris_source','whole_competition',f'raw.ridge.{view}',f'source_apd.prototype.{view}')])
        pairs.append((f'source_apd.huber.{view}',sig))
    pairs.append(('source_apd.ridge.local4','source_apd.ridge.global'))
    for dataset in ('deepglobe_road','paco_part'):
        rows=[r for r in manifest if r['dataset']==dataset];ids=[r['episode_id'] for r in rows]
        cs=[r for r in coverage_rows if r['dataset']==dataset];ci={r['episode_id']:r for r in cs}
        groups=components(rows);sizes=Counter(groups)
        ref_counts=Counter(r['reference_photo_id'] for r in rows);query_counts=Counter(r['query_photo_id'] for r in rows)
        duplicates[dataset]=dict(n_episodes=len(rows),query_photo_unique=len(query_counts),reference_photo_unique=len(ref_counts),
            distinct_photos_both_roles=len(set(ref_counts)|set(query_counts)),
            within_episode_same_photo=sum(r['query_photo_id']==r['reference_photo_id'] for r in rows),
            unique_query_rgb=len(set(r['query_rgb_hash'] for r in rows)),unique_reference_rgb=len(set(r['reference_rgb_hash'] for r in rows)),
            unique_reference_input_and_mask=len(set((r['reference_rgb_hash'],r['reference_mask_hash']) for r in rows)),
            n_photo_components=len(sizes),max_photo_component=max(sizes.values()),photo_component_sizes=sorted(sizes.values(),reverse=True),
            n_fold_class=len(set((r['fold'],r['loader_class_id']) for r in rows)),
            fold_class_counts=dict(Counter(str(r['fold']) for r in {(r['fold'],r['loader_class_id']):r for r in rows}.values())),
            repeated_query_photos={p:n for p,n in query_counts.items() if n>1},
            repeated_reference_photos={p:n for p,n in ref_counts.items() if n>1})
        metrics=['ref_coverage','reference_max_coverage','fit_n','holdout_fg_mass_fraction','holdout_fg_coverage','holdout_max_coverage',
            'fit_target_mean','fit_weight_mass_on_positive_target','mixed_target_mass_fraction','label_half_hardening_loses_fg_mass',
            'balanced_binary_equivalent_target_abs_gap','holdout_fg_mass_within_one_patch_of_train','holdout_all_patches_within_one_patch_of_train']
        coverage[dataset]={k:qstats([c[k] for c in cs if c[k] is not None]) for k in metrics}
        coverage[dataset]['counts']={k:sum(c[k] for c in cs) for k in ('zero_output_global','zero_output_local','apd_applied')}
        for k in ['full_majority_fg','fit_majority_fg','holdout_majority_fg','full_high_fg','fit_high_fg','holdout_high_fg','holdout_fg_mass']:
            coverage[dataset]['counts'][k+'_zero']=sum(c[k]==0 for c in cs)
        coverage[dataset]['pooled_fg_mass_lost_by_half_hardening']=sum(c['label_half_hardening_loses_fg_mass']*c['full_fg_mass'] for c in cs)/sum(c['full_fg_mass'] for c in cs)
        for roi in ('global','foris_foreground','new_unary_foreground'):
            values={eid:{**raw[eid]['rois'][roi],**old[eid]['rois'][roi]} for eid in ids}
            validity[dataset+'/'+roi]=dict(n=len(rows),n_valid=sum(values[eid]['P0']['auc'] is not None for eid in ids),
                no_fg=[eid for eid in ids if not values[eid]['P0']['fg']],no_bg=[eid for eid in ids if not values[eid]['P0']['bg']])
            for a,b in pairs:
                for metric in ('auc','ap','tpr_at_fpr05'):
                    comparisons['/'.join((dataset,roi,a+'__minus__'+b,metric))]=pairing([values[eid][a] for eid in ids],[values[eid][b] for eid in ids],rows,metric)
            # Descriptive bounds chosen by legal reference coverage and existing APD branch; no rule fitted.
            median=np.median([c['ref_coverage'] for c in cs])
            group_defs={'reference_area_low_half':lambda c:c['ref_coverage']<=median,'reference_area_high_half':lambda c:c['ref_coverage']>median,
                        'reference_has_no_majority_FG':lambda c:c['full_majority_fg']==0,
                        'APD_on':lambda c:c['apd_applied'],'APD_off':lambda c:not c['apd_applied']}
            for group,filt in group_defs.items():
                eids=[eid for eid in ids if filt(ci[eid])]
                d=[values[eid]['source_apd.ridge.local4']['auc']-values[eid]['source_apd.ridge.global']['auc'] for eid in eids if values[eid]['source_apd.ridge.global']['auc'] is not None]
                boundary['/'.join((dataset,roi,group))]=dict(n=len(d),local_minus_global_AUC=qstats(d) if d else None,reference_area_split=float(median))
    inputs=[PILOT/'manifest.json',PILOT/'inference.jsonl',STUDY/'raw/episodes.jsonl',STUDY/'existing/episodes.jsonl',
            STUDY/'raw/sealed.json',STUDY/'existing/sealed.json',STUDY/'raw/frozen/historical_invariance_support.py',
            OLD/'screen12_native200_v3/methods_summary.json',OLD/'screen12_native200_v3/screen12_native200_v3/score/report.json']
    result=dict(scope='fixed exposed Deep100/PACO100; conditional fine128 rank, not original mIoU',
        bootstrap=dict(seed=SEED,draws=DRAWS,note='descriptive paired diagnostics after exposure; no multiplicity adjustment or independent confirmation'),
        input_sha256={str(p):sha(p) for p in inputs},script_sha256=sha(__file__),
        independent_recalculation=dict(field_files_verified=verify_fields,prediction_files_verified=len(manifest),reference_masks_verified=len(manifest),
            query_masks_verified=len(manifest),rank_values_checked=checked,max_absolute_error=max_error,threads=1,new_encodings=0,new_predictions=0,new_fits=0),
        duplicates=duplicates,validity=validity,coverage=coverage,comparisons=comparisons,boundary=boundary,old_native200=old_audit(),
        elapsed_seconds=time.monotonic()-start)
    write(OUT/'audit.json',result)
    (OUT/'coverage_episodes.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in coverage_rows))
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'],verified=result['independent_recalculation'],duplicate_summary={k:{x:v[x] for x in ['n_episodes','query_photo_unique','reference_photo_unique','n_photo_components','max_photo_component','n_fold_class']} for k,v in duplicates.items()})))

if __name__=='__main__':main()
