#!/usr/bin/env python3
"""CPU-only, photo-purged leave-one-class-out donor scoring on SAVED policies.

This is a policy-restricted supervised measuring device, not arbitrary subset
reconstruction, a full information upper bound, or a new segmentation run.
No image, feature cache, GPU, or model is read. Query native I/U is copied only
from precomputed GT-free policies; pool-easy is a frozen diagnostic anchor.

Manifest: {"records":[{"e":0,"c":72,"image_ids":["support","query","donor1",...]}]}.
image_ids must follow EXACT saved donor order; parent reconstructs that order
with the original seed-0 gallery. Same-photo IDs across roles must be identical.
Training drops the entire episode if ANY support/query/donor photo appears in
ANY held-class episode, because donor features depend on the complete episode.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

# CPU fits are small; cap native BLAS/OpenMP threads before importing numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np

FEATURES = ('link', 'agree', 'fmsg', 'fmsb', 'marg', 'area')
POLICIES = ('pool', '1shot', 'keep-link', 'keep-agree', 'keep-fmsg', 'keep-fmsb',
            'keep-marg', 'keep-link+agree+marg', 'link>=0.5')
ANCHORS = ('1shot', 'pool', 'keep-agree', 'pool-easy')
SEED = 2044


def photo_id(value):
    """Only path and extension normalization; no guessed semantic/numeric IDs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every manifest image ID must be a nonempty string')
    return Path(value).stem


def feature_matrix(pool):
    """No donor quality or query I/U enters the features."""
    x = np.column_stack([pool[key] for key in FEATURES]).astype(float)
    return np.column_stack((x, x[:, 5]**2))


def policy_memberships(pool):
    """Exact root probe_reliab policies, including its two distinct tie orders."""
    n = len(pool['link']); nk = max(1, int(round(.62*n)))
    masks = {'pool':np.ones(n, bool), '1shot':np.zeros(n, bool)}
    for key in FEATURES[:5]:
        # root: sorted(o, key=lambda y:-f[key][y]); Python stable ties.
        ids = sorted(range(n), key=lambda i:-pool[key][i])[:nk]
        masks['keep-'+key] = np.zeros(n, bool); masks['keep-'+key][ids] = True
    z = {key:(np.asarray(pool[key])-np.mean(pool[key]))/(np.std(pool[key])+1e-6)
         for key in ('link', 'agree', 'marg')}
    ids = np.argsort(-(z['link']+z['agree']+z['marg']))[:nk]
    masks['keep-link+agree+marg'] = np.zeros(n, bool)
    masks['keep-link+agree+marg'][ids] = True
    masks['link>=0.5'] = np.asarray(pool['link']) >= .5
    return masks


def nominate_policy(probabilities, pool):
    """Only donor easy probabilities and GT-free membership; never saved I/U."""
    p = np.clip(np.asarray(probabilities, float), 1e-6, 1-1e-6)
    masks = policy_memberships(pool); scores = {}
    for policy in POLICIES:
        g = masks[policy]
        scores[policy] = float(np.sum(np.where(g, np.log(p), np.log1p(-p))))
    # Exact score ties prefer pool, then 1shot, then the listed frozen order.
    choice = max(POLICIES, key=lambda policy:scores[policy])
    return choice, masks[choice], scores


def load_inputs(result_file, manifest_file):
    source = json.loads(Path(result_file).read_text())
    manifest = json.loads(Path(manifest_file).read_text())
    table = {}; records = source['records']; checked = []
    for row in manifest['records']:
        key = int(row['e'])
        if key in table: raise ValueError(f'Duplicate manifest episode {key}')
        ids = [photo_id(value) for value in row['image_ids']]
        table[key] = dict(c=int(row['c']), ids=ids, raw_ids=row['image_ids'])
    seen = set()
    for record in records:
        e, c = int(record['e']), int(record['c'])
        if e in seen: raise ValueError(f'Duplicate saved episode {e}')
        seen.add(e)
        if e not in table or table[e]['c'] != c:
            raise ValueError(f'Missing/mismatched manifest class for episode {e}')
        if 'pool' not in record: raise ValueError(f'No saved donor features for episode {e}')
        n = len(record['pool']['qual'])
        if n < 4: raise ValueError(f'Episode {e} lacks the root probe complete-policy contract (N>=4)')
        if len(table[e]['ids']) != n+2:
            raise ValueError(f'Manifest episode {e}: expected support/query/{n} ordered donors')
        ids = table[e]['ids']
        if len(set(ids[2:])) != n or set(ids[2:]) & set(ids[:2]):
            raise ValueError(f'Manifest episode {e}: donors duplicate or equal support/query')
        for key in FEATURES+('qual',):
            values = np.asarray(record['pool'][key], float)
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError(f'Episode {e}: invalid {key} array')
        quality = np.asarray(record['pool']['qual'], float)
        if ((quality < 0)|(quality > 1)).any(): raise ValueError(f'Episode {e}: invalid donor quality')
        for policy in set(POLICIES+ANCHORS):
            if policy not in record['iu']: raise ValueError(f'Episode {e}: missing saved policy I/U: {policy}')
            pair = np.asarray(record['iu'][policy], float)
            if pair.shape != (2,) or not np.isfinite(pair).all() or not (0 <= pair[0] <= pair[1]):
                raise ValueError(f'Episode {e}: invalid native I/U for {policy}')
        checked.append(dict(e=e,c=c,pool=record['pool'],iu=record['iu'],image_ids=ids,
                            raw_image_ids=table[e]['raw_ids']))
    return source, checked


def class_split(records, held_class):
    held = [r for r in records if r['c']==held_class]
    held_ids = set(photo for r in held for photo in r['image_ids'])
    candidates = [r for r in records if r['c']!=held_class]
    training = []; removed = []
    for row in candidates:
        shared = sorted(set(row['image_ids']) & held_ids)
        if shared:
            removed.append(dict(e=row['e'],c=row['c'],shared_photo_ids=shared,
                                removed_episode_image_ids=row['image_ids']))
        else: training.append(row)
    if set(photo for r in training for photo in r['image_ids']) & held_ids:
        raise AssertionError('Photo purge failed')
    return training, held, dict(held_class=held_class,held_photo_ids=sorted(held_ids),
        held_episodes=len(held), held_donors=sum(len(r['pool']['qual']) for r in held),
        candidate_training_episodes=len(candidates),
        candidate_training_donors=sum(len(r['pool']['qual']) for r in candidates),
        removed_training_episodes=len(removed),
        removed_training_donors=sum(len(r['pool']['qual']) for r in candidates if any(x['e']==r['e'] for x in removed)),
        removed_training_photo_ids=sorted(set(p for r in removed for p in r['removed_episode_image_ids'])),
        removed=removed,training_episode_ids=[r['e'] for r in training],
        training_photo_ids=sorted(set(p for r in training for p in r['image_ids'])),
        training_classes=sorted(set(r['c'] for r in training)))


def make_models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    return dict(logistic=make_pipeline(StandardScaler(), LogisticRegression(C=1,max_iter=2000,random_state=SEED)),
        hgb=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05,
                l2_regularization=1,min_samples_leaf=20,random_state=SEED,early_stopping=False))


def donor_metrics(labels, probabilities):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels, bool); p = np.asarray(probabilities, float); pred = p>=.5
    positive, negative = int(y.sum()), int((~y).sum())
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())
    return dict(donors=len(y),easy_donors=positive,hard_donors=negative,
        auc=float(roc_auc_score(y,p)) if positive and negative else None,
        sensitivity=tp/positive if positive else None, specificity=(negative-fp)/negative if negative else None,
        predicted_easy_fraction=float(pred.mean()) if len(y) else None,
        easy_fraction=float(y.mean()) if len(y) else None,
        brier=float(np.mean((p-y)**2)) if len(y) else None,
        log_loss=float(-np.mean(np.where(y,np.log(np.clip(p,1e-6,1-1e-6)),np.log1p(-np.clip(p,1e-6,1-1e-6))))) if len(y) else None)


def class_miou(records, key):
    grouped = {}
    for r in records:
        i,u = r['iu'][key]; a=grouped.setdefault(r['c'],[0.,0.]);a[0]+=i;a[1]+=u
    per_class = {str(c):100*i/max(u,1) for c,(i,u) in sorted(grouped.items())}
    return dict(miou=float(np.mean(list(per_class.values()))) if grouped else None,per_class=per_class)


def paired_interval(records, baseline, candidate, reps=2000):
    if not records: return None
    classes = sorted(set(r['c'] for r in records)); rng=np.random.default_rng(SEED)
    arrays={c:np.array([[*r['iu'][baseline],*r['iu'][candidate]] for r in records if r['c']==c]) for c in classes}
    delta=[]
    for _ in range(reps):
        values=[]
        for c in classes:
            data=arrays[c]; draws=data[rng.integers(0,len(data),len(data))].sum(0)
            values.append(draws[2]/max(draws[3],1)-draws[0]/max(draws[1],1))
        delta.append(100*np.mean(values))
    return dict(delta_pp=class_miou(records,candidate)['miou']-class_miou(records,baseline)['miou'],
                ci95_pp=np.quantile(delta,[.025,.975]).tolist(),
                procedure='2000 within-class paired episode resamples; frozen cross-fit predictions, not refitted; single-fold descriptive interval')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file',default='results/probe_reliab_f0_400.json')
    ap.add_argument('--manifest');ap.add_argument('--out');ap.add_argument('--self-check',action='store_true')
    a=ap.parse_args()
    if a.self_check: self_check();return
    if not a.manifest or not a.out: ap.error('--manifest and --out are required')
    source,records=load_inputs(a.file,a.manifest)
    if float(source.get('args',{}).get('keep',.62)) != .62:
        raise ValueError('Saved probe must use frozen keep=.62')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    sentinel_count=sum(sum(all(r['pool'][key][i]==-1 for key in FEATURES[:5]) and r['pool']['area'][i]==-9
                           for i in range(len(r['pool']['qual']))) for r in records)
    report=dict(state='RUNNING',args=vars(a),fold=0,dataset='COCO-20i',episodes=len(records),
        numpy_version=np.__version__,empty_p1_sentinel_donors=sentinel_count,
        feature_names=list(FEATURES)+['area_squared'], models=dict(
            logistic='Training-only StandardScaler + LogisticRegression(C=1,max_iter=2000,random_state=2044)',
            hgb='HistGradientBoostingClassifier(depth=3,iter=200,lr=.05,l2=1,leaf=20,seed=2044,early_stopping=False)'),
        contract=dict(labels='Cached donor patch-mask IoU >=.5; patch quality, not query task quality',
            split='Leave one class out; purge entire training episode if any role photo overlaps any held episode',
            photo_id='Path basename without extension; manifest must already identify equivalent photos consistently',
            manifest_order='support,query,donors in original probe order; externally reconstructed, not independently verified from this JSON',
            policies=list(POLICIES),policy_keep='Top max(1,round(.62*N)); feature ties Python-stable, combo np.argsort default exactly root',
            policy_choice='Sum kept*log(p_easy)+(1-kept)*log(1-p_easy), clip [1e-6,1-1e-6]; exact ties frozen list prefers pool then 1shot',
            task_metric='Query native I/U copied from saved GT-free policy; no arbitrary predicted subset I/U reconstruction',
            anchors=list(ANCHORS),oracle_use='pool-easy only diagnostic anchor, never a selectable policy; two-stage policies not selectable',
            interpretation='Supervised nominee/policy-restricted measuring device; not a complete evidence upper bound. No hyperparameter search.',
            empty_p1='Sentinel features retained; empty P1 does not establish semantic absence',
            dependence='Held class/photo purge prevents training reuse; donor rows share episodes and test galleries. One fold, no independent final method score.'),
        source_sha256={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.file,a.manifest,__file__]},
        splits=[],model_results={name:dict(records=[],class_metrics=[]) for name in ('logistic','hgb')})
    def write():
        report['elapsed_s']=time.time()-start
        tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(out)
    write()
    try:
        try:
            from threadpoolctl import threadpool_limits
            fit_context=lambda:threadpool_limits(limits=1)
        except ImportError: fit_context=nullcontext
        import sklearn
        report['sklearn_version']=sklearn.__version__
        held_classes=sorted(set(r['c'] for r in records))
        for step,c in enumerate(held_classes,1):
            training,held,split=class_split(records,c);report['splits'].append(split)
            if not training:
                split['state']='NO_TRAINING_AFTER_PURGE';write();print(f'class {c}: no training after purge',flush=True);continue
            x=np.concatenate([feature_matrix(r['pool']) for r in training]);y=np.concatenate([np.asarray(r['pool']['qual'])>=.5 for r in training])
            split.update(training_donors=len(y),training_easy_donors=int(y.sum()),training_hard_donors=int((~y).sum()))
            if len(np.unique(y))<2:
                split['state']='ONE_TRAINING_LABEL_AFTER_PURGE';write();print(f'class {c}: one training label after purge',flush=True);continue
            models=make_models();split['state']='FITTING';write()
            for name,model in models.items():
                fit_start=time.time()
                with fit_context(): model.fit(x,y)
                output=report['model_results'][name];all_y=[];all_p=[]
                for row in held:
                    probabilities=model.predict_proba(feature_matrix(row['pool']))[:,1]
                    choice,kept,scores=nominate_policy(probabilities,row['pool'])
                    # Labels/native I/U are read ONLY after the nomination was frozen.
                    quality=np.asarray(row['pool']['qual']);truth=quality>=.5
                    all_y.extend(truth.tolist());all_p.extend(probabilities.tolist())
                    iu={anchor:row['iu'][anchor] for anchor in ANCHORS};iu['nominee']=row['iu'][choice]
                    output['records'].append(dict(e=row['e'],c=row['c'],donor_photo_ids=row['image_ids'][2:],
                        donor_easy_probabilities=probabilities.tolist(),donor_easy_labels=truth.tolist(),
                        chosen_policy=choice,chosen_donor_indices=np.flatnonzero(kept).tolist(),
                        chosen_donor_photo_ids=[photo for photo,keep in zip(row['image_ids'][2:],kept) if keep],
                        membership_likelihood=scores,iu=iu))
                metrics=donor_metrics(all_y,all_p);metrics.update(c=c,fit_s=time.time()-fit_start,training_donors=len(y),
                    removed_training_episodes=split['removed_training_episodes'],held_episodes=len(held))
                output['class_metrics'].append(metrics)
            split['state']='COMPLETED';write()
            print(f'{step}/{len(held_classes)} held class {c}; train {len(y)} donors, purge {split["removed_training_episodes"]} episodes; '+
                  ' | '.join(f'{name} AUC {report["model_results"][name]["class_metrics"][-1]["auc"]}' for name in models),flush=True)
        for name,output in report['model_results'].items():
            scored=output['records'];ys=[v for r in scored for v in r['donor_easy_labels']];ps=[v for r in scored for v in r['donor_easy_probabilities']]
            output['donor_metrics']=donor_metrics(ys,ps)
            sensitivities=[r['sensitivity'] for r in output['class_metrics'] if r['sensitivity'] is not None]
            aucs=[r['auc'] for r in output['class_metrics'] if r['auc'] is not None]
            output.update(macro_class_sensitivity=float(np.mean(sensitivities)) if sensitivities else None,
                          macro_class_auc=float(np.mean(aucs)) if aucs else None,
                          coverage=dict(episodes=len(scored),total_episodes=len(records),episode_fraction=len(scored)/max(len(records),1),
                                        classes=len(output['class_metrics']),total_classes=len(set(r['c'] for r in records)),
                                        donors=len(ys),total_donors=sum(len(r['pool']['qual']) for r in records)),
                          native_query={key:class_miou(scored,key) for key in ANCHORS+('nominee',)},
                          paired_native={key:paired_interval(scored,key,'nominee') for key in ANCHORS},
                          chosen_policy_counts={key:sum(r['chosen_policy']==key for r in scored) for key in POLICIES})
        report['state']='COMPLETED';write()
        print(json.dumps({name:{key:output[key] for key in ('donor_metrics','coverage','native_query','chosen_policy_counts')}
                          for name,output in report['model_results'].items()}),flush=True)
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex));write();raise


def self_check():
    pool={key:[.1,.1,.8,.9] for key in FEATURES};pool['qual']=[0,0,1,1]
    masks=policy_memberships(pool)
    assert masks['keep-link'].tolist()==[False,False,True,True]
    assert masks['link>=0.5'].tolist()==[False,False,True,True]
    assert feature_matrix(pool).shape==(4,7)
    assert nominate_policy([.9]*4,pool)[0]=='pool'
    assert nominate_policy([.1]*4,pool)[0]=='1shot'
    assert nominate_policy([.5]*4,pool)[0]=='pool' # exact tie preference
    rows=[dict(e=0,c=0,image_ids=['s0','q0','shared']),
          dict(e=1,c=1,image_ids=['shared','q1','d1']), # donor -> support role leak
          dict(e=2,c=1,image_ids=['s2','q2','d2']),
          dict(e=3,c=2,image_ids=['s3','q3','q0'])] # query -> donor role leak
    for r in rows:r['pool']=pool
    train,test,split=class_split(rows,0)
    assert [r['e'] for r in train]==[2] and split['removed_training_episodes']==2
    assert set(split['training_photo_ids']).isdisjoint(split['held_photo_ids'])
    assert photo_id('/data/image.PNG')==photo_id('/mask/image.png')
    scored=[dict(c=0,iu={'1shot':[2,4],'nominee':[3,4]}),dict(c=1,iu={'1shot':[1,2],'nominee':[1,2]})]
    assert class_miou(scored,'1shot')['miou']==50
    assert paired_interval(scored,'1shot','nominee',20)['delta_pp']==12.5
    assert 'pool-easy' not in masks and 'two-stage-oracle' not in masks
    print('CPU self-check passed: exact policy memberships/ties, seven label-free features, cross-role photo purge, native class-IoU and paired interval, oracle excluded from nomination.')


if __name__=='__main__':main()
