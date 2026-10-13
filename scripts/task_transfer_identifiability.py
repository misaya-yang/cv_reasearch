"""Separate within-image decodability, cross-image transfer and shared-rule capacity.

This is explicitly an ORACLE DIAGNOSTIC, never a segmentation submission.
Reference-only fields are generated/sealed before query labels are opened.
Subsequent spatially held-out query and joint fits use query labels solely to
identify which limitation warrants an independent algorithm experiment.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import opus55_reference_mask_trace as base
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from raw_feature_cache import tensor_hash

OUT = base.ASSETS/'a/task_transfer_identifiability60_20261010'
REPRESENTATIONS = ('unit_raw', 'fixed_position_projection')
LAMBDA = .01
SAMPLES_PER_ROLE = 128
BLOCK = 8
torch.set_num_threads(2)


def read(p):
    return json.loads(Path(p).read_text())


def coverage(path, digest):
    assert base.sha(path)==digest
    a = np.asarray(Image.open(path).convert('L'))>0
    t = F.interpolate(torch.from_numpy(a.astype('float32'))[None,None], (1024,1024), mode='nearest')[0,0]
    return F.avg_pool2d(t[None,None],16)[0,0].numpy().reshape(-1)


def load_raw(pair):
    result, receipts = [], []
    for role in ('reference','query'):
        binding=pair['raw_binding'][role]
        assert base.sha(binding['entry'])==binding['entry_sha256']
        entry=read(binding['entry']); path=Path(binding['entry']).parent/entry['file']
        assert base.sha(path)==entry['file_sha256']
        assert entry['input_tensor_hash']==binding['input_tensor_hash']
        with np.load(path,allow_pickle=False) as z: a=z['O/24'].copy()
        assert a.dtype==np.float32 and a.shape==(4096,1024)
        assert tensor_hash(a)==entry['features']['O/24']['tensor_sha256']
        result.append(F.normalize(torch.from_numpy(a),dim=1))
        receipts.append(dict(role=role,key=binding['key'],payload_sha256=entry['file_sha256'],
                             array_sha256=tensor_hash(a)))
    return result,receipts


def split_grid():
    y,x=np.mgrid[:64,:64]
    return (((y//BLOCK+x//BLOCK)%2).reshape(-1)).astype(np.int64)


def sample(x,c,allowed):
    """Balanced binary pixel risk, compressed by deterministic role quadrature.

    Mixed tokens may appear in both roles; duplicate positions are collapsed
    with their exact occurrence weights, not hard-assigned or counted twice.
    """
    pool=np.flatnonzero(allowed)
    fg=c[pool].astype('float64'); bg=1-fg
    if fg.sum()==0 or bg.sum()==0:
        return None
    occurrence=[]
    for role in (fg,bg):
        cumulative=np.cumsum(role)
        occurrence.append(pool[np.searchsorted(cumulative,(np.arange(SAMPLES_PER_ROLE)+.5)*cumulative[-1]/SAMPLES_PER_ROLE)])
    ids=np.unique(np.concatenate(occurrence))
    nf=np.bincount(np.searchsorted(ids,occurrence[0]),minlength=len(ids))
    nb=np.bincount(np.searchsorted(ids,occurrence[1]),minlength=len(ids))
    a=(nf+nb)/(2*SAMPLES_PER_ROLE)
    y=(nf-nb)/(nf+nb)
    return x[ids],torch.from_numpy(y),torch.from_numpy(a),ids


@torch.inference_mode()
def fit(samples, validate=False):
    """GPU Gram/readout, CPU FP64 small solve; fixed classical ridge diagnostic."""
    if any(s is None for s in samples):
        return None
    x=torch.cat([s[0] for s in samples]).double()
    y=torch.cat([s[1] for s in samples])
    a=torch.cat([s[2]/len(samples) for s in samples])
    mx=(a[:,None]*x).sum(0); my=(a*y).sum()
    xc=x-mx;yc=y-my
    gx=xc.float().to('mps')
    gram=(gx@gx.T).cpu().double()
    matrix=gram+torch.diag(LAMBDA/a)
    dual=torch.linalg.solve(matrix,yc)
    w=xc.T@dual;b=my-mx@w
    residual=float((matrix@dual-yc).norm()/yc.norm().clamp_min(1e-15))
    check=None
    if validate:
        exact=xc@xc.T+torch.diag(LAMBDA/a)
        exact_w=xc.T@torch.linalg.solve(exact,yc);exact_b=my-mx@exact_w
        error=float((w-exact_w).abs().max())
        check=dict(max_gram_error=float((gram-xc@xc.T).abs().max()),
                   max_coefficient_error=error,bias_error=float(abs(b-exact_b)),
                   sampled_score_error=float(((x@w+b)-(x@exact_w+exact_b)).abs().max()))
        assert check['sampled_score_error']<2e-5,check
    return dict(w=w,b=b,diagnostics=dict(sampled_positions=len(x),dual_residual=residual,numerical_check=check))


@torch.inference_mode()
def predict(fit_result,x):
    if fit_result is None:
        return None
    score=(x.to('mps')@fit_result['w'].float().to('mps')+fit_result['b'].float().to('mps')).cpu().numpy()
    assert np.isfinite(score).all()
    return score


def source_stage(out):
    pairs=read(base.OUT/'pairs.json')
    if (out/'source_sealed.json').exists():
        verify(out)
        return
    out.mkdir(parents=True,exist_ok=True)
    for name in ('source_fields','source_records','oracle_fields','oracle_records','frozen'):
        (out/name).mkdir(exist_ok=True)
    if not (out/'config.json').exists():
        shutil.copyfile(__file__,out/'frozen/task_transfer_identifiability.py')
        base.write(out/'pairs.json',pairs)
        base.write(out/'config.json',dict(n_pairs=60,real_tasks_per_pair=2,representations=REPRESENTATIONS,
            source_sha256=base.sha(__file__),pairs_sha256=base.sha(out/'pairs.json'),
            origin_pairs_sha256=base.sha(base.OUT/'pairs.json'),query_truth_sha256=base.sha(base.OUT/'query_truth.json'),
            basis_sha256=base.sha(base.ASSETS/'native_assets/positional_basis.pt'),
            lambda_=LAMBDA,samples_per_role=SAMPLES_PER_ROLE,spatial_split='two checkerboard folds of8x8-token blocks; fixed before labels',
            target='balanced binary pixel risk from continuous mask coverage; exact128 occurrences per role',
            purpose='Distinguish missing native representation, support-to-query transfer, and failure to identify an existing shared rule',
            primary_comparisons=['R->R versus R->Q','R->Q versus Q->Q','joint->Q versus Q->Q','joint->R versus R->R'],
            additional='R->Q with threshold fit on query training blocks: explicitly query-label oracle calibration',
            no_FoRIS=True,no_pseudo_masks=True,no_new_DINO=True,
            oracle_warning='Query and joint probes use query GT; they are diagnostics, never deployable methods or benchmark scores',
            limits='Within-image contextual dependence remains despite blocked holdout; these are exposed development tasks; finite ridge capacity is not an impossibility proof'))
    cfg=verify(out,require_seal=False)
    basis=torch.load(base.ASSETS/'native_assets/positional_basis.pt',map_location='cpu',weights_only=True)['basis'].float()
    projection=(torch.eye(1024)-basis@basis.T).to('mps')
    split=split_grid();records=[];start=time.monotonic()
    for i,pair in enumerate(pairs):
        rp=out/'source_records'/f'{i:03d}.json'
        if rp.exists():
            rec=read(rp);assert base.sha(out/'source_fields'/rec['file'])==rec['sha256'];records.append(rec);continue
        attempts=[]
        with base.deny_labels(base.OUT,attempts):
            native,receipts=load_raw(pair)
            features=dict(unit_raw=native,fixed_position_projection=[F.normalize((x.to('mps')@projection.T).cpu(),dim=1) for x in native])
            fields={};stats={}
            for v in ('A','B'):
                rbind=pair['variants'][v];c=coverage(rbind['path'],rbind['file_sha256'])
                for rep,(r,q) in features.items():
                    full=fit([sample(r,c,np.ones(4096,bool))],validate=(i==0))
                    fields[v+'/'+rep+'/full_r_to_q']=predict(full,q)
                    for fold in (0,1):
                        fitted=fit([sample(r,c,split==fold)],validate=(i==0))
                        key=v+'/'+rep+'/'+str(fold)
                        if fitted is None:
                            stats[key]=dict(valid=False,reason='reference_training_block_missing_role')
                        else:
                            fields[key+'/r_to_r']=predict(fitted,r)
                            fields[key+'/r_to_q']=predict(fitted,q)
                            fields[key+'/coefficient']=fitted['w'].numpy()
                            fields[key+'/bias']=fitted['b'].numpy()
                            stats[key]=dict(valid=True,**fitted['diagnostics'])
            assert not attempts
            file=f'{i:03d}.npz';np.savez_compressed(out/'source_fields'/file,**fields)
        rec=dict(index=i,id=pair['id'],dataset=pair['dataset'],file=file,sha256=base.sha(out/'source_fields'/file),
            raw=receipts,fit=stats,query_label_access_attempts=attempts,new_encodings=0)
        base.write(rp,rec);records.append(rec)
        if (i+1)%10==0:print(json.dumps(dict(stage='REFERENCE_ONLY',n=i+1,seconds=time.monotonic()-start)),flush=True)
    base.write(out/'source_sealed.json',dict(n=60,records=records,config_sha256=base.sha(out/'config.json'),
        source_seconds=time.monotonic()-start,query_GT_reads=0,new_encodings=0))


def verify(out,require_seal=True):
    cfg=read(out/'config.json')
    assert base.sha(__file__)==cfg['source_sha256']==base.sha(out/'frozen/task_transfer_identifiability.py')
    assert base.sha(out/'pairs.json')==cfg['pairs_sha256']
    assert base.sha(base.ASSETS/'native_assets/positional_basis.pt')==cfg['basis_sha256']
    if require_seal:
        assert read(out/'source_sealed.json')['config_sha256']==base.sha(out/'config.json')
    return cfg


def metrics(score,c,keep):
    if score is None:
        return None
    y=c[keep].astype(float);s=score[keep].astype(float)
    if y.sum()==0 or (1-y).sum()==0:
        return None
    p=s>0
    balanced_risk=.5*np.sum(y*(s-1)**2)/y.sum()+.5*np.sum((1-y)*(s+1)**2)/(1-y).sum()
    return dict(auc=base.weighted_auc(s,y,1-y),soft_iou=float(100*np.sum(y*p)/(y.sum()+np.sum((1-y)*p))),
                balanced_squared_risk=float(balanced_risk),foreground_mass=float(y.sum()),background_mass=float((1-y).sum()))


def threshold(score,c,keep):
    s,y=score[keep].astype(float),c[keep].astype(float)
    if y.sum()==0 or (1-y).sum()==0:return None
    order=np.argsort(s,kind='stable');s,y=s[order],y[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];sizes=np.diff(np.r_[starts,len(s)])
    pos=np.add.reduceat(y,starts);tp=np.cumsum(pos[::-1]);fp=np.cumsum((sizes-pos)[::-1])
    k=len(starts)-1-int(np.argmax(tp/(y.sum()+fp)))
    return float(s[starts[k-1]]) if k else float(np.nextafter(s[0],-np.inf))


def oracle_stage(out):
    verify(out);pairs=read(out/'pairs.json');truth=read(base.OUT/'query_truth.json')
    assert base.sha(base.OUT/'query_truth.json')==read(out/'config.json')['query_truth_sha256']
    basis=torch.load(base.ASSETS/'native_assets/positional_basis.pt',map_location='cpu',weights_only=True)['basis'].float()
    projection=(torch.eye(1024)-basis@basis.T).to('mps');split=split_grid();results=[];start=time.monotonic()
    seal=read(out/'source_sealed.json')
    for i,pair in enumerate(pairs):
        output=out/'oracle_records'/f'{i:03d}.json'
        if output.exists():results.append(read(output));continue
        native,receipts=load_raw(pair)
        features=dict(unit_raw=native,fixed_position_projection=[F.normalize((x.to('mps')@projection.T).cpu(),dim=1) for x in native])
        rec=seal['records'][i];assert rec['id']==pair['id'] and base.sha(out/'source_fields'/rec['file'])==rec['sha256']
        measures=[];fields={}
        with np.load(out/'source_fields'/rec['file']) as source:
            for v in ('A','B'):
                rb=pair['variants'][v];qb=truth[pair['id']]['masks'][v]
                cr=coverage(rb['path'],rb['file_sha256']);cq=coverage(qb['path'],qb['sha256'])
                for rep,(r,q) in features.items():
                    for fold in (0,1):
                        train=split==fold;test=~train;key=v+'/'+rep+'/'+str(fold)
                        if not rec['fit'][key]['valid']:
                            measures.append(dict(variant=v,representation=rep,fold=fold,valid=False,reason='source_missing_role'));continue
                        rs,qs=sample(r,cr,train),sample(q,cq,train)
                        qfit=fit([qs],validate=(i==0));joint=fit([rs,qs],validate=(i==0))
                        if qfit is None or joint is None:
                            measures.append(dict(variant=v,representation=rep,fold=fold,valid=False,reason='query_training_missing_role'));continue
                        candidates={'r_to_r':source[key+'/r_to_r'],'r_to_q':source[key+'/r_to_q'],
                            'q_to_q':predict(qfit,q),'q_to_r':predict(qfit,r),
                            'joint_to_q':predict(joint,q),'joint_to_r':predict(joint,r)}
                        t=threshold(candidates['r_to_q'],cq,train)
                        if t is not None:candidates['r_to_q_oracle_threshold']=candidates['r_to_q']-t
                        measured={name:metrics(value,cr if name.endswith('_r') else cq,test) for name,value in candidates.items()}
                        if any(value is None for value in measured.values()):
                            measures.append(dict(variant=v,representation=rep,fold=fold,valid=False,reason='heldout_missing_role'));continue
                        for name,value in candidates.items():fields[key+'/'+name]=value
                        rw=torch.from_numpy(source[key+'/coefficient']);qw=qfit['w']
                        angle=float(F.cosine_similarity(rw[None],qw[None]))
                        measures.append(dict(variant=v,representation=rep,fold=fold,valid=True,metrics=measured,
                            oracle_threshold=t,reference_query_direction_cosine=angle,
                            query_fit=qfit['diagnostics'],joint_fit=joint['diagnostics']))
        file=f'{i:03d}.npz';np.savez_compressed(out/'oracle_fields'/file,**fields)
        value=dict(index=i,id=pair['id'],dataset=pair['dataset'],measures=measures,
            oracle_fields_sha256=base.sha(out/'oracle_fields'/file),query_labels_used=True,never_deployable=True)
        base.write(output,value);results.append(value)
        if (i+1)%10==0:print(json.dumps(dict(stage='ORACLE_DIAGNOSTIC',n=i+1,seconds=time.monotonic()-start)),flush=True)
    summary={}
    for d in sorted({p['dataset'] for p in pairs}):
        summary[d]={}
        for rep in REPRESENTATIONS:
            pairwise=[]
            for row in results:
                if row['dataset']!=d:continue
                valid=[m for m in row['measures'] if m['valid'] and m['representation']==rep]
                if not valid:continue
                names=set.intersection(*(set(m['metrics']) for m in valid))
                pairwise.append(dict(id=row['id'],n_valid_folds=len(valid),
                    metrics={name:{metric:float(np.mean([m['metrics'][name][metric] for m in valid])) for metric in ('auc','soft_iou','balanced_squared_risk')} for name in names},
                    direction_cosine=float(np.mean([m['reference_query_direction_cosine'] for m in valid]))))
            names=set.intersection(*(set(p['metrics']) for p in pairwise))
            means={name:{metric:float(np.mean([p['metrics'][name][metric] for p in pairwise])) for metric in ('auc','soft_iou','balanced_squared_risk')} for name in names}
            contrasts={}
            for a,b in [('r_to_r','r_to_q'),('q_to_q','r_to_q'),('q_to_q','joint_to_q'),('joint_to_r','r_to_r'),('r_to_q_oracle_threshold','r_to_q')]:
                contrasts[a+' minus '+b]={m:base.bootstrap([p['metrics'][a][m]-p['metrics'][b][m] for p in pairwise]) for m in ('auc','soft_iou')}
            summary[d][rep]=dict(n_pairs=len(pairwise),n_valid_folds=sum(p['n_valid_folds'] for p in pairwise),means=means,
                contrasts=contrasts,mean_direction_cosine=float(np.mean([p['direction_cosine'] for p in pairwise])),pairs=pairwise)
    base.write(out/'report.json',dict(state='ORACLE_DIAGNOSTIC_COMPLETE',datasets=summary,
        oracle_seconds=time.monotonic()-start,new_encodings=0,query_GT_used_for_oracles=True,
        warning='Not segmentation benchmark scores: heldout soft token IoU and AUC, oracle use of query train-block labels. Within-image context is shared.',
        source_seal_sha256=base.sha(out/'source_sealed.json'),config_sha256=base.sha(out/'config.json')))
    print(json.dumps({d:{rep:{k:v for k,v in s.items() if k in ('n_pairs','n_valid_folds','means','mean_direction_cosine')} for rep,s in v.items()} for d,v in summary.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['source','oracle','all'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    if args.mode in ('source','all'):source_stage(args.out)
    if args.mode in ('oracle','all'):oracle_stage(args.out)


if __name__=='__main__':main()
