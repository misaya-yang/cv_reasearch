"""Post-seal saved-field ranking/zero-point diagnosis; no new predictions."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
RUN=REPO.parent/'cv_data/a/reference_focus_head200_20261010'
FIELDS=tuple(a+'.'+v for a in ('actual','derived') for v in ('global','local4','equal'))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str]).encode());h.update(a.tobytes());return h.hexdigest()


def statistics(score,fg,bg):
    s=np.asarray(score,np.float64).reshape(-1);f=np.asarray(fg,np.float64).reshape(-1);b=np.asarray(bg,np.float64).reshape(-1)
    nf,nb=float(f.sum()),float(b.sum())
    result=dict(fg_mass=nf,bg_mass=nb,
        fg_score_mean=float(f@s/nf) if nf else None,bg_score_mean=float(b@s/nb) if nb else None,
        fg_nonpositive_fraction=float(f[s<=0].sum()/nf) if nf else None,
        bg_positive_fraction=float(b[s>0].sum()/nb) if nb else None)
    if not nf or not nb:return dict(result,auc=None,ap=None,tpr_at_fpr05=None)
    keep=f+b>0;f,b,s=f[keep],b[keep],s[keep]
    order=np.argsort(s,kind='stable');s,f,b=s[order],f[order],b[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    f,b=np.add.reduceat(f,starts),np.add.reduceat(b,starts)
    auc=float((f*(np.cumsum(b)-.5*b)).sum()/(nf*nb))
    tp,fp=np.cumsum(f[::-1]),np.cumsum(b[::-1]);ap=float((f[::-1]*tp/(tp+fp)).sum()/nf)
    tpr=float(np.max(np.r_[0.,tp[fp/nb<=.05]/nf]))
    return dict(result,auc=auc,ap=ap,tpr_at_fpr05=tpr)


def main():
    assert not (ROOT/'query.json').exists(),'Never replace an evaluated field study'
    seal,cfg=read(RUN/'sealed.json'),read(RUN/'config.json')
    assert seal['state']=='ALL_PREDICTIONS_SEALED' and seal['n']==200
    assert sha(RUN/'config.json')==seal['config_sha256']
    manifest,tasks=read(RUN/'manifest.json'),read(RUN/'tasks.json')
    records=[json.loads(l) for l in (RUN/'inference.jsonl').read_text().splitlines()]
    assert sha(RUN/'inference.jsonl')==seal['inference_sha256']
    source=read(ROOT/'reference.json');assert source['state']=='REFERENCE_ONLY_SEALED'
    recipe=dict(source_seal_sha256=sha(RUN/'sealed.json'),reference_diagnostics_sha256=sha(ROOT/'reference.json'),
        fields=FIELDS,rois=['all','FoRIS_foreground','actual_derived_equal_crf_disagreement'],
        metric='fixed fine128 scores, foreground/background pixel mass inside canonical1024 ROI; macro episode AUC/AP/TPR. Zero-sign errors are node-sign diagnostics, not full rendered mask mIoU.',
        truth='torch nearest original query binary mask to1024; exact same score frame',
        query_features_read=False,new_masks=0,encoder_calls=0,raw_writes=0,
        no_threshold_search=True,bias_decomposition='actual-minus-derived field score difference minus frozen source intercept difference; diagnostic only',
        preregistration='Post-hoc development diagnosis, frozen transform before this script reads query GT; not untouched confirmation')
    (ROOT/'query_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    # Independent tied/ordered weighted-rank sanity checks, not method tests.
    assert statistics([0,1],[0,1],[1,0])['auc']==1
    assert statistics([1],[3],[7])['auc']==.5
    started,results=time.perf_counter(),[]
    for row,task,rec in zip(manifest,tasks,records):
        assert row['episode_id']==task['episode_id']==rec['episode_id']
        field_path=RUN/'fields'/rec['filename'];prediction_path=RUN/'predictions'/rec['filename']
        assert sha(field_path)==rec['fields_sha256'] and sha(prediction_path)==rec['prediction_sha256']
        assert sha(task['baseline']['path'])==task['baseline']['sha256']
        with np.load(field_path,allow_pickle=False) as z:fields={n:z[n].copy() for n in FIELDS}
        with np.load(prediction_path,allow_pickle=False) as z:
            actual=np.unpackbits(z['cli1024/actual.equal.crf'],count=1024*1024).reshape(1024,1024).astype(bool)
            derived=np.unpackbits(z['cli1024/derived.equal.crf'],count=1024*1024).reshape(1024,1024).astype(bool)
        with np.load(task['baseline']['path'],allow_pickle=False) as z:
            baseline=np.unpackbits(z['cli1024/foris.crf'],count=1024*1024).reshape(1024,1024).astype(bool)
        with Image.open(row['query_mask_path']) as im:original=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert arrsha(original)==row['query_mask_hash']
        truth=F.interpolate(torch.from_numpy(original).float()[None,None],(1024,1024),mode='nearest')[0,0].numpy()>.5
        rois={'all':np.ones_like(truth),'FoRIS_foreground':baseline,'actual_derived_equal_crf_disagreement':actual^derived}
        metrics={}
        bias=float(rec['diagnostics']['fits']['actual']['bias']-rec['diagnostics']['fits']['derived']['bias'])
        for roi_name,roi in rois.items():
            fg=(truth&roi).reshape(128,8,128,8).sum((1,3)).astype(np.float64)
            bg=(~truth&roi).reshape(128,8,128,8).sum((1,3)).astype(np.float64)
            metrics[roi_name]={n:statistics(fields[n],fg,bg) for n in FIELDS}
            for view in ('global','local4','equal'):
                delta=fields['actual.'+view].astype(np.float64)-fields['derived.'+view].astype(np.float64)
                metrics[roi_name]['shift.'+view]=dict(
                    frozen_intercept_difference=bias,
                    fg_total_score_change=float((fg*delta).sum()/fg.sum()) if fg.sum() else None,
                    bg_total_score_change=float((bg*delta).sum()/bg.sum()) if bg.sum() else None,
                    fg_direction_only_change=float((fg*(delta-bias)).sum()/fg.sum()) if fg.sum() else None,
                    bg_direction_only_change=float((bg*(delta-bias)).sum()/bg.sum()) if bg.sum() else None)
        results.append(dict(index=rec['index'],episode_id=row['episode_id'],dataset=row['dataset'],metrics=metrics))
    payload=dict(state='POSTSEAL_SAVED_QUERY_ANALYSIS_COMPLETE',n=len(results),results=results,
        recipe_sha256=sha(ROOT/'query_recipe.json'),script_sha256=sha(__file__),
        query_GT_after_seal=True,query_raw_reads=0,new_predictions=0,encoder_calls=0,seconds=time.perf_counter()-started)
    (ROOT/'query.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=payload['state'],n=len(results),seconds=payload['seconds'])),flush=True)


if __name__=='__main__':main()
