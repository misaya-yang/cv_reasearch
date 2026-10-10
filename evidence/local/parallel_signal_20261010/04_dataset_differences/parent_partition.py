"""PACO100 GT-only ranking diagnosis: target versus parent-internal/external BG."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import json
import numpy as np
import analyze as a

def mass(x):return x.reshape(128,8,128,8).sum(axis=(1,3)).ravel().astype(float)
def weighted_auc(score,fg,bg):
    keep=(fg+bg)>0;s,f,b=score[keep],fg[keep],bg[keep]
    if f.sum()==0 or b.sum()==0:return None
    ix=np.argsort(s,kind='stable');s,f,b=s[ix],f[ix],b[ix]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    f,b=np.add.reduceat(f,starts),np.add.reduceat(b,starts)
    return float(np.sum(f*(np.cumsum(b)-.5*b))/(f.sum()*b.sum()))

manifest=[r for r in a.read(a.PILOT/'manifest.json') if r['dataset']=='paco_part']
parents={r['episode_id']:r for r in a.lines(a.DATA/'a/paco_fast9_600_20261009/analysis/task_granularity/episodes.jsonl')}
infer={r['episode_id']:r for r in a.lines(a.PILOT/'inference.jsonl')}
seal={r['episode_id']:r for r in a.read(a.SIGNAL/'raw/sealed.json')['receipts']}
eseal={r['episode_id']:r for r in a.read(a.SIGNAL/'existing/sealed.json')['receipts']}
out=[]
for row in manifest:
    eid=row['episode_id'];pp=parents[eid];inf=infer[eid];ss=seal[eid];es=eseal[eid]
    assert pp['query']['part_binary_tensor_sha256']==row['query_mask_hash']
    p=pp['parent_mask_path'];assert a.sha(p)==pp['parent_mask_sha256'];a.receipts[p]=pp['parent_mask_sha256']
    with np.load(p,allow_pickle=False) as z:
        parent=np.unpackbits(z['cli1024/parent'],count=1024**2).reshape(1024,1024).astype(bool)
        truth=np.unpackbits(z['cli1024/target'],count=1024**2).reshape(1024,1024).astype(bool)
        hw=tuple(z['original_hw']);original=np.unpackbits(z['original/target'],count=int(np.prod(hw))).reshape(hw).astype(np.uint8)
        assert a.array_sha(original)==row['query_mask_hash']
        h,w=hw
        assert np.array_equal(truth,original[(np.arange(1024)*h//1024)[:,None],(np.arange(1024)*w//1024)[None,:]])
    pred=a.PILOT/'predictions'/inf['filename'];assert a.sha(pred)==inf['prediction_sha256'];a.receipts[str(pred)]=inf['prediction_sha256']
    with np.load(pred,allow_pickle=False) as z:
        base=np.unpackbits(z['cli1024/foris.crf'],count=1024**2).reshape(1024,1024).astype(bool)
        unary=np.unpackbits(z['cli1024/role.unary'],count=1024**2).reshape(1024,1024).astype(bool)
    p=a.SIGNAL/'raw/fields'/ss['filename'];assert a.sha(p)==ss['sha256'];a.receipts[str(p)]=ss['sha256']
    with np.load(p,allow_pickle=False) as z:
        fields={k:z[k].ravel().astype(float) for k in ('source_apd.ridge.global','source_apd.ridge.local4')}
    p=a.SIGNAL/'existing/fields'/es['filename'];assert a.sha(p)==es['sha256'];a.receipts[str(p)]=es['sha256']
    with np.load(p,allow_pickle=False) as z:fields.update({k:z[k].ravel().astype(float) for k in ('P0','foris_source')})
    roi_stats={}
    for roi,mask in [('global',np.ones_like(base)),('foris_foreground',base),('new_unary_foreground',unary&~base)]:
        fg=mass(truth&mask);roi_stats[roi]={}
        for part,bgmask in [('all_BG',~truth),('inside_parent_BG',~truth&parent),('outside_parent_BG',~truth&~parent)]:
            bg=mass(bgmask&mask)
            roi_stats[roi][part]=dict(fg=float(fg.sum()),bg=float(bg.sum()),auc={k:weighted_auc(v,fg,bg) for k,v in fields.items()})
    out.append(dict(episode_id=eid,rois=roi_stats))

(a.OUT/'parent_partition_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out))
summary={}
for roi in out[0]['rois']:
    summary[roi]={}
    for part in out[0]['rois'][roi]:
        rows=[r['rois'][roi][part] for r in out]
        valid=[r for r in rows if all(v is not None for v in r['auc'].values())]
        delta=[r['auc']['source_apd.ridge.local4']-r['auc']['source_apd.ridge.global'] for r in valid]
        summary[roi][part]=dict(n_valid=len(valid),macro_auc={k:float(np.mean([r['auc'][k] for r in valid])) for k in fields},
            local_minus_whole=a.summary(delta),local_better=sum(x>0 for x in delta),fg_mass=sum(r['fg'] for r in rows),bg_mass=sum(r['bg'] for r in rows))
a.write(a.OUT/'parent_partition_summary.json',dict(n=100,summary=summary,query_GT_and_parent_GT_diagnostic_only=True,new_predictions=0,
    same_target_mass_across_BG_partitions=True,warning='Target mass includes GT outside associated parent; no annotation clipping. Different BG subsets and valid n must not be confused with causal view effects.',
    source_sha256=a.receipts,script_sha256=a.sha(__file__)))
print(json.dumps(summary),flush=True)
