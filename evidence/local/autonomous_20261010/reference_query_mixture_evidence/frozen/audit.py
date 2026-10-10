"""Independent sealed-field/reference-query first-moment audit.

All600 scores are recomputed from saved scalar fields and exact masks. Only the
predeclared two episodes0/100 optionally reread O24 to spot-check generation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[4]
RUN=REPO.parent/'cv_data/a/reference_query_mixture600_20261010'
SPOT_INDICES=(0,100)
HASHES,CHECKS={},Counter()


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def sha(p):
    p=Path(p).resolve()
    if str(p) not in HASHES:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        HASHES[str(p)]=h.hexdigest()
    return HASHES[str(p)]


def check_sha(p,d):assert sha(p)==d,(str(p),'SHA mismatch');CHECKS['file_SHA']+=1


def array_sha(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256(json.dumps([list(x.shape),x.dtype.str]).encode());h.update(x.tobytes());return h.hexdigest()


def coverage(row,role):
    with Image.open(row[role+'_mask_path']) as im:x=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert array_sha(x)==row[role+'_mask_hash'];CHECKS[role+'_mask_identity']+=1
    yy=np.arange(1024,dtype=np.int64)*x.shape[0]//1024
    xx=np.arange(1024,dtype=np.int64)*x.shape[1]//1024
    canvas=x[yy[:,None],xx[None,:]]
    c=canvas.reshape(64,16,64,16).mean((1,3)).reshape(-1)
    assert c.mean()==canvas.mean()
    CHECKS[role+'_canonical_coverage']+=1
    return c


def auc(s,c):
    order=np.argsort(s,kind='stable');v=s[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    p,n=np.add.reduceat(c[order],starts),np.add.reduceat((1-c)[order],starts)
    return float(np.sum(p*(np.cumsum(n)-n+.5*n))/(p.sum()*n.sum()))


def stats(values):
    v=np.asarray(values,dtype=float)
    return dict(mean=float(v.mean()),median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))


def identities():
    cfg,seal=read(RUN/'config.json'),read(RUN/'sealed.json')
    assert cfg['n']==seal['n']==600 and seal['state']=='ALL_DIAGNOSTIC_FIELDS_SEALED'
    assert seal['query_GT_reads']==seal['baseline_reads']==0
    check_sha(RUN/'config.json',seal['config_sha256']);check_sha(RUN/'records.jsonl',seal['records_sha256'])
    check_sha(RUN/'frozen/diagnose.py',cfg['source_sha256'])
    parent=Path(cfg['parent_root'])
    for rel,d in cfg['parent_hashes'].items():check_sha(parent/rel,d)
    for p,d in cfg['external_source_sha256'].items():check_sha(p,d)
    check_sha(cfg['profile_path'],cfg['profile_sha256']);check_sha(cfg['basis_path'],cfg['basis_sha256'])
    records,manifest,tasks=lines(RUN/'records.jsonl'),read(parent/'manifest.json'),read(parent/'tasks.json')
    prs=read(parent/'sealed.json')['receipts'];assert len(records)==len(manifest)==len(tasks)==len(prs)==600
    cache=Path(cfg['profile_path']).parent
    for index,(row,t,rec,pr) in enumerate(zip(manifest,tasks,records,prs)):
        assert row['episode_id']==t['episode_id']==rec['episode_id']==pr['episode_id']
        assert rec['index']==index and rec['apd']==pr['apd_applied']
        assert rec['raw_requests']==t['raw']
        assert all(rec[k]==0 for k in ('query_GT_reads','baseline_reads','encoder_forward','raw_cache_writes'))
        assert rec['reference_mask_array_sha256']==row['reference_mask_hash']
        check_sha(RUN/'fields'/rec['filename'],rec['fields_sha256'])
        for request in rec['raw_requests']:
            ep=cache/request['key']/'entry.json';check_sha(ep,request['entry_sha256']);entry=read(ep)
            assert entry['file_sha256']==request['payload_sha256'] and entry['features']['O/24']['tensor_sha256']==request['tensor_sha256']
            CHECKS['raw_metadata_parent_binding']+=1
    (OUT/'frozen').mkdir(exist_ok=True);shutil.copyfile(__file__,OUT/'frozen/audit.py')
    write(OUT/'ANALYSIS_FROZEN.json',dict(source_sha256=sha(__file__),diagnostic_source=cfg['source_sha256'],
        seal_sha256=sha(RUN/'sealed.json'),query_GT_reads=0,raw_spot_check_indices=list(SPOT_INDICES),
        recipe='600 exactmask/coverage/saved scalarAUC,moments,unclipped priorerror decomposition,sourcecentroid geometry; optionalonly0/100actualO24CPU replay; no segmenter'))
    return cfg,manifest,records


def raw_spot(cfg,row,rec,index):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    cache=Path(cfg['profile_path']).parent;arrays=[]
    for req in rec['raw_requests']:
        folder=cache/req['key'];entry=read(folder/'entry.json');p=folder/entry['file']
        check_sha(p,req['payload_sha256'])
        with np.load(p,allow_pickle=False) as z:x=z['O/24'].copy()
        assert x.dtype==np.float32 and x.shape==(4096,1024) and array_sha(x)==req['tensor_sha256']
        arrays.append(torch.from_numpy(x));CHECKS['spot_O24_array_identity']+=1
    r,q=[F.normalize(x,dim=1) for x in arrays]
    c=torch.from_numpy(coverage(row,'reference'))
    basis=torch.load(cfg['basis_path'],map_location='cpu',weights_only=True)['basis'].float()
    projection=torch.eye(1024)-basis@basis.T
    result={}
    with np.load(RUN/'fields'/rec['filename'],allow_pickle=False) as z:
        for rep in cfg['representations']:
            rr,qq=(r,q) if rep=='rawunit' or not rec['apd'] else [F.normalize(x@projection.T,dim=1) for x in (r,q)]
            rr,qq=rr.double(),qq.double();fg=(c[:,None]*rr).sum(0)/c.sum();bg=((1-c)[:,None]*rr).sum(0)/(1-c).sum()
            d=fg-bg;s=(((qq-bg)@d)/d.square().sum()).numpy();cent=torch.stack((fg,bg,qq.mean(0))).numpy()
            assert np.max(np.abs(s-z[rep+'/query_coordinate']))<1e-11
            assert np.max(np.abs(cent-z[rep+'/centroids']))<1e-12
            result[rep]=dict(maximum_scalar_difference=float(np.max(np.abs(s-z[rep+'/query_coordinate']))),
                            maximum_centroid_difference=float(np.max(np.abs(cent-z[rep+'/centroids']))))
            CHECKS['spot_actual_feature_moment_replay']+=1
    return dict(index=index,episode_id=row['episode_id'],apd=rec['apd'],representations=result,encoder_calls=0)


def main():
    started=time.monotonic();cfg,manifest,recs=identities()
    parent=read(RUN/'results.json');check_sha(RUN/'scored_episodes.jsonl',parent['scored_sha256']);check_sha(RUN/'sealed.json',parent['seal_sha256'])
    original={r['episode_id']:r for r in lines(RUN/'scored_episodes.jsonl')};rows=[];maxima=Counter()
    for row,rec in zip(manifest,recs):
        c=coverage(row,'query');cr=coverage(row,'reference');prior=float(c.mean());values={}
        assert 0<prior<1
        with np.load(RUN/'fields'/rec['filename'],allow_pickle=False) as z:
            for rep,info in rec['representations'].items():
                if info['degenerate']:values[rep]={'degenerate':True};continue
                s=z[rep+'/query_coordinate'];cent=z[rep+'/centroids']
                assert s.shape==(4096,) and cent.shape==(3,1024) and s.dtype==cent.dtype==np.float64 and np.isfinite(s).all() and np.isfinite(cent).all()
                rf,rb,mq=cent;d=rf-rb;sep=np.linalg.norm(d);coord=float(s.mean());closed=float((mq-rb)@d/(d@d));clip=float(np.clip(coord,0,1))
                assert abs(coord-closed)<1e-12 and abs(coord-info['unconstrained_query_prior_coordinate'])<1e-12
                assert abs(cr.mean()-info['reference_prior'])<1e-12 and abs(sep-info['conditional_mean_separation'])<1e-12
                convex=float(np.linalg.norm(mq-rb-clip*d)/sep)
                assert abs(convex-info['convex_segment_residual_over_source_separation'])<1e-12
                assert abs(info['source_conditional_fg_coordinate']-1)<1e-10 and abs(info['source_conditional_bg_coordinate'])<1e-10
                fg=float((c*s).sum()/c.sum());bg=float(((1-c)*s).sum()/(1-c).sum())
                rawerr=coord-prior;fbias=prior*(fg-1);bbias=(1-prior)*bg;de=abs(rawerr-fbias-bbias)
                assert de<1e-12;maxima['decomposition_error']=max(maxima['decomposition_error'],de)
                val=dict(**info,query_prior_GT=prior,signed_prior_error=clip-prior,absolute_prior_error=abs(clip-prior),
                    query_conditional_fg_coordinate=fg,query_conditional_bg_coordinate=bg,query_conditional_gap=fg-bg,
                    foreground_shift_bias=fbias,background_shift_bias=bbias,raw_prior_error=rawerr,
                    bias_decomposition_error=de,prototype_direction_area_auc=auc(s,c))
                expected=original[row['episode_id']]['representations'][rep]
                for key,v in val.items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):
                        difference=abs(float(v)-float(expected[key]));assert difference<1e-11,(row['episode_id'],rep,key,difference)
                        maxima['parent_numeric_difference']=max(maxima['parent_numeric_difference'],difference)
                    else:assert v==expected[key]
                CHECKS['representation_moment_AUC_prior_and_ledger']=CHECKS['representation_moment_AUC_prior_and_ledger']+1
                values[rep]=val
        rows.append(dict(episode_id=row['episode_id'],dataset=row['dataset'],representations=values))
    groups={}
    for ds in sorted({r['dataset'] for r in rows}):
        rr=[r for r in rows if r['dataset']==ds];groups[ds]={}
        for rep in cfg['representations']:
            valid=[r['representations'][rep] for r in rr if not r['representations'][rep]['degenerate']]
            numbers={k:stats([r[k] for r in valid]) for k,v in valid[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)} if valid else {}
            own=dict(n=len(rr),valid=len(valid),moments=numbers,negative_or_zero_query_role_gap=sum(r['query_conditional_gap']<=0 for r in valid),
                     coordinate_outside_unit_interval=sum(r['coordinate_outside_unit_interval'] for r in valid))
            recorded=parent['groups'][ds][rep]
            assert all(own[k]==recorded[k] for k in ('n','valid','negative_or_zero_query_role_gap','coordinate_outside_unit_interval'))
            for k,v in numbers.items():
                for stat,n in v.items():assert abs(n-recorded['moments'][k][stat])<1e-11,(ds,rep,k,stat)
            groups[ds][rep]=own;CHECKS['dataset_representation_pool_identity']+=1
    print(json.dumps(dict(state='ALL600_SAVED_SCALAR_VERIFIED',checks=dict(CHECKS),maxima=dict(maxima))),flush=True)
    # Optional spot replay is limited to predeclared0/100, not selected by scores.
    spots=[raw_spot(cfg,manifest[i],recs[i],i) for i in SPOT_INDICES] if '--raw-spots' in sys.argv else []
    output=dict(state='VERIFIED',n=600,groups=groups,checks=dict(CHECKS),maximum_errors=dict(maxima),raw_spots=spots,
        raw_feature_arrays_loaded=len(spots)*2,full600_raw_feature_replay=False,source_input_SHA=HASHES,
        source_generation_seconds=read(RUN/'sealed.json')['seconds'],audit_seconds=time.monotonic()-started,
        encoder_calls=0,new_segmentation_masks=0,
        limits='Necessary scalarfirstmoment condition only; reference E_F=1/E_B=0 byconstruction; globalcentroid/sourceclippedline consistency not full distribution transfer; exposed panels; raw error decomposes, clipped error doesnot')
    write(OUT/'verification.json',output)
    (OUT/'scored_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    print(json.dumps(dict(state=output['state'],n=600,seconds=output['audit_seconds'],raw_feature_arrays_loaded=output['raw_feature_arrays_loaded'],checks=output['checks'])),flush=True)


if __name__=='__main__':main()
