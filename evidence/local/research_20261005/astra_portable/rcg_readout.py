#!/usr/bin/env python3
"""Reference-corrected reciprocal-graph readout (RCG), frozen DINOv3 cache.

Input contract: q,r [4096,1024], cov and FoRIS score [64,64]. No query labels,
pre/native masks, class IDs, images, weights, fitting, or model downloads in predict.
The FINAL recipe is alpha=.5, cross-image k=10, query graph k=20, lambda=16,
confidence floor=.1. These are development-selected constants, not claimed a priori.

  python rcg_readout.py infer --root ROOT --manifest episodes.csv --out RUN
  python rcg_readout.py evaluate --root ROOT --manifest episodes.csv --out RUN

The original manifest_dev241_full.json schema is also supported. All records are
required; missing caches are errors, never silently dropped. Run on CPU.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,hashlib,json,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.stats import rankdata
from scipy.sparse.csgraph import connected_components

CONFIG={'arm':'RCG','alpha':0.5,'cross_image_k':10,'query_k':20,'source_purity':0.9,
        'lambda':16.0,'confidence_floor':0.1,'query_gt_in_inference':False,
        'native_mask_in_inference':False,'finalizer':'bilinear 1024, align_corners=False, >0.5; no re-minmax',
        'parameter_provenance':'development-selected; other-three-fold recipe selection is reported separately',
        'cg_rtol':1e-7,'cg_atol':1e-9,'cg_maxiter':300}

def rank(x: np.ndarray) -> np.ndarray:
    return ((rankdata(x.ravel(),method='average')-0.5)/x.size).astype(np.float32)

def minmax(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    return (x-x.min())/max(float(x.max()-x.min()),1e-6)

def mask_from_field(z: np.ndarray) -> np.ndarray:
    x=torch.from_numpy(np.ascontiguousarray(z,dtype=np.float32))[None,None]
    return F.interpolate(x,(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()>.5

def unpack(x: np.ndarray) -> np.ndarray:
    if x.dtype!=np.uint8 or x.size!=131072:raise ValueError('Expected 1024x1024 np.packbits uint8 mask')
    return np.unpackbits(x).reshape(1024,1024).astype(bool)

@torch.inference_mode()
def predict(q: torch.Tensor|np.ndarray,r: torch.Tensor|np.ndarray,
            cov: np.ndarray,score: np.ndarray) -> tuple[np.ndarray,dict[str,Any]]:
    """Return a 64x64 float32 field and solver metadata. No evaluation inputs."""
    q=torch.as_tensor(q,device='cpu').float();r=torch.as_tensor(r,device='cpu').float()
    cov=np.asarray(cov);score=np.asarray(score)
    if q.shape!=(4096,1024) or r.shape!=(4096,1024) or cov.shape!=(64,64) or score.shape!=(64,64):
        raise ValueError('Invalid cache shapes; regular 64x64 / 1024x1024 mapping is required')
    if not (torch.isfinite(q).all() and torch.isfinite(r).all() and np.isfinite(cov).all() and np.isfinite(score).all()):
        raise ValueError('Nonfinite inputs')
    if cov.min()<0 or cov.max()>1:raise ValueError('Reference coverage must lie in [0,1]')
    if (q.norm(dim=1)==0).any() or (r.norm(dim=1)==0).any():raise ValueError('Zero-norm tokens')
    q=F.normalize(q,dim=1);r=F.normalize(r,dim=1)
    fi=np.flatnonzero(cov.ravel()>=CONFIG['source_purity'])
    if not len(fi):fi=np.flatnonzero(cov.ravel()==cov.max())
    sim=q@r.T
    dq=sim.topk(CONFIG['cross_image_k'],dim=1).values.mean(1)
    dr=sim.topk(CONFIG['cross_image_k'],dim=0).values.mean(0)
    guide=((2*sim[:,fi]-dr[fi][None,:]).max(1).values-dq).numpy()
    del sim,dr,dq,r
    s=minmax(score).ravel()
    y=(s+CONFIG['alpha']*(rank(guide)-rank(s))).astype(np.float64)
    # Directed adaptive 20-NN query graph, then retain reciprocal edges only.
    sim=q@q.T;sim.fill_diagonal_(-2)
    values,idx=sim.topk(CONFIG['query_k'],dim=1);del sim,q
    distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).numpy().ravel()
    w=sparse.csr_matrix((weights,(np.repeat(np.arange(4096),CONFIG['query_k']),idx.numpy().ravel())),shape=(4096,4096))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8)
    degree=np.asarray(w.sum(1)).ravel()
    a=CONFIG['confidence_floor']+np.abs(2*s-1);a=a/a.mean();a=a.astype(np.float64)
    H=sparse.diags(a)+CONFIG['lambda']*(sparse.diags(degree)-w)
    rhs=a*y;iterations=[0]
    def callback(_:np.ndarray)->None:iterations[0]+=1
    z,status=cg(H,rhs,x0=y,rtol=CONFIG['cg_rtol'],atol=CONFIG['cg_atol'],maxiter=CONFIG['cg_maxiter'],callback=callback)
    if status!=0:raise RuntimeError(f'CG did not converge: status={status}')
    residual=float(np.linalg.norm(H@z-rhs)/max(np.linalg.norm(rhs),1e-12))
    ncomp,labels=connected_components(w,directed=False)
    info={'source_pure_tokens':int(len(fi)),'graph_undirected_edges':int(w.nnz//2),'graph_components':int(ncomp),
          'graph_isolated_tokens':int((degree==0).sum()),'cg_iterations':iterations[0],'cg_relative_residual':residual,
          'weighted_mean_error':float(abs(np.dot(a,z-y))),
          'maximum_principle_error':float(max(0,z.max()-y.max(),y.min()-z.min()))}
    return z.reshape(64,64).astype(np.float32),info

def load_rows(path:Path)->list[dict[str,Any]]:
    if path.suffix.lower()=='.csv':rows=list(csv.DictReader(path.open()))
    else:
        d=json.loads(path.read_text())
        if isinstance(d,list):rows=d
        elif 'episodes' in d:rows=d['episodes']
        elif 'rows' in d:rows=d['rows']
        else:
            rows=[]
            for f,rs in d.items():
                if isinstance(rs,list):rows.extend(dict(r,fold=r.get('fold',f)) for r in rs)
    unique={}
    for rr in rows:
        r=dict(rr)
        for k in ['fold','e','c']:r[k]=int(r[k])
        r['key']=f"{r['fold']}_{r['e']}_{r['c']}"
        for role in ['support','query']:
            if not isinstance(r.get(role),str):raise ValueError('Manifest must contain support and query photo filenames')
        if r['key'] in unique:
            old=unique[r['key']]
            if any(old[k]!=r[k] for k in ['fold','e','c','support','query']):raise ValueError('Conflicting duplicate episode')
        unique[r['key']]=r
    return sorted(unique.values(),key=lambda r:(r['fold'],r['e'],r['c']))

def groups(rows:list[dict[str,Any]])->np.ndarray:
    parent=list(range(len(rows)));seen={}
    def find(i:int)->int:
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for role in ['support','query']:
            photo=Path(r[role]).name
            if photo in seen:
                x,y=find(i),find(seen[photo]);parent[max(x,y)]=min(x,y)
            seen[photo]=i
    ids={};out=[]
    for i in range(len(rows)):
        root=find(i);ids.setdefault(root,len(ids));out.append(ids[root])
    return np.array(out,dtype=int)

def metric(iu:np.ndarray,classes:np.ndarray,weights:np.ndarray|None=None)->float:
    if weights is None:weights=np.ones(len(classes))
    I=np.bincount(classes,weights=weights*iu[:,0],minlength=classes.max()+1)
    U=np.bincount(classes,weights=weights*iu[:,1],minlength=classes.max()+1)
    return float(100*np.mean(I[U>0]/U[U>0]))

def infer(args:argparse.Namespace,rows:list[dict[str,Any]])->None:
    out=args.out
    if (out/'sealed.json').exists():raise FileExistsError('Use a fresh output directory; existing sealed run is immutable')
    (out/'fields').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2));(out/'manifest.json').write_text(json.dumps(rows,indent=2))
    sealed={};infos=[];t=time.perf_counter()
    for n,row in enumerate(rows):
        k=row['key'];start=time.perf_counter()
        fp=args.root/'cache/evidence_v1/feat'/f'{k}.pt';pp=args.root/'results/extent_v1/run/packets'/f'{k}.npz'
        feat=torch.load(fp,map_location='cpu',weights_only=True)
        with np.load(pp,allow_pickle=False) as p:cov=p['cov'].copy();score=p['score'].copy()
        z,info=predict(feat['q'],feat['r'],cov,score)
        path=out/'fields'/f'{k}.npz';maskpath=out/'masks'/f'{k}.npz'
        np.savez_compressed(path,RCG=z);np.savez_compressed(maskpath,mask=np.packbits(mask_from_field(z)))
        sealed[k]={'field_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256(maskpath.read_bytes()).hexdigest()}
        info.update(key=k,elapsed_seconds=time.perf_counter()-start);infos.append(info)
        if n%20==0:print(f'infer {n+1}/{len(rows)} elapsed={time.perf_counter()-t:.1f}s',flush=True)
    (out/'audits.json').write_text(json.dumps(infos,indent=2));(out/'sealed.json').write_text(json.dumps(sealed,indent=2))

def evaluate(args:argparse.Namespace,rows:list[dict[str,Any]])->None:
    out=args.out;sealed=json.loads((out/'sealed.json').read_text());records=[];arrays={a:[] for a in ['RCG','FORIS_PRE','FORIS_NATIVE']}
    mismatch=0
    for row in rows:
        k=row['key'];path=out/'masks'/f'{k}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sealed[k]['mask_sha256']:raise ValueError('Sealed prediction changed')
        with np.load(path) as p:pred=unpack(p['mask'])
        with np.load(args.root/'results/extent_v1/run/packets'/f'{k}.npz',allow_pickle=False) as p:
            truth=unpack(p['truth']);pre=unpack(p['pre']);native=unpack(p['native']);score=p['score'].copy()
        err=int(np.count_nonzero(mask_from_field(minmax(score))!=pre));mismatch+=err
        if err:raise AssertionError(f'FoRIS pre not reproduced: {k}, {err} pixels')
        for nm,m in [('RCG',pred),('FORIS_PRE',pre),('FORIS_NATIVE',native)]:
            I=int(np.count_nonzero(m&truth));U=int(np.count_nonzero(m|truth));arrays[nm].append([I,U])
            records.append({'key':k,'fold':row['fold'],'c':row['c'],'arm':nm,'intersection':I,'union':U,'iou':I/U})
    arrays={a:np.asarray(x,dtype=np.int64) for a,x in arrays.items()};cl=np.array([r['c'] for r in rows]);fo=np.array([r['fold'] for r in rows]);gr=groups(rows);G=gr.max()+1
    draws=np.random.RandomState(0).randint(G,size=(2000,G));W=np.stack([np.bincount(d,minlength=G) for d in draws])[:,gr]
    v={a:metric(x,cl) for a,x in arrays.items()};samples={a:np.array([metric(x,cl,w) for w in W]) for a,x in arrays.items()}
    result={'config':CONFIG,'N':len(rows),'classes':int(len(set(cl))),'photo_groups':int(G),'pre_reproduction_mismatched_pixels':mismatch,'scores':v,'contrasts':{}}
    for a in ['FORIS_PRE','FORIS_NATIVE']:
        d=arrays['RCG'][:,0]/arrays['RCG'][:,1]-arrays[a][:,0]/arrays[a][:,1]
        result['contrasts'][a]={'gain':v['RCG']-v[a],'ci95':np.percentile(samples['RCG']-samples[a],[2.5,97.5]).tolist(),
                              'fold_gain':[metric(arrays['RCG'][fo==f],cl[fo==f])-metric(arrays[a][fo==f],cl[fo==f]) for f in sorted(set(fo))],
                              'up':int((d>1e-12).sum()),'down':int((d< -1e-12).sum()),'tie':int((abs(d)<=1e-12).sum())}
    (out/'report.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'IU.npz',**arrays)
    with (out/'episode_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.save(out/'bootstrap_photo_draws.npy',draws);print(json.dumps(result,indent=2))

def main()->None:
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','evaluate','all']);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);rows=load_rows(a.manifest)
    if not rows:raise ValueError('Empty manifest')
    for r in rows:
        for rel in [f"cache/evidence_v1/feat/{r['key']}.pt",f"results/extent_v1/run/packets/{r['key']}.npz"]:
            if not (a.root/rel).is_file():raise FileNotFoundError(a.root/rel)
    if a.stage in ['infer','all']:infer(a,rows)
    if a.stage in ['evaluate','all']:evaluate(a,rows)
if __name__=='__main__':main()
