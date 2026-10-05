"""Exact selected frozen components, with unused experiment families removed.
The original function bodies are retained. Only insid3.predict is renamed locally.
No data, labels, host pre/native masks or downloads occur at module import.
"""
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from scipy import sparse
from scipy.sparse.linalg import cg
import rcg_readout as rcg
import hypothesis_source_contrast as hyp
ARMS=['reference_mean','reference_nn','cluster_nn_mean','insid3_cached','without_intra','without_occupancy','ward_bidir','ward_oneway','ward_random_reverse','ward_refmean','ward_nn_all','ward_insid3']
def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

C=SimpleNamespace(unit=unit)
def insid3_predict(q,r,cov):
    q=C.unit(np.asarray(q,dtype=np.float32));r=C.unit(np.asarray(r,dtype=np.float32));fg=np.asarray(cov).ravel()>=.5
    n=len(q);shape=np.asarray(cov).shape
    if n!=int(np.prod(shape)) or len(r)!=fg.size:raise ValueError('grid mismatch')
    if not fg.any() or fg.all():
        label=bool(fg.any());m=np.full(shape,float(label),np.float32)
        return {a:m for a in ARMS},np.arange(n),dict(fallback='constant_source_role',clusters=n)
    mu=C.unit(r[fg].mean(0));mub=C.unit(r[~fg].mean(0));cross=q@mu
    qr=q@r.T;fgmax=qr[:,fg].max(1);bgmax=qr[:,~fg].max(1);margin=fgmax-bgmax
    candidates=(cross>0)&fg[qr.argmax(1)]
    if not (cross>0).any():candidates=(cross>np.quantile(cross,.9))&fg[qr.argmax(1)]
    del qr
    dist=1-np.clip(q@q.T,-1,1);np.fill_diagonal(dist,0)
    lab=AgglomerativeClustering(n_clusters=None,metric='precomputed',linkage='average',distance_threshold=.4).fit_predict(dist)
    del dist
    k=int(lab.max())+1;areas=np.bincount(lab,minlength=k);sums=np.zeros((k,q.shape[1]),np.float32);np.add.at(sums,lab,q)
    proto=C.unit(sums);cross_mean=np.bincount(lab,weights=cross,minlength=k)/areas
    cand_n=np.bincount(lab,weights=candidates,minlength=k);aw=cand_n/areas
    meanmargin=np.bincount(lab,weights=margin,minlength=k)/areas
    if candidates.any():
        seed=int(np.argmax(np.where(cand_n>0,proto@mu,-np.inf)));intra=proto@proto[seed];aw[seed]=1
        combined=cross_mean*intra*aw;nointra=cross_mean*aw;nooccupancy=cross_mean*intra
    else:
        seed=-1;intra=np.zeros(k);combined=nointra=nooccupancy=np.zeros(k)
    masks={'reference_mean':(q@(mu-mub)>0).reshape(shape),
           'reference_nn':(margin>0).reshape(shape),
           'cluster_nn_mean':(meanmargin>0)[lab].reshape(shape),
           'insid3_cached':(combined>.2)[lab].reshape(shape),
           'without_intra':(nointra>.2)[lab].reshape(shape),
           'without_occupancy':(nooccupancy>.2)[lab].reshape(shape)}
    meta=dict(fallback=None,clusters=k,source_fg=int(fg.sum()),candidate_tokens=int(candidates.sum()),seed=seed,
              seed_area=int(areas[seed])if seed>=0 else 0,
              positive_clusters={a:int(np.unique(lab[np.asarray(v).ravel()]).size)for a,v in masks.items()},
              areas=areas.tolist(),occupancy=aw.tolist(),cross_mean=cross_mean.tolist(),intra=intra.tolist(),
              combined=combined.tolist(),cluster_nn_margin=meanmargin.tolist())
    return masks,lab,meta

def render(mask):return F.interpolate(torch.from_numpy(np.asarray(mask,np.float32))[None,None],size=(1024,1024),mode='bilinear',align_corners=False)[0,0].numpy()>.5

@torch.inference_mode()
def mean_control(q,r,cov,score,rcg):
    """Only MEAN_a0.25_l16 from the locked reciprocal_controls implementation."""
    q=F.normalize(torch.as_tensor(q).float(),dim=1);r=F.normalize(torch.as_tensor(r).float(),dim=1)
    fi=np.flatnonzero(np.asarray(cov).ravel()>=.9)
    if not len(fi):fi=np.flatnonzero(np.asarray(cov).ravel()==np.max(cov))
    guide=(q@F.normalize(r[fi].mean(0),dim=0)).numpy()
    s=rcg.minmax(score).ravel();y=(s+.25*(rcg.rank(guide)-rcg.rank(s))).astype(np.float64)
    sim=q@q.T;sim.fill_diagonal_(-2);values,idx=sim.topk(20,dim=1);del sim
    distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).numpy().ravel()
    w=sparse.csr_matrix((weights,(np.repeat(np.arange(4096),20),idx.numpy().ravel())),shape=(4096,4096))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8);degree=np.asarray(w.sum(1)).ravel()
    a=.1+np.abs(2*s-1);a=(a/a.mean()).astype(np.float64)
    H=sparse.diags(a)+16*(sparse.diags(degree)-w);rhs=a*y;iterations=[0]
    def cb(_):iterations[0]+=1
    z,status=cg(H,rhs,x0=y,rtol=1e-7,atol=1e-9,maxiter=300,callback=cb)
    if status:raise RuntimeError('Locked mean control CG failure: '+str(status))
    info=dict(source_pure_tokens=int(len(fi)),graph_undirected_edges=int(w.nnz//2),cg_iterations=iterations[0],
              cg_relative_residual=float(np.linalg.norm(H@z-rhs)/max(np.linalg.norm(rhs),1e-12)))
    return z.reshape(64,64).astype(np.float32),info

def predict_episode(q,r,cov,score,ins,hyp,rcg):
    original,lab,meta=ins.predict(q,r,cov)
    if meta.get('fallback') is not None:
        raise RuntimeError('Locked bank encountered '+str(meta['fallback'])+'; record and resolve before GT, no silent new fallback')
    if meta['candidate_tokens']==0:
        # Explicit locked no-finite-score fallback: preserve the actual default.
        # Keep K=0 and seed index=-1; do not invent a candidate or inspect GT.
        if meta['seed']!=-1 or original['insid3_cached'].any():raise AssertionError('Unexpected no-candidate default')
        names=list(hyp.NAMES);masks=np.empty((0,*cov.shape),bool);seeds=np.empty(0,np.int32)
        values=np.empty((0,len(names)),np.float64);default=-1;extra={}
        choices={n:-1 for n in names};fallback={n:True for n in names}
        outputs={n:np.packbits(render(original['insid3_cached']))for n in names}
        info=dict(default_seed=-1,hypotheses=[],eligible_candidate_count=0,
                  fallback_reason='No original eligible seed; preserve original empty default for every hypothesis selector')
    else:
        masks,info,extra=hyp.observe(q,r,cov,lab.reshape(cov.shape),meta)
        names=list(info['hypotheses'][0]['scores']);seeds=np.array([h['seed']for h in info['hypotheses']],np.int32)
        default=int(np.flatnonzero(seeds==meta['seed'])[0]);values=np.array([[h['scores'][n]if h['scores'][n]is not None else np.nan for n in names]for h in info['hypotheses']],np.float64)
        if not np.array_equal(masks[default],original['insid3_cached']):raise AssertionError('Default complete hypothesis parity')
        choices={};fallback={};outputs={}
        for j,name in enumerate(names):
            valid=np.isfinite(values[:,j]);fallback[name]=not bool(valid.any());ix=int(np.argmax(np.where(valid,values[:,j],-np.inf)))if valid.any()else default
            choices[name]=ix;outputs[name]=np.packbits(render(masks[ix]))
    choices['insid3_default']=default;outputs['insid3_default']=np.packbits(render(original['insid3_cached']))
    z,ri=rcg.predict(q,r,cov,score);mz,mi=mean_control(q,r,cov,score,rcg)
    if ri['graph_undirected_edges']!=mi['graph_undirected_edges']:raise AssertionError('RCG/mean query graph mismatch')
    outputs.update(RCG=np.packbits(rcg.mask_from_field(z)),MEAN_CONTROL=np.packbits(rcg.mask_from_field(mz)),RCG_field=z,MEAN_CONTROL_field=mz,
                   masks=masks,seeds=seeds,score_names=np.array(names),scores=values,
                   choice_names=np.array(list(choices)),choice_indices=np.array(list(choices.values())),default_index=np.array(default),
                   cluster_labels=lab.reshape(cov.shape).astype(np.int16),work_hw=np.array([1024,1024]),**extra)
    info.update(choices=choices,fallback=fallback,bank=meta,RCG=ri,MEAN_CONTROL=mi)
    return outputs,info

def predict_all(q,r,cov,score):
    raw,audit=predict_episode(q,r,cov,score,SimpleNamespace(predict=insid3_predict),hyp,rcg)
    base={k:raw[k] for k in ['RCG','MEAN_CONTROL','insid3_default','source_contrast_mean','scalar_contrast']}
    R=base['RCG'];S=base['source_contrast_mean'];T=base['scalar_contrast']
    for tag,arr in [('source',S),('scalar',T),('default',base['insid3_default'])]:
        base['RCG_or_'+tag]=R|arr
        base['RCG_and_'+tag]=R&arr
    base.update(majority3=(R&S)|(R&T)|(S&T),conservative_delete=R&(S|T),double_support_add=R|(S&T),all_intersection=R&S&T,all_union=R|S|T)
    base['native_pre_from_score']=np.packbits(rcg.mask_from_field(rcg.minmax(score)))
    fields={'RCG_field':raw['RCG_field'],'MEAN_CONTROL_field':raw['MEAN_CONTROL_field']}
    return base,fields,audit
