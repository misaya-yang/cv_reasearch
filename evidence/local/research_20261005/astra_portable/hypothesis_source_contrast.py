"""Fixed complete-hypothesis selector from the two predeclared hard cases.
No query truth, host score, fitting, area gate or confidence switch.
"""
import numpy as np
from scipy.ndimage import binary_dilation
ST=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
NAMES=['source_contrast_mean','foreground_mean_cos','foreground_mean_dot','scalar_contrast','query_saliency','relative_context_cos']
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def observe(q,r,cov,lab,metadata):
    q=unit(np.asarray(q,np.float32));r=unit(np.asarray(r,np.float32));fg=np.asarray(cov)>=.5
    ring=binary_dilation(fg,structure=ST)&~fg
    if not fg.any():raise ValueError('Source FG absent; declared dataset violates role contract')
    mf=r[fg.ravel()].mean(0);mfu=unit(mf)
    mr=r[ring.ravel()].mean(0)if ring.any()else np.full(r.shape[1],np.nan,np.float32)
    dr=unit(mf-mr)if ring.any()else None
    a=np.asarray(metadata['areas']);K=len(a);labels=lab.ravel();sums=np.zeros((K,q.shape[1]),np.float32);np.add.at(sums,labels,q);proto=unit(sums)
    cs=np.bincount(labels,weights=q@mfu,minlength=K)/a
    if np.max(abs(cs-np.asarray(metadata['cross_mean'])))>=1e-6:raise ValueError('Original source affinity mismatch')
    oldseed=metadata['seed'];counts=np.rint(a*np.asarray(metadata['occupancy'])).astype(int)
    counts[oldseed]=metadata['candidate_tokens']-counts[np.arange(K)!=oldseed].sum();rawocc=counts/a
    hypotheses=[];masks=[];hmean=[];ringmean=[];bgmean=[];combined=[]
    for seed in np.flatnonzero(counts>0):
        aw=rawocc.copy();aw[seed]=1;score=cs*(proto@proto[seed])*aw
        mask=(score>.2)[labels].reshape(lab.shape);outside=binary_dilation(mask,structure=ST)&~mask
        vals={n:None for n in NAMES};nan=np.full(q.shape[1],np.nan,np.float32)
        mq=q[mask.ravel()].mean(0)if mask.any()else nan.copy()
        mc=q[outside.ravel()].mean(0)if outside.any()else nan.copy()
        mb=q[~mask.ravel()].mean(0)if (~mask).any()else nan.copy()
        if mask.any():
            vals['foreground_mean_cos']=float(unit(mq)@mfu)
            vals['foreground_mean_dot']=float(mq@mfu)
            if dr is not None:vals['source_contrast_mean']=float(mq@dr)
            if outside.any():
                dq=mq-mc;norm=float(np.linalg.norm(dq));vals['query_saliency']=norm
                if dr is not None:
                    vals['scalar_contrast']=float(dq@dr)
                    if norm>1e-12:vals['relative_context_cos']=float(dq@dr/norm)
        hypotheses.append(dict(seed=int(seed),default_seed=bool(seed==oldseed),foreground_tokens=int(mask.sum()),context_tokens=int(outside.sum()),
                               scores=vals,native_seed_score=float(proto[seed]@mfu)))
        masks.append(mask);hmean.append(mq);ringmean.append(mc);bgmean.append(mb);combined.append(score)
    hm=np.asarray(hmean,np.float32);rm=np.asarray(ringmean,np.float32);bm=np.asarray(bgmean,np.float32)
    arrays=dict(q_H_mean=hm,q_ring_mean=rm,q_global_complement_mean=bm,
                r_FG_mean=mf,r_ring_mean=mr,reference_cov=np.asarray(cov),
                reference_H_projection=r@unit(hm).T,reference_BG_projection=r@unit(bm).T,
                combined_cluster_scores=np.asarray(combined,np.float32))
    info=dict(default_seed=int(oldseed),reference_foreground_tokens=int(fg.sum()),reference_context_tokens=int(ring.sum()),
              reference_contrast_norm=float(np.linalg.norm(mf-mr))if ring.any()else None,
              hypotheses=hypotheses,eligibility='All original candidate-count-positive seed clusters; no GT/host filter',
              per_score_validity='Source contrast needs nonempty H and reference ring only; query-context controls separately need query exterior ring')
    return np.asarray(masks,bool),info,arrays
