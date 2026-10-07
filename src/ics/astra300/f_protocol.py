"""The frozen F-family protocol, separate from A B0 and A calibration.

Only complete reference labels enter fitting. All out-of-block fits rebuild
dictionaries after removing the held block and its one-token buffer. Original
reference pixels are read only in the calibration evaluator, never in a kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.special import expit, logsumexp

from . import common

EPS=1e-6
WEIGHTS=(0.,.25,1.)


def unit(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)


def mad(x):
    x=np.asarray(x,np.float32)
    return max(float(np.median(np.abs(x-np.median(x)))) if x.size else 0.,EPS)


def edges(hw):
    x=np.arange(np.prod(hw)).reshape(hw)
    return np.r_[np.c_[x[:-1].ravel(),x[1:].ravel()],np.c_[x[:,:-1].ravel(),x[:,1:].ravel()]].astype(int)


def spherical_modes(x,maximum=8):
    """Row-first, farthest spherical initialization; 20 Lloyd steps, no empty."""
    x=unit(x)
    if not len(x):return np.empty((0,x.shape[1]),np.float32)
    selected=[0];nearest=x@x[0]
    for _ in range(1,min(maximum,len(x))):
        candidates=np.setdiff1d(np.arange(len(x)),selected)
        j=int(candidates[np.argmin(nearest[candidates])])
        if nearest[j]>=1-EPS:break
        selected.append(j);nearest=np.maximum(nearest,x@x[j])
    modes=x[selected].copy()
    for _ in range(20):
        assignment=(x@modes.T).argmax(1)
        modes=unit(np.stack([x[assignment==j].mean(0) for j in range(len(modes)) if np.any(assignment==j)]))
    return modes


def quarters(hw):
    y,x=np.indices(hw)
    return ((y>=hw[0]/2)*2+(x>=hw[1]/2)).ravel()


def buffered_folds(hw,valid):
    blocks=quarters(hw);result=[]
    for block in range(4):
        held=np.asarray(valid,bool)&(blocks==block)
        excluded=ndimage.binary_dilation(held.reshape(hw),structure=np.ones((3,3))).ravel()
        result.append((np.asarray(valid,bool)&~excluded,held))
    return result


def _model(r,coverage,valid):
    r=np.asarray(r,np.float32);valid=np.asarray(valid,np.float32)
    wf=coverage*valid;wb=(1-coverage)*valid
    pure_f=(coverage>=.9)&(valid>0);pure_b=(coverage<=.1)&(valid>0)
    if wf.sum()<=0:return dict(empty_f=True,valid=valid,coverage=coverage)
    mu_f=unit((r*wf[:,None]).sum(0))
    fm=spherical_modes(r[pure_f]) if pure_f.any() else mu_f[None]
    if wb.sum()<=0:
        bank=r[wf>0]
        if len(bank)<2:threshold=1.
        else:
            sim=bank@bank.T;np.fill_diagonal(sim,-np.inf)
            threshold=float(np.quantile(sim.max(1),.05))
        return dict(empty_f=False,single_class=True,mu_f=mu_f,fm=fm,bm=np.empty((0,r.shape[1]),np.float32),bank_f=bank,threshold=threshold,valid=valid,coverage=coverage,pure_f=pure_f,pure_b=pure_b)
    mu_b=unit((r*wb[:,None]).sum(0))
    bm=spherical_modes(r[pure_b]) if pure_b.any() else mu_b[None]
    return dict(empty_f=False,single_class=False,mu_f=mu_f,mu_b=mu_b,fm=fm,bm=bm,bank_f=r[pure_f] if pure_f.any() else mu_f[None],bank_b=r[pure_b] if pure_b.any() else mu_b[None],valid=valid,coverage=coverage,pure_f=pure_f,pure_b=pure_b)


def _scores(q,r,model):
    q=np.asarray(q,np.float32)
    if model["empty_f"]:
        return np.full((len(q),3),-100,np.float32)
    if model["single_class"]:
        score=(q@model["bank_f"].T).max(1)-model["threshold"]
        return np.repeat((score/.1)[:,None],3,axis=1)
    mean=(q@model["mu_f"]-q@model["mu_b"])/.1
    nn=((q@model["bank_f"].T).max(1)-(q@model["bank_b"].T).max(1))/.1
    profile=q@r.T/.1
    fweight=model["coverage"]*model["valid"];bweight=(1-model["coverage"])*model["valid"]
    lf=logsumexp(profile+np.where(fweight>0,np.log(np.maximum(fweight,EPS)),-np.inf)[None,:],axis=1)-np.log(fweight.sum())
    lb=logsumexp(profile+np.where(bweight>0,np.log(np.maximum(bweight,EPS)),-np.inf)[None,:],axis=1)-np.log(bweight.sum())
    return np.c_[mean,nn,lf-lb].astype(np.float32)


@dataclass
class FContext:
    ep: object
    q: np.ndarray
    r: np.ndarray
    hw: tuple
    rhw: tuple
    c: np.ndarray
    valid: np.ndarray
    s: np.ndarray
    p0: np.ndarray
    m0: np.ndarray
    M0: np.ndarray
    f: np.ndarray
    b: np.ndarray
    fsim: np.ndarray
    bsim: np.ndarray
    heads: np.ndarray
    edges: np.ndarray
    capacity: np.ndarray
    candidates: list
    pieces: list
    source: object
    factor_scale: float
    trace: dict=field(default_factory=dict)
    resources: object=None

    @property
    def n(self):return len(self.q)
    @property
    def fm(self):return self.f
    @property
    def bm(self):return self.b
    @property
    def edge_weight(self):return self.capacity
    @property
    def Q(self):return self.candidates
    @property
    def P(self):
        masks=[]
        for ids in self.pieces:
            mask=np.zeros(self.n,bool);mask[ids]=True;masks.append(mask)
        return masks
    @property
    def F(self):return self.valid&(self.c>=.9)
    @property
    def B(self):return self.valid&(self.c<=.1)
    @property
    def pure(self):return self.F|self.B

    def energy(self,y,s=None,edges=None,weights=None):
        s=self.s if s is None else np.asarray(s,np.float32).ravel()
        e=self.edges if edges is None else np.asarray(edges,int)
        w=self.capacity if weights is None else np.asarray(weights,np.float32)
        y=np.asarray(y,bool).ravel()
        return float(np.logaddexp(0,s).sum()-s@y+w@(y[e[:,0]]!=y[e[:,1]]))


def exact_e0(c,s=None):
    from ics.methods.pro_paired_environment import exact_potts_cut
    y,info=exact_potts_cut(c.s if s is None else s,c.edges,c.capacity)
    return np.asarray(y,bool),info


def _candidate_pools(c,scales):
    masks=[];seen=set()
    def add(y):
        y=np.asarray(y,bool).ravel();key=np.packbits(y).tobytes()
        if key not in seen and len(masks)<16:seen.add(key);masks.append(y)
    for j in range(3):
        for threshold in (-.5,0.,.5):add(c.heads[:,j]/scales[j]>threshold)
    for j in range(3):add(exact_e0(c,c.heads[:,j])[0])
    add(c.m0);add(np.zeros(c.n,bool));add(np.ones(c.n,bool))
    pieces=[];keys=set()
    for a,b in combinations(masks,2):
        labels,count=ndimage.label((a!=b).reshape(c.hw))
        for j in range(1,count+1):
            ids=np.flatnonzero(labels.ravel()==j)
            if len(ids)>1:
                key=tuple(ids)
                if key not in keys:keys.add(key);pieces.append(ids)
    yy,xx=np.indices(c.hw);gy=np.minimum(3,yy*4//c.hw[0]);gx=np.minimum(3,xx*4//c.hw[1])
    cells=gy*4+gx
    for cell in range(16):
        ids=np.flatnonzero(cells.ravel()==cell)
        if len(ids)>1 and tuple(ids) not in keys:keys.add(tuple(ids));pieces.append(ids)
    # Round-robin spatial bins before controversy mass; row/ID stable ties.
    disagreement=np.zeros(c.n,int)
    for a,b in combinations(masks,2):disagreement+=(a!=b)
    buckets=[[] for _ in range(16)]
    for ids in pieces:
        cell=int(cells.ravel()[ids].min());buckets[cell].append(ids)
    for bucket in buckets:bucket.sort(key=lambda ids:(-int(disagreement[ids].sum()),tuple(ids)))
    selected=[]
    while any(buckets) and len(selected)<64:
        for bucket in buckets:
            if bucket and len(selected)<64:selected.append(bucket.pop(0))
    discarded=[ids.tolist() for bucket in buckets for ids in bucket]
    return masks,selected,discarded


def build_context(ep,*,resources=None):
    common.validate(ep)
    q=np.asarray(ep.q,np.float32);r=np.asarray(ep.r,np.float32)
    valid=np.asarray(ep.wvalid,np.float32)>0;cov=np.asarray(common.coverage(ep),np.float32)
    model=_model(r,cov,np.asarray(ep.wvalid,np.float32))
    heads=_scores(q,r,model);s=heads[:,0];p0=expit(s).astype(np.float32)
    if model["empty_f"]:f=np.empty((0,r.shape[1]),np.float32);b=f.copy()
    else:f=model["fm"];b=model["bm"]
    e=edges(ep.q_hw);distance=np.sum((q[e[:,0]]-q[e[:,1]])**2,axis=1)
    scale=float(np.median(distance)) if len(distance) else 0.
    capacity=.25*np.exp(-distance/(scale+EPS))
    folds=buffered_folds(ep.r_hw,valid);blocks=quarters(ep.r_hw)
    source_heads=np.zeros((len(r),3),np.float32);source_f=np.zeros((len(r),max(len(f),1)),np.float32);source_b=np.zeros((len(r),max(len(b),1)),np.float32)
    source_oof=np.zeros(len(r),bool);fold_calls=0;fold_models=[]
    for train,hold in folds:
        if not train.any() or not hold.any():fold_models.append(None);continue
        fitted=_model(r,cov,np.asarray(ep.wvalid,np.float32)*train)
        fold_models.append(fitted)
        if fitted["empty_f"] or fitted.get("single_class"):continue
        source_heads[hold]=_scores(r[hold],r,fitted)
        # Mode indices refer to that fold's row-first/farthest dictionary.
        # Missing slots have an impossible cosine, not replicated max scores.
        # No full-fit mode is used to score held points or align their IDs.
        ff=r[hold]@fitted["fm"].T;bb=r[hold]@fitted["bm"].T
        source_f[hold]=-2.;source_b[hold]=-2.
        source_f[np.ix_(hold,np.arange(min(ff.shape[1],source_f.shape[1])))]=ff[:,:source_f.shape[1]]
        source_b[np.ix_(hold,np.arange(min(bb.shape[1],source_b.shape[1])))]=bb[:,:source_b.shape[1]]
        source_oof[hold]=True;fold_calls+=1
    source_edges=edges(ep.r_hw)
    role_pairs=[]
    for a,bidx in source_edges:
        if valid[a] and valid[bidx]:
            if cov[a]>=.9 and cov[bidx]<=.1:role_pairs.append([r[a],r[bidx]])
            if cov[bidx]>=.9 and cov[a]<=.1:role_pairs.append([r[bidx],r[a]])
    src=SimpleNamespace(heads=source_heads,coverage=cov,valid=np.asarray(ep.wvalid,np.float32),
                        train_valid=valid,oof_valid=source_oof,blocks=blocks,hw=ep.r_hw,
                        fold_train=[x[0] for x in folds],fold_test=[x[1] for x in folds],fold_models=fold_models,
                        fsim=source_f,bsim=source_b,edges=source_edges,
                        role_pairs=np.asarray(role_pairs,np.float32).reshape(-1,2,r.shape[1]))
    c=FContext(ep,q,r,ep.q_hw,ep.r_hw,cov,valid,s,p0,s>0,
               common.U(ep,p0.reshape(ep.q_hw),.5),f,b,q@f.T if len(f) else np.empty((len(q),0)),
               q@b.T if len(b) else np.empty((len(q),0)),heads,e,capacity,[],[],src,
               mad(source_heads[source_oof,0]),resources=resources)
    scales=[mad(source_heads[source_oof,j]) for j in range(3)]
    c.candidates,c.pieces,discarded=_candidate_pools(c,scales)
    c.trace.update(protocol="F_native_FP32_single_R_original",source_oof_fit_calls=fold_calls,
                   source_oof_valid_points=int(source_oof.sum()),source_training_mask_hash=common.array_hash(valid),
                   prototype_modes=[len(f),len(b)],source_head_scales=scales,discarded_P=discarded,
                   empty_reference_fg=model["empty_f"],single_class=model.get("single_class",False),
                   missing_pure_roles=[not bool(((cov>=.9)&valid).any()),not bool(((cov<=.1)&valid).any())])
    return c


def kernel_result(ep,c,out,*,method=None):
    """One R; binary optimizers use F's specified token-unary disagreement."""
    info=dict(c.trace,**out.get("info",{}),quality="unknown",query_GT_read=False,method=method)
    probability=out.get("probability")
    labels=out.get("y",out.get("labels"))
    unary=out.get("margin",out.get("unary",c.s))
    if probability is not None:
        return common.Result(np.asarray(probability,np.float32).reshape(c.hw),.5,info=info)
    if labels is None:
        return common.Result(expit(np.asarray(unary,np.float32)).reshape(c.hw),.5,info=info)
    candidate=np.asarray(labels,np.float32).reshape(c.hw)
    mask=common.disagreement(ep,candidate,p0=(c.s>0).reshape(c.hw).astype(np.float32),mask0=c.M0,threshold=.5,baseline_threshold=.5)
    optimizer=common.U(ep,candidate,.5)
    same_unary=common.U(ep,(np.asarray(unary)>0).reshape(c.hw).astype(np.float32),.5)
    continuous=common.U(ep,expit(np.asarray(unary)).reshape(c.hw),.5)
    info.update(binary_optimizer_readout=True,readout_checks={name:dict(foreground_pixels=int(value.sum()),sha256=common.array_hash(value)) for name,value in (("optimizer",optimizer),("same_unary",same_unary),("continuous",continuous))})
    return common.Result(candidate,.5,mask,info)


def _reference_query(ep,train):
    # The calibration kernel has no complete source mask and cannot see held
    # labels through a resource provider. All fitting weights exclude buffer.
    native=common.as_episode(ep)
    original=tuple(ep.reference_mask.shape) if ep.reference_mask is not None else (tuple(ep.r_rgb.shape[:2]) if ep.r_rgb is not None else ep.r_hw)
    return replace(native,q=np.asarray(ep.r),q_hw=ep.r_hw,q_valid=np.asarray(ep.wvalid),
                   q_rgb=ep.r_rgb,query_geometry=dict(ep.reference_geometry),original_shape=original,
                   wf=np.asarray(ep.wf)*train,wvalid=np.asarray(ep.wvalid)*train,
                   reference_mask=None,artifacts={},provider=None)


def calibrate_weight(ep,kernel,*,weights=WEIGHTS,resources=None):
    """F source IoU selection of 0/.25/1, once-rendered *complete* masks."""
    if tuple(weights)!=WEIGHTS:raise ValueError("F primary weight choices are exactly {0,.25,1}")
    if ep.reference_mask is None:
        return 0.,dict(calibration="F_original_reference_IoU",fallback="complete_original_reference_mask_unavailable",weights=list(WEIGHTS),effective_folds=0)
    target=np.asarray(ep.reference_mask,bool)
    folds=buffered_folds(ep.r_hw,np.asarray(ep.wvalid)>0)
    valid_folds=[(train,hold) for train,hold in folds if train.any() and hold.any() and np.any(np.asarray(ep.wf)[train]>0) and np.any(np.asarray(ep.wb)[train]>0)]
    if len(valid_folds)<2:
        return 0.,dict(calibration="F_original_reference_IoU",fallback="fewer_than_two_valid_buffered_folds",weights=list(WEIGHTS),effective_folds=len(valid_folds))
    rows=[];fit_calls=0
    for weight in WEIGHTS:
        scores=[]
        for train,hold in valid_folds:
            fold_ep=_reference_query(ep,train);context=build_context(fold_ep,resources=resources)
            out=kernel(context,weight);fit_calls+=1
            result=kernel_result(fold_ep,context,out)
            mask=common.render(fold_ep,result)["original"]
            domain=common.U(fold_ep,hold.reshape(ep.r_hw).astype(np.float32),.5)
            intersection=np.count_nonzero(mask&target&domain);union=np.count_nonzero((mask|target)&domain)
            scores.append(intersection/union if union else 1.)
        rows.append(dict(weight=weight,fold_IoU=list(map(float,scores)),mean_IoU=float(np.mean(scores))))
    selected=max(rows,key=lambda x:(x["mean_IoU"],-x["weight"]))["weight"]
    return selected,dict(calibration="F_original_reference_IoU",weights=list(WEIGHTS),effective_folds=len(valid_folds),
                         trials=rows,selected_weight=selected,fold_kernel_calls=fit_calls,unknown_and_one_patch_buffer_removed=True,
                         fitting_dictionary_rebuilt_after_split=True,reference_GT_read_only_after_complete_mask=True,
                         source_scope="shared frozen encoding context; not independent cross-image validation")


def full_profile_control(ep,*,same_candidates=False):
    """All M reference similarities, identical graph and F final readout."""
    c=build_context(ep);score=c.heads[:,2]
    if same_candidates:
        costs=[c.energy(y,s=score) for y in c.candidates]
        j=min(range(len(costs)),key=lambda i:(float(costs[i]),int(c.candidates[i].sum()),i))
        y=c.candidates[j];info=dict(selected_candidate=j,candidate_energies=list(map(float,costs)))
    else:y,info=exact_e0(c,s=score)
    return kernel_result(ep,c,dict(y=y,unary=score,info=dict(control="complete_reference_profile_same_graph",all_reference_tokens=len(ep.r),**info)),method="F_full_profile_control")


def profile_same_edits_control(ep,result):
    """Match *actual original-pixel* additions/deletions, without query GT."""
    c=build_context(ep);actual=common.render(ep,result)["original"]
    add=int(np.count_nonzero(actual&~c.M0));delete=int(np.count_nonzero(~actual&c.M0))
    field=common.continuous_original(ep,c.heads[:,2].reshape(c.hw)).ravel()
    y=c.M0.ravel().copy();ids=np.arange(len(y))
    candidates=ids[~y];order=np.lexsort((candidates,-field[candidates]));y[candidates[order[:add]]]=True
    candidates=ids[c.M0.ravel()];order=np.lexsort((candidates,field[candidates]));y[candidates[order[:delete]]]=False
    return common.Result(mask_original=y.reshape(ep.original_shape),info=dict(control="complete_profile_exact_original_edit_budget",added_pixels=add,deleted_pixels=delete,all_reference_tokens=len(ep.r),query_GT_read=False,quality="unknown"))
