"""CPU candidate: reference mode-affinity covariance on actual mask components.

Spatial geometry is absent from this descriptor. Mean similarity and mode
occupancy do not determine the covariance of mode responses. This remains an
empirically unvalidated transfer hypothesis, not a covariance-alignment claim.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import time

import numpy as np
from scipy import ndimage

from .reference_occupancy import aggregate_tree,cluster,spatial_tree,unit
from .reference_shape import STRUCTURE,component_totals,optimize


@dataclass(frozen=True)
class Config:
    reference_modes:int=8
    lloyd_steps:int=5
    purity:float=.9
    covariance_norm_floor:float=1e-6
    strength:float=1.0
    evidence_penalty_scale:float=3.0
    minimum_component_pixels:int=8
    proposals:int=48
    maximum_rounds:int=4
    improvement_tolerance:float=1e-10


def response_features(response):
    response=np.asarray(response,dtype=np.float64)
    if response.ndim!=2 or not np.isfinite(response).all():raise ValueError('Finite affinity rows required')
    outer=response[:,:,None]*response[:,None,:]
    return np.column_stack((np.ones(len(response)),response,outer.reshape(len(response),-1)))


def covariances(totals,k):
    count=np.maximum(totals[:,0],1e-12)
    mean=totals[:,1:1+k]/count[:,None]
    second=totals[:,1+k:].reshape(-1,k,k)/count[:,None,None]
    covariance=second-mean[:,:,None]*mean[:,None,:]
    return .5*(covariance+covariance.transpose(0,2,1))


def make_bank(reference_response,labels,pure,coverage,cfg):
    k=reference_response.shape[1]
    features=response_features(reference_response)
    matrices=[]
    for label in range(1,int(labels.max())+1):
        take=(labels.ravel()==label)&pure
        if int(take.sum())<cfg.minimum_component_pixels:continue
        weight=coverage.ravel()[take]
        total=(features[take]*weight[:,None]).sum(0,keepdims=True)
        matrix=covariances(total,k)[0]
        if np.linalg.norm(matrix)<=cfg.covariance_norm_floor:continue
        matrices.append(matrix)
    if not matrices:return None
    matrices=np.stack(matrices)
    return dict(k=k,matrices=matrices,norm2=np.sum(matrices*matrices,axis=(1,2)),
                trace=np.trace(matrices,axis1=1,axis2=2),mode='full')


def covariance_debts(matrices,valid,bank):
    debt=np.zeros(len(matrices))
    if not valid.any():return debt
    selected=matrices[valid]
    if bank['mode']=='full':
        delta=selected[:,None,:,:]-bank['matrices'][None,:,:,:]
        distance=np.sum(delta*delta,axis=(2,3))/bank['norm2'][None,:]
    elif bank['mode']=='trace':
        trace=np.trace(selected,axis1=1,axis2=2)
        distance=(trace[:,None]-bank['trace'][None,:])**2/(bank['trace'][None,:]**2)
    else:raise ValueError('Unknown covariance comparison mode')
    distance=np.minimum(distance.min(1),np.finfo(np.float64).max/2)
    debt[valid]=distance/(1+distance)
    return debt


def evaluate(mask,unary,features,bank,cfg):
    labels,_=ndimage.label(mask,structure=STRUCTURE)
    totals=component_totals(labels,features)
    valid=totals[:,0]>=cfg.minimum_component_pixels;valid[0]=False
    debt=covariance_debts(covariances(totals,bank['k']),valid,bank)
    fit=np.bincount(labels.ravel(),weights=unary,minlength=len(totals));fit[0]=0
    positive=np.bincount(labels.ravel(),weights=np.maximum(unary,0),minlength=len(totals));positive[0]=0
    # Identity consistency discounts existing positive evidence. Adding a weak
    # pixel does not introduce an extra area-proportional mismatch charge.
    energies=fit-cfg.strength*cfg.evidence_penalty_scale*positive*debt;energies[0]=0
    return float(energies.sum()),labels,energies,debt


def proposal_masks(children,features,unary,bank,cfg):
    n=len(unary)
    totals=aggregate_tree(features,children)
    valid=totals[:,0]>=cfg.minimum_component_pixels
    debt=covariance_debts(covariances(totals,bank['k']),valid,bank)
    fit=aggregate_tree(unary,children)
    positive=aggregate_tree(np.maximum(unary,0),children)
    score=(fit-cfg.strength*cfg.evidence_penalty_scale*positive*debt)/np.sqrt(np.maximum(totals[:,0],1))
    eligible=np.flatnonzero(valid)
    order=eligible[np.argsort(-score[eligible],kind='stable')[:cfg.proposals]]
    candidates=[]
    for node in order:
        mask=np.zeros(n,dtype=bool);stack=[int(node)]
        while stack:
            item=stack.pop()
            if item<n:mask[item]=True
            else:stack.extend(children[item-n])
        candidates.append(mask)
    return candidates


def predict(q,r,coverage,base,cfg=Config()):
    started=time.perf_counter()
    if (cfg.reference_modes<2 or cfg.lloyd_steps<1 or not 0<cfg.purity<=1
            or cfg.covariance_norm_floor<=0 or cfg.strength<0 or cfg.evidence_penalty_scale<=0 or cfg.minimum_component_pixels<2
            or cfg.proposals<1 or cfg.maximum_rounds<1 or cfg.improvement_tolerance<=0):
        raise ValueError('Invalid covariance configuration')
    q,r=unit(q),unit(r)
    base=np.asarray(base,dtype=np.float64);cov=np.asarray(coverage,dtype=np.float64)
    if (base.ndim!=2 or base.shape!=cov.shape or len(q)!=base.size or len(r)!=base.size
            or q.shape[1]!=r.shape[1] or not np.isfinite(base).all() or not np.isfinite(cov).all()
            or cov.min()<0 or cov.max()>1 or cov.max()<=0):
        raise ValueError('Aligned query/reference/base and valid reference mask required')
    pure=cov.ravel()>=cfg.purity
    if not pure.any():pure=cov.ravel()==cov.max()
    centers,_,_=cluster(r[pure],cfg.reference_modes,cfg.lloyd_steps,cov.ravel()[pure])
    k=len(centers)
    reference_response=np.einsum('nd,kd->nk',r,centers,optimize=False).astype(np.float64)
    margin_field=base.copy()
    reference_margin=None
    response=None
    if k>=2:
        ordered=np.sort(reference_response[pure],axis=1)
        reference_margin=float(np.average(ordered[:,-1]-ordered[:,-2],weights=cov.ravel()[pure]))
        response=np.einsum('nd,kd->nk',q,centers,optimize=False).astype(np.float64)
        ordered_query=np.sort(response,axis=1)
        margin_field=np.clip((ordered_query[:,-1]-ordered_query[:,-2])/max(reference_margin,1e-6),0,1).reshape(base.shape)
    labels,_=ndimage.label(cov>.5,structure=STRUCTURE)
    bank=make_bank(reference_response,labels,pure,cov,cfg) if k>=2 else None
    mask=base>.5
    if bank is None or cfg.strength==0 or base.max()<=.5:
        return dict(token_mask=mask,trace_control=mask.copy(),mode_margin_field=margin_field,
                    info=dict(config=asdict(cfg),modes=k,reference_templates=0 if bank is None else len(bank['matrices']),
                              abstention=True,wall_seconds=time.perf_counter()-started,
                              query_gt_used=False,new_encoder_forwards=0))
    features=response_features(response)
    children=spatial_tree(q,base.shape)
    mask,search=optimize(base,features,children,bank,cfg,proposal_function=proposal_masks,energy_function=evaluate)
    trace_bank=dict(bank,mode='trace')
    trace,control_search=optimize(base,features,children,trace_bank,cfg,
                                  proposal_function=proposal_masks,energy_function=evaluate)
    return dict(token_mask=mask,trace_control=trace,mode_margin_field=margin_field,
                info=dict(config=asdict(cfg),modes=k,reference_templates=len(bank['matrices']),
                          reference_covariances=bank['matrices'].tolist(),reference_trace=bank['trace'].tolist(),
                          reference_mode_margin=reference_margin,
                          search=search,trace_control_search=control_search,abstention=False,
                          added_tokens=int((mask&~(base>.5)).sum()),deleted_tokens=int((~mask&(base>.5)).sum()),
                          wall_seconds=time.perf_counter()-started,query_gt_used=False,new_encoder_forwards=0,
                          segmentation_benefit='unmeasured',complete_dataset_minutes='unmeasured'))
