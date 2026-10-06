"""CPU candidate: reference silhouette moments on actual foreground components.

The feature tree supplies bounded proposals. Shape energy is always recomputed
on the physical four-connected mask, not on the proposal tree's partition.
Optimization is finite monotone local search, not a global optimality claim.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import time

import numpy as np
from scipy import ndimage

from .reference_occupancy import aggregate_tree,spatial_tree,unit


@dataclass(frozen=True)
class Config:
    strength:float=1.0
    minimum_component_pixels:int=8
    proposals:int=48
    maximum_rounds:int=4
    improvement_tolerance:float=1e-10


STRUCTURE=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)


def raw_features(z):
    z=np.asarray(z,dtype=np.complex128)
    square=z*z
    radial=(z.real*z.real+z.imag*z.imag)
    return np.column_stack((np.ones(z.size),z.real,z.imag,square.real,square.imag,radial,
                            (square*z).real,(square*z).imag,(radial*z).real,(radial*z).imag,radial*radial))


def coordinate_features(shape,original_hw=None):
    yy,xx=np.indices(shape)
    original_hw=tuple(shape) if original_hw is None else tuple(original_hw)
    if (len(original_hw)!=2 or not np.isfinite(original_hw).all()
            or min(original_hw)<=0 or any(float(v)!=int(v) for v in original_hw)):
        raise ValueError('Positive integer original image height/width required')
    height,width=original_hw
    dx=width/shape[1]/max(height,width);dy=height/shape[0]/max(height,width)
    z=((xx-(shape[1]-1)/2)*dx+1j*(yy-(shape[0]-1)/2)*dy).ravel()
    # Three-point Gauss quadrature integrates each occupied rectangular pixel
    # cell's degree-four raw moments exactly, rather than treating it as a point.
    nodes=np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])/2
    weights=np.array([5/18,4/9,5/18])
    features=np.zeros((z.size,11))
    for i,x in enumerate(nodes):
        for j,y in enumerate(nodes):
            features+=weights[i]*weights[j]*raw_features(z+dx*x+1j*dy*y)
    features[:,0]=1
    return features


def signatures(totals,minimum):
    n=totals[:,0]
    denominator=np.maximum(n,1)
    mu=(totals[:,1]+1j*totals[:,2])/denominator
    m20=(totals[:,3]+1j*totals[:,4])/denominator
    m11=totals[:,5]/denominator
    m30=(totals[:,6]+1j*totals[:,7])/denominator
    m21=(totals[:,8]+1j*totals[:,9])/denominator
    m22=totals[:,10]/denominator
    variance=m11-np.abs(mu)**2
    valid=(n>=minimum)&(variance>1e-16)
    variance=np.maximum(variance,1e-16)
    central20=m20-mu*mu
    central30=m30-3*mu*m20+2*mu**3
    central22=(m22-4*np.real(np.conj(mu)*m21)+2*np.real(np.conj(mu)**2*m20)
               +4*np.abs(mu)**2*m11-3*np.abs(mu)**4)
    signature=np.column_stack((np.clip(np.abs(central20)/variance,0,1),
                               np.log1p(np.abs(central30)/variance**1.5),
                               np.log(np.maximum(central22/variance**2,1))))
    return signature,valid


def component_totals(labels,features):
    flat=labels.ravel();count=int(flat.max())+1
    return np.column_stack([np.bincount(flat,weights=features[:,i],minlength=count)
                            for i in range(features.shape[1])])


def reference_bank(coverage,features,cfg):
    labels,_=ndimage.label(coverage>.5,structure=STRUCTURE)
    totals=component_totals(labels,features)
    signature,valid=signatures(totals,cfg.minimum_component_pixels)
    valid[0]=False
    return signature[valid]


def debts(signature,valid,bank):
    debt=np.zeros(len(signature),dtype=np.float64)
    if valid.any():
        delta=signature[valid,None,:]-bank[None,:,:]
        distance=(delta*delta).sum(2).min(1)
        debt[valid]=distance/(1+distance)
    return debt


def evaluate(mask,unary,features,bank,cfg):
    labels,_=ndimage.label(mask,structure=STRUCTURE)
    totals=component_totals(labels,features)
    signature,valid=signatures(totals,cfg.minimum_component_pixels)
    valid[0]=False
    debt=debts(signature,valid,bank)
    fit=np.bincount(labels.ravel(),weights=unary,minlength=len(totals))
    fit[0]=0
    component_energy=fit-cfg.strength*totals[:,0]*debt
    component_energy[0]=0
    return float(component_energy.sum()),labels,component_energy,debt


def proposal_masks(children,features,unary,bank,cfg):
    n=len(unary)
    totals=aggregate_tree(features,children)
    signature,valid=signatures(totals,cfg.minimum_component_pixels)
    debt=debts(signature,valid,bank)
    fit=aggregate_tree(unary,children)
    priority=(fit-cfg.strength*totals[:,0]*debt)/np.sqrt(np.maximum(totals[:,0],1))
    eligible=np.flatnonzero(valid)
    order=eligible[np.argsort(-priority[eligible],kind='stable')[:cfg.proposals]]
    masks=[]
    for node in order:
        mask=np.zeros(n,dtype=bool);stack=[int(node)]
        while stack:
            item=stack.pop()
            if item<n:mask[item]=True
            else:stack.extend(children[item-n])
        masks.append(mask)
    return masks


def optimize(base,features,children,bank,cfg):
    shape=base.shape;unary=(base-.5).ravel()
    mask=base>.5
    candidates=proposal_masks(children,features,unary,bank,cfg)
    history=[];evaluations=0
    current,labels,component_energy,_=evaluate(mask,unary,features,bank,cfg);evaluations+=1
    initial=current
    for iteration in range(cfg.maximum_rounds):
        changed=False
        # Deleting a whole negative-energy physical component is exactly improving.
        negative=component_energy < -cfg.improvement_tolerance
        negative[0]=False
        if negative.any():
            previous=current;mask=mask&~negative[labels]
            current,labels,component_energy,_=evaluate(mask,unary,features,bank,cfg);evaluations+=1
            if current+cfg.improvement_tolerance<previous:raise RuntimeError('Nonmonotone component deletion')
            history.append(dict(round=iteration,move='delete_negative_components',before=previous,after=current))
            changed=True
        best=current;winner=None;winner_name=None
        flat=mask.ravel()
        for region in candidates:
            touched=np.unique(labels.ravel()[region]);touched=touched[touched>0]
            affected=np.isin(labels.ravel(),touched) if len(touched) else np.zeros(flat.shape,dtype=bool)
            for name,proposed in (('add_region',flat|region),('remove_region',flat&~region),
                                  ('replace_touched_components',(flat&~affected)|region)):
                if np.array_equal(proposed,flat):continue
                value,_,_,_=evaluate(proposed.reshape(shape),unary,features,bank,cfg);evaluations+=1
                if value>best+cfg.improvement_tolerance:
                    best=value;winner=proposed.copy();winner_name=name
        if winner is not None:
            previous=current;mask=winner.reshape(shape)
            current,labels,component_energy,_=evaluate(mask,unary,features,bank,cfg);evaluations+=1
            history.append(dict(round=iteration,move=winner_name,before=previous,after=current))
            changed=True
        if not changed:break
    return mask,dict(initial_energy=initial,final_energy=current,moves=history,
                     energy_evaluations=evaluations,proposal_count=len(candidates),
                     global_optimum_claimed=False)


def predict(q,coverage,base,cfg=Config(),*,reference_hw=None,query_hw=None):
    started=time.perf_counter()
    if (cfg.strength<0 or cfg.minimum_component_pixels<2 or cfg.proposals<1
            or cfg.maximum_rounds<1 or cfg.improvement_tolerance<=0):
        raise ValueError('Invalid shape-prior configuration')
    base=np.asarray(base,dtype=np.float64);cov=np.asarray(coverage,dtype=np.float64)
    q=unit(q)
    if (base.ndim!=2 or min(base.shape)<1 or base.shape!=cov.shape or len(q)!=base.size
            or not np.isfinite(base).all() or not np.isfinite(cov).all()
            or cov.min()<0 or cov.max()>1 or cov.max()<=0):
        raise ValueError('Aligned query features/base and reference coverage required')
    if reference_hw is None or query_hw is None:
        mask=base>.5
        return dict(token_mask=mask,compactness_control=mask.copy(),
                    info=dict(config=asdict(cfg),original_geometry_missing=True,abstention=True,
                              wall_seconds=time.perf_counter()-started,query_gt_used=False,new_encoder_forwards=0))
    features=coordinate_features(base.shape,query_hw)
    bank=reference_bank(cov,coordinate_features(cov.shape,reference_hw),cfg)
    mask=base>.5
    if not len(bank) or cfg.strength==0 or base.max()<=.5:
        return dict(token_mask=mask,compactness_control=mask.copy(),
                    info=dict(config=asdict(cfg),reference_shapes=len(bank),abstention=True,
                              wall_seconds=time.perf_counter()-started,query_gt_used=False,new_encoder_forwards=0))
    children=spatial_tree(q,base.shape)
    mask,search=optimize(base,features,children,bank,cfg)
    # Same algorithm/information budget with a generic square, not a new method.
    side=int(np.ceil(np.sqrt(cfg.minimum_component_pixels)))
    square=coordinate_features((side,side)).sum(0,keepdims=True)
    generic,_=signatures(square,cfg.minimum_component_pixels)
    control,control_search=optimize(base,features,children,generic,cfg)
    return dict(token_mask=mask,compactness_control=control,
                info=dict(config=asdict(cfg),reference_shapes=len(bank),reference_bank=bank.tolist(),
                          reference_image_hw=list(reference_hw),query_image_hw=list(query_hw),
                          search=search,generic_square_control=control_search,abstention=False,
                          added_tokens=int((mask&~(base>.5)).sum()),deleted_tokens=int((~mask&(base>.5)).sum()),
                          wall_seconds=time.perf_counter()-started,query_gt_used=False,new_encoder_forwards=0,
                          segmentation_benefit='unmeasured',complete_dataset_minutes='unmeasured'))
