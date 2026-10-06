"""CPU candidate: held-out landmark consensus for reference-mask alignment.

Similarities alone can match the wrong arrangement of appearance parts. This
candidate tests one global similarity transform against landmarks not used to
generate that transform. Class transfer of this geometry remains unverified.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import itertools
import time

import numpy as np

from .reference_occupancy import cluster,unit


@dataclass(frozen=True)
class Config:
    reference_modes:int=8
    lloyd_steps:int=5
    purity:float=.9
    peaks_per_mode:int=8
    peak_separation:float=3.0
    minimum_anchor_separation:float=1.0
    scale_min:float=.125
    scale_max:float=8.0
    standardized_support:float=1.5
    support_fraction:float=.75
    median_deviation_floor:float=.01
    pose_nms_iou:float=.5
    geometry_weight:float=.5


def sample(field,points,outside=0.0):
    field=np.asarray(field,dtype=np.float64)
    points=np.asarray(points)
    x,y=points.real,points.imag
    valid=(x>=0)&(y>=0)&(x<=field.shape[1]-1)&(y<=field.shape[0]-1)
    x=np.clip(x,0,field.shape[1]-1);y=np.clip(y,0,field.shape[0]-1)
    x0,y0=np.floor(x).astype(int),np.floor(y).astype(int)
    x1,y1=np.minimum(x0+1,field.shape[1]-1),np.minimum(y0+1,field.shape[0]-1)
    dx,dy=x-x0,y-y0
    value=(field[y0,x0]*(1-dx)*(1-dy)+field[y0,x1]*dx*(1-dy)
           +field[y1,x0]*(1-dx)*dy+field[y1,x1]*dx*dy)
    return np.where(valid,value,outside)


def peaks(response,shape,cfg):
    yy,xx=np.indices(shape)
    positions=(xx+1j*yy).ravel()
    choices=[]
    for row in response:
        available=np.ones(len(row),dtype=bool)
        selected=[]
        for _ in range(min(cfg.peaks_per_mode,len(row))):
            if not available.any():break
            index=int(np.argmax(np.where(available,row,-np.inf)))
            selected.append(positions[index])
            available&=np.abs(positions-positions[index])>=cfg.peak_separation
            available[index]=False
        choices.append(selected)
    return choices


def propose(anchors,choices,cfg):
    result=[]
    for i,j in itertools.combinations(range(len(anchors)),2):
        if abs(anchors[j]-anchors[i])<cfg.minimum_anchor_separation:continue
        for qi,qj in itertools.product(choices[i],choices[j]):
            for mirror in (False,True):
                reference=np.conj(anchors) if mirror else anchors
                a=(qj-qi)/(reference[j]-reference[i])
                if not cfg.scale_min<=abs(a)<=cfg.scale_max:continue
                b=qi-a*reference[i]
                result.append((a,b,mirror,i,j))
    return result


def evaluate(anchors,standardized,poses,cfg):
    k=len(anchors)
    if k<4 or not poses:return []
    coefficients=np.array([pose[0] for pose in poses])
    translations=np.array([pose[1] for pose in poses])
    mirror=np.array([pose[2] for pose in poses],dtype=bool)
    first=np.array([pose[3] for pose in poses],dtype=int)
    second=np.array([pose[4] for pose in poses],dtype=int)
    reference=np.where(mirror[:,None],np.conj(anchors)[None,:],anchors[None,:])
    positions=coefficients[:,None]*reference+translations[:,None]
    values=np.column_stack([sample(standardized[t],positions[:,t],outside=-3.0) for t in range(k)])
    validation=np.ones(values.shape,dtype=bool)
    rows=np.arange(len(poses));validation[rows,first]=validation[rows,second]=False
    support=((values>=cfg.standardized_support)&validation).sum(1)
    held_mean=np.where(validation,values,0).sum(1)/(k-2)
    whole_mean=values.mean(1)
    accepted=((support>=np.ceil(cfg.support_fraction*(k-2)))
              &(held_mean>=cfg.standardized_support)&(whole_mean>=cfg.standardized_support)
              &(values[rows,first]>=cfg.standardized_support)&(values[rows,second]>=cfg.standardized_support))
    scores=[]
    for row in np.flatnonzero(accepted):
        a,b,mirror,i,j=poses[row]
        scores.append(dict(a=a,b=b,mirror=mirror,pair=(i,j),
                           validation_mean=float(held_mean[row]),whole_mean=float(whole_mean[row]),
                           held_out_support=int(support[row])))
    return sorted(scores,key=lambda row:(-row['validation_mean'],-row['whole_mean'],abs(row['a']),
                                          row['a'].real,row['a'].imag,row['b'].real,row['b'].imag,row['mirror']))


def aligned_mask(coverage,shape,pose):
    yy,xx=np.indices(shape)
    inverse=((xx+1j*yy)-pose['b'])/pose['a']
    if pose['mirror']:inverse=np.conj(inverse)
    return sample(coverage,inverse,outside=0.0)


def select_prior(coverage,shape,scored,cfg):
    prior=np.zeros(shape,dtype=np.float64)
    selected=[]
    masks=[]
    seen=set()
    for pose in scored:
        identity=(round(pose['a'].real,6),round(pose['a'].imag,6),
                  round(pose['b'].real,6),round(pose['b'].imag,6),pose['mirror'])
        if identity in seen:continue
        seen.add(identity)
        warped=aligned_mask(coverage,shape,pose)
        mask=warped>.5
        if not mask.any():continue
        redundant=False
        for previous in masks:
            intersection=int((mask&previous).sum())
            union=int((mask|previous).sum())
            if intersection/union>=cfg.pose_nms_iou:
                redundant=True;break
        if redundant:continue
        masks.append(mask);prior=np.maximum(prior,warped)
        selected.append(dict(a=[float(pose['a'].real),float(pose['a'].imag)],
                             b=[float(pose['b'].real),float(pose['b'].imag)],mirror=pose['mirror'],
                             proposal_pair=list(pose['pair']),validation_mean=pose['validation_mean'],
                             whole_mean=pose['whole_mean'],held_out_support=pose['held_out_support']))
    return prior,selected


def predict(q,r,coverage,base,cfg=Config()):
    started=time.perf_counter()
    if (cfg.reference_modes<4 or cfg.lloyd_steps<1 or cfg.peaks_per_mode<1
            or cfg.peak_separation<=0 or cfg.minimum_anchor_separation<=0
            or not 0<cfg.scale_min<=cfg.scale_max or cfg.standardized_support<=0
            or not 0<cfg.support_fraction<=1 or cfg.median_deviation_floor<=0
            or not 0<cfg.pose_nms_iou<=1 or not 0<=cfg.geometry_weight<1 or not 0<cfg.purity<=1):
        raise ValueError('Invalid constellation configuration')
    q,r=unit(q),unit(r)
    cov=np.asarray(coverage,dtype=np.float64)
    base=np.asarray(base,dtype=np.float64)
    if (base.ndim!=2 or cov.shape!=base.shape or len(q)!=base.size or len(r)!=base.size
            or q.shape[1]!=r.shape[1] or not np.isfinite(base).all() or not np.isfinite(cov).all()
            or cov.min()<0 or cov.max()>1 or cov.max()<=0):
        raise ValueError('Aligned query/reference grids and valid reference coverage required')
    foreground=cov.ravel()>=cfg.purity
    if not foreground.any():foreground=cov.ravel()==cov.max()
    centers,labels,_=cluster(r[foreground],cfg.reference_modes,cfg.lloyd_steps,cov.ravel()[foreground])
    k=len(centers)
    yy,xx=np.indices(base.shape)
    positions=(xx+1j*yy).ravel()[foreground]
    anchors=np.array([np.average(positions[labels==i],weights=cov.ravel()[foreground][labels==i])
                      for i in range(k)])
    if k<4 or cfg.geometry_weight==0:
        return dict(field=base.copy(),prior=np.zeros_like(base),bag_field=base.copy(),
                    info=dict(config=asdict(cfg),landmarks=k,abstention=True,poses=[],
                              wall_seconds=time.perf_counter()-started,
                              query_gt_used=False,new_encoder_forwards=0))
    response=np.einsum('nd,kd->kn',q,centers,optimize=False)
    # Distinct landmarks must carry mode-specific evidence: common similarity to
    # every mode is not evidence of the reference's spatial part configuration.
    best=response.argmax(0)
    top=response.max(0)
    second=np.partition(response,-2,axis=0)[-2]
    specific=response-top[None,:]
    specific[best,np.arange(base.size)]=top-second
    center=np.median(specific,axis=1,keepdims=True)
    scale=np.maximum(1.4826*np.median(np.abs(specific-center),axis=1,keepdims=True),cfg.median_deviation_floor)
    standardized=np.clip((specific-center)/scale,-3,3).reshape((k,)+base.shape)
    bag_prior=.5+standardized.max(axis=0)/6
    bag_field=(1-cfg.geometry_weight)*np.clip(base,0,1)+cfg.geometry_weight*bag_prior
    choices=peaks(specific,base.shape,cfg)
    poses=propose(anchors,choices,cfg)
    scored=evaluate(anchors,standardized,poses,cfg)
    prior,selected=select_prior(cov,base.shape,scored,cfg)
    if selected:
        field=(1-cfg.geometry_weight)*np.clip(base,0,1)+cfg.geometry_weight*prior
    else:
        field=base.copy()
    info=dict(config=asdict(cfg),landmarks=k,anchors=[[float(x.real),float(x.imag)] for x in anchors],
              proposed_poses=len(poses),consensus_poses=len(scored),retained_poses=len(selected),
              poses=selected,abstention=not bool(selected),
              clipped_base_values=int(((base<0)|(base>1)).sum()) if selected else 0,
              added_tokens=int(((field>.5)&(base<=.5)).sum()),
              deleted_tokens=int(((field<=.5)&(base>.5)).sum()),
              wall_seconds=time.perf_counter()-started,query_gt_used=False,new_encoder_forwards=0,
              segmentation_benefit='unmeasured',complete_dataset_minutes='unmeasured')
    return dict(field=field,prior=prior,bag_field=bag_field,info=info)
