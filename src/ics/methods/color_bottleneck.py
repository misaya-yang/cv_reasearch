"""CPU candidate: seeded query-color bottleneck basins plus frozen base evidence.

Minimum bottleneck means the largest local edge contrast along a path, not
the range of RGB values along that path. These differ on smooth gradients.
No empirical segmentation gain, semantic novelty or minute budget is asserted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    foreground_seed: float = 0.75
    background_seed: float = 0.25
    color_weight: float = 0.5


def resize_field(field, shape):
    """Bilinear half-pixel coordinates and replicated borders, in float64."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all() or min(field.shape) < 1:
        raise ValueError('Finite nonempty 2D field required')
    if field.shape == tuple(shape):
        return field.copy()
    rows = np.clip((np.arange(shape[0]) + .5) * field.shape[0] / shape[0] - .5, 0, field.shape[0]-1)
    cols = np.clip((np.arange(shape[1]) + .5) * field.shape[1] / shape[1] - .5, 0, field.shape[1]-1)
    r0, c0 = rows.astype(int), cols.astype(int)
    r1, c1 = np.minimum(r0+1,field.shape[0]-1), np.minimum(c0+1,field.shape[1]-1)
    ry, cx = rows-r0, cols-c0
    horizontal = field[:,c0]*(1-cx)[None,:]+field[:,c1]*cx[None,:]
    return horizontal[r0]*(1-ry)[:,None]+horizontal[r1]*ry[:,None]


def color_edges(rgb):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('H,W,3 RGB required')
    if rgb.dtype==np.uint8:
        colors=rgb.astype(np.float64)/255
    else:
        colors=rgb.astype(np.float64)
        if not np.isfinite(colors).all() or colors.min()<0 or colors.max()>1:
            raise ValueError('RGB floats must be finite and in [0,1]')
    h,w=colors.shape[:2]
    ids=np.arange(h*w).reshape(h,w)
    left=np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()]
    right=np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
    flat=colors.reshape(-1,3)
    cost=np.sqrt(np.sum((flat[left]-flat[right])**2,axis=1)/3)
    return left,right,cost


def adjacency(n,left,right,cost):
    left,right=np.asarray(left),np.asarray(right)
    cost=np.asarray(cost,dtype=np.float64)
    if (n<1 or left.dtype.kind not in 'iu' or right.dtype.kind not in 'iu'
            or left.ndim!=1 or right.shape!=left.shape or cost.shape!=left.shape
            or not np.isfinite(cost).all() or np.any(cost<0)
            or np.any(left<0) or np.any(right>=n) or np.any(left>=right)):
        raise ValueError('Canonical undirected nonnegative graph required')
    source=np.r_[left,right]
    order=np.argsort(source,kind='stable')
    degree=np.bincount(source,minlength=n)
    indptr=np.r_[0,np.cumsum(degree)]
    return indptr,np.r_[right,left][order],np.r_[cost,cost][order]


def minimum_bottleneck(graph,seeds):
    """Multi-source Dijkstra over the min/max path semiring."""
    indptr,neighbors,cost=graph
    n=len(indptr)-1
    seeds=np.asarray(seeds)
    if seeds.dtype!=np.bool_ or seeds.shape!=(n,) or not seeds.any():
        raise ValueError('Nonempty boolean seed mask required')
    distance=np.full(n,np.inf)
    distance[seeds]=0
    heap=[(0.0,int(i)) for i in np.flatnonzero(seeds)]
    heapq.heapify(heap)
    while heap:
        barrier,node=heapq.heappop(heap)
        if barrier>distance[node]:
            continue
        for offset in range(indptr[node],indptr[node+1]):
            nxt=int(neighbors[offset])
            candidate=max(barrier,float(cost[offset]))
            if candidate<distance[nxt]:
                distance[nxt]=candidate
                heapq.heappush(heap,(candidate,nxt))
    return distance


def predict(rgb,base_field,cfg=Config()):
    started=time.perf_counter()
    if not (0<=cfg.background_seed<cfg.foreground_seed<=1 and 0<=cfg.color_weight<1):
        raise ValueError('Ordered seed thresholds and color_weight in [0,1) required')
    left,right,cost=color_edges(rgb)
    shape=np.asarray(rgb).shape[:2]
    original=resize_field(base_field,shape)
    base=np.clip(original,0,1)
    n=base.size
    if float(np.ptp(base))<1e-12 or cfg.color_weight==0:
        return dict(field=base,color_field=base.copy(),intermediate_base=original,
                    info=dict(config=asdict(cfg),constant_or_zero_weight=True,
                              wall_seconds=time.perf_counter()-started,
                              query_gt_used=False,new_encoder_forwards=0))
    foreground=(base.ravel()>=cfg.foreground_seed)
    background=(base.ravel()<=cfg.background_seed)
    if not foreground.any() or not background.any():
        return dict(field=base,color_field=base.copy(),intermediate_base=original,
                    info=dict(config=asdict(cfg),missing_seed_abstention=True,
                              foreground_seeds=int(foreground.sum()),background_seeds=int(background.sum()),
                              wall_seconds=time.perf_counter()-started,
                              query_gt_used=False,new_encoder_forwards=0))
    graph=adjacency(n,left,right,cost)
    df=minimum_bottleneck(graph,foreground)
    db=minimum_bottleneck(graph,background)
    if not np.isfinite(df).all() or not np.isfinite(db).all():
        raise RuntimeError('Unexpected disconnected RGB grid')
    denominator=df+db
    # Exact zero/zero ties contain no color preference and keep the base.
    color=np.divide(db,denominator,out=base.ravel().copy(),where=denominator>0).reshape(shape)
    field=(1-cfg.color_weight)*base+cfg.color_weight*color
    info=dict(config=asdict(cfg),rgb_grid=list(shape),undirected_edges=len(left),
              foreground_seeds=int(foreground.sum()),background_seeds=int(background.sum()),
              missing_seed_abstention=False,
              zero_barrier_ties=int((denominator==0).sum()),
              added_grid_pixels=int(((field>.5)&(base<=.5)).sum()),
              deleted_grid_pixels=int(((field<=.5)&(base>.5)).sum()),
              clipped_base_values=int(((original<0)|(original>1)).sum()),
              wall_seconds=time.perf_counter()-started,
              query_gt_used=False,new_encoder_forwards=0,
              segmentation_benefit='unmeasured',complete_dataset_minutes='unmeasured')
    return dict(field=field,color_field=color,intermediate_base=original,info=info)
