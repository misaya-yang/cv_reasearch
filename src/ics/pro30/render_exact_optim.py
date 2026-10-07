"""Bounded exact CPU100 renderer optimization; original remains a control.

No coordinate, operation grouping, dtype-conversion or threshold change. Only
the coordinate arrays broadcast and the large outputs are evaluated in row tiles.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial
import time
import numpy as np

from ics.cpu100.common import render as original_render
from .common import Result,readonly

_AXES=OrderedDict()


def _axis(index,source_length):
    index=np.clip(index,0,source_length-1)
    lower=np.floor(index).astype(int);upper=np.minimum(lower+1,source_length-1)
    return lower,upper,index-lower


def _resize_axes(source_hw,target_hw):
    key=(tuple(source_hw),tuple(target_hw))
    if key in _AXES:
        axes=_AXES.pop(key);_AXES[key]=axes;return axes
    # Keep the exact source expression order: multiply by source integer BEFORE
    # dividing by target integer. A precomputed floating scale is not equivalent.
    y=(np.arange(target_hw[0])+.5)*source_hw[0]/target_hw[0]-.5
    x=(np.arange(target_hw[1])+.5)*source_hw[1]/target_hw[1]-.5
    axes=(_axis(y,source_hw[0]),_axis(x,source_hw[1]));_AXES[key]=axes
    while len(_AXES)>16:_AXES.popitem(last=False)
    return axes


def _sample_rows(array,yaxis,xaxis,first,last):
    y0,y1,fy=yaxis;x0,x1,fx=xaxis
    y0=y0[first:last,None];y1=y1[first:last,None];fy=fy[first:last,None]
    x0=x0[None,:];x1=x1[None,:];fx=fx[None,:]
    # Literal sample_grid grouping, including each multiply and each addition.
    return ((1-fy)*((1-fx)*array[y0,x0]+fx*array[y0,x1])
            +fy*((1-fx)*array[y1,x0]+fx*array[y1,x1]))


def resize_exact(array,shape,*,block_rows=64,threshold=None):
    array=np.asarray(array)
    if array.ndim!=2:raise ValueError('This bounded renderer only accepts scalar fields')
    if block_rows<1:raise ValueError('Positive row tile required')
    yaxis,xaxis=_resize_axes(array.shape,shape)
    out=np.empty(shape,dtype=float if threshold is None else bool)
    for first in range(0,shape[0],block_rows):
        last=min(first+block_rows,shape[0]);value=_sample_rows(array,yaxis,xaxis,first,last)
        out[first:last]=value if threshold is None else value>threshold
    return out


def render_exact_blocked(ep,result,*,block_rows=64):
    margin=np.asarray(result.margin,float).reshape(ep.q_hw)
    if not np.isfinite(margin).all():raise ValueError('Method must return finite complete signed margin')
    g=ep.query_geometry
    if g:
        view=g['view_side'];sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
        y=(oy+(np.arange(64)+.5)*sh/64)*ep.q_hw[0]/view-.5
        x=(ox+(np.arange(64)+.5)*sw/64)*ep.q_hw[1]/view-.5
        field=_sample_rows(margin,_axis(y,margin.shape[0]),_axis(x,margin.shape[1]),0,64)
    else:field=resize_exact(margin,(64,64),block_rows=block_rows)
    field=field.astype(np.float32)
    work=resize_exact(field,(1024,1024),block_rows=block_rows,threshold=0.)
    original=resize_exact(work.astype(np.float32),ep.original_shape,block_rows=block_rows,threshold=.5)
    return dict(margin=field,work=work,original=original)


def finish_exact(ep,margin,method_id,info=None,*,invalid_value=None):
    from ics.cpu100.common import Result as CPUResult
    start=time.perf_counter();field=np.asarray(margin,float).reshape(ep.q_hw).copy()
    if invalid_value is not None:field.ravel()[ep.q_valid<=0]=invalid_value
    output=render_exact_blocked(ep,CPUResult(field,{}))
    metadata=dict(info or {},method_id=method_id,query_GT_read=False,
        renderer='CPU100 exact blocked64: physical64 FP32→bilinear1024 >0→binary bilinear original >.5',
        renderer_seconds=time.perf_counter()-start,work_mask_shape=[1024,1024],original_mask_shape=list(ep.original_shape),
        complete_all_instances=True,renderer_optimization='broadcast axes and64row tiles; unchanged arithmetic andthresholds')
    return Result(field,0.,readonly(output['original']),metadata)


def m06_exact_blocked(ep,mode='skew'):
    from .differential_06 import differential
    return differential(ep,mode=mode,renderer=finish_exact)


METHODS={'PRO30_M06':m06_exact_blocked}
CONTROLS={}
for mode in ('zero_e','pointwise_quadratic','center_neighbor_bilinear','divergence_unary','gradient_e','reverse','permuted_K'):
    CONTROLS['PRO30_M06__'+mode]=partial(m06_exact_blocked,mode=mode)

def original_m06_control(ep):
    from .differential_06 import differential
    out=differential(ep)
    out.info.update(method_id='PRO30_M06__original_renderer',control_only=True,identical_inference_object=True)
    return out

CONTROLS['PRO30_M06__original_renderer']=original_m06_control
