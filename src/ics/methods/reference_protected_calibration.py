"""Reference-conditioned calibration between two existing score coordinates.

Neither coordinate nor the role tendency is assumed to be calibrated probability.
Read both coordinates to pixels first, then use the reference FG/BG tendency to
select between their lower/upper envelopes. A uniform control matches the exact
L1 upward correction budget. This is a development hypothesis, not a gain claim.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _read(values, shape):
    if values.shape == shape:
        return values.copy()
    return F.interpolate(torch.from_numpy(values)[None,None],shape,
                         mode='bilinear',align_corners=False)[0,0].numpy()


def predict(native, anchor, role, *, target_hw=(1024,1024)):
    inputs=[np.ascontiguousarray(x,dtype=np.float32) for x in (native,anchor,role)]
    if any(x.ndim!=2 or x.shape!=inputs[0].shape or not np.isfinite(x).all() for x in inputs):
        raise ValueError('Require matching finite 2D coordinate and reference-role fields')
    if inputs[2].min()<0 or inputs[2].max()>1:
        raise ValueError('Reference role tendency must be in [0,1]')
    if len(target_hw)!=2 or min(target_hw)<1:
        raise ValueError('Require positive pixel output geometry')
    n,a,p=(_read(x,tuple(target_hw)) for x in inputs)
    low,high=np.minimum(n,a),np.maximum(n,a)
    gap=high-low
    budget=float(gap.sum(dtype=np.float64))
    # Reference p is used without tuning, discretization or query labels.
    weighted=(p*gap).astype(np.float32)
    gamma=float(weighted.sum(dtype=np.float64)/budget) if budget else .5
    protected=np.where(p==1,high,np.where(p==0,low,low+weighted)).astype(np.float32)
    uniform=(high.copy() if gamma==1 else low.copy() if gamma==0
             else low+np.float32(gamma)*gap)
    actual_role=float((protected-low).sum(dtype=np.float64))
    actual_uniform=float((uniform-low).sum(dtype=np.float64))
    return dict(protected=protected,uniform=uniform,
                info=dict(gamma=gamma,available_envelope_budget=budget,
                    role_correction_l1=actual_role,uniform_correction_l1=actual_uniform,
                    correction_budget_relative_difference=abs(actual_role-actual_uniform)/max(actual_role,1e-12),
                    role_min=float(p.min()),role_max=float(p.max()),
                    query_GT_in_inference=False,encoder_calls=0,
                    target_hw=list(target_hw),interpolate_before_envelopes=True))
