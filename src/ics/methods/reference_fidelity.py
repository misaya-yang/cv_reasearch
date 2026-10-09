"""Positive-agreement fidelity, retaining the actual parent graph and unary."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .rcg import rank


def build(parent,a,s,guide,unary):
    if not sparse.isspmatrix_csr(parent) or parent.dtype!=np.float64:
        raise ValueError('Use the actual saved FP64 CSR parent H')
    a,s,g,y=(np.asarray(v).reshape(-1) for v in (a,s,guide,unary))
    if a.dtype!=np.float64 or y.dtype!=np.float64 or s.dtype!=np.float32 or g.dtype!=np.float32:
        raise ValueError('Preserve parent FP64 A/unary and FP32 s/guide')
    if any(v.shape!=a.shape or not np.isfinite(v).all() for v in (a,s,g,y)) or parent.shape!=(len(a),len(a)) or (a<=0).any():
        raise ValueError('Invalid parent fields')
    coo=parent.tocoo();off=coo.row!=coo.col
    rows,cols=coo.row[off],coo.col[off];weights=-coo.data[off]/16
    if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid graph weights')
    w=sparse.csr_matrix((weights,(rows,cols)),shape=parent.shape);asym=w-w.T
    if asym.nnz and np.abs(asym.data).max()>1e-12:raise ValueError('Asymmetric parent graph')
    degree=np.bincount(rows,weights=weights,minlength=len(a))
    group=(s>.5)&(y>.5);n=int(group.sum());u=np.zeros(len(a),np.float32)
    if n:u[group]=rank(g[group])
    mass=float(degree[group].sum());constant=bool(n and np.ptp(g[group])==0)
    fallback=('nG_less_than2_identity' if n<2 else 'constant_G_guide_identity' if constant else
              'zero_G_degree_identity' if mass==0 else None)
    delta=np.zeros(len(a),np.float64);control=delta.copy();eta=0.
    if fallback is None:
        delta=16*degree*u.astype(np.float64)
        eta=float((degree[group]*u[group].astype(np.float64)).sum()/mass)
        control[group]=16*degree[group]*eta
    budget_error=abs(float(delta.sum()-control.sum()))
    if budget_error>max(1e-12,1e-12*float(delta.sum())):raise RuntimeError('Total fidelity increment mismatch')
    result=dict(arrays=dict(group=group,u=u,degree=degree,candidate_delta_a=delta,matched_delta_a=control),
        diagnostics=dict(n_group=n,guide_unique_in_G=int(np.unique(g[group]).size),fallback=fallback,
            eta=eta,degree_mass_G=mass,candidate_delta_sum=float(delta.sum()),matched_delta_sum=float(control.sum()),
            mass_match_error=budget_error,candidate_rhs_increment_sum=float(delta@y),matched_rhs_increment_sum=float(control@y),
            same_rhs_increment_presumed=False,graph_unchanged=True,unary_guide_unchanged=True,
            outside_G_A_unchanged=True,constant_guide_identity_is_explicit_fallback=True,encoder_calls=0,query_GT_in_inference=False))
    for kind,change in [('candidate',delta),('matched',control)]:
        new=parent.copy() if not np.count_nonzero(change) else (parent+sparse.diags(change)).tocsr()
        result[kind+'_H']=new;result[kind+'_Hdiff']=(new-parent).tocsr()
        result[kind+'_a']=a+change
        result['arrays'][kind+'_a']=a+change
        result['arrays'][kind+'_rhs']=(a+change)*y
    return result
