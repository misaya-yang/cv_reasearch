"""Localized conditional-CDF bilateral graph gating, with matched edge mass.

Only the graph changes. The saved parent unary/A and its diagonal floating-point
residual are preserved. This is classical rank gating, not new task information.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .rcg import rank


def conditional_cdf(guide, group):
    g=np.asarray(guide,dtype=np.float32).reshape(-1);group=np.asarray(group,bool).reshape(-1)
    if g.shape!=group.shape or not np.isfinite(g).all():raise ValueError('Invalid guide/group')
    n=int(group.sum())
    if not n:return np.full(len(g),.5,np.float32)
    values=np.sort(g[group]);lo=np.searchsorted(values,g,side='left');hi=np.searchsorted(values,g,side='right')
    u=((lo.astype(np.float64)+.5*(hi-lo))/n).astype(np.float32)
    if not np.array_equal(u[group],rank(g[group])):raise RuntimeError('Conditional CDF must equal locked rank in G')
    return u


def _update(parent,rows,cols,delta):
    n=parent.shape[0];degree=np.bincount(rows,weights=delta,minlength=n).astype(np.float64)
    if not np.count_nonzero(delta):return parent.copy(),degree,sparse.csr_matrix(parent.shape,dtype=np.float64)
    dw=sparse.csr_matrix((delta,(rows,cols)),shape=parent.shape)
    new=parent+16*(sparse.diags(degree)-dw)
    return new,degree,(new-parent).tocsr()


def build(parent,s,guide):
    if not sparse.isspmatrix_csr(parent) or parent.dtype!=np.float64 or parent.shape[0]!=parent.shape[1]:
        raise ValueError('Use the actual saved FP64 CSR parent H')
    s=np.asarray(s,dtype=np.float32).reshape(-1);g=np.asarray(guide,dtype=np.float32).reshape(-1)
    if s.shape!=g.shape or len(s)!=parent.shape[0] or not np.isfinite(s).all() or not np.isfinite(g).all():raise ValueError('Invalid source fields')
    group=s>.5;u=conditional_cdf(g,group);n=int(group.sum())
    coo=parent.tocoo();off=coo.row!=coo.col;rows=coo.row[off].copy();cols=coo.col[off].copy()
    weights=(-coo.data[off]/16).astype(np.float64)
    if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Parent off-diagonals do not define nonnegative W')
    w=sparse.csr_matrix((weights,(rows,cols)),shape=parent.shape)
    asymmetry=w-w.T
    if asymmetry.nnz and np.abs(asymmetry.data).max()>1e-12:raise ValueError('Parent graph is asymmetric')
    affected=group[rows]|group[cols];factors=np.ones(len(weights),np.float64)
    constant=bool(n and np.ptp(g[group])==0)
    fallback='nG_less_than2_identity' if n<2 else 'constant_G_guide_identity' if constant else None
    if fallback is None:
        uu=u.astype(np.float64);factors[affected]=1-np.abs(uu[rows[affected]]-uu[cols[affected]])
    candidate_weights=weights*factors
    mass=float(weights[affected].sum(dtype=np.float64));candidate_mass=float(candidate_weights[affected].sum(dtype=np.float64))
    eta=1. if mass==0 else candidate_mass/mass
    matched_weights=weights.copy();matched_weights[affected]*=eta
    if fallback is not None:matched_weights=weights.copy();eta=1.
    matched_mass=float(matched_weights[affected].sum(dtype=np.float64))
    if abs(matched_mass-candidate_mass)>max(1e-12,1e-12*mass):raise RuntimeError('Localized affected-edge mass does not close')
    if not np.array_equal(candidate_weights[~affected],weights[~affected]) or not np.array_equal(matched_weights[~affected],weights[~affected]):
        raise RuntimeError('Edges outside E changed')
    candidate,degree_candidate,diff_candidate=_update(parent,rows,cols,candidate_weights-weights)
    matched,degree_matched,diff_matched=_update(parent,rows,cols,matched_weights-weights)
    return dict(candidate_H=candidate,matched_H=matched,candidate_Hdiff=diff_candidate,matched_Hdiff=diff_matched,
        arrays=dict(group=group,u=u,edge_rows=rows,edge_cols=cols,parent_weights=weights,factors=factors,affected_edges=affected,
                    candidate_weights=candidate_weights,matched_weights=matched_weights,
                    candidate_delta_degree=degree_candidate,matched_delta_degree=degree_matched),
        diagnostics=dict(n_group=n,fallback=fallback,constant_G_guide=constant,cdf_defined=bool(n),
            cdf_dtype='FP32 exactly matching locked rank; weight arithmetic FP64',affected_directed_edges=int(affected.sum()),
            eta=eta,parent_affected_mass=mass,candidate_affected_mass=candidate_mass,matched_affected_mass=matched_mass,
            edge_mass_absolute_error=abs(matched_mass-candidate_mass),outside_E_bit_exact=True,
            parent_diagonal_residual_preserved='Hnew=Hparent+16*(diag(sum deltaW)-deltaW); never rebuild original H',
            unary_unchanged=True,a_unchanged=True,no_degree_renormalization=True,encoder_calls=0,query_GT_in_inference=False))
