"""Keep a reference atom's physical identity across two visual observations.

Joint and factorized class-conditional KDEs share the exact role marginals.
The joint density binds both query views to the same reference location;
the factorized density can use different locations for each view. A coverage-
stratified permutation destroys binding while preserving each view/role bag.
This is a falsifiable information probe, not a proven original segmenter.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

TEMPERATURE=.07
ARMS=('whole_reference','factorized','bound','role_marginal_shuffle')


def unit(raw,basis):
    if raw.dtype!=torch.float32 or raw.ndim!=2 or raw.device.type!='cpu' or not torch.isfinite(raw).all():
        raise ValueError('Require preserved CPU FP32 feature rows')
    x=F.normalize(raw,dim=1)
    # Fixed positional subspace, common to every view and every arm.
    x=x-(x@basis)@basis.T
    if not (x.norm(dim=1)>0).all():raise ValueError('Degenerate projected direction')
    return F.normalize(x,dim=1)


def coverage_permutation(coverage):
    c=np.asarray(coverage);permutation=np.arange(len(c));rng=np.random.default_rng(0)
    groups=[np.flatnonzero(c==v) for v in np.unique(c)]
    # Index-based group order makes the same permutation survive exchanging
    # foreground and background, c -> 1-c.
    groups.sort(key=lambda ids:int(ids[0]))
    for ids in groups:permutation[ids]=rng.permutation(ids)
    assert np.array_equal(c[permutation],c)
    changed=permutation!=np.arange(len(c))
    return permutation,dict(changed_positions=int(changed.sum()),strata=len(groups),
        changed_fg_mass_fraction=float(c[changed].sum()/c.sum()),
        changed_bg_mass_fraction=float((1-c[changed]).sum()/(1-c).sum()))


def _weights(c,device):
    c=np.asarray(c,dtype=np.float64)
    if c.ndim!=1 or not np.isfinite(c).all() or c.min()<0 or c.max()>1 or c.sum()==0 or (1-c).sum()==0:
        raise ValueError('Both lawful reference roles must have positive physical mass')
    result=[]
    for weights in (c,1-c):
        log=np.full(len(c),-np.inf,dtype=np.float32);keep=weights>0
        log[keep]=np.log(weights[keep]/weights.sum()).astype(np.float32)
        result.append(torch.from_numpy(log).to(device))
    return result


@torch.inference_mode()
def fields(r0,r1,coverage,q0,q1,whole_reference,whole_coverage,*,device='mps',chunk=2048):
    if r0.shape!=r1.shape or q0.shape!=q1.shape or r0.shape[1]!=q0.shape[1]:
        raise ValueError('Require matching paired view dimensions')
    for x in (r0,r1,q0,q1,whole_reference):
        if x.dtype!=torch.float32 or x.ndim!=2 or not torch.isfinite(x).all() or (x.norm(dim=1)-1).abs().max()>1e-3:
            raise ValueError('Require actual finite unit view descriptors')
    permutation,shuffle_info=coverage_permutation(coverage)
    fg,bg=_weights(coverage,device);full_fg,full_bg=_weights(whole_coverage,device)
    a,b,full=r0.to(device),r1.to(device),whole_reference.to(device)
    perm=torch.from_numpy(permutation).to(device)
    output={name:[] for name in ARMS};marginal_error=0.
    for start in range(0,len(q0),chunk):
        x,y=q0[start:start+chunk].to(device),q1[start:start+chunk].to(device)
        l0=x@a.T/TEMPERATURE;l1=y@b.T/TEMPERATURE
        def ratio(logits,left,right):
            return torch.logsumexp(logits+left,dim=1)-torch.logsumexp(logits+right,dim=1)
        marginal0=ratio(l0,fg,bg);marginal1=ratio(l1,fg,bg)
        output['factorized'].append((marginal0+marginal1).cpu())
        output['bound'].append(ratio(l0+l1,fg,bg).cpu())
        shuffled=l1[:,perm]
        output['role_marginal_shuffle'].append(ratio(l0+shuffled,fg,bg).cpu())
        marginal_error=max(marginal_error,float((marginal1-ratio(shuffled,fg,bg)).abs().max().cpu()))
        output['whole_reference'].append(ratio(x@full.T/TEMPERATURE,full_fg,full_bg).cpu())
    if marginal_error>2e-5:raise RuntimeError('Coverage-stratified shuffle changed a role marginal')
    values={name:torch.cat(parts).numpy() for name,parts in output.items()}
    if not all(np.isfinite(x).all() for x in values.values()):raise RuntimeError('Nonfinite density ratio')
    return values,dict(shuffle=shuffle_info,max_role_marginal_shuffle_error=marginal_error,
        temperature=TEMPERATURE,reference_fg_mass=float(np.asarray(coverage).sum()),
        reference_bg_mass=float((1-np.asarray(coverage)).sum()),
        meaning='log class-conditional joint density ratio; not calibrated query posterior',
        pairing_information='bound-factorized is the FG versus BG difference of cross-view dependence terms')
