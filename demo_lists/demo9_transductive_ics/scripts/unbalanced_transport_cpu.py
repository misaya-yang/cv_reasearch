#!/usr/bin/env python3
"""CPU-only UOT algebra/identity/quota diagnostics; no task-gain experiment."""
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PATH=Path(__file__).resolve().parents[1]/'tics/unbalanced_transport.py'
spec=importlib.util.spec_from_file_location('unbalanced_transport',PATH)
uot=importlib.util.module_from_spec(spec);spec.loader.exec_module(uot)


def main():
    torch.set_num_threads(1);torch.manual_seed(2044)
    dtype=torch.float64
    ref=F.normalize(torch.randn(12,5,dtype=dtype),dim=1);fg=torch.arange(12)%3==0
    query=F.normalize(torch.randn(9,5,dtype=dtype),dim=1)
    shared=dict(max_iterations=1000,tolerance=1e-10,epsilon=.2,rho_query=.15,rho_reference=.12)
    a,da=uot.transport_scores(ref,fg,query,chunk_size=3,**shared)
    b,db=uot.transport_scores(ref,fg,query,chunk_size=20,**shared)
    assert torch.allclose(a,b,atol=1e-10,rtol=1e-10)
    assert abs(da['convex_primal_value']-db['convex_primal_value'])<1e-10
    assert da['unbalanced_stationarity_max_abs']<1e-8
    # Exact rho=0 reduction to full-anchor KDE under SAME class prior/kernel.
    k,dk=uot.full_anchor_kde_scores(ref,fg,query,epsilon=.2,chunk_size=3,tolerance=1e-10)
    z,dz=uot.unbalanced_transport_scores(ref,fg,query,epsilon=.2,rho_query=0,rho_reference=0,chunk_size=3,tolerance=1e-10)
    assert torch.allclose(k,z,atol=1e-12)
    weights=torch.where(fg,torch.full((12,),.5/int(fg.sum()),dtype=dtype),torch.full((12,),.5/int((~fg).sum()),dtype=dtype))
    kernel=torch.exp(-(1-query@ref.T)/.2)*weights[None]
    direct=kernel[:,fg].sum(1)/kernel.sum(1)
    assert torch.allclose(k,direct,atol=1e-12)
    # Same descriptors/labels, permuted support order or query order gives identity.
    pi=torch.randperm(len(ref));qi=torch.randperm(len(query))
    perm,_=uot.transport_scores(ref[pi],fg[pi],query[qi],chunk_size=4,**shared)
    assert torch.allclose(perm,a[qi],atol=1e-10)
    # Clean toy changes true query FG share; truth is evaluator-only, NEVER passed.
    reference=torch.tensor([[1.,0.]]*2+[[-1.,0.]]*8,dtype=dtype)
    label=torch.tensor([True]*2+[False]*8);mass=[];failures={'unbalanced_soft_quota':0,'balanced_soft_quota':0}
    for count in (0,2,10,18,20):
        x=torch.tensor([[1.,0.]]*count+[[-1.,0.]]*(20-count),dtype=dtype)
        su,du=uot.unbalanced_transport_scores(reference,label,x,chunk_size=4,max_iterations=1000,tolerance=1e-10)
        sb,db=uot.balanced_transport_scores(reference,label,x,chunk_size=4,max_iterations=1000,tolerance=1e-10)
        sk,dk=uot.full_anchor_kde_scores(reference,label,x,chunk_size=4,tolerance=1e-10)
        expected=count/20
        assert abs(float(su.mean())-expected)<.01
        assert abs(float(sb.mean())-.5)<1e-8
        assert db['row_marginal_L1']<1e-8 and db['column_marginal_L1']<1e-8
        if abs(expected-.5)>=.4:
            failures['unbalanced_soft_quota']+=int(abs(float(su.mean())-expected)>.05)
            failures['balanced_soft_quota']+=int(abs(float(sb.mean())-expected)>.05)
        mass.append(dict(toy_query_FG=expected,KDE_mean_FG=float(sk.mean()),UOT_mean_FG=float(su.mean()),
            balanced_mean_FG=float(sb.mean()),UOT_transported_FG=du['transported_FG_fraction'],UOT_total_mass=du['transported_mass']))
    # Identity labels on clearly separated anchors are recovered, with no foreground-size input.
    identity,_=uot.unbalanced_transport_scores(reference,label,reference,chunk_size=3,max_iterations=1000,tolerance=1e-10)
    assert torch.equal(identity>.5,label)
    failed=False
    try:uot.transport_scores(ref,fg,query,max_iterations=1,tolerance=1e-12)
    except uot.TransportConvergenceError:failed=True
    assert failed,'unconverged solution silently accepted'
    failed=False
    try:uot.transport_scores(ref,fg,query,max_ram_bytes=32)
    except MemoryError:failed=True
    assert failed,'RAM cap ignored'
    print(json.dumps(dict(state='CPU_NUMERICS_PASSED',chunk_parity_max_abs=float((a-b).abs().max()),
        stationarity=da['unbalanced_stationarity_max_abs'],rho0_KDE_identity=True,support_query_permutation_identity=True,
        clear_feature_identity_labels=True,explicit_convergence_failure=True,explicit_RAM_failure=True,
        induced_foreground_quota_failure_counts=failures,toy_mass=mass,
        limitation='Clean toy verifies solver/protocol only; zero real episodes, no task mIoU or empirical gain'),indent=2))


if __name__=='__main__':main()
