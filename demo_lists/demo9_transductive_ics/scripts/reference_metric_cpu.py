#!/usr/bin/env python3
"""CPU-only algorithm/gradient checks; synthetic data is not segmentation gain."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_metric import (MetricConfig, preference, protected_step,
    metric_distances, density_ratio, local_triplets, ReferenceMetric, correct_host_field)


@torch.no_grad()
def original_line_search(design,margins,mass,config):
    """Frozen pre-fix algorithm, diagnostic only; never a deployment fallback."""
    a,m,p=design.double(),margins.double(),mass.double()
    value=torch.ones(a.shape[1],dtype=torch.float64)
    eye=torch.eye(len(value),dtype=torch.float64)
    def evaluate(v):
        z=(config.margin-m-a@(v-1))/config.temperature;s=z.sigmoid()
        loss=(p*F.softplus(z)).sum()+config.regularization/2*(v-1).square().sum()
        grad=config.regularization*(v-1)-a.T@(p*s)/config.temperature
        hessian=config.regularization*eye+a.T@((p*s*(1-s)/config.temperature**2)[:,None]*a)
        return loss,grad,hessian
    for iteration in range(config.max_iterations):
        loss,grad,hessian=evaluate(value)
        if float(grad.abs().max())<=config.tolerance:
            return dict(converged=True,maxgrad=float(grad.abs().max()),iterations=iteration)
        step=torch.linalg.solve(hessian,grad);directional=(grad*step).sum();rate=1.
        for _ in range(40):
            candidate=value-rate*step
            if evaluate(candidate)[0]<=loss-1e-4*rate*directional:
                value=candidate;break
            rate*=.5
        else:break
    residual=float(evaluate(value)[1].abs().max())
    return dict(converged=residual<=config.tolerance,maxgrad=residual,iterations=iteration+1)


def near_stationary_check(path=None):
    """D1024/r32 FP32 inputs with the UNCHANGED default F64 stationarity gate."""
    config=MetricConfig()
    if path is None:
        # This seeded physical triplet design stalled above the original gate
        # on the local CPU BLAS before the fix. Its exact objective reduction
        # rounding can vary across BLAS builds; the repaired gate must pass all.
        generator=torch.Generator(device='cpu').manual_seed(129)
        features=F.normalize(torch.randn(128,1024,generator=generator),dim=1)
        directions=torch.linalg.qr(torch.randn(1024,32,generator=generator),mode='reduced')[0]
        indices=torch.randint(0,128,(4096,3),generator=generator)
        positive=features[indices[:,0]]-features[indices[:,1]]
        negative=features[indices[:,0]]-features[indices[:,2]]
        design=(negative@directions).square()-(positive@directions).square()
        margins=negative.square().sum(1)-positive.square().sum(1)
        mass=torch.full((4096,),1/4096)
        source='synthetic_seed129_full1024_physical_differences'
    else:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        design,margins,mass=(saved[k] for k in ('design','margins','mass'))
        if 'config' in saved:config=MetricConfig(**saved['config'])
        source='explicit_failed_design_no_feature_extraction'
    assert design.dtype==torch.float32 and design.shape[1]==32
    assert config.tolerance==1e-9,'This regression must not loosen the original production gate'
    baseline=original_line_search(design,margins,mass,config)
    design=design.detach().requires_grad_()
    result=preference(design,margins,mass,config)
    # Verify the actual F64 solver iterate, not its required FP32 output cast.
    a,m,p,v,_=result.grad_fn.saved_tensors
    sigmoid=((config.margin-m-a@(v-1))/config.temperature).sigmoid()
    gradient=config.regularization*(v-1)-a.T@(p*sigmoid)/config.temperature
    residual=float(gradient.detach().abs().max())
    assert residual<=config.tolerance
    result.square().mean().backward()
    assert torch.isfinite(design.grad).all()
    return dict(source=source,input_dtype='float32',solver_dtype='float64',rank=32,
        stationarity_threshold_unchanged=config.tolerance,original_line_search=baseline,
        repaired_maxgrad=residual,backward_finite=True,
        objective_regularizer_temperature_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path)
    p.add_argument('--regression-design',type=Path,help='Optional small explicit failed design/margins/mass .pt; no models/images')
    a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2048)
    near_stationary=near_stationary_check(a.regression_design)
    config=replace(MetricConfig(),rank=3,regularization=1.,tolerance=1e-11)
    design=(torch.randn(11,3,dtype=torch.float64)*.12).requires_grad_()
    margins=(torch.randn(11,dtype=torch.float64)*.08).requires_grad_()
    mass=torch.full((11,),1/11,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,y,z:preference(x,y,z,config),(design,margins,mass),eps=1e-5,atol=2e-5,rtol=2e-4)
    v=preference(design,margins,mass,config);w,alpha=protected_step(v,design,margins,config)
    guard=margins>=config.margin
    assert w.min()>=config.minimum-1e-10 and w.max()<=config.maximum+1e-10
    assert ((margins+design@(w-1))[guard]>=config.retain_fraction*margins[guard]-1e-10).all()
    # A unique ACTIVE margin constraint catches an accidentally detached alpha.
    m=torch.tensor([.3,.8],dtype=torch.float64)
    aa=torch.tensor([[-1.,0.,0.],[0.,.1,0.]],dtype=torch.float64)
    vv=torch.tensor([2.,1.2,1.1],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:protected_step(x,aa,m,config)[0],(vv,),eps=1e-5,atol=1e-5)
    u=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64))[0]
    q=F.normalize(torch.randn(13,7,dtype=torch.float64),dim=1)
    s=F.normalize(torch.randn(17,7,dtype=torch.float64),dim=1)
    weights=torch.tensor([.25,1.2,4.],dtype=torch.float64)
    matrix=torch.eye(7,dtype=torch.float64)+(u*(weights-1))@u.T
    explicit=torch.einsum('nmd,dc,nmc->nm',q[:,None]-s[None],matrix,q[:,None]-s[None])
    distance=metric_distances(q,s,u,weights)
    distance_error=float((distance-explicit).abs().max());assert distance_error<1e-12
    fg=torch.arange(17)<6
    full=density_ratio(q,s,fg,u,weights,query_chunk=20,anchor_chunk=20)
    chunk=density_ratio(q,s,fg,u,weights,query_chunk=3,anchor_chunk=2)
    assert torch.allclose(full,chunk,atol=1e-12)
    assert torch.allclose(full,-density_ratio(q,s,~fg,u,weights),atol=1e-12)
    features=F.normalize(torch.randn(64,7,dtype=torch.float64),dim=1)
    coverage=(torch.arange(64).reshape(8,8)%8<4).flatten().double()
    triples,pi=local_triplets(features,coverage,(8,8),config)
    assert len(triples)>0 and torch.allclose(pi.sum(),pi.new_tensor(1.))
    model=ReferenceMetric(7,config).double()
    delta,audit=model(features,coverage,q,(8,8))
    loss=delta.square().mean()+.01*delta.mean();loss.backward()
    assert model.dictionary.grad is not None and torch.isfinite(model.dictionary.grad).all()
    assert model.gain_parameter.grad is not None and torch.isfinite(model.gain_parameter.grad)
    original=torch.randn(32,32)
    assert correct_host_field(original,torch.zeros(64),(8,8)) is original
    empty_delta,empty_audit=model(features,torch.ones(64,dtype=torch.float64),q,(8,8))
    assert not empty_delta.any() and empty_audit['state']=='FALLBACK_NO_LEGAL_TRIPLETS'
    report=dict(state='PASSED',scope='Synthetic CPU math/implementation, no DINO/data/GPU/task efficacy',
        implicit_gradcheck=True,protected_active_alpha_gradcheck=True,metric_matrix_maxerror=distance_error,
        chunked_density_parity=True,label_complement_antisymmetry=True,full_dictionary_gradient=True,
        exact_zero_correction_host_identity=True,degenerate_reference_fallback=True,
        near_stationary_roundoff_regression=near_stationary,audit=audit)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report))
    print(json.dumps(report))


if __name__=='__main__':main()
