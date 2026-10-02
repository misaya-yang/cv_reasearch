#!/usr/bin/env python3
"""Local counterexamples for a conditional APD rescue; NOT a DINO experiment.

No encoder, new dataset, learned head, GPU, simulator score reported as mIoU,
or candidate registration. The purpose is to reject insufficient guarantees.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def normalize(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True)


def stable_direction(contrasts):
    mean=contrasts.mean(0)
    _,singular,vh=np.linalg.svd(contrasts-mean,full_matrices=False)
    rank=int(np.count_nonzero(singular>1e-10))
    nuisance=vh[:rank].T
    residual=mean-nuisance@(nuisance.T@mean)
    return residual,nuisance


def density_margin(query,fg,bg,bandwidth=.1):
    def logmean(a):
        z=normalize(query)@normalize(a).T/bandwidth
        high=z.max(1,keepdims=True)
        return high[:,0]+np.log(np.exp(z-high).mean(1))
    return logmean(fg)-logmean(bg)


def gaussian_error(a,b,c,d,e,t):
    variance=c+2*t*d+t*t*e
    if variance<=0:raise ValueError('Positive query variance required')
    standardized=(a+t*b)/(2*math.sqrt(variance))
    return .5*math.erfc(standardized/math.sqrt(2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Keep existing analytic receipts')
    rng=np.random.default_rng(2049)
    f=normalize(rng.normal(size=(12,6)));g=rng.uniform(.1,1,size=(12,1))
    scalar_error=float(np.max(np.abs(normalize(g*f)-normalize(f))))
    pos=np.diag([1.,1.,0.]);native=np.eye(3)-pos
    # The two lawful-view worlds share class content but reverse layout.
    fg=normalize(np.array([[1.,3.,1.],[1.,-3.,1.]]))
    bg=normalize(np.array([[-1.,-3.,1.],[-1.,3.,1.]]))
    projected_contrasts=(fg-bg)@pos
    residual,nuisance=stable_direction(projected_contrasts)
    rescue=native+np.outer(residual,residual)/(residual@residual)
    restore_error=float(np.linalg.norm(rescue@residual-residual))
    nuisance_error=float(np.linalg.norm(rescue@nuisance))
    query=normalize(np.array([[1.,-3.,1.],[-1.,3.,1.]]));truth=np.array([True,False])
    scores={
        'native_APD':density_margin(query@native,fg[:1]@native,bg[:1]@native),
        'APD_off_single_view':density_margin(query,fg[:1],bg[:1]),
        'conditional_rescue':density_margin(query@rescue,fg[:1]@rescue,bg[:1]@rescue),
        'naive_same_information_view_bank':density_margin(query,fg,bg)}
    toy={k:dict(margin=v.tolist(),correct=int(((v>0)==truth).sum()),units=2) for k,v in scores.items()}
    # Identical observed contrasts have two admissible latent explanations.
    observed=projected_contrasts
    semantic=np.tile(residual,(2,1));noise_semantic=observed-semantic
    noise_only=observed.copy()
    world_identity=bool(np.array_equal(semantic+noise_semantic,noise_only))
    gaussian=[]
    for name,a0,b0,c0,d0,e0 in [
        ('useful_removed_signal',0.,1.,1.,0.,.05),
        ('opposite_query_layout',1.,-1.,1.,0.,.01),
        ('positive_signal_but_high_variance',1.,.2,1.,0.,100.),
        ('aligned_low_variance_signal',1.,.5,1.,0.,.1)]:
        gaussian.append(dict(name=name,A=a0,B=b0,C=c0,D=d0,E=e0,
            risk_native=gaussian_error(a0,b0,c0,d0,e0,0),
            risk_full_rescue=gaussian_error(a0,b0,c0,d0,e0,1),
            initial_SNR_derivative_numerator=b0*c0-a0*d0))
    output=dict(state='COMPLETED_LOCAL_CONDITIONAL_MECHANISM_ANALYSIS',
        source=dict(path=str(a.source),sha256=hashlib.sha256(a.source.read_bytes()).hexdigest()),
        scalar_normalization_max_error_F64=scalar_error,
        conditional_rescue=dict(restore_error=restore_error,nuisance_deletion_error=nuisance_error,
            observed_contrasts=observed.tolist(),stable_residual=residual.tolist(),rank=int(nuisance.shape[1])),
        same_observations_semantic_vs_fixed_nuisance=world_identity,
        artificial_three_coordinate_matching=toy,gaussian_risk_counterexamples=gaussian,
        decision='Do not register or queue this proposal on the strength of preserved support contrast. Its positive toy is matched by the naive same-information bank and fixed layout bias remains unidentifiable.',
        actual_DINO_features_or_masks_used=False,GPU_used=False,query_GT_in_real_model=False,
        real_task_gain_measured=False,full_FoRIS_operator_executed=False,
        derivative='For t=eta^2, R(t)=(A+tB)/sqrt(C+2tD+t^2E); sign R-prime = sign[BC-AD+t(BD-AE)]. Positive contrast B alone is insufficient.')
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(state=output['state'],scalar_error=scalar_error,
        same_information_toy_correct={k:v['correct'] for k,v in toy.items()},
        indistinguishable_latent_worlds=world_identity,
        real_task_gain_measured=False,new_candidate_registered=False)))


if __name__=='__main__':main()
