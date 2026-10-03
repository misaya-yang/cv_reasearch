#!/usr/bin/env python3
"""Independent NumPy structural audit of the supplied metric research fixture.

This neither runs a segmentation model nor promotes the optional initializer.
The real-run cause remains unidentified without actual support-force records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def centered(c):return c-np.trace(c)/len(c)*np.eye(len(c))


def diagonal_energy(c,u):return float(np.square(np.diag(u.T@centered(c)@u)).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve previous analytic evidence')
    source=a.package/'metric_scientific_checks_followon_2049/initializer_cancellation_results.json'
    fixture=json.loads(source.read_text());cs=np.asarray(fixture['support_force_matrices'],dtype=float)
    identity=np.eye(2);rotated=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
    mean=cs.mean(0)
    stats=dict(mean_centered_energy=float(np.square(centered(mean)).sum()),
        mean_task_centered_energy=float(np.square(np.stack([centered(c) for c in cs])).sum((1,2)).mean()),
        canonical_task_diagonal_energy=float(np.mean([diagonal_energy(c,identity) for c in cs])),
        rotated_task_diagonal_energy=float(np.mean([diagonal_energy(c,rotated) for c in cs])),
        commutator_norm=float(np.linalg.norm(cs[0]@cs[1]-cs[1]@cs[0])))
    rng=np.random.default_rng(2049);dimension,rank,samples=64,8,2048
    force=np.zeros((dimension,dimension));force[0,0]=1;force[1,1]=-1
    captured=[]
    for _ in range(samples):
        u=np.linalg.qr(rng.normal(size=(dimension,rank)),mode='reduced')[0]
        # Exact rank-two contraction; avoid unnecessary dense BLAS and stale
        # FP-status warnings emitted by the author's system NumPy backend.
        if not np.isfinite(u).all():raise ValueError('Nonfinite QR result')
        captured.append(np.square(u[0]**2-u[1]**2).sum()/2)
    captured=np.asarray(captured);se=float(captured.std(ddof=1)/np.sqrt(samples))
    monte_carlo=dict(scope='Independent random-QR mathematical fixture, not trained-U evidence',
        D=dimension,rank=rank,samples=samples,seed=2049,
        theoretical_capture_ratio=2*rank/(dimension*(dimension+2)),
        measured_capture_ratio=float(captured.mean()),standard_error=se,
        predicted_D1024_rank32_ratio=64/(1024*1026))
    output=dict(state='COMPLETED_LOCAL_REFERENCE_MECHANISM_AUDIT',
        package_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fixture_structure=stats,random_basis_geometry=monte_carlo,
        package_synthetic_control_losses={k:v.get('loss') for k,v in fixture['controls'].items()},
        mechanism='Reference labels create episode force C_e; the adaptive scales see only diag(U^T C_e U), then inner curvature/regularization and the guard determine coefficient size.',
        coefficient_bound='||r_star|| <= ||g0||/lambda, g0=-diag(U^T C_e U)/T. Small r does not imply small g0 because curvature may shrink the response.',
        real_initialization='The actual repo pilot used seed2048 Gaussian-QR, not the package optional mean-force spectral initializer.',
        incorrect_inferences=['Mean centered force cancels => every episode lacks information',
            'The commuting two-task fixture refutes common-basis expressivity',
            'Toy adaptation beats a frozen-basis global ablation => beats independently learned global U and w',
            'Random-basis expectation certifies the already trained selected dictionary'],
        missing_real_diagnostics=['g0 by task','projected full force and diagonal capture',
            'mean versus taskwise centered-force energy','inner Hessian curvature',
            'preferred r before alpha','applied alpha/r','actual density increment versus host margin'],
        real_DINO_gain_measured=False,GPU_used=False,received_package_modified=False)
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(state=output['state'],fixture=stats,random=monte_carlo)))


if __name__=='__main__':main()
