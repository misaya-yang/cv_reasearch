#!/usr/bin/env python3
"""Independent CPU checks of RICE proposal equations, not segmentation results.

No torch, model, data, labels, server or GPU. Counterexamples constrain what
the proposal can claim; they neither accept nor reject its task-level premise.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def logsumexp(x):
    peak = np.max(x)
    return float(peak + np.log(np.exp(x - peak).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kernel-checks', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing analysis')
    rng = np.random.default_rng(2051)
    d, witnesses, nuisance_count = 12, 4, 7
    nuisance = rng.normal(size=(d, nuisance_count)) / np.sqrt(nuisance_count)
    differences = rng.normal(size=(d, witnesses))
    epsilon = .2
    sn = epsilon * np.eye(d) + nuisance @ nuisance.T
    sd = differences @ differences.T
    invd = np.linalg.solve(sn, differences)
    small = differences.T @ invd
    eigenvalues, eigenvectors = np.linalg.eigh(small)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    w = invd @ (eigenvectors / np.sqrt(eigenvalues)[None, :])
    whiten_error = float(np.max(np.abs(w.T @ sn @ w - np.eye(witnesses))))
    eigen_error = float(np.max(np.abs(sd @ w - (sn @ w) * eigenvalues[None, :])))
    woodbury = (differences - nuisance @ np.linalg.solve(
        epsilon * np.eye(nuisance_count) + nuisance.T @ nuisance, nuisance.T @ differences)) / epsilon
    woodbury_error = float(np.max(np.abs(woodbury - invd)))
    assert max(whiten_error, eigen_error, woodbury_error) < 1e-10

    # Sn is SPD, hence full-span even when the discriminant ratio is large.
    u = np.array([1., 0.])
    counter_sn = np.eye(2)
    counter_sd = 100 * np.outer(u, u)
    rayleigh = float((u @ counter_sd @ u) / (u @ counter_sn @ u))
    assert rayleigh == 100

    ring = []
    for size in (1, 4, 16, 64):
        values = np.full(size, 2.)
        naive = 2. - logsumexp(values)
        temperature = .3
        normalized = 2. - temperature * (logsumexp(values / temperature) - np.log(size))
        assert abs(normalized) < 1e-12
        ring.append(dict(pixels=size, original_ring_contrast=naive,
                         temperature_scaled_logmeanexp_contrast=normalized))

    # Two parts legally share one token under the stated fractional constraints.
    a = np.array([.5, .5]); b = np.array([1.]); transport = np.array([[.5], [.5]])
    assert np.all(transport.sum(1) <= a) and np.all(transport.sum(0) <= b)
    missing_penalty = .5; negative_evidence = -.2
    unmatched = -missing_penalty * a.sum()
    matched = float(transport.sum() * negative_evidence)
    assert matched > unmatched

    # Total covariance: pairing carries no extra effect once both terms are
    # summed with gamma=1 and within uses the cross-view per-position mean.
    features = rng.normal(size=(4, 7, 6))
    def covariances(x):
        pixel_mean = x.mean(0)
        group_mean = pixel_mean.mean(0)
        residual = (x - pixel_mean[None]).reshape(-1, x.shape[-1])
        context = residual.T @ residual / len(residual)
        within = (pixel_mean - group_mean).T @ (pixel_mean - group_mean) / len(pixel_mean)
        augmented = (x - group_mean).reshape(-1, x.shape[-1])
        augmented = augmented.T @ augmented / len(augmented)
        return context, within, augmented
    context, within, augmented = covariances(features)
    shuffled = features.copy()
    for view in range(1, len(shuffled)):
        shuffled[view] = shuffled[view, rng.permutation(shuffled.shape[1])]
    sc, sw, sa = covariances(shuffled)
    covariance_error = float(np.abs(context + within - augmented).max())
    assert covariance_error < 1e-12
    assert np.allclose(augmented, sa, atol=1e-12)
    assert np.allclose(context + within, sc + sw, atol=1e-12)

    # Merely sharing the nuisance span does not erase contrast. Scaling Sn up
    # reduces the ratio; it is relative amplitude, not span inclusion, that matters.
    report = dict(state='COMPLETED_CPU_RICE_EQUATION_AUDIT',
        proposal_sha256=hashlib.sha256(args.proposal.read_bytes()).hexdigest(),
        generalized_eigenconstruction=dict(d=d, discriminant_columns=witnesses,
            nuisance_columns=nuisance_count, normalization_max_error=whiten_error,
            eigen_equation_max_error=eigen_error, woodbury_max_error=woodbury_error,
            validated=True, cost_caveat='Dual dimension equals number of nuisance columns, not discriminant rank; choose primal/dual or declared exact rank compression.'),
        span_counterexample=dict(nuisance_full_rank=2, discriminant_rank=1,
            discriminant_span_contained=True, generalized_top_eigenvalue=rayleigh,
            conclusion='Span inclusion does not imply small eigenvalues; epsilonI makes nuisance span full regardless.'),
        ring_constant_evidence=ring,
        fractional_assignment_counterexample=dict(parts=2, tokens=1, a=a.tolist(), b=b.tolist(),
            feasible_assignment=transport.tolist(), all_mass_matched=True,
            conclusion='Column capacity limits total mass but allows one token to represent several parts fractionally.'),
        missing_penalty_counterexample=dict(evidence=negative_evidence, kappa=missing_penalty,
            score_if_unmatched=float(unmatched), score_if_matched=matched,
            conclusion='The penalty rewards matching whenever evidence exceeds -kappa; it is not a positive-evidence constraint.'),
        total_covariance_identity=dict(views=4, aligned_positions=7, dimensions=6,
            context_plus_within_equals_augmented_max_error=covariance_error,
            gamma1_pair_shuffle_sum_max_change=float(np.abs(context + within - sc - sw).max()),
            pair_shuffle_context_frobenius_change=float(np.linalg.norm(context - sc)),
            scope='Within is computed from cross-view per-position means within one foreground part; equal population weighting. If gamma=1, paired covariance plus within equals ordinary augmented part covariance. Original-view within or gamma!=1 differs.'),
        calibration_scope='Held reference views share identity, pose and pixels; AUC cannot certify cross-image edit precision. Fit W/parts/negatives without the held view.',
        prior_reachability_scope='The existing .0231216 and +.1841pp limits are conditional ideal-arithmetic bounds for a frozen E3 intervention, not a FP32 certificate or a bound on all metrics.',
        next_mechanism='Compare matched raw geometry, same-view simple covariance/ordinary discriminant, and counterfactual-nuisance generalized discriminant, using the same full FG/BG evidence and continuous query decision. Defer ring, OT and override gate.',
        segmentation_gain_measured=False, GPU_used=False, downloaded_assets=False,
        proposal_modified=False)
    if args.kernel_checks:
        source = Path(__file__).resolve().parents[1] / 'tics/rice_subspace.py'
        spec = importlib.util.spec_from_file_location('rice_numpy_kernel', source)
        kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
        checks = []
        for nuisance_columns in (0, 4, 20):
            v = rng.normal(size=(d, nuisance_columns)) * .1
            direct_inverse = np.linalg.solve(.2 * np.eye(d) + v @ v.T, differences)
            actual_inverse = kernel.inverse_nuisance_action(v, differences, .2)
            assert np.allclose(actual_inverse, direct_inverse, atol=1e-11, rtol=1e-11)
            ww, info = kernel.reference_subspace(differences, v, .2, rank_cap=4, retained_energy=1.)
            ss = .2 * np.eye(d) + v @ v.T
            assert np.allclose(ww.T @ ss @ ww, np.eye(4), atol=1e-10, rtol=1e-10)
            fg, bg = differences.T[:2], -differences.T[:2]
            q = np.vstack((fg, bg))
            dense = kernel.signed_kernel_evidence(q, fg, bg, ww, .3, query_chunk=100, anchor_chunk=100)
            streamed = kernel.signed_kernel_evidence(q, fg, bg, ww, .3, query_chunk=1, anchor_chunk=1)
            duplicated = kernel.signed_kernel_evidence(q, fg, np.repeat(bg, 3, axis=0), ww, .3)
            assert np.allclose(dense, streamed, atol=1e-10, rtol=1e-10)
            assert np.allclose(dense, duplicated, atol=1e-10, rtol=1e-10)
            assert (dense[:2] > 0).all() and (dense[2:] < 0).all()
            checks.append(dict(nuisance_columns=nuisance_columns, selected_rank=info['rank'],
                inverse_max_error=float(np.abs(actual_inverse - direct_inverse).max()),
                chunk_max_error=float(np.abs(dense - streamed).max()),
                repeated_BG_max_error=float(np.abs(dense - duplicated).max())))
        empty, info = kernel.reference_subspace(np.zeros((d, 4)), np.zeros((d, 0)), .2)
        assert empty.shape == (d, 0) and info['state'] == 'NO_REFERENCE_DISCRIMINANT'
        report['implemented_kernel_checks'] = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            cases=checks, zero_contrast_fallback=True, no_postprojection_normalization=True,
            real_images=False, segmentation_score=False)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(state=report['state'], generalized_eigen_error=eigen_error,
        woodbury_error=woodbury_error, counterexample_eigenvalue=rayleigh,
        original_constant_ring_range=[ring[0]['original_ring_contrast'], ring[-1]['original_ring_contrast']],
        fractional_multi_part_feasible=True, total_covariance_identity_error=covariance_error,
        no_real_task_score=True)))


if __name__ == '__main__':
    main()
