"""Explicit-feature verification and complete synthetic kernel classifier checks."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.reference_quadratic import predict, solve_kernel


def explicit_features(x):
    return .5*np.column_stack((np.ones(len(x)), np.sqrt(2)*x,
                               (x[:, :, None]*x[:, None, :]).reshape(len(x), -1)))


def fixture():
    q = np.zeros((4096, 1024), dtype=np.float32)
    r = np.zeros_like(q)
    q[:, 2] = 1
    r[:, 2] = 1
    cov = np.zeros((64, 64))
    cov[2:10, 2:10] = 1
    for row in range(2, 10):
        for col in range(2, 10):
            i = row*64+col
            r[i] = 0
            r[i, 0 if col < 6 else 1] = 1 if row < 6 else -1
    background = np.flatnonzero(cov.ravel() == 0)
    r[background[::2], 2] = -1
    target = np.zeros((64, 64), dtype=bool)
    target[20:28, 4:12] = True
    q[target.ravel()] = 0
    q[target.ravel(), :2] = .55
    q[target.ravel(), 2] = np.sqrt(1 - 2*.55**2)
    base = np.full((64, 64), .1)
    base[target] = .8
    return q, r, cov, base, target


def main():
    rng = np.random.RandomState(20261013)
    maximum_difference = 0.
    for _ in range(24):
        anchors = rng.normal(size=(10, 5))
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        labels = np.r_[np.ones(5), -np.ones(5)]
        weights = rng.uniform(.5, 1.5, 10)
        weights[:5] *= .5/weights[:5].sum()
        weights[5:] *= .5/weights[5:].sum()
        gram_cos = np.einsum('id,jd->ij', anchors, anchors, optimize=False)
        coefficient, info = solve_kernel(((1 + gram_cos)*.5)**2, labels, weights, .01)
        phi = explicit_features(anchors)
        gram_explicit = np.einsum('id,jd->ij', phi, phi, optimize=False)
        np.testing.assert_allclose(gram_explicit, ((1+gram_cos)*.5)**2, atol=1e-14)
        normal = np.einsum('ni,n,nj->ij', phi, weights, phi, optimize=False) + .01*np.eye(phi.shape[1])
        target_vector = np.einsum('ni,n,n->i', phi, weights, labels, optimize=False)
        primal = np.linalg.solve(normal, target_vector)
        query = rng.normal(size=(17, 5))
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        query_cos = np.einsum('nd,kd->nk', query, anchors, optimize=False)
        dual_score = np.einsum('nk,k->n', ((1+query_cos)*.5)**2, coefficient, optimize=False)
        primal_score = np.einsum('nd,d->n', explicit_features(query), primal, optimize=False)
        difference = float(np.max(np.abs(dual_score-primal_score)))
        maximum_difference = max(maximum_difference, difference)
        assert difference < 1e-10 and info['relative_linear_residual'] < 1e-12
        assert info['minimum_regularized_eigenvalue'] > 0
    q, r, cov, base, target = fixture()
    result = predict(q, r, cov, base)
    np.testing.assert_array_equal(result['field'] > .5, target)
    for key in ('linear_control', 'kernel_mean_control', 'nearest_control'):
        assert not (result[key] > .5).any()
    np.testing.assert_array_equal(result['subspace_control'] > .5, target)
    # Final predictor preserves direction. The initially considered homogeneous
    # square kernel cannot distinguish antipodal foreground/background features.
    anti_r, anti_q = np.zeros_like(r), np.zeros_like(q)
    anti_r[:, 0] = -1
    anti_r[cov.ravel() == 1, 0] = 1
    anti_q[:, 0] = -1
    anti_q[target.ravel(), 0] = 1
    anti = predict(anti_q, anti_r, cov, base)
    np.testing.assert_array_equal(anti['field'] > .5, target)
    assert np.max(anti['homogeneous_control']) - np.min(anti['homogeneous_control']) < 1e-12
    assert not np.array_equal(anti['homogeneous_control'] > .5, target)
    np.testing.assert_array_equal(anti['nearest_control'] > .5, target)
    # Final-method failure: changed true query appearance lies much closer to
    # reference background. A correct strong base is discarded, not merely left.
    shifted_q = q.copy()
    shifted_q[target.ravel(), :2] = .3
    shifted_q[target.ravel(), 2] = np.sqrt(1 - 2*.3**2)
    shifted = predict(shifted_q, r, cov, base)
    assert not (shifted['field'] > .5).any()
    assert (base[target] > .5).all()
    absent = predict(q, r, np.ones_like(cov), base)
    assert absent['info']['abstention']
    np.testing.assert_array_equal(absent['field'], base)
    reversed_result = predict(-q, -r, cov, base)
    np.testing.assert_allclose(reversed_result['field'], result['field'], atol=1e-10)
    # Exercise the nominal32-anchor path with dense, full-dimensional inputs.
    # No masks from this timing fixture are judged as semantic predictions.
    dense_q = rng.normal(size=(4096, 1024)).astype(np.float32)
    dense_r = rng.normal(size=(4096, 1024)).astype(np.float32)
    dense_cov = np.zeros((64, 64))
    dense_cov[:, :32] = 1
    dense = predict(dense_q, dense_r, dense_cov, base)
    assert dense['info']['foreground_modes'] == dense['info']['background_modes'] == 16
    assert dense['info']['quadratic_solver']['relative_linear_residual'] < 1e-10
    assert np.isfinite(dense['field']).all()
    report = dict(
        kind='analytical_and_synthetic_complete_algorithm_checks', real_episodes=0,
        independent_explicit_feature_primal_cases=24, maximum_primal_dual_score_difference=maximum_difference,
        complete_output=dict(target_tokens=64, false_tokens=0),
        pointwise_maximum_query_similarity=dict(foreground=.55, background=float(np.sqrt(1-2*.55**2))),
        first_contrast_also_solved_by_subspace_control=True,
        nonhomogeneous_kernel_preserves_antipodal_polarity=True,
        discarded_homogeneous_construction=dict(antipodal_class_scores_identical=True, fails_separation=True),
        antipodal_contrast_also_solved_by_nearest_control=True,
        query_appearance_shift_counterexample=dict(strong_base_target_tokens=64, candidate_target_tokens=0),
        missing_background_abstains=True, common_feature_sign_invariance=True,
        dimensions=[4096, 1024], candidate_info=result['info'],
        dense_32_anchor_timing=dense['info'],
        timing_scope='Synthetic inputs already loaded; includes five controls, excludes shared MEAN, I/O,1024 readout and cohort scoring',
        real_gain='unmeasured', real_runtime='unmeasured', no_gpu=True, no_server=True,
    )
    Path(__file__).with_name('quadratic_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(primal_dual_difference=maximum_difference, complete_output=report['complete_output'],
                          subspace_also_solves_first=True, appearance_shift_counterexample=True, real_episodes=0)))


if __name__ == '__main__':
    main()
