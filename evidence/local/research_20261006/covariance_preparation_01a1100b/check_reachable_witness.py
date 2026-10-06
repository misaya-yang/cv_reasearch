"""Reachable normalized-feature witness for the unchanged covariance predictor.

Synthetic construction only: no real image, DINO output, label or remote resource.
Unlike a freely specified affinity table, every affinity here is a cosine between
unit feature vectors and the modes learned by the actual default implementation.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.reference_covariance import Config, covariances, covariance_debts, predict, response_features
from ics.methods.reference_occupancy import Config as TransportConfig, cluster, occupancy_debt, unit


def matrix(rows):
    return covariances(response_features(rows).sum(0, keepdims=True), rows.shape[1])[0]


def main():
    # Eight modes with four correlated pairs. Positive definite eigenvalues
    # are .5 and 1.5, so Cholesky realizes the Gram matrix with unit vectors.
    gram = np.eye(8)
    for j in range(0, 8, 2):
        gram[j, j + 1] = gram[j + 1, j] = .5
    reference = np.linalg.cholesky(gram)
    permutation = np.array([0, 2, 1, 3, 4, 6, 5, 7])
    desired_wrong = gram[:, permutation]
    wrong = np.linalg.solve(reference, desired_wrong.T).T
    # Exact common scale from the unit-ball feasibility bound, not a fitted
    # method parameter: the largest squared norm is 5/3, scale=sqrt(3/5).
    maximum_norm2 = float(np.max(np.sum(wrong * wrong, axis=1)))
    scale = 1 / np.sqrt(maximum_norm2)
    np.testing.assert_allclose(maximum_norm2, 5 / 3, atol=1e-14)

    def lift(rows):
        projected = scale * rows
        return np.column_stack((projected, np.sqrt(np.maximum(0, 1 - np.sum(projected**2, axis=1)))))

    correct, distractor = lift(reference), lift(wrong)
    shape = (64, 64)
    q = np.zeros((4096, 1024), dtype=np.float32)
    r = np.zeros_like(q)
    q[:, 9] = 1
    r[:, 9] = 1
    coverage = np.zeros(shape)
    coverage[2:10, 2:10] = 1
    base = np.full(shape, .1)
    target = np.zeros(shape, dtype=bool)
    false = np.zeros(shape, dtype=bool)
    target[24:32, 2:10] = True
    false[24:32, 40:48] = True
    base[target | false] = .8
    for row in range(8):
        for col in range(8):
            ri = (row + 2) * 64 + col + 2
            ti = (row + 24) * 64 + col + 2
            fi = (row + 24) * 64 + col + 40
            r[ri] = 0
            r[ri, :8] = reference[col]
            q[ti] = 0
            q[ti, :9] = correct[col]
            q[fi] = 0
            q[fi, :9] = distractor[col]
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1, atol=1e-7)
    cfg = Config()
    pure = coverage.ravel() >= cfg.purity
    centers, _, weights = cluster(r[pure], cfg.reference_modes, cfg.lloyd_steps, coverage.ravel()[pure])
    assert len(centers) == 8
    ref_response = np.einsum('nd,kd->nk', unit(r[pure]), centers, optimize=False)
    true_response = np.einsum('nd,kd->nk', unit(q[target.ravel()]), centers, optimize=False)
    false_response = np.einsum('nd,kd->nk', unit(q[false.ravel()]), centers, optimize=False)
    # These quantities agree for actual learned modes, including pointwise
    # sorted rows at corresponding positions, not just scalar aggregate means.
    tolerance = 1e-7
    np.testing.assert_allclose(true_response.mean(0), false_response.mean(0), atol=tolerance)
    np.testing.assert_allclose(np.sort(true_response, axis=1), np.sort(false_response, axis=1), atol=tolerance)
    np.testing.assert_array_equal(np.bincount(true_response.argmax(1), minlength=8),
                                  np.bincount(false_response.argmax(1), minlength=8))
    ref_cov = matrix(ref_response)
    true_cov, false_cov = matrix(true_response), matrix(false_response)
    np.testing.assert_allclose(np.trace(true_cov), np.trace(false_cov), atol=tolerance)
    bank = dict(k=8, matrices=ref_cov[None], norm2=np.array([np.sum(ref_cov**2)]),
                trace=np.array([np.trace(ref_cov)]), mode='full')
    full = covariance_debts(np.stack((true_cov, false_cov)), np.ones(2, dtype=bool), bank)
    trace = covariance_debts(np.stack((true_cov, false_cov)), np.ones(2, dtype=bool), dict(bank, mode='trace'))
    assert full[0] < 1 / 3 < full[1]
    np.testing.assert_allclose(trace[0], trace[1], atol=tolerance)
    # Uniform reference/query mode masses make the transport problem invariant
    # to this column permutation too. This is the occupancy cost on this
    # fixture, not a claim about every regional occupancy decoder.
    transport = []
    for values in (true_response[:8], false_response[:8]):
        cost, _ = occupancy_debt(np.full((1, 8), 1 / 8), weights, 1 - values, TransportConfig())
        transport.append(float(cost[0]))
    np.testing.assert_allclose(transport[0], transport[1], atol=tolerance)
    result = predict(q, r, coverage, base)  # Default config; no supplied bank.
    assert not result['info']['abstention']
    np.testing.assert_array_equal(result['token_mask'], target)
    np.testing.assert_array_equal(result['trace_control'], target | false)
    np.testing.assert_array_equal(result['mode_margin_field'] > .5, target | false)
    for move in result['info']['search']['moves']:
        assert move['after'] > move['before']
    source = ROOT / 'src/ics/methods/reference_covariance.py'
    report = dict(
        kind='reachable_unit_feature_synthetic_witness', real_episodes=0,
        dimensions=dict(tokens=4096, feature_dimensions=1024, reference_modes=8),
        configuration='unchanged default Config()',
        feasibility=dict(maximum_unscaled_wrong_norm2=maximum_norm2, common_scale=float(scale),
                         unit_feature_norms_checked=True, modes_learned_by_actual_cluster=True),
        matched_controls=dict(column_means=True, mode_occupancy=True, covariance_trace=True,
                              pointwise_sorted_cosines=True, pointwise_best_and_runner_margin=True,
                              uniform_mass_transport_extra_cost=transport, tolerance=tolerance),
        covariance_debts=dict(full_true=float(full[0]), full_false=float(full[1]),
                              trace_true=float(trace[0]), trace_false=float(trace[1])),
        complete_token_outputs=dict(covariance_true=64, covariance_false=0,
                                    trace_true=64, trace_false=64, margin_true=64, margin_false=64),
        search=result['info']['search'],
        scope='Existence of an extra usable covariance cue through the actual predictor. Not real DINO reachability, frequency, semantic transfer, mIoU, novelty or cohort runtime evidence.',
        real_gain='unmeasured', new_independent_methods=0, no_gpu=True, no_server=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    Path(__file__).with_name('reachable_witness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(controls_matched=True, complete_outputs=report['complete_token_outputs'],
                          covariance_debts=report['covariance_debts'], real_episodes=0)))


if __name__ == '__main__':
    main()
