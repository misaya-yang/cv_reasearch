"""Concave mixture fitting and complete synthetic segmentation, including failure."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy.optimize import minimize_scalar
from ics.methods.reference_prior_shift import Config, fit_prior, predict


def fixture(small=False):
    q = np.zeros((4096, 1024), dtype=np.float32)
    r = np.zeros_like(q)
    q[:, 2] = 1
    r[:, 2] = 1
    cov = np.zeros((64, 64))
    cov[2:10, 2:10] = 1
    for row in range(2, 10):
        for col in range(2, 10):
            i = row * 64 + col
            r[i] = 0
            r[i, 0 if col < 6 else 1] = 1
    background = np.flatnonzero(cov.ravel() == 0)
    r[background[:len(background)//5]] = 0
    r[background[:len(background)//5], 1] = 1
    target = np.zeros((64, 64), dtype=bool)
    distractor = np.zeros_like(target)
    if small:
        target[20:28, 4:12] = True
    else:
        target[20:36, :16] = True
    distractor[40:60, 16:56] = True
    q[target.ravel()] = 0
    q[target.ravel(), 0] = 1
    q[distractor.ravel()] = 0
    q[distractor.ravel(), 1] = 1
    base = np.full((64, 64), .1)
    return q, r, cov, base, target, distractor


def main():
    rng = np.random.RandomState(20261012)
    maximum_difference = 0.
    for _ in range(32):
        f, b = np.exp(rng.normal(size=(2, 80)))
        fitted, info = fit_prior(f, b, .01, .99)
        independent = minimize_scalar(lambda p: -np.mean(np.log(p*f + (1-p)*b)),
                                      bounds=(.01, .99), method='bounded', options=dict(xatol=1e-12))
        alternatives = [.01, .99, float(independent.x)]
        best = min(alternatives, key=lambda p: -np.mean(np.log(p*f + (1-p)*b)))
        maximum_difference = max(maximum_difference, abs(fitted-best))
        assert abs(fitted-best) < 1e-6
        assert info['negative_average_second_derivative'] >= 0
    # Exact expected category counts from an identifiable two-density mixture.
    f = np.array([.7, .2, .1])
    b = np.array([.1, .3, .6])
    count = np.array([2800, 2700, 4500])  # .3*f + .7*b, no sampled count noise.
    estimated, _ = fit_prior(np.repeat(f, count), np.repeat(b, count), .0001, .9999)
    assert abs(estimated-.3) < 1e-8
    lower, _ = fit_prior(np.ones(8), np.full(8, 2.), .01, .99)
    upper, _ = fit_prior(np.full(8, 2.), np.ones(8), .01, .99)
    assert lower == .01 and upper == .99
    q, r, cov, base, target, distractor = fixture()
    result = predict(q, r, cov, base)
    np.testing.assert_array_equal(result['field'] > .5, target)
    np.testing.assert_array_equal(result['balanced_control'] > .5, target | distractor)
    np.testing.assert_array_equal(result['margin_control'] > .5, target | distractor)
    assert not (result['reference_prior_control'] > .5).any()
    # Exact decimal purity endpoints belong to their foreground/background
    # sample sets. Equal weights within each set preserve fitted densities.
    endpoints = predict(q, r, .1 + .8 * cov, base)
    assert endpoints['info']['foreground_samples'] == 64
    assert endpoints['info']['background_samples'] == 4032
    np.testing.assert_allclose(endpoints['field'], result['field'], atol=1e-10)
    # Density transfer/overlap failure remains visible; this does not change
    # bandwidth, prior bounds or any method setting after observing the failure.
    small_q, small_r, small_cov, small_base, small_target, _ = fixture(small=True)
    small = predict(small_q, small_r, small_cov, small_base)
    assert not (small['field'] > .5).any()
    assert small_target.sum() == 64
    # Identical class-conditional score densities give a flat prior likelihood.
    same = np.zeros_like(r)
    same[:, 0] = 1
    flat = predict(q, same, cov, base)
    assert flat['info']['abstention']
    np.testing.assert_array_equal(flat['field'], base)
    absent = predict(q, r, np.ones_like(cov), base)
    assert absent['info']['abstention']
    np.testing.assert_array_equal(absent['field'], base)
    # Orthogonal common sign reversal preserves reference/query score densities.
    reversed_result = predict(-q, -r, cov, base)
    np.testing.assert_allclose(reversed_result['field'], result['field'], atol=1e-10)
    report = dict(
        kind='analytical_and_synthetic_complete_algorithm_checks', real_episodes=0,
        independent_scalar_optimizer_cases=32, maximum_prior_difference=maximum_difference,
        exact_expected_mixture_prior=dict(expected=.3, fitted=float(estimated)),
        boundary_optima_checked=True, flat_likelihood_abstains=True,
        missing_background_abstains=True, common_feature_sign_invariance=True,
        purity_endpoints_and_uniform_within_class_weight_invariance=True,
        complete_output=dict(target_tokens=256, false_tokens=0,
                             balanced_false_tokens=800, margin_false_tokens=800,
                             reference_prior_target_tokens=0),
        estimated_query_prior=float(result['info']['query_prior']), actual_synthetic_target_fraction=float(target.mean()),
        reference_prior=float(result['info']['reference_prior']),
        small_target_counterexample=dict(target_tokens=64, recovered_target_tokens=0,
                                         output_tokens=0, fitted_prior=float(small['info']['query_prior'])),
        candidate_info=result['info'], dimensions=[4096, 1024],
        timing_scope='Synthetic arrays already loaded; excludes shared baseline construction, I/O,1024 readout and cohort scoring',
        real_gain='unmeasured', real_runtime='unmeasured', no_gpu=True, no_server=True,
    )
    Path(__file__).with_name('prior_shift_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(complete_output=report['complete_output'],
                          small_target_counterexample=report['small_target_counterexample'],
                          prior_difference=maximum_difference, real_episodes=0)))


if __name__ == '__main__':
    main()
