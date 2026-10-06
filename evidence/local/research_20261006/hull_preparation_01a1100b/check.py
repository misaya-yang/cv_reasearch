"""Bounded CPU checks; synthetic vectors, not DINO measurements. No parameter sweep."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from ics.experiment import render, sha
from ics.methods.reference_hull import Config, hull_distance, predict
from ics.methods.reference_quadratic import predict as quadratic_predict

OUT = Path(__file__).resolve().parent


def directions(angles, dimension=1024):
    vectors = np.zeros((len(angles), dimension), dtype=np.float32)
    vectors[:, :2] = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return vectors


def fixture():
    theta = np.arange(3) * 2*np.pi/3
    f = directions(np.ravel(np.stack((theta-.4, theta+.4), axis=1)))
    b = directions(theta+.2)
    r = np.concatenate((np.tile(f, (16, 1)), np.tile(b, (1332, 1)), b[[0, 1, 2, 0]]))
    cov = np.r_[np.ones(96), np.zeros(3996), np.full(4, .5)].reshape(64, 64)
    q = b[np.arange(4096) % 3].copy()
    base = np.full((64, 64), .1, dtype=np.float32)
    truth = np.zeros((64, 64), dtype=bool)
    for index, (row, col) in enumerate(((8, 8), (26, 26), (44, 44))):
        block = np.arange(4096).reshape(64, 64)[row:row+8, col:col+8].ravel()
        q[block] = directions(theta[[index]])[0]
        base.ravel()[block] = .45
        truth.ravel()[block] = True
    return dict(q=q, r=r, cov=cov, base=base), truth, theta, f, b


def counts(field, truth):
    mask = field > .5
    return dict(TP=int((mask & truth).sum()), FP=int((mask & ~truth).sum()),
                FN=int((~mask & truth).sum()), IoU=float((mask & truth).sum()/max(1, (mask | truth).sum())))


def exact_cost(query, anchors):
    start = np.eye(len(anchors))[np.argmin(np.sum((anchors-query)**2, axis=1))]
    fit = minimize(lambda weight: np.sum((weight@anchors-query)**2), start,
                   jac=lambda weight: 2*anchors@(weight@anchors-query), method='SLSQP',
                   bounds=[(0, 1)]*len(anchors),
                   constraints=[dict(type='eq', fun=lambda weight: weight.sum()-1,
                                     jac=lambda weight: np.ones(len(weight)))],
                   options=dict(ftol=1e-13, maxiter=1000))
    assert fit.success, fit.message
    assert np.min(fit.x) >= -1e-10 and abs(fit.x.sum()-1) < 1e-10
    return float(fit.fun)


def main():
    started = time.perf_counter()
    inputs, truth, theta, foreground, background = fixture()
    result = predict(**dict(q=inputs['q'], r=inputs['r'], coverage=inputs['cov'], base=inputs['base']))
    quadratic = quadratic_predict(inputs['q'], inputs['r'], inputs['cov'], inputs['base'])
    arms = dict(base=inputs['base'], hull=result['field'], nearest=result['nearest_control'],
                centroid=result['centroid_control'], linear_span=result['subspace_control'],
                affine_span=result['affine_control'], quadratic=quadratic['field'])
    token = {key: counts(value, truth) for key, value in arms.items()}
    assert token['hull']['TP'] == 192 and token['hull']['FP'] == 0
    assert all(token[key]['TP'] == 0 for key in ('base', 'nearest', 'centroid', 'linear_span', 'affine_span'))
    truth1024 = np.repeat(np.repeat(truth, 16, axis=0), 16, axis=1)
    full = {key: counts(render(value).astype(float), truth1024) for key, value in arms.items()}
    np.savez_compressed(OUT / 'witness_inputs.npz', **inputs,
                        unread_query_truth=np.array(['never read by inference'], dtype=object))
    np.savez_compressed(OUT / 'witness_outputs.npz', **arms,
                        **result['bounds'], truth_token=truth, truth1024_packed=np.packbits(truth1024),
                        **{key+'_1024_packed': np.packbits(render(value)) for key, value in arms.items()})
    # The same geometry admits a genuine semantic failure. A target can have a
    # background appearance; optimization certainty cannot protect its label.
    failure_q = background[np.arange(4096) % 3].copy()
    failure_base = inputs['base'].copy()
    failure_base[truth] = .8
    failure = predict(failure_q, inputs['r'], inputs['cov'], failure_base)
    failure_token = dict(base=counts(failure_base, truth), hull=counts(failure['field'], truth))
    np.savez_compressed(OUT / 'semantic_failure_outputs.npz', base=failure_base, hull=failure['field'],
                        truth_token=truth, **failure['bounds'],
                        base1024_packed=np.packbits(render(failure_base)),
                        hull1024_packed=np.packbits(render(failure['field'])))
    assert failure_token['base']['TP'] == 192 and failure_token['hull']['TP'] == 0
    # Both hulls contain the same query point: every unresolved value is unchanged,
    # including arbitrary continuous baseline values, rather than forcing a tie.
    shared = np.zeros_like(inputs['q']); shared[:, 0] = 1
    tie_base = np.linspace(-.2, 1.2, 4096).reshape(64, 64)
    tied = predict(shared, shared, inputs['cov'], tie_base)
    assert np.array_equal(tied['field'], tie_base) and tied['info']['fallback_rows'] == 4096
    missing = predict(inputs['q'], inputs['r'], np.ones((64, 64)), inputs['base'])
    assert missing['info']['abstention'] and np.array_equal(missing['field'], inputs['base'])
    # Independent small convex solver check of objective intervals, including the
    # witness and a deterministic generic full-rank case. No hyperparameter search.
    rng = np.random.RandomState(71)
    generic_a = rng.randn(5, 7); generic_a /= np.linalg.norm(generic_a, axis=1)[:, None]
    generic_q = rng.randn(9, 7); generic_q /= np.linalg.norm(generic_q, axis=1)[:, None]
    checks = []
    for name, q, a in [('generic', generic_q, generic_a),
                        ('witness_FG', directions(np.r_[theta, theta+.2], 2).astype(float), foreground[:, :2].astype(float)),
                        ('witness_BG', directions(np.r_[theta, theta+.2], 2).astype(float), background[:, :2].astype(float))]:
        distances = hull_distance(q, a)
        exact = np.array([exact_cost(query, a) for query in q])
        lower_error = float(np.max(distances['lower']-exact))
        upper_error = float(np.max(exact-distances['cost']))
        assert lower_error < 1e-9 and upper_error < 1e-9
        checks.append(dict(case=name, rows=len(q), lower=distances['lower'].tolist(),
                           upper=distances['cost'].tolist(), SLSQP=exact.tolist(),
                           maximum_lower_above_SLSQP=lower_error,
                           maximum_SLSQP_above_upper=upper_error, solver=distances['info']))
    bad = 0
    for cfg in (replace(Config(), maximum_iterations=0), replace(Config(), foreground_modes=17),
                replace(Config(), gap_tolerance=float('nan')), replace(Config(), lloyd_steps=1.5)):
        try:
            predict(inputs['q'], inputs['r'], inputs['cov'], inputs['base'], cfg)
        except ValueError:
            bad += 1
    assert bad == 4
    report = dict(scope='synthetic CPU implementation checks; no real frozen DINO gain measurement',
                  config=result['info']['config'], witness=dict(token=token, complete1024=full, method=result['info'],
                  class_moments=dict(fg_mean=foreground[:, :2].astype(float).mean(0).tolist(),
                    bg_mean=background[:, :2].astype(float).mean(0).tolist(),
                    fg_second_moment=(foreground[:, :2].astype(float).T@foreground[:, :2].astype(float)/6).tolist(),
                    bg_second_moment=(background[:, :2].astype(float).T@background[:, :2].astype(float)/3).tolist()),
                  quadratic_maximum_distance_from_half=float(np.max(np.abs(quadratic['field']-.5)))),
                  semantic_failure=dict(token=failure_token,
                    complete1024=dict(base=counts(render(failure_base).astype(float), truth1024),
                                      hull=counts(render(failure['field']).astype(float), truth1024)),
                    interpretation='A real target with a BG anchor appearance is confidently deleted; bounds are numerical only'),
                  shared_point=dict(fallback_rows=tied['info']['fallback_rows'], exact_base_preservation=True),
                  missing_reference_class=dict(abstention=True, exact_base_preservation=True),
                  SLSQP_bound_checks=checks, invalid_configuration_checks=bad,
                  query_GT_used_by_prediction=False, encoder_forwards=0,
                  total_seconds=time.perf_counter()-started,
                  code_sha256={p: sha(ROOT/p) for p in ['src/ics/methods/reference_hull.py', 'scripts/run_reference_hull.py']})
    (OUT / 'checks.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(token=token, complete1024=full, solver=result['info'], total_seconds=report['total_seconds']), indent=2))


if __name__ == '__main__':
    main()
