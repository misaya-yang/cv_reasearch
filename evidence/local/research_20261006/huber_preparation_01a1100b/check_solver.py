"""Independent bounded optimizer, analytical examples and synthetic sparse CPU cost."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy.optimize import minimize
from ics.methods.huber_graph import Config, predict, quadratic_control


def independent_solution(y, a, left, right, weight, cfg):
    y, a = y.ravel(), a.ravel()
    def loss(z):
        difference = z[left] - z[right]
        radius = cfg.transition
        magnitude = np.abs(difference)
        term = np.where(magnitude <= radius, .5 * magnitude**2,
                        radius * (magnitude-.5*radius))
        slope = cfg.graph_lambda * weight * np.clip(difference, -radius, radius)
        gradient = a * (z-y)
        np.add.at(gradient, left, slope)
        np.add.at(gradient, right, -slope)
        return .5*np.sum(a*(z-y)**2)+np.sum(cfg.graph_lambda*weight*term), gradient
    result = minimize(loss, np.clip(y,0,1), jac=True, method='L-BFGS-B', bounds=[(0,1)]*len(y),
                      options=dict(gtol=1e-12, ftol=1e-15, maxiter=5000))
    assert result.success, result.message
    return result.x, float(result.fun)


def main():
    cfg = Config(gap_per_token=1e-12)
    rng = np.random.RandomState(20261007)
    left, right = np.triu_indices(6, 1)
    error = 0.0
    gap_coverages = []
    for _ in range(16):
        y = rng.uniform(-.2,1.2,size=(2,3))
        a = rng.uniform(.2,2,size=(2,3))
        weights = rng.uniform(.01,.2,size=len(left))
        solution, info = predict(y,a,left,right,weights,cfg)
        expected, value = independent_solution(y,a,left,right,weights,cfg)
        distance = float(np.max(np.abs(solution.ravel()-expected)))
        error = max(error, distance)
        assert distance <= info['maximum_field_error_bound']+1e-7
        assert value >= info['dual']-1e-8 and value <= info['primal']+1e-8
        gap_coverages.append(info['gap'])
    # In the entire quadratic branch, Huber and quadratic objectives coincide.
    y = rng.uniform(0,1,size=(2,3)); a = np.ones_like(y)
    weights = np.full(len(left),.05)
    quadratic, _ = quadratic_control(y,a,left,right,weights)
    huber, _ = predict(y,a,left,right,weights,Config(transition=2,gap_per_token=1e-12))
    np.testing.assert_allclose(huber,quadratic,atol=1e-5)
    # Analytical saturated star: force per edge lambda*w*rho=.04.
    y = np.array([[.9,.1,.1,.1,.1,.1,.1]])
    a = np.ones_like(y)
    left, right = np.zeros(6,dtype=int), np.arange(1,7)
    weights = np.full(6,.05)
    huber, info = predict(y,a,left,right,weights,cfg)
    expected = np.array([[.66,.14,.14,.14,.14,.14,.14]])
    np.testing.assert_allclose(huber,expected,atol=1e-5)
    quadratic, _ = quadratic_control(y,a,left,right,weights)
    assert huber[0,0] > .5 and quadratic[0,0] < .5
    star = dict(target=y.tolist(), huber=huber.tolist(), quadratic=quadratic.tolist(),
                hypothetical_source_positive_token_survives=bool(huber[0,0]>.5),
                empirical_segmentation=False)
    # Isolated nodes and zero regularization obey the exact box projection.
    empty = np.array([],dtype=int)
    isolated, isolated_info = predict(y,a,empty,empty,np.array([],dtype=float),cfg)
    np.testing.assert_array_equal(isolated,y)
    zero, _ = predict(y,a,left,right,weights,Config(graph_lambda=0))
    np.testing.assert_array_equal(zero,y)
    rejected = False
    try:
        predict(y,a,left,right,weights,Config(maximum_iterations=1,check_every=1,gap_per_token=1e-16))
    except RuntimeError:
        rejected = True
    assert rejected
    # Synthetic full-size degree-20 sparse graph, not an actual DINO graph/cohort.
    n = 4096
    i = np.arange(n)
    pairs = np.concatenate([np.sort(np.column_stack([i,(i+shift)%n]),axis=1)
                            for shift in (1,3,7,11,17,23,31,41,53,67)])
    pairs = np.unique(pairs,axis=0)
    left, right = pairs.T
    weights = np.full(len(left),.05)
    y = rng.uniform(0,1,size=(64,64))
    a = .1+np.abs(2*y-1); a /= a.mean()
    _, full_info = predict(y,a,left,right,weights)
    report = dict(kind='analytical_and_synthetic_only', real_episodes=0,
                  independent_bounded_optimizer_cases=16, maximum_coordinate_difference=error,
                  maximum_small_case_gap=max(gap_coverages),
                  quadratic_branch_equivalence_checked=True, saturated_star=star,
                  isolated_and_zero_lambda_checked=True, uncertified_solver_rejected=rejected,
                  synthetic_graph=full_info, real_miou='unmeasured', real_runtime='unmeasured',
                  no_gpu=True,no_server=True)
    Path(__file__).with_name('solver_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(maximum_coordinate_difference=error, saturated_star=star,
                          synthetic_seconds=full_info['wall_seconds'],
                          iterations=full_info['iterations'],gap=full_info['gap'],real_episodes=0)))


if __name__ == '__main__':
    main()
