"""Small CPU checks of the fixed Pro M2 math. No natural segmentation data."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.ics.methods import pro_reference_relations as m


def complete_edges(n):
    return np.column_stack(np.triu_indices(n, 1))


def independent_tables(a, coverage, edges, independent=False):
    k = a.shape[1]
    weights = np.c_[1-coverage.ravel(), coverage.ravel()]
    marginal = np.array([(weights[:, s, None]*a).sum(0) + 1/k for s in range(2)])
    marginal /= (weights.sum(0) + 1)[:, None]
    counts = np.zeros((2, 2, k, k))
    for i, j in edges:
        for s in range(2):
            for t in range(2):
                counts[s, t] += weights[i, s]*weights[j, t]*np.outer(a[i], a[j])
                counts[s, t] += weights[j, s]*weights[i, t]*np.outer(a[j], a[i])
    if np.any(counts.sum((2, 3)) == 0):
        return None
    result = counts / counts.sum((2, 3))[:, :, None, None]
    if independent:
        result = np.array([[np.outer(row.sum(1), row.sum(0)) for row in state] for state in result])
    return .9*result + .1*marginal[:, None, :, None]*marginal[None, :, None, :]


def run():
    started = time.perf_counter()
    rng = np.random.default_rng(6406)
    cfg = m.Config()
    edges = complete_edges(12)
    a = np.c_[np.linspace(.05, .95, 12), np.linspace(.95, .05, 12)]
    p = np.array([[1e-14, 1-1e-14], [.65, .35]])
    factorized = p[:, None, :, None]*p[None, :, None, :]
    factorized_raw = m.relation_values(factorized, a, edges)
    assert np.array_equal(factorized_raw, np.zeros(len(edges)))

    table_error = 0.
    for independent in (False, True):
        coverage = rng.uniform(size=(3, 4))
        tables, info = m.estimate_tables(a, coverage, edges, pair_independent=independent)
        target = independent_tables(a, coverage, edges, independent)
        table_error = max(table_error, float(np.max(np.abs(tables-target))))
        assert np.allclose(tables, target, atol=2e-15, rtol=0)
        assert np.allclose(tables.sum((2, 3)), 1, atol=1e-15)
        direct = []
        for i, j in edges:
            likelihood = np.array([[a[i]@tables[s, t]@a[j] for t in range(2)] for s in range(2)])
            direct.append(.25*np.log(likelihood[1, 1]*likelihood[0, 0]/(likelihood[1, 0]*likelihood[0, 1])))
        direct = np.array(direct)
        direct[np.abs(direct) < 1e-10] = 0
        assert np.allclose(m.relation_values(tables, a, edges), np.clip(direct, -2, 2), atol=2e-15, rtol=0)

    base = np.array([[0., .51, .19, .97], [.12, .88, .63, .3], [.65, .48, .03, 1.]])
    zero = sparse.csr_matrix((base.size, base.size))
    field, _, zero_info = m.solve_field(base, zero)
    assert np.array_equal(field, base)
    candidate, work = m.render_original(field, (41, 59))
    original, original_work = m.render_original(base, (41, 59))
    assert np.array_equal(candidate, original) and np.array_equal(work, original_work)
    wrong, wrong_work = m.render_original(expit(base-.5), (41, 59))
    wrong_pixel_difference = int(np.count_nonzero(work != wrong_work))
    assert wrong_pixel_difference > 0, 'Witness must catch accidental sigmoid readout'

    optimizer_rows = []
    for _ in range(8):
        raw = rng.uniform(-2, 2, len(edges))
        relation, normal_info = m.normalize_relation(edges, raw, 12)
        field, z, solver = m.solve_field(base, relation)
        dense = relation.toarray()
        unary = base.ravel()-.5

        def objective(x):
            spin = 2*x-1
            return (-unary@x - .9*.5*(spin@dense@spin)
                    + np.sum(x*np.log(x)+(1-x)*np.log1p(-x)))

        def gradient(x):
            return -unary-1.8*(dense@(2*x-1))+np.log(x)-np.log1p(-x)

        result = minimize(objective, expit(unary), jac=gradient, method='BFGS',
                          options={'gtol': 1e-10, 'maxiter': 300})
        gap = float(np.max(np.abs(z.ravel()-result.x)))
        assert gap <= solver['fixed_point_error_bound'] + 2e-8
        assert np.max(np.abs(gradient(result.x))) < 1e-7
        assert np.max(np.asarray(abs(relation).sum(1))) <= 1+1e-14
        assert np.array_equal(relation.toarray(), relation.toarray().T)
        assert np.allclose(field.ravel(), base.ravel()+1.8*dense@(2*z.ravel()-1), atol=1e-15)
        optimizer_rows.append(dict(maximum_optimizer_difference=gap, **solver))

    # One supplied mathematical table witness, not a learned/default feature witness.
    p_same = np.array([[.4, .1], [.1, .4]])
    p_cross = np.array([[.1, .4], [.4, .1]])
    tables = np.zeros((2, 2, 2, 2))
    for s in range(2):
        for t in range(2):
            tables[s, t] = .9*(p_same if s == t else p_cross)+.1*.25
    hard_roles = np.eye(2)[[0, 1, 0]]
    small_edges = complete_edges(3)
    signed_raw = m.relation_values(tables, hard_roles, small_edges)
    small_base = np.array([[.9, .7, .9]])
    sign_masks = {}
    for arm, raw in dict(signed=signed_raw, zero=0*signed_raw,
                         positive=np.maximum(signed_raw, 0), absolute=np.abs(signed_raw)).items():
        matrix, _ = m.normalize_relation(small_edges, raw, 3)
        field, _, info = m.solve_field(small_base, matrix)
        if arm == 'signed':
            assert field.max() > 1, 'Signed output must not be silently clamped'
        sign_masks[arm] = (field > .5).ravel().tolist()
    assert sign_masks['signed'] == [True, False, True]
    assert all(sign_masks[arm] == [True, True, True] for arm in ('zero', 'positive', 'absolute'))

    # <=16-token injectable all-arm path, including block leaveout through the real estimator.
    coverage = np.tile([1., 1., 1., 1., 0., 0., 0., 0.], 2).reshape(16, 1)
    roles = np.eye(2)[np.tile([0, 0, 1, 1, 0, 0, 1, 1], 2)]
    reference_edges = complete_edges(16)
    fields, core = m.infer_roles(roles, roles[:12], coverage, base,
                                {'injected': reference_edges}, {'injected': edges})
    assert set(fields) == set(m.ARMS)
    assert np.array_equal(fields['zero'], base)
    assert core['relation_arms']['absolute']['negative_edges'] == 0
    assert core['relation_arms']['positive']['negative_edges'] == 0
    rr, cc = np.indices(coverage.shape)
    groups = (2*((rr//8) % 2)+(cc//8) % 2).ravel()
    deleted = []
    for held_out in range(4):
        kept = reference_edges[(groups[reference_edges[:, 0]] != held_out) & (groups[reference_edges[:, 1]] != held_out)]
        tables = independent_tables(roles, coverage, kept)
        deleted.append(m.relation_values(tables, roles[:12], edges))
    deleted = np.asarray(deleted)
    block_raw = np.where(np.all(deleted > 0, 0) | np.all(deleted < 0, 0), deleted.mean(0), 0)
    block_matrix, _ = m.normalize_relation(edges, block_raw, 12)
    block_field, _, _ = m.solve_field(base, block_matrix)
    assert np.array_equal(block_field, fields['block'])

    # Default dictionary and the four edge types on a small synthetic feature packet.
    r = np.eye(4)[np.tile([0, 1, 2, 3], 4)]
    q = np.eye(4)[np.tile([3, 1, 0, 2], 4)]
    cov = np.tile([1., 1., 0., 0.], 4).reshape(4, 4)
    score = rng.uniform(size=(4, 4)).astype(np.float32)
    default = m.predict(q, r, cov, score)
    assert default['info']['dictionary']['actual_roles'] == 4
    assert set(default['info']['core']['table_types']) == {'spatial1', 'spatial2', 'spatial4', 'mutual20'}
    cached_base = (score-score.min())/max(float(score.max()-score.min()), 1e-6)
    assert np.array_equal(default['zero_control'], cached_base)
    for key in ('field', *(f'{arm}_control' for arm in m.ARMS if arm != 'signed')):
        assert np.isfinite(default[key]).all()
    empty = m.predict(q, r, np.zeros_like(cov), score)
    assert all(not empty[key].any() for key in ('field', *(f'{arm}_control' for arm in m.ARMS if arm != 'signed')))
    no_background = m.predict(q, r, np.ones_like(cov), score)
    assert np.array_equal(no_background['field'], cached_base)
    zero_features = m.predict(np.zeros_like(q), np.zeros_like(r), cov, score)
    assert np.isfinite(zero_features['field']).all()
    assert zero_features['info']['dictionary']['actual_roles'] == 1

    # Exact top20 boundary ties and mutuality on a 25-token tiny feature matrix.
    feature = np.eye(5)[np.arange(25) % 5]
    graph = m.mutual_edges(feature)
    cosine = feature@feature.T
    np.fill_diagonal(cosine, -np.inf)
    order = np.argsort(-cosine, axis=1, kind='stable')[:, :20]
    expected = np.array([(i, j) for i in range(25) for j in range(i+1, 25)
                         if j in order[i] and i in order[j]])
    assert np.array_equal(graph, expected)
    cfg_batch = m.Config(affinity_batch=3)
    assert np.array_equal(graph, m.mutual_edges(feature, cfg_batch))

    # Independent direct equal-type mean, including zero-valued inactive type.
    union, value = m.combine_types({'a': np.array([[0, 1], [1, 2]]), 'b': np.array([[0, 1]])},
                                  {'a': np.array([1., 2.]), 'b': np.array([0.])}, 3)
    assert np.array_equal(union, [[0, 1], [1, 2]]) and np.array_equal(value, [.5, 2.])
    source = ROOT/'src/ics/methods/pro_reference_relations.py'
    report = dict(status='passed', checks='small synthetic/independent math only; no natural episode',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_relative=str(source.relative_to(ROOT)), pro_recipe=m.Config().__dict__,
                  factorized_exact_zero=True, factorized_minimum_probability=float(factorized.min()),
                  weighted_table_maximum_difference=table_error, zero_original_renderer_pixel_difference=0,
                  rejected_sigmoid_work_pixel_difference=wrong_pixel_difference,
                  fixed_point_optimizer_checks=optimizer_rows, injected_table_witness_masks=sign_masks,
                  injected_roles_six_arm_info=core, default_small_packet_info=default['info'],
                  mutual_tie_and_batch_parity=True, empty_reference_empty=True,
                  full_reference_zero_relation=True, zero_features_top1_finite=True,
                  real_episodes=0, real_segmentation_gain='unmeasured',
                  runtime_scope='all tiny checks, includes CPU renderer import; not real candidate runtime',
                  wall_seconds=time.perf_counter()-started)
    destination = Path(__file__).with_name('math_check.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'source_sha256', 'weighted_table_maximum_difference',
                                                'rejected_sigmoid_work_pixel_difference', 'real_episodes', 'wall_seconds')}))


if __name__ == '__main__':
    run()
