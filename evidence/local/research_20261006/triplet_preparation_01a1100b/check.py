"""Synthetic observable-role proofs, solver checks and semantic counterexample."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp, xlogy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from ics.methods import reference_triplet_relations as method
from ics.methods import pro_reference_relations as pair
from ics.experiment import render, sha

OUT = Path(__file__).resolve().parent


def reference():
    roles, coverage, triads = [], [], []
    for label in method.STATES:
        for bits in method.STATES:
            count = 3 if np.prod(2*label-1)*np.prod(2*bits-1) > 0 else 1
            for _ in range(count):
                triads.append(np.arange(len(roles), len(roles)+3))
                roles.extend((np.arange(3)*2+bits).tolist())
                coverage.extend(label.tolist())
    return np.eye(6)[roles], np.asarray(coverage).reshape(1, -1), np.asarray(triads)


def pair_control(a_ref, a_query, coverage, base, triads):
    ref_edges = pair.canonical_edges(np.concatenate([triads[:, [0, 1]], triads[:, [0, 2]], triads[:, [1, 2]]]), len(a_ref))
    query_edges = np.array([[0, 1], [0, 2], [1, 2]])
    fields, info = pair.infer_roles(a_ref, a_query, coverage, base, {'triad_pairs': ref_edges}, {'triad_pairs': query_edges})
    return fields['signed'], info


def energy(p, unary, triads, value):
    m = 2*p-1
    return float(np.sum(xlogy(p, p)+xlogy(1-p, 1-p)-unary*p)-.9*np.sum(value*np.prod(m[triads], axis=1)))


def hessian(p, triads, value):
    result = np.diag(1/(p*(1-p)))
    m = 2*p-1
    for (i, j, k), coefficient in zip(triads, value):
        for left, right, other in ((i, j, k), (i, k, j), (j, k, i)):
            result[left, right] -= 4*.9*coefficient*m[other]
            result[right, left] -= 4*.9*coefficient*m[other]
    return result


def main():
    started = time.perf_counter()
    # Adding any degree<=2 term cannot affect the third label Walsh coefficient.
    spins = method.SPINS
    lower = 2 + spins@np.array([.3, -.4, .7]) + .2*spins[:, 0]*spins[:, 1] - .5*spins[:, 1]*spins[:, 2]
    logs = lower+.37*method.PARITY
    assert abs(method.third_order_log(logs)-.37) < 1e-14
    assert abs(method.third_order_log(method.remove_third_log(logs))) < 1e-14
    lower2 = -.3 + spins@np.array([-.1, .7, -.2])-.5*spins[:, 1]*spins[:, 2]
    mixed = logsumexp(np.stack([lower, lower2]), axis=0)-np.log(2)
    mixing_third = float(method.third_order_log(mixed))
    assert abs(mixing_third) > 1e-4  # Disproves the pre-soft-marginalization ablation shortcut.
    a_ref, coverage, triads = reference()
    query_triads = np.array([[0, 1, 2]])
    a_query = np.eye(6)[[1, 3, 5]]
    base = np.array([[.499, .8, .8]])
    fields, info = method.infer_roles(a_ref, a_query, coverage, base, {'one': triads}, {'one': query_triads})
    pair_field, pair_info = pair_control(a_ref, a_query, coverage, base, triads)
    assert fields['signed'][0, 0] > .5 and pair_field[0, 0] < .5
    assert np.array_equal(fields['zero'], base) and np.array_equal(fields['no_third'], base)
    assert np.array_equal(pair_field, base)
    tables, _ = method.estimate_tables(a_ref, coverage, triads)
    raw, _ = method.relation_values(tables, a_query, query_triads)
    value, norm = method.normalize_relation(query_triads, raw, 3)
    field, pgrid, solver = method.solve_field(base, query_triads, value)
    p = pgrid.ravel()
    unary = base.ravel()-.5
    optimum = minimize(lambda p: energy(p, unary, query_triads, value), np.full(3, .5),
                       jac=lambda p: logit(p)-unary-2*.9*method.messages(2*p-1, query_triads, value),
                       bounds=[(1e-8, 1-1e-8)]*3, method='L-BFGS-B',
                       options=dict(ftol=1e-15, gtol=1e-10, maxiter=1000))
    assert optimum.success and np.max(np.abs(optimum.x-p)) < 1e-6
    eigenvalue = float(np.linalg.eigvalsh(hessian(p, query_triads, value)).min())
    assert eigenvalue >= .4-1e-12 and solver['contraction_bound'] <= .9+1e-12
    # Negative relation corrects a designated background with strong FG neighbors,
    # but deletes a designated true FG with exactly the same observable input.
    negative_query = np.eye(6)[[0, 3, 5]]
    strong_neighbors = np.array([[.501, .8, .8]])
    negative, negative_info = method.infer_roles(a_ref, negative_query, coverage, strong_neighbors,
                                                {'one': triads}, {'one': query_triads})
    assert negative['signed'][0, 0] < .5 and negative['absolute'][0, 0] > .5
    assert np.array_equal(negative['zero'], strong_neighbors)
    # Pure reference only has label111: all types inactive, exact baseline fallback.
    missing, missing_info = method.infer_roles(a_ref, a_query, np.ones_like(coverage), base,
                                              {'one': triads}, {'one': query_triads})
    assert np.array_equal(missing['signed'], base)
    # Full legal packet exercises default feature->roles and all fixed right-angle
    # geometry, independent from the injected parity construction above.
    rng = np.random.RandomState(47)
    role_ids = rng.randint(6, size=4096)
    support_ids = rng.randint(6, size=4096)
    q = np.zeros((4096, 1024), dtype=np.float32); q[np.arange(4096), role_ids] = 1
    r = np.zeros_like(q); r[np.arange(4096), support_ids] = 1
    cov = rng.randint(2, size=(64, 64)).astype(np.float32)
    score = rng.rand(64, 64).astype(np.float32)
    np.savez_compressed(OUT/'legal_inputs.npz', q=q, r=r, cov=cov, score=score,
                        unread_query_truth=np.array(['not inference input'], dtype=object))
    full = method.predict(q, r, cov, score)
    assert all(np.isfinite(full[key]).all() for key in ('field', 'zero_control', 'absolute_control', 'no_third_control'))
    assert np.array_equal(full['zero_control'], full['no_third_control'])
    assert all(s['contraction_bound'] <= .9+1e-12 for s in full['info']['core']['solvers'].values())
    np.savez_compressed(OUT/'synthetic_outputs.npz', parity_signed=fields['signed'], parity_zero=fields['zero'],
                        parity_pair_control=pair_field, negative_signed=negative['signed'],
                        negative_absolute=negative['absolute'], negative_base=strong_neighbors,
                        negative1024_packed=np.packbits(render(negative['signed'])),
                        negative_base1024_packed=np.packbits(render(strong_neighbors)),
                        full_field=full['field'], full1024_packed=np.packbits(render(full['field'])))
    report = dict(scope='synthetic CPU preparation; no real DINO experiment or claimed gain',
                  config=full['info']['config'], unary_pair_cancellation_error=float(abs(method.third_order_log(logs)-.37)),
                  pre_marginalization_ablation_counterexample=mixing_third,
                  parity=dict(reference_tokens=len(a_ref), reference_triads=len(triads), base=base.tolist(),
                              signed=fields['signed'].tolist(), pair_pro_m2=pair_field.tolist(), raw=raw.tolist(),
                              relation=norm, solvers=info['solvers'], pair_core=pair_info),
                  energy=dict(optimizer_success=bool(optimum.success), maximum_probability_difference=float(np.max(np.abs(optimum.x-p))),
                              minimum_hessian_eigenvalue=eigenvalue, analytic_lower_bound=.4,
                              contraction_bound=solver['contraction_bound'], residual=solver['fixed_point_residual']),
                  negative=dict(base=strong_neighbors.tolist(), signed=negative['signed'].tolist(), absolute=negative['absolute'].tolist(),
                                correct_BG_case='same observable first token is designated BG: signed removes false target',
                                semantic_failure='same observable first token designated FG: signed deletes a true target',
                                complete1024_all_FG_truth=dict(base_TP=int(render(strong_neighbors).sum()),
                                    signed_TP=int(render(negative['signed']).sum()),
                                    deleted_true_pixels=int((render(strong_neighbors)&~render(negative['signed'])).sum())),
                                info=negative_info),
                  zero_and_no_third_exact_base=True, missing_label_state_exact_base=True,
                  default_complete1024_shape=list(render(full['field']).shape), default_method=full['info'],
                  query_GT_used_by_inference=False, new_encoder_forwards=0,
                  wall_seconds=time.perf_counter()-started,
                  code_sha256={path: sha(ROOT/path) for path in ['src/ics/methods/reference_triplet_relations.py', 'scripts/run_reference_triplet_relations.py']})
    (OUT/'checks.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('energy','pre_marginalization_ablation_counterexample','zero_and_no_third_exact_base','wall_seconds')}, indent=2))


if __name__ == '__main__':
    main()
