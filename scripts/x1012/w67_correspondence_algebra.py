"""W67 algebra fixtures, not a real-image segmentation experiment.

No encoder, image/feature loader, query labels, training or parameter search.
The dense implementation deliberately targets small, inspectable fixtures.
"""
from pathlib import Path
import hashlib
import json
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def grid_neighbors(hw):
    h, w = hw
    adjacency = np.zeros((h * w, h * w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            i = y * w + x
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    adjacency[i, yy * w + xx] = 1
    degree = adjacency.sum(1, keepdims=True)
    return np.divide(adjacency, degree, out=np.zeros_like(adjacency), where=degree > 0)


def centered_proximity(hw, radius):
    xy = np.indices(hw).reshape(2, -1).T
    distance = np.abs(xy[:, None] - xy[None]).max(axis=2)
    proximity = (distance <= radius).astype(np.float64)
    return (proximity - proximity.mean(0, keepdims=True)
            - proximity.mean(1, keepdims=True) + proximity.mean())


def message_grid(prior, query_hw, reference_hw, radius, batch=64):
    """A_Q T H_R W_R H_R without dense reference-matrix multiplication.

    Four query neighbors, zero-padded square reference proximity, exact double
    centering. Batching only changes memory use, not the rule.
    """
    qh, qw = query_hw; rh, rw = reference_hw
    n, m = qh * qw, rh * rw
    t = np.asarray(prior, dtype=np.float64)
    if t.shape != (n, m) or radius < 0 or int(radius) != radius:
        raise ValueError("Invalid grid or integer radius")
    grid = t.reshape(qh, qw, m)
    neighbor = np.zeros_like(grid); degree = np.zeros((qh, qw, 1))
    neighbor[1:] += grid[:-1]; degree[1:] += 1
    neighbor[:-1] += grid[1:]; degree[:-1] += 1
    neighbor[:, 1:] += grid[:, :-1]; degree[:, 1:] += 1
    neighbor[:, :-1] += grid[:, 1:]; degree[:, :-1] += 1
    np.divide(neighbor, degree, out=neighbor, where=degree > 0)
    neighbor = neighbor.reshape(n, rh, rw)
    low_y = np.maximum(np.arange(rh) - radius, 0)
    high_y = np.minimum(np.arange(rh) + radius + 1, rh)
    low_x = np.maximum(np.arange(rw) - radius, 0)
    high_x = np.minimum(np.arange(rw) + radius + 1, rw)
    counts = (high_y - low_y)[:, None] * (high_x - low_x)[None, :]
    result = np.empty((n, m))
    for start in range(0, n, batch):
        values = neighbor[start:start + batch]
        prefix = np.pad(values, ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
        window = (prefix[:, high_y[:, None], high_x[None, :]]
                  - prefix[:, low_y[:, None], high_x[None, :]]
                  - prefix[:, high_y[:, None], low_x[None, :]]
                  + prefix[:, low_y[:, None], low_x[None, :]])
        centered = window - values.sum((1, 2))[:, None, None] * counts[None] / m
        centered -= centered.mean((1, 2), keepdims=True)
        result[start:start + len(values)] = centered.reshape(len(values), m)
    return result


def repair(kernel, query_graph, reference_relation, beta=4.0):
    """One simultaneous correspondence update, retaining every kernel row sum.

    No minimizer or globally monotone energy claim is made for this one-step rule.
    """
    k = np.asarray(kernel, dtype=np.float64)
    if (k.ndim != 2 or not np.isfinite(k).all() or (k < 0).any()
            or (k.sum(1) <= 0).any()):
        raise ValueError("Require a nonnegative kernel with positive row sums")
    if query_graph.shape != (len(k), len(k)):
        raise ValueError("Query graph shape mismatch")
    if reference_relation.shape != (k.shape[1], k.shape[1]):
        raise ValueError("Reference relation shape mismatch")
    if beta == 0:
        return k.copy(), np.zeros_like(k)
    mass = k.sum(1, keepdims=True)
    prior = k / mass
    message = query_graph @ prior @ reference_relation
    log_prior = np.full_like(prior, -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    logits = log_prior + beta * message
    logits -= logits.max(1, keepdims=True)
    posterior = np.exp(logits)
    posterior /= posterior.sum(1, keepdims=True)
    return posterior * mass, message


def share(kernel, labels):
    return kernel @ labels / kernel.sum(1)


def main():
    rng = np.random.default_rng(20261016)
    t = rng.random((5, 7)); t /= t.sum(1, keepdims=True)
    y = rng.random(7)
    c = t @ y
    outer_error = float(np.max(np.abs(t @ np.outer(y, y) @ t.T - np.outer(c, c))))
    assert outer_error < 1e-12
    a, b = rng.random((5, 5)), rng.random((7, 7))
    separable_error = float(np.max(np.abs((a @ t @ b) @ y - a @ (t @ (b @ y)))))
    assert separable_error < 1e-12

    q = grid_neighbors((1, 4)); relation = centered_proximity((1, 4), 1)
    labels = np.array([1., 1., 0., 0.])
    kernel = np.array([[9., 1., 1., 1.], [1., 5., 5., 1.],
                       [1., 5., 5., 1.], [1., 1., 1., 9.]])
    corrected, message = repair(kernel, q, relation)
    before, after = share(kernel, labels), share(corrected, labels)
    assert before[1] == before[2] == .5
    assert after[1] > .5 and after[2] < .5
    assert np.array_equal(repair(kernel, q, relation, beta=0)[0], kernel)
    row_mass_error = float(np.max(np.abs(corrected.sum(1) - kernel.sum(1))))
    assert row_mass_error < 1e-12

    uniform = np.ones((4, 4))
    uniform_error = float(np.max(np.abs(repair(uniform, q, relation)[0] - uniform)))
    assert uniform_error < 1e-12
    fast_errors = []
    for query_hw, reference_hw, radius in [((1, 4), (1, 4), 1),
                                           ((3, 4), (4, 5), 2),
                                           ((1, 1), (3, 2), 1),
                                           ((2, 2), (3, 3), 9)]:
        z = rng.random((np.prod(query_hw), np.prod(reference_hw)))
        z /= z.sum(1, keepdims=True)
        dense = grid_neighbors(query_hw) @ z @ centered_proximity(reference_hw, radius)
        fast = message_grid(z, query_hw, reference_hw, radius, batch=3)
        error = float(np.max(np.abs(fast - dense)))
        assert error < 1e-12
        fast_errors.append(dict(query_hw=query_hw, reference_hw=reference_hw,
                                radius=radius, maximum_error=error))
    biased = np.tile([100., 1., 1., 1.], (4, 1))
    biased_share = share(repair(biased, q, relation)[0], labels)
    assert abs(float(biased_share.mean()) - float(labels.mean())) > .01

    permutation = np.array([1, 0, 3, 2])
    null_relation = relation[np.ix_(permutation, permutation)]
    null_share = share(repair(kernel, q, null_relation)[0], labels)
    assert np.max(np.abs(null_share - after)) > 1e-6
    assert np.allclose(np.linalg.eigvalsh(relation), np.linalg.eigvalsh(null_relation))

    wrong = kernel.copy(); wrong[[0, 3]] = wrong[[3, 0]]
    wrong_after = share(repair(wrong, q, relation)[0], labels)
    assert wrong_after[1] < .5 and wrong_after[2] > .5
    # The middle truth is the same stipulated synthetic target in both fixtures.
    # A wrong contextual anchor reverses the previously helpful decision.

    strong = np.array([[1e-6, 1e-6, 1e-6, 1.],
                       [1e-6, .9, 1e-6, .1],
                       [1e-6, 1e-6, 1e-6, 1.],
                       [1e-6, 1e-6, 1e-6, 1.]])
    strong_before = float(share(strong, labels)[1])
    strong_after = float(share(repair(strong, q, relation)[0], labels)[1])
    assert strong_before > .89 and strong_after < .5
    # This is capacity to override an anchor, not a certificate that overriding is correct.

    sigmoid = lambda x: float(1 / (1 + np.exp(-x)))
    p = .9
    lower = sigmoid(np.log(p / (1 - p)) - 1)
    assert lower > .5
    record = dict(
        status="PASS", scope="Synthetic algebra only; no real-image efficacy evidence",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        real_images=0, encoder_calls=0, real_query_GT_reads=0, model_parameter_updates=0,
        beta=4., fixture_reference_radius=1,
        identities=dict(label_outer_product_error=outer_error,
                        separable_filter_error=separable_error,
                        zero_strength_bit_identity=True, maximum_row_mass_error=row_mass_error,
                        uniform_kernel_error=uniform_error),
        fast_operator_vs_independent_dense=fast_errors,
        positive_witness=dict(reference_labels=labels.tolist(), kernel=kernel.tolist(),
                              before=before.tolist(), after=after.tolist(), message=message.tolist()),
        relation_permutation_control=dict(permutation=permutation.tolist(), after=null_share.tolist(),
                                          original_unary_and_kernel_unchanged=True,
                                          relation_spectrum_unchanged=True),
        counterexample=dict(wrong_anchor_kernel=wrong.tolist(), after=wrong_after.tolist(),
                            conclusion="Incorrect but confident context can reverse both middle decisions"),
        stronger_edit_capacity=dict(before=strong_before, after=strong_after,
                                    conclusion="Can overturn high membership; correctness is unverified"),
        no_column_mass_constraint=dict(reference_share=float(labels.mean()),
                                       mean_output_share=float(biased_share.mean())),
        old_bounded_message_rule=dict(maximum_log_odds_change=1.,
                                      unflippable_foreground_above=sigmoid(1),
                                      unflippable_background_below=sigmoid(-1),
                                      prior_09_minimum_posterior=lower))
    target = ROOT / "evidence/research/w67_correspondence_algebra.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
