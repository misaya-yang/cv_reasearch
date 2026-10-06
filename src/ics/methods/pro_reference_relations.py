"""Pro M2 four-state reference-label relations, CPU cached-feature variant.

The mathematical mechanism follows the supplied Pro report, lines 388--519.
This does not reproduce its RGB/FP32 DINO producer: callers must bind their
cached feature/score producer. No query labels, class IDs, or fold inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import sparse
from scipy.special import expit


ARMS = ('signed', 'zero', 'positive', 'absolute', 'pair_independent', 'block')


@dataclass(frozen=True)
class Config:
    roles: int = 32
    kmeans_maxiter: int = 20
    role_topk: int = 2
    role_tau: float = .07
    feature_neighbors: int = 20
    product_mix: float = .1
    zero_tolerance: float = 1e-10
    raw_clip: float = 2.
    interaction_lambda: float = .9
    fixed_point_maxiter: int = 250
    fixed_point_atol: float = 1e-6
    reference_block_width: int = 8
    affinity_batch: int = 256  # Computation/memory choice, not graph truncation.


def _validate_config(cfg):
    if any(getattr(cfg, key) != getattr(Config(), key) for key in asdict(cfg)
           if key != 'affinity_batch') or cfg.affinity_batch < 1:
        raise ValueError('This version fixes the Pro M2 recipe; only affinity batching may vary')


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty feature matrix required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(np.divide(x, norm, out=np.zeros_like(x), where=norm > 0))


def _cosine(left, right):
    # Some macOS Accelerate builds report stale FP flags on bounded dot products.
    # Validate the actual result rather than treating those flags as an output.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        result = left @ right.T
    if not np.isfinite(result).all() or np.max(np.abs(result), initial=0) > 1.001:
        raise FloatingPointError('Nonfinite or unbounded normalized cosine product')
    return result


def role_dictionary(features, cfg=Config()):
    """Farthest-point spherical Lloyd; seed 0, lower IDs resolve all ties.

    Empty clusters are removed without reinitialization. Zero feature vectors
    remain zero. The Pro report does not specify the first farthest-point seed;
    choosing the first combined R/Q token is disclosed in the preparation card.
    """
    _validate_config(cfg)
    x = _unit(features)
    count = min(cfg.roles, len(x))
    seeds = [0]
    nearest = _cosine(x, x[:1]).ravel()
    chosen = np.zeros(len(x), dtype=bool)
    chosen[0] = True
    for _ in range(1, count):
        candidate = np.where(chosen, np.inf, nearest)
        nxt = int(np.argmin(candidate))
        seeds.append(nxt)
        chosen[nxt] = True
        nearest = np.maximum(nearest, _cosine(x, x[nxt:nxt+1]).ravel())
    centers = x[seeds].copy()
    # CSR rows are clusters, with ascending token columns. Its dot product
    # accumulates the same FP64 token values in the same per-cluster order as
    # np.add.at, without NumPy's unbuffered scalar scatter over every channel.
    # Convert once; the incidence matrix has exactly one nonzero per token.
    x64 = x.astype(np.float64)
    token_ids = np.arange(len(x))
    incidence_values = np.ones(len(x), dtype=np.float64)
    previous = None
    iterations = 0
    for iterations in range(1, cfg.kmeans_maxiter + 1):
        labels = _cosine(x, centers).argmax(1)
        if previous is not None and np.array_equal(labels, previous):
            break
        mass = np.bincount(labels, minlength=len(centers))
        valid = mass > 0
        incidence = sparse.csr_matrix((incidence_values, (labels, token_ids)),
                                      shape=(len(centers), len(x)))
        sums = incidence @ x64
        centers = _unit(sums[valid])
        previous = (np.cumsum(valid) - 1)[labels]
    labels = _cosine(x, centers).argmax(1)
    valid = np.bincount(labels, minlength=len(centers)) > 0
    centers = centers[valid]
    return centers, dict(actual_roles=len(centers), lloyd_iterations=iterations,
                         first_seed='combined_reference_then_query_token_0')


def role_assignment(features, centers, cfg=Config()):
    x, centers = _unit(features), _unit(centers)
    similarity = _cosine(x, centers)
    top = np.argsort(-similarity, axis=1, kind='stable')[:, :min(2, len(centers))]
    logits = np.take_along_axis(similarity.astype(np.float64), top, axis=1) / cfg.role_tau
    probability = np.exp(logits - logits.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    a = np.zeros((len(x), len(centers)), dtype=np.float64)
    np.put_along_axis(a, top, probability, axis=1)
    return a


def _roles(a):
    a = np.asarray(a, dtype=np.float64)
    if (a.ndim != 2 or not len(a) or not a.shape[1] or not np.isfinite(a).all()
            or np.any(a < 0) or not np.allclose(a.sum(1), 1, rtol=0, atol=1e-12)
            or np.any(np.count_nonzero(a, axis=1) > 2)):
        raise ValueError('Role rows must be finite top2 probability vectors')
    return a


def _pack(a):
    index = np.argsort(-a, axis=1, kind='stable')[:, :min(2, a.shape[1])]
    return index, np.take_along_axis(a, index, axis=1)


def canonical_edges(edges, n):
    edges = np.asarray(edges, dtype=np.int64)
    if not edges.size:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError('Edges require valid token endpoints')
    edges = np.sort(edges, axis=1)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def mutual_edges(features, cfg=Config()):
    """Exact cosine mutual20, blockwise memory; smaller graphs use min(20,N-1)."""
    x = _unit(features)
    n, neighbors = len(x), min(cfg.feature_neighbors, len(x) - 1)
    if not neighbors:
        return np.empty((0, 2), dtype=np.int64)
    indices = np.empty((n, neighbors), dtype=np.int64)
    for start in range(0, n, cfg.affinity_batch):
        stop = min(start + cfg.affinity_batch, n)
        similarity = _cosine(x[start:stop], x)
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        thresholds = np.partition(similarity, n - neighbors, axis=1)[:, n - neighbors]
        for row, threshold in enumerate(thresholds):
            ids = np.flatnonzero(similarity[row] >= threshold)
            order = np.lexsort((ids, -similarity[row, ids]))
            indices[start + row] = ids[order[:neighbors]]
    directed = sparse.csr_matrix((np.ones(n * neighbors, dtype=bool),
                                 (np.repeat(np.arange(n), neighbors), indices.ravel())), shape=(n, n))
    reciprocal = sparse.triu(directed.multiply(directed.T), k=1).tocoo()
    return canonical_edges(np.column_stack((reciprocal.row, reciprocal.col)), n)


def edge_types(features, shape, cfg=Config()):
    if len(shape) != 2 or int(np.prod(shape)) != len(features):
        raise ValueError('Feature-grid shape mismatch')
    ids = np.arange(len(features)).reshape(shape)
    result = {}
    for distance in (1, 2, 4):
        horizontal = np.column_stack((ids[:, :-distance].ravel(), ids[:, distance:].ravel()))
        vertical = np.column_stack((ids[:-distance, :].ravel(), ids[distance:, :].ravel()))
        result[f'spatial{distance}'] = canonical_edges(np.concatenate((horizontal, vertical)), len(features))
    result['mutual20'] = mutual_edges(features, cfg)
    return result


def estimate_tables(a_ref, coverage, edges, cfg=Config(), *, pair_independent=False):
    """Four weighted directed label-state tables with the same .1 product mixture."""
    a = _roles(a_ref)
    m = np.asarray(coverage, dtype=np.float64).ravel()
    if m.shape != (len(a),) or not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
        raise ValueError('Reference coverage must match roles and lie in [0,1]')
    edges = canonical_edges(edges, len(a))
    w = np.column_stack((1 - m, m))
    k = a.shape[1]
    marginal = (w.T @ a + 1 / k) / (w.sum(0)[:, None] + 1)
    index, probability = _pack(a)
    left, right = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
    pair = index[left, :, None] * k + index[right, None, :]
    affinity = probability[left, :, None] * probability[right, None, :]
    counts = np.zeros((2, 2, k, k), dtype=np.float64)
    for s in (0, 1):
        for t in (0, 1):
            weight = affinity * (w[left, s] * w[right, t])[:, None, None]
            counts[s, t] = np.bincount(pair.ravel(), weights=weight.ravel(), minlength=k*k).reshape(k, k)
    totals = counts.sum(axis=(2, 3))
    if np.any(totals == 0):
        return None, dict(active=False, state_edge_mass=totals.tolist())
    empirical = counts / totals[:, :, None, None]
    if pair_independent:
        empirical = empirical.sum(3)[:, :, :, None] * empirical.sum(2)[:, :, None, :]
    product = marginal[:, None, :, None] * marginal[None, :, None, :]
    tables = (1 - cfg.product_mix) * empirical + cfg.product_mix * product
    return tables, dict(active=True, state_edge_mass=totals.tolist())


def relation_values(tables, a_query, edges, cfg=Config()):
    """Injectable table-to-edge core. No epsilon that would break factorization."""
    a = _roles(a_query)
    edges = canonical_edges(edges, len(a))
    if tables is None or not len(edges):
        return np.zeros(len(edges), dtype=np.float64)
    tables = np.asarray(tables, dtype=np.float64)
    k = a.shape[1]
    if tables.shape != (2, 2, k, k) or not np.isfinite(tables).all() or np.any(tables <= 0):
        raise ValueError('Strictly positive finite four-state tables required')
    index, probability = _pack(a)
    left, right = edges.T
    affinity = probability[left, :, None] * probability[right, None, :]
    log_likelihood = np.empty((2, 2, len(edges)))
    for s in (0, 1):
        for t in (0, 1):
            lookup = tables[s, t][index[left, :, None], index[right, None, :]]
            likelihood = np.sum(lookup * affinity, axis=(1, 2))
            if np.any(likelihood <= 0):
                raise FloatingPointError('Underflowed relation likelihood; no absolute epsilon fallback')
            log_likelihood[s, t] = np.log(likelihood)
    raw = .25 * (log_likelihood[1, 1] + log_likelihood[0, 0]
                 - log_likelihood[1, 0] - log_likelihood[0, 1])
    raw[np.abs(raw) < cfg.zero_tolerance] = 0
    return np.clip(raw, -cfg.raw_clip, cfg.raw_clip)


def combine_types(query_groups, values, n):
    """Equal type average on the union, including inactive zero-valued types."""
    keys = sorted(query_groups)
    edges = np.concatenate([query_groups[key] for key in keys])
    raw = np.concatenate([values[key] for key in keys])
    if not len(edges):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    codes, inverse = np.unique(edges[:, 0] * n + edges[:, 1], return_inverse=True)
    raw = np.bincount(inverse, weights=raw) / np.bincount(inverse)
    return np.column_stack((codes // n, codes % n)), raw


def normalize_relation(edges, raw, n):
    original = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = canonical_edges(original, n)
    if not np.array_equal(original, edges):
        raise ValueError('Raw relations require already canonical, unique edge order')
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (len(edges),) or not np.isfinite(raw).all():
        raise ValueError('One finite raw value per canonical edge required')
    degree = np.bincount(edges.ravel(), weights=np.repeat(np.abs(raw), 2), minlength=n)
    scale = float(degree.max(initial=0))
    scaled = raw / scale if scale else np.zeros_like(raw)
    matrix = sparse.csr_matrix((np.r_[scaled, scaled],
                               (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    matrix.eliminate_zeros()
    return matrix, dict(raw_maximum_absolute_degree=scale, negative_edges=int((raw < 0).sum()),
                        positive_edges=int((raw > 0).sum()), nonzero_edges=int(matrix.nnz // 2))


def solve_field(base, relation, cfg=Config()):
    """Contractive FP64 solve, with an a posteriori fixed-point error certificate."""
    _validate_config(cfg)
    base = np.asarray(base, dtype=np.float64)
    if not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError('Normalized finite unary base in [0,1] required')
    matrix = sparse.csr_matrix(relation, dtype=np.float64)
    n = base.size
    if matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
        raise ValueError('Relation matrix mismatch')
    if np.max(np.abs(matrix.diagonal()), initial=0) > 0 or np.max(np.abs((matrix - matrix.T).data), initial=0) > 1e-12:
        raise ValueError('Relation must be symmetric with zero diagonal')
    row_sum = float(np.asarray(abs(matrix).sum(1)).max(initial=0))
    if row_sum > 1 + 1e-12:
        raise ValueError('Relation maximum absolute row sum exceeds 1')
    contraction = cfg.interaction_lambda * row_sum
    unary = base.ravel() - .5
    z = expit(unary)
    delta = 0.
    for iteration in range(1, cfg.fixed_point_maxiter + 1):
        updated = expit(unary + 2 * cfg.interaction_lambda * (matrix @ (2*z - 1)))
        delta = float(np.max(np.abs(updated - z)))
        z = updated
        if delta < cfg.fixed_point_atol:
            break
    residual = float(np.max(np.abs(expit(unary + 2 * cfg.interaction_lambda * (matrix @ (2*z - 1))) - z)))
    if not np.isfinite(z).all() or delta >= cfg.fixed_point_atol:
        raise RuntimeError(f'Pro relation fixed point failed: delta={delta}, residual={residual}')
    field = base.ravel() + 2 * cfg.interaction_lambda * (matrix @ (2*z - 1))
    info = dict(iterations=iteration, iterate_delta=delta, fixed_point_residual=residual,
                contraction_bound=contraction, fixed_point_error_bound=residual / (1 - contraction),
                conservative_delta_error_bound=delta / (1 - contraction), maximum_absolute_row_sum=row_sum)
    return field.reshape(base.shape), z.reshape(base.shape), info


def infer_roles(a_ref, a_query, coverage, base, ref_groups, query_groups, cfg=Config()):
    """Inject role probabilities and graphs to exercise all six mathematical arms."""
    _validate_config(cfg)
    a_ref, a_query = _roles(a_ref), _roles(a_query)
    coverage, base = np.asarray(coverage, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if (coverage.ndim != 2 or coverage.size != len(a_ref) or base.ndim != 2 or base.size != len(a_query)
            or a_ref.shape[1] != a_query.shape[1] or set(ref_groups) != set(query_groups) or not ref_groups):
        raise ValueError('Aligned coverage/base roles and matching nonempty graph type dictionaries required')
    if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError('Invalid reference coverage')
    if not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError('Finite normalized base required')
    ref_groups = {key: canonical_edges(value, len(a_ref)) for key, value in ref_groups.items()}
    query_groups = {key: canonical_edges(value, len(a_query)) for key, value in query_groups.items()}
    if not coverage.any():
        return {arm: np.zeros_like(base) for arm in ARMS}, dict(reason='empty_reference', solvers={})

    def estimate(groups, independent=False):
        values, reports = {}, {}
        for key in sorted(groups):
            tables, reports[key] = estimate_tables(a_ref, coverage, groups[key], cfg,
                                                  pair_independent=independent)
            values[key] = relation_values(tables, a_query, query_groups[key], cfg)
        edges, raw = combine_types(query_groups, values, len(a_query))
        return edges, raw, reports

    edges, signed, table_info = estimate(ref_groups)
    _, independent, _ = estimate(ref_groups, True)
    rr, cc = np.indices(coverage.shape)
    group = (2*((rr // 8) % 2) + (cc // 8) % 2).ravel()
    deleted = []
    for held_out in range(4):
        remaining = {key: edge[(group[edge[:, 0]] != held_out) & (group[edge[:, 1]] != held_out)]
                     for key, edge in ref_groups.items()}
        _, value, _ = estimate(remaining)
        deleted.append(value)
    deleted = np.asarray(deleted)
    consistent = np.all(deleted > 0, axis=0) | np.all(deleted < 0, axis=0)
    block = np.where(consistent, deleted.mean(0), 0)
    raw_arms = dict(signed=signed, zero=np.zeros_like(signed), positive=np.maximum(signed, 0),
                    absolute=np.abs(signed), pair_independent=independent, block=block)
    fields, solvers, relation_info = {}, {}, {}
    for arm, raw in raw_arms.items():
        relation, relation_info[arm] = normalize_relation(edges, raw, len(a_query))
        fields[arm], _, solvers[arm] = solve_field(base, relation, cfg)
    return fields, dict(table_types=table_info, query_union_edges=len(edges),
                        relation_arms=relation_info, solvers=solvers,
                        block_consistent_edges=int(consistent.sum()))


def render_original(field, original_shape, work_shape=(1024, 1024)):
    """Pro renderer: FP32, bilinear/strict .5, binary, bilinear/strict .5.

    CPU torch import is lazy; only needed for the complete renderer parity check.
    """
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))[None, None]
    work = functional.interpolate(tensor, work_shape, mode='bilinear', align_corners=False) > .5
    original = functional.interpolate(work.float(), tuple(original_shape), mode='bilinear', align_corners=False) > .5
    return original[0, 0].numpy(), work[0, 0].numpy()


def predict(q, r, coverage, score, cfg=Config(), *, dictionary=None,
            ref_groups=None, query_groups=None, score_is_normalized=False):
    """Default CPU cached variant; raw score minmax matches current cache helper.

    Returns signed `field` and five named controls plus all-arm metadata. Passing
    precomputed centers/graphs is optional and must be disclosed by the caller.
    Producer identity and mapping from cached q/r to the Pro representation are
    unresolved caller bindings, not assertions made by this module.
    """
    started = time.perf_counter()
    _validate_config(cfg)
    q, r = _unit(q), _unit(r)
    coverage, score = np.asarray(coverage, dtype=np.float64), np.asarray(score, dtype=np.float32)
    if (coverage.ndim != 2 or score.ndim != 2 or coverage.size != len(r) or score.size != len(q)
            or q.shape[1] != r.shape[1] or not np.isfinite(coverage).all() or not np.isfinite(score).all()
            or np.any((coverage < 0) | (coverage > 1))):
        raise ValueError('Finite aligned feature/coverage/score packet required')
    base = (score.astype(np.float64) if score_is_normalized else
            ((score-score.min()) / max(float(score.max()-score.min()), 1e-6)).astype(np.float64))
    if np.any((base < 0) | (base > 1)):
        raise ValueError('Normalized score outside [0,1]')
    if not coverage.any():
        fields, core_info = {arm: np.zeros_like(base) for arm in ARMS}, dict(reason='empty_reference')
        dictionary_info = dict(actual_roles=0)
    else:
        dictionary_started = time.perf_counter()
        if dictionary is None:
            centers, dictionary_info = role_dictionary(np.concatenate((r, q)), cfg)
        else:
            centers = _unit(dictionary)
            if centers.shape[1] != q.shape[1] or len(centers) > cfg.roles:
                raise ValueError('Injected role dictionary shape invalid')
            dictionary_info = dict(actual_roles=len(centers), injected=True)
        a_ref, a_query = role_assignment(r, centers, cfg), role_assignment(q, centers, cfg)
        dictionary_info['dictionary_and_assignment_seconds'] = time.perf_counter()-dictionary_started
        graph_started = time.perf_counter()
        ref_groups = edge_types(r, coverage.shape, cfg) if ref_groups is None else ref_groups
        reference_graph_seconds = time.perf_counter()-graph_started
        graph_started = time.perf_counter()
        query_groups = edge_types(q, score.shape, cfg) if query_groups is None else query_groups
        query_graph_seconds = time.perf_counter()-graph_started
        core_started = time.perf_counter()
        fields, core_info = infer_roles(a_ref, a_query, coverage, base, ref_groups, query_groups, cfg)
        core_info.update(reference_graph_seconds=reference_graph_seconds,
                         query_graph_seconds=query_graph_seconds,
                         all_six_arm_core_seconds=time.perf_counter()-core_started)
    info = dict(config=asdict(cfg), dictionary=dictionary_info, core=core_info,
                method_id='pro_m2_reference_relations_cached_v0',
                score_contract='normalized caller field' if score_is_normalized else 'current cached score minmax denominator floor1e-6',
                cache_producer_binding='caller must bind; not original RGB Pro FP32 encoding reproduction',
                input_role_order='reference then query', query_gt_used=False, new_encoder_forwards=0,
                real_segmentation_gain='unmeasured', real_complete_runtime='unmeasured',
                wall_seconds=time.perf_counter()-started)
    return dict(field=fields['signed'], **{f'{arm}_control': fields[arm] for arm in ARMS if arm != 'signed'}, info=info)
