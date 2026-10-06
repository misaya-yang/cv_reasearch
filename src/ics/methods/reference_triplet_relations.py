"""Reference-supervised three-role label interactions; unvalidated CPU candidate.

The third Walsh coefficient of eight query label-state log likelihoods removes
all unary/pairwise terms in those logs. This algebra does not establish novelty,
semantic transfer, or real DINO segmentation gains. Optimization is strictly
convex after a fixed incident-mass normalization; zero relation returns the
original normalized unary field exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.special import expit

from . import pro_reference_relations as pair

STATES = np.array(list(np.ndindex(2, 2, 2)), dtype=np.int64)
SPINS = 2*STATES-1
PARITY = np.prod(SPINS, axis=1)
ARMS = ('signed', 'zero', 'absolute', 'no_third')


@dataclass(frozen=True)
class Config:
    product_mix: float = .1
    raw_clip: float = 2.
    zero_tolerance: float = 1e-10
    interaction_lambda: float = .9
    incident_mass_limit: float = .5
    fixed_point_maxiter: int = 250
    fixed_point_atol: float = 1e-7
    distances: tuple = (1, 2, 4)

    def __post_init__(self):
        # JSON converts tuple metadata to lists; restore the fixed recipe type.
        object.__setattr__(self, 'distances', tuple(self.distances))


def _validate_config(cfg):
    if cfg != Config():
        raise ValueError('This candidate fixes its recipe; no hyperparameter search interface')


def canonical_triads(triads, n):
    triads = np.asarray(triads, dtype=np.int64)
    if not triads.size:
        return np.empty((0, 3), dtype=np.int64)
    if triads.ndim != 2 or triads.shape[1] != 3 or np.any(triads < 0) or np.any(triads >= n):
        raise ValueError('Triads require three valid token endpoints')
    triads = np.sort(triads, axis=1)
    if np.any(np.diff(triads, axis=1) == 0):
        raise ValueError('Self interactions are not allowed')
    return np.unique(triads, axis=0)


def spatial_triads(shape, cfg=Config()):
    """Fixed right angles: center, vertical arm, horizontal arm; four directions."""
    _validate_config(cfg)
    if len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
        raise ValueError('Positive integer grid H/W required')
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    result = {}
    for distance in cfg.distances:
        for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            inside = (rr+dy*distance >= 0) & (rr+dy*distance < h) & (cc+dx*distance >= 0) & (cc+dx*distance < w)
            r, c = rr[inside], cc[inside]
            result[f'd{distance}_v{dy}_h{dx}'] = np.column_stack((r*w+c, (r+dy*distance)*w+c, r*w+c+dx*distance))
    return result


def ordered_triads(triads, n):
    triads = np.asarray(triads, dtype=np.int64)
    if not triads.size:
        return np.empty((0, 3), dtype=np.int64)
    if triads.ndim != 2 or triads.shape[1] != 3 or np.any(triads < 0) or np.any(triads >= n):
        raise ValueError('Ordered triads require valid endpoints')
    if np.any(np.diff(np.sort(triads, axis=1), axis=1) == 0):
        raise ValueError('Three distinct endpoints required')
    return np.unique(triads, axis=0)


def estimate_tables(a_ref, coverage, triads, cfg=Config()):
    """Eight P(role_center,role_vertical,role_horizontal | label-state) tables.

    All label states use the same .1 product shrinkage and smoothed class-role
    marginal. A missing state makes this geometric type inactive, not imputed
    from query truth. Fractional reference coverage weights are observable inputs.
    """
    _validate_config(cfg)
    a = pair._roles(a_ref)
    coverage = np.asarray(coverage, dtype=np.float64).ravel()
    if coverage.shape != (len(a),) or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError('Finite reference coverage in [0,1] required')
    triads = ordered_triads(triads, len(a))
    k = a.shape[1]
    w = np.column_stack((1-coverage, coverage))
    marginal = (np.einsum('ns,nk->sk', w, a, optimize=False)+1/k)/(w.sum(0)[:, None]+1)
    index, probability = pair._pack(a)
    i, j, l = triads.T
    codes = (index[i, :, None, None]*k*k + index[j, None, :, None]*k + index[l, None, None, :])
    affinity = probability[i, :, None, None]*probability[j, None, :, None]*probability[l, None, None, :]
    counts = np.empty((8, k, k, k), dtype=np.float64)
    for state, (s, t, u) in enumerate(STATES):
        weight = affinity*(w[i, s]*w[j, t]*w[l, u])[:, None, None, None]
        counts[state] = np.bincount(codes.ravel(), weights=weight.ravel(), minlength=k**3).reshape(k, k, k)
    totals = counts.sum((1, 2, 3))
    if np.any(totals == 0):
        return None, dict(active=False, triads=len(triads), label_state_mass=totals.tolist())
    empirical = counts/totals[:, None, None, None]
    product = np.stack([marginal[s, :, None, None]*marginal[t, None, :, None]*marginal[u, None, None, :]
                        for s, t, u in STATES])
    return (1-cfg.product_mix)*empirical+cfg.product_mix*product, dict(active=True, triads=len(triads), label_state_mass=totals.tolist())


def third_order_log(log_likelihood):
    """Hadamard/Walsh coefficient in label spins; shape (8, ...)."""
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    if log_likelihood.ndim < 1 or log_likelihood.shape[0] != 8 or not np.isfinite(log_likelihood).all():
        raise ValueError('Eight finite label-state log likelihoods required')
    return np.einsum('s,s...->...', PARITY/8, log_likelihood, optimize=False)


def remove_third_log(log_likelihood):
    """Keep all degree<=2 log components after query-role marginalization.

    Ablating before a soft-role log-sum-exp could reintroduce a third label
    interaction. This control therefore acts on the observable query likelihood
    logs; its resulting three-body relation is exactly zero by construction.
    """
    coefficient = third_order_log(log_likelihood)
    return log_likelihood-PARITY.reshape((8,)+(1,)*np.ndim(coefficient))*coefficient


def relation_values(tables, a_query, triads, cfg=Config()):
    _validate_config(cfg)
    a = pair._roles(a_query)
    triads = ordered_triads(triads, len(a))
    if tables is None or not len(triads):
        return np.zeros(len(triads)), np.zeros(len(triads))
    k = a.shape[1]
    tables = np.asarray(tables, dtype=np.float64)
    if tables.shape != (8, k, k, k) or not np.isfinite(tables).all() or np.any(tables <= 0):
        raise ValueError('Strictly positive eight-state role tables required')
    index, probability = pair._pack(a)
    i, j, l = triads.T
    affinity = probability[i, :, None, None]*probability[j, None, :, None]*probability[l, None, None, :]
    logs = np.empty((8, len(triads)))
    for state in range(8):
        lookup = tables[state][index[i, :, None, None], index[j, None, :, None], index[l, None, None, :]]
        likelihood = (lookup*affinity).sum((1, 2, 3))
        if np.any(likelihood <= 0):
            raise FloatingPointError('Underflowed likelihood; no absolute epsilon fallback')
        logs[state] = np.log(likelihood)
    values = []
    for value in (third_order_log(logs), third_order_log(remove_third_log(logs))):
        value[np.abs(value) < cfg.zero_tolerance] = 0
        values.append(np.clip(value, -cfg.raw_clip, cfg.raw_clip))
    return tuple(values)


def combine_types(groups, values, n):
    """Canonicalize symmetric three-spin terms; average repeated geometric types."""
    keys = sorted(groups)
    triads = np.concatenate([np.sort(ordered_triads(groups[key], n), axis=1) for key in keys])
    raw = np.concatenate([values[key] for key in keys])
    if not len(triads):
        return np.empty((0, 3), dtype=np.int64), np.empty(0)
    unique, inverse = np.unique(triads, axis=0, return_inverse=True)
    return unique, np.bincount(inverse, weights=raw)/np.bincount(inverse)


def normalize_relation(triads, raw, n, cfg=Config()):
    _validate_config(cfg)
    original = np.asarray(triads, dtype=np.int64).reshape(-1, 3)
    triads = canonical_triads(original, n)
    if not np.array_equal(original, triads):
        raise ValueError('Relation requires canonical unique triads')
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (len(triads),) or not np.isfinite(raw).all():
        raise ValueError('One finite relation per triad required')
    incident = np.bincount(triads.ravel(), weights=np.repeat(np.abs(raw), 3), minlength=n)
    scale = float(incident.max(initial=0))
    # A single global scale preserves signs and relative geometric strength.
    factor = max(1., scale/cfg.incident_mass_limit)
    scaled = raw/factor
    return scaled, dict(raw_maximum_incident_absolute_mass=scale, normalization_factor=factor,
                        maximum_incident_absolute_mass=scale/factor,
                        positive_triads=int((raw > 0).sum()), negative_triads=int((raw < 0).sum()),
                        nonzero_triads=int(np.count_nonzero(raw)))


def messages(m, triads, interaction):
    """sum of J*m_j*m_k for each incident token; no repeated endpoints."""
    if not len(triads):
        return np.zeros(len(m))
    i, j, k = triads.T
    value = np.column_stack((interaction*m[j]*m[k], interaction*m[i]*m[k], interaction*m[i]*m[j]))
    return np.bincount(triads.ravel(), weights=value.ravel(), minlength=len(m))


def solve_field(base, triads, interaction, cfg=Config()):
    """Strictly convex entropy-minus-cubic energy, contractive mean-field map.

    E(p)=sum[p log p+(1-p)log(1-p)-(s-.5)p]-lambda sum J m_i m_j m_k.
    Incident |J| <= .5 implies map infinity norm <= lambda=.9 and
    Hessian >= 4*(1-lambda) I by diagonal dominance. Returned field is
    s+2*lambda*messages(m), so zero terms preserve s exactly.
    """
    _validate_config(cfg)
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or not base.size or not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
        raise ValueError('Finite normalized HxW unary base in [0,1] required')
    n = base.size
    original = np.asarray(triads, dtype=np.int64).reshape(-1, 3)
    triads = canonical_triads(original, n)
    interaction = np.asarray(interaction, dtype=np.float64)
    if not np.array_equal(original, triads) or interaction.shape != (len(triads),) or not np.isfinite(interaction).all():
        raise ValueError('Canonical triad interaction mismatch')
    incident = np.bincount(triads.ravel(), weights=np.repeat(np.abs(interaction), 3), minlength=n)
    maximum = float(incident.max(initial=0))
    if maximum > cfg.incident_mass_limit+1e-12:
        raise ValueError('Maximum incident absolute mass exceeds .5')
    contraction = 2*cfg.interaction_lambda*maximum
    unary = base.ravel()-.5
    p = expit(unary)
    if not np.any(interaction):
        return base.copy(), p.reshape(base.shape), dict(iterations=0, fixed_point_residual=0.,
                    contraction_bound=0., fixed_point_error_bound=0., maximum_incident_absolute_mass=0.,
                    entropy_hessian_lower_bound=4., zero_relation_exact_unary=True)
    for iteration in range(1, cfg.fixed_point_maxiter+1):
        updated = expit(unary+2*cfg.interaction_lambda*messages(2*p-1, triads, interaction))
        delta = float(np.max(np.abs(updated-p)))
        p = updated
        if delta < cfg.fixed_point_atol:
            break
    residual = float(np.max(np.abs(expit(unary+2*cfg.interaction_lambda*messages(2*p-1, triads, interaction))-p)))
    if not np.isfinite(p).all() or delta >= cfg.fixed_point_atol:
        raise RuntimeError(f'Triplet fixed point did not converge: delta={delta}, residual={residual}')
    field = base.ravel()+2*cfg.interaction_lambda*messages(2*p-1, triads, interaction)
    return field.reshape(base.shape), p.reshape(base.shape), dict(iterations=iteration,
               fixed_point_residual=residual, contraction_bound=contraction,
               fixed_point_error_bound=residual/(1-contraction), maximum_incident_absolute_mass=maximum,
               entropy_hessian_lower_bound=4*(1-contraction), zero_relation_exact_unary=False)


def infer_roles(a_ref, a_query, coverage, base, ref_groups, query_groups, cfg=Config()):
    """Injectable observable-role core; no query labels or semantic IDs."""
    _validate_config(cfg)
    a_ref, a_query = pair._roles(a_ref), pair._roles(a_query)
    coverage, base = np.asarray(coverage, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if (coverage.ndim != 2 or coverage.size != len(a_ref) or base.ndim != 2 or base.size != len(a_query)
            or a_ref.shape[1] != a_query.shape[1] or not ref_groups or set(ref_groups) != set(query_groups)):
        raise ValueError('Reference/query roles, coverage/unary and geometric types must align')
    ref_groups = {key: ordered_triads(value, len(a_ref)) for key, value in ref_groups.items()}
    query_groups = {key: ordered_triads(value, len(a_query)) for key, value in query_groups.items()}
    values, removed, table_info = {}, {}, {}
    for key in sorted(ref_groups):
        tables, table_info[key] = estimate_tables(a_ref, coverage, ref_groups[key], cfg)
        values[key], removed[key] = relation_values(tables, a_query, query_groups[key], cfg)
    triads, raw = combine_types(query_groups, values, len(a_query))
    _, no_third = combine_types(query_groups, removed, len(a_query))
    fields, solvers, relations = {}, {}, {}
    for arm, value in dict(signed=raw, zero=np.zeros_like(raw), absolute=np.abs(raw), no_third=no_third).items():
        normalized, relations[arm] = normalize_relation(triads, value, len(a_query), cfg)
        fields[arm], _, solvers[arm] = solve_field(base, triads, normalized, cfg)
    return fields, dict(table_types=table_info, query_triads=len(triads), relations=relations, solvers=solvers,
                        no_third_control='remove degree3 from query label-state logs after role marginalization; exactly zero by algebra')


def predict(q, r, coverage, score, cfg=Config(), *, score_is_normalized=False,
            dictionary=None, ref_groups=None, query_groups=None, include_pair_control=False):
    started = time.perf_counter()
    _validate_config(cfg)
    q, r = pair._unit(q), pair._unit(r)
    coverage = np.asarray(coverage, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64 if score_is_normalized else np.float32)
    if (coverage.ndim != 2 or score.ndim != 2 or coverage.size != len(r) or score.size != len(q)
            or q.shape[1] != r.shape[1] or not np.isfinite(coverage).all() or not np.isfinite(score).all()
            or np.any((coverage < 0) | (coverage > 1))):
        raise ValueError('Aligned frozen q/r, reference coverage and raw score required')
    base = score.astype(np.float64) if score_is_normalized else ((score-score.min())/max(float(score.max()-score.min()), 1e-6)).astype(np.float64)
    if np.any((base < 0) | (base > 1)):
        raise ValueError('Normalized unary score outside [0,1]')
    if dictionary is None:
        centers, dictionary_info = pair.role_dictionary(np.concatenate((r, q)))
    else:
        centers = pair._unit(dictionary)
        if centers.shape[1] != q.shape[1] or len(centers) > pair.Config().roles:
            raise ValueError('Injected dictionary shape invalid')
        dictionary_info = dict(actual_roles=len(centers), injected=True)
    a_ref, a_query = pair.role_assignment(r, centers), pair.role_assignment(q, centers)
    ref_groups = spatial_triads(coverage.shape, cfg) if ref_groups is None else ref_groups
    query_groups = spatial_triads(score.shape, cfg) if query_groups is None else query_groups
    fields, core = infer_roles(a_ref, a_query, coverage, base, ref_groups, query_groups, cfg)
    info = dict(config=asdict(cfg), dictionary=dictionary_info, core=core,
                method_id='reference_triplet_relations_v0', query_gt_used=False, new_encoder_forwards=0,
                roles='shared Pro M2 spherical dictionary32 and top2 assignment, reference then query',
                score_contract='normalized caller score' if score_is_normalized else 'raw cached score minmax denominator floor1e-6',
                relation_scope='only third label-log coefficient; lower orders discarded',
                real_segmentation_gain='unmeasured', real_complete_runtime='unmeasured')
    output = dict(field=fields['signed'], **{f'{arm}_control': fields[arm] for arm in ARMS if arm != 'signed'})
    if include_pair_control:
        control = pair.predict(q, r, coverage, base, dictionary=centers, score_is_normalized=True)
        output['pair_m2_control'] = control['field']
        info['pair_control'] = control['info']
        info['pair_control_contract'] = 'existing Pro M2 signed, all spatial1/2/4 and mutual20 types, same dictionary and normalized unary'
    info['wall_seconds'] = time.perf_counter()-started
    return dict(**output, info=info)
