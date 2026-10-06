"""Pro M3 v0: fixed held-out role prediction on an adjacent Ward tree.

No encoder or query-label access. No candidate pruning except the proved gain
upper bound. Full-native fallback must be supplied by the caller when needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import time

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


ARMS = ('pro_m3_heldout', 'pro_m3_all_role.control',
        'pro_m3_mean.control', 'pro_m3_zero.control')


@dataclass(frozen=True)
class Config:
    templates_max: int = 8
    roles_max: int = 8
    bg_max: int = 16
    min_roles: int = 4
    subregions_max: int = 16
    cluster_iterations: int = 20
    refinements: int = 5
    dummy_cost: float = 1.0
    dummy_validation: float = -2.0
    contrast_tau: float = .07
    relation_scale: float = .1
    validation_clip: float = 4.0
    validation_weight: float = .5


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def spherical_roles(x, requested, iterations=20):
    """Initial seed = first/minimum token ID; then deterministic farthest point.

    Zero means stay zero; empty roles are deleted without reseeding. Actual
    role order follows surviving initialization order.
    """
    x = unit(x)
    if not len(x):
        return np.empty((0, x.shape[1]))
    k = min(requested, len(x))
    seeds = [0]
    nearest = x @ x[0]
    while len(seeds) < k:
        distance = 1 - nearest
        distance[seeds] = -np.inf
        nxt = int(np.argmax(distance))
        seeds.append(nxt)
        nearest = np.maximum(nearest, x @ x[nxt])
    centers = x[seeds].copy()
    previous = None
    for _ in range(iterations):
        assignments = np.argmax(x @ centers.T, axis=1)
        sums = np.zeros_like(centers)
        np.add.at(sums, assignments, x)
        counts = np.bincount(assignments, minlength=len(centers))
        keep = counts > 0
        new = unit(sums[keep])
        unchanged = previous is not None and np.array_equal(previous, assignments) and keep.all()
        centers = new
        if unchanged:
            break
        previous = assignments if keep.all() else None
    return centers


def ward_tree(q, shape):
    """Exact adjacency-constrained Ward costs, updated after every merge.

    Heap entries for dead clusters are discarded; active clusters never change
    membership. This is Ward, not sorted original edges/single linkage.
    """
    h, w = shape
    n, d = q.shape
    if h * w != n:
        raise ValueError('Query shape mismatch')
    children = np.empty((max(n - 1, 0), 2), dtype=np.int32)
    size = np.zeros(2 * n - 1, dtype=np.int32)
    minimum = np.zeros(2 * n - 1, dtype=np.int32)
    sums = np.zeros((2 * n - 1, d), dtype=np.float64)
    size[:n], minimum[:n], sums[:n] = 1, np.arange(n), q
    neighbors = {i: set() for i in range(n)}
    heap = []

    def push(a, b):
        if minimum[a] > minimum[b]:
            a, b = b, a
        diff = sums[a] / size[a] - sums[b] / size[b]
        cost = float(size[a]) * float(size[b]) / (size[a] + size[b]) * float(diff @ diff)
        heapq.heappush(heap, (cost, int(minimum[a]), int(minimum[b]), a, b))

    for i in range(n):
        y, x = divmod(i, w)
        for j in ([i + 1] if x + 1 < w else []) + ([i + w] if y + 1 < h else []):
            neighbors[i].add(j)
            neighbors[j].add(i)
            push(i, j)
    active = np.ones(2 * n - 1, dtype=bool)
    active[n:] = False
    merge_costs = []
    for v in range(n, 2 * n - 1):
        while heap:
            cost, _, _, a, b = heapq.heappop(heap)
            if active[a] and active[b]:
                break
        else:
            raise RuntimeError('Ward adjacency graph disconnected')
        children[v - n] = a, b
        size[v], minimum[v], sums[v] = size[a] + size[b], min(minimum[a], minimum[b]), sums[a] + sums[b]
        adjacent = (neighbors.pop(a) | neighbors.pop(b)) - {a, b}
        active[a] = active[b] = False
        active[v] = True
        neighbors[v] = adjacent
        for other in adjacent:
            neighbors[other].discard(a)
            neighbors[other].discard(b)
            neighbors[other].add(v)
            push(v, other)
        merge_costs.append(cost)
    return children, size, minimum, unit(sums / size[:, None]).astype(np.float32), merge_costs


def node_subregions(v, children, size, minimum, limit=16):
    n = len(size) // 2 + 1
    parts = [int(v)]
    while len(parts) < limit:
        eligible = [p for p in parts if p >= n]
        if not eligible:
            break
        largest = min(eligible, key=lambda p: (-int(size[p]), int(minimum[p])))
        parts.remove(largest)
        parts.extend(int(p) for p in children[largest - n])
    return np.asarray(sorted(parts, key=lambda p: int(minimum[p])), dtype=np.int32)


def assignment(cost, dummy_cost, counter):
    """One private dummy per role, with no accidental indexing of dummy IDs."""
    k, l = cost.shape
    augmented = np.full((k, l + k), np.inf, dtype=np.float64)
    augmented[:, :l] = cost
    augmented[np.arange(k), l + np.arange(k)] = dummy_cost
    rows, cols = linear_sum_assignment(augmented)
    counter['hungarian_calls'] += 1
    result = np.full(k, -1, dtype=np.int32)
    real = cols < l
    result[rows[real]] = cols[real]
    return result


def fit_roles(cross_cost, reference_distance, query_distance, cfg, counter):
    k = len(cross_cost)
    current = assignment(cross_cost, cfg.dummy_cost, counter)

    def objective(match):
        value = sum(cross_cost[a, j] if j >= 0 else cfg.dummy_cost for a, j in enumerate(match))
        for a in range(k):
            for b in range(a + 1, k):
                if match[a] >= 0 and match[b] >= 0:
                    value += (reference_distance[a, b] - query_distance[match[a], match[b]]) ** 2 / k
        return float(value)

    best, best_value = current.copy(), objective(current)
    seen = {tuple(current)}
    for _ in range(cfg.refinements):
        costs = cross_cost.copy()
        for a in range(k):
            for b in range(k):
                if a != b and current[b] >= 0:
                    costs[a] += (reference_distance[a, b] - query_distance[:, current[b]]) ** 2 / k
        proposed = assignment(costs, cfg.dummy_cost, counter)
        value = objective(proposed)
        if tuple(proposed) in seen or value >= best_value:
            break
        best, best_value = proposed.copy(), value
        seen.add(tuple(proposed))
        current = proposed
    return best


def predict_remaining(reference_distance, query_distance, fit_ids, held_ids, fit_match, cfg, counter):
    """Deliberately has no held-out cross-image cost parameter."""
    real_fit = np.flatnonzero(fit_match >= 0)
    if len(real_fit) < 2:
        counter['degenerate_directions'] += 1
        return None, None
    available = np.setdiff1d(np.arange(len(query_distance)), fit_match[real_fit])
    predicted_cost = np.zeros((len(held_ids), len(available)), dtype=np.float64)
    for a in real_fit:
        predicted_cost += (reference_distance[np.ix_(held_ids, [fit_ids[a]])]
                           - query_distance[np.ix_(available, [fit_match[a]])].T) ** 2 / len(real_fit)
    local_match = assignment(predicted_cost, cfg.dummy_cost, counter)
    match = np.full(len(held_ids), -1, dtype=np.int32)
    real = local_match >= 0
    match[real] = available[local_match[real]]
    residual = np.full(len(held_ids), np.nan)
    residual[real] = predicted_cost[np.flatnonzero(real), local_match[real]]
    return match, residual


def heldout_value(reference_distance, query_distance, cross_cosine, bg_cosine, cfg, counter):
    k = len(reference_distance)
    values = []
    for fit_ids, held_ids in ((np.arange(0, k, 2), np.arange(1, k, 2)),
                             (np.arange(1, k, 2), np.arange(0, k, 2))):
        counter['directions_total'] += 1
        fit = fit_roles(1 - cross_cosine[fit_ids], reference_distance[np.ix_(fit_ids, fit_ids)],
                        query_distance, cfg, counter)
        match, residual = predict_remaining(reference_distance, query_distance, fit_ids, held_ids, fit, cfg, counter)
        if match is None:
            values.extend([cfg.dummy_validation] * len(held_ids))
            continue
        for b, j in zip(range(len(held_ids)), match):
            counter['predicted_roles'] += 1
            if j < 0:
                counter['dummy_roles'] += 1
                values.append(cfg.dummy_validation)
            else:
                # This is the first access to this direction's held-out cost.
                values.append((cross_cosine[held_ids[b], j] - bg_cosine[j]) / cfg.contrast_tau
                              - residual[b] / cfg.relation_scale)
    return float(np.mean(values))


def all_role_value(reference_distance, query_distance, cross_cosine, bg_cosine, cfg, counter):
    match = fit_roles(1 - cross_cosine, reference_distance, query_distance, cfg, counter)
    real = np.flatnonzero(match >= 0)
    counter['all_role_total'] += len(match)
    counter['all_role_dummy'] += int((match < 0).sum())
    if len(real) < 2:
        return cfg.dummy_validation
    values = np.full(len(match), cfg.dummy_validation)
    for a in real:
        other = real[real != a]
        residual = np.mean((reference_distance[a, other] - query_distance[match[a], match[other]]) ** 2)
        values[a] = (cross_cosine[a, match[a]] - bg_cosine[match[a]]) / cfg.contrast_tau - residual / cfg.relation_scale
    return float(np.mean(values))


def decode_tree(children, whole_gain):
    n = len(children) + 1
    optimal = np.zeros(2 * n - 1)
    choices = np.zeros(2 * n - 1, dtype=np.int8)  # 0 BG, 1 children, 2 whole FG
    for v in range(2 * n - 1):
        split = float(optimal[children[v - n]].sum()) if v >= n else -np.inf
        if split > 0:
            optimal[v], choices[v] = split, 1
        if whole_gain[v] > optimal[v]:
            optimal[v], choices[v] = whole_gain[v], 2
    mask = np.zeros(n, dtype=bool)
    stack = [(2 * n - 2, False)]
    while stack:
        v, forced = stack.pop()
        if forced or choices[v] == 2:
            if v < n:
                mask[v] = True
            else:
                stack.extend((int(c), True) for c in children[v - n])
        elif choices[v] == 1:
            stack.extend((int(c), False) for c in children[v - n])
    return mask, float(optimal[-1])


def predict(q, r, cov, score, cfg=Config(), *, native_fallback=None):
    started = time.perf_counter()
    q, r = unit(q), unit(r)
    cov, score = np.asarray(cov, dtype=np.float64), np.asarray(score, dtype=np.float64)
    if cov.ndim != 2 or score.shape != cov.shape or q.shape != r.shape or q.shape[0] != cov.size:
        raise ValueError('Require aligned q/r and coverage/score grids')
    if not all(np.isfinite(a).all() for a in (q, r, cov, score)) or np.any((cov < 0) | (cov > 1)):
        raise ValueError('Nonfinite inputs or invalid coverage')
    info = dict(config=asdict(cfg), query_gt_used=False, new_encoder_forwards=0,
                first_cluster_seed='smallest token ID', bg_definition='complement of effective coarse FG',
                real_runtime='unmeasured', hungarian_calls=0, predicted_roles=0,
                dummy_roles=0, degenerate_directions=0, directions_total=0,
                all_role_total=0, all_role_dummy=0, template_count=0)
    if not np.any(cov > 0):
        return {'fields': {a: np.zeros_like(cov, dtype=np.float32) for a in ARMS},
                'info': dict(info, fallback='empty_reference')}
    foreground = cov >= .5
    if not foreground.any():
        foreground = cov == cov.max()
    components, component_count = label(foreground)
    candidates = []
    for component in range(1, component_count + 1):
        ids = np.flatnonzero(components.ravel() == component)
        roles = spherical_roles(r[ids], min(cfg.roles_max, len(ids)), cfg.cluster_iterations)
        if len(roles) >= cfg.min_roles:
            candidates.append((int(ids[0]), roles, unit(r[ids].mean(0))))
    if len(candidates) > cfg.templates_max:
        selected = [0]
        centers = np.stack([t[2] for t in candidates])
        nearest = centers @ centers[0]
        while len(selected) < cfg.templates_max:
            distances = 1 - nearest
            distances[selected] = -np.inf
            nxt = int(np.argmax(distances))
            selected.append(nxt)
            nearest = np.maximum(nearest, centers @ centers[nxt])
        candidates = [candidates[i] for i in selected]
    info['template_count'] = len(candidates)
    info['template_role_counts'] = [len(t[1]) for t in candidates]
    info['reference_components'] = component_count
    if not candidates:
        if native_fallback is None:
            raise ValueError('No structural template: full FoRIS native fallback mask required; score thresholding is not equivalent')
        mask = native_fallback() if callable(native_fallback) else native_fallback
        mask = np.asarray(mask)
        if mask.shape != (1024, 1024) or mask.dtype != bool:
            raise ValueError('Full-native fallback must be a bool 1024x1024 mask')
        return {'fields': {}, 'masks': {a: mask.copy() for a in ARMS},
                'info': dict(info, fallback='full_foris_native_no_template', seconds=time.perf_counter() - started)}
    bg_ids = np.flatnonzero(~foreground.ravel())
    bg = spherical_roles(r[bg_ids], min(cfg.bg_max, len(bg_ids)), cfg.cluster_iterations) if len(bg_ids) else np.empty((0, r.shape[1]))
    span = float(np.ptp(score))
    s = np.zeros_like(score) if span < 1e-9 else (score - score.min()) / span
    clipped = np.clip(s.ravel(), 1 / (1 + np.exp(4)), 1 / (1 + np.exp(-4)))
    unary = np.log(clipped / (1 - clipped))
    tree_start = time.perf_counter()
    children, sizes, minimum, means, costs = ward_tree(q, cov.shape)
    info['ward_seconds'] = time.perf_counter() - tree_start
    n = len(q)
    unary_sums = np.empty(2 * n - 1)
    unary_sums[:n] = unary
    for offset, pair in enumerate(children):
        unary_sums[n + offset] = unary_sums[pair].sum()
    fg_mean = unit(r[foreground.ravel()].mean(0))
    bg_cosine = np.max(bg @ means.T, axis=0) if len(bg) else np.zeros(len(means))
    mean_value = np.clip((means @ fg_mean - bg_cosine) / cfg.contrast_tau, -4, 4)
    validation = {'heldout': np.zeros(2 * n - 1), 'all': np.zeros(2 * n - 1), 'mean': mean_value, 'zero': np.zeros(2 * n - 1)}
    # Leaves have <2 actual subregions: every structure control uses V=0.
    validation['mean'][:n] = 0
    cosine_banks = [roles @ means.T for _, roles, _ in candidates]
    distance_banks = [1 - roles @ roles.T for _, roles, _ in candidates]
    raw_validation = {'heldout': np.zeros(2 * n - 1), 'all': np.zeros(2 * n - 1),
                      'evaluated': np.zeros(2 * n - 1, dtype=bool)}
    skipped = 0
    matching_start = time.perf_counter()
    kappa = math.log1p(n)
    for v in range(n, 2 * n - 1):
        if unary_sums[v] + 2 * sizes[v] - kappa <= 0:
            skipped += 1
            continue
        parts = node_subregions(v, children, sizes, minimum, cfg.subregions_max)
        part_means = means[parts].astype(np.float64)
        query_distance = 1 - np.einsum('id,jd->ij', part_means, part_means, optimize=False)
        heldout, all_roles = [], []
        for bank, reference_distance in zip(cosine_banks, distance_banks):
            cosine = bank[:, parts]
            heldout.append(heldout_value(reference_distance, query_distance, cosine, bg_cosine[parts], cfg, info))
            all_roles.append(all_role_value(reference_distance, query_distance, cosine, bg_cosine[parts], cfg, info))
        raw_validation['heldout'][v] = max(heldout)
        raw_validation['all'][v] = max(all_roles)
        raw_validation['evaluated'][v] = True
        validation['heldout'][v] = np.clip(raw_validation['heldout'][v], -4, 4)
        validation['all'][v] = np.clip(raw_validation['all'][v], -4, 4)
    info.update(matching_seconds=time.perf_counter() - matching_start,
                gain_upper_bound_skipped_nodes=skipped, nodes=2 * n - 1,
                region_cost=kappa, last_ward_cost=float(costs[-1]) if costs else 0,
                fallback=None, bg_roles=len(bg))
    fields, objectives = {}, {}
    for arm, key in zip(ARMS, ('heldout', 'all', 'mean', 'zero')):
        gain = unary_sums + cfg.validation_weight * sizes * validation[key] - kappa
        mask, value = decode_tree(children, gain)
        fields[arm] = mask.reshape(cov.shape).astype(np.float32)
        objectives[arm] = value
    info.update(seconds=time.perf_counter() - started, objectives=objectives,
                dummy_rate=info['dummy_roles'] / max(info['predicted_roles'], 1),
                degenerate_direction_rate=info['degenerate_directions'] / max(info['directions_total'], 1),
                all_role_dummy_rate=info['all_role_dummy'] / max(info['all_role_total'], 1))
    return {'fields': fields, 'info': info, 'diagnostics': raw_validation}
