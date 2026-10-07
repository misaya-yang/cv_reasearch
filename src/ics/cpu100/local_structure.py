"""CPU local DINO structure candidates; natural-image benefits are unmeasured.

The Gram classifier does not transfer reference silhouette or part order.
Synthetic invariance witnesses are conditions, not DINO domain guarantees.
"""
from __future__ import annotations

import time

import numpy as np

from .common import Episode, Result, unit, validate


STENCIL = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
           (-2, 0), (2, 0), (0, -2), (0, 2))
CHUNK = 64


def local_signatures(features, hw, valid):
    """Full 18D signature and controls; bounded nine-patch feature chunks.

    Sorted spectrum and center row are invariant to feature orthogonal changes.
    Centering cancels an additive component of its direct operands. Because
    this function first normalizes, this does not cover generic additions to
    native DINO. The witnessed affine change has a common added component
    orthogonal to the signal span, hence the same norm at every token.
    """
    features = unit(features)
    valid = np.asarray(valid, dtype=float)
    h, w = hw
    rr, cc = np.indices(hw)
    ids, weights = [], []
    for dy, dx in STENCIL:
        y, x = rr + dy, cc + dx
        inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        idx = (np.clip(y, 0, h - 1) * w + np.clip(x, 0, w - 1)).ravel()
        ids.append(idx)
        weights.append(inside.ravel() * valid[idx])
    ids, weights = np.stack(ids, 1), np.stack(weights, 1)
    full = np.empty((len(features), 18))
    diagonal = np.empty((len(features), 9))
    trace_only = np.empty((len(features), 1))
    inactive = 0
    for start in range(0, len(features), CHUNK):
        ix, wt = ids[start:start + CHUNK], weights[start:start + CHUNK]
        x = features[ix]
        mass = wt.sum(1)
        mu = np.einsum("bkd,bk->bd", x, wt) / np.maximum(mass[:, None], 1e-15)
        centered = (x - mu[:, None]) * np.sqrt(wt[:, :, None])
        gram = np.einsum("bkd,bld->bkl", centered, centered, optimize=True)
        trace = np.trace(gram, axis1=1, axis2=2)
        active = trace > 1e-12
        normalized = np.divide(gram, trace[:, None, None],
                               out=np.zeros_like(gram), where=active[:, None, None])
        spectrum = np.clip(np.linalg.eigvalsh(normalized)[:, ::-1], 0, None)
        signature = np.concatenate((spectrum, np.sort(normalized[:, 0], axis=1)), axis=1)
        full[start:start + len(ix)] = signature
        diagonal[start:start + len(ix)] = np.sort(np.diagonal(normalized, axis1=1, axis2=2), axis=1)
        trace_only[start:start + len(ix), 0] = trace / np.maximum(mass, 1)
        inactive += int((~active).sum())
    return dict(full=full, diagonal=diagonal, trace=trace_only), dict(
        zero_variation_tokens=inactive, feature_chunk=CHUNK,
        maximum_feature_chunk_elements=CHUNK * 9 * features.shape[1])


def distances2(a, b):
    return np.maximum(np.sum(a * a, axis=1)[:, None]
                      + np.sum(b * b, axis=1)[None]
                      - 2 * np.einsum("id,jd->ij", a, b, optimize=False), 0)


def kernel_margin(q, r, wf, wb):
    """Fixed class-average kernel rule; neither query labels nor query seeds."""
    reference_valid = wf + wb > 0
    r, wf, wb = r[reference_valid], wf[reference_valid], wb[reference_valid]
    if wf.sum() == 0:
        return np.full(len(q), -1.), dict(degenerate="empty_reference_foreground")
    if wb.sum() == 0:
        return np.full(len(q), 1.), dict(degenerate="empty_reference_background")
    nearest = []
    for start in range(0, len(r), CHUNK):
        d = distances2(r[start:start + CHUNK], r)
        d[d <= 1e-12] = np.inf
        nearest.extend(np.min(d, axis=1).tolist())
    nearest = np.asarray(nearest)
    distinct = nearest[np.isfinite(nearest)]
    if not len(distinct):
        return np.zeros(len(q)), dict(degenerate="all_reference_descriptors_equal", bandwidth_squared=0.)
    h2 = float(np.median(distinct))
    foreground, background = wf / wf.sum(), wb / wb.sum()
    margin = np.empty(len(q))
    unsupported = 0
    for start in range(0, len(q), CHUNK):
        kernel = np.exp(-distances2(q[start:start + CHUNK], r) / h2)
        a = np.sum(kernel * foreground[None], axis=1)
        b = np.sum(kernel * background[None], axis=1)
        den = a + b
        margin[start:start + len(kernel)] = np.divide(a - b, den, out=np.zeros_like(a), where=den > 1e-250)
        unsupported += int((den <= 1e-250).sum())
    return margin, dict(bandwidth_squared=h2, valid_reference_descriptors=len(r),
                        kernel_chunk=CHUNK, unsupported_query_descriptors=unsupported)


def _predict(ep: Episode, arm: str) -> Result:
    started = time.perf_counter()
    validate(ep)
    r, ri = local_signatures(ep.r, ep.r_hw, ep.wvalid)
    q, qi = local_signatures(ep.q, ep.q_hw, ep.q_valid)
    wf, wb = ep.wf, ep.wb
    if arm == "label_permuted":
        # Deterministic full-label intervention, not an additional method.
        valid = ep.wvalid > 0
        original_fraction = wf[valid] / ep.wvalid[valid]
        order = np.random.default_rng(0).permutation(len(original_fraction))
        wf = wf.copy()
        wf[valid] = ep.wvalid[valid] * original_fraction[order]
        wb = ep.wvalid - wf
        key = "full"
    else:
        key = arm
    margin, ki = kernel_margin(q[key], r[key], wf, wb)
    margin[ep.q_valid <= 0] = -1
    return Result(margin.reshape(ep.q_hw), dict(
        method_id="local_001", arm=arm, reference=ri, query=qi, classifier=ki,
        stencil=[list(v) for v in STENCIL], source_id=ep.source_id,
        query_ground_truth_used=False, query_seeds_used=False,
        new_encoder_forwards=0, natural_segmentation_gain="unmeasured",
        invariance_scope="orthogonal changes; common-scale-plus-added-orthogonal-component witnesses with equal token norms; no generic additive invariance",
        wall_seconds=time.perf_counter() - started))


def predict(ep: Episode, method_id: str = "local_001") -> Result:
    if method_id == "local_001":
        return _predict(ep, "full")
    if method_id == "local_002":
        return correspondence(ep)
    if method_id == "local_003":
        return patch_reconstruction(ep)
    raise KeyError(method_id)


def pooled_reference(ep):
    """At most 16x16 anchors, preserving fractional MR labels and padding."""
    h, w = ep.r_hw
    ph, pw = min(h, 16), min(w, 16)
    y, x = np.indices(ep.r_hw)
    group = ((y * ph // h) * pw + x * pw // w).ravel()
    n = ph * pw
    mass = np.bincount(group, weights=ep.wvalid, minlength=n)
    foreground = np.bincount(group, weights=ep.wf, minlength=n)
    features = np.zeros((n, ep.r.shape[1]))
    np.add.at(features, group, ep.r * ep.wvalid[:, None])
    coordinates = np.zeros((n, 2))
    positions = np.column_stack(((y.ravel() + .5) / h, (x.ravel() + .5) / w))
    np.add.at(coordinates, group, positions * ep.wvalid[:, None])
    active = mass > 0
    return (unit(features[active]), foreground[active] / mass[active],
            coordinates[active] / mass[active, None])


def query_graph(ep):
    """Four neighbors, no image wrap, query padding produces zero edges."""
    h, w = ep.q_hw
    y, x = np.indices(ep.q_hw)
    neighbor = np.zeros((len(ep.q), 4), dtype=int)
    weight = np.zeros((len(ep.q), 4))
    q = unit(ep.q)
    for j, (dy, dx) in enumerate(STENCIL[1:5]):
        yy, xx = y + dy, x + dx
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        ix = (np.clip(yy, 0, h - 1) * w + np.clip(xx, 0, w - 1)).ravel()
        neighbor[:, j] = ix
        similarity = np.einsum("nd,nd->n", q, q[ix])
        weight[:, j] = np.maximum(similarity, 0) * inside.ravel()
        weight[:, j] *= np.minimum(ep.q_valid, ep.q_valid[ix])
    return neighbor, weight


def correspondence(ep: Episode, arm="coordinate") -> Result:
    """Bounded multi-label coordinate ICM; no global optimum is claimed."""
    started = time.perf_counter()
    validate(ep)
    reference, fraction, coordinates = pooled_reference(ep)
    fg, bg = np.flatnonzero(fraction > .5), np.flatnonzero(fraction <= .5)
    if not len(fg) or not len(bg):
        constant = 1. if len(fg) else -1.
        margin = np.full(len(ep.q), constant)
        margin[ep.q_valid <= 0] = -1
        return Result(margin.reshape(ep.q_hw), dict(method_id="local_002", arm=arm,
            degenerate="foreground_or_background_lost_after_shared_pooling",
            native_foreground_weight=float(ep.wf.sum()), pooled_foreground_anchors=len(fg),
            pooled_background_anchors=len(bg), wall_seconds=time.perf_counter() - started,
            natural_segmentation_gain="unmeasured"))
    q = unit(ep.q)
    similarity = np.einsum("nd,md->nm", q, reference, optimize=False)
    # Stable sorting resolves cosine ties by original pooled reference ID.
    foreground = fg[np.argsort(-similarity[:, fg], axis=1, kind="stable")[:, :4]]
    background = bg[np.argsort(-similarity[:, bg], axis=1, kind="stable")[:, :4]]
    candidates = np.sort(np.concatenate((foreground, background), axis=1), axis=1)
    unary = 1 - np.take_along_axis(similarity, candidates, axis=1)
    # Invalid padding is outside the declared optimization domain. It cannot
    # bias a global-start comparison merely through arbitrary encoded features.
    unary[ep.q_valid <= 0] = 0
    labels = fraction[candidates] > .5
    if arm == "permuted_coordinates":
        coordinates = coordinates[np.random.default_rng(0).permutation(len(coordinates))]
    y, x = np.indices(ep.q_hw)
    position = np.column_stack(((y.ravel() + .5) / ep.q_hw[0],
                                (x.ravel() + .5) / ep.q_hw[1]))
    displacement = coordinates[candidates] - position[:, None]
    neighbor, weight = query_graph(ep)
    pooled_side = max(min(ep.r_hw[0], 16), min(ep.r_hw[1], 16))
    # A pooled reference anchor represents a finite spatial cell. Using only
    # the finer query spacing makes all different-anchor costs saturate, erasing
    # coordinate geometry on a native64 / pooled16 grid. This registered energy
    # revision uses the coarser sampling spacing; it is not a parameter sweep.
    radius_squared = (2 * max(1 / max(ep.q_hw), 1 / pooled_side)) ** 2

    def pair_cost(i, states):
        neighbors = neighbor[i]
        if arm == "label_potts":
            difference = labels[i, :, None] != labels[neighbors, states[neighbors]][None]
        else:
            other = displacement[neighbors, states[neighbors]]
            difference = np.minimum(np.sum((displacement[i, :, None] - other[None]) ** 2, axis=2)
                                    / radius_squared, 1.)
        return .1 * np.sum(difference * weight[i][None], axis=1)

    def energy(states):
        values = unary[np.arange(len(q)), states].copy()
        values += .5 * np.asarray([pair_cost(i, states)[states[i]] for i in range(len(q))])
        return float(values.sum()), values

    # Only exact zero-weight separation of the original MRF is used. There is
    # no query label/GT-based component construction or additional initial state.
    component = np.full(len(q), -1, dtype=int)
    components = 0
    for seed in range(len(q)):
        if component[seed] >= 0:
            continue
        component[seed] = components
        stack = [seed]
        while stack:
            i = stack.pop()
            for other in neighbor[i, weight[i] > 0]:
                if component[other] < 0:
                    component[other] = components
                    stack.append(int(other))
        components += 1

    starts = [np.argmin(unary, axis=1),
              np.argmin(np.where(labels, unary, np.inf), axis=1),
              np.argmin(np.where(~labels, unary, np.inf), axis=1)]
    if arm == "unary":
        states = starts[0]
        objectives, changes = [float(unary[np.arange(len(q)), states].sum())], []
        assembled_energy = objectives[0]
        chosen_component_start = None
    else:
        solutions, costs, objectives, changes = [], [], [], []
        for initial in starts:
            current = initial.copy()
            start_changes = []
            for _ in range(6):
                changed = 0
                for order in (range(len(q)), range(len(q) - 1, -1, -1)):
                    for i in order:
                        if ep.q_valid[i] <= 0:
                            continue
                        choice = int(np.argmin(unary[i] + pair_cost(i, current)))
                        changed += int(choice != current[i])
                        current[i] = choice
                start_changes.append(changed)
            value, per_token = energy(current)
            objectives.append(value)
            changes.append(start_changes)
            solutions.append(current.copy())
            costs.append(np.bincount(component, weights=per_token, minlength=components))
        chosen_component_start = np.argmin(np.asarray(costs), axis=0)
        state_by_start = np.asarray(solutions)
        states = state_by_start[chosen_component_start[component], np.arange(len(q))]
        assembled_energy, _ = energy(states)
        if assembled_energy > min(objectives) + 1e-10:
            raise AssertionError("Disconnected original energy must not increase after componentwise choice")
    selected = candidates[np.arange(len(q)), states]
    margin = 2 * fraction[selected] - 1
    margin[ep.q_valid <= 0] = -1
    return Result(margin.reshape(ep.q_hw), dict(method_id="local_002", arm=arm,
        pooled_reference_anchors=len(reference), candidates_per_token=candidates.shape[1],
        objective_by_start=objectives, state_changes_by_pass=changes,
        graph_components=components, assembled_objective=assembled_energy,
        componentwise_start_choice=None if chosen_component_start is None else chosen_component_start.tolist(),
        solver_revision=1,
        energy_revision=1, compatibility_radius_squared=radius_squared,
        pooled_reference_side=pooled_side,
        optimizer="fixed six forward/backward ICM passes, three starts; approximate MAP",
        edge_weight="nonnegative query cosine times physical edge validity",
        coordinate_scope="letterboxed patch-grid unit square, no silhouette warp",
        query_ground_truth_used=False, query_seeds_used=False, new_encoder_forwards=0,
        natural_segmentation_gain="unmeasured", wall_seconds=time.perf_counter() - started))


def tensor_index(hw, valid):
    h, w = hw
    y, x = np.indices(hw)
    indices, weights = [], []
    for dy, dx in STENCIL:
        yy, xx = y + dy, x + dx
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        ix = (np.clip(yy, 0, h - 1) * w + np.clip(xx, 0, w - 1)).ravel()
        indices.append(ix)
        weights.append(inside.ravel() * valid[ix])
    return np.stack(indices, 1), np.stack(weights, 1)


def patch_rows(features, indices, weights, rows, *, central=False):
    if central:
        return unit(features[rows])
    ix, wt = indices[rows], weights[rows]
    values = features[ix]
    mean = np.einsum("bkd,bk->bd", values, wt) / np.maximum(wt.sum(1)[:, None], 1e-15)
    tensor = ((values - mean[:, None]) * np.sqrt(wt[:, :, None])).reshape(len(rows), -1)
    return unit(tensor)


def tensor_dictionary(features, indices, weights, eligible, count):
    """Streaming farthest-patch anchors; no Nr x 9D tensor allocation."""
    eligible = np.asarray(eligible, int)
    chosen, tensors = [], []
    closest = np.full(len(eligible), np.inf)
    first = 0
    for _ in range(count):
        idx = int(eligible[first])
        chosen.append(idx)
        anchor = patch_rows(features, indices, weights, np.asarray([idx]))[0]
        tensors.append(anchor)
        for start in range(0, len(eligible), CHUNK):
            rows = eligible[start:start + CHUNK]
            patches = patch_rows(features, indices, weights, rows)
            d = np.sum((patches - anchor[None]) ** 2, axis=1)
            closest[start:start + len(rows)] = np.minimum(closest[start:start + len(rows)], d)
        used = np.isin(eligible, np.asarray(chosen))
        closest[used] = -np.inf
        first = int(np.argmax(closest))
    return np.asarray(tensors), np.asarray(chosen)


def project_nonnegative_l1(a):
    """Euclidean projection onto a>=0, sum(a)<=1, independently per row."""
    nonnegative = np.maximum(a, 0)
    active = nonnegative.sum(1) > 1
    if not np.any(active):
        return nonnegative
    positive = nonnegative[active]
    ordered = np.sort(positive, axis=1)[:, ::-1]
    partial = np.cumsum(ordered, axis=1) - 1
    rho = np.sum(ordered - partial / np.arange(1, a.shape[1] + 1)[None] > 0, axis=1)
    threshold = partial[np.arange(len(positive)), rho - 1] / rho
    nonnegative[active] = np.maximum(positive - threshold[:, None], 0)
    return nonnegative


def reconstruction_objective(query, dictionary, *, nearest=False):
    """Fixed convex residual, not a sparsity or exact-mixture claim."""
    gram = np.einsum("id,jd->ij", dictionary, dictionary, optimize=False)
    cross = np.einsum("nd,kd->nk", query, dictionary, optimize=False)
    norm = np.sum(query * query, axis=1)
    if nearest:
        objective = np.min(norm[:, None] + np.diag(gram)[None] - 2 * cross, axis=1)
        return np.maximum(objective, 0), dict(projected_gradient_residual=None)
    hessian = gram + .05 * np.eye(len(dictionary))
    step = 1 / max(float(np.linalg.eigvalsh(hessian)[-1]), 1e-12)
    coefficient = np.zeros_like(cross)
    for _ in range(80):
        gradient = np.einsum("nk,kj->nj", coefficient, hessian, optimize=False) - cross
        coefficient = project_nonnegative_l1(coefficient - step * gradient)
    gradient = np.einsum("nk,kj->nj", coefficient, hessian, optimize=False) - cross
    residual = np.max(np.abs(project_nonnegative_l1(coefficient - step * gradient) - coefficient))
    objective = norm - 2 * np.sum(coefficient * cross, axis=1)
    objective += np.einsum("nk,kj,nj->n", coefficient, hessian, coefficient, optimize=False)
    return np.maximum(objective, 0), dict(projected_gradient_residual=float(residual),
                                         iterations=80, exact_optimum_claimed=False)


def patch_reconstruction(ep: Episode, arm="patch") -> Result:
    started = time.perf_counter()
    validate(ep)
    if ep.wf.sum() == 0 or ep.wb.sum() == 0:
        margin = np.full(len(ep.q), 1. if ep.wf.sum() else -1.)
        margin[ep.q_valid <= 0] = -1
        return Result(margin.reshape(ep.q_hw), dict(method_id="local_003", arm=arm,
            degenerate="empty_reference_class", wall_seconds=time.perf_counter() - started,
            natural_segmentation_gain="unmeasured"))
    r, q = unit(ep.r), unit(ep.q)
    ri, rw = tensor_index(ep.r_hw, ep.wvalid)
    qi, qw = tensor_index(ep.q_hw, ep.q_valid)
    fraction = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    valid = ep.wvalid > 0
    fg, bg = np.flatnonzero(valid & (fraction >= .9)), np.flatnonzero(valid & (fraction <= .1))
    if not len(fg):
        fg = np.flatnonzero(valid & (fraction == fraction[valid].max()))
    if not len(bg):
        bg = np.flatnonzero(valid & (fraction == fraction[valid].min()))
    # Equal actual dictionary sizes prevent an L2 benefit caused only by class
    # atom count; no additional query data or duplicate-column trick is used.
    count = min(16, len(fg), len(bg))
    fd, fa = tensor_dictionary(r, ri, rw, fg, count)
    bd, ba = tensor_dictionary(r, ri, rw, bg, count)
    if arm == "central_hull":
        fd, bd = unit(r[fa]), unit(r[ba])
    if arm == "permuted_order":
        order = np.random.default_rng(0).permutation(9)
        fd = fd.reshape(len(fd), 9, -1)[:, order].reshape(len(fd), -1)
        bd = bd.reshape(len(bd), 9, -1)[:, order].reshape(len(bd), -1)
    margin, errors, residual = np.zeros(len(q)), [], 0.
    for start in range(0, len(q), CHUNK):
        rows = np.arange(start, min(start + CHUNK, len(q)))
        tensors = patch_rows(q, qi, qw, rows, central=arm == "central_hull")
        a, ai = reconstruction_objective(tensors, fd, nearest=arm == "nearest_patch")
        b, bi = reconstruction_objective(tensors, bd, nearest=arm == "nearest_patch")
        margin[rows] = np.divide(b - a, a + b, out=np.zeros_like(a), where=a + b > 1e-12)
        for value in (ai["projected_gradient_residual"], bi["projected_gradient_residual"]):
            if value is not None:
                residual = max(residual, value)
        errors.extend(np.column_stack((a, b)).tolist())
    margin[ep.q_valid <= 0] = -1
    return Result(margin.reshape(ep.q_hw), dict(method_id="local_003", arm=arm,
        atoms_per_class=count, foreground_anchor_ids=fa.tolist(), background_anchor_ids=ba.tolist(),
        maximum_projected_gradient_residual=residual if arm != "nearest_patch" else None,
        class_average_objective=np.mean(np.asarray(errors), axis=0).tolist(),
        optimization="fixed 80-step projected gradient on nonnegative l1 ball; L2=.05; no sparse guarantee",
        query_ground_truth_used=False, query_seeds_used=False, new_encoder_forwards=0,
        natural_segmentation_gain="unmeasured", wall_seconds=time.perf_counter() - started))


METHODS = {"local_001": predict, "local_002": correspondence, "local_003": patch_reconstruction}
CONTROLS = {
    "local_001.diagonal_control": lambda ep: _predict(ep, "diagonal"),
    "local_001.trace_control": lambda ep: _predict(ep, "trace"),
    "local_001.label_permuted_control": lambda ep: _predict(ep, "label_permuted"),
    "local_002.unary_control": lambda ep: correspondence(ep, "unary"),
    "local_002.label_potts_control": lambda ep: correspondence(ep, "label_potts"),
    "local_002.permuted_coordinates_control": lambda ep: correspondence(ep, "permuted_coordinates"),
    "local_003.nearest_patch_control": lambda ep: patch_reconstruction(ep, "nearest_patch"),
    "local_003.central_hull_control": lambda ep: patch_reconstruction(ep, "central_hull"),
    "local_003.permuted_order_control": lambda ep: patch_reconstruction(ep, "permuted_order"),
}
