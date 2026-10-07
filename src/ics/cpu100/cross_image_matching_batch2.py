"""Four further reviewed, fixed CPU cross-image decisions and strong controls."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

from .common import Result, prototype_margin, unit, validate
from .cross_image_matching import _compress, _dot, _fallback, _roles


def _dictionary(ep, maximum=32):
    fg, bg, groups, coverage = _roles(ep, maximum)
    if not len(fg) or not len(bg):
        return None
    atoms = np.concatenate((fg, bg))
    roles = np.array([np.average(coverage[g], weights=ep.wvalid[g]) for g in groups])
    return atoms, roles, len(fg)


def active_nnls(gram, response, norm2=1.):
    """Exact finite active-set enumeration for at most three chosen atoms."""
    k = len(response)
    if k > 3:
        raise ValueError("Only the reviewed <=3 active-set solver is allowed")
    best = np.zeros(k)
    cost = float(norm2)
    for bitset in range(1, 1 << k):
        ids = np.array([j for j in range(k) if bitset & (1 << j)])
        candidate = np.linalg.lstsq(gram[np.ix_(ids, ids)], response[ids], rcond=1e-10)[0]
        if np.any(candidate < -1e-10):
            continue
        candidate = np.maximum(candidate, 0)
        error = float(norm2 - 2 * candidate @ response[ids]
                      + candidate @ gram[np.ix_(ids, ids)] @ candidate)
        if error < cost - 1e-12:
            best = np.zeros(k); best[ids] = candidate
            cost = error
    return best, max(0., cost)


def omp3(response, gram, norm2):
    coefficients = np.zeros_like(response)
    costs = np.asarray(norm2).copy()
    for row in range(len(response)):
        chosen = []
        for _ in range(min(3, gram.shape[0])):
            residual_dot = response[row] - gram @ coefficients[row]
            residual_dot[chosen] = -np.inf
            atom = int(np.argmax(residual_dot))
            if residual_dot[atom] <= 1e-12:
                break
            chosen.append(atom)
            values, cost = active_nnls(gram[np.ix_(chosen, chosen)], response[row, chosen], norm2[row])
            coefficients[row] = 0; coefficients[row, chosen] = values
            costs[row] = cost
    return coefficients, costs


def _sparse(ep, kind):
    validate(ep)
    bank = _dictionary(ep)
    if bank is None:
        return _fallback(ep, f"sparse_{kind}", "missing reference role")
    atoms, roles, nf = bank
    gram, response = _dot(atoms, atoms.T), _dot(ep.q, atoms.T)
    norm2 = np.sum(ep.q * ep.q, 1)
    coefficients, errors = omp3(response, gram, norm2)
    diagnostics = {}
    if kind == "joint":
        f, b = coefficients * roles, coefficients * (1 - roles)
        ef = norm2 - 2 * np.sum(b * response, 1) + np.sum(_dot(b, gram) * b, 1)
        eb = norm2 - 2 * np.sum(f * response, 1) + np.sum(_dot(f, gram) * f, 1)
        margin = ef - eb
    elif kind == "vote":
        margin = _dot(coefficients, 2 * roles - 1)
    elif kind == "independent":
        _, efg = omp3(response[:, :nf], gram[:nf, :nf], norm2)
        _, ebg = omp3(response[:, nf:], gram[nf:, nf:], norm2)
        margin = ebg - efg
    elif kind == "dense_cone":
        class_errors = []
        retained_ranks = []
        for sl in (slice(None, nf), slice(nf, None)):
            g = gram[sl, sl]; rhs = response[:, sl]
            values, vectors = np.linalg.eigh(g)
            active = values > max(float(values.max()), 1e-300) * 1e-10
            retained_ranks.append(int(active.sum()))
            a = np.sqrt(values[active])[:, None] * vectors[:, active].T
            b = (rhs @ vectors[:, active]) / np.sqrt(values[active])
            error = np.empty(len(ep.q))
            for i in range(len(ep.q)):
                alpha = nnls(a, b[i], maxiter=max(100, 10 * g.shape[0]))[0]
                error[i] = max(0., norm2[i] - 2 * alpha @ rhs[i] + alpha @ g @ alpha)
            class_errors.append(error)
        margin = class_errors[1] - class_errors[0]
        diagnostics = dict(cone_eigen_relative_tolerance=1e-10, cone_retained_ranks=retained_ranks)
    elif kind == "hull":
        from ics.methods.reference_hull import hull_distance
        fg = hull_distance(ep.q, atoms[:nf]); bg = hull_distance(ep.q, atoms[nf:])
        margin = bg["cost"] - fg["cost"]
        diagnostics = dict(hull_maximum_fw_gap=max(fg["info"]["maximum_gap"], bg["info"]["maximum_gap"]),
                           hull_unconverged_class_rows=fg["info"]["unconverged_rows"]+bg["info"]["unconverged_rows"],
                           hull_costs="upper feasible costs; stored FW gap bounds distance error")
    else:
        raise ValueError(kind)
    margin[ep.q_valid <= 0] = 0
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism=f"joint_sparse_role_{kind}", reference_modes=[nf, len(atoms)-nf],
        maximum_sparse_atoms=3, joint_mean_residual=float(errors.mean()),
        mean_nonzero_coefficients=float(np.count_nonzero(coefficients > 1e-10, axis=1).mean()),
        empirical_gain="unknown", greedy_not_global_l0=True, **diagnostics))


def joint_sparse_role_removal(ep): return _sparse(ep, "joint")
def independent_sparse_control(ep): return _sparse(ep, "independent")
def sparse_coefficient_vote_control(ep): return _sparse(ep, "vote")
def dense_class_cone_control(ep): return _sparse(ep, "dense_cone")
def convex_hull_control(ep): return _sparse(ep, "hull")


def facility_greedy(cost, weights, fee=.05):
    """Reviewed finite greedy cover; no claim of globally optimal facility set."""
    unary = _dot(weights, cost)
    opened = [int(np.argmin(unary))]
    current = cost[:, opened[0]].copy()
    trace = [float(weights @ current + fee)]
    while len(opened) < cost.shape[1]:
        gain = weights @ np.maximum(current[:, None] - cost, 0)
        gain[opened] = -np.inf
        index = int(np.argmax(gain))
        if gain[index] <= fee + 1e-12:
            break
        opened.append(index)
        current = np.minimum(current, cost[:, index])
        trace.append(float(weights @ current + fee * len(opened)))
    return sorted(opened), trace


def _facility(ep, kind):
    validate(ep)
    bank = _dictionary(ep)
    if bank is None or ep.q_valid.sum() <= 0:
        return _fallback(ep, f"facility_{kind}", "empty role or query")
    atoms, roles, nf = bank
    cost = np.clip(1 - _dot(ep.q, atoms.T), 0, 2)
    weights = ep.q_valid / ep.q_valid.sum()
    opened, trace = facility_greedy(cost, weights)
    if kind == "all": opened = list(range(len(atoms)))
    elif kind == "unary_count":
        opened = sorted(np.argsort(weights @ cost, kind="stable")[:len(opened)].tolist())
    elif kind == "random_count":
        opened = sorted(np.random.default_rng(0).choice(len(atoms), len(opened), replace=False).tolist())
    elif kind != "greedy": raise ValueError(kind)
    assignment = np.asarray(opened)[cost[:, opened].argmin(1)]
    margin = 2 * roles[assignment] - 1
    margin[ep.q_valid <= 0] = 0
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism=f"exemplar_facility_{kind}", opened_atoms=opened,
        total_atoms=len(atoms), source_role_modes=[nf, len(atoms)-nf],
        activation_fee=.05, greedy_objective_trace=trace,
        prior="shared-explanation complexity prior; depends on query area fractions",
        capacities="no upper column capacity or source-marginal requirement", empirical_gain="unknown"))


def exemplar_facility_cover(ep): return _facility(ep, "greedy")
def all_exemplars_control(ep): return _facility(ep, "all")
def facility_unary_same_count_control(ep): return _facility(ep, "unary_count")
def facility_random_same_count_control(ep): return _facility(ep, "random_count")


def cross_affinity(query, reference):
    sim = _dot(query, reference.T)
    keep = np.zeros_like(sim, bool)
    row = np.argsort(-sim, axis=1, kind="stable")[:, :min(4, sim.shape[1])]
    keep[np.arange(len(query))[:, None], row] = True
    col = np.argsort(-sim, axis=0, kind="stable")[:min(4, sim.shape[0])]
    keep[col, np.arange(len(reference))[None, :]] = True
    return np.exp((sim - sim.max(1, keepdims=True)) / .1) * keep


def path_endpoints(weight, exclude_query=True, exclude_reference=True):
    """Weighted length-three paths, normalized only at final role readout."""
    g = _dot(weight.T, weight)
    endpoints = _dot(weight, g)
    if exclude_query:
        endpoints -= np.sum(weight * weight, axis=1)[:, None] * weight
    if exclude_reference:
        endpoints -= weight * np.diag(g)
    if exclude_query and exclude_reference:
        endpoints += weight**3
    if endpoints.min(initial=0) < -1e-9:
        raise FloatingPointError("Negative path mass beyond roundoff")
    return np.maximum(endpoints, 0)


def _paths(ep, kind):
    validate(ep)
    bank = _dictionary(ep, 16)
    query, assignment, _ = _compress(ep.q, ep.q_valid, 64)
    if bank is None or not len(query):
        return _fallback(ep, f"paths_{kind}", "empty role or query")
    atoms, roles, nf = bank
    w = cross_affinity(query, atoms)
    onehop = _dot(w, 2 * roles - 1) / w.sum(1)
    if kind == "onehop":
        mode_margin = onehop; pathless = np.zeros(len(query), bool)
    else:
        end = path_endpoints(w, kind in {"nonreturn", "no_query_return"},
                             kind in {"nonreturn", "no_reference_return"})
        mass = end.sum(1); pathless = mass <= 1e-14
        mode_margin = np.divide(_dot(end, 2 * roles - 1), mass,
                                out=onehop.copy(), where=~pathless)
    margin = np.zeros(len(ep.q)); active = assignment >= 0
    margin[active] = mode_margin[assignment[active]]
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism=f"nonreturn_path_{kind}", pathless_modes=int(pathless.sum()),
        reference_modes=[nf, len(atoms)-nf], query_modes=len(query),
        path_dependence="paths are not independent statistical samples",
        normalization="weighted paths, not a step-normalized Markov probability", empirical_gain="unknown"))


def nonreturn_path_consensus(ep): return _paths(ep, "nonreturn")
def ordinary_three_hop_control(ep): return _paths(ep, "ordinary")
def one_hop_path_control(ep): return _paths(ep, "onehop")
def no_query_return_control(ep): return _paths(ep, "no_query_return")
def no_reference_return_control(ep): return _paths(ep, "no_reference_return")


def opponent_dictionary(ep):
    fg, bg, _, _ = _roles(ep, 32)
    k = min(len(fg), len(bg))
    if not k: return None
    fg, _, _ = _compress(fg, np.ones(len(fg)), k)
    bg, _, _ = _compress(bg, np.ones(len(bg)), k)
    k = min(len(fg), len(bg))
    return fg[:k], bg[:k]


def _opponents(ep, kind):
    validate(ep)
    bank = opponent_dictionary(ep)
    if bank is None:
        return _fallback(ep, f"opponents_{kind}", "empty reference role")
    fg, bg = bank
    cost = np.clip(1 - _dot(fg, bg.T), 0, 2)
    if kind in {"matching", "raw_mean"}:
        left, right = linear_sum_assignment(cost)
        direction = fg[left] - bg[right]
    elif kind == "nearest":
        right = cost.argmin(1); direction = fg - bg[right]
    elif kind == "allpairs":
        direction = (fg[:, None] - bg[None, :]).reshape(-1, fg.shape[1])
    elif kind == "random":
        right = np.random.default_rng(0).permutation(len(bg)); direction = fg - bg[right]
    else: raise ValueError(kind)
    norm = np.linalg.norm(direction, axis=1)
    if kind == "raw_mean":
        margin = _dot(ep.q, direction.mean(0)); kept = len(direction)
    else:
        direction = direction[norm > 1e-10]
        if not len(direction):
            return _fallback(ep, f"opponents_{kind}", "all opponent pairs coincide")
        direction = unit(direction); kept = len(direction)
        margin = np.median(_dot(ep.q, direction.T), axis=1)
    margin[ep.q_valid <= 0] = 0
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism=f"source_opponents_{kind}", source_mode_k=len(fg), used_directions=kept,
        capacity_scope="reference opponent construction only; query columns unrestricted",
        optimal_source_matching_cost=float(cost[linear_sum_assignment(cost)].sum()),
        raw_mean_identity="mean matched(F-B) is independent of permutation", empirical_gain="unknown"))


def source_opponent_matching(ep): return _opponents(ep, "matching")
def nearest_opponent_median_control(ep): return _opponents(ep, "nearest")
def allpair_opponent_median_control(ep): return _opponents(ep, "allpairs")
def random_opponent_median_control(ep): return _opponents(ep, "random")
def raw_matched_mean_control(ep): return _opponents(ep, "raw_mean")


METHODS = {
    "cross_image_joint_sparse_role_removal": joint_sparse_role_removal,
    "cross_image_exemplar_facility_cover": exemplar_facility_cover,
    "cross_image_nonreturn_path_consensus": nonreturn_path_consensus,
    "cross_image_source_opponent_matching": source_opponent_matching,
}

CONTROLS = {
    "cross_image_independent_sparse_control": independent_sparse_control,
    "cross_image_sparse_coefficient_vote_control": sparse_coefficient_vote_control,
    "cross_image_dense_class_cone_control": dense_class_cone_control,
    "cross_image_convex_hull_control": convex_hull_control,
    "cross_image_all_exemplars_control": all_exemplars_control,
    "cross_image_facility_unary_same_count_control": facility_unary_same_count_control,
    "cross_image_facility_random_same_count_control": facility_random_same_count_control,
    "cross_image_ordinary_three_hop_control": ordinary_three_hop_control,
    "cross_image_one_hop_path_control": one_hop_path_control,
    "cross_image_no_query_return_control": no_query_return_control,
    "cross_image_no_reference_return_control": no_reference_return_control,
    "cross_image_nearest_opponent_median_control": nearest_opponent_median_control,
    "cross_image_allpair_opponent_median_control": allpair_opponent_median_control,
    "cross_image_random_opponent_median_control": random_opponent_median_control,
    "cross_image_raw_matched_mean_control": raw_matched_mean_control,
}
