"""Three reviewed CPU correspondence mechanisms, with uncounted controls.

They use legal native DINO tokens and the complete reference mask only.  No
measured segmentation gain is claimed; fixed versions retain their failures.
"""
from __future__ import annotations

import numpy as np

from .common import Episode, Result, prototype_margin, unit, validate


def _dot(a, b):
    # Some macOS Accelerate builds expose stale FP flags; validate the product.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = a @ b
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite correspondence product")
    return out


def _compress(x, weight, maximum):
    """Deterministic FPS + five weighted spherical Lloyd steps.

    Partial valid cells affect mode locations; zero-weight padding never enters
    the dictionary.  Returned modes have no imposed population capacities.
    """
    selected = np.flatnonzero(np.asarray(weight) > 0)
    if not len(selected):
        return np.empty((0, x.shape[1])), np.full(len(x), -1, int), []
    y, w = unit(x[selected]), np.asarray(weight, float)[selected]
    center = unit(np.sum(y * w[:, None], axis=0))
    seed = int(np.argmax(_dot(y, center)))
    seeds = [seed]
    similarity = _dot(y, y[seed])
    for _ in range(1, min(maximum, len(y))):
        index = int(np.argmin(similarity))
        if similarity[index] >= 1 - 1e-12:
            break
        seeds.append(index)
        similarity = np.maximum(similarity, _dot(y, y[index]))
    modes = y[seeds].copy()
    for _ in range(5):
        labels = _dot(y, modes.T).argmax(1)
        sums = np.stack([np.sum(y[labels == k] * w[labels == k, None], axis=0)
                         for k in range(len(modes))])
        keep = np.linalg.norm(sums, axis=1) > 1e-12
        if not keep.any():
            modes = y[:1].copy()
            break
        modes = unit(sums[keep])
    labels = _dot(y, modes.T).argmax(1)
    used = np.unique(labels)
    modes = modes[used]
    labels = np.searchsorted(used, labels)
    full_labels = np.full(len(x), -1, int)
    full_labels[selected] = labels
    groups = [selected[labels == k] for k in range(len(modes))]
    return modes, full_labels, groups


def _fallback(ep, mechanism, reason):
    return Result(prototype_margin(ep), dict(mechanism=mechanism, fallback=reason,
                                            empirical_gain="unknown"))


def _roles(ep, maximum=64):
    coverage = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    fg, _, fg_groups = _compress(ep.r, ep.wvalid * (coverage >= .5), maximum)
    bg, _, bg_groups = _compress(ep.r, ep.wvalid * (coverage < .5), maximum)
    return fg, bg, fg_groups + bg_groups, coverage


def _top_mean(a, k=3):
    k = min(k, a.shape[1])
    return np.partition(a, a.shape[1] - k, axis=1)[:, -k:].mean(1)


def _csls(ep, correction):
    validate(ep)
    fg, bg, _, _ = _roles(ep)
    if not len(fg) or not len(bg):
        return _fallback(ep, "csls_hubness", "missing hard reference role")
    query, _, _ = _compress(ep.q, ep.q_valid, 256)
    if not len(query):
        return _fallback(ep, "csls_hubness", "no valid query token")
    reference = np.concatenate((fg, bg))
    cross = _dot(query, reference.T)
    k = min(10, len(query))
    hub = np.partition(cross, len(query) - k, axis=0)[-k:].mean(0)
    margin = np.empty(len(ep.q))
    for start in range(0, len(ep.q), 256):
        score = 2 * _dot(ep.q[start:start + 256], reference.T)
        if correction:
            score -= hub
        margin[start:start + 256] = (_top_mean(score[:, :len(fg)])
                                    - _top_mean(score[:, len(fg):]))
    margin[ep.q_valid <= 0] = 0
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism="csls_hubness" if correction else "same_dictionary_top3_cosine_control",
        reference_modes=[len(fg), len(bg)], query_modes=len(query),
        hub_density_min=float(hub.min()), hub_density_max=float(hub.max()),
        query_side_density="row constant cancels exactly in FG-minus-BG support",
        valid_partial_weights="weighted mode locations; equal Q-mode density weights",
        empirical_gain="unknown", capacities="none"))


def csls_hubness(ep):
    return _csls(ep, True)


def cosine_dictionary_control(ep):
    return _csls(ep, False)


def _shift_margin(ep, shift):
    moved = unit(ep.r + shift)
    fg = unit(np.sum(moved * ep.wf[:, None], axis=0))
    bg = unit(np.sum(moved * ep.wb[:, None], axis=0))
    out = _dot(ep.q, fg - bg)
    out[ep.q_valid <= 0] = 0
    return out.reshape(ep.q_hw)


def background_anchor_shift(ep):
    validate(ep)
    name = "background_anchor_shift"
    if ep.wf.sum() <= 0 or ep.wb.sum() <= 0:
        return _fallback(ep, name, "empty reference role")
    coverage = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    background, _, _ = _compress(ep.r, ep.wvalid * (coverage <= .1), 64)
    query, _, _ = _compress(ep.q, ep.q_valid, 256)
    if min(len(background), len(query)) < 2:
        return _fallback(ep, name, "insufficient background or query modes")
    sim = _dot(background, query.T)
    b_order = np.argsort(-sim, axis=1, kind="stable")
    q_order = np.argsort(-sim, axis=0, kind="stable")
    matches = []
    for i, order in enumerate(b_order):
        j = int(order[0])
        if (q_order[0, j] == i and sim[i, j] - sim[i, order[1]] > .05
                and sim[i, j] - sim[q_order[1, j], j] > .05):
            matches.append((i, j))
    info = dict(mechanism=name, matches=len(matches), empirical_gain="unknown",
                valid_partial_weights="weighted mode locations; equal residual vote per mode")
    if len(matches) < 3:
        result = _fallback(ep, name, "fewer than three mutual margin anchors")
        result.info.update(info)
        return result
    delta = np.stack([query[j] - background[i] for i, j in matches])
    distance = np.linalg.norm(delta[:, None] - delta[None, :], axis=2)
    index = int(np.argmin(distance.sum(1)))
    shift = delta[index]
    consensus = float(np.mean(distance[index] <= .1))
    info.update(shift_norm=float(np.linalg.norm(shift)), residual_consensus=consensus)
    if consensus < 2 / 3 or info["shift_norm"] > .5:
        result = _fallback(ep, name, "inconsistent or excessive anchor translation")
        result.info.update(info)
        return result
    return Result(_shift_margin(ep, shift), dict(info, fallback=False))


def whole_mean_shift_control(ep):
    validate(ep)
    if ep.wf.sum() <= 0 or ep.wb.sum() <= 0 or ep.q_valid.sum() <= 0:
        return _fallback(ep, "whole_mean_shift_control", "empty reference/query role")
    mean_r = np.average(ep.r, axis=0, weights=ep.wvalid)
    mean_q = np.average(ep.q, axis=0, weights=ep.q_valid)
    return Result(_shift_margin(ep, mean_q - mean_r), dict(
        mechanism="whole_mean_shift_control", empirical_gain="unknown",
        not_counted=True, assumption="whole-image shift mixes semantic composition"))


def _softmax(a):
    a = a - a.max(1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(1, keepdims=True)


def gram_objective(p, cost, dq, dr):
    """Fixed finite soft-association objective; not a global optimality claim."""
    n = len(p)
    mass = p.sum(0)
    relational = (np.sum(dq * dq) + mass @ ((dr * dr) @ mass)
                  - 2 * np.sum((_dot(dq, _dot(p, dr))) * p)) / n**2
    entropy = np.sum(p * np.log(np.maximum(p, 1e-300))) / n
    return float(.2 * np.sum(p * cost) / n + relational + .02 * entropy)


def gram_gradient(p, cost, dq, dr):
    """Gradient representative on the row-simplex tangent space.

    ``gram_objective`` substitutes row sums of one into its DQ-squared term.
    This representative retains the row-constant derivative of the original
    four-index loss.  It is equivalent on zero-row-sum directions and in the
    normalized mirror update, but is not that substituted function's ambient
    gradient.  The audit therefore uses tangent directions explicitly.
    """
    n = len(p)
    mass = p.sum(0)
    relational = (np.sum(dq * dq, axis=1)[:, None]
                  + ((dr * dr) @ mass)[None, :]
                  - 2 * _dot(dq, _dot(p, dr))) * (2 / n**2)
    return .2 * cost / n + relational + .02 * (np.log(np.maximum(p, 1e-300)) + 1) / n


def _gram(ep, mode):
    validate(ep)
    fg, bg, groups, coverage = _roles(ep, maximum=32)
    query, labels, _ = _compress(ep.q, ep.q_valid, 64)
    if not len(fg) or not len(bg) or not len(query):
        return _fallback(ep, f"gram_{mode}", "empty valid role or query")
    reference = np.concatenate((fg, bg))
    labels_r = np.array([np.average(coverage[g], weights=ep.wvalid[g]) for g in groups])
    cost = np.clip(1 - _dot(query, reference.T), 0, 2)
    dq = np.clip(1 - _dot(query, query.T), 0, 2)
    dr = np.clip(1 - _dot(reference, reference.T), 0, 2)
    p = _softmax(-cost / .1)
    objective = [gram_objective(p, cost, dq, dr)]
    if mode == "relation":
        for _ in range(40):
            p = _softmax(np.log(np.maximum(p, 1e-300))
                         - .25 * len(p) * gram_gradient(p, cost, dq, dr))
            objective.append(gram_objective(p, cost, dq, dr))
    elif mode == "profile":
        quantiles = np.linspace(0, 1, 17)
        qprofile = np.quantile(dq, quantiles, axis=1).T
        rprofile = np.quantile(dr, quantiles, axis=1).T
        discrepancy = np.mean((qprofile[:, None] - rprofile[None, :])**2, axis=2)
        p = _softmax(-discrepancy / .02)
    elif mode != "appearance":
        raise ValueError(mode)
    score = 2 * _dot(p, labels_r) - 1
    margin = np.zeros(len(ep.q))
    active = labels >= 0
    margin[active] = score[labels[active]]
    return Result(margin.reshape(ep.q_hw), dict(
        mechanism=f"free_column_gram_{mode}", reference_modes=[len(fg), len(bg)],
        query_modes=len(query), steps=40 if mode == "relation" else 0,
        objective_trace=objective, maximum_row_sum_error=float(np.max(np.abs(p.sum(1) - 1))),
        maximum_column_mass=float(p.sum(0).max()), column_capacities="none",
        valid_partial_weights="weighted mode locations; uniform mode relation weights",
        optimizer="fixed mirror descent, no global optimum claimed", empirical_gain="unknown"))


def free_column_gram_matching(ep):
    return _gram(ep, "relation")


def appearance_assignment_control(ep):
    return _gram(ep, "appearance")


def sorted_gram_profile_control(ep):
    return _gram(ep, "profile")


METHODS = {
    "cross_image_csls_hubness": csls_hubness,
    "cross_image_background_anchor_shift": background_anchor_shift,
    "cross_image_free_column_gram_matching": free_column_gram_matching,
}

CONTROLS = {
    "cross_image_cosine_dictionary_control": cosine_dictionary_control,
    "cross_image_whole_mean_shift_control": whole_mean_shift_control,
    "cross_image_appearance_assignment_control": appearance_assignment_control,
    "cross_image_sorted_gram_profile_control": sorted_gram_profile_control,
}
