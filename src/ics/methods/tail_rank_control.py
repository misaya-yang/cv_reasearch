"""Fixed positive-tail CDF rearrangement of the actual parent MEAN unary.

This is a classical head-only rank control. It adds no reference information,
encoder, learned parameter, new graph, or established segmentation gain.
"""
from __future__ import annotations

import numpy as np

from .rcg import rank


def build_unaries(s, guide, parent_unary):
    """Change only token G=(s>.5); preserve the original unary outside G."""
    s, guide, parent = (np.asarray(v).reshape(-1) for v in (s, guide, parent_unary))
    if s.dtype != np.float32 or guide.dtype != np.float32 or parent.dtype != np.float64:
        raise ValueError('Use parent FP32 s/guide and FP64 unary without reprocessing')
    if s.shape != guide.shape or s.shape != parent.shape or not len(s):
        raise ValueError('Parent input shapes differ or are empty')
    if not all(np.isfinite(v).all() for v in (s, guide, parent)):
        raise ValueError('Nonfinite parent input')
    group = s > .5
    n = int(group.sum())
    candidate, no_reference, amplitude = parent.copy(), parent.copy(), parent.copy()
    rank_s = rank(s[group]) if n else np.empty(0, np.float32)
    rank_guide = rank(guide[group]) if n else np.empty(0, np.float32)
    if n >= 2:
        candidate[group] = (s[group] + .25 * (rank_guide - rank_s)).astype(np.float64)
        no_reference[group] = s[group].astype(np.float64)
    base = s[group].astype(np.float64)
    delta_global = parent[group] - base
    delta_local = candidate[group] - base
    global_budget = float(np.abs(delta_global).sum(dtype=np.float64))
    local_budget = float(np.abs(delta_local).sum(dtype=np.float64))
    centered = delta_global - delta_global.mean(dtype=np.float64) if n else delta_global.copy()
    centered_budget = float(np.abs(centered).sum(dtype=np.float64))
    if n < 2:
        # The frozen small-group fallback takes precedence over the centered
        # budget formula (one centered element is always zero).
        kappa = None
    elif centered_budget == 0:
        if local_budget != 0:
            raise ValueError('Zero centered global budget requires zero local budget')
        kappa = 0.
        if n >= 2:
            amplitude[group] = base
    else:
        kappa = local_budget / centered_budget
        amplitude[group] = base + kappa * centered
    amplitude_budget = float(np.abs(amplitude[group] - base).sum(dtype=np.float64))
    budget_error = abs(amplitude_budget - local_budget)
    if budget_error > max(1e-12, 1e-12 * local_budget):
        raise RuntimeError('Actual FP64 amplitude-control L1 budget does not close')
    if any(not np.array_equal(v[~group], parent[~group]) for v in (candidate, no_reference, amplitude)):
        raise RuntimeError('The outside-G parent unary must remain bit exact')
    return dict(group=group, rank_s_group=rank_s, rank_guide_group=rank_guide,
                candidate_unary=candidate, no_reference_unary=no_reference,
                amplitude_unary=amplitude,
                diagnostics=dict(group='token-level original s>.5', n_group=n,
                                 fallback='nG_less_than2_reuse_parent_unary' if n < 2 else None,
                                 outside_group_parent_bit_exact=True,
                                 amplitude_kappa=kappa, global_group_l1=global_budget,
                                 centered_global_group_l1=centered_budget,
                                 candidate_group_l1=local_budget, amplitude_group_l1=amplitude_budget,
                                 global_signed_correction_sum=float(delta_global.sum(dtype=np.float64)),
                                 candidate_signed_correction_sum=float(delta_local.sum(dtype=np.float64)),
                                 amplitude_signed_correction_sum=float((amplitude[group] - base).sum(dtype=np.float64)),
                                 amplitude_global_correction_centered=True,
                                 amplitude_l1_absolute_error=budget_error,
                                 amplitude_budget_from_actual_FP32_to_FP64_candidate=True,
                                 amplitude_clipped=False, amplitude_parameter_search=False,
                                 alpha=.25, query_GT_in_inference=False,
                                 group_is_postCRF_diagnostic_ROI=False,
                                 constant_guide_is_graph_only_claim=False))


def solve_parent(unary, a, h):
    """Same actual parent H/A and CG settings; there is no graph rebuilding."""
    from scipy.sparse.linalg import cg
    unary, a = np.asarray(unary).reshape(-1), np.asarray(a).reshape(-1)
    if unary.dtype != np.float64 or a.dtype != np.float64 or a.shape != unary.shape:
        raise ValueError('Actual parent unary/A must be equally shaped FP64 vectors')
    if h.shape != (len(unary), len(unary)) or not np.isfinite(unary).all() or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Invalid parent system')
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    z, status = cg(h, a * unary, x0=unary, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    if status:
        raise RuntimeError('Tail-rank parent CG did not converge: ' + str(status))
    return z.astype(np.float32), dict(cg_iterations=iterations[0], cg_rtol=1e-7,
        cg_atol=1e-9, cg_maxiter=300, lambda_value=16., graph='saved actual parent H/A',
        cg_relative_residual=float(np.linalg.norm(h @ z - a * unary) / max(np.linalg.norm(a * unary), 1e-12)),
        encoder_calls=0, query_GT_in_inference=False)
