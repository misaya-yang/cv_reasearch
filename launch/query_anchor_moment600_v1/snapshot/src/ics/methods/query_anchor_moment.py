"""Query-internal foreground moment from unique reference-to-query anchors.

Reference tokens select anchors, but their feature distribution never enters the
moment contrast. The enrichment/representativity assumptions are conditional;
nearest-neighbor selection bias is not assumed away as an observed property.
"""
from __future__ import annotations

ARMS = ("native.cache.control", "query_anchor_moment", "query_anchor_isotropic.control",
        "query_anchor_moment.graph")
CONFIG = dict(
    reference_purity=0.9,
    reference_selection="cov >= .9; if empty, every token attaining max(cov), exactly the existing RCG rule",
    anchor_selection="unique argmax query-token index for each selected reference foreground token; ties use first index",
    moment="mean(unit_query[A]) - mean(unit_query[all]); neither mean separately normalized",
    direction="unit(Cq^-1 * moment); no query-label-based orientation flip",
    covariance="existing analytic centered Ledoit-Wolf query covariance; no tuned shrinkage",
    isotropic_control="unit(moment), with identical anchors, native binarizer and CUDA CRF",
    conditional_theory=(
        "Under representative within-query class-conditional anchor sampling, "
        "muA=rho*muFg+(1-rho)*muBg and muQ=pi*muFg+(1-pi)*muBg imply "
        "d=(rho-pi)*(muFg-muBg). Foreground enrichment rho>pi fixes positive orientation "
        "without knowing pi or equating cross-image foreground distributions."
    ),
    selection_bias=(
        "In general d=(rho-pi)*delta_query+epsilon, including class-conditional nearest-neighbor "
        "selection bias and finite-sample error. Enrichment and representative sampling are "
        "not proved or measured on DINO by this inference run."
    ),
    fallback="For numerically zero query moment use the existing unit(mean(unit(reference pure-FG tokens))) cue; if that mean is zero use its first selected token",
    fallback_scope="Only this declared degenerate case reintroduces a direct cross-image foreground cue; no rows dropped",
    graph="existing mean_graph.control, alpha=0, lambda=16, k=20, fidelity_floor=.1; no reranking",
    native_score_or_mask_in_method=False,
    source_background_mean_used=False,
    preprocessing_prior="retained native conditional Part1 output; cached q/r explicitly unit normalized",
    query_labels_in_inference=False,
    encoder_forwards=0,
)


def predict(q, r, cov):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .query_covariance import inverse_means
    from . import mean_graph

    q, r = F.normalize(q.float(), dim=1), F.normalize(r.float(), dim=1)
    foreground = np.flatnonzero(np.asarray(cov).ravel() >= CONFIG["reference_purity"])
    maximum_coverage_fallback = not len(foreground)
    if maximum_coverage_fallback:
        foreground = np.flatnonzero(np.asarray(cov).ravel() == np.asarray(cov).max())
    fg = torch.as_tensor(foreground, dtype=torch.long)
    similarities = q @ r[fg].T
    nearest = similarities.argmax(dim=0)
    del similarities
    anchors = torch.unique(nearest, sorted=True)
    mu_anchor = q[anchors].double().mean(dim=0)
    mu_query = q.double().mean(dim=0)
    moment = mu_anchor - mu_query
    norm = moment.norm()
    threshold = torch.finfo(torch.float64).eps * max(1.0, float(mu_anchor.norm()), float(mu_query.norm()))
    fallback = bool(norm <= threshold)
    if fallback:
        cue = r[fg].double().mean(dim=0)
        if bool(cue.norm() == 0):
            cue = r[fg[0]].double()
        direction = F.normalize(cue, p=2, dim=0)
        covariance_receipt = dict(shrinkage=None, numerical_floor_applied=False,
                                  covariance_not_used_in_degenerate_fallback=True)
        isotropic = direction
    else:
        # A zero second RHS merely reuses the established linear solver; it is
        # not a reference background prototype and contains no reference data.
        solution, covariance_receipt, _ = inverse_means(q, moment, torch.zeros_like(moment), "covariance")
        direction = F.normalize(solution[:, 2], p=2, dim=0)
        isotropic = moment / norm
    score = (q.double() @ direction).float().reshape(64, 64)
    score_isotropic = (q.double() @ isotropic).float().reshape(64, 64)
    score_graph, graph_receipt = mean_graph.control(q, r, cov, score.numpy(), device="cpu")
    graph_receipt = dict(graph_receipt, cue_identity="query-anchor covariance linear score; no FoRIS response",
                         finalizer="source native minmax/binarizer then identical native CUDA CRF")
    fields = {ARMS[1]: score.numpy(), ARMS[2]: score_isotropic.numpy(), ARMS[3]: score_graph,
              "source_fg_indices": foreground.astype(np.int64), "anchor_indices": anchors.numpy(),
              "reference_nn_indices": nearest.numpy(), "mu_anchor": mu_anchor.float().numpy(),
              "mu_query": mu_query.float().numpy(), "moment": moment.float().numpy(),
              "unit_w": direction.float().numpy(), "unit_w_isotropic": isotropic.float().numpy()}
    receipt = dict(reference_tokens=len(foreground), unique_query_anchors=len(anchors),
                   duplicate_nearest_indices_removed=len(foreground) - len(anchors),
                   maximum_coverage_reference_fallback=maximum_coverage_fallback,
                   moment_norm=float(norm), numerical_zero_threshold=threshold,
                   degenerate_moment_fallback=fallback,
                   fallback_cue=CONFIG["fallback"] if fallback else None,
                   orientation_to_query_moment=float(direction @ moment),
                   covariance=covariance_receipt, graph=graph_receipt,
                   query_gt_used=False, anchor_purity_or_prevalence_measured=False)
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise ValueError("Non-finite query-anchor field")
    return fields, receipt
