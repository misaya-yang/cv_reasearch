"""Closed-form query covariance correction with a matched isotropic control.

Only the annotated reference supplies class means. All query tokens supply a
centered Ledoit-Wolf covariance; query labels, fitted coefficients and an encoder
are absent. The native feature geometry, foreground anchor and complete tail
remain in use. Native CUDA CRF is a separate required finalization phase.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

ARMS = ("native.cache.control", "query_covariance", "query_isotropic.control")
CONFIG = dict(
    covariance="Ledoit-Wolf analytic shrinkage of centered query covariance to trace(C)/d * I",
    covariance_divisor="n_query_tokens",
    covariance_precision="float64",
    shrinkage_parameters="none; analytic query-only estimate",
    reference_means="all native-downsampled foreground/background tokens; not separately normalized",
    direction="unit(C^-1 * (muFG - muBG)) with positive foreground orientation",
    evidence="aFG=q^T*C^-1*muFG/||C^-1*delta||; aBG=q^T*C^-1*muBG/||C^-1*delta||; s2=aFG-aBG",
    isotropic_control="C=I with identical means, evidence construction and complete native tail",
    foreground_anchor="unchanged native normalized foreground mean in raw feature coordinates",
    covariance_interpretation=(
        "Conditional: homoscedastic Gaussian query classes and reference delta parallel to query delta "
        "make total-covariance inverse and within-class inverse discriminant directions parallel by "
        "Sherman-Morrison. Neither assumption is observed or verified by this experiment."
    ),
    optimization="For positive definite shrunk C, the direction maximizes (w^T delta)^2/(w^T C w)",
    numerical_floor="Only if Cholesky detects degenerate covariance; d * float64_eps * max(1, eigenvalue scale)",
    fallback="native Part2 evidence for empty reference background or exactly zero mean contrast",
    query_labels_in_inference=False,
    encoder_forwards=0,
)


def native_minmax(value):
    value = value - value.min()
    return value / value.max().clamp_min(1e-6)


def ledoit_wolf(query):
    """Analytic LW estimate without an n x n Gram matrix or a tuned lambda.

    With S=X^T X/n, delta=||S-mu I||_F^2/d and
    beta=(mean_i ||x_i||^4-||S||_F^2)/(d*n), shrinkage is
    clip(beta,0,delta)/delta. This is the standard centered LW estimator.
    """
    x = query.to(torch.float64)
    x = x - x.mean(dim=0)
    n, d = x.shape
    sample = (x.T @ x) / n
    mu = sample.diagonal().mean()
    norm_squared = sample.square().sum()
    delta = (norm_squared / d - mu.square()).clamp_min(0)
    beta = ((x.square().sum(dim=1).square().mean() - norm_squared) / (d * n)).clamp_min(0)
    beta = torch.minimum(beta, delta)
    shrinkage = beta / delta if bool(delta > 0) else torch.zeros_like(delta)
    covariance = (1 - shrinkage) * sample
    covariance.diagonal().add_(shrinkage * mu)
    receipt = dict(query_tokens=n, dimensions=d, query_tokens_used="all", centered=True,
                   shrinkage=float(shrinkage), shrinkage_beta=float(beta), shrinkage_delta=float(delta),
                   target_variance=float(mu), sample_trace=float(sample.diagonal().sum()),
                   covariance_trace=float(covariance.diagonal().sum()), numerical_floor_applied=False)
    return covariance, receipt


def inverse_means(query, mu_fg, mu_bg, mode):
    """Solve all three right-hand sides together; never invert the matrix."""
    fg, bg = mu_fg.to(torch.float64), mu_bg.to(torch.float64)
    means = torch.stack((fg, bg, fg - bg), dim=1)
    if mode == "isotropic":
        return means, dict(covariance="identity", shrinkage=None, numerical_floor_applied=False), None
    if mode != "covariance":
        raise ValueError(f"Unknown readout: {mode}")
    covariance, receipt = ledoit_wolf(query)
    factor, info = torch.linalg.cholesky_ex(covariance)
    if int(info):
        # A numerical floor is a declared degeneracy repair, never a tuning knob.
        values, vectors = torch.linalg.eigh(covariance)
        scale = max(1.0, float(values.abs().max()))
        floor = covariance.shape[0] * torch.finfo(torch.float64).eps * scale
        receipt.update(numerical_floor_applied=True, numerical_floor=floor,
                       minimum_eigenvalue_before_floor=float(values.min()),
                       floored_eigenvalues=int((values < floor).sum()), cholesky_failure_index=int(info))
        covariance = (vectors * values.clamp_min(floor)[None]) @ vectors.T
        covariance = (covariance + covariance.T) / 2
        factor = torch.linalg.cholesky(covariance)
    solution = torch.cholesky_solve(means, factor)
    receipt["solve_relative_residual"] = float((covariance @ solution - means).norm() / means.norm())
    return solution, receipt, covariance


def reference_means(fmaps, masks, downsample_mask):
    """Use the exact reference-mask downsampler and source token coordinates."""
    if fmaps.shape[1] != 2 or masks.shape[0] != 1:
        raise ValueError("This method requires exactly one annotated reference and one query")
    h, w = fmaps.shape[-2:]
    mask = downsample_mask(masks[0:1], h, w)
    reference = fmaps[0, 0]
    fg, bg = reference[:, mask], reference[:, ~mask]
    if not fg.shape[1]:
        raise ValueError("No foreground tokens; native Part2 has no valid reference")
    means = (fg.mean(dim=1), bg.mean(dim=1) if bg.shape[1] else None)
    return means, dict(foreground_tokens=fg.shape[1], background_tokens=bg.shape[1],
                       mask_downsampling="source utils.data.downsample_mask")


def replacement_evidence(query, mu_fg, mu_bg, mode):
    """Return coherent native-shaped (score,sf,sbn) in raw coordinates."""
    inverse, receipt, covariance = inverse_means(query, mu_fg, mu_bg, mode)
    norm = inverse[:, 2].norm()
    if not bool(norm > 0) or not torch.isfinite(inverse).all():
        raise ValueError("Non-finite or zero covariance-corrected reference contrast")
    unit_w = inverse[:, 2] / norm
    delta = mu_fg.to(torch.float64) - mu_bg.to(torch.float64)
    orientation = unit_w @ delta
    if not bool(orientation > 0):
        raise ValueError("Corrected direction lost positive foreground orientation")
    responses = (query.to(torch.float64) @ inverse[:, :2] / norm).to(torch.float32)
    a_fg, a_bg = responses[:, 0].reshape(64, 64), responses[:, 1].reshape(64, 64)
    score = a_fg - a_bg
    direct = (query.to(torch.float64) @ unit_w).to(torch.float32).reshape(64, 64)
    denominator = (unit_w @ covariance @ unit_w) if covariance is not None else unit_w.square().sum()
    receipt.update(mode=mode, contrast_norm=float((mu_fg - mu_bg).norm()),
                   inverse_contrast_norm=float(norm), orientation=float(orientation),
                   fisher_ratio=float(orientation.square() / denominator),
                   maximum_fisher_ratio=float(delta @ inverse[:, 2]),
                   evidence_score_closure_max_abs=float((score - direct).abs().max()),
                   native_bg_weight_used_in_replacement=False, native_fallback=False)
    fields = dict(a_fg=a_fg, a_bg=a_bg, unit_w=unit_w.float(), raw_mu_fg=mu_fg, raw_mu_bg=mu_bg)
    return (score, native_minmax(a_fg), native_minmax(a_bg)), fields, receipt


@torch.no_grad()
def prefinal_arms(host, fmaps, native_module):
    """One native Part2 and three direct source Part3/Part4/binarizer tails."""
    masks = host._ref_masks.unsqueeze(1)
    n_refs, h, w = host._ref_images.shape[0], fmaps.shape[-2], fmaps.shape[-1]
    start = time.monotonic()
    native = host._part2_background_suppression(fmaps_norm=fmaps, ref_masks=masks,
                                               n_refs=n_refs, h=h, w=w)
    part2_seconds = time.monotonic() - start
    if native is None:
        raise ValueError("No native Part2 output for the reference")
    native_s2, native_sf, native_sbn, anchor, denoised = native
    (mu_fg, mu_bg), mean_receipt = reference_means(fmaps, masks, native_module.downsample_mask)
    if not torch.equal(anchor, F.normalize(mu_fg, p=2, dim=0)):
        raise RuntimeError("Original native foreground anchor differs from normalized raw foreground mean")
    query = fmaps[0, n_refs].flatten(1).T
    fallback = "empty_reference_background" if mu_bg is None else (
        "zero_reference_mean_contrast" if bool((mu_fg - mu_bg).norm() == 0) else None)
    prefinal, fields, receipts, timings = {}, {}, {}, {}
    for arm, mode in zip(ARMS, ("native", "covariance", "isotropic")):
        started = time.monotonic()
        extra = {}
        if mode == "native" or fallback is not None:
            s2, sf, sbn = native_s2, native_sf, native_sbn
            receipt = dict(mode=mode, native_fallback=mode != "native", fallback_reason=fallback,
                           foreground_anchor_preserved=True, gated_target_preserved=True)
        else:
            (s2, sf, sbn), extra, receipt = replacement_evidence(query, mu_fg, mu_bg, mode)
            receipt.update(foreground_anchor_preserved=True, gated_target_preserved=True)
        receipt["reference"] = mean_receipt
        times = dict(shared_native_part2_seconds=part2_seconds,
                     replacement_seconds=time.monotonic() - started)
        begin = time.monotonic()
        s3, cand, prior = host._part3_clustering(s2, sf=sf, mu_fg=anchor,
                ref_feats_raw=fmaps[:, :n_refs], tgt_feat_raw=fmaps[:, n_refs],
                ref_masks=masks, n_refs=n_refs, h=h, w=w)
        times["part3_seconds"] = time.monotonic() - begin
        begin = time.monotonic()
        score = host._part4_semantic_consistency_correction(s3, sf=sf, sbn=sbn,
                cand_soft=cand, seed_prior=prior, tgt_feat=denoised)
        times["part4_seconds"] = time.monotonic() - begin
        begin = time.monotonic()
        pre = host._binarize_response(score, target_hw=host._tgt_image.shape[-2:])
        times["binarize_seconds"] = time.monotonic() - begin
        values = dict(s2=s2, sf=sf, sbn=sbn, mu_fg=anchor, s3=s3, cand_soft=cand,
                      seed_prior=prior, score=score, gated_target_norm=denoised.norm(dim=1)[0], **extra)
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError(f"Non-finite stage output: {arm}")
        if tuple(pre.shape) != (1024, 1024) or pre.dtype != torch.bool:
            raise ValueError("Native binarizer must return a bool 1024x1024 mask")
        prefinal[arm] = pre
        fields.update({f"{arm}.{name}": value for name, value in values.items()})
        receipts[arm], timings[arm] = receipt, times
    for arm in ARMS[1:]:
        for name in ("mu_fg", "cand_soft", "seed_prior", "gated_target_norm"):
            if not torch.equal(fields[f"{ARMS[0]}.{name}"], fields[f"{arm}.{name}"]):
                raise RuntimeError(f"Intervention changed retained raw geometry or auxiliary: {arm}.{name}")
    return prefinal, fields, receipts, timings
