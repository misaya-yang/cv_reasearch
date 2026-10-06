"""Fixed-correspondence inverse label inference with a native continuous prior.

Unlike reconstruction.py, no query residual histogram becomes a mask. Query
tokens form a dictionary for each reference token. A fixed sparse row-stochastic
operator A reconstructs reference features, then reference coverage constrains
A z. No correspondence, area, threshold or weight is selected using query GT.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .reconstruction import _simplex

CONFIG = {
    "name": "fixed_correspondence_inverse_labels",
    "version": 1,
    "neighbors": 8,
    "dictionary_steps": 32,
    "reference_chunk": 128,
    "inverse_steps": 96,
    "source_total_mass": 1.0,
    "native_total_mass": 1.0,
    "source_role_balance": "equal FG/BG coverage mass",
    "native_normalization": "(score-min)/(max-min), denominator floored at 1e-6",
    "control": "same_A_and_native_prior_direct_label_transfer",
    "extra_encoder_forwards": 0,
    "layers": "existing cached layer only",
}


def _operator(q, r):
    """Reference rows reconstruct themselves using convex query dictionaries."""
    k = min(CONFIG["neighbors"], len(q))
    all_indices, all_weights, residuals = [], [], []
    for start in range(0, len(r), CONFIG["reference_chunk"]):
        references = r[start : start + CONFIG["reference_chunk"]]
        indices = (references @ q.T).topk(k, dim=-1, largest=True, sorted=True).indices
        atoms = q[indices]
        gram = atoms @ atoms.transpose(1, 2)
        cross = (atoms * references[:, None]).sum(-1)
        weights = torch.full(indices.shape, 1 / k, dtype=q.dtype, device=q.device)
        lipschitz = gram.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-7)
        for _ in range(CONFIG["dictionary_steps"]):
            gradient = (gram @ weights[..., None]).squeeze(-1) - cross
            weights = _simplex(weights - gradient / lipschitz)
        residuals.append((references - (weights[..., None] * atoms).sum(1)).square().sum(-1))
        all_indices.append(indices)
        all_weights.append(weights)
    return torch.cat(all_indices), torch.cat(all_weights), torch.cat(residuals)


def _role_weights(coverage):
    fg, bg = coverage.sum(), (1 - coverage).sum()
    if bool((fg > 1e-7) & (bg > 1e-7)):
        return 0.5 * (coverage / fg + (1 - coverage) / bg)
    return torch.full_like(coverage, 1 / len(coverage))


def _run(q, r, cov, score, *, device="cpu", extras=None, inverse):
    del extras
    started = time.perf_counter()
    shape = tuple(score.shape)
    if len(shape) != 2:
        raise ValueError("score must be an HxW grid")
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    r = torch.as_tensor(r, dtype=torch.float32, device=device)
    coverage = torch.as_tensor(cov, dtype=torch.float32, device=device).reshape(-1)
    raw = torch.as_tensor(score, dtype=torch.float32, device=device).reshape(-1)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("q/r must be NxD with shared feature dimension")
    if len(q) == 0 or len(r) == 0 or len(q) != len(raw) or len(r) != len(coverage):
        raise ValueError("empty input or incompatible token/grid counts")
    if not all(bool(torch.isfinite(x).all()) for x in (q, r, coverage, raw)):
        raise ValueError("inputs must be finite")
    if bool(((coverage < 0) | (coverage > 1)).any()):
        raise ValueError("reference coverage must lie in [0,1]")

    with torch.no_grad():
        q = torch.nn.functional.normalize(q, dim=-1)
        r = torch.nn.functional.normalize(r, dim=-1)
        initial = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
        indices, weights, residuals = _operator(q, r)
        row_weights = _role_weights(coverage) * CONFIG["source_total_mass"]
        prior = CONFIG["native_total_mass"] / len(q)

        def apply(z):
            return (z[indices] * weights).sum(-1)

        def transpose(v):
            result = torch.zeros(len(q), dtype=q.dtype, device=q.device)
            result.scatter_add_(0, indices.reshape(-1), (v[:, None] * weights).reshape(-1))
            return result

        column_mass = transpose(row_weights)
        label_mass = transpose(row_weights * coverage)

        def inverse_objective(z):
            return 0.5 * (row_weights * (apply(z) - coverage).square()).sum() + 0.5 * prior * (z - initial).square().sum()

        def direct_objective(z):
            return 0.5 * (row_weights[:, None] * weights * (z[indices] - coverage[:, None]).square()).sum() + 0.5 * prior * (z - initial).square().sum()

        if inverse:
            z = initial.clone()
            # A is nonnegative with unit row sums. The maximum column mass
            # therefore bounds ||A' W A||_infinity and its spectral norm.
            lipschitz = column_mass.max() + prior
            for _ in range(CONFIG["inverse_steps"]):
                gradient = transpose(row_weights * (apply(z) - coverage)) + prior * (z - initial)
                z = (z - gradient / lipschitz).clamp(0, 1)
            gradient = transpose(row_weights * (apply(z) - coverage)) + prior * (z - initial)
            stationarity = (z - (z - gradient / lipschitz).clamp(0, 1)).abs().max()
            objective = inverse_objective
        else:
            # Exact minimizer of sum_r W_r sum_i A_ri (z_i-y_r)^2 plus
            # the same native quadratic prior. No inverse-label coupling.
            z = (label_mass + prior * initial) / (column_mass + prior)
            gradient = (column_mass + prior) * z - label_mass - prior * initial
            stationarity = gradient.abs().max()
            objective = direct_objective

        info = {
            "config": dict(CONFIG),
            "method": "inverse_labels" if inverse else "direct_transfer_control",
            "device": str(device),
            "status": "ok",
            "native_prior_is_pre_crf": True,
            "uses_query_truth": False,
            "correspondence_updated_by_labels": False,
            "query_tokens": len(q),
            "reference_tokens": len(r),
            "actual_neighbors": int(indices.shape[1]),
            "optimization_steps": CONFIG["inverse_steps"] if inverse else 0,
            "source_mass": float(row_weights.sum().item()),
            "native_mass": float(prior * len(q)),
            "column_mass_max": float(column_mass.max().item()),
            "unconstrained_query_fraction": float((column_mass <= 1e-12).float().mean().item()),
            "reference_feature_reconstruction_error": float(residuals.mean().item()),
            "source_coverage_error_before": float((row_weights * (apply(initial) - coverage).square()).sum().item()),
            "source_coverage_error_after": float((row_weights * (apply(z) - coverage).square()).sum().item()),
            "optimized_objective_before": float(objective(initial).item()),
            "optimized_objective_after": float(objective(z).item()),
            "inverse_objective_after": float(inverse_objective(z).item()),
            "stationarity_residual": float(stationarity.item()),
            "native_patch_foreground_fraction": float((initial > 0.5).float().mean().item()),
            "output_patch_foreground_fraction": float((z > 0.5).float().mean().item()),
            "changed_patch_fraction": float(((initial > 0.5) != (z > 0.5)).float().mean().item()),
            "soft_change_mean_abs": float((z - initial).abs().mean().item()),
            "cuda_scatter_roundoff": "scatter_add may differ at floating-point roundoff level on CUDA",
        }
        field = z.reshape(shape).cpu().numpy().astype(np.float32)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    info["runtime_seconds"] = time.perf_counter() - started
    return field, info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Joint query-label inverse problem; fixed source correspondence and prior."""
    return _run(q, r, cov, score, device=device, extras=extras, inverse=True)


def control(q, r, cov, score, *, device="cpu", extras=None):
    """Same correspondence and prior, exact independent direct-label solution."""
    return _run(q, r, cov, score, device=device, extras=extras, inverse=False)
