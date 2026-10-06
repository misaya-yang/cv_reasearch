"""Local convex reconstruction with cross-coordinate stress testing.

This is an unvalidated complete mechanism, not the retired RICE eigenspace.
Reference FG and BG each explain every query token using convex combinations
of their eight nearest reference tokens. Coefficients are fitted in one fixed
coordinate half and evaluated in the other, then the halves are exchanged.
DINO coordinates are correlated: this split is NOT independent validation.
No query annotation, native-mask gate, candidate bank or fitted source cut is
used. The complete output uses a fixed Otsu cut on the query residual margin.
"""

from __future__ import annotations

import time

import numpy as np
import torch

CONFIG = {
    "name": "cross_coordinate_local_convex_reconstruction",
    "version": 1,
    "neighbors": 8,
    "optimization_steps": 32,
    "query_chunk": 128,
    "coordinate_seed": 0,
    "otsu_bins": 256,
    "reference_fg_coverage": 0.5,
    "extra_encoder_forwards": 0,
    "layers": "existing cached layer only",
    "control": "uniform_same_neighbors_same_coordinate_split",
}


def _simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto nonnegative weights summing to one."""
    ordered, _ = values.sort(dim=-1, descending=True)
    sums = ordered.cumsum(dim=-1) - 1
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device)
    active = ordered - sums / ranks > 0
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1)
    theta = sums.gather(-1, rho - 1) / rho
    return (values - theta).clamp_min(0)


def _validated_errors(query, reference, fit_dims, eval_dims, optimize):
    """Process chunks so the local dictionary does not allocate N*K*D globally."""
    k = min(CONFIG["neighbors"], len(reference))
    fit_r, eval_r = reference[:, fit_dims], reference[:, eval_dims]
    errors = []
    fit_norm = fit_r.square().sum(-1)
    residual_objectives = []
    for start in range(0, len(query), CONFIG["query_chunk"]):
        fit_q = query[start : start + CONFIG["query_chunk"], fit_dims]
        eval_q = query[start : start + CONFIG["query_chunk"], eval_dims]
        # Euclidean neighbors within a coordinate half; full vectors were
        # normalized once before splitting, so no half-specific renormalizing.
        distances = fit_q.square().sum(-1, keepdim=True) + fit_norm - 2 * fit_q @ fit_r.T
        indices = distances.topk(k, dim=-1, largest=False, sorted=True).indices
        atoms = fit_r[indices]
        weights = torch.full(indices.shape, 1 / k, device=query.device)
        if optimize:
            gram = atoms @ atoms.transpose(1, 2)
            cross = (atoms * fit_q[:, None, :]).sum(-1)
            # Absolute row-sum bounds spectral norm, ensuring a safe step.
            lipschitz = gram.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-7)
            for _ in range(CONFIG["optimization_steps"]):
                gradient = (gram @ weights[..., None]).squeeze(-1) - cross
                weights = _simplex(weights - gradient / lipschitz)
        fit_reconstruction = (atoms * weights[..., None]).sum(1)
        residual_objectives.append((fit_q - fit_reconstruction).square().sum(-1))
        eval_reconstruction = (eval_r[indices] * weights[..., None]).sum(1)
        errors.append((eval_q - eval_reconstruction).square().sum(-1))
    return torch.cat(errors), torch.cat(residual_objectives)


def _query_cut(values):
    """Fixed histogram Otsu cut, with a zero-margin cut for constant maps."""
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-10:
        return 0.0, "constant_margin_zero_cut"
    histogram, edges = np.histogram(values, bins=CONFIG["otsu_bins"], range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = histogram.cumsum(dtype=np.float64)
    weighted = (histogram * centers).cumsum()
    denominator = weight[:-1] * (len(values) - weight[:-1])
    between = np.full(len(denominator), -np.inf)
    valid = denominator > 0
    between[valid] = (weighted[-1] * weight[:-1][valid] - weighted[:-1][valid] * len(values)) ** 2 / denominator[valid]
    return float(edges[int(np.argmax(between)) + 1]), "query_otsu"


def _run(q, r, cov, score, *, device="cpu", extras=None, optimize):
    del extras
    started = time.perf_counter()
    shape = tuple(np.shape(score))
    if len(shape) != 2:
        raise ValueError("score must be an HxW grid")
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    r = torch.as_tensor(r, dtype=torch.float32, device=device)
    coverage = torch.as_tensor(cov, dtype=torch.float32, device=device).reshape(-1)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("q and r must be NxD token matrices with the same D")
    if len(q) != int(np.prod(shape)) or len(r) != len(coverage):
        raise ValueError("query/score or reference/coverage token counts differ")
    if q.shape[1] < 2:
        raise ValueError("at least two feature coordinates are required")
    if not torch.isfinite(q).all() or not torch.isfinite(r).all() or not torch.isfinite(coverage).all():
        raise ValueError("non-finite input")
    if ((coverage < 0) | (coverage > 1)).any():
        raise ValueError("cov must contain foreground coverage in [0,1]")
    foreground = coverage >= CONFIG["reference_fg_coverage"]
    n_fg = int(foreground.sum().item())
    info = {"config": dict(CONFIG), "reference_fg_tokens": n_fg, "reference_bg_tokens": len(r) - n_fg,
            "uses_native_values": False, "coordinate_split_is_independent": False,
            "method": "optimized_simplex" if optimize else "uniform_control", "device": str(device)}
    if n_fg == 0 or n_fg == len(r):
        # Defined no-evidence behavior, not a GT/native fallback.
        fill = 0.0 if n_fg == 0 else 1.0
        info.update(status="degenerate_reference", runtime_seconds=time.perf_counter() - started)
        return np.full(shape, fill, dtype=np.float32), info
    with torch.no_grad():
        q = torch.nn.functional.normalize(q, dim=-1)
        r = torch.nn.functional.normalize(r, dim=-1)
        permutation = np.random.RandomState(CONFIG["coordinate_seed"]).permutation(q.shape[1])
        a = torch.as_tensor(permutation[::2], device=device)
        b = torch.as_tensor(permutation[1::2], device=device)
        fg_error = torch.zeros(len(q), device=device)
        bg_error = torch.zeros_like(fg_error)
        fit_fg = torch.zeros_like(fg_error)
        fit_bg = torch.zeros_like(fg_error)
        for fitting, evaluation in ((a, b), (b, a)):
            error, fit = _validated_errors(q, r[foreground], fitting, evaluation, optimize)
            fg_error += error
            fit_fg += fit
            error, fit = _validated_errors(q, r[~foreground], fitting, evaluation, optimize)
            bg_error += error
            fit_bg += fit
        margin = (bg_error - fg_error).cpu().numpy()
        cut, cut_status = _query_cut(margin)
        # Preserve the cut exactly while avoiding saturation. The shared
        # renderer bilinearly upsamples this soft field and thresholds at .5.
        scale = max(float(np.max(np.abs(margin - cut))), 1e-7)
        field = 0.5 + (margin - cut) / (2 * scale)
        info.update(status="ok", cut=cut, cut_status=cut_status,
                    margin_min=float(margin.min()), margin_max=float(margin.max()),
                    foreground_fraction=float(np.mean(field > 0.5)),
                    fg_eval_residual=float(fg_error.mean().item()), bg_eval_residual=float(bg_error.mean().item()),
                    fg_fit_residual=float(fit_fg.mean().item()), bg_fit_residual=float(fit_bg.mean().item()))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    info["runtime_seconds"] = time.perf_counter() - started
    return field.reshape(shape).astype(np.float32), info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Return the complete low-resolution soft mask and diagnostic metadata."""
    return _run(q, r, cov, score, device=device, extras=extras, optimize=True)


def control(q, r, cov, score, *, device="cpu", extras=None):
    """Same neighbors, coordinate split and cut; uniform reconstruction weights."""
    return _run(q, r, cov, score, device=device, extras=extras, optimize=False)
