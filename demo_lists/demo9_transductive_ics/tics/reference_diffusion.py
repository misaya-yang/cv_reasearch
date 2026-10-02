"""Reference-clamped full-patch manifold diffusion, with no learned weights.

Inputs are frozen support/query patch features, the ONLY support mask, and
native query geometry features. There is no query-label argument. Cross-image
edge identities are reciprocal feature kNN. Support labels determine clamped
values, class-balanced cross weights, and the native KDE fidelity target.
All graph weights are nonnegative; positive fidelity makes the query system
SPD even if cross edges are absent. Neither inference nor the controls impose
a foreground quota. This is a proposed method, not a performance claim.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    knn: int = 5
    cross_temperature: float = .10
    geometry_temperature: float = .20
    kde_temperature: float = .10
    spatial_strength: float = 1.0
    cross_strength: float = 1.0
    fidelity: float = 1.0
    chunk_rows: int = 256
    cg_tolerance: float = 1e-6
    cg_max_iterations: int = 200


@dataclass
class DiffusionSystem:
    query_count: int
    spatial_i: Tensor
    spatial_j: Tensor
    spatial_weight: Tensor
    cross_query: Tensor
    cross_support: Tensor
    cross_weight: Tensor
    cross_degree: Tensor
    cross_rhs: Tensor
    diagonal: Tensor
    rhs: Tensor
    target: Tensor
    fidelity: float

    def matvec(self, value: Tensor) -> Tensor:
        result = (self.fidelity + self.cross_degree) * value
        difference = self.spatial_weight * (value[self.spatial_i] - value[self.spatial_j])
        result = result.index_add(0, self.spatial_i, difference)
        return result.index_add(0, self.spatial_j, -difference)


def _unit(features: Tensor) -> Tensor:
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    return F.normalize(features.to(dtype), dim=-1)


def _topk(source: Tensor, target: Tensor, count: int, chunk_rows: int) -> Tensor:
    return torch.cat([(source[start:start + chunk_rows] @ target.T).topk(
        min(count, len(target)), dim=1).indices
        for start in range(0, len(source), chunk_rows)], dim=0)


def _kde_margin(query: Tensor, support: Tensor, labels: Tensor,
                temperature: float, chunk_rows: int) -> Tensor:
    fg, bg = support[labels], support[~labels]
    output = []
    for start in range(0, len(query), chunk_rows):
        scores = (query[start:start + chunk_rows] @ support.T - 1) / temperature
        output.append(scores[:, labels].logsumexp(1) - math.log(len(fg))
                      - scores[:, ~labels].logsumexp(1) + math.log(len(bg)))
    return torch.cat(output)


def _validate_config(config: DiffusionConfig) -> None:
    positive = (config.cross_temperature, config.geometry_temperature,
                config.kde_temperature, config.fidelity, config.cg_tolerance)
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        raise ValueError("Temperatures, fidelity and tolerance must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in
           (config.spatial_strength, config.cross_strength)):
        raise ValueError("Graph strengths must be finite and nonnegative")
    if min(config.knn, config.chunk_rows, config.cg_max_iterations) < 1:
        raise ValueError("Positive integer kNN, chunk size and iteration count required")


@torch.no_grad()
def build_system(support_features: Tensor, support_mask: Tensor,
                 query_features: Tensor, query_shape: tuple[int, int],
                 geometry_query: Tensor, config: DiffusionConfig = DiffusionConfig(),
                 variant: str = "diffusion") -> DiffusionSystem:
    """Construct an SPD operator without a dense query-by-query matrix.

    variants share the identical full-patch features and fixed parameters:
      diffusion: spatial + reciprocal cross edges + native KDE fidelity;
      boundary_only: spatial + same KDE fidelity, no cross edges;
      plain_label_propagation: same graph/fidelity, zero fidelity target;
      native_kde: no edges, same fidelity target (cheap same-feature control).
    This function requires both support labels. The public wrapper below handles
    degeneracies by preserving a supplied host score exactly.
    """
    _validate_config(config)
    if variant not in {"diffusion", "boundary_only", "plain_label_propagation", "native_kde"}:
        raise ValueError("Unknown diffusion/control variant")
    if support_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("Expected full patch features [N,D]")
    n, dimensions = query_features.shape
    if support_features.shape[1] != dimensions or geometry_query.ndim != 2 or len(geometry_query) != n:
        raise ValueError("Feature dimensions/grid lengths disagree")
    if n != query_shape[0] * query_shape[1] or min(query_shape) < 1:
        raise ValueError("Query shape must match all query patches")
    tensors = (support_features, query_features, geometry_query, support_mask)
    if any(x.device != query_features.device for x in tensors):
        raise ValueError("All inputs must be on the same device")
    if any(not bool(torch.isfinite(x).all()) for x in tensors):
        raise ValueError("Nonfinite graph inputs")
    if any(bool((x.norm(dim=-1) <= 1e-12).any()) for x in tensors[:3]):
        raise ValueError("Zero feature vector has no feature correspondence")
    if support_mask.numel() != len(support_features):
        raise ValueError("Only support mask must match all support patches")
    if not bool(((support_mask == 0) | (support_mask == 1)).all()):
        raise ValueError("Support mask must be binary")
    labels = support_mask.reshape(-1).bool()
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("Both support foreground and background are required")
    query = _unit(query_features)
    support = _unit(support_features).to(query.dtype)
    geometry = _unit(geometry_query).to(query.dtype)
    grid = torch.arange(n, device=query.device).reshape(query_shape)
    edge_i = torch.cat((grid[:, :-1].reshape(-1), grid[:-1, :].reshape(-1)))
    edge_j = torch.cat((grid[:, 1:].reshape(-1), grid[1:, :].reshape(-1)))
    spatial = torch.exp(((geometry[edge_i] * geometry[edge_j]).sum(-1) - 1)
                        / config.geometry_temperature) * config.spatial_strength
    if variant == "native_kde":
        spatial = torch.zeros_like(spatial)

    cq = torch.empty(0, dtype=torch.long, device=query.device)
    cs = cq.clone()
    cw = query.new_empty(0)
    if variant not in {"boundary_only", "native_kde"} and config.cross_strength > 0:
        forward = _topk(query, support, config.knn, config.chunk_rows)
        backward = _topk(support, query, config.knn, config.chunk_rows)
        candidates_q = torch.arange(n, device=query.device)[:, None].expand_as(forward).reshape(-1)
        candidates_s = forward.reshape(-1)
        reciprocal = (backward[candidates_s] == candidates_q[:, None]).any(1)
        cq, cs = candidates_q[reciprocal], candidates_s[reciprocal]
        similarity = (query[cq] * support[cs]).sum(-1)
        # Exactly the equal class prior used by KDE. A node's class count is
        # unchanged under a complete foreground/background label swap.
        fg_count = labels.sum().to(query.dtype)
        bg_count = (~labels).sum().to(query.dtype)
        class_count = torch.where(labels[cs], fg_count, bg_count)
        balance = len(support) / (2 * class_count)
        cw = (torch.exp((similarity - 1) / config.cross_temperature)
              * balance * config.cross_strength)
    cross_degree = query.new_zeros(n).index_add(0, cq, cw)
    signed_labels = labels.to(query.dtype) * 2 - 1
    cross_rhs = query.new_zeros(n).index_add(0, cq, cw * signed_labels[cs])
    target = torch.tanh(_kde_margin(query, support, labels,
                                  config.kde_temperature, config.chunk_rows) / 2)
    if variant == "plain_label_propagation":
        target = torch.zeros_like(target)
    diagonal = cross_degree + config.fidelity
    diagonal = diagonal.index_add(0, edge_i, spatial).index_add(0, edge_j, spatial)
    return DiffusionSystem(n, edge_i, edge_j, spatial, cq, cs, cw, cross_degree,
                           cross_rhs, diagonal, config.fidelity * target + cross_rhs,
                           target, config.fidelity)


@torch.no_grad()
def solve_cg(system: DiffusionSystem, tolerance: float = 1e-6,
             max_iterations: int = 200) -> tuple[Tensor, dict]:
    """Jacobi-preconditioned CG with a final recomputed relative residual."""
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive solver tolerance and iteration count required")
    value = system.target.clone()
    residual = system.rhs - system.matvec(value)
    norm = system.rhs.norm().clamp_min(torch.finfo(value.dtype).eps)
    preconditioned = residual / system.diagonal
    direction = preconditioned.clone()
    rz = torch.dot(residual, preconditioned)
    iterations = 0
    for iteration in range(max_iterations):
        if float(residual.norm() / norm) <= tolerance:
            break
        product = system.matvec(direction)
        curvature = torch.dot(direction, product)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
            raise RuntimeError("CG lost positive curvature; SPD contract violated")
        step = rz / curvature
        value = value + step * direction
        residual = residual - step * product
        preconditioned = residual / system.diagonal
        next_rz = torch.dot(residual, preconditioned)
        direction = preconditioned + (next_rz / rz) * direction
        rz = next_rz
        iterations = iteration + 1
    relative = float((system.rhs - system.matvec(value)).norm() / norm)
    audit = dict(iterations=iterations, relative_residual=relative,
                 converged=bool(math.isfinite(relative) and relative <= tolerance * 2))
    return value, audit


@torch.no_grad()
def reference_diffusion(support_features: Tensor, support_mask: Tensor,
                        query_features: Tensor, query_shape: tuple[int, int],
                        geometry_query: Tensor, *, fallback_score: Tensor,
                        config: DiffusionConfig = DiffusionConfig(),
                        variant: str = "diffusion") -> tuple[Tensor, dict]:
    """Return patch signed scores; caller keeps the agreed host upsample rule.

    fallback_score is the unchanged host signed patch field, NOT query GT. It
    is required so missing support classes/nonfinite features/nonconvergence
    can return an exact lawful fallback instead of a fabricated empty mask.
    Shape/device/programming errors remain explicit exceptions.
    """
    _validate_config(config)
    if fallback_score.shape != (len(query_features),) or fallback_score.device != query_features.device:
        raise ValueError("Fallback must be a native signed query patch field on the same device")
    if not bool(torch.isfinite(fallback_score).all()):
        raise ValueError("Native fallback must be finite")
    audit = dict(method="reference_clamped_manifold_diffusion", variant=variant,
                 config=asdict(config), query_gt_used=False,
                 edge_rule="Label-independent reciprocal feature kNN; label-dependent equal-class cross strength",
                 foreground_quota=False)
    labels = support_mask.reshape(-1)
    if not bool((labels == 1).any()) or not bool((labels == 0).any()):
        return fallback_score.clone(), {**audit, "state": "FALLBACK_MISSING_SUPPORT_CLASS"}
    if any(not bool(torch.isfinite(x).all()) for x in
           (support_features, query_features, geometry_query)):
        return fallback_score.clone(), {**audit, "state": "FALLBACK_NONFINITE_FEATURES"}
    if any(bool((x.norm(dim=-1) <= 1e-12).any()) for x in
           (support_features, query_features, geometry_query)):
        return fallback_score.clone(), {**audit, "state": "FALLBACK_ZERO_FEATURES"}
    system = build_system(support_features, support_mask, query_features, query_shape,
                          geometry_query, config, variant)
    value, solver = solve_cg(system, config.cg_tolerance, config.cg_max_iterations)
    audit.update(solver)
    audit.update(spatial_edges=len(system.spatial_weight), cross_edges=len(system.cross_weight),
                 minimum_fidelity=config.fidelity)
    if not solver["converged"] or not bool(torch.isfinite(value).all()):
        return fallback_score.clone(), {**audit, "state": "FALLBACK_NONCONVERGED"}
    return value, {**audit, "state": "SOLVED"}
