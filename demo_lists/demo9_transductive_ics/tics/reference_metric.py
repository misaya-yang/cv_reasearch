"""Reference-constrained metric and density correction; no encoder/GT access.

Shared rank-one metrics are established (SCML); this implementation is a
candidate, not a novelty/gain claim. Solve the strictly convex preference to
stationarity, differentiate implicitly, then take a differentiable protected
step. Host correction scale is a GLOBAL train/development-fitted parameter.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricConfig:
    rank: int = 32
    margin: float = .1
    temperature: float = .05
    regularization: float = .1
    retain_fraction: float = .5
    minimum: float = .25
    maximum: float = 4.
    bandwidth: float = .1
    max_anchors_per_class: int = 256
    positives: int = 2
    negatives: int = 4
    exclusion_radius: int = 2
    max_iterations: int = 60
    tolerance: float = 1e-9


def _validate_config(c):
    if not (c.margin > 0 and c.temperature > 0 and c.regularization > 0
            and 0 <= c.retain_fraction < 1 and 0 < c.minimum < 1 < c.maximum
            and c.bandwidth > 0 and c.max_iterations > 0 and c.tolerance > 0):
        raise ValueError('Positive temperatures/regularizer, valid bounds and protection required')


class _Preference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, design, margins, mass, kappa, temperature, regularizer, max_iter, tolerance):
        if design.ndim != 2 or margins.shape != mass.shape or margins.shape != design.shape[:1]:
            raise ValueError('Expected design[T,r], margins/mass[T]')
        if not (torch.isfinite(design).all() and torch.isfinite(margins).all()
                and torch.isfinite(mass).all() and (mass >= 0).all() and mass.sum() > 0):
            raise ValueError('Finite positive-mass design required')
        # Double precision and explicit convergence matter for the implicit VJP.
        a, m, p = design.double(), margins.double(), mass.double()
        v = torch.ones(a.shape[1], dtype=a.dtype, device=a.device)
        eye = torch.eye(len(v), dtype=a.dtype, device=a.device)

        def evaluate(value):
            z = (kappa - m - a @ (value - 1)) / temperature
            s = z.sigmoid()
            loss = (p * F.softplus(z)).sum() + regularizer / 2 * (value - 1).square().sum()
            grad = regularizer * (value - 1) - a.T @ (p * s) / temperature
            hessian = regularizer * eye + a.T @ ((p * s * (1 - s) / temperature ** 2)[:, None] * a)
            return loss, grad, hessian

        converged = False
        for _ in range(int(max_iter)):
            loss, grad, h = evaluate(v)
            if float(grad.abs().max()) <= tolerance:
                converged = True
                break
            step = torch.linalg.solve(h, grad)
            directional = (grad * step).sum()
            # Near the minimizer, the predicted decrease can be smaller than
            # one objective ULP. Literal Armijo can then reject a full Newton
            # step (or accept an unchanged rounded iterate) despite a much
            # smaller actual stationarity residual. This does NOT relax the
            # original convergence gate or change the objective/Hessian.
            ulp = torch.nextafter(loss, loss.new_tensor(float('inf'))) - loss
            roundoff_bound = 8 * ulp
            gradient_norm = grad.abs().max()
            accepted = False
            rate = 1.
            for _ in range(40):
                candidate = v - rate * step
                candidate_loss, candidate_grad, _ = evaluate(candidate)
                if rate * directional <= roundoff_bound:
                    # At floating-point objective resolution, require BOTH
                    # a bounded objective roundoff and strict improvement in
                    # the very same max-gradient norm used for stationarity.
                    admissible = (candidate_loss <= loss + roundoff_bound
                                  and torch.isfinite(candidate_grad).all()
                                  and candidate_grad.abs().max() < gradient_norm)
                else:
                    admissible = candidate_loss <= loss - 1e-4 * rate * directional
                if admissible:
                    v = candidate
                    accepted = True
                    break
                rate *= .5
            if not accepted:
                # No silent differentiation of an unconverged point.
                break
        _, grad, h = evaluate(v)
        if not converged and float(grad.abs().max()) > tolerance:
            raise RuntimeError(f'Preference Newton not stationary: maxgrad={float(grad.abs().max()):.3g}')
        if not torch.isfinite(v).all():
            raise RuntimeError('Nonfinite preference')
        ctx.save_for_backward(a, m, p, v, h)
        ctx.kappa, ctx.temperature = kappa, temperature
        ctx.input_dtypes = (design.dtype, margins.dtype, mass.dtype)
        return v.to(design.dtype)

    @staticmethod
    def backward(ctx, upstream):
        a, m, p, v, h = ctx.saved_tensors
        t = ctx.temperature
        dual = torch.linalg.solve(h.T, upstream.double())
        s = ((ctx.kappa - m - a @ (v - 1)) / t).sigmoid()
        adual = a @ dual
        beta = p * s * (1 - s) / t ** 2
        ga = (p * s / t)[:, None] * dual[None] - (beta * adual)[:, None] * (v - 1)[None]
        gm = -beta * adual
        gp = s * adual / t
        dtypes = ctx.input_dtypes
        return ga.to(dtypes[0]), gm.to(dtypes[1]), gp.to(dtypes[2]), None, None, None, None, None


def preference(design, margins, mass, config=MetricConfig()):
    _validate_config(config)
    return _Preference.apply(design, margins, mass, config.margin, config.temperature,
                             config.regularization, config.max_iterations, config.tolerance)


def protected_step(value, design, margins, config=MetricConfig()):
    """Largest feasible step along preference; NOT constrained-global optimum.

    min ties use PyTorch amin's subgradient. Ordinary finite-difference claims
    are restricted to unique active constraints. No detach of alpha.
    """
    _validate_config(config)
    delta = value - 1
    # Only constraints violated at alpha=1 can constrain a step in [0,1].
    # This is exact and avoids evaluating tiny/zero inactive denominators.
    positive_active = delta > config.maximum - 1
    negative_active = delta < config.minimum - 1
    pos_den = torch.where(positive_active, delta, torch.ones_like(delta))
    neg_den = torch.where(negative_active, -delta, torch.ones_like(delta))
    positive = torch.where(positive_active, (config.maximum - 1) / pos_den, torch.ones_like(delta))
    negative = torch.where(negative_active, (1 - config.minimum) / neg_den, torch.ones_like(delta))
    slope = design @ delta
    guard = margins >= config.margin
    # Clamp before division, never hide invalid division behind where.
    budget = (1 - config.retain_fraction) * margins.clamp_min(0)
    valid = guard & (slope < -budget)
    guard_den = torch.where(valid, -slope, torch.ones_like(slope))
    guarded = torch.where(valid, budget / guard_den, torch.ones_like(slope))
    limits = torch.cat([value.new_ones(1), positive, negative, guarded])
    alpha = limits.amin().clamp_min(0)
    weights = 1 + alpha * delta
    return weights, alpha


def local_triplets(features, coverage, grid, config=MetricConfig()):
    """Reference labels only; mixed patches omitted; original neighbours frozen.

    Deterministic evenly spaced anchor subsampling; Cartesian positive/negative
    witnesses. No query tensor or annotation is accepted.
    """
    if features.ndim != 2 or coverage.shape != features.shape[:1] or math.prod(grid) != len(features):
        raise ValueError('features[P,D], coverage[P], matching patch grid required')
    f = F.normalize(features.detach(), dim=-1)
    foreground, background = coverage >= .9, coverage <= .1
    yy, xx = torch.meshgrid(torch.arange(grid[0], device=f.device), torch.arange(grid[1], device=f.device), indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], 1)
    groups = []
    for own, other in ((foreground, background), (background, foreground)):
        pool = own.nonzero().flatten()
        opposites = other.nonzero().flatten()
        if len(pool) < 2 or not len(opposites):
            continue
        n = min(len(pool), config.max_anchors_per_class)
        anchors = pool[torch.linspace(0, len(pool) - 1, n, device=f.device).round().long()]
        records = []
        for anchor in anchors:
            far = (coords[pool] - coords[anchor]).abs().amax(1) > config.exclusion_radius
            positives = pool[far]
            if not len(positives):
                continue
            pos = positives[(f[positives] @ f[anchor]).topk(min(config.positives, len(positives))).indices]
            neg = opposites[(f[opposites] @ f[anchor]).topk(min(config.negatives, len(opposites))).indices]
            for p in pos:
                for n_ in neg:
                    records.append(torch.stack([anchor, p, n_]))
        if records:
            groups.append(torch.stack(records))
    if len(groups) != 2:
        return torch.empty(0, 3, device=f.device, dtype=torch.long), f.new_empty(0)
    triples = torch.cat(groups)
    mass = torch.cat([f.new_full((len(g),), .5 / len(g)) for g in groups])
    return triples, mass


def difference_design(features, directions, triples):
    a, p, n = features[triples[:, 0]], features[triples[:, 1]], features[triples[:, 2]]
    positive, negative = a - p, a - n
    margins = negative.square().sum(1) - positive.square().sum(1)
    design = (negative @ directions).square() - (positive @ directions).square()
    return margins, design


def metric_distances(query, anchors, directions, weights):
    """No normalization AFTER the metric transform; original inputs are unit."""
    base = query.square().sum(1)[:, None] + anchors.square().sum(1)[None] - 2 * query @ anchors.T
    q, s = query @ directions, anchors @ directions
    change = (weights - 1)
    extra = (q.square() * change).sum(1)[:, None] + (s.square() * change).sum(1)[None] - 2 * (q * change) @ s.T
    return (base + extra).clamp_min(0)  # Only numerical negative roundoff under an SPD metric.


def density_ratio(query, reference, foreground, directions, weights, bandwidth=.1,
                  query_chunk=128, anchor_chunk=512):
    if not foreground.any() or foreground.all():
        raise ValueError('Both reference labels required')
    scores = []
    for block in query.split(query_chunk):
        categories = []
        for mask in (foreground, ~foreground):
            anchors = reference[mask]
            aggregate = block.new_full((len(block),), -float('inf'))
            for subset in anchors.split(anchor_chunk):
                logits = -metric_distances(block, subset, directions, weights) / (2 * bandwidth)
                aggregate = torch.logaddexp(aggregate, logits.logsumexp(1))
            categories.append(aggregate - math.log(len(anchors)))
        scores.append(categories[0] - categories[1])
    return torch.cat(scores)


class ReferenceMetric(nn.Module):
    """Learn dictionary/global correction gain on BASE episodes; test fits w only.

    beta is a bounded learned fusion coefficient, NOT a calibrated Bayes factor.
    All learned controls receive the same coefficient/labels/budget opportunity.
    """
    def __init__(self, dimension, config=MetricConfig(), seed=2048):
        super().__init__()
        _validate_config(config)
        if not 1 <= config.rank <= dimension:
            raise ValueError('rank <= feature dimension required')
        g = torch.Generator().manual_seed(seed)
        self.dictionary = nn.Parameter(torch.randn(dimension, config.rank, generator=g) / math.sqrt(dimension))
        self.gain_parameter = nn.Parameter(torch.tensor(-2.))
        self.config = config

    def directions(self):
        q, r = torch.linalg.qr(self.dictionary, mode='reduced')
        if (r.diagonal().abs() < 1e-7).any():
            raise RuntimeError('Dictionary QR rank deficient; preserve failure, do not silently change rank')
        return q

    def adapt(self, reference, coverage, grid):
        unit = F.normalize(reference, dim=-1)
        triples, mass = local_triplets(unit, coverage, grid, self.config)
        directions = self.directions()
        if not len(triples):
            return directions, unit.new_ones(self.config.rank), dict(state='FALLBACK_NO_LEGAL_TRIPLETS', alpha=0., triples=0)
        margins, design = difference_design(unit, directions, triples)
        desired = preference(design, margins, mass, self.config)
        weights, alpha = protected_step(desired, design, margins, self.config)
        return directions, weights, dict(state='ADAPTED', alpha=float(alpha.detach()), triples=len(triples),
                                         min_weight=float(weights.detach().min()), max_weight=float(weights.detach().max()))

    def forward(self, reference, coverage, query, grid):
        directions, weights, audit = self.adapt(reference, coverage, grid)
        if audit['state'].startswith('FALLBACK'):
            return query.new_zeros(len(query)), audit
        r, q = F.normalize(reference, dim=-1), F.normalize(query, dim=-1)
        foreground = coverage >= .5
        changed = density_ratio(q, r, foreground, directions, weights, self.config.bandwidth)
        identity = density_ratio(q, r, foreground, directions, torch.ones_like(weights), self.config.bandwidth)
        correction = self.config.bandwidth * (changed - identity)
        audit['gain'] = float((4 * self.gain_parameter.sigmoid()).detach())
        return 4 * self.gain_parameter.sigmoid() * correction, audit


def correct_host_field(host_field, patch_delta, grid):
    """Continuous correction before threshold. Exact identity for zero correction."""
    if patch_delta.numel() != math.prod(grid):
        raise ValueError('Delta and grid differ')
    if not patch_delta.requires_grad and torch.count_nonzero(patch_delta) == 0:
        return host_field
    delta = F.interpolate(patch_delta.reshape(1, 1, *grid), size=host_field.shape[-2:], mode='bilinear', align_corners=False)
    return host_field + delta.reshape(host_field.shape)
