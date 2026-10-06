"""Reference foreground/background score densities with unlabeled query prior fit.

The one-dimensional mixture likelihood is concave. Its optimum does not prove
that reference class-conditional densities transfer to the query, or that the
resulting posterior is calibrated for semantic segmentation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .reference_occupancy import unit


@dataclass(frozen=True)
class Config:
    bins: int = 64
    bandwidth: float = .1
    uniform_mass: float = .01
    purity: float = .9
    minimum_reference_samples: int = 8
    prior_tolerance: float = 1e-10
    maximum_bisections: int = 64
    identifiability_tolerance: float = 1e-12


def fit_prior(foreground, background, lower, upper, cfg=Config()):
    f = np.asarray(foreground, dtype=np.float64)
    b = np.asarray(background, dtype=np.float64)
    if (f.shape != b.shape or f.ndim != 1 or not len(f) or not np.isfinite(f).all()
            or not np.isfinite(b).all() or np.any(f <= 0) or np.any(b <= 0)
            or not 0 < lower < upper < 1):
        raise ValueError('Positive aligned mixture densities and a valid prior interval required')
    delta = f - b

    def derivative(prior):
        return float(np.mean(delta / (b + prior * delta)))

    left, right = lower, upper
    left_derivative, right_derivative = derivative(left), derivative(right)
    iterations = 0
    if left_derivative <= 0:
        prior, optimum = left, 'lower_boundary'
    elif right_derivative >= 0:
        prior, optimum = right, 'upper_boundary'
    else:
        optimum = 'interior'
        for iterations in range(1, cfg.maximum_bisections + 1):
            middle = .5 * (left + right)
            if derivative(middle) > 0:
                left = middle
            else:
                right = middle
            if right - left <= cfg.prior_tolerance:
                break
        prior = .5 * (left + right)
    mixture = b + prior * delta
    return prior, dict(optimum=optimum, iterations=iterations,
                       derivative_at_lower=left_derivative, derivative_at_upper=right_derivative,
                       derivative_at_solution=derivative(prior), final_bracket=[left, right],
                       average_log_likelihood=float(np.mean(np.log(mixture))),
                       negative_average_second_derivative=float(np.mean((delta / mixture)**2)))


def density(scores, weights, edges, centers, cfg):
    hist = np.histogram(scores, bins=edges, weights=weights)[0].astype(np.float64)
    hist /= hist.sum()
    kernel = np.exp(-.5 * ((centers[:, None] - centers[None, :]) / cfg.bandwidth)**2)
    kernel /= kernel.sum(axis=0, keepdims=True)
    mass = (np.einsum('ij,j->i', kernel, hist, optimize=False) + cfg.uniform_mass / cfg.bins) / (1 + cfg.uniform_mass)
    return mass / (edges[1] - edges[0])


def predict(q, r, coverage, base, cfg=Config()):
    started = time.perf_counter()
    if (cfg.bins < 8 or cfg.bandwidth <= 0 or cfg.uniform_mass <= 0 or not .5 < cfg.purity <= 1
            or cfg.minimum_reference_samples < 2 or cfg.prior_tolerance <= 0
            or cfg.maximum_bisections < 1 or cfg.identifiability_tolerance <= 0):
        raise ValueError('Invalid prior-shift configuration')
    q, r = unit(q), unit(r)
    coverage = np.asarray(coverage, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if (base.ndim != 2 or base.size < 3 or coverage.shape != base.shape or base.size != len(q)
            or len(r) != len(q) or q.shape[1] != r.shape[1] or not np.isfinite(base).all()
            or not np.isfinite(coverage).all() or coverage.min() < 0 or coverage.max() > 1):
        raise ValueError('Aligned finite feature/mask/base inputs required')
    fields = dict(field=base.copy(), balanced_control=base.copy(), reference_prior_control=base.copy(),
                  margin_control=base.copy())
    info = dict(config=asdict(cfg), abstention=True, query_gt_used=False, new_encoder_forwards=0,
                real_gain='unmeasured', complete_dataset_minutes='unmeasured')
    foreground = coverage.ravel() >= cfg.purity
    background = (1 - coverage.ravel()) >= cfg.purity
    if min(int(foreground.sum()), int(background.sum())) < cfg.minimum_reference_samples:
        info['reason'] = 'missing_reference_foreground_or_background_samples'
    else:
        means = []
        for take, weights in ((foreground, coverage.ravel()), (background, 1 - coverage.ravel())):
            vector = np.average(r[take], axis=0, weights=weights[take])
            norm = np.linalg.norm(vector)
            means.append(None if norm < 1e-8 else vector / norm)
        if any(vector is None for vector in means):
            info['reason'] = 'degenerate_reference_centroid'
        else:
            direction = means[0] - means[1]
            reference_score = np.clip(np.einsum('nd,d->n', r, direction, optimize=False), -2, 2)
            query_score = np.clip(np.einsum('nd,d->n', q, direction, optimize=False), -2, 2)
            edges = np.linspace(-2, 2, cfg.bins + 1)
            centers = .5 * (edges[1:] + edges[:-1])
            f = density(reference_score[foreground], coverage.ravel()[foreground], edges, centers, cfg)
            b = density(reference_score[background], 1 - coverage.ravel()[background], edges, centers, cfg)
            f_query, b_query = np.interp(query_score, centers, f), np.interp(query_score, centers, b)
            lower, upper = 1 / base.size, 1 - 1 / base.size
            reference_prior = float(np.clip(coverage.mean(), lower, upper))

            def posterior(prior):
                return (prior * f_query / (prior * f_query + (1 - prior) * b_query)).reshape(base.shape)

            fields['balanced_control'] = posterior(.5)
            fields['reference_prior_control'] = posterior(reference_prior)
            fields['margin_control'] = np.clip(.5 + .25 * query_score.reshape(base.shape), 0, 1)
            info.update(reference_prior=reference_prior, foreground_samples=int(foreground.sum()),
                        background_samples=int(background.sum()))
            if np.max(np.abs(f_query - b_query) / np.maximum(f_query, b_query)) <= cfg.identifiability_tolerance:
                info['reason'] = 'query_likelihood_flat_in_prior'
            else:
                prior, fit = fit_prior(f_query, b_query, lower, upper, cfg)
                fields['field'] = posterior(prior)
                info.update(abstention=False, query_prior=float(prior), fit=fit)
    info['wall_seconds'] = time.perf_counter() - started
    return dict(**fields, info=info)
