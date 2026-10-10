"""Compact scene-conditioned reference-role reconstruction on frozen features.

The scene supplies a nonlinear feature basis, never segmentation pseudo-labels.
The annotated reference alone supplies the binary-role risk.  This is a complete
signed readout and a falsifiable transductive-kernel hypothesis, not an asserted
novelty or accuracy result.  No backbone, FoRIS field, query label, threshold fit,
spatial smoother or renderer is used here.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch


TEMPERATURE = .07
RIDGE_STRENGTH = .01
PER_ROLE_SAMPLES = 128
MAX_COMPONENTS = 128
MAX_CANDIDATES = 512
LLOYD_STEPS = 4
OCCUPANCY_PSEUDOCOUNT = .5


def _features(value: Any, name: str) -> torch.Tensor:
    x = torch.as_tensor(value)
    if x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError(f'{name} must be CPU FP32 [tokens, channels]')
    if not x.shape[0] or not x.shape[1] or not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} must have nonempty finite features')
    norms = torch.linalg.vector_norm(x, dim=1)
    if not bool((norms > 0).all()) or float((norms-1).abs().max()) > 1e-3:
        raise ValueError(f'{name} must be caller-normalized unit features')
    return x.detach()


def _grid(grid: Any, n: int, *, infer: bool = False) -> tuple[int, int] | None:
    if grid is None:
        side = math.isqrt(n)
        return (side, side) if infer and side*side == n else None
    if len(grid) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in grid):
        raise ValueError('query_grid_hw must contain two positive integers')
    hw = (int(grid[0]), int(grid[1]))
    if hw[0]*hw[1] != n:
        raise ValueError('query_grid_hw does not match query token count')
    return hw


def _reservoir_ids(n: int, grid: tuple[int, int] | None) -> torch.Tensor:
    """Fixed mask-independent spatial strata; all tokens when n<=512."""
    if n <= MAX_CANDIDATES:
        return torch.arange(n)
    if grid is None:
        return torch.floor((torch.arange(MAX_CANDIDATES, dtype=torch.float64)+.5)
                           * n/MAX_CANDIDATES).long()
    h, w = grid
    ny = min(h, max(1, int(math.sqrt(MAX_CANDIDATES*h/w))))
    nx = min(w, MAX_CANDIDATES//ny)
    yy = torch.floor((torch.arange(ny, dtype=torch.float64)+.5)*h/ny).long()
    xx = torch.floor((torch.arange(nx, dtype=torch.float64)+.5)*w/nx).long()
    return (yy[:, None]*w+xx[None, :]).reshape(-1)


@torch.inference_mode()
def _mixture(cloud: torch.Tensor, grid: tuple[int, int] | None) -> dict:
    """Deterministic spherical FPS then four unlabeled Lloyd updates.

    Duplicate components are removed at initialization. Empty Lloyd components
    retain their previous unit vector; the fixed Jeffreys half-count makes every
    occupancy positive. Final hard occupancies are recomputed after the final
    center update. They are model masses, not foreground-area estimates.
    """
    started = time.perf_counter()
    reservoir = _reservoir_ids(len(cloud), grid)
    sample = cloud[reservoir]
    mean = cloud.mean(0)
    first = int(torch.argmax(sample @ mean))
    selected = [first]
    best_cosine = sample @ sample[first]
    for _ in range(1, min(MAX_COMPONENTS, len(sample))):
        candidate = int(torch.argmin(best_cosine))
        if float(best_cosine[candidate]) >= 1-1e-6:
            break
        selected.append(candidate)
        best_cosine = torch.maximum(best_cosine, sample @ sample[candidate])
    chosen = torch.tensor(selected, dtype=torch.int64)
    centers = sample[chosen].clone()
    empty_history = []
    for _ in range(LLOYD_STEPS):
        assigned = torch.argmax(cloud @ centers.T, dim=1)
        mass = torch.bincount(assigned, minlength=len(centers))
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assigned, cloud)
        norm = torch.linalg.vector_norm(sums, dim=1)
        active = norm > 1e-12
        centers[active] = sums[active]/norm[active, None]
        empty_history.append(int((mass == 0).sum()))
    assigned = torch.argmax(cloud @ centers.T, dim=1)
    mass = torch.bincount(assigned, minlength=len(centers)).double()
    occupancy = (mass+OCCUPANCY_PSEUDOCOUNT)/(len(cloud)+OCCUPANCY_PSEUDOCOUNT*len(centers))
    return dict(centers=centers, occupancy=occupancy, assignment=assigned,
                diagnostics=dict(component_count=len(centers),
                    reservoir_ids=reservoir.tolist(), landmark_ids=reservoir[chosen].tolist(),
                    hard_occupancies=mass.long().tolist(), occupancies=occupancy.tolist(),
                    empty_components_by_step=empty_history,
                    final_empty_components=int((mass == 0).sum()),
                    cloud_tokens=len(cloud), grid_hw=grid,
                    center_unit_norm_max_error=float((centers.norm(dim=1)-1).abs().max()),
                    mixture_seconds=time.perf_counter()-started))


@torch.inference_mode()
def _basis(x: torch.Tensor, mixture: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # Cosines retain shared FP32 feature arithmetic; probability/fit math is FP64.
    logits = (x @ mixture['centers'].T).double()/TEMPERATURE
    occupancy = mixture['occupancy']
    responsibility = torch.softmax(logits+occupancy.log()[None], dim=1)
    return responsibility/occupancy.sqrt()[None], responsibility


def _role_sample(coverage: np.ndarray) -> dict:
    c_full = np.asarray(coverage).reshape(-1)
    if (not len(c_full) or not np.isfinite(c_full).all()
            or (c_full < 0).any() or (c_full > 1).any()):
        raise ValueError('reference_coverage must be finite nonempty values in [0,1]')
    if c_full.sum(dtype=np.float64) <= 0 or (1-c_full).sum(dtype=np.float64) <= 0:
        raise ValueError('Both reference roles must have positive area mass')
    sampled = []
    # Preserve the caller coverage dtype for historical quantile-id parity.
    for role in (c_full, 1-c_full):
        cumulative = np.cumsum(role)
        sampled.extend(np.searchsorted(cumulative, (np.arange(PER_ROLE_SAMPLES)+.5)
                                       * cumulative[-1]/PER_ROLE_SAMPLES, side='left'))
    ids = np.unique(sampled)
    if (ids < 0).any() or (ids >= len(c_full)).any():
        raise ValueError('Reference role quantile sampling produced invalid ids')
    c = torch.from_numpy(c_full[ids].copy()).double()
    positive, negative = c/c.sum(), (1-c)/(1-c).sum()
    weights = .5*(positive+negative)
    target = (positive-negative)/(positive+negative)
    return dict(ids=torch.from_numpy(ids), weights=weights, target=target,
                positive=positive, negative=negative, coverage=c,
                diagnostics=dict(sample_ids=ids.tolist(), sample_count=len(ids),
                    selected_foreground_mass=float(c.sum()),
                    selected_background_mass=float((1-c).sum()),
                    full_foreground_mass=float(c_full.sum(dtype=np.float64)),
                    full_background_mass=float((1-c_full).sum(dtype=np.float64)),
                    coverage_dtype=c_full.dtype.str,
                    target_weighted_mean=float((weights*target).sum()/weights.sum()),
                    collapsed_loss_constant=float((weights*(1-target.square())).sum())))


@torch.inference_mode()
def _ridge(x: torch.Tensor, sample: dict) -> dict:
    """Exact weighted binary-role dual ridge with an unpenalized intercept."""
    x = x.double()
    y, a = sample['target'], sample['weights']
    mean_x = (a[:, None]*x).sum(0)/a.sum()
    mean_y = (a*y).sum()/a.sum()
    centered_x, centered_y = x-mean_x, y-mean_y
    matrix = centered_x @ centered_x.T+torch.diag(RIDGE_STRENGTH/a)
    dual = torch.linalg.solve(matrix, centered_y)
    coefficient = centered_x.T @ dual
    bias = mean_y-mean_x @ coefficient
    pred = x @ coefficient+bias
    residual = pred-y
    eigenvalues = torch.linalg.eigvalsh(matrix)
    normal_residual = matrix @ dual-centered_y
    gradient = centered_x.T @ (a*residual)+RIDGE_STRENGTH*coefficient
    binary_loss = (.5*sample['positive']*(pred-1).square()
                   +.5*sample['negative']*(pred+1).square()).sum()
    collapsed_loss = (a*residual.square()).sum()+sample['diagnostics']['collapsed_loss_constant']
    if not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(bias)):
        raise RuntimeError('Nonfinite context transport ridge solution')
    return dict(coefficient=coefficient, bias=bias, mean_x=mean_x,
                diagnostics=dict(basis_channels=x.shape[1],
                    dual_condition_number=float(eigenvalues[-1]/eigenvalues[0]),
                    dual_relative_residual=float(normal_residual.norm()/centered_y.norm().clamp_min(1e-15)),
                    weighted_stationarity_l2=float(gradient.norm()),
                    intercept_stationarity=float((a*residual).sum()),
                    reference_binary_role_loss=float(binary_loss),
                    binary_collapsed_loss_absolute_error=float((binary_loss-collapsed_loss).abs()),
                    reference_weighted_rmse=float((a*residual.square()).sum().sqrt()),
                    regularized_objective=float(binary_loss+RIDGE_STRENGTH*coefficient.square().sum()),
                    coefficient_l2=float(coefficient.norm()), bias=float(bias)))


@torch.inference_mode()
def _kernel_fit(reference: torch.Tensor, query: torch.Tensor, sample: dict,
                mixture: dict) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    ref_basis, ref_responsibility = _basis(reference, mixture)
    query_basis, query_responsibility = _basis(query, mixture)
    fit = _ridge(ref_basis[sample['ids']], sample)
    query_score = query_basis @ fit['coefficient']+fit['bias']
    reference_score = ref_basis @ fit['coefficient']+fit['bias']
    # These are evidence-coverage diagnostics, never probabilities or readout gates.
    selected_a = ref_responsibility[sample['ids']]
    positive_mass = sample['positive'] @ selected_a
    negative_mass = sample['negative'] @ selected_a
    prior = mixture['occupancy']
    role_values = fit['coefficient']/prior.sqrt()
    diagnostic = dict(mixture=mixture['diagnostics'], fit=fit['diagnostics'],
        reference_score=reference_score.float().tolist(),
        foreground_responsibility_mass=positive_mass.tolist(),
        background_responsibility_mass=negative_mass.tolist(),
        latent_role_values=role_values.tolist(),
        query_assignment_entropy_mean=float(-(query_responsibility
                                             *query_responsibility.clamp_min(1e-300).log()).sum(1).mean()),
        reference_assignment_entropy_mean=float(-(ref_responsibility
                                                 *ref_responsibility.clamp_min(1e-300).log()).sum(1).mean()),
        scene_integrated_squared_role_value=float((prior*role_values.square()).sum()),
        regularizer_identity_error=float(((prior*role_values.square()).sum()
                                         -fit['coefficient'].square().sum()).abs()),
        score_min=float(query_score.min()), score_max=float(query_score.max()),
        score_mean=float(query_score.mean()), positive_token_fraction=float((query_score > 0).double().mean()),
        kernel_fit_seconds=time.perf_counter()-started)
    return query_score.float().numpy(), diagnostic


@torch.inference_mode()
def fit_predict(reference_features: Any, reference_coverage: Any,
                query_features: Any, query_grid_hw: Any = None) -> dict:
    """Return complete signed fields and matched source-only controls.

    Inputs are caller-normalized CPU FP32 reference/query features (e.g. O24
    after one shared APD branch) and exact continuous reference patch coverage.
    fields['scene'] uses an unlabeled query basis. fields['source_kernel'] uses
    the identical mixture/basis/role-risk machinery on unlabeled reference
    features. fields['support_ridge'] fits the identical role risk directly in
    the shared original feature space. All fields use strict zero as decision.

    No per-dataset setting or tunable runtime argument is exposed. Mixture
    occupancy changes feature responsibilities and the function norm; it does
    not impose a foreground size or class-balance constraint on the query.
    reference_score and source fit diagnostics measure reconstruction only,
    never assert source-to-query identity or confidence calibration.
    """
    started = time.perf_counter()
    reference = _features(reference_features, 'reference_features')
    query = _features(query_features, 'query_features')
    if reference.shape[1] != query.shape[1]:
        raise ValueError('Reference and query feature channels must match')
    sample = _role_sample(np.asarray(reference_coverage))
    if np.asarray(reference_coverage).size != len(reference):
        raise ValueError('Reference coverage does not match reference token count')
    query_grid = _grid(query_grid_hw, len(query))
    source_grid = _grid(None, len(reference), infer=True)
    scene_mixture = _mixture(query, query_grid)
    scene_field, scene_diagnostic = _kernel_fit(reference, query, sample, scene_mixture)
    source_mixture = _mixture(reference, source_grid)
    source_field, source_diagnostic = _kernel_fit(reference, query, sample, source_mixture)
    support_started = time.perf_counter()
    support = _ridge(reference[sample['ids']], sample)
    support_field = (query.double() @ support['coefficient']+support['bias']).float().numpy()
    support_reference = (reference.double() @ support['coefficient']+support['bias']).float()
    support_diagnostic = dict(fit=support['diagnostics'], reference_score=support_reference.tolist(),
                             support_fit_seconds=time.perf_counter()-support_started)
    fields = dict(scene=scene_field, source_kernel=source_field, support_ridge=support_field)
    if any(not np.isfinite(value).all() for value in fields.values()):
        raise RuntimeError('Nonfinite autonomous context transport field')
    return dict(fields=fields, diagnostics=dict(
        method='query_basis_binary_role_reconstruction',
        status='unvalidated transductive kernel hypothesis; no originality assertion',
        constants=dict(temperature=TEMPERATURE, ridge_strength=RIDGE_STRENGTH,
            per_role_samples=PER_ROLE_SAMPLES, max_components=MAX_COMPONENTS,
            max_candidates=MAX_CANDIDATES, lloyd_steps=LLOYD_STEPS,
            occupancy_pseudocount=OCCUPANCY_PSEUDOCOUNT),
        input=dict(reference_tokens=len(reference), query_tokens=len(query),
            feature_channels=reference.shape[1], query_grid_hw=query_grid,
            dtype='caller unit FP32; FP32 similarities, FP64 basis and fit',
            implicit_normalization=False, implicit_APD=False),
        sample=sample['diagnostics'], scene=scene_diagnostic,
        source_kernel=source_diagnostic, support_ridge=support_diagnostic,
        decision='strict signed zero; no minmax or reference threshold',
        query_labels_used=False, query_class_size_constraint=False,
        segmentation_pseudolabels_used=False, FoRIS_field_used=False,
        backbone_calls=0, grid_used_for='mask-independent reservoir strata only',
        total_module_seconds=time.perf_counter()-started))
