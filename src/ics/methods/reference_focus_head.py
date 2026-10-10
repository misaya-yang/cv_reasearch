"""Matched real/derived reference-focus binary-role readouts on frozen O24.

One human reference annotation supplies whole and focus observations. Exact
inverse-overlap pixel weights preserve full-reference FG/BG area mass before
balanced binary-role quadrature. Real and derived focus features share every
label, physical weight, sampled ID and loss term. This is an added-observation
experiment, not occupancy reconstruction, a calibrated probability, or a claim
of an original segmentation method.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PER_ROLE_OCCURRENCES = 128
RIDGE_STRENGTH = .01
RAW_GRID = (64, 64)
QUERY_GRID = (128, 128)
FOCUS_SIZE = 512
CORNER_BOXES = ((0, 0, 512, 512), (512, 0, 1024, 512),
                (0, 512, 512, 1024), (512, 512, 1024, 1024))


def _raw(value: Any, name: str, channels: int | None = None) -> torch.Tensor:
    x = torch.as_tensor(value)
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2
            or x.shape[0] != 4096 or not x.shape[1]):
        raise ValueError(f'{name} must be CPU FP32 [4096, channels] actual O24')
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f'{name} feature channels differ')
    norm = torch.linalg.vector_norm(x, dim=1)
    if not bool(torch.isfinite(x).all()) or not bool((torch.isfinite(norm) & (norm > 0)).all()):
        raise ValueError(f'{name} must have finite nonzero raw token norms')
    return x.detach()


def _box(value: Any) -> tuple[int, int, int, int]:
    if len(value) != 4 or any(not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError('focus_box_xyxy must contain four integer coordinates')
    x0, y0, x1, y1 = map(int, value)
    if (not (0 <= x0 <= 512 and 0 <= y0 <= 512)
            or x1-x0 != FOCUS_SIZE or y1-y0 != FOCUS_SIZE or x0 % 16 or y0 % 16):
        raise ValueError('Require one16-aligned512 focus box inside the1024 reference canvas')
    return x0, y0, x1, y1


def _mask(value: Any) -> torch.Tensor:
    mask = torch.as_tensor(value)
    if mask.device.type != 'cpu' or tuple(mask.shape) != (1024, 1024):
        raise ValueError('Require lawful CPU reference_mask_canvas1024 [1024,1024]')
    if not bool(torch.isfinite(mask).all()) or not bool(((mask >= 0) & (mask <= 1)).all()):
        raise ValueError('Reference mask must be finite in [0,1]')
    return mask.detach().double()


@torch.inference_mode()
def _representation(raw: torch.Tensor, apply_apd: bool,
                    projection: torch.Tensor | None) -> torch.Tensor:
    # Preserve prior study's F.normalize and input strides. In particular,
    # whole64->128 interpolation is channel-strided before normalization.
    unit = F.normalize(raw, dim=1)
    if apply_apd:
        projected = unit @ projection.T
        norm = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        if not bool((torch.isfinite(norm) & (norm > 0)).all()):
            raise ValueError('Shared APD has a nonfinite or zero projected token')
        unit = F.normalize(projected, dim=1)
    return unit


@torch.inference_mode()
def _derived_focus(whole: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    x0, y0, x1, y1 = box
    channels = whole.shape[1]
    grid = whole.reshape(64, 64, channels).permute(2, 0, 1)[None]
    cropped = grid[:, :, y0//16:y1//16, x0//16:x1//16]
    return F.interpolate(cropped, size=RAW_GRID, mode='bilinear', align_corners=False
                         )[0].permute(1, 2, 0).reshape(4096, channels).contiguous()


@torch.inference_mode()
def _reference_measure(mask: torch.Tensor, box: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = box
    whole_c = mask.reshape(64, 16, 64, 16).mean((1, 3))
    crop_mask = mask[y0:y1, x0:x1]
    # Crop512->model1024 by nearest mask resizing has exact8x8 physical areas.
    focus_c = crop_mask.reshape(64, 8, 64, 8).mean((1, 3))
    whole_area = torch.full((64, 64), 256., dtype=torch.float64)
    whole_area[y0//16:y1//16, x0//16:x1//16] = 128.
    focus_area = torch.full((64, 64), 32., dtype=torch.float64)
    coverage = torch.cat((whole_c.reshape(-1), focus_c.reshape(-1)))
    physical_area = torch.cat((whole_area.reshape(-1), focus_area.reshape(-1)))
    foreground = physical_area*coverage
    background = physical_area*(1-coverage)
    pixel_fg, pixel_bg = mask.sum(), (1-mask).sum()
    if not float(pixel_fg) > 0 or not float(pixel_bg) > 0:
        raise ValueError('Both full-reference roles must have positive area')
    fg_error, bg_error = (foreground.sum()-pixel_fg).abs(), (background.sum()-pixel_bg).abs()
    if max(float(fg_error), float(bg_error)) > 1e-8:
        raise RuntimeError('Inverse-overlap reference role masses do not close')
    return dict(coverage=coverage, physical_area=physical_area,
        foreground=foreground, background=background,
        diagnostics=dict(reference_footprints='whole16x16, focus8x8 canonical pixels',
            inverse_overlap_area='whole outside256, inside128; focus32',
            full_reference_foreground_pixels=float(pixel_fg),
            full_reference_background_pixels=float(pixel_bg),
            pooled_foreground_pixel_mass=float(foreground.sum()),
            pooled_background_pixel_mass=float(background.sum()),
            foreground_mass_absolute_error=float(fg_error), background_mass_absolute_error=float(bg_error),
            full_reference_foreground_fraction=float(pixel_fg/mask.numel()),
            pooled_foreground_fraction=float(foreground.sum()/physical_area.sum()),
            whole_weighted_area=float(whole_area.sum()), focus_weighted_area=float(focus_area.sum()),
            geometry_does_not_establish_visual_purity=True))


def _role_quadrature(measure: dict) -> dict:
    """128 equal-risk occurrences per physical-area-weighted role.

    A mixed observed token may occur in both roles. Collapsing repeated token
    IDs gives a_j=(nF+nB)/256 and y_j=(nF-nB)/(nF+nB). This is exactly the
    empirical binary-role risk, not selected-coverage regression or independent
    validation. It does not give extra statistical confidence to duplicate views.
    """
    occurrence_ids = {}
    for role, name in (('foreground', 'foreground'), ('background', 'background')):
        mass = measure[role].numpy()
        cumulative = np.cumsum(mass, dtype=np.float64)
        positions = (np.arange(PER_ROLE_OCCURRENCES, dtype=np.float64)+.5)*cumulative[-1]/PER_ROLE_OCCURRENCES
        ids = np.searchsorted(cumulative, positions, side='left')
        if (ids < 0).any() or (ids >= len(mass)).any():
            raise RuntimeError('Reference physical-role quantile index failure')
        occurrence_ids[name] = ids
    ids = np.unique(np.r_[occurrence_ids['foreground'], occurrence_ids['background']])
    n = len(measure['coverage'])
    fg_count = np.bincount(occurrence_ids['foreground'], minlength=n)[ids]
    bg_count = np.bincount(occurrence_ids['background'], minlength=n)[ids]
    total_count = fg_count+bg_count
    if fg_count.sum() != PER_ROLE_OCCURRENCES or bg_count.sum() != PER_ROLE_OCCURRENCES or (total_count <= 0).any():
        raise RuntimeError('Balanced reference occurrence counts do not close')
    normalization = 2*PER_ROLE_OCCURRENCES
    weights = torch.from_numpy(total_count.astype(np.float64)/normalization)
    target = torch.from_numpy((fg_count-bg_count).astype(np.float64)/total_count)
    fg_risk = torch.from_numpy(fg_count.astype(np.float64)/normalization)
    bg_risk = torch.from_numpy(bg_count.astype(np.float64)/normalization)
    # Balanced role counts make the analytical weighted target mean exactly0.
    target_mean = float((weights*target).sum())
    if abs(target_mean) > 1e-12:
        raise RuntimeError('Balanced binary-role target mean does not close')
    return dict(ids=torch.from_numpy(ids), weights=weights, target=target,
        foreground_risk=fg_risk, background_risk=bg_risk,
        collapse_constant=(weights*(1-target.square())).sum(),
        diagnostics=dict(sample_ids=ids.tolist(), sample_count=len(ids),
            foreground_occurrence_ids=occurrence_ids['foreground'].tolist(),
            background_occurrence_ids=occurrence_ids['background'].tolist(),
            foreground_occurrence_counts=fg_count.tolist(), background_occurrence_counts=bg_count.tolist(),
            foreground_occurrences=int(fg_count.sum()), background_occurrences=int(bg_count.sum()),
            whole_selected_tokens=int((ids < 4096).sum()), focus_selected_tokens=int((ids >= 4096).sum()),
            cross_role_aliased_tokens=int(((fg_count > 0) & (bg_count > 0)).sum()),
            total_risk_weight=float(weights.sum()), foreground_risk_weight=float(fg_risk.sum()),
            background_risk_weight=float(bg_risk.sum()), numerical_weighted_target_mean=target_mean,
            analytical_weighted_target_mean=0., collapse_constant=float((weights*(1-target.square())).sum()),
            empirical_risk='sum nF/256(s-1)^2 + nB/256(s+1)^2; exact after id collapse',
            sampling='128 midpoint physical-area role-mass quantiles per role over whole then focus',
            quadrature_is_not_independent_validation=True))


@torch.inference_mode()
def _fit(features: torch.Tensor, measure: dict, sample: dict) -> dict:
    x = features[sample['ids']].double()
    a, y = sample['weights'], sample['target']
    mean_x = (a[:, None]*x).sum(0)/a.sum()
    centered_x = x-mean_x
    matrix = centered_x @ centered_x.T+torch.diag(RIDGE_STRENGTH/a)
    dual = torch.linalg.solve(matrix, y)
    coefficient = centered_x.T @ dual
    # Analytical target mean0 prevents numerical constant fields choosing a role.
    bias = -mean_x @ coefficient
    fitted = x @ coefficient+bias
    residual = fitted-y
    binary_loss = (sample['foreground_risk']*(fitted-1).square()
                   +sample['background_risk']*(fitted+1).square()).sum()
    collapsed = (a*residual.square()).sum()+sample['collapse_constant']
    eigenvalues = torch.linalg.eigvalsh(matrix)
    normal_residual = matrix @ dual-y
    gradient = centered_x.T @ (a*residual)+RIDGE_STRENGTH*coefficient
    full_score = features.double() @ coefficient+bias
    fg = measure['foreground']/measure['foreground'].sum()
    bg = measure['background']/measure['background'].sum()
    full_loss = .5*(fg*(full_score-1).square()+bg*(full_score+1).square()).sum()
    if not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(bias)):
        raise RuntimeError('Nonfinite reference-focus role solution')
    return dict(coefficient=coefficient, bias=bias,
        diagnostics=dict(dual_condition_number=float(eigenvalues[-1]/eigenvalues[0]),
            dual_relative_residual=float(normal_residual.norm()/y.norm().clamp_min(1e-15)),
            coefficient_stationarity_l2=float(gradient.norm()),
            intercept_stationarity=float((a*residual).sum()),
            sample_binary_role_loss=float(binary_loss),
            full_physical_binary_role_loss=float(full_loss),
            binary_collapsed_loss_absolute_error=float((binary_loss-collapsed).abs()),
            coefficient_l2=float(coefficient.norm()), bias=float(bias),
            source_loss_scope='same correlated reference whole/focus views; no query or independent holdout'))


@torch.inference_mode()
def _query_representations(whole: torch.Tensor, corners: list[torch.Tensor],
                           apply_apd: bool, projection: torch.Tensor | None) -> dict:
    channels = whole.shape[1]
    global_raw = F.interpolate(whole.reshape(64, 64, channels).permute(2, 0, 1)[None],
        size=QUERY_GRID, mode='bilinear', align_corners=False
        )[0].permute(1, 2, 0).reshape(16384, channels)
    global_unit = _representation(global_raw, apply_apd, projection)
    fine = torch.empty((128, 128, channels), dtype=torch.float32)
    for corner, (x0, y0, x1, y1) in zip(corners, CORNER_BOXES):
        fine[y0//8:y1//8, x0//8:x1//8] = _representation(corner, apply_apd, projection).reshape(64, 64, channels)
    return dict(global_features=global_unit, local4_features=fine.reshape(16384, channels))


def _field_diagnostic(field: torch.Tensor) -> dict:
    return dict(minimum=float(field.min()), maximum=float(field.max()), mean=float(field.mean()),
                positive_fraction=float((field > 0).double().mean()))


@torch.inference_mode()
def fit_predict(reference_whole_raw: Any, reference_focus_raw: Any,
                reference_mask_canvas1024: Any, focus_box_xyxy: Any,
                query_whole_raw: Any, native4_raw: Any, *, apply_apd: bool,
                projection: Any = None) -> dict:
    """Complete signed actual/derived fields with identical lawful supervision.

    The caller supplies the actual newly encoded focus O24 and the same original
    whole/native4 raw arrays used by existing200 episodes. Its legal mask selects
    the fixed focus box outside this module. apply_apd and projection preserve one
    original whole-query branch for all R/Q views; there is no view selector.

    fields['actual.equal'] is the fixed candidate. global/local4 are observation
    controls; fields['derived.equal'] is the same-label same-input-budget
    interpolated-whole-reference control. All fields are128x128 CPU FP32 signed
    role scores with strict zero decision. They are not foreground occupancy or
    calibrated probabilities. No query GT, FoRIS mask, renderer or CRF enters.
    """
    started = time.perf_counter()
    whole = _raw(reference_whole_raw, 'reference_whole_raw')
    channels = whole.shape[1]
    focus = _raw(reference_focus_raw, 'reference_focus_raw', channels)
    query = _raw(query_whole_raw, 'query_whole_raw', channels)
    if len(native4_raw) != 4:
        raise ValueError('Require exactly NW/NE/SW/SE actual native query quadrants')
    corners = [_raw(value, f'native4_raw[{i}]', channels) for i, value in enumerate(native4_raw)]
    if not isinstance(apply_apd, (bool, np.bool_)):
        raise ValueError('apply_apd must be the caller fixed boolean branch')
    if apply_apd:
        projection = torch.as_tensor(projection)
        if (projection.device.type != 'cpu' or projection.dtype != torch.float32
                or tuple(projection.shape) != (channels, channels)
                or not bool(torch.isfinite(projection).all())):
            raise ValueError('Shared active APD requires finite CPU FP32 [channels,channels] projection')
    box = _box(focus_box_xyxy)
    measure = _reference_measure(_mask(reference_mask_canvas1024), box)
    sample = _role_quadrature(measure)
    source_started = time.perf_counter()
    whole_features = _representation(whole, bool(apply_apd), projection)
    actual_features = torch.cat((whole_features, _representation(focus, bool(apply_apd), projection)))
    derived_raw = _derived_focus(whole, box)
    derived_features = torch.cat((whole_features, _representation(derived_raw, bool(apply_apd), projection)))
    fits = {name: _fit(features, measure, sample)
            for name, features in (('actual', actual_features), ('derived', derived_features))}
    fit_seconds = time.perf_counter()-source_started
    query_started = time.perf_counter()
    observations = _query_representations(query, corners, bool(apply_apd), projection)
    query_seconds = time.perf_counter()-query_started
    readout_started = time.perf_counter()
    fields, field_diagnostics = {}, {}
    for name, fit in fits.items():
        global_score = observations['global_features'].double() @ fit['coefficient']+fit['bias']
        local_score = observations['local4_features'].double() @ fit['coefficient']+fit['bias']
        for view, score in (('global', global_score), ('local4', local_score),
                            ('equal', .5*(global_score+local_score))):
            if not bool(torch.isfinite(score).all()):
                raise RuntimeError('Nonfinite reference-focus query field')
            fields[f'{name}.{view}'] = score.float().reshape(128, 128).numpy()
            field_diagnostics[f'{name}.{view}'] = _field_diagnostic(score)
    return dict(fields=fields, diagnostics=dict(
        method='real_reference_focus_balanced_role_vs_derived_reference',
        status='fixed added-observation experiment; no originality or performance assertion',
        fixed_default='actual.equal', matched_control='derived.equal',
        constants=dict(per_role_occurrences=PER_ROLE_OCCURRENCES, ridge_strength=RIDGE_STRENGTH,
            focus_size=FOCUS_SIZE, query_grid_hw=QUERY_GRID),
        input=dict(feature_channels=channels, actual_raw_arrays=7,
            focus_box_xyxy=box, apply_apd=bool(apply_apd),
            source_order='whole4096 then focus4096; same physical-view IDs for both fits',
            representation='norm raw tokens then same caller APD projection and norm; Qwhole raw64->128 before norm',
            derived_focus='whole raw32x32 exact16-aligned crop, bilinear32->64 align_corners=False; no encoder'),
        physical_measure=measure['diagnostics'], quadrature=sample['diagnostics'],
        fits={name: fit['diagnostics'] for name, fit in fits.items()}, fields=field_diagnostics,
        actual_minus_derived={view: dict(
            maximum_absolute_difference=float(np.max(np.abs(fields[f'actual.{view}']-fields[f'derived.{view}']))),
            sign_disagreement_fraction=float(np.mean((fields[f'actual.{view}'] > 0) != (fields[f'derived.{view}'] > 0))))
            for view in ('global', 'local4', 'equal')},
        score_semantics='balanced binary-role squared risk; not area occupancy or calibrated probability',
        source_pixel_prior_preserved_before_role_balance=True,
        query_labels_used=False, segmentation_pseudolabels_used=False, FoRIS_fields_used=False,
        new_encoder_calls_in_head=0, raw_writes=0, sensor_inversion=False, graph_optimization=False,
        decision='strict signed zero; fixed equal views, no per-dataset selection or reference threshold',
        source_processing_and_fit_seconds=fit_seconds, query_processing_seconds=query_seconds,
        readout_seconds=time.perf_counter()-readout_started, total_head_seconds=time.perf_counter()-started))
