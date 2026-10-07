"""Legal Ward/isolated128-view CLS revision0; not a new independent observable.

Actual global CLS already exists in diagnose_object_cls_dev241.py. This fixes a
legal full-output revision with matched crop-patch controls and bounded MEAN
edits; neither small resolution nor descriptor rows increase method count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
from PIL import Image


DESCRIPTORS = ('cls', 'patch_all', 'patch_masked')


@dataclass(frozen=True)
class Config:
    view_side: int = 128
    partition_counts: tuple = (4, 8, 16, 32)
    reference_background_regions: int = 8
    neutral_rgb: tuple = (124, 116, 104)
    purity: float = .9
    cosine_temperature: float = .07
    differential_weight: float = .05


def unit(value):
    value = np.asarray(value, np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm <= 1e-12):
        raise ValueError('Finite nonzero vectors required')
    return value/norm


def reference_coverage(mask, shape=(64, 64)):
    resized = np.asarray(Image.fromarray(np.asarray(mask, np.uint8)).resize(
        (1024, 1024), Image.Resampling.NEAREST), np.float64)
    return resized.reshape(shape[0], 1024//shape[0], shape[1], 1024//shape[1]).mean(axis=(1, 3))


def region_pixels(indices, image_shape, shape=(64, 64)):
    mask = np.zeros(int(np.prod(shape)), np.uint8)
    mask[np.asarray(indices, np.int64)] = 1
    return np.asarray(Image.fromarray(mask.reshape(shape)).resize(
        (image_shape[1], image_shape[0]), Image.Resampling.NEAREST)).astype(bool)


def isolated_view(rgb, mask, cfg=Config()):
    """Physical original-image aspect, bbox, bilinear RGB, centered neutral128."""
    rgb, mask = np.asarray(rgb), np.asarray(mask, bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('Aligned uint8 RGB and binary region mask required')
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError('Cannot encode an empty isolated region')
    top, left, bottom, right = int(rows.min()), int(cols.min()), int(rows.max()+1), int(cols.max()+1)
    crop, crop_mask = rgb[top:bottom, left:right].copy(), mask[top:bottom, left:right]
    crop[~crop_mask] = cfg.neutral_rgb
    height, width = crop_mask.shape
    scale = cfg.view_side/max(height, width)
    sh, sw = max(1, round(height*scale)), max(1, round(width*scale))
    resized = np.asarray(Image.fromarray(crop).resize((sw, sh), Image.Resampling.BILINEAR))
    alpha = np.asarray(Image.fromarray(crop_mask.astype(np.float32)).resize(
        (sw, sh), Image.Resampling.BILINEAR), np.float64)
    canvas = np.empty((cfg.view_side, cfg.view_side, 3), np.uint8)
    canvas[:] = cfg.neutral_rgb
    coverage = np.zeros((cfg.view_side, cfg.view_side), np.float64)
    oy, ox = (cfg.view_side-sh)//2, (cfg.view_side-sw)//2
    canvas[oy:oy+sh, ox:ox+sw] = resized
    coverage[oy:oy+sh, ox:ox+sw] = alpha
    patch_coverage = coverage.reshape(8, 16, 8, 16).mean(axis=(1, 3))
    return canvas, patch_coverage, dict(original_bbox_tlbr=[top, left, bottom, right],
                 original_bbox_hw=[height, width], resized_hw=[sh, sw], padding_top_left=[oy, ox],
                 physical_aspect_preserved=True, isolated_original_pixels=int(mask.sum()),
                 labeled_query_pixels=0)


def margin(descriptor, foreground, background):
    descriptor, foreground, background = unit(descriptor), unit(foreground), unit(background)
    foreground_score = np.einsum('...d,d->...', descriptor, foreground, optimize=False)
    background_score = np.max(np.einsum('...d,kd->...k', descriptor, background, optimize=False), axis=-1)
    return foreground_score-background_score


def assemble_fields(base, partitions, region_scores, native_region_scores, native_point_scores,
                    foreground_scores, native_foreground_scores, cfg=Config()):
    """One complete partition per scale gives each token exactly four votes."""
    base = np.asarray(base, np.float64)
    count = base.size
    native = np.zeros(count)
    evidence = {name: np.zeros(count) for name in DESCRIPTORS}
    foreground = np.zeros(count)
    native_foreground = np.zeros(count)
    for scale in cfg.partition_counts:
        visited = np.zeros(count, np.int64)
        for ids in partitions[scale]:
            key = tuple(map(int, ids))
            native[ids] += np.tanh(native_region_scores[key]/cfg.cosine_temperature)/len(cfg.partition_counts)
            for name in DESCRIPTORS:
                evidence[name][ids] += np.tanh(region_scores[name][key]/cfg.cosine_temperature)/len(cfg.partition_counts)
            foreground[ids] += np.tanh(foreground_scores[key]/cfg.cosine_temperature)/len(cfg.partition_counts)
            native_foreground[ids] += np.tanh(native_foreground_scores[key]/cfg.cosine_temperature)/len(cfg.partition_counts)
            visited[ids] += 1
        if not np.all(visited == 1):
            raise ValueError('All query tokens must appear exactly once in every fixed partition')
    fields = {'mean': base.copy()}
    for name in DESCRIPTORS:
        fields[name] = base+cfg.differential_weight*(evidence[name]-native).reshape(base.shape)
    fields['cls_foreground_only'] = base+cfg.differential_weight*(foreground-native_foreground).reshape(base.shape)
    fields['native_roi_mean'] = base+cfg.differential_weight*(native-np.tanh(
        np.asarray(native_point_scores)/cfg.cosine_temperature)).reshape(base.shape)
    maximum = max(float(np.max(np.abs(field-base))) for field in fields.values())
    if maximum > 2*cfg.differential_weight+1e-12:
        raise RuntimeError('Bounded complete-field edit contract violated')
    return fields, dict(maximum_absolute_edit=maximum, exact_edit_bound=2*cfg.differential_weight,
                       no_clamp_or_minmax=True, confident_tokens_outside_0_4_to_0_6_protected=True)


def render(field, original_shape):
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(np.asarray(field, np.float32).copy())[None, None]
    work = functional.interpolate(tensor, (1024, 1024), mode='bilinear', align_corners=False) > .5
    original = functional.interpolate(work.float(), original_shape, mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


def predict(reference_rgb, full_reference_mask, query_rgb, q, r, base, encoder, *, producer_binding, cfg=Config()):
    """Actual RGB/known-R-mask, legal native-feature Ward regions -> full masks.

    encoder(viewRGB128, geometric_patch_coverage) returns actual CLS and pooled
    final-patch descriptors; injected testing encoders must identify themselves.
    No query annotation, class/fold or GT-selected region argument is accepted.
    """
    if cfg != Config() or not isinstance(producer_binding, dict) or not producer_binding.get('producer'):
        raise ValueError('Fixed revision0 config and explicit native/new-view producer binding required')
    from .pro_message_extrapolation import ward_partitions
    started = time.perf_counter()
    q, r, base = unit(q), unit(r), np.asarray(base, np.float64)
    reference_rgb, query_rgb = np.asarray(reference_rgb), np.asarray(query_rgb)
    mask = np.asarray(full_reference_mask, bool)
    if (base.shape != (64, 64) or q.shape != r.shape or len(q) != base.size or
            mask.shape != reference_rgb.shape[:2] or not np.isfinite(base).all()):
        raise ValueError('Aligned64x64 cache/base and full original reference mask required')
    calls, reference_descriptors, views, query_descriptors = [], {}, [], {}
    def abstain(reason):
        fields = {'mean': base.copy()}
        for name in (*DESCRIPTORS, 'cls_foreground_only', 'native_roi_mean'):
            fields[name] = base.copy()
        return dict(fields=fields, masks={name: dict(zip(('work', 'original'), render(value, query_rgb.shape[:2])))
                    for name, value in fields.items()}, info=dict(abstention=True, reason=reason,
                    new_encoder_forwards=0, query_gt_used=False, independent_new_method_count=0, revision=0,
                    producer_binding=producer_binding, wall_seconds=time.perf_counter()-started))
    if not mask.any() or mask.all():
        return abstain('missing_known_reference_class')
    coverage = reference_coverage(mask)
    foreground = coverage.ravel() >= cfg.purity
    if not foreground.any():
        foreground = coverage.ravel() == coverage.max()
    background = (1-coverage).ravel() >= cfg.purity
    if not background.any():
        return abstain('no_pure_reference_background_cell_for_native_control')
    partitions, query_ward = ward_partitions(q, base.shape, cfg.partition_counts)
    bg_regions, bg_ward = ward_partitions(r, base.shape, (cfg.reference_background_regions,), background)
    bg_regions = bg_regions[cfg.reference_background_regions]
    native_foreground = unit(np.average(r[foreground], axis=0, weights=coverage.ravel()[foreground]))
    native_background = unit(np.stack([r[ids].mean(0) for ids in bg_regions]))
    native_point = margin(q, native_foreground, native_background)
    cache = {}

    def encode(rgb, selected, role):
        view, patch_coverage, view_info = isolated_view(rgb, selected, cfg)
        image_hash = hashlib.sha256(view.tobytes()+patch_coverage.tobytes()).hexdigest()
        if image_hash in cache:
            views.append(dict(role=role, reused_exact_view=True, view_sha256=image_hash, **view_info))
            return cache[image_hash]
        tick = time.perf_counter()
        result = encoder(view, patch_coverage)
        result = {name: unit(result[name]) for name in DESCRIPTORS}
        if any(value.ndim != 1 or value.shape != result['cls'].shape for value in result.values()):
            raise ValueError('Actual CLS and crop patch descriptors must share a finite vector space')
        elapsed = time.perf_counter()-tick
        calls.append(dict(role=role, seconds=elapsed, view_size=[128, 128], new_image_forwards=1))
        cache[image_hash] = result
        views.append(dict(role=role, reused_exact_view=False, view_sha256=image_hash, **view_info))
        return result

    reference_descriptors['foreground'] = encode(reference_rgb, mask, 'reference_FG')
    bg = []
    for index, ids in enumerate(bg_regions):
        selected = region_pixels(ids, mask.shape) & ~mask
        if selected.any():
            bg.append(encode(reference_rgb, selected, f'reference_BG_{index}'))
    if not bg:
        raise ValueError('No legal full-mask-negative reference crop')
    banks = {name: np.stack([row[name] for row in bg]) for name in DESCRIPTORS}
    region_scores = {name: {} for name in DESCRIPTORS}
    native_regions, fg_scores, native_fg_scores = {}, {}, {}
    for scale in cfg.partition_counts:
        for slot, ids in enumerate(partitions[scale]):
            key = tuple(map(int, ids))
            if key in query_descriptors:
                continue
            descriptor = encode(query_rgb, region_pixels(ids, query_rgb.shape[:2]), f'query_K{scale}_ROI{slot}')
            query_descriptors[key] = descriptor
            native_descriptor = unit(q[ids].mean(0))
            native_regions[key] = float(margin(native_descriptor, native_foreground, native_background))
            native_fg_scores[key] = float(np.sum(native_descriptor*native_foreground))
            fg_scores[key] = float(np.sum(descriptor['cls']*reference_descriptors['foreground']['cls']))
            for name in DESCRIPTORS:
                region_scores[name][key] = float(margin(descriptor[name], reference_descriptors['foreground'][name], banks[name]))
    fields, bound = assemble_fields(base, partitions, region_scores, native_regions, native_point,
                                    fg_scores, native_fg_scores, cfg)
    masks = {name: dict(zip(('work', 'original'), render(value, query_rgb.shape[:2]))) for name, value in fields.items()}
    logical_images = 1+len(bg)+sum(len(regions) for regions in partitions.values())
    info = dict(config=asdict(cfg), revision=0, independent_new_method_count=0, query_gt_used=False,
                producer_binding=producer_binding, input_information='R RGB/full binary mask,Q RGB/native cache/same MEAN only',
                query_ward=query_ward, background_ward=bg_ward, query_logical_ROIs=60,
                unique_query_ROIs=len(query_descriptors), known_background_views=len(bg),
                descriptor_image_requests=logical_images, new_encoder_forwards=len(calls),
                exact_view_reuse=len(views)-len(calls), calls=calls, views=views, bound=bound,
                wall_seconds=time.perf_counter()-started, abstention=False,
                semantics='heuristic differential evidence, not probability calibration or joint optimality')
    return dict(fields=fields, masks=masks, info=info,
                descriptors=dict(reference_foreground=reference_descriptors['foreground'], reference_background=banks,
                                 query=list(query_descriptors.values()),
                                 query_region_ids=[list(key) for key in query_descriptors]))


class ActualCLS128Encoder:
    """Existing offline Eva: final norm token0, no default pool/head substitution."""
    def __init__(self, wrapper, device='cpu'):
        import torch
        self.model = wrapper.m.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.device = device
        if (type(self.model).__module__ != 'timm.models.eva' or type(self.model).__name__ != 'Eva' or
                self.model.cls_token is None or self.model.num_prefix_tokens != 5):
            raise ValueError('Bind actual Eva CLS index0+four-register interface')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __call__(self, view, patch_coverage):
        import torch
        rgb = torch.from_numpy(np.asarray(view, np.uint8).copy()).permute(2, 0, 1).to(self.device, torch.float32)/255
        mean = torch.tensor((.485, .456, .406), device=self.device)[:, None, None]
        std = torch.tensor((.229, .224, .225), device=self.device)[:, None, None]
        with torch.inference_mode(), torch.autocast(device_type=self.device, enabled=False):
            tokens = self.model.forward_features(((rgb-mean)/std)[None])
        if tuple(tokens.shape) != (1, 69, 1024) or tokens.dtype != torch.float32:
            raise ValueError('Dynamic128 producer must return raw final-normalized69x1024 tokens')
        raw = tokens[0].detach().cpu().numpy().astype(np.float64)
        patches = unit(raw[5:])
        coverage = np.asarray(patch_coverage, np.float64).ravel()
        if coverage.shape != (64,) or coverage.sum() <= 0 or np.any(coverage < 0):
            raise ValueError('Legal view geometric coverage required for matched patch pool')
        return dict(cls=unit(raw[0]), patch_all=unit(patches.mean(0)),
                    patch_masked=unit(np.average(patches, axis=0, weights=coverage)))
