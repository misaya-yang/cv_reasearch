"""Fixed same-whole-image-context removal response revision0, independent count0.

This measures directions between normalized original/removal representations;
it does not rename absolute crop CLS or claim semantic causality.
"""
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
from PIL import Image

from .object_crop_cls import unit, reference_coverage, region_pixels, margin, render
from .pro_message_extrapolation import ward_partitions


CHANNELS = ('cls', 'patch_all', 'patch_masked')


@dataclass(frozen=True)
class Config:
    view_side: int = 128
    partition_counts: tuple = (4, 8, 16, 32)
    reference_background_regions: int = 8
    neutral_rgb: tuple = (124, 116, 104)
    purity: float = .9
    cosine_temperature: float = .07
    differential_weight: float = .05
    minimum_response_norm: float = 1e-6


def contextual_views(rgb, selected, cfg=Config()):
    """Whole original image is the fixed common crop; only selected pixels change."""
    rgb, selected = np.asarray(rgb), np.asarray(selected, bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or selected.shape != rgb.shape[:2]:
        raise ValueError('Aligned original RGB and legal removal mask required')
    h, w = selected.shape
    factor = cfg.view_side/max(h, w)
    sh, sw = max(1, round(h*factor)), max(1, round(w*factor))
    oy, ox = (cfg.view_side-sh)//2, (cfg.view_side-sw)//2
    edited = rgb.copy()
    edited[selected] = cfg.neutral_rgb
    canvases = []
    for image in (rgb, edited):
        canvas = np.empty((cfg.view_side, cfg.view_side, 3), np.uint8)
        canvas[:] = cfg.neutral_rgb
        canvas[oy:oy+sh, ox:ox+sw] = np.asarray(Image.fromarray(image).resize((sw, sh), Image.Resampling.BILINEAR))
        canvases.append(canvas)
    coverage = np.zeros((cfg.view_side, cfg.view_side), np.float64)
    coverage[oy:oy+sh, ox:ox+sw] = np.asarray(Image.fromarray(selected.astype(np.float32)).resize(
        (sw, sh), Image.Resampling.BILINEAR), np.float64)
    alpha = coverage.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel()
    outside_unchanged = np.array_equal(edited[~selected], rgb[~selected])
    if not outside_unchanged:
        raise ValueError('Complete natural outside-context must remain unchanged')
    return *canvases, alpha, dict(original_hw=[h, w], crop_tlbr=[0, 0, h, w],
        resized_hw=[sh, sw], padding_top_left=[oy, ox], physical_aspect_preserved=True,
        outside_region_original_pixels_identical=outside_unchanged,
        selected_original_pixels=int(selected.sum()), labeled_query_pixels=0,
        ROI_view_coverage_sum=float(coverage.sum()), ROI_patch_coverage_sum=float(alpha.sum()),
        actual_view_changed_pixel_count=int(np.any(canvases[0] != canvases[1], axis=-1).sum()),
        original_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        selected_sha256=hashlib.sha256(selected.tobytes()).hexdigest())


def describe(tokens, coverage):
    tokens = np.asarray(tokens, np.float64)
    coverage = np.asarray(coverage, np.float64)
    if tokens.ndim != 2 or tokens.shape[0] != 69 or coverage.shape != (64,) or coverage.sum() <= 0:
        raise ValueError('69 actual CLS/register/patch tokens and aligned geometric coverage required')
    patches = unit(tokens[5:])
    return dict(cls=unit(tokens[0]), patch_all=unit(patches.mean(0)),
                patch_masked=unit(np.average(patches, axis=0, weights=coverage)))


def response(original, removed, cfg=Config()):
    delta = unit(original)-unit(removed)
    norm = float(np.linalg.norm(delta))
    if norm <= cfg.minimum_response_norm:
        return None, norm
    return delta/norm, norm


def fields_from_scores(base, partitions, scores, native_regions, native_point, cfg=Config()):
    base = np.asarray(base, np.float64)
    native = np.zeros(base.size)
    votes = {name: np.zeros(base.size) for name in scores}
    for scale in cfg.partition_counts:
        seen = np.zeros(base.size, np.int64)
        for ids in partitions[scale]:
            key = tuple(map(int, ids))
            native[ids] += np.tanh(native_regions[key]/cfg.cosine_temperature)/len(cfg.partition_counts)
            for name in scores:
                # Invalid response abstains locally; never substitute a guessed direction.
                value = scores[name][key]
                if value is None:
                    value = native_regions[key]
                votes[name][ids] += np.tanh(value/cfg.cosine_temperature)/len(cfg.partition_counts)
            seen[ids] += 1
        if not np.all(seen == 1):
            raise ValueError('Every token must have one vote at each fixed Ward scale')
    fields = {'mean': base.copy()}
    for name, vote in votes.items():
        fields[name] = base+cfg.differential_weight*(vote-native).reshape(base.shape)
    fields['native_roi_mean'] = base+cfg.differential_weight*(native-np.tanh(
        np.asarray(native_point)/cfg.cosine_temperature)).reshape(base.shape)
    bound = max(float(np.max(np.abs(field-base))) for field in fields.values())
    if bound > 2*cfg.differential_weight+1e-12:
        raise RuntimeError('Fixed complete-readout bound violated')
    return fields, dict(maximum_absolute_edit=bound, exact_edit_bound=.1,
        high_confidence_tokens_outside_0_4_to_0_6_protected=True, probability_calibration=False)


def predict(reference_rgb, full_reference_mask, query_rgb, q, r, base, encoder, *, producer_binding, cfg=Config()):
    """RGB/fullR-mask/legalWard -> six complete fields/masks; no Q labels accepted."""
    if cfg != Config() or not producer_binding.get('producer'):
        raise ValueError('Fixed revision and explicit actual encoder/native binding required')
    started = time.perf_counter()
    q, r, base = unit(q), unit(r), np.asarray(base, np.float64)
    reference_rgb, query_rgb, mask = np.asarray(reference_rgb), np.asarray(query_rgb), np.asarray(full_reference_mask, bool)
    if base.shape != (64, 64) or q.shape != r.shape or len(q) != base.size or mask.shape != reference_rgb.shape[:2]:
        raise ValueError('Aligned native64 features, same MEAN and original full reference mask required')
    coverage = reference_coverage(mask).ravel()
    names = ('delta_cls', 'delta_patch_all', 'delta_patch_masked', 'zero_original_cls')
    if not mask.any() or mask.all() or not np.any(coverage <= .1):
        fields = {name: base.copy() for name in ('mean', *names, 'native_roi_mean')}
        return dict(fields=fields, masks={name: dict(zip(('work', 'original'), render(value, query_rgb.shape[:2])))
            for name, value in fields.items()}, info=dict(abstention=True, new_encoder_forwards=0,
            query_gt_used=False, independent_new_method_count=0, reason='missing_known_reference_classes',
            wall_seconds=time.perf_counter()-started))
    pure = coverage >= cfg.purity
    if not pure.any():
        pure = coverage == coverage.max()
    foreground = unit(np.average(r[pure], axis=0, weights=coverage[pure]))
    partitions, query_ward = ward_partitions(q, base.shape, cfg.partition_counts)
    bg_partitions, bg_ward = ward_partitions(r, base.shape, (8,), coverage <= .1)
    bg_regions = bg_partitions[8]
    native_bg = unit(np.stack([r[ids].mean(0) for ids in bg_regions]))
    native_point = margin(q, foreground, native_bg)
    cache, calls, pairs = {}, [], []
    cache_requests = 0

    def encode(view, role):
        nonlocal cache_requests
        cache_requests += 1
        digest = hashlib.sha256(view.tobytes()).hexdigest()
        if digest in cache:
            return cache[digest], digest
        before = time.perf_counter()
        tokens = np.asarray(encoder(view), np.float64)
        if tokens.ndim != 2 or tokens.shape[0] != 69 or not np.isfinite(tokens).all():
            raise ValueError('Actual final norm prefix+patch tokens required')
        cache[digest] = tokens
        calls.append(dict(role=role, seconds=time.perf_counter()-before, view_size=[128, 128],
                          view_sha256=digest, new_image_forwards=1))
        return tokens, digest

    def pair(rgb, selected, role):
        original, removed, alpha, geometry = contextual_views(rgb, selected, cfg)
        if alpha.sum() <= 0:
            pairs.append(dict(role=role, geometric_zeroROI=True, response_norms={channel: 0. for channel in CHANNELS}, **geometry))
            return {channel: None for channel in CHANNELS}, None, {channel: 0. for channel in CHANNELS}
        a, ah = encode(original, role+'.original')
        b, bh = encode(removed, role+'.removed')
        ad, bd = describe(a, alpha), describe(b, alpha)
        directions, norms = {}, {}
        for channel in CHANNELS:
            directions[channel], norms[channel] = response(ad[channel], bd[channel], cfg)
        pairs.append(dict(role=role, original_view_sha256=ah, removed_view_sha256=bh,
                          response_norms=norms, **geometry))
        return directions, ad, norms

    rfg, roriginal, fg_norms = pair(reference_rgb, mask, 'reference_FG')
    rbg = []
    for index, ids in enumerate(bg_regions):
        selected = region_pixels(ids, mask.shape) & ~mask
        if selected.any():
            rbg.append(pair(reference_rgb, selected, f'reference_BG_{index}')[0])
    banks = {channel: [row[channel] for row in rbg if row[channel] is not None] for channel in CHANNELS}
    channel_active = {channel: rfg[channel] is not None and len(banks[channel]) > 0 for channel in CHANNELS}
    scores = {name: {} for name in names}
    query_descriptors, native_regions, descriptor_rows = {}, {}, []
    for scale in cfg.partition_counts:
        for slot, ids in enumerate(partitions[scale]):
            key = tuple(map(int, ids))
            if key in query_descriptors:
                continue
            directions, original, norms = pair(query_rgb, region_pixels(ids, query_rgb.shape[:2]), f'query_K{scale}_ROI{slot}')
            query_descriptors[key] = directions
            native_regions[key] = float(margin(unit(q[ids].mean(0)), foreground, native_bg))
            for channel in CHANNELS:
                valid = channel_active[channel] and directions[channel] is not None
                scores['delta_'+channel][key] = float(margin(directions[channel], rfg[channel],
                    np.stack(banks[channel]))) if valid else None
            # All matched no-removal R views have identical whole-image CLS: margin exactly0.
            scores['zero_original_cls'][key] = 0. if original is not None and roriginal is not None else None
            direction_cosines = {channel: float(np.sum(directions['cls']*directions[channel]))
                if directions['cls'] is not None and directions[channel] is not None else None
                for channel in ('patch_all', 'patch_masked')}
            descriptor_rows.append(dict(region_ids=list(key), response_norms=norms,
                direction_cosines_with_CLS=direction_cosines,
                scores={name: scores[name][key] for name in names}))
    fields, bound = fields_from_scores(base, partitions, scores, native_regions, native_point, cfg)
    masks = {name: dict(zip(('work', 'original'), render(field, query_rgb.shape[:2]))) for name, field in fields.items()}
    info = dict(revision=0, independent_new_method_count=0, query_gt_used=False, config=asdict(cfg),
        producer_binding=producer_binding, complete_context='entire original image, one common physical128 canvas per R/Q',
        query_logical_ROIs=60, unique_query_ROIs=len(query_descriptors), known_BG_views=len(rbg),
        logical_image_requests=2+1+len(rbg)+60, new_encoder_forwards=len(calls), original_views_reused=True,
        query_member_reuse=60-len(query_descriptors), encoder_token_cache_requests=cache_requests,
        exact_RGB_hash_reuse=cache_requests-len(calls),
        newly_encoded_original_views=sum(call['role'].endswith('.original') for call in calls),
        newly_encoded_removed_views=sum(call['role'].endswith('.removed') for call in calls),
        channel_active=channel_active, reference_FG_response_norms=fg_norms,
        zero_response_threshold=cfg.minimum_response_norm, pairs=pairs, calls=calls,
        original_CLS_margin_identically_zero=True, query_ward=query_ward, background_ward=bg_ward,
        bound=bound, wall_seconds=time.perf_counter()-started, abstention=False,
        semantic_causality_claim=False, generalized_quality_claim=False)
    return dict(fields=fields, masks=masks, info=info, region_scores=descriptor_rows)


class ActualRemoval128Encoder:
    def __init__(self, wrapper):
        from .object_crop_cls import ActualCLS128Encoder
        bound = ActualCLS128Encoder(wrapper, 'cpu')
        self.model = bound.model

    def __call__(self, view):
        import torch
        image = torch.from_numpy(np.asarray(view, np.uint8).copy()).permute(2, 0, 1).float()/255
        mean, std = image.new_tensor((.485, .456, .406))[:, None, None], image.new_tensor((.229, .224, .225))[:, None, None]
        with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
            tokens = self.model.forward_features(((image-mean)/std)[None])
        if tuple(tokens.shape) != (1, 69, 1024) or tokens.dtype != torch.float32:
            raise ValueError('Actual frozen Eva128 final-LN69x1024 interface required')
        return tokens[0].detach().numpy().astype(np.float64)
