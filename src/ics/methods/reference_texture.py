"""Reference-supervised RGB radial power fractions, fixed CPU candidate.

All query patches are evaluated. Only the supplied reference mask labels
training patches. No encoder, query labels, class IDs, or object proposals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from PIL import Image


METHOD_ID = 'reference_rgb_radial_power_texture_v1'
FIELD_KEYS = ('field', 'color_histogram_control', 'variance_control', 'peak_control', 'zero_control', 'standalone_control')


@dataclass(frozen=True)
class Config:
    rgb_size: int = 256
    field_size: int = 64
    windows: tuple = (16, 32)
    radial_cuts: tuple = (1/16, 1/8, 1/4)
    color_bins: int = 8
    minimum_reference_patches: int = 8
    maximum_correction: float = .125
    prototype_zero_tolerance: float = 1e-12
    fft_batch: int = 256


def _fixed(cfg):
    expected = asdict(Config())
    actual = asdict(cfg)
    actual['windows'], actual['radial_cuts'] = tuple(actual['windows']), tuple(actual['radial_cuts'])
    if actual != expected:
        raise ValueError('Fixed single recipe; parameter search is not supported')


def canonical_rgb(rgb, cfg=Config()):
    """Existing PIL RGB -> canonical1024 bilinear -> RGB256 bilinear."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8:
        raise ValueError('Nonempty uint8 original RGB required')
    image = Image.fromarray(rgb, 'RGB').resize((1024, 1024), Image.Resampling.BILINEAR)
    return np.asarray(image.resize((cfg.rgb_size, cfg.rgb_size), Image.Resampling.BILINEAR)).copy()


def canonical_mask(mask, reference_shape, cfg=Config()):
    mask = np.asarray(mask)
    if mask.shape != tuple(reference_shape) or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError('Complete binary reference mask must match original reference RGB')
    image = Image.fromarray((mask != 0).astype(np.uint8)*255).resize((1024, 1024), Image.Resampling.NEAREST)
    return np.asarray(image.resize((cfg.rgb_size, cfg.rgb_size), Image.Resampling.NEAREST)) != 0


def _windows(array, width, *, mask=False):
    # 64-grid centers at (4*i+1.5,4*j+1.5), aligned with half-pixel resizing.
    pad = width//2-2
    padding = ((pad, pad), (pad, pad)) + (((0, 0),) if array.ndim == 3 else ())
    if mask:
        padded = np.pad(array.astype(np.int8), padding, mode='constant', constant_values=-1)
    else:
        padded = np.pad(array, padding, mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (width, width), axis=(0, 1))[::4, ::4]
    return np.moveaxis(windows, 2, -1) if array.ndim == 3 else windows


def radial_masks(width, cfg=Config()):
    f = np.fft.fftfreq(width)
    radius2 = f[:, None]**2+f[None, :]**2
    non_dc = radius2 > 0
    result = []
    lower = 0.
    for upper in (*cfg.radial_cuts, np.inf):
        result.append(non_dc & (radius2 > lower**2) & (radius2 <= upper**2))
        lower = upper
    return np.asarray(result)


def patch_power(patches, cfg=Config()):
    """SUM power in each ring / SUM all non-DC power, never mean/bin-count."""
    patches = np.asarray(patches, dtype=np.float64)
    width = patches.shape[-1]
    if patches.ndim != 3 or patches.shape[1:] != (width, width) or width not in cfg.windows:
        raise ValueError('Fixed square patch widths required')
    centered = patches-patches.mean((1, 2), keepdims=True)
    spectrum = np.fft.fft2(centered, axes=(-2, -1))
    power = spectrum.real**2+spectrum.imag**2
    power[:, 0, 0] = 0
    total = power.sum((1, 2))
    constant = patches.max((1, 2)) == patches.min((1, 2))
    total[constant] = 0
    ring_power = np.stack([power[:, ring].sum(1) for ring in radial_masks(width, cfg)], axis=1)
    fraction = np.divide(ring_power, total[:, None], out=np.zeros_like(ring_power), where=total[:, None] > 0)
    variance = total/(width**4)
    index = fraction.argmax(1)  # Lowest band index on an exact tie, including flat patches.
    peak = np.column_stack((index/3., fraction[np.arange(len(patches)), index]))
    return fraction, variance, peak


def descriptors(rgb256, cfg=Config()):
    rgb = np.asarray(rgb256)
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError('Canonical uint8 RGB256 required')
    rgb = rgb.astype(np.float64)/255.
    gray = rgb.mean(2)
    n = cfg.field_size**2
    texture, variance, peak, color = [], [], [], []
    for width in cfg.windows:
        gray_view, rgb_view = _windows(gray, width), _windows(rgb, width)
        fractions = np.empty((n, 4)); values = np.empty((n, 1)); peaks = np.empty((n, 2))
        histograms = np.empty((n, 24))
        for start in range(0, n, cfg.fft_batch):
            ids = np.arange(start, min(start+cfg.fft_batch, n))
            rows, columns = ids//64, ids % 64
            fractions[ids], values[ids, 0], peaks[ids] = patch_power(gray_view[rows, columns], cfg)
            colors = rgb_view[rows, columns]
            bins = np.minimum((colors*cfg.color_bins).astype(np.int64), cfg.color_bins-1)
            for channel in range(3):
                histogram = np.stack([(bins[..., channel] == index).mean((1, 2))
                                      for index in range(cfg.color_bins)], axis=1)
                histograms[ids, channel*cfg.color_bins:(channel+1)*cfg.color_bins] = histogram
        texture.append(fractions); variance.append(values); peak.append(peaks); color.append(histograms)
    return dict(texture=np.concatenate(texture, 1), variance=np.concatenate(variance, 1),
                peak=np.concatenate(peak, 1), color=np.mean(color, axis=0))


def pure_reference_patches(mask256, cfg=Config()):
    fg = np.ones((64, 64), dtype=bool)
    bg = np.ones_like(fg)
    for width in cfg.windows:
        view = _windows(np.asarray(mask256), width, mask=True)
        fg &= np.all(view == 1, axis=(-2, -1))
        bg &= np.all(view == 0, axis=(-2, -1))
    return fg.ravel(), bg.ravel()


def prototype_contrast(reference, query, foreground, background, cfg=Config()):
    mf, mb = reference[foreground].mean(0), reference[background].mean(0)
    separation = float(np.sum((mf-mb)**2))
    info = dict(foreground_prototype=mf.tolist(), background_prototype=mb.tolist(),
                squared_prototype_separation=separation)
    if separation <= cfg.prototype_zero_tolerance:
        info['abstention'] = True
        return np.zeros(len(query)), info
    df, db = np.sum((query-mf)**2, axis=1), np.sum((query-mb)**2, axis=1)
    g = np.divide(db-df, db+df, out=np.zeros_like(db), where=db+df > 0)
    info.update(abstention=False, contrast_min=float(g.min()), contrast_max=float(g.max()))
    return g, info


def render(field, original_shape):
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))[None, None]
    work = functional.interpolate(tensor, (1024, 1024), mode='bilinear', align_corners=False) > .5
    original = functional.interpolate(work.float(), tuple(original_shape), mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


def predict(reference_rgb, reference_mask, query_rgb, base, *, original_shape=None, cfg=Config()):
    started = time.perf_counter()
    _fixed(cfg)
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (64, 64) or not np.isfinite(base).all():
        raise ValueError('Bound finite MEAN64 required')
    original_shape = tuple(np.asarray(query_rgb).shape[:2]) if original_shape is None else tuple(original_shape)
    if len(original_shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in original_shape):
        raise ValueError('Positive integer original query H/W required')
    rr, qr = canonical_rgb(reference_rgb, cfg), canonical_rgb(query_rgb, cfg)
    mask = canonical_mask(reference_mask, np.asarray(reference_rgb).shape[:2], cfg)
    fields = {key: base.copy() for key in FIELD_KEYS}
    info = dict(config=asdict(cfg), method_id=METHOD_ID, independent_method_increment=1,
                query_gt_used=False, new_encoder_forwards=0, real_segmentation_gain='unmeasured',
                ring_definition='ring SUM power / all non-DC SUM power; not mean frequency density',
                rotation_scope='radial fractions of a fixed RGB256 patch; upstream uint8 PIL interpolation may change rotated pixels',
                peak_control='per window max radial power fraction: band index/3 and fraction; smallest index ties',
                color_control='mean of the same two normalized 3x8 RGB histograms, 24 dimensions',
                reference_purity='both full windows in image and entirely known FG or BG; no cov64 approximation',
                standalone_contract='.5+.5*main g if main prototype is separated; otherwise explicit base fallback',
                standalone_abstention=True,
                abstention=True, original_shape=list(original_shape))
    fg, bg = pure_reference_patches(mask, cfg)
    info.update(reference_foreground_patches=int(fg.sum()), reference_background_patches=int(bg.sum()))
    if not np.asarray(reference_mask).any():
        fields = {key: np.zeros_like(base) for key in FIELD_KEYS}
        info['reason'] = 'empty_reference'
    elif min(int(fg.sum()), int(bg.sum())) < cfg.minimum_reference_patches:
        info['reason'] = 'insufficient_complete_pure_reference_windows'
    else:
        descriptor_started = time.perf_counter()
        rd, qd = descriptors(rr, cfg), descriptors(qr, cfg)
        info['reference_query_descriptor_seconds'] = time.perf_counter()-descriptor_started
        contrasts = {}
        for key, descriptor in (('field', 'texture'), ('color_histogram_control', 'color'),
                                ('variance_control', 'variance'), ('peak_control', 'peak')):
            g, contrasts[key] = prototype_contrast(rd[descriptor], qd[descriptor], fg, bg, cfg)
            fields[key] = base+cfg.maximum_correction*g.reshape(64, 64)
            if key == 'field' and not contrasts[key]['abstention']:
                fields['standalone_control'] = (.5+.5*g).reshape(64, 64)
        info.update(abstention=contrasts['field']['abstention'], prototypes=contrasts,
                    standalone_abstention=contrasts['field']['abstention'],
                    maximum_absolute_correction=float(np.max(np.abs(fields['field']-base))),
                    added_tokens=int(((fields['field'] > .5) & (base <= .5)).sum()),
                    deleted_tokens=int(((fields['field'] <= .5) & (base > .5)).sum()))
    masks_work, masks_original = {}, {}
    for key, field in fields.items():
        masks_work[key], masks_original[key] = render(field, original_shape)
    info['predict_with_all_renderers_seconds'] = time.perf_counter()-started
    return dict(**fields, masks_work=masks_work, masks_original=masks_original, info=info)
