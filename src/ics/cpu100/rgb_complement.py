"""Five bounded RGB observations that complement a direct DINO prototype.

All reference labels are known reference labels. No query label, host score,
class name, external image, or encoder is accepted. Missing actual RGB or the
complete reference mask is an unavailable input, not a successful fallback.
Synthetic checks establish possible action, never natural-data improvement.
"""
from __future__ import annotations

from functools import partial
import time

import numpy as np
from PIL import Image

from .common import Episode, Result, prototype_margin, validate
from ics.methods.direct_dino_features import unit


class RGBUnavailable(ValueError):
    """Physical RGB/reference-mask inputs were not supplied."""


RGB_SIDE = 256
CORRECTION = 0.5
BLOCK = 128
TRIPLETS = (
    ((1, 0), (1, 0)), ((0, 1), (0, 1)),
    ((1, 0), (0, 1)), ((1, 0), (0, -1)),
    ((1, 1), (1, 0)), ((1, 1), (0, 1)),
    ((1, -1), (1, 0)), ((1, -1), (0, -1)),
    ((2, 0), (-1, 1)), ((0, 2), (1, -1)),
    ((2, 1), (-1, 0)), ((1, 2), (0, -1)),
)


def _require(ep):
    validate(ep)
    if ep.q_rgb is None or ep.r_rgb is None or ep.reference_mask is None:
        raise RGBUnavailable("unavailable: actual Q/R uint8 RGB and complete known R mask required")
    for rgb, geom, role in ((ep.q_rgb, ep.query_geometry, "query"),
                            (ep.r_rgb, ep.reference_geometry, "reference")):
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8 or min(a.shape[:2]) < 1:
            raise ValueError(f"{role}: original uint8 H,W,3 physical RGB required")
        if geom and tuple(geom.get("original_hw", a.shape[:2])) != a.shape[:2]:
            raise ValueError(f"{role}: original RGB/geometry shape mismatch")
    mask = np.asarray(ep.reference_mask)
    if mask.shape != np.shape(ep.r_rgb)[:2] or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("Complete binary reference mask must align with original R RGB")
    if max(*ep.q_hw, *ep.r_hw) > 64:
        raise ValueError("CPU contract supports native feature grids up to 64 per axis")
    return ep


def _canvas(array, geometry, *, labels=False):
    """Preserve recorded physical rectangle; never reinterpret a letterbox."""
    a = np.asarray(array)
    if labels:
        a = (a != 0).astype(np.uint8)
    if geometry:
        view = int(geometry["view_side"])
        sh, sw = map(int, geometry["resized_hw"])
        oy, ox = map(int, geometry["padding_top_left"])
        if min(sh, sw) <= 0 or min(oy, ox) < 0 or max(oy + sh, ox + sw) > view:
            raise ValueError("Invalid recorded physical RGB transform")
        # All current native views are 128/1024, exactly compatible with 256.
        target = np.asarray((sh, sw, oy, ox)) * (RGB_SIDE / view)
        if not np.allclose(target, np.rint(target), atol=1e-12):
            raise ValueError("RGB256 would round the recorded physical transform; explicit adapter required")
        sh, sw, oy, ox = map(int, np.rint(target))
    else:
        sh = sw = RGB_SIDE
        oy = ox = 0
    interp = Image.Resampling.NEAREST if labels else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(a).resize((sw, sh), interp))
    if labels:
        out = np.full((RGB_SIDE, RGB_SIDE), -1, np.int8)
    else:
        out = np.full((RGB_SIDE, RGB_SIDE, 3), (124, 116, 104), np.uint8)
    out[oy:oy + sh, ox:ox + sw] = resized
    return out


def _token_means(canvas, hw):
    h, w = hw
    if RGB_SIDE % h or RGB_SIDE % w:
        raise ValueError("RGB256 and native grid must have exact integer cell alignment")
    return (canvas.reshape(h, RGB_SIDE // h, w, RGB_SIDE // w, 3)
            .mean(axis=(1, 3)).reshape(h * w, 3) / 255.)


def _windows(canvas, hw, width, *, labels=False):
    h, w = hw
    cy = (np.arange(h) + .5) * RGB_SIDE / h - .5
    cx = (np.arange(w) + .5) * RGB_SIDE / w - .5
    sy, sx = cy - (width - 1) / 2, cx - (width - 1) / 2
    if not np.allclose(sy, np.rint(sy)) or not np.allclose(sx, np.rint(sx)):
        raise ValueError("Patch window centers would require a new interpolation adapter")
    pad = width
    padding = ((pad, pad), (pad, pad)) + (((0, 0),) if canvas.ndim == 3 else ())
    if labels:
        padded = np.pad(canvas, padding, constant_values=-1)
    else:
        padded = np.pad(canvas, padding, mode="reflect")
    # This view is not copied; descriptor functions copy only bounded blocks.
    sliding = np.lib.stride_tricks.sliding_window_view(padded, (width, width), axis=(0, 1))
    rows, cols = np.meshgrid(np.rint(sy).astype(int) + pad,
                             np.rint(sx).astype(int) + pad, indexing="ij")
    result = sliding[rows.ravel(), cols.ravel()]
    return np.moveaxis(result, 1, -1) if canvas.ndim == 3 else result


def _prepared(ep, width=16):
    _require(ep)
    qr = _canvas(ep.q_rgb, ep.query_geometry)
    rr = _canvas(ep.r_rgb, ep.reference_geometry)
    mask = _canvas(ep.reference_mask, ep.reference_geometry, labels=True)
    rw = _windows(rr, ep.r_hw, width)
    qw = _windows(qr, ep.q_hw, width)
    labels = _windows(mask, ep.r_hw, width, labels=True)
    fg = np.all(labels == 1, axis=(1, 2))
    bg = np.all(labels == 0, axis=(1, 2))
    return rw, qw, fg, bg, qr, rr


def _class_advantage(r, q, fg, bg, *, hellinger=False, minimum=4):
    if fg.sum() < minimum or bg.sum() < minimum:
        return np.zeros(len(q)), dict(abstention="insufficient_pure_R_windows",
                                      R_FG_windows=int(fg.sum()), R_BG_windows=int(bg.sum()))
    mf, mb = r[fg].mean(0), r[bg].mean(0)
    if hellinger:
        mf, mb = np.sqrt(np.maximum(mf, 0)), np.sqrt(np.maximum(mb, 0))
        q = np.sqrt(np.maximum(q, 0))
    if np.linalg.norm(mf - mb) <= 1e-12:
        return np.zeros(len(q)), dict(abstention="R_observation_class_means_equal")
    df = ((q - mf) ** 2).sum(1)
    db = ((q - mb) ** 2).sum(1)
    h = np.divide(db - df, db + df, out=np.zeros(len(q)), where=db + df > 1e-12)
    return h, dict(R_FG_windows=int(fg.sum()), R_BG_windows=int(bg.sum()))


def _result(ep, h, info, method, started, *, rgb_only=False):
    base = prototype_margin(ep)
    h = np.asarray(h, float).reshape(ep.q_hw)
    h = np.where(ep.q_valid.reshape(ep.q_hw) > 0, h, 0.)
    margin = h.copy() if rgb_only else base + CORRECTION * h
    if ep.wf.sum() == 0:
        margin[:] = -1.
    elif ep.wb.sum() == 0:
        margin[:] = 1.
    info.update(method=method, query_GT_used=False, new_encoder_calls=0,
                output_semantics="complete signed margin, zero threshold; not probability",
                input_type="actual RGB plus labeled R and supplied frozen DINO grid",
                source_id=ep.source_id, DINO_producer=ep.producer,
                correction_cap=CORRECTION, RGB_side=RGB_SIDE,
                valid_added_tokens=int(((margin > 0) & (base <= 0) & (ep.q_valid.reshape(ep.q_hw) > 0)).sum()),
                valid_deleted_tokens=int(((margin <= 0) & (base > 0) & (ep.q_valid.reshape(ep.q_hw) > 0)).sum()),
                wall_seconds=time.perf_counter() - started,
                natural_segmentation_benefit="unmeasured")
    return Result(margin=margin, info=info)


def _weighted_median(x, weights):
    keep = weights > 0
    if not keep.any():
        return np.zeros(np.asarray(x).shape[1])
    x, weights = np.asarray(x)[keep], np.asarray(weights)[keep]
    out = []
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        cum = np.cumsum(weights[order])
        out.append(x[order[np.searchsorted(cum, weights.sum() / 2, side="left")], j])
    return np.asarray(out)


def _laplace(x, weights):
    center = _weighted_median(x, weights)
    spread = np.maximum(_weighted_median(np.abs(x - center), weights), 1 / 255)
    return center, spread


def _laplace_logp(x, parameters):
    center, spread = parameters
    return -(np.abs(x - center) / spread + np.log(2 * spread)).sum(1)


def radiometric_color(ep, arm="primary"):
    started = time.perf_counter()
    _require(ep)
    qr = _token_means(_canvas(ep.q_rgb, ep.query_geometry), ep.q_hw)
    rr = _token_means(_canvas(ep.r_rgb, ep.reference_geometry), ep.r_hw)
    base = prototype_margin(ep).ravel()
    anchors = (base <= -.25) & (ep.q_valid >= .9)
    info = dict(Q_background_anchors=int(anchors.sum()),
                nuisance_assumption="shared diagonal photometric transform and comparable BG composition")
    aligned = qr.copy()
    if anchors.sum() >= 8 and ep.wb.sum() > 0:
        rmedian, rmad = _laplace(rr, ep.wb)
        qmedian, qmad = _laplace(qr[anchors], np.ones(anchors.sum()))
        scale = np.clip(rmad / qmad, .5, 2.)
        aligned = (qr - qmedian) * scale + rmedian
        info.update(nuisance_fitted=True, nuisance_scale=scale.tolist())
    else:
        info.update(nuisance_fitted=False, nuisance_abstention="fewer_than_8_Q_BG_anchors")
    pf, pb = _laplace(rr, ep.wf), _laplace(rr, ep.wb)
    if arm == "unaligned":
        values = qr
    else:
        values = aligned
    if arm in ("primary", "prototype", "unaligned", "rgb_only"):
        df = ((values - pf[0]) ** 2).sum(1)
        db = ((values - pb[0]) ** 2).sum(1)
        h = np.divide(db - df, db + df, out=np.zeros(len(qr)), where=db + df > 1e-12)
    else:
        h = np.tanh((_laplace_logp(values, pf) - _laplace_logp(values, pb)) / 3)
    if np.linalg.norm(pf[0] - pb[0]) < 1e-12 and np.linalg.norm(pf[1] - pb[1]) < 1e-12:
        h[:] = 0
        info["abstention"] = "R_RGB_class_likelihoods_equal"
    return _result(ep, h, info, "RGB01." + arm, started, rgb_only=arm == "rgb_only")


_ordinal_codes = np.arange(256, dtype=np.uint16)
_ordinal_min = np.minimum.reduce([((_ordinal_codes >> j) | (_ordinal_codes << (8 - j))) & 255
                                  for j in range(8)])
_ordinal_lookup = np.searchsorted(np.unique(_ordinal_min), _ordinal_min)
_ordinal_bins = len(np.unique(_ordinal_min))


def _gray(patches):
    return patches.astype(np.float64).mean(-1) / 255.


def ordinal_descriptor(patches):
    out = np.empty((len(patches), _ordinal_bins), float)
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
              (1, 1), (1, 0), (1, -1), (0, -1))
    for start in range(0, len(patches), BLOCK):
        gray = _gray(patches[start:start + BLOCK])
        center = gray[:, 1:-1, 1:-1]
        codes = np.zeros(center.shape, np.uint16)
        for bit, (dy, dx) in enumerate(shifts):
            neighbor = gray[:, 1 + dy:gray.shape[1] - 1 + dy,
                            1 + dx:gray.shape[2] - 1 + dx]
            codes |= (neighbor >= center).astype(np.uint16) << bit
        bins = _ordinal_lookup[codes].reshape(len(gray), -1)
        index = bins + np.arange(len(gray))[:, None] * _ordinal_bins
        hist = np.bincount(index.ravel(), minlength=len(gray) * _ordinal_bins).reshape(len(gray), -1)
        out[start:start + len(gray)] = hist / bins.shape[1]
    return out


def color_histogram(patches):
    out = np.empty((len(patches), 24), float)
    for start in range(0, len(patches), BLOCK):
        p = patches[start:start + BLOCK].reshape(-1, patches.shape[1] ** 2, 3)
        rows = np.arange(len(p))[:, None] * 8
        out[start:start + len(p)] = np.concatenate([
            np.bincount((p[..., channel] // 32 + rows).ravel(), minlength=len(p) * 8)
            .reshape(len(p), 8) / p.shape[1] for channel in range(3)], axis=1)
    return out


def power_descriptor(patches, arm="full"):
    width = patches.shape[1]
    frequency = np.fft.fftfreq(width)
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    bands = [(radius > lo) & (radius <= hi) for lo, hi in
             ((0, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, np.inf))]
    dim = width * width if arm == "full" else 4
    out = np.empty((len(patches), dim), float)
    for start in range(0, len(patches), BLOCK):
        gray = _gray(patches[start:start + BLOCK])
        centered = gray - gray.mean((1, 2), keepdims=True)
        power = np.abs(np.fft.fft2(centered, axes=(-2, -1))) ** 2
        total = power.sum((1, 2))
        fractions = np.divide(power, total[:, None, None], out=np.zeros_like(power),
                              where=total[:, None, None] > 1e-12)
        if arm == "full":
            value = fractions.reshape(len(gray), -1)
        else:
            value = np.stack([fractions[:, band].sum(1) for band in bands], axis=1)
        out[start:start + len(gray)] = value
    return out


def bispectral_descriptor(patches):
    out = np.empty((len(patches), 2 * len(TRIPLETS)), float)
    width = patches.shape[1]
    for start in range(0, len(patches), BLOCK):
        gray = _gray(patches[start:start + BLOCK])
        centered = gray - gray.mean((1, 2), keepdims=True)
        rotated = []
        for rotation in range(4):
            spectrum = np.fft.fft2(np.rot90(centered, rotation, axes=(-2, -1)), axes=(-2, -1))
            entries = []
            for k, l in TRIPLETS:
                a = spectrum[:, k[0] % width, k[1] % width]
                b = spectrum[:, l[0] % width, l[1] % width]
                c = spectrum[:, (k[0] + l[0]) % width, (k[1] + l[1]) % width]
                triple = a * b * np.conj(c)
                magnitude = np.abs(a) * np.abs(b) * np.abs(c)
                normalized = np.divide(triple, magnitude, out=np.zeros_like(triple), where=magnitude > 1e-10)
                entries.extend((normalized.real, normalized.imag))
            rotated.append(np.stack(entries, axis=1))
        out[start:start + len(gray)] = np.mean(rotated, axis=0)
    return out


def ordinal_texture(ep, arm="primary"):
    started = time.perf_counter()
    rw, qw, fg, bg, _, _ = _prepared(ep)
    if arm == "color":
        descriptor, hellinger = color_histogram, True
    elif arm in ("full_power", "radial_power"):
        descriptor, hellinger = partial(power_descriptor, arm="full" if arm == "full_power" else "radial"), True
    elif arm == "variance":
        descriptor, hellinger = lambda p: _gray(p).var((1, 2))[:, None], False
    else:
        descriptor, hellinger = ordinal_descriptor, True
    h, info = _class_advantage(descriptor(rw), descriptor(qw), fg, bg, hellinger=hellinger)
    return _result(ep, h, info, "RGB02." + arm, started)


def phase_texture(ep, arm="primary"):
    started = time.perf_counter()
    rw, qw, fg, bg, _, _ = _prepared(ep)
    descriptor = (partial(power_descriptor, arm="full") if arm == "full_power"
                  else ordinal_descriptor if arm == "ordinal" else bispectral_descriptor)
    h, info = _class_advantage(descriptor(rw), descriptor(qw), fg, bg,
                              hellinger=arm in ("full_power", "ordinal"))
    info["phase_invariance_scope"] = "periodic translation cancels triple phase; 90-degree rotations averaged"
    return _result(ep, h, info, "RGB03." + arm, started)


def patch_descriptor(patches):
    values = patches.astype(np.float64) / 255.
    means = values.mean((1, 2))
    centered = (values - means[:, None, None]).reshape(len(values), -1)
    return np.concatenate((unit(centered), means / 4.), axis=1)


def _farthest_indices(values, maximum=64):
    if len(values) <= maximum:
        return np.arange(len(values))
    selected = [0]
    distance = ((values - values[0]) ** 2).sum(1)
    for _ in range(1, maximum):
        nxt = int(np.argmax(distance))
        if distance[nxt] <= 1e-12:
            break
        selected.append(nxt)
        distance = np.minimum(distance, ((values - values[nxt]) ** 2).sum(1))
    return np.asarray(selected)


def _nearest_distance(query, dictionary):
    out = np.empty(len(query))
    norms = (dictionary ** 2).sum(1)
    for start in range(0, len(query), BLOCK):
        q = query[start:start + BLOCK]
        distances = ((q ** 2).sum(1)[:, None] + norms[None, :]
                     - 2 * np.einsum("id,jd->ij", q, dictionary, optimize=False))
        out[start:start + len(q)] = np.maximum(distances, 0).min(1)
    return out


def patch_dictionary(ep, arm="primary"):
    started = time.perf_counter()
    rw, qw, fg, bg, _, _ = _prepared(ep, width=8)
    if arm in ("color", "full_power", "ordinal", "variance"):
        descriptor = (color_histogram if arm == "color" else ordinal_descriptor if arm == "ordinal"
                      else partial(power_descriptor, arm="full") if arm == "full_power"
                      else lambda p: _gray(p).var((1, 2))[:, None])
        h, info = _class_advantage(descriptor(rw), descriptor(qw), fg, bg,
                                  hellinger=arm != "variance")
        return _result(ep, h, info, "RGB04." + arm, started)
    r, q = patch_descriptor(rw), patch_descriptor(qw)
    info = dict(R_FG_windows=int(fg.sum()), R_BG_windows=int(bg.sum()))
    if not fg.any() or not bg.any():
        info["abstention"] = "missing_pure_R_class_windows"
        h = np.zeros(len(q))
    else:
        f, b = r[fg], r[bg]
        if arm == "mean":
            f, b = f.mean(0, keepdims=True), b.mean(0, keepdims=True)
        else:
            f, b = f[_farthest_indices(f)], b[_farthest_indices(b)]
        df, db = _nearest_distance(q, f), _nearest_distance(q, b)
        h = np.divide(db - df, db + df, out=np.zeros(len(q)), where=db + df > 1e-12)
        info.update(FG_dictionary_size=len(f), BG_dictionary_size=len(b))
    return _result(ep, h, info, "RGB04." + arm, started)


def _feature_neighborhoods(values, hw):
    h, w = hw
    if min(h, w) < 3:
        raise ValueError("3x3 relation observer requires at least 3 tokens per axis")
    a = values.reshape(h, w, -1)
    padded = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
    return np.moveaxis(windows, 2, -1).reshape(h * w, 9, -1)


def coupling_descriptor(features, colors, hw, *, uncoupled=False, permuted=False):
    h, w = hw
    if min(h, w) < 3:
        raise ValueError("3x3 relation observer requires at least 3 tokens per axis")
    ids = np.arange(h * w).reshape(h, w)
    padded_ids = np.pad(ids, ((1, 1), (1, 1)), mode="reflect")
    index = np.lib.stride_tricks.sliding_window_view(padded_ids, (3, 3)).reshape(h * w, 9)
    a, b = np.triu_indices(9, 1)
    out = np.empty((len(features), 16 if uncoupled else 3))
    for start in range(0, len(features), 32):
        window_index = index[start:start + 32]
        f, c = features[window_index], colors[window_index]
        gram = np.einsum("bid,bjd->bij", f, f, optimize=False)
        fd = np.maximum(1 - gram[:, a, b], 0)
        if permuted:
            c = c[:, (4, 0, 8, 2, 6, 1, 7, 3, 5)]
        cd = ((c[:, a] - c[:, b]) ** 2).sum(2)
        if uncoupled:
            rows = np.arange(len(f))[:, None] * 8
            fh = np.bincount((np.minimum((fd / 2 * 8).astype(int), 7) + rows).ravel(),
                             minlength=len(f) * 8).reshape(len(f), 8) / 36
            ch = np.bincount((np.minimum((cd / 3 * 8).astype(int), 7) + rows).ravel(),
                             minlength=len(f) * 8).reshape(len(f), 8) / 36
            value = np.concatenate((fh, ch), axis=1)
        else:
            x, y = fd - fd.mean(1, keepdims=True), cd - cd.mean(1, keepdims=True)
            denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
            corr = np.divide((x * y).sum(1), denominator,
                             out=np.zeros(len(f)), where=denominator > 1e-12)
            value = np.column_stack((corr, np.median(fd, axis=1), np.median(cd, axis=1)))
        out[start:start + len(f)] = value
    return out


def neighborhood_coupling(ep, arm="primary"):
    started = time.perf_counter()
    _require(ep)
    rr = _token_means(_canvas(ep.r_rgb, ep.reference_geometry), ep.r_hw)
    qr = _token_means(_canvas(ep.q_rgb, ep.query_geometry), ep.q_hw)
    kwargs = dict(uncoupled=arm == "uncoupled", permuted=arm == "permuted")
    r = coupling_descriptor(ep.r, rr, ep.r_hw, **kwargs)
    q = coupling_descriptor(ep.q, qr, ep.q_hw, **kwargs)
    wf = _feature_neighborhoods(ep.wf[:, None], ep.r_hw)[..., 0]
    wb = _feature_neighborhoods(ep.wb[:, None], ep.r_hw)[..., 0]
    valid = _feature_neighborhoods(ep.wvalid[:, None], ep.r_hw)[..., 0]
    fg = np.all((wf >= .95) & (valid >= .95), axis=1)
    bg = np.all((wb >= .95) & (valid >= .95), axis=1)
    h, info = _class_advantage(r, q, fg, bg, hellinger=arm == "uncoupled")
    qvalid = _feature_neighborhoods(ep.q_valid[:, None], ep.q_hw)[..., 0]
    h[~np.all(qvalid >= .95, axis=1)] = 0
    return _result(ep, h, info, "RGB05." + arm, started)


METHODS = {
    "RGB01": radiometric_color,
    "RGB02": ordinal_texture,
    "RGB03": phase_texture,
    "RGB04": patch_dictionary,
    "RGB05": neighborhood_coupling,
}

CONTROLS = {
    "RGB01.unaligned": partial(radiometric_color, arm="unaligned"),
    "RGB01.prototype": partial(radiometric_color, arm="prototype"),
    "RGB01.laplace": partial(radiometric_color, arm="laplace"),
    "RGB01.rgb_only": partial(radiometric_color, arm="rgb_only"),
    "RGB02.color": partial(ordinal_texture, arm="color"),
    "RGB02.variance": partial(ordinal_texture, arm="variance"),
    "RGB02.full_power": partial(ordinal_texture, arm="full_power"),
    "RGB02.radial_power": partial(ordinal_texture, arm="radial_power"),
    "RGB03.full_power": partial(phase_texture, arm="full_power"),
    "RGB03.ordinal": partial(phase_texture, arm="ordinal"),
    "RGB04.mean": partial(patch_dictionary, arm="mean"),
    "RGB04.color": partial(patch_dictionary, arm="color"),
    "RGB04.variance": partial(patch_dictionary, arm="variance"),
    "RGB04.full_power": partial(patch_dictionary, arm="full_power"),
    "RGB04.ordinal": partial(patch_dictionary, arm="ordinal"),
    "RGB05.uncoupled": partial(neighborhood_coupling, arm="uncoupled"),
    "RGB05.permuted": partial(neighborhood_coupling, arm="permuted"),
}
