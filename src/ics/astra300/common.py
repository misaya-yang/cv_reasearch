"""Legal Astra inputs, explicit resources, and one continuous physical readout.

This module supplies the A-family B0 only. Other groups own their documented
baseline. Frozen descriptors cannot stand in for missing internal observations
or for an explicitly required host field/renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, fields, replace
import hashlib
from collections.abc import Mapping

import numpy as np
from ics.cpu100.common import Episode as CPU100Episode
from ics.cpu100.common import load_episode as _load_native_episode
from ics.methods.direct_dino_features import resize, sample_grid, unit

EPS = 1e-6


class ArtifactUnavailable(RuntimeError):
    """The exact observation required by a card has not been supplied."""


Unavailable = ArtifactUnavailable


@dataclass(frozen=True)
class Episode(CPU100Episode):
    artifacts: dict = dataclass_field(default_factory=dict)
    provider: object = None


@dataclass
class Result:
    field: np.ndarray | None = None
    threshold: float = 0.
    mask_original: np.ndarray | None = None
    info: dict = dataclass_field(default_factory=dict)

    @property
    def margin(self):
        """Compatibility accessor; never reinterpret probabilities as margins."""
        return None if self.field is None else np.asarray(self.field) - self.threshold


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def readonly(value):
    """Own an immutable copy; caller changes cannot silently change a resource."""
    a = np.array(value, copy=True)
    a.setflags(write=False)
    return a


def as_episode(ep, *, artifacts=None, provider=None):
    if isinstance(ep, Episode) and artifacts is None and provider is None:
        return ep
    values = {f.name: getattr(ep, f.name) for f in fields(CPU100Episode)}
    supplied = getattr(ep, 'artifacts', {}) if artifacts is None else artifacts
    sealed = {name: readonly(value) if isinstance(value, np.ndarray) else value
              for name, value in supplied.items()}
    selected_provider = getattr(ep, 'provider', None) if provider is None else provider
    return Episode(**values, artifacts=sealed, provider=selected_provider)


def load_episode(row):
    """Reuse the strict native loader; extra packs are separate explicit assets.

    Processed host fields remain artifacts, never relabeled as raw DINO tokens.
    Direct array artifacts must be attached by the caller; this loader does not
    deserialize arbitrary files or execute providers from a manifest.
    """
    ep = _load_native_episode(row)
    return as_episode(ep, artifacts=row.get('artifacts', {}))


def validate(ep):
    q, r = np.asarray(ep.q), np.asarray(ep.r)
    if (q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]
            or len(q) != np.prod(ep.q_hw) or len(r) != np.prod(ep.r_hw)):
        raise ValueError('Aligned native descriptor grids required')
    for a in (q, r, ep.wf, ep.wvalid, ep.q_valid):
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite episode input')
    if (np.asarray(ep.wf).shape != (len(r),) or np.asarray(ep.wvalid).shape != (len(r),)
            or np.asarray(ep.q_valid).shape != (len(q),)
            or np.any(ep.wf < 0) or np.any(ep.wf > ep.wvalid)
            or np.any(ep.wvalid < 0) or np.any(ep.wvalid > 1)
            or np.any(ep.q_valid < 0) or np.any(ep.q_valid > 1)):
        raise ValueError('Legal soft coverage and validity required')
    for a in (q, r):
        norms = np.linalg.norm(a, axis=1)
        if np.any((norms > EPS) & (np.abs(norms - 1) > 1e-5)):
            raise ValueError('Native descriptors must be unit-normalized')
    if len(ep.original_shape) != 2 or min(ep.original_shape) < 1:
        raise ValueError('Actual original query H/W required')
    return ep


def artifact(ep, name):
    available = getattr(ep, 'artifacts', {})
    if name in available:
        value = available[name]
    else:
        provider = getattr(ep, 'provider', None)
        if provider is None:
            raise ArtifactUnavailable('Required actual artifact unavailable: ' + name)
        try:
            value = provider(ep, name) if callable(provider) else provider.require(ep, name)
        except (KeyError, NotImplementedError) as error:
            raise ArtifactUnavailable('Required actual artifact unavailable: ' + name) from error
    if value is None or isinstance(value, ArtifactUnavailable):
        raise ArtifactUnavailable('Required actual artifact unavailable: ' + name)
    return value


require_artifact = artifact


def require_artifacts(ep, *names):
    return tuple(artifact(ep, name) for name in names)


def continuous_original(ep, value, *, geometry=None):
    """One half-pixel bilinear interpolation from continuous grid to original Q."""
    a = np.asarray(value, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('A finite two-dimensional continuous field is required')
    g = ep.query_geometry if geometry is None else geometry
    if not g:
        return resize(a, tuple(ep.original_shape))
    view = int(g['view_side']); sh, sw = map(int, g['resized_hw']); oy, ox = map(int, g['padding_top_left'])
    if min(sh, sw) <= 0 or min(oy, ox) < 0 or oy + sh > view or ox + sw > view:
        raise ValueError('Invalid physical resize/crop/pad geometry')
    # a spans the recorded encoder canvas. There is no 64/1024 binary stage.
    y = (oy + (np.arange(ep.original_shape[0]) + .5) * sh / ep.original_shape[0]) * a.shape[0] / view - .5
    x = (ox + (np.arange(ep.original_shape[1]) + .5) * sw / ep.original_shape[1]) * a.shape[1] / view - .5
    xx, yy = np.meshgrid(x, y)
    return sample_grid(a, yy, xx)


def U(ep, score, threshold=0., *, geometry=None):
    if not np.isfinite(threshold):
        raise ValueError('Finite fixed threshold required')
    return continuous_original(ep, score, geometry=geometry) > threshold


def render(ep, result):
    if not isinstance(result, Result):
        raise TypeError('Astra Result(field, threshold, mask_original, info) required')
    physical = None
    if result.field is not None:
        if result.info.get('field_space') == 'original':
            physical = np.asarray(result.field, float).copy()
            if physical.shape != tuple(ep.original_shape) or not np.isfinite(physical).all():
                raise ValueError('Original-space continuous field must match original query H/W')
        else:
            physical = continuous_original(ep, result.field)
    if result.mask_original is not None:
        mask = np.asarray(result.mask_original)
        if mask.dtype != bool or mask.shape != tuple(ep.original_shape):
            raise ValueError('Explicit binary output must have actual original query shape')
        mask = mask.copy()
    elif physical is not None:
        if not np.isfinite(result.threshold):
            raise ValueError('Finite output threshold required')
        mask = physical > result.threshold
    else:
        raise ValueError('A complete field or original binary mask is required')
    return dict(field=physical, margin=None if physical is None else physical - result.threshold,
                original=mask, info=dict(result.info, renderer='continuous_once_then_strict_threshold'))


def host_baseline(ep):
    field, mask, producer, renderer = require_artifacts(ep, 'mean_field', 'mask0', 'host_producer', 'host_renderer')
    if not isinstance(producer, Mapping) or not producer or not isinstance(renderer, (Mapping, str)):
        raise ArtifactUnavailable('Explicit host producer and renderer binding required')
    expected_hashes = ep.producer.get('source_image_hashes')
    if expected_hashes is not None and producer.get('source_image_hashes') != expected_hashes:
        raise ArtifactUnavailable('Host source image hashes do not match native episode')
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != tuple(ep.original_shape):
        raise ValueError('Host M0 must be the actual original-size binary mask')
    threshold = renderer.get('threshold', .5) if isinstance(renderer, Mapping) else .5
    return Result(np.asarray(field), threshold, mask.copy(), dict(host_producer=dict(producer), host_renderer=renderer))


def disagreement(ep, candidate, p0=None, mask0=None, *, threshold=.5, baseline_threshold=.5):
    """where(R(candidate) != R(p0), R(candidate), actual host M0)."""
    if p0 is None:
        p0 = artifact(ep, 'p0')
    if mask0 is None:
        # Requiring bindings avoids quietly swapping MEAN for a weak prototype.
        mask0 = host_baseline(ep).mask_original
    mask0 = np.asarray(mask0)
    if mask0.dtype != bool or mask0.shape != tuple(ep.original_shape):
        raise ValueError('Actual original M0 binary mask required')
    p1 = U(ep, candidate, threshold)
    baseline = U(ep, p0, baseline_threshold)
    return np.where(p1 != baseline, p1, mask0)


H = disagreement
W = disagreement
Z = disagreement


def fps(x, maximum=64, ids=None):
    """Observed row-first anchors; farthest and ID-stable ties, no synthetic mean."""
    if not isinstance(maximum, int) or not 0 <= maximum <= 64:
        raise ValueError('Anchor budget must be an integer in [0,64]')
    x = unit(np.asarray(x))
    ids = np.arange(len(x)) if ids is None else np.asarray(ids, int)
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique stable anchor IDs required')
    if not len(x) or maximum == 0:
        return np.empty((0, x.shape[1])), np.empty(0, int)
    order = np.argsort(ids, kind='stable')
    selected = [int(order[0])]; closest = x @ x[selected[0]]
    for _ in range(1, min(maximum, len(x))):
        candidates = order[~np.isin(order, selected)]
        new = int(candidates[np.argmin(closest[candidates])])
        if closest[new] >= 1 - EPS:
            break
        selected.append(new); closest = np.maximum(closest, x @ x[new])
    return readonly(x[selected]), ids[selected].copy()


anchors = fps


def coverage(ep):
    return np.divide(ep.wf, ep.wvalid, out=np.zeros_like(np.asarray(ep.wf, float)), where=np.asarray(ep.wvalid) > 0)


def roles(ep):
    c = coverage(ep); valid = np.asarray(ep.wvalid) > 0
    return c, valid & (c >= .9), valid & (c <= .1)


def _top5(x, bank):
    sim = unit(x) @ unit(bank).T
    k = min(5, len(bank))
    return np.partition(sim, sim.shape[1] - k, axis=1)[:, -k:].mean(1)


def a_b0(ep):
    """A source-specified B0; no implicit B/D/E/F baseline replacement."""
    validate(ep); c, pure_f, pure_b = roles(ep)
    info = dict(baseline='A_B0_64anchors_5NN_cosine_difference', quality='unknown')
    if np.sum(ep.wf) <= 0:
        return Result(np.full(ep.q_hw, -1.), info=dict(info, fallback='empty_reference_F'))
    if np.sum(ep.wb) <= 0:
        bank = ep.r[np.asarray(ep.wf) > 0]
        if len(bank) < 2:
            return Result(np.full(ep.q_hw, -1.), info=dict(info, fallback='single_class_insufficient_LOO'))
        sim = bank @ bank.T; np.fill_diagonal(sim, -np.inf); k = min(5, len(bank) - 1)
        leave_support = np.partition(sim, len(bank) - k, axis=1)[:, -k:].mean(1)
        threshold = float(np.quantile(1 - leave_support, .95))
        fbank, _ = fps(bank)
        field = threshold - (1 - _top5(ep.q, fbank))
        return Result(field.reshape(ep.q_hw), info=dict(info, fallback='single_class_LOO_distance95',
                                                        support_distance95=threshold, single_class_limit=True))
    banks = []
    for pure, weight in ((pure_f, ep.wf), (pure_b, ep.wb)):
        if pure.any():
            bank, _ = fps(ep.r[pure], ids=np.flatnonzero(pure))
        else:
            bank = unit(np.sum(ep.r * np.asarray(weight)[:, None], axis=0))[None]
        banks.append(bank)
    score = _top5(ep.q, banks[0]) - _top5(ep.q, banks[1])
    return Result(score.reshape(ep.q_hw), info=dict(info, anchors=[len(v) for v in banks],
                                                   missing_pure_weighted_mean=[not pure_f.any(), not pure_b.any()]))


B0 = a_b0


def spatial_folds(ep, *, buffer=1):
    """2x2 holds; unknown and their one-token buffer are excluded before fitting."""
    yy, xx = np.indices(ep.r_hw)
    block = ((yy >= ep.r_hw[0] / 2) * 2 + (xx >= ep.r_hw[1] / 2)).ravel()
    valid = np.asarray(ep.wvalid) > 0
    from scipy.ndimage import binary_dilation
    folds = []
    for k in range(4):
        hold = valid & (block == k)
        excluded = hold.reshape(ep.r_hw)
        if buffer:
            excluded = binary_dilation(excluded, structure=np.ones((3, 3)), iterations=buffer)
        train = valid & ~excluded.ravel()
        if train.any() and hold.any():
            folds.append((train, hold))
    return folds


def _training_episode(ep, train):
    # Remove label-derived artifacts and complete MR, which contains held labels.
    clean = {name: value for name, value in getattr(ep, 'artifacts', {}).items()
             if name in ('midlayers', 'preLN', 'QKV', 'attention', 'encoder_binding')}
    native = as_episode(ep)
    return replace(native, wf=readonly(np.asarray(ep.wf) * train),
                   wvalid=readonly(np.asarray(ep.wvalid) * train), reference_mask=None,
                   artifacts=clean, provider=None)


def calibrate_reference(ep, callback, configs=(None,), *, fallback_threshold=0.):
    """A C_R callback(fold_ep, train, hold, config)->full reference score.

    The callback sees unknown held labels removed, has no complete MR, and must
    rebuild every label-dependent fit. It returns a reference-grid Result or
    array; full refit is the owner's responsibility after the selected config.
    Threshold loss is FG/BG-balanced soft-coverage absolute error. This is A C_R,
    not the F family's original-size IoU weight calibration.
    """
    configs = tuple(configs)
    if not 1 <= len(configs) <= 12:
        raise ValueError('A C_R allows <=12 total joint configurations')
    folds = spatial_folds(ep); c = coverage(ep)
    if len(folds) < 2:
        return configs[0], float(fallback_threshold), dict(calibration='A_C_R', fallback='fewer_than_two_spatial_folds')
    candidates = []; fit_calls = 0
    for index, config in enumerate(configs):
        scores, targets, wf, wb = [], [], [], []
        for train, hold in folds:
            out = callback(_training_episode(ep, train), train.copy(), hold.copy(), config); fit_calls += 1
            value = out.field if isinstance(out, Result) else out
            value = np.asarray(value, float).ravel()
            if value.shape != (len(ep.r),) or not np.isfinite(value[hold]).all():
                raise ValueError('Calibration callback must return complete finite reference-grid score')
            scores.extend(value[hold]); targets.extend(c[hold]); wf.extend(np.asarray(ep.wf)[hold]); wb.extend(np.asarray(ep.wb)[hold])
        scores, targets, wf, wb = map(np.asarray, (scores, targets, wf, wb))
        if wf.sum() <= 0 or wb.sum() <= 0:
            continue
        # Strict >t means thresholds at observed values exhaust all partitions.
        thresholds = np.unique(np.r_[scores, 0., np.nextafter(scores.min(), -np.inf)])
        for t in thresholds:
            error = np.abs((scores > t).astype(float) - targets)
            loss = .5 * (float(error @ wf / wf.sum()) + float(error @ wb / wb.sum()))
            candidates.append((loss, abs(float(t)), index, float(t)))
    if not candidates:
        return configs[0], float(fallback_threshold), dict(calibration='A_C_R', fallback='missing_balanced_validation_roles', fold_fit_calls=fit_calls)
    loss, _, index, threshold = min(candidates)
    return configs[index], threshold, dict(calibration='A_C_R', source_scope='shared-context not independent cross-image validation',
                                           balanced_coverage_error=loss, folds=len(folds), fold_fit_calls=fit_calls,
                                           configurations=len(configs), held_labels_removed=True)


C_R = calibrate_reference
