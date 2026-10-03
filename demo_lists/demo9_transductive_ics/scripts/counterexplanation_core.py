"""CPU primitives for a bounded complete-negative-reference observation probe.

These are not a validated segmentation method. Negative fields must be produced
by complete public FoRIS with a different LEGAL reference mask; fg_max-bg_max
does not qualify. The extractor alone can be cached because its input is [S,Q]
RGB, independent of the reference mask. Never persist that feature cache.
"""
from collections import deque
from contextlib import contextmanager

import numpy as np
from scipy import ndimage


def response_density(field, uniform_on_constant=False):
    """A response-mass distribution, explicitly not a calibrated posterior."""
    x = np.asarray(field, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite source field")
    y = x - x.min()
    if y.max() <= np.finfo(float).eps * max(1., np.abs(x).max()):
        return np.full_like(x, 1/x.size) if uniform_on_constant else None
    y += np.finfo(float).eps * max(1., y.max())
    return y / y.sum()


def counterexplanation_cut(positive, negatives, levels, native=.5, uniform_control=False):
    """Use candidate INSIDE-OUTSIDE evidence difference, not raw summed mass.

    The original sum D(empty)=D(full)=0 and favors area via the integral.
    Dividing inside/outside mass removes that direct area multiplier but DOES
    NOT remove small-region variance or semantic-seam bias. Degenerate negative
    observations are discarded; all degenerate or identical observations must
    return the native cut instead of manufacturing score-contrast evidence.
    """
    s = np.asarray(positive, float)
    p = response_density(s)
    ns = [response_density(x, uniform_on_constant=uniform_control) for x in negatives]
    ns = [x for x in ns if x is not None]
    if p is None or not ns:
        return native, "no_nonconstant_negative", None
    if all(np.max(np.abs(p-n)) <= 32*np.finfo(float).eps for n in ns):
        evidence = np.zeros_like(p)
        return native, "identical_observations", evidence
    # The pointwise excess is for the information/AUC diagnostic ONLY. The
    # selection below compares COMPLETE negative explanations, avoiding a
    # pixelwise union of incompatible negative hypotheses.
    evidence = p - np.maximum.reduce(ns)
    sn = (s - s.min()) / (s.max() - s.min())
    candidates = []
    for t in levels:
        mask = sn > t
        # Same validity window as the prepared extent experiment; not tuned.
        if mask.sum() < 4 or mask.sum() > .9 * mask.size:
            continue
        positive_contrast = p[mask].mean()-p[~mask].mean()
        negative_contrast = max(n[mask].mean()-n[~mask].mean() for n in ns)
        value = positive_contrast-negative_contrast
        candidates.append((float(value), float(t)))
    if not candidates:
        return native, "no_valid_level", evidence
    best = max(x[0] for x in candidates)
    ties = [t for v, t in candidates if abs(v - best) <= 1e-12]
    return min(ties, key=lambda t: (abs(t - native), t)), "selected", evidence


def positive_only_cut(positive, levels, native=.5):
    """Explicit naive control; SAME inside/outside operator as the main arm.

    A uniform negative's contrast is zero, so positive-only and uniform-response
    controls must be exactly equal. Neither is swallowed by main-arm fallback.
    """
    return counterexplanation_cut(positive, [np.ones_like(positive)], levels,
                                 native=native, uniform_control=True)


def _connected_exact(background, seed, area):
    """Fixed 4-neighbour breadth-first fill; labels, never query GT, bound it."""
    h, w = background.shape
    result = np.zeros_like(background)
    visited = np.zeros_like(background)
    queue = deque([seed]); visited[seed] = True
    filled = 0
    while queue and filled < area:
        y, x = queue.popleft()
        result[y, x] = True; filled += 1
        for yy, xx in ((y-1, x), (y, x-1), (y, x+1), (y+1, x)):
            if 0 <= yy < h and 0 <= xx < w and background[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = True; queue.append((yy, xx))
    if filled != area:
        raise ValueError("connected component smaller than labelled foreground")
    return result


def negative_proposals(reference_fg, priority, randomized=False, seed=0):
    """At most two connected, exact-area masks, every pixel known reference BG.

    Priority is the positive query score routed back to the reference by the
    existing back_idx, then resized to ORIGINAL reference geometry. It carries
    no query labels. These masks are hypothesis proposals, NOT semantic concepts.
    Random control uses identical component/area/connection restrictions.
    """
    fg = np.asarray(reference_fg, bool)
    score = np.asarray(priority, float)
    if score.shape != fg.shape or not np.isfinite(score).all():
        raise ValueError("priority geometry or finite-value contract")
    area = int(fg.sum())
    if area == 0:
        return [], "empty_labelled_foreground"
    bg = ~fg
    labels, _ = ndimage.label(bg)
    sizes = np.bincount(labels.ravel())
    eligible = bg & (sizes[labels] >= area)
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return [], "no_area_matched_connected_background"
    rng = np.random.default_rng(seed)
    order = rng.permutation(ids) if randomized else ids[np.argsort(-score.ravel()[ids], kind="stable")]
    selected = []
    for index in order:
        point = np.unravel_index(index, fg.shape)
        if selected and selected[0][point]:
            continue
        proposal = _connected_exact(bg, point, area)
        assert not (proposal & fg).any() and proposal.sum() == area
        selected.append(proposal)
        if len(selected) == 2:
            break
    return selected, "ok" if selected else "no_distinct_proposal"


@contextmanager
def cache_public_extractor(host):
    """Only raw image encoding is reused; COMPLETE public source still runs.

    Actual DINO native cached/uncached mask equality is a mandatory runtime gate,
    not something this wrapper or a synthetic CPU test establishes.
    """
    import torch
    had, old = "_extract_features" in host.__dict__, host.__dict__.get("_extract_features")
    original = host._extract_features
    state = dict(calls=0, reuses=0, images=None, raw=None)
    def extract(images):
        if state["raw"] is None:
            state["images"] = images.detach().clone()
            state["raw"] = original(images).detach().clone()
            state["calls"] += 1
        else:
            if not torch.equal(images, state["images"]):
                raise RuntimeError("cached [S,Q] image contract changed")
            state["reuses"] += 1
        return state["raw"].clone()
    try:
        host._extract_features = extract
        yield state
    finally:
        if had:
            host._extract_features = old
        else:
            delattr(host, "_extract_features")
        state["raw"] = state["images"] = None
