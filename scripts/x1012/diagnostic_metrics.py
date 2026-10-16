"""Exact scoring-only diagnostics. Query labels/valid masks never enter inference."""
import numpy as np

SCORING_VERSION = 'exact_distinct_valid_v1'
DIAGNOSTIC_CLASSES = ('threshold_recoverable', 'reference_probe_gap', 'sh_score_gap', 'unresolved_by_probes')


def exact_best_iou(field, gt, valid=None):
    """Max IoU of field > t over every distinct score boundary, without splitting ties.

    IoU uses the repository convention I / max(U, 1), including empty GT.
    The result is an oracle for this one field, not for a representation or model family.
    """
    f, g = np.asarray(field), np.asarray(gt, dtype=bool)
    if f.shape != g.shape:
        raise ValueError('field and gt must have identical shapes')
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        if v.shape != g.shape:
            raise ValueError('valid and gt must have identical shapes')
        f, g = f[v], g[v]
    else:
        f, g = f.ravel(), g.ravel()
    if not np.isfinite(f).all():
        raise ValueError('non-finite score in the scoring region')
    if not f.size or not g.any():
        return 0.0
    order = np.argsort(-f.astype(np.float64, copy=False), kind='stable')
    scores = f[order]
    # A strict threshold can select a complete score group, never a prefix of its ties.
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    intersection = np.cumsum(g[order], dtype=np.int64)[ends]
    predicted = ends + 1
    union = int(g.sum()) + predicted - intersection
    return float(np.max(intersection / np.maximum(union, 1)))


def classify_diagnostic(row):
    """Positive witnesses first; failed probes do not establish impossibility."""
    if row['cut_F'] >= .5:
        return 'threshold_recoverable'
    if row['cut_PQ'] >= .5:
        return 'reference_probe_gap' if row['cut_PR'] < .3 else 'sh_score_gap'
    return 'unresolved_by_probes'


def legacy_classification(row):
    """Original W45 rule, retained only to separate scoring changes from interpretation."""
    if row['cut_PQ'] < .5:
        return 'representation'
    if row['cut_F'] >= .5:
        return 'readout'
    return 'transfer' if row['cut_PR'] < .3 else 'other'
