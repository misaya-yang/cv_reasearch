"""Same-information reference-only RBF SVM control, not a new method.

CPU NumPy/sklearn only. Native full feature coordinates are never normalized,
projected or trained. Query labels, area priors and region selection are absent.
Output is an uncalibrated dimensionless signed margin (foreground positive).
"""
from dataclasses import dataclass
import math
import warnings
import numpy as np


def _features(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite [N,D] in full native coordinates')
    return result


def _sample(values, count):
    return values[np.arange(count, dtype=np.int64) * len(values) // count]


def reference_bandwidth(foreground, background, per_class_cap=128):
    """tau=median(nonzero unordered reference-pair squared distances)/2.

    Use equal counts min(nFG,nBG,128) per class, sampled at evenly spaced input
    indices. Only this bandwidth sample is reduced; SVM fitting keeps every
    supplied lawful anchor. Pair distance uses direct differences so identical
    descriptors have exact zero distance, not cancellation-created variance.
    """
    fg, bg = _features(foreground, 'foreground'), _features(background, 'background')
    if fg.shape[1] != bg.shape[1] or int(per_class_cap) != per_class_cap or per_class_cap < 1:
        raise ValueError('common feature coordinates and positive integer sampling cap required')
    count = min(len(fg), len(bg), int(per_class_cap))
    audit = dict(bandwidth_rule='median_nonzero_unordered_balanced_reference_pair_d2/2',
                 bandwidth_samples_per_class=count, bandwidth_reference_only=True,
                 post_feature_normalization=False, query_used=False)
    if count == 0:
        return None, dict(audit, state='MISSING_REFERENCE_CLASS')
    sample = np.concatenate((_sample(fg, count), _sample(bg, count)))
    distances = []
    with np.errstate(over='ignore', invalid='ignore'):
        for index in range(len(sample) - 1):
            delta = sample[index + 1:] - sample[index]
            pair = np.einsum('ij,ij->i', delta, delta, optimize=False)
            if not np.isfinite(pair).all():
                return None, dict(audit, state='NONFINITE_REFERENCE_PAIR_DISTANCE')
            distances.extend(pair[pair > 0].tolist())
    if not distances:
        return None, dict(audit, state='NO_POSITIVE_REFERENCE_BANDWIDTH')
    median = float(np.median(distances))
    return median / 2, dict(audit, state='REFERENCE_BANDWIDTH_READY',
                          nonzero_pair_count=len(distances), median_pair_d2=median, bandwidth=median / 2)


def class_balanced_weights(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [-1, 1]).all():
        raise ValueError('binary labels must be a vector in {-1,+1}')
    positive, negative = int((labels == 1).sum()), int((labels == -1).sum())
    if min(positive, negative) < 1:
        raise ValueError('both reference classes required')
    return np.where(labels == 1, len(labels) / (2 * positive), len(labels) / (2 * negative))


def rbf_kernel(left, right, bandwidth):
    """exp(-||x-y||^2/(2*tau)); no coordinates or kernel scales are fitted."""
    x, y = _features(left, 'left'), _features(right, 'right')
    if x.shape[1] != y.shape[1] or not math.isfinite(float(bandwidth)) or bandwidth <= 0:
        raise ValueError('common coordinates and finite positive bandwidth required')
    with np.errstate(over='ignore', invalid='ignore'):
        xx = np.einsum('ij,ij->i', x, x, optimize=False)
        yy = np.einsum('ij,ij->i', y, y, optimize=False)
        xy = np.einsum('ij,kj->ik', x, y, optimize=False)
        distances = xx[:, None] + yy[None] - 2 * xy
    if not np.isfinite(distances).all():
        raise ArithmeticError('nonfinite query/reference kernel distance')
    return np.exp(-np.maximum(distances, 0) / (2 * float(bandwidth)))


@dataclass
class ReferenceKernelSVM:
    """Selected reference classifier only; no annotation/evaluator access."""
    reference_features: np.ndarray
    reference_labels: np.ndarray
    bandwidth: float
    estimator: object
    semantic_orientation: int
    audit: dict

    @property
    def dual_coefficients(self):
        return self.semantic_orientation * self.estimator.dual_coef_[0]

    @property
    def intercept(self):
        return self.semantic_orientation * float(self.estimator.intercept_[0])

    def decision_function(self, query, chunk_rows=128):
        """Return ALL signed margins; caller scores GT only after predictions.

        No L2 normalization, min/max, sigmoid, threshold, quota or CRF here.
        Distant queries approach the fitted intercept; BG is NOT guaranteed.
        """
        q = _features(query, 'query')
        if q.shape[1] != self.reference_features.shape[1] or int(chunk_rows) != chunk_rows or chunk_rows < 1:
            raise ValueError('common coordinates and positive integer chunk required')
        outputs = []
        for begin in range(0, len(q), int(chunk_rows)):
            gram = rbf_kernel(q[begin:begin + int(chunk_rows)], self.reference_features, self.bandwidth)
            outputs.append(self.semantic_orientation * self.estimator.decision_function(gram))
        result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ArithmeticError('nonfinite SVM decision margin')
        return result

    def representer_margin(self, query):
        q = _features(query, 'query')
        anchors = self.reference_features[self.estimator.support_]
        return rbf_kernel(q, anchors, self.bandwidth) @ self.dual_coefficients + self.intercept


def fit_reference_kernel_svm(foreground, background, C=1., max_iterations=10000):
    """Fit sklearn.SVC(kernel='precomputed') on unchanged full coordinates.

    SVC minimizes .5||f||_H^2+C sum_i weight_i hinge_i, where
    weight_i=N/(2*n_class_i). This equals .5||f||^2+(CN/2)(mean_FG hinge
    +mean_BG hinge), or lambda/2||f||^2+mean_FG+mean_BG with lambda=2/(CN).
    C=1 is fixed by default; there is no hyperparameter/GT selection here.

    Partial conflicting descriptors remain in the soft-margin problem. For
    numerical label-reversal symmetry, orientation is the sign of the earliest
    lexicographic descriptor's NONZERO normalized FG-minus-BG empirical mass.
    Rows sort by feature keys, then canonical label for conflicting ties.
    Semantic orientation maps the canonical margin back to FG+1/BG-1.
    Swapping FG/BG fits the identical canonical problem and negates the margin.
    This relabeling/reordering preserves the class-balanced objective exactly.

    Return (model,audit). Missing classes, zero bandwidth, IDENTICAL normalized
    class empirical distributions, unavailable sklearn and nonconvergence
    return explicit (None,audit). Partial conflicts are audited, NOT excluded.
    They NEVER produce an all-background mask or a fabricated fitted model.
    """
    fg, bg = _features(foreground, 'foreground'), _features(background, 'background')
    if fg.shape[1] != bg.shape[1] or not math.isfinite(float(C)) or C <= 0:
        raise ValueError('common feature coordinates and finite positive C required')
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError('finite positive integer iteration budget required')
    bandwidth, band_audit = reference_bandwidth(fg, bg)
    audit = dict(band_audit, C=float(C), foreground_anchors=len(fg), background_anchors=len(bg),
                 feature_dimension=fg.shape[1], query_ground_truth_used=False,
                 classifier='existing_sklearn_SVC_precomputed_RBF',
                 margin_units='uncalibrated_dimensionless_signed_SVM_margin', foreground_sign=1,
                 far_query_background_guaranteed=False, max_iterations=int(max_iterations))
    if not len(fg) or not len(bg):
        return None, audit
    x = np.concatenate((fg, bg))
    labels = np.concatenate((np.ones(len(fg)), -np.ones(len(bg)))).astype(np.int8)
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    positive = np.bincount(inverse[labels == 1], minlength=inverse.max() + 1)
    negative = np.bincount(inverse[labels == -1], minlength=inverse.max() + 1)
    conflicting = (positive > 0) & (negative > 0)
    # Exact rational comparison avoids introducing orientation through floating
    # cancellation when unequal class sizes have identical normalized masses.
    mass_numerator = np.asarray([int(p)*len(bg)-int(n)*len(fg)
                                 for p,n in zip(positive,negative)], dtype=object)
    nonzero = np.flatnonzero(mass_numerator)
    audit.update(conflict_count=int(conflicting.sum()),
                 conflicting_unique_descriptors=int(conflicting.sum()),
                 conflicting_foreground_anchors=int(positive[conflicting].sum()),
                 conflicting_background_anchors=int(negative[conflicting].sum()),
                 partial_conflicts_retained=True,
                 class_distribution_identical=not len(nonzero))
    if bandwidth is None:
        return None, audit
    if not len(nonzero):
        return None, dict(audit, state='IDENTICAL_CLASS_DISTRIBUTIONS', trained=False)
    orientation = 1 if mass_numerator[nonzero[0]] > 0 else -1
    canonical_labels = labels * orientation
    order = np.lexsort((canonical_labels,) + tuple(x[:, column] for column in range(x.shape[1] - 1, -1, -1)))
    x, labels, canonical_labels = x[order].copy(), labels[order].copy(), canonical_labels[order].copy()
    weights = class_balanced_weights(labels)
    audit.update(sample_weight_foreground=len(x) / (2 * len(fg)),
                 sample_weight_background=len(x) / (2 * len(bg)),
                 equivalent_lambda=2 / (float(C) * len(x)),
                 canonical_order='lexicographic_unchanged_feature_rows_then_canonical_label_ties',
                 orientation_rule='first_nonzero_normalized_class_mass_difference_in_lexicographic_descriptor_order',
                 orientation_descriptor_index=int(nonzero[0]), semantic_orientation=orientation)
    try:
        from sklearn.svm import SVC
    except ImportError:
        return None, dict(audit, state='SKLEARN_UNAVAILABLE', trained=False)
    gram = rbf_kernel(x, x, bandwidth)
    np.fill_diagonal(gram, 1.)
    estimator = SVC(C=float(C), kernel='precomputed', probability=False,
                    tol=1e-9, shrinking=False, max_iter=int(max_iterations), cache_size=64.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(gram, canonical_labels, sample_weight=weights)
    if estimator.fit_status_ != 0:
        return None, dict(audit, state='SVM_NOT_CONVERGED', trained=False,
                          warnings=[str(w.message) for w in caught])
    audit.update(state='REFERENCE_KERNEL_SVM_FITTED', trained=True,
                 support_vectors=len(estimator.support_), solver_tolerance=1e-9,
                 iterations=np.asarray(getattr(estimator, 'n_iter_', [])).tolist(), probability_calibration=False,
                 feature_normalization_or_projection=False)
    return ReferenceKernelSVM(x, labels, bandwidth, estimator, orientation, audit), audit
