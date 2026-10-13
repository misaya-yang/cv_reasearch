"""Small, label-free region operations and exact union-IoU selection for A6."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

KMEANS = (16, 32, 64, 128, 256)
TAUS = (.5, .6, .7)
GEOMETRIES = tuple(f'kmeans{k}' for k in KMEANS) + tuple(f'average_tau{t}' for t in TAUS) + ('spatial64',)
FEATURES = ('mean_c', 'max_c', 'mean_log_e', 'relative_mean_log_e',
            'log_area_fraction', 'log1p_mnn_count', 'mnn_fraction',
            'log1p_fg_mnn_mass', 'log_max_region_kernel_fg',
            'log_max_region_kernel_bg', 'region_kernel_fg_bg_contrast',
            'neighbor_mean_c', 'neighbor_max_c')


def compact(labels):
    return np.unique(np.asarray(labels).reshape(-1), return_inverse=True)[1].astype(np.uint16)


def grid_connectivity(side=64):
    from scipy.sparse import coo_matrix
    grid = np.arange(side * side).reshape(side, side)
    a = np.r_[grid[:, :-1].ravel(), grid[:-1, :].ravel()]
    b = np.r_[grid[:, 1:].ravel(), grid[1:, :].ravel()]
    return coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])),
                      shape=(side * side, side * side)).tocsr()


def average_tree_and_spatial(q):
    """One exact cosine/average tree for every tau; no PCA approximation."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.cluster import AgglomerativeClustering
    distance = (1 - q @ q.T).clamp(0, 2).numpy()
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='average')
    result = {}
    for tau in TAUS:
        # FoRIS merges strictly below 1-tau; fcluster's cut is inclusive.
        threshold = np.nextafter(np.float64(1 - tau), -np.inf)
        result[f'average_tau{tau}'] = compact(fcluster(tree, threshold, criterion='distance'))
    spatial = AgglomerativeClustering(n_clusters=64, metric='precomputed',
                                     linkage='average', connectivity=grid_connectivity()).fit_predict(distance)
    result['spatial64'] = compact(spatial)
    return tree, result


def region_means(values, labels):
    labels = np.asarray(labels, np.int64)
    values = torch.as_tensor(np.asarray(values, np.float32))
    ids = torch.from_numpy(labels)
    count = torch.bincount(ids, minlength=int(labels.max()) + 1).float()
    result = torch.zeros((len(count),) + tuple(values.shape[1:]), dtype=torch.float32)
    result.index_add_(0, ids, values)
    return (result / count.reshape((-1,) + (1,) * (values.ndim - 1))).numpy()


def neighbor_c(labels, mean_c):
    """Four-neighbor region graph, weighted by shared token-boundary length."""
    side = int(np.sqrt(len(labels)))
    grid = np.asarray(labels).reshape(side, side)
    a = np.r_[grid[:, :-1].ravel(), grid[:-1, :].ravel()]
    b = np.r_[grid[:, 1:].ravel(), grid[1:, :].ravel()]
    different = a != b
    a, b = a[different], b[different]
    src, dst = np.r_[a, b], np.r_[b, a]
    count = np.bincount(src, minlength=len(mean_c))
    total = np.bincount(src, weights=mean_c[dst], minlength=len(mean_c))
    average = np.divide(total, count, out=mean_c.astype(np.float64).copy(), where=count > 0)
    maximum = np.full(len(mean_c), -np.inf)
    np.maximum.at(maximum, src, mean_c[dst])
    maximum[count == 0] = mean_c[count == 0]
    return average, maximum


def describe(labels, c, log_e, mutual, fg_mutual, token_kernel_f, token_kernel_b, kernel_shift):
    count = np.bincount(labels).astype(np.float32)
    scalars = np.column_stack((c, log_e, mutual, fg_mutual))
    means = region_means(scalars, labels)
    peak = np.full(len(count), -np.inf)
    np.maximum.at(peak, labels, c)
    log_fg = np.log(np.maximum(region_means(token_kernel_f, labels).max(1), 1e-30)) + kernel_shift
    log_bg = np.log(np.maximum(region_means(token_kernel_b, labels).max(1), 1e-30)) + kernel_shift
    adjacent, adjacent_peak = neighbor_c(labels, means[:, 0])
    features = np.column_stack((means[:, 0], peak, means[:, 1], means[:, 1] - np.median(log_e),
                               np.log(count / len(labels)), np.log1p(means[:, 2] * count), means[:, 2],
                               np.log1p(means[:, 3] * count), log_fg, log_bg, log_fg - log_bg,
                               adjacent, adjacent_peak)).astype(np.float32)
    if features.shape != (len(count), len(FEATURES)) or not np.isfinite(features).all():
        raise ValueError('Non-finite or invalid regional descriptors')
    return features


def reference_kernels(kernel, coverage, reference_labels):
    """Mean token-pair kernel to each reference region's soft FG/BG subset.

    The caller uses one common exp shift for the entire image pair and adds it
    back after taking logs. Per-query-row softmax would change this descriptor.
    """
    ids = torch.as_tensor(reference_labels.astype(np.int64))
    onehot = torch.nn.functional.one_hot(ids, int(ids.max()) + 1).float()
    y = torch.as_tensor(coverage, dtype=torch.float32)
    fg, bg = onehot * y[:, None], onehot * (1 - y[:, None])
    weights = torch.cat((fg, bg), 1)
    masses = weights.sum(0)
    value = kernel @ weights
    value = torch.where(masses[None] > 0, value / masses.clamp_min(1e-30)[None], torch.zeros_like(value))
    split = fg.shape[1]
    return value[:, :split].numpy(), value[:, split:].numpy()


def selection(tp, fp, gt_area):
    """Exact arbitrary-subset union oracle: every GT pixel stays in union."""
    tp, fp = np.asarray(tp, np.int64), np.asarray(fp, np.int64)
    order = np.argsort(-tp / np.maximum(tp + fp, 1), kind='stable')
    inter = np.r_[0, np.cumsum(tp[order])]
    union = gt_area + np.r_[0, np.cumsum(fp[order])]
    cut = int(np.argmax(inter / np.maximum(union, 1)))
    keep = np.zeros(len(tp), dtype=bool)
    keep[order[:cut]] = True
    return keep


def token_pixel_counts(gt, valid):
    side = 64
    return tuple(np.asarray(x, np.int64).reshape(side, 16, side, 16).sum((1, 3)).ravel()
                 for x in (gt, valid))


def region_pixel_counts(labels, token_gt, token_valid):
    n = np.bincount(labels, weights=token_valid, minlength=int(labels.max()) + 1).astype(np.int64)
    tp = np.bincount(labels, weights=token_gt, minlength=len(n)).astype(np.int64)
    return n, tp


def metric_counts(keep, n, tp, gt_area, valid_area):
    """Exactly the nearest-region pixel metric, with pr.metric's JSON schema."""
    keep = np.asarray(keep, bool)
    predicted, inter = int(n[keep].sum()), int(tp[keep].sum())
    fp, fn = predicted - inter, int(gt_area) - inter
    union = inter + fp + fn
    error = (float(abs(np.log2(predicted / gt_area))) if predicted and gt_area
             else 0. if not predicted and not gt_area else None)
    return dict(I=inter, U=union, TP=inter, FP=fp, FN=fn, GT=int(gt_area), predicted=predicted,
                IoU=100 * inter / max(union, 1), target_fraction=int(gt_area) / max(int(valid_area), 1),
                abs_log2_area_error=error, zero_predicted_area=predicted == 0,
                infinite_area_error=error is None)


def photo_train_half(photo, seed=20261012):
    digest = hashlib.sha256(f'{seed}\0{photo}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % 2 == 0


def fit_logistic(x, y, area_weights):
    """Train-only, area-weighted soft BCE; each photo episode has total mass 1."""
    x = torch.as_tensor(np.asarray(x, np.float32))
    y = torch.as_tensor(np.asarray(y, np.float32))
    weight = torch.as_tensor(np.asarray(area_weights, np.float32))
    if len(x) == 0 or not bool(weight.sum() > 0):
        raise ValueError('No valid training regions')
    weight = weight / weight.sum()
    mean = (x * weight[:, None]).sum(0)
    sd = ((x - mean).square() * weight[:, None]).sum(0).sqrt().clamp_min(1e-6)
    xn = (x - mean) / sd
    theta = torch.zeros(x.shape[1] + 1, requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], max_iter=100, history_size=10,
                                  tolerance_grad=1e-7, tolerance_change=1e-10,
                                  line_search_fn='strong_wolfe')
    evaluations = 0

    def closure():
        nonlocal evaluations
        evaluations += 1
        optimizer.zero_grad()
        logits = xn @ theta[:-1] + theta[-1]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * weight).sum()
        loss = loss + 1e-4 * theta[:-1].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    standardized = theta.detach()
    coef = standardized[:-1] / sd
    intercept = standardized[-1] - mean @ coef
    return dict(coef=coef.tolist(), intercept=float(intercept), train_mean=mean.tolist(), train_sd=sd.tolist(),
                standardized_coef=standardized[:-1].tolist(), standardized_intercept=float(standardized[-1]),
                n_regions=len(x), loss_evaluations=evaluations, l2=1e-4,
                max_iter=100, threshold_probability=.5, target='valid-pixel soft foreground coverage',
                weighting='region valid area / episode valid area; episodes equally weighted')


def predict(model, x):
    return np.asarray(x) @ np.asarray(model['coef'], np.float32) + model['intercept'] > 0
