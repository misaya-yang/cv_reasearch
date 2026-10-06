"""Pro M4: paired reference transplants, environment covariance, exact Potts cut.

The encoder is injected and frozen. Only canonical points inside reference
transplants receive labels; untouched query pixels never enter the fit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import time

import numpy as np
from PIL import Image


ARMS = ('paired', 'mean', 'class_lda', 'ref_canvas')


@dataclass(frozen=True)
class Config:
    work_size: int = 1024
    long_edges: tuple = (128, 256)
    centers_xy: tuple = ((256, 256), (768, 768))
    minimum_background_side: int = 32
    seam_margin: int = 16
    window_side: int = 16
    minimum_coverage: float = .9
    minimum_points: int = 4
    maximum_points: int = 32
    minimum_ridge: float = 1e-6
    minimum_separation: float = 1e-8
    potts_weight: float = .25


class GeometryFailure(ValueError):
    pass


def _fixed(cfg):
    if cfg != Config():
        raise ValueError('This version fixes the supplied Pro M4 recipe')


def _rgb(value):
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError('Require RGB uint8 HxWx3')
    return value


def canonical_inputs(reference, reference_mask, query, cfg=Config()):
    _fixed(cfg)
    reference, query = _rgb(reference), _rgb(query)
    mask = np.asarray(reference_mask)
    if mask.shape != reference.shape[:2] or not np.isin(mask, (0, 1)).all():
        raise ValueError('A complete binary reference mask matching reference RGB is required')
    size = (cfg.work_size, cfg.work_size)
    r = np.asarray(Image.fromarray(reference).resize(size, Image.Resampling.BILINEAR)).copy()
    q = np.asarray(Image.fromarray(query).resize(size, Image.Resampling.BILINEAR)).copy()
    m = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)).astype(bool)
    return r, m, q


def maximum_background_rectangle(mask):
    """Largest all-zero rectangle; ties by (top,left,bottom,right), half-open."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Two-dimensional mask required')
    height = np.zeros(mask.shape[1], dtype=np.int64)
    best, area = None, 0
    for row in range(mask.shape[0]):
        height = np.where(mask[row], 0, height + 1)
        stack = []
        for col in range(mask.shape[1] + 1):
            h, start = (int(height[col]) if col < mask.shape[1] else 0), col
            while stack and stack[-1][1] > h:
                left, previous = stack.pop()
                rectangle = (row + 1 - previous, left, row + 1, col)
                candidate_area = previous * (col - left)
                if candidate_area > area or (candidate_area == area and candidate_area and
                                               (best is None or rectangle < best)):
                    best, area = rectangle, candidate_area
                start = left
            if h and (not stack or stack[-1][1] < h):
                stack.append((start, h))
    return best


def _resize_shape(height, width, edge):
    scale = edge / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def canonical_points(mask, cfg=Config()):
    """Row-major first/ties, Euclidean farthest points in original crop geometry.

    Candidates are pixel centers of the smallest resized crop. A 16x16
    half-open window [row-8:row+8,col-8:col+8] defines discrete coverage.
    Corresponding original-mask pixel and resized center must both be FG.
    """
    height, width = mask.shape
    sh, sw = _resize_shape(height, width, min(cfg.long_edges))
    small = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((sw, sh), Image.Resampling.NEAREST))
    if min(sh, sw) < 2 * cfg.seam_margin + 1:
        raise GeometryFailure('positive_crop_too_narrow_for_seam_exclusion')
    rr, cc = np.meshgrid(np.arange(sh), np.arange(sw), indexing='ij')
    valid = ((rr + .5 >= cfg.seam_margin) & (rr + .5 <= sh - cfg.seam_margin) &
             (cc + .5 >= cfg.seam_margin) & (cc + .5 <= sw - cfg.seam_margin) & (small != 0))
    rows, cols = np.nonzero(valid)
    half = cfg.window_side // 2
    integral = np.pad(small.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r0, r1, c0, c1 = rows-half, rows+half, cols-half, cols+half
    coverage = (integral[r1, c1]-integral[r0, c1]-integral[r1, c0]+integral[r0, c0]) / cfg.window_side**2
    xy = np.column_stack(((cols + .5)/sw, (rows + .5)/sh))
    original_rows = np.minimum((xy[:, 1]*height).astype(int), height-1)
    original_cols = np.minimum((xy[:, 0]*width).astype(int), width-1)
    eligible = (coverage >= cfg.minimum_coverage) & mask[original_rows, original_cols]
    xy, coverage = xy[eligible], coverage[eligible]
    if len(xy) < cfg.minimum_points:
        raise GeometryFailure('fewer_than_four_valid_canonical_points')
    geometry = xy * np.array([width, height])
    selected = [0]
    distance = np.sum((geometry-geometry[0])**2, axis=1)
    for _ in range(1, min(cfg.maximum_points, len(xy))):
        distance[selected] = -np.inf
        nxt = int(np.argmax(distance))  # First row-major index resolves ties.
        selected.append(nxt)
        distance = np.minimum(distance, np.sum((geometry-geometry[nxt])**2, axis=1))
    return xy[selected], coverage[selected], (sh, sw)


def prepare_geometry(reference, mask, cfg=Config()):
    _fixed(cfg)
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise GeometryFailure('empty_reference_foreground')
    positive_box = (int(rows.min()), int(cols.min()), int(rows.max()+1), int(cols.max()+1))
    negative_box = maximum_background_rectangle(mask)
    if negative_box is None or min(negative_box[2]-negative_box[0], negative_box[3]-negative_box[1]) < cfg.minimum_background_side:
        raise GeometryFailure('no_background_rectangle_with_both_sides_at_least_32')
    top, left, bottom, right = positive_box
    crop = reference[top:bottom, left:right].copy()
    crop_mask = mask[top:bottom, left:right].copy()
    top, left, bottom, right = negative_box
    background = reference[top:bottom, left:right].copy()
    points, coverage, smallest_shape = canonical_points(crop_mask, cfg)
    conditions = []
    for edge in cfg.long_edges:
        height, width = _resize_shape(*crop_mask.shape, edge)
        for cx, cy in cfg.centers_xy:
            left, top = cx-width//2, cy-height//2
            conditions.append(dict(long_edge=edge, center_xy=[cx, cy],
                                   box_tlbr=[top, left, top+height, left+width]))
    return dict(positive=crop, positive_mask=crop_mask, negative=background,
                positive_box=positive_box, negative_box=negative_box, points_xy=points,
                minimum_scale_coverage=coverage, smallest_shape=smallest_shape, conditions=conditions)


def transplant(canvas, crop, condition):
    top, left, bottom, right = condition['box_tlbr']
    if not (0 <= top < bottom <= canvas.shape[0] and 0 <= left < right <= canvas.shape[1]):
        raise ValueError('Transplant box outside canvas')
    result = canvas.copy()
    result[top:bottom, left:right] = np.asarray(Image.fromarray(crop).resize(
        (right-left, bottom-top), Image.Resampling.BILINEAR))
    return result


def mapped_points(points, condition):
    top, left, bottom, right = condition['box_tlbr']
    return points*np.array([right-left, bottom-top]) + np.array([left, top])


def unit(value):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-12):
        raise ValueError('Finite nonzero encoder features required')
    return value/norm


def sample_points(feature_map, points_xy, canvas_shape=(1024, 1024)):
    """Bilinear sample raw final-LN features at patch-center coordinates, then unit."""
    f = np.asarray(feature_map, dtype=np.float64)
    if f.ndim != 3 or not np.isfinite(f).all():
        raise ValueError('Encoder must return finite HxWxD final-LN features')
    gh, gw = f.shape[:2]
    x = np.clip(points_xy[:, 0]*gw/canvas_shape[1]-.5, 0, gw-1)
    y = np.clip(points_xy[:, 1]*gh/canvas_shape[0]-.5, 0, gh-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, gw-1), np.minimum(y0+1, gh-1)
    dx, dy = (x-x0)[:, None], (y-y0)[:, None]
    sampled = ((1-dy)*((1-dx)*f[y0, x0]+dx*f[y0, x1]) +
               dy*((1-dx)*f[y1, x0]+dx*f[y1, x1]))
    return unit(sampled)


def collect_interventions(canvas, geometry, encode, label):
    banks, log = [], []
    for condition in geometry['conditions']:
        paired = []
        xy = mapped_points(geometry['points_xy'], condition)
        for sign, crop_name in ((1, 'positive'), (-1, 'negative')):
            edited = transplant(canvas, geometry[crop_name], condition)
            features = encode(edited, f'{label}.{crop_name}.{len(log)//2}')
            paired.append(sample_points(features, xy, canvas.shape[:2]))
            log.append(dict(condition=condition, reference_crop=crop_name, training_label=sign,
                            labeled_points_xy=xy.tolist(), untouched_query_pixels_labeled=0))
        banks.append(paired)
    # y,T,m,D; no original-query feature is included in these banks.
    return np.asarray(banks, dtype=np.float64).transpose(1, 0, 2, 3), log


def covariance_factors(bank):
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 4 or bank.shape[0] != 2 or bank.shape[1] != 4 or not 4 <= bank.shape[2] <= 32:
        raise ValueError('Require the paired 2x4xm xD bank, with 4<=m<=32')
    if not np.isfinite(bank).all():
        raise ValueError('Nonfinite intervention bank')
    _, conditions, points, dimension = bank.shape
    point_means = bank.mean(axis=1)
    class_means = point_means.mean(axis=1)
    env = (bank-point_means[:, None]).reshape(-1, dimension).T / np.sqrt(2*conditions*points)
    parts = (point_means-class_means[:, None]).reshape(-1, dimension).T / np.sqrt(2*points)
    ridge = max(float(np.sum(env**2)/dimension), Config().minimum_ridge)
    return class_means, env, parts, ridge


def inverse_direction(factor, delta, ridge):
    """Thin-SVD inverse of UU.T+ridge I; never materialize a DxD covariance."""
    factor, delta = np.asarray(factor, np.float64), np.asarray(delta, np.float64)
    if factor.ndim != 2 or delta.shape != (factor.shape[0],) or ridge <= 0:
        raise ValueError('Aligned covariance factor, difference and positive ridge required')
    if not np.isfinite(factor).all() or not np.isfinite(delta).all():
        raise ValueError('Finite statistics required')
    vectors, singular, _ = np.linalg.svd(factor, full_matrices=False)
    projection = vectors.T@delta
    direction = (delta-vectors@projection)/ridge + vectors@(projection/(singular**2+ridge))
    residual = np.linalg.norm(ridge*direction+factor@(factor.T@direction)-delta) / max(np.linalg.norm(delta), 1e-15)
    if not np.isfinite(direction).all() or residual > 1e-6:
        raise RuntimeError(f'Closed-form covariance solve failed: relative residual {residual}')
    return direction, dict(relative_residual=float(residual), factor_columns=factor.shape[1],
                          positive_ridge=float(ridge), maximum_singular_value=float(singular.max(initial=0)))


def normalized_logits(query, direction, means, cfg=Config()):
    delta = means[0]-means[1]
    separation = float(direction@delta)
    if not np.isfinite(separation) or separation <= cfg.minimum_separation:
        return None, dict(reason='nonfinite_or_small_class_separation', separation=separation)
    bias = float(.5*direction@(means[0]+means[1]))
    logits = 2*(query@direction-bias)/separation
    if not np.isfinite(logits).all():
        return None, dict(reason='nonfinite_query_logits', separation=separation)
    return logits, dict(separation=separation, midpoint=bias,
                        intervention_mean_logits=(2*(means@direction-bias)/separation).tolist())


def spatial_edges(query, shape, cfg=Config()):
    ids = np.arange(len(query)).reshape(shape)
    edges = np.concatenate((np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel())),
                            np.column_stack((ids[:-1].ravel(), ids[1:].ravel()))))
    distance = np.maximum(0., 1-np.einsum('ij,ij->i', query[edges[:, 0]], query[edges[:, 1]]))
    sigma = max(float(np.median(distance)) if len(distance) else 0., 1e-4)
    weight = np.exp(-distance/sigma)
    degree_mean = float(2*weight.sum()/len(query))
    capacity = cfg.potts_weight*weight/max(degree_mean, 1e-8)
    return edges, capacity, dict(edge_sigma=sigma, average_weighted_degree=degree_mean,
                                 four_neighbor_edges=len(edges))


def potts_energy(mask, logits, edges, capacity):
    mask, logits = np.asarray(mask, bool).ravel(), np.asarray(logits, np.float64).ravel()
    unary = np.where(mask, np.logaddexp(0, -logits), np.logaddexp(0, logits)).sum()
    return float(unary+np.sum(capacity*(mask[edges[:, 0]] != mask[edges[:, 1]])))


def exact_potts_cut(logits, edges, capacity):
    """Floating-capacity Dinic s-t cut; no integer quantization or local search."""
    logits, capacity = np.asarray(logits, np.float64).ravel(), np.asarray(capacity, np.float64)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    n = len(logits)
    if (not n or capacity.shape != (len(edges),) or not np.isfinite(logits).all() or
            not np.isfinite(capacity).all() or np.any(capacity < 0) or
            np.any(edges < 0) or np.any(edges >= n)):
        raise ValueError('Finite unary costs and nonnegative valid Potts edges required')
    source, sink = n, n+1
    graph = [[] for _ in range(n+2)]

    def add(left, right, cap):
        graph[left].append([right, len(graph[right]), float(cap)])
        graph[right].append([left, len(graph[left])-1, 0.])

    for i in range(n):
        add(source, i, np.logaddexp(0, logits[i]))  # Source side means foreground.
        add(i, sink, np.logaddexp(0, -logits[i]))
    for (left, right), cap in zip(edges, capacity):
        add(int(left), int(right), cap)
        add(int(right), int(left), cap)
    flow, phases = 0., 0
    while True:
        level = [-1]*(n+2)
        level[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for target, _, cap in graph[vertex]:
                if cap > 0 and level[target] < 0:
                    level[target] = level[vertex]+1
                    queue.append(target)
        if level[sink] < 0:
            break
        phases += 1
        pointer = [0]*(n+2)
        # Iterative DFS avoids recursion limits on a full 4096-token graph.
        path_vertices, path_edges = [source], []
        while path_vertices:
            vertex = path_vertices[-1]
            if vertex == sink:
                amount = min(graph[v][i][2] for v, i in path_edges)
                for v, i in path_edges:
                    target, reverse, cap = graph[v][i]
                    graph[v][i][2] = max(0., cap-amount)
                    graph[target][reverse][2] += amount
                flow += amount
                path_vertices, path_edges = [source], []
                continue
            while pointer[vertex] < len(graph[vertex]):
                target, _, cap = graph[vertex][pointer[vertex]]
                if cap > 0 and level[target] == level[vertex]+1:
                    break
                pointer[vertex] += 1
            if pointer[vertex] == len(graph[vertex]):
                level[vertex] = -1
                path_vertices.pop()
                if path_edges:
                    previous, _ = path_edges.pop()
                    pointer[previous] += 1
            else:
                path_edges.append((vertex, pointer[vertex]))
                path_vertices.append(graph[vertex][pointer[vertex]][0])
    reachable, queue = {source}, deque([source])
    while queue:
        vertex = queue.popleft()
        for target, _, cap in graph[vertex]:
            if cap > 0 and target not in reachable:
                reachable.add(target)
                queue.append(target)
    mask = np.array([i in reachable for i in range(n)])
    energy = potts_energy(mask, logits, edges, capacity)
    gap = abs(energy-flow)
    if gap > 1e-8*max(1., abs(energy)):
        raise RuntimeError(f'Cut/maxflow energy certificate failed: {gap}')
    return mask, dict(energy=energy, maximum_flow=flow, certificate_gap=gap, phases=phases,
                     capacity_semantics='floating point, no integer quantization')


def render(coarse, original_shape):
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(np.asarray(coarse, np.float32).copy())[None, None]
    work = functional.interpolate(tensor, (1024, 1024), mode='bilinear', align_corners=False) > .5
    original = functional.interpolate(work.to(torch.float32), original_shape,
                                      mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


def _native_binding(binding):
    if (not isinstance(binding, dict) or binding.get('method') != 'full_foris_native' or
            not isinstance(binding.get('recipe_sha256'), str) or len(binding['recipe_sha256']) != 64):
        raise ValueError('Bind the complete FoRIS native recipe before inference; no default baseline')


def predict(reference_rgb, reference_mask, query_rgb, encoder, native_fallback, *,
            native_binding, encoder_binding, include_ref_canvas=True, cfg=Config()):
    """Actual RGB -> complete work/original masks, with explicit lazy native fallback.

    encoder(RGB_uint8_1024) returns raw final-LN HxWxD. native_fallback receives
    only original R/mask/Q and the already encoded original-Q map (or None);
    returns mask_work, mask_original, info.encoder_forwards. A bound cached
    native callback is legal and must disclose cached rather than deployment cost.
    """
    _fixed(cfg)
    _native_binding(native_binding)
    if not isinstance(encoder_binding, dict) or not encoder_binding.get('producer'):
        raise ValueError('An explicit frozen-encoder producer binding is required')
    started = time.perf_counter()
    r, mask, q = canonical_inputs(reference_rgb, reference_mask, query_rgb, cfg)
    original_shape = np.asarray(query_rgb).shape[:2]
    calls, native, original_query = [], None, None
    arms = ARMS if include_ref_canvas else ARMS[:3]

    def encode(image, role):
        tick = time.perf_counter()
        features = np.asarray(encoder(image), dtype=np.float32)
        if features.ndim != 3 or features.shape[:2] != (64, 64) or not np.isfinite(features).all():
            raise ValueError('Encoder must return finite raw final-LN 64x64xD features')
        calls.append(dict(role=role, seconds=time.perf_counter()-tick, shape=list(features.shape)))
        return features

    def fallback(reason, category):
        nonlocal native
        if native is None:
            native = native_fallback(reference_rgb, reference_mask, query_rgb, original_query)
            if (np.asarray(native['mask_work']).shape != (1024, 1024) or
                    np.asarray(native['mask_original']).shape != original_shape or
                    not np.isin(native['mask_work'], (0, 1)).all() or
                    not np.isin(native['mask_original'], (0, 1)).all() or
                    not isinstance(native.get('info'), dict) or 'encoder_forwards' not in native['info']):
                raise ValueError('Bound native fallback must return both binary masks and actual encoder cost')
        return dict(mask_work=np.asarray(native['mask_work'], bool).copy(),
                    mask_original=np.asarray(native['mask_original'], bool).copy(),
                    info=dict(fallback=True, fallback_category=category, reason=reason,
                              native_binding=native_binding, native_cost=native['info']))

    try:
        geometry = prepare_geometry(r, mask, cfg)
    except GeometryFailure as error:
        output = {arm: fallback(str(error), 'geometry') for arm in arms}
        return dict(arms=output, info=dict(config=asdict(cfg), encoder_binding=encoder_binding,
                    encoder_calls=calls, encoder_forwards=0, query_gt_used=False,
                    wall_seconds=time.perf_counter()-started, real_segmentation_validation=False))
    original_query = encode(q, 'original_query')
    query = unit(original_query).reshape(-1, original_query.shape[-1])
    bank, intervention_log = collect_interventions(q, geometry, encode, 'query_canvas')
    means, env, parts, ridge = covariance_factors(bank)
    delta = means[0]-means[1]
    edges, capacity, graph_info = spatial_edges(query, (64, 64), cfg)
    output, statistics = {}, {}

    def finish(arm, class_means, factor, regularization, mean_only=False):
        difference = class_means[0]-class_means[1]
        if mean_only:
            direction, solve = difference, dict(solver='mean_direction', reused_main_ridge=ridge)
        else:
            direction, solve = inverse_direction(factor, difference, regularization)
        logits, normalization = normalized_logits(query, direction, class_means, cfg)
        statistics[arm] = dict(solver=solve, normalization=normalization,
                               ridge=regularization, mean_vectors=class_means.tolist())
        if logits is None:
            output[arm] = fallback(normalization['reason'], 'statistical')
        else:
            coarse, certificate = exact_potts_cut(logits, edges, capacity)
            work, original = render(coarse.reshape(64, 64), original_shape)
            output[arm] = dict(field=coarse.reshape(64, 64).astype(np.float32),
                               mask_work=work, mask_original=original,
                               info=dict(fallback=False, cut=certificate))

    finish('paired', means, env, ridge)
    finish('mean', means, env, ridge, mean_only=True)
    finish('class_lda', means, np.concatenate((env, parts), axis=1), ridge)
    if include_ref_canvas:
        ref_bank, ref_log = collect_interventions(r, geometry, encode, 'reference_canvas')
        ref_means, ref_env, _, ref_ridge = covariance_factors(ref_bank)
        finish('ref_canvas', ref_means, ref_env, ref_ridge)
        intervention_log += ref_log
    info = dict(config=asdict(cfg), encoder_binding=encoder_binding, native_binding=native_binding,
                positive_box=list(geometry['positive_box']), negative_box=list(geometry['negative_box']),
                canonical_points_xy=geometry['points_xy'].tolist(),
                minimum_scale_coverage=geometry['minimum_scale_coverage'].tolist(),
                conditions=geometry['conditions'], interventions=intervention_log,
                statistics=statistics, graph=graph_info, encoder_calls=calls,
                encoder_forwards=len(calls), primary_encoder_forwards=9,
                additional_ref_canvas_forwards=8 if include_ref_canvas else 0,
                native_fallback_encoder_forwards=0 if native is None else native['info']['encoder_forwards'],
                query_gt_used=False, untouched_query_pixels_labeled=0,
                covariance_object='same_canonical_content_cross_condition_residuals',
                ridge_shared_between_paired_and_class_lda=True,
                wall_seconds=time.perf_counter()-started, real_segmentation_validation=False)
    return dict(arms=output, info=info)


class FrozenTimmRGBEncoder:
    """Adapter for the repository's existing locally loaded TimmDINOv3 wrapper."""
    def __init__(self, model, device='cpu'):
        import torch
        self.model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __call__(self, rgb):
        import torch
        tensor = torch.from_numpy(_rgb(rgb).copy()).permute(2, 0, 1).to(self.device, torch.float32)/255
        mean = torch.tensor((.485, .456, .406), device=self.device)[:, None, None]
        std = torch.tensor((.229, .224, .225), device=self.device)[:, None, None]
        with torch.inference_mode(), torch.autocast(device_type=str(self.device).split(':')[0], enabled=False):
            features = self.model.get_intermediate_layers(((tensor-mean)/std)[None], n=1, reshape=True)[0]
        if tuple(features.shape) != (1, 1024, 64, 64):
            raise ValueError('The supplied M4 DINOv3-L producer requires final 1024x64x64 features')
        return features[0].permute(1, 2, 0).cpu().numpy()
