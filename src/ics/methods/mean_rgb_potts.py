"""Fixed simple control: complete MEAN unary plus RGB128 boundary Potts cut.

No novelty claim, learned parameters, seeds, encoder calls or query labels. This
uses all four-neighbor RGB edges and a globally optimized binary output, rather
than propagating a foreground/background seed basin. The Pro M4 float-capacity
solver and two-stage binary renderer are reused without altering M4 source.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import time
import numpy as np
from .pro_paired_environment import exact_potts_cut, potts_energy, render


@dataclass(frozen=True)
class Config:
    rgb_grid: tuple = (128, 128)
    potts_weight: float = .25
    minimum_edge_sigma: float = 1e-4
    minimum_average_degree: float = 1e-8


CONFIG = dict(method='locked_MEAN_RGB128_boundary_Potts_control_v1',
              role='strong simple complete readout control; not an original mechanism',
              unary='ell=2*bilinear(MEAN64,RGB128)-1; no clipping or re-minmax',
              graph='all undirected four-neighbor RGB128 grid edges',
              rgb_distance='mean squared RGB channel difference, RGB in [0,1]',
              sigma='max(median(all RGB edge distances),1e-4)',
              capacity='.25*exp(-distance/sigma)/max(mean weighted degree,1e-8)',
              parameter_source='Pro M4 Potts .25, median scale and normalization floors; no GT fitting',
              solver='Pro M4 exact_potts_cut; float64 capacities, no integer quantization',
              field_semantics='binary globally optimized cut, not continuous probability',
              renderer='binary RGB128 -> bilinear1024 >.5 -> binary -> bilinear original >.5',
              extra_encoder_calls=0, query_labels_in_inference=False)


def rgb_edge_capacities(rgb, cfg=Config()):
    """All horizontal/vertical edges, with no edge pruning or seed restrictions."""
    if cfg != Config():
        raise ValueError('Fixed control recipe; parameter search is not supported')
    colors = np.asarray(rgb)
    if colors.ndim != 3 or colors.shape[2] != 3 or min(colors.shape[:2]) < 1:
        raise ValueError('Nonempty HxWx3 RGB required')
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float64)/255.
    else:
        colors = colors.astype(np.float64)
        if not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError('RGB floats must be finite in [0,1]')
    height, width = colors.shape[:2]
    ids = np.arange(height*width).reshape(height, width)
    edges = np.concatenate((np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel())),
                            np.column_stack((ids[:-1].ravel(), ids[1:].ravel()))))
    flat = colors.reshape(-1, 3)
    distance = np.mean((flat[edges[:, 0]]-flat[edges[:, 1]])**2, axis=1)
    sigma = max(float(np.median(distance)) if len(distance) else 0., cfg.minimum_edge_sigma)
    affinity = np.exp(-distance/sigma)
    degree_mean = float(2*affinity.sum()/len(flat))
    capacity = cfg.potts_weight*affinity/max(degree_mean, cfg.minimum_average_degree)
    return edges, capacity, dict(rgb_grid=[height, width], vertices=len(flat),
                                 undirected_edges=len(edges), edge_sigma=sigma,
                                 average_weighted_degree=degree_mean,
                                 pairwise_mass=float(capacity.sum()),
                                 zero_capacity_edges=int(np.count_nonzero(capacity == 0)),
                                 all_grid_edges_retained=True)


def resize_unary(base, shape):
    import torch
    import torch.nn.functional as functional
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 2 or not base.size or not np.isfinite(base).all():
        raise ValueError('Finite nonempty MEAN field required')
    tensor = torch.from_numpy(base.copy())[None, None]
    return functional.interpolate(tensor, tuple(shape), mode='bilinear',
                                  align_corners=False)[0, 0].numpy().astype(np.float64)


def predict(q, r, cov, base, rgb, *, original_shape=(1024, 1024), cfg=Config()):
    """Aligned caches plus a recomputed MEAN field -> complete binary cut/masks.

    q/r and cov bind episode alignment; they add no second semantic feature
    readout. Original shape is optional size metadata, never labels/class/fold.
    """
    started = time.perf_counter()
    if cfg != Config():
        raise ValueError('Fixed control recipe; parameter search is not supported')
    q, r, cov, base, rgb = map(np.asarray, (q, r, cov, base, rgb))
    if (q.ndim != 2 or r.shape != q.shape or q.shape[0] != base.size
            or base.shape != (64, 64) or cov.shape != base.shape
            or not np.isfinite(q).all() or not np.isfinite(r).all()
            or not np.isfinite(cov).all() or np.any((cov < 0) | (cov > 1))
            or rgb.shape != (*cfg.rgb_grid, 3)):
        raise ValueError('Aligned finite q/r, coverage/MEAN64 and bound RGB128 required')
    if (len(original_shape) != 2 or any(int(v) != v or v <= 0 for v in original_shape)):
        raise ValueError('Positive integer original H/W required')
    mean128 = (resize_unary(base, cfg.rgb_grid) if cov.any() else
               np.zeros(cfg.rgb_grid, dtype=np.float64))
    logits = 2*mean128-1
    graph_started = time.perf_counter()
    edges, capacity, graph_info = rgb_edge_capacities(rgb, cfg)
    graph_seconds = time.perf_counter()-graph_started
    cut_started = time.perf_counter()
    cut, certificate = exact_potts_cut(logits, edges, capacity)
    field = cut.reshape(cfg.rgb_grid).astype(np.float32)
    cut_seconds = time.perf_counter()-cut_started
    render_started = time.perf_counter()
    work, original = render(field, tuple(int(v) for v in original_shape))
    # This paired unary-only arm controls RGB128 resize/binary/render effects.
    unary = (logits > 0).astype(np.float32)
    unary_work, unary_original = render(unary, tuple(int(v) for v in original_shape))
    render_seconds = time.perf_counter()-render_started
    info = dict(config=asdict(cfg), recipe=CONFIG, graph=graph_info, cut=certificate,
                unary_only_energy=potts_energy(unary, logits, edges, capacity),
                grid_added_pixels=int(np.count_nonzero((field > .5) & (unary <= .5))),
                grid_deleted_pixels=int(np.count_nonzero((field <= .5) & (unary > .5))),
                graph_seconds=graph_seconds, cut_seconds=cut_seconds,
                render_seconds=render_seconds, wall_seconds=time.perf_counter()-started,
                query_gt_used=False, new_encoder_forwards=0,
                empty_reference=bool(not cov.any()),
                native_MEAN_cost_included=False, independent_mechanisms=0,
                segmentation_benefit='unmeasured',
                work_shape=[1024,1024], original_shape=list(map(int, original_shape)))
    return dict(field=field, token_mask=field.astype(bool), mask_work=work, mask_original=original,
                unary_control=unary, unary_mask_work=unary_work, unary_mask_original=unary_original,
                info=info)
