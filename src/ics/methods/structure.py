"""Full-image correspondence compatibility, without query labels or model loading.

The complete candidate alternates a cross-image appearance likelihood with joint
within-image correspondence messages. It does not optimize or gate an RCG mask.
This is an experimental inference rule, not an established semantic guarantee.
"""
from __future__ import annotations

import time
import numpy as np
import torch

CONFIG = {
    'name': 'structure_correspondence_v1',
    'temperature': 0.05,
    'neighbors': 12,
    'iterations': 4,
    'message_strength': 0.5,
    'query_block': 256,
    'extra_encoder_forwards': 0,
    'layers': ['cached_final'],
    'selection': 'fixed_before_query_GT',
}


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _prepare(q, r, cov, score, device):
    q, r = _tensor(q, device), _tensor(r, device)
    coverage = _tensor(cov, device).reshape(-1)
    shape = tuple(np.shape(score))
    if len(shape) != 2 or q.ndim != 2 or r.ndim != 2:
        raise ValueError('Expected q/r token matrices and a two-dimensional score grid')
    if q.shape[0] != shape[0] * shape[1] or r.shape[0] != coverage.numel():
        raise ValueError('Token counts do not agree with score/reference coverage grids')
    if q.shape[1] != r.shape[1] or q.shape[0] < 2 or r.shape[0] < 2:
        raise ValueError('Feature dimensions must match and each image needs two tokens')
    if not all(torch.isfinite(x).all().item() for x in (q, r, coverage)):
        raise ValueError('Nonfinite input')
    if coverage.min() < 0 or coverage.max() > 1:
        raise ValueError('Reference coverage must lie in [0, 1]')
    q = torch.nn.functional.normalize(q, dim=1)
    r = torch.nn.functional.normalize(r, dim=1)
    fg, bg = coverage.sum(), (1 - coverage).sum()
    if fg <= 0 or bg <= 0:
        raise ValueError('Both reference foreground and background are required')
    # Equal role priors, but no constraint on the query foreground fraction.
    fg_weight = coverage / fg
    bg_weight = (1 - coverage) / bg
    prior = (fg_weight + bg_weight) / 2
    label = fg_weight / (fg_weight + bg_weight)
    logits = (q @ r.T) / CONFIG['temperature'] + prior.log()[None]
    assignment = logits.softmax(1)
    return q, r, prior, label, logits, assignment, shape


def _graph(x):
    """A directed row-stochastic feature kNN graph, excluding self edges."""
    n = len(x)
    k = min(CONFIG['neighbors'], n - 1)
    rows, cols, values = [], [], []
    for start in range(0, n, CONFIG['query_block']):
        stop = min(start + CONFIG['query_block'], n)
        affinity = x[start:stop] @ x.T
        idx = torch.arange(start, stop, device=x.device)
        affinity[torch.arange(stop - start, device=x.device), idx] = -torch.inf
        val, col = affinity.topk(k, dim=1)
        # Relative weights avoid a feature-density-dependent row mass.
        val = (val / CONFIG['temperature']).softmax(1)
        rows.append(idx[:, None].expand_as(col).reshape(-1))
        cols.append(col.reshape(-1))
        values.append(val.reshape(-1))
    edge = torch.stack((torch.cat(rows), torch.cat(cols)))
    return torch.sparse_coo_tensor(edge, torch.cat(values), (n, n)).coalesce()


def _entropy(t):
    return float((-(t * t.clamp_min(1e-20).log()).sum(1)).mean().item())


def _run(q, r, cov, score, device, mode, return_context=False):
    start = time.perf_counter()
    q, r, prior, label, logits, t, shape = _prepare(q, r, cov, score, device)
    initial = t
    history = [{'iteration': 0, 'entropy': _entropy(t),
                'foreground_mass': float((t @ label).mean().item())}]
    if mode != 'independent':
        aq, ar = _graph(q), _graph(r)
        for iteration in range(CONFIG['iterations']):
            message = torch.sparse.mm(aq, t)
            message = torch.sparse.mm(ar, message.T.contiguous()).T.contiguous()
            message = message / message.sum(1, keepdim=True).clamp_min(1e-20)
            if mode == 'diffusion':
                t = (initial + message) / 2
            else:
                # A message is compared with the same graph applied to the source
                # prior. This removes a static source-degree/role prior effect.
                null = torch.sparse.mm(ar, prior[:, None]).T
                t = (logits + CONFIG['message_strength'] *
                     (message.clamp_min(1e-20).log() - null.clamp_min(1e-20).log())).softmax(1)
            history.append({'iteration': iteration + 1, 'entropy': _entropy(t),
                            'foreground_mass': float((t @ label).mean().item())})
    field = (t @ label).reshape(shape).detach().cpu().numpy().astype(np.float32)
    info = {'config': dict(CONFIG), 'mode': mode, 'history': history,
                   'elapsed_seconds': time.perf_counter() - start,
                   'extra_encoder_forwards': 0, 'query_gt_used': False,
                   'source_tokens': len(r), 'query_tokens': len(q),
                   'correspondence_matrix_mib': t.numel() * t.element_size() / 2**20}
    if return_context:
        return field, info, (logits, label, shape)
    return field, info


@torch.inference_mode()
def predict(q, r, cov, score, *, device='cpu', extras=None):
    return _run(q, r, cov, score, device, 'compatibility')


@torch.inference_mode()
def control(q, r, cov, score, *, device='cpu', extras=None):
    """Same reference/query inputs, class-balanced independent kernel transfer."""
    return _run(q, r, cov, score, device, 'independent')


@torch.inference_mode()
def diffusion_control(q, r, cov, score, *, device='cpu', extras=None):
    """Same graphs/iterations: separable diffusion without likelihood feedback."""
    return _run(q, r, cov, score, device, 'diffusion')


@torch.inference_mode()
def entropy_control(q, r, cov, score, *, device='cpu', extras=None):
    """Independent transfer with its mean assignment entropy matched to primary.

    This uses no labels: one scalar temperature is chosen from the primary's
    assignment entropy. It distinguishes correspondence information from generic
    sharpening. It deliberately includes the primary's computation in its cost.
    """
    started = time.perf_counter()
    _, primary_info, context = _run(q, r, cov, score, device, 'compatibility', return_context=True)
    logits, label, shape = context
    target = primary_info['history'][-1]['entropy']
    low, high = 0.01, 20.0
    for _ in range(16):
        scale = (low + high) / 2
        t = (logits * scale).softmax(1)
        if _entropy(t) > target:
            low = scale
        else:
            high = scale
    scale = (low + high) / 2
    t = (logits * scale).softmax(1)
    field = (t @ label).reshape(shape).cpu().numpy().astype(np.float32)
    return field, {'config': dict(CONFIG), 'mode': 'matched_entropy_independent',
                   'entropy_target': target, 'entropy_observed': _entropy(t),
                   'inverse_temperature_multiplier': scale,
                   'query_gt_used': False, 'extra_encoder_forwards': 0,
                   'elapsed_seconds': time.perf_counter() - started}


additional_controls = {
    'diffusion': diffusion_control,
    'entropy': entropy_control,
}
