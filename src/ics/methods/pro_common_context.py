"""Pro M1: real-block, read-only shared-query suffix and complete graph readout.

The adapter invokes the encoder's actual blocks, with their actual RoPE arguments,
and intercepts fused SDPA. It never reconstructs Eva/DINO projections by guesswork.
Native capture and audit are required before using a new encoder implementation.
SDPA interception is process-global: model calls must be serialized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .intervention import _SDPA_LOCK


@dataclass(frozen=True)
class Config:
    start_block: int = 20  # zero-based: pre block21 = raw output of block20
    suffix_blocks: int = 4
    probes_per_role: int = 4
    temperature: float = .07
    chunk_size: int = 64
    graph_lambda: float = 16.
    audit_atol: float = 1e-5
    audit_rtol: float = 1e-4


def _clone(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_clone(v) for v in x)
    if isinstance(x, list):
        return [_clone(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


def _digest(tensors):
    h = hashlib.sha256()
    for value in tensors:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_receipt(model):
    files = {}
    for value in (type(model), type(model.blocks[0]), type(model.blocks[0].attn)):
        filename = inspect.getsourcefile(value)
        if filename and Path(filename).is_file():
            files[str(Path(filename).resolve())] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return dict(model_class=f'{type(model).__module__}.{type(model).__name__}',
                block_class=f'{type(model.blocks[0]).__module__}.{type(model.blocks[0]).__name__}',
                source_sha256=files, adapter='actual_block_forward_and_actual_SDPA_QKV')


def _validate_model(model, cfg):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Require frozen evaluation model')
    if cfg.start_block < 0 or cfg.start_block + cfg.suffix_blocks != len(model.blocks):
        raise ValueError('Suffix must end at the actual last block')
    if cfg.chunk_size < 1 or cfg.probes_per_role != 4 or cfg.temperature != .07:
        raise ValueError('Invalid chunk or changed Pro M1 v0 parameters')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Require FP32 actual model parameters')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise ValueError('Disable autocast for the bound FP32 native/suffix contract')
    for block in model.blocks[cfg.start_block:]:
        if not getattr(block.attn, 'fused_attn', False):
            raise RuntimeError('Actual attention must use fused SDPA; unfused implementation is not silently replaced')
    if not hasattr(model, 'norm'):
        raise RuntimeError('Missing actual final token norm')


@torch.inference_mode()
def capture_native(model, images, cfg=Config()):
    """Capture actual paired RGB forward, raw pre-block states, and post-RoPE KV.

    images is [reference,query]. Raw capture is from block input hooks, never the
    norm=True intermediate-layer wrapper. Nonzero SDPA mask/dropout fails closed.
    """
    _validate_model(model, cfg)
    if images.ndim != 4 or images.shape[0] != 2 or images.dtype != torch.float32:
        raise ValueError('Require FP32 paired [reference,query] images')
    if not _SDPA_LOCK.acquire(blocking=False):
        raise RuntimeError('Concurrent model call during native capture')
    original = F.scaled_dot_product_attention
    layers, handles, current = {}, [], {'index': None}

    def pre(index):
        def hook(module, args, kwargs):
            current['index'] = index
            if index >= cfg.start_block:
                if (not args or not torch.is_tensor(args[0]) or args[0].ndim != 3
                        or args[0].shape[0] != 2 or args[0].dtype != torch.float32):
                    raise RuntimeError('Require actual block(x, ...), paired NLC input')
                layers[index] = dict(args=_clone(args), kwargs=_clone(kwargs), calls=0)
        return hook

    def sdpa(q, k, v, attn_mask=None, dropout_p=0., is_causal=False, **kwargs):
        index = current['index']
        if index in layers:
            if (q.ndim != 4 or q.shape != k.shape or k.shape != v.shape or q.shape[0] != 2
                    or q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32
                    or attn_mask is not None or dropout_p != 0 or is_causal):
                raise RuntimeError('Unsupported native SDPA layout/mask/dropout; audit the actual Eva variant')
            layer = layers[index]
            layer.update(k=k[1:2].detach().clone(), v=v[1:2].detach().clone(),
                         sdpa_kwargs=_clone(kwargs), heads=int(q.shape[1]))
            layer['calls'] += 1
        return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, **kwargs)

    started = time.perf_counter()
    try:
        for index, block in enumerate(model.blocks):
            handles.append(block.register_forward_pre_hook(pre(index), with_kwargs=True))
        F.scaled_dot_product_attention = sdpa
        final = model.forward_features(images)
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()
        _SDPA_LOCK.release()
    if not torch.is_tensor(final) or final.ndim != 3 or final.shape[0] != 2:
        raise RuntimeError('forward_features must return actual final normalized NLC token tensor')
    if any(layer['calls'] != 1 for layer in layers.values()) or len(layers) != cfg.suffix_blocks:
        raise RuntimeError('Expected exactly one captured SDPA per suffix block')
    prefix = int(model.num_prefix_tokens)
    n = int(final.shape[1]) - prefix
    side = math.isqrt(n)
    if side * side != n or n <= 9 or prefix < 0:
        raise RuntimeError('Require square patch grid with keys remaining after 3x3 exclusion')
    h20 = layers[cfg.start_block]['args'][0]
    if h20.shape != final.shape:
        raise RuntimeError('Native token count changed through suffix')
    tensors = [h20, final] + [t for layer in layers.values() for t in (layer['k'], layer['v'])]
    return dict(layers=layers, raw_h20=h20, final=final.detach().clone(), prefix=prefix,
                side=side, memory_hash=_digest(tensors), capture_seconds=time.perf_counter()-started,
                source=source_receipt(model))


def memory_hash(cache):
    return _digest([cache['raw_h20'], cache['final']] +
                   [t for layer in cache['layers'].values() for t in (layer['args'][0], layer['k'], layer['v'])])


@torch.inference_mode()
def replay(model, cache, patch_state, cfg=Config(), *, uniform=False, exclude_local=True):
    """One probe at every Q spatial slot, actual block operations, fixed Q memory.

    All probe locations are batched through real blocks; only attention matrices
    are chunked. This correctness-first adapter recomputes actual paired QKV/MLP,
    including unused reference work. Report its measured cost; no FLOPs claim.
    """
    n, prefix = cache['side']**2, cache['prefix']
    if patch_state.shape != (n, cache['raw_h20'].shape[-1]) or not torch.isfinite(patch_state).all():
        raise ValueError('Expected finite raw patch state at every Q slot')
    p = patch_state.detach().clone()
    original = F.scaled_dot_product_attention
    ids = torch.arange(n, device=p.device)
    rows, cols = ids // cache['side'], ids % cache['side']
    for index, layer in cache['layers'].items():
        calls = 0

        def sdpa(q, k, v, attn_mask=None, dropout_p=0., is_causal=False, **kwargs):
            nonlocal calls
            calls += 1
            if (q.shape != k.shape or q.shape != v.shape or q.shape[0] != 2
                    or attn_mask is not None or dropout_p != 0 or is_causal
                    or kwargs != layer['sdpa_kwargs']):
                raise RuntimeError('Suffix SDPA signature changed relative to native capture')
            out = torch.zeros_like(q)
            for start in range(0, n, cfg.chunk_size):
                end = min(n, start+cfg.chunk_size)
                allowed = torch.ones((end-start, prefix+n), dtype=torch.bool, device=q.device)
                if exclude_local:
                    allowed[:, :prefix] = False
                    nearby = ((rows[start:end, None]-rows[None, :]).abs() <= 1) & ((cols[start:end, None]-cols[None, :]).abs() <= 1)
                    allowed[:, prefix:] = ~nearby
                if not bool(allowed.any(1).all()):
                    raise RuntimeError('No legal query keys after exclusion')
                if uniform:
                    weight = allowed.to(q.dtype) / allowed.sum(1, keepdim=True)
                    value = torch.einsum('ij,bhjd->bhid', weight, layer['v'])
                else:
                    value = original(q[1:2, :, prefix+start:prefix+end], layer['k'], layer['v'],
                                     attn_mask=allowed[None, None], dropout_p=0., is_causal=False, **kwargs)
                out[1:2, :, prefix+start:prefix+end] = value
            return out

        args = list(_clone(layer['args']))
        args[0][1, prefix:] = p
        if not _SDPA_LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent model call during suffix replay')
        try:
            F.scaled_dot_product_attention = sdpa
            result = model.blocks[index](*args, **_clone(layer['kwargs']))
        finally:
            F.scaled_dot_product_attention = original
            _SDPA_LOCK.release()
        if calls != 1 or not torch.is_tensor(result) or result.shape != args[0].shape:
            raise RuntimeError('Actual block contract changed; expected one SDPA and NLC output')
        p = result[1, prefix:].detach().clone()
    return model.norm(p)


@torch.inference_mode()
def audit_native(model, cache, cfg=Config()):
    before = memory_hash(cache)
    actual = replay(model, cache, cache['raw_h20'][1, cache['prefix']:], cfg, exclude_local=False)
    expected = cache['final'][1, cache['prefix']:]
    difference = float((actual-expected).abs().max())
    passed = bool(torch.allclose(actual, expected, atol=cfg.audit_atol, rtol=cfg.audit_rtol))
    after = memory_hash(cache)
    if not passed or before != after:
        raise RuntimeError(f'Native self-probe audit failed: maxabs={difference}, memory_unchanged={before == after}')
    return dict(native_self_maxabs=difference, atol=cfg.audit_atol, rtol=cfg.audit_rtol,
                memory_unchanged=True, local_and_prefix_exclusion=False,
                level='actual_runtime_encoder; toy only if supplied model is toy')


def _project(x, basis):
    if basis is not None:
        x = x - (x @ basis) @ basis.T
    return F.normalize(x, dim=-1)


def select_probes(features, coverage, foreground, maximum=4):
    cov = torch.as_tensor(coverage, device=features.device).reshape(-1)
    if foreground:
        candidates = torch.where(cov >= .9)[0]
        if not len(candidates):
            candidates = torch.where(cov == cov.max())[0]
    else:
        if bool((cov == 1).all()):
            return []
        candidates = torch.where(cov <= .1)[0]
        if not len(candidates):
            candidates = torch.where(cov == cov.min())[0]
    points = features[candidates]
    mean = F.normalize(points.mean(0), dim=0)
    chosen = [int(torch.argmax(points @ mean))]
    while len(chosen) < min(maximum, len(candidates)):
        distance = (1-points @ points[chosen].T).min(1).values
        distance[chosen] = -torch.inf
        chosen.append(int(torch.argmax(distance)))
    return [int(candidates[j]) for j in chosen]


def _margin(similarity, foreground_count, temperature):
    fg = similarity[:, :foreground_count]
    result = temperature * (torch.logsumexp(fg/temperature, dim=1)-math.log(foreground_count))
    bg = similarity[:, foreground_count:]
    if bg.shape[1]:
        result -= temperature * (torch.logsumexp(bg/temperature, dim=1)-math.log(bg.shape[1]))
    return result


@torch.inference_mode()
def guides(model, cache, coverage, cfg=Config(), *, basis=None):
    cov = np.asarray(coverage)
    n, prefix = cache['side']**2, cache['prefix']
    if cov.size != n or not np.isfinite(cov).all() or cov.min() < 0 or cov.max() > 1:
        raise ValueError('Coverage must be finite mask fractions aligned to patches')
    if cov.max() == 0:
        return None, dict(empty_reference=True)
    if basis is not None:
        basis = torch.as_tensor(basis, dtype=torch.float32, device=cache['final'].device)
        if basis.ndim != 2 or basis.shape[0] != cache['final'].shape[-1] or not torch.isfinite(basis).all():
            raise ValueError('Invalid fixed native projection basis')
    reference = _project(cache['final'][0, prefix:], basis)
    query = _project(cache['final'][1, prefix:], basis)
    foreground = select_probes(reference, cov, True, cfg.probes_per_role)
    background = select_probes(reference, cov, False, cfg.probes_per_role)
    selected = foreground + background
    plain = _margin(query @ reference[selected].T, len(foreground), cfg.temperature)
    pure = np.flatnonzero(cov.ravel() >= .9)
    if not len(pure):
        pure = np.flatnonzero(cov.ravel() == cov.max())
    mean = query @ F.normalize(reference[pure].mean(0), dim=0)
    fields, timings = {'plain': plain.cpu().numpy(), 'mean-unit': mean.cpu().numpy()}, {}
    before = memory_hash(cache)
    for arm, uniform in (('ctx', False), ('uniform', True)):
        started = time.perf_counter()
        own = _project(replay(model, cache, cache['raw_h20'][1, prefix:], cfg, uniform=uniform), basis)
        similarities = []
        for probe in selected:
            raw = cache['raw_h20'][0, prefix+probe].expand(n, -1)
            transformed = _project(replay(model, cache, raw, cfg, uniform=uniform), basis)
            similarities.append((own*transformed).sum(-1))
        field = _margin(torch.stack(similarities, dim=1), len(foreground), cfg.temperature)
        fields[arm] = field.cpu().numpy()
        timings[arm] = time.perf_counter()-started
    if memory_hash(cache) != before:
        raise RuntimeError('Native memory was modified')
    return fields, dict(config=asdict(cfg), foreground_probe_ids=foreground, background_probe_ids=background,
                        trajectories=1+len(selected), suffix_seconds=timings, memory_unchanged=True,
                        projection_fixed=True, query_gt_used=False)


def graph_readout(query_features, score, guide_fields, cfg=Config()):
    """Shared Pro rho=1 graph readout, distinct from historical MEAN rho=.25."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from .rcg import minmax, rank
    query = F.normalize(torch.as_tensor(query_features, dtype=torch.float32, device='cpu'), dim=-1)
    n = len(query)
    side = math.isqrt(n)
    raw_score = np.asarray(score)
    if side*side != n or raw_score.size != n or n <= 20 or not np.isfinite(raw_score).all():
        raise ValueError('Require square score grid and >20 query patches')
    if not torch.isfinite(query).all():
        raise ValueError('Nonfinite graph descriptors')
    s = minmax(raw_score).reshape(-1)
    similarity = query @ query.T
    similarity.fill_diagonal_(-2)
    values, indices = similarity.topk(20, dim=1)
    distances = (1-values).clamp_min(0)
    weights = torch.exp(-distances/distances[:, -1:].clamp_min(1e-6)).numpy().ravel()
    graph = sparse.csr_matrix((weights, (np.repeat(np.arange(n), 20), indices.numpy().ravel())), shape=(n, n))
    graph = graph.multiply(graph.T)
    graph.data = np.sqrt(graph.data)
    graph = graph/max(float(np.asarray(graph.sum(1)).mean()), 1e-8)
    fidelity = .1+np.abs(2*s-1)
    fidelity = (fidelity/fidelity.mean()).astype(np.float64)
    matrix = sparse.diags(fidelity)+cfg.graph_lambda*(sparse.diags(np.asarray(graph.sum(1)).ravel())-graph)
    output, residuals = {}, {}
    for arm, field in guide_fields.items():
        if np.asarray(field).size != n or not np.isfinite(field).all():
            raise ValueError('Nonfinite or misaligned guide')
        y = (s+rank(np.asarray(field))-rank(s)).astype(np.float64)
        rhs = fidelity*y
        z, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
        if status:
            raise RuntimeError(f'M1 graph CG failed in {arm}: {status}')
        output[arm] = z.reshape(side, side).astype(np.float32)
        residuals[arm] = float(np.linalg.norm(matrix@z-rhs)/max(np.linalg.norm(rhs), 1e-12))
    return output, dict(rho=1., graph_lambda=cfg.graph_lambda, residuals=residuals,
                        graph_storage_dtype=str(graph.dtype), historical_MEAN_score_reused=False)


def render(field, original_hw, working_size=1024):
    tensor = torch.as_tensor(np.ascontiguousarray(field), dtype=torch.float32)[None, None]
    work = F.interpolate(tensor, (working_size, working_size), mode='bilinear', align_corners=False) > .5
    original = F.interpolate(work.float(), tuple(original_hw), mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


@torch.inference_mode()
def predict(model, images, coverage, score, original_hw, cfg=Config(), *, basis=None):
    started = time.perf_counter()
    cache = capture_native(model, images, cfg)
    audit = audit_native(model, cache, cfg)
    gs, info = guides(model, cache, coverage, cfg, basis=basis)
    if gs is None:
        fields = {arm: np.zeros_like(np.asarray(score), dtype=np.float32) for arm in ('ctx', 'plain', 'mean-unit', 'uniform')}
        graph_info = dict(empty_reference=True)
    else:
        q = _project(cache['final'][1, cache['prefix']:], None if basis is None else torch.as_tensor(basis, device=images.device, dtype=torch.float32))
        fields, graph_info = graph_readout(q.cpu(), score, gs, cfg)
    masks = {arm: render(field, original_hw) for arm, field in fields.items()}
    return dict(fields=fields, masks=masks, guides=gs,
                info=dict(probe_info=info, graph=graph_info, native_audit=audit, source=cache['source'],
                          capture_seconds=cache['capture_seconds'], total_seconds=time.perf_counter()-started,
                          added_native_audit_suffix=True, actual_segmentation_quality='unmeasured'))
