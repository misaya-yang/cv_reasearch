"""Pro M5: native-QK outside-to-inside value-message extrapolation.

The caller supplies actual frozen blocks and native H20; final-token caches are
insufficient. No model is constructed, downloaded or executed on import. The
adapter contract deliberately requires native QK/LN/RoPE and block forward,
instead of guessing a DINO implementation from attribute names.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import time
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F


ARMS = ('extrapolate', 'endpoint', 'hard_channel', 'native_region', 'zero',
        'response_objectness', 'mass_objectness')


@dataclass(frozen=True)
class Config:
    partition_counts: tuple[int, ...] = (4, 8, 16, 32)
    reference_bg_regions: int = 8
    beta: float = .75
    difference_scale: float = 4.
    descriptor_norm_cap: float = .5
    margin_tau: float = .07
    attention_query_chunk: int = 64
    native_atol: float = 5e-5
    native_rtol: float = 5e-5


def _config(cfg):
    for key, value in asdict(Config()).items():
        if key != 'attention_query_chunk' and getattr(cfg, key) != value:
            raise ValueError('M5 v0 recipe is fixed: ' + key)
    if cfg.attention_query_chunk < 1:
        raise ValueError('Positive attention query chunk required')


def unit(x):
    """Zero-norm vectors stay zero, as specified by the common Pro contract."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm >= 1e-8)


def ward_partitions(features, shape, counts, active=None):
    """Exact adjacent Ward merges, ordered by cost then minimum leaf IDs.

    On a disconnected active mask all adjacent merges are exhausted first. If
    more connected components remain than the requested BG budget, merge the
    pair with closest (unnormalized) feature means, as specified for M5 BG.
    This fallback never occurs for the complete query grid. No score is used.
    """
    x = unit(features)
    h, w = shape
    n = h * w
    if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
        raise ValueError('Finite aligned Ward features required')
    active = np.ones(n, dtype=bool) if active is None else np.asarray(active, dtype=bool).reshape(-1)
    if len(active) != n:
        raise ValueError('Invalid active region mask')
    leaves = np.flatnonzero(active)
    if not len(leaves):
        return {int(k): [] for k in counts}, dict(nonlocal_component_merges=0)
    if any(int(k) != k or k < 1 for k in counts):
        raise ValueError('Positive integer partition counts required')
    targets = {min(int(k), len(leaves)) for k in counts}
    minimum_target = min(targets)
    size = {int(i): 1 for i in leaves}
    sums = {int(i): x[i].copy() for i in leaves}
    minimum = {int(i): int(i) for i in leaves}
    children = {}
    neighbors = {int(i): set() for i in leaves}
    heap = []

    def edge(a, b, nonlocal_edge=False):
        if minimum[a] > minimum[b]:
            a, b = b, a
        difference = sums[a] / size[a] - sums[b] / size[b]
        distance = float(difference @ difference)
        cost = distance if nonlocal_edge else size[a] * size[b] / (size[a] + size[b]) * distance
        heapq.heappush(heap, (cost, minimum[a], minimum[b], a, b))

    for i in leaves:
        i = int(i)
        for j in (i + 1 if i % w + 1 < w else -1, i + w if i // w + 1 < h else -1):
            if j >= 0 and active[j]:
                neighbors[i].add(j); neighbors[j].add(i); edge(i, j)
    snapshots = {}
    alive = set(size)
    next_id = n
    nonlocal_mode = False
    nonlocal_merges = 0
    while True:
        if len(alive) in targets:
            snapshots[len(alive)] = sorted(alive, key=minimum.get)
        if len(alive) <= minimum_target:
            break
        selected = None
        while heap:
            _, _, _, a, b = heapq.heappop(heap)
            if a in alive and b in alive and (nonlocal_mode or b in neighbors[a]):
                selected = (a, b); break
        if selected is None:
            nonlocal_mode = True
            ids = sorted(alive, key=minimum.get)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    edge(a, b, True)
            continue
        a, b = selected
        node = next_id; next_id += 1
        children[node] = (a, b)
        size[node] = size[a] + size[b]
        sums[node] = sums[a] + sums[b]
        minimum[node] = min(minimum[a], minimum[b])
        touching = (neighbors[a] | neighbors[b]) - {a, b}
        neighbors[node] = touching
        alive.remove(a); alive.remove(b); alive.add(node)
        if nonlocal_mode:
            nonlocal_merges += 1
            for other in sorted(alive - {node}, key=minimum.get):
                edge(node, other, True)
        else:
            for other in touching:
                neighbors[other].difference_update((a, b)); neighbors[other].add(node)
                edge(node, other)

    def decode(node):
        stack, pixels = [node], []
        while stack:
            item = stack.pop()
            if item < n:
                pixels.append(item)
            else:
                stack.extend(children[item])
        return np.asarray(sorted(pixels), dtype=np.int64)

    out = {int(k): [decode(node) for node in snapshots[min(int(k), len(leaves))]] for k in counts}
    for partition in out.values():
        ids = np.concatenate(partition)
        if len(ids) != len(leaves) or not np.array_equal(np.sort(ids), leaves):
            raise RuntimeError('Ward partition lost or duplicated active pixels')
    return out, dict(nonlocal_component_merges=nonlocal_merges,
                     disconnected_merge_distance='squared_Euclidean_raw_feature_means',
                     tie_rule='cost_then_ordered_minimum_leaf_ids')


class BlockAdapter(Protocol):
    """Exact native implementation boundary; all states are [tokens, width]."""
    scale: float

    def native_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return [heads,tokens,head_width], actual LN/QKnorm/RoPE/bias included."""
        ...

    def native_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Actual block forward, including native attention and all residuals."""
        ...

    def values(self, hidden: torch.Tensor) -> torch.Tensor:
        """Actual LN1 and V projection, [heads,region_tokens,head_width]."""
        ...

    def finish(self, hidden: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Message [heads,region_tokens,head_width]; actual proj/LS/MLP/residual."""
        ...


@dataclass
class FunctionalBlockAdapter:
    """Inject native callbacks, including implementations with fused/gated FFNs."""
    scale: float
    native_qkv: Callable
    native_forward: Callable
    values: Callable
    finish: Callable


class StandardPreNormBlockAdapter:
    """Strict conventional eval block adapter with explicit native-QK callback.

    The callback is REQUIRED because actual RoPE/masks and QK norms differ
    across official/timm DINO versions. It must call the real implementation.
    No approximate RoPE is supplied. Combined qkv's V slice retains real bias.
    Other block layouts must use FunctionalBlockAdapter, not attribute guesses.
    """
    def __init__(self, block, *, native_qkv, native_forward, contract):
        required = ('norm1', 'attn', 'ls1', 'norm2', 'mlp', 'ls2')
        if block.training or any(not hasattr(block, name) for name in required):
            raise ValueError('An actual supported pre-norm eval block is required')
        if contract != 'pre_norm_attention_ls1_residual_pre_norm_mlp_ls2_residual':
            raise ValueError('Explicit standard residual-layout contract required')
        if any(not hasattr(block.attn, name) for name in ('qkv', 'num_heads', 'scale', 'proj')):
            raise ValueError('Actual combined qkv attention interface required')
        if any(module.training for module in block.modules()):
            raise ValueError('Every native module must be in eval mode')
        self.block, self.native_qkv, self.native_forward = block, native_qkv, native_forward
        self.scale = float(block.attn.scale)

    def values(self, hidden):
        block = self.block
        projected = block.attn.qkv(block.norm1(hidden.unsqueeze(0)))
        _, tokens, tripled = projected.shape
        heads = int(block.attn.num_heads)
        if tripled % (3 * heads):
            raise ValueError('Invalid native qkv dimensions')
        v = projected.reshape(1, tokens, 3, heads, tripled // (3 * heads)).permute(2, 0, 3, 1, 4)[2]
        if hasattr(block.attn, 'v_norm'):
            v = block.attn.v_norm(v)
        return v[0]

    def finish(self, hidden, message):
        block = self.block
        residual = hidden.unsqueeze(0)
        message = message.transpose(0, 1).reshape(1, len(hidden), -1)
        projected = block.attn.proj(message)
        if hasattr(block.attn, 'proj_drop'):
            projected = block.attn.proj_drop(projected)
        projected = block.ls1(projected)
        if hasattr(block, 'drop_path1'):
            projected = block.drop_path1(projected)
        residual = residual + projected
        update = block.ls2(block.mlp(block.norm2(residual)))
        if hasattr(block, 'drop_path2'):
            update = block.drop_path2(update)
        return (residual + update)[0]


@dataclass
class NativeCache:
    h20: torch.Tensor
    patch_ids: torch.Tensor
    qkv: tuple
    native_states: tuple
    adapters: tuple
    final_norm: Callable
    project: Callable
    projected_native: torch.Tensor
    producer: dict


@torch.inference_mode()
def build_native_cache(h20, patch_ids, adapters, final_norm, project, *, producer):
    """Reconstruct four actual native blocks once; retain QKV, not dense attention."""
    if h20.ndim != 2 or h20.dtype not in (torch.float32, torch.float64) or not torch.isfinite(h20).all():
        raise ValueError('Finite FP32 native H20 [all tokens,width] required')
    adapters = tuple(adapters)
    if len(adapters) != 4:
        raise ValueError('Exactly actual blocks21-24 required')
    patch_ids = torch.as_tensor(patch_ids, dtype=torch.long, device=h20.device)
    if (patch_ids.ndim != 1 or not len(patch_ids) or len(torch.unique(patch_ids)) != len(patch_ids)
            or patch_ids.min() < 0 or patch_ids.max() >= len(h20)):
        raise ValueError('Unique exact patch sequence indices required; prefix tokens are external')
    current, cached, states = h20, [], []
    for adapter in adapters:
        if not np.isfinite(adapter.scale) or adapter.scale <= 0:
            raise ValueError('Actual native attention scale required')
        q, k, v = adapter.native_qkv(current)
        if (q.ndim != 3 or q.shape != k.shape or q.shape != v.shape or q.shape[1] != len(current)
                or q.shape[0] * q.shape[2] != current.shape[1]):
            raise ValueError('Native post-QKnorm/post-RoPE QKV shape mismatch')
        for tensor in (q, k, v):
            if tensor.device != current.device or tensor.dtype != current.dtype or not torch.isfinite(tensor).all():
                raise ValueError('Native QKV must preserve dtype/device and be finite')
        cached.append((q.detach(), k.detach(), v.detach()))
        current = adapter.native_forward(current)
        if current.shape != h20.shape or not torch.isfinite(current).all():
            raise ValueError('Actual native block output invalid')
        states.append(current.detach())
    projected = project(final_norm(current))[patch_ids]
    if projected.ndim != 2 or projected.shape[0] != len(patch_ids) or not torch.isfinite(projected).all():
        raise ValueError('Actual final LN then fixed linear episode projection required')
    return NativeCache(h20.detach(), patch_ids, tuple(cached), tuple(states), adapters,
                       final_norm, project, projected.detach(), dict(producer))


def grouped_message(logits, inside_values, outside_values, beta, *, redistribute=True):
    """Grouped logsumexp/softmax; no clipped denominator or 1-p subtraction.

    logits are already split (inside, outside), each [head,query,key].
    Returns message and p_out. Empty outside gives exactly p_out=0.
    """
    inside_logits, outside_logits = logits
    if not 0 <= beta <= 1 or inside_logits.shape[-1] < 1:
        raise ValueError('Inside self-key and beta in [0,1] required')
    if not torch.isfinite(inside_logits).all() or not torch.isfinite(outside_logits).all():
        raise FloatingPointError('Nonfinite native attention logits')
    inner_lse = torch.logsumexp(inside_logits, dim=-1)
    inner_mean = torch.softmax(inside_logits, dim=-1) @ inside_values
    if outside_logits.shape[-1] == 0:
        return inner_mean, torch.zeros_like(inner_lse)
    outer_lse = torch.logsumexp(outside_logits, dim=-1)
    total_lse = torch.logaddexp(inner_lse, outer_lse)
    p_in, p_out = torch.exp(inner_lse - total_lse), torch.exp(outer_lse - total_lse)
    outer_mean = torch.softmax(outside_logits, dim=-1) @ outside_values
    weight = p_in + (1 - beta) * p_out if redistribute else p_in
    message = weight[..., None] * inner_mean + beta * p_out[..., None] * outer_mean
    if not torch.isfinite(message).all():
        raise FloatingPointError('Nonfinite intervened attention message')
    return message, p_out


@torch.inference_mode()
def replay_region(cache, region, beta, cfg=Config(), *, audit_native=False, redistribute=True):
    _config(cfg)
    region = torch.as_tensor(region, dtype=torch.long, device=cache.h20.device)
    if region.ndim != 1 or not len(region) or len(torch.unique(region)) != len(region):
        raise ValueError('A nonempty unique patch ROI is required')
    ids = cache.patch_ids[region]
    all_ids = torch.arange(len(cache.h20), device=ids.device)
    outside = torch.ones(len(cache.h20), dtype=torch.bool, device=ids.device)
    outside[ids] = False
    outside_ids = all_ids[outside]
    hidden = cache.h20[ids].clone()
    maximum_error, mass_sum, mass_count = 0., 0., 0
    for layer, (adapter, (q, k, native_v)) in enumerate(zip(cache.adapters, cache.qkv)):
        values = adapter.values(hidden)
        if values.shape != native_v[:, ids].shape or not torch.isfinite(values).all():
            raise ValueError('Actual branch LN1/V output shape invalid')
        messages = []
        for start in range(0, len(ids), cfg.attention_query_chunk):
            query_ids = ids[start:start + cfg.attention_query_chunk]
            # Only native Q/K are read. CLS/register keys remain in outside_ids.
            logits = (q[:, query_ids] @ k.transpose(-2, -1)) * adapter.scale
            message, p_out = grouped_message((logits[..., ids], logits[..., outside_ids]),
                                            values, native_v[:, outside_ids], beta,
                                            redistribute=redistribute)
            messages.append(message)
            mass_sum += float(p_out.sum().cpu()); mass_count += p_out.numel()
        hidden = adapter.finish(hidden, torch.cat(messages, dim=1))
        if not torch.isfinite(hidden).all():
            raise FloatingPointError('Nonfinite branch state')
        if audit_native:
            expected = cache.native_states[layer][ids]
            error = float((hidden - expected).abs().max().cpu())
            maximum_error = max(maximum_error, error)
            if not torch.allclose(hidden, expected, atol=cfg.native_atol, rtol=cfg.native_rtol):
                raise RuntimeError(f'beta1 actual native replay failed at block{21+layer}: maxabs={error}')
    projected = cache.project(cache.final_norm(hidden))
    if not torch.isfinite(projected).all():
        raise FloatingPointError('Nonfinite final LN/projected region state')
    mean = projected.mean(0).cpu().numpy().astype(np.float64)
    return mean, dict(beta=float(beta),native_maximum_error=maximum_error,
                      mean_outside_mass=mass_sum / max(mass_count, 1), region_tokens=len(ids),
                      processed_suffix_token_layers=4 * len(ids),
                      qk_source='frozen_native_post_norm_post_rope',redistribute=redistribute)


def descriptor(native, endpoint, cfg=Config()):
    native, endpoint = np.asarray(native), np.asarray(endpoint)
    norm = float(np.linalg.norm(native))
    difference = cfg.difference_scale * (native - endpoint)
    eta = min(1., cfg.descriptor_norm_cap * norm / (float(np.linalg.norm(difference)) + 1e-8))
    if norm < 1e-8:
        return dict(old=np.zeros_like(native), extrapolate=np.zeros_like(native),
                    endpoint=np.zeros_like(native), zero_native=True, eta=eta,
                    relative_response=0., difference=difference)
    return dict(old=unit(native), extrapolate=unit(native - eta * difference), endpoint=unit(endpoint),
                zero_native=False, eta=eta, difference=difference,
                relative_response=float(np.linalg.norm(difference)) / max(norm, 1e-8))


@torch.inference_mode()
def region_bank(cache, regions, cfg):
    rows, audits = [], []
    for region in regions:
        native = cache.projected_native[torch.as_tensor(region, device=cache.h20.device)].mean(0).cpu().numpy()
        beta1, audit = replay_region(cache, region, 1., cfg, audit_native=True)
        if not np.allclose(beta1, native, atol=cfg.native_atol, rtol=cfg.native_rtol):
            raise RuntimeError('beta1 pooled descriptor failed actual native audit')
        endpoint, soft_audit = replay_region(cache, region, cfg.beta, cfg)
        hard, hard_audit = replay_region(cache, region, 0., cfg)
        row = descriptor(native, endpoint, cfg)
        row['hard_channel'] = unit(hard) if not row['zero_native'] else np.zeros_like(hard)
        row['mean_outside_mass'] = soft_audit['mean_outside_mass']
        rows.append(row)
        audits.append(dict(native=audit,endpoint=soft_audit,hard=hard_audit,eta=row['eta'],
                           zero_native=row['zero_native']))
    return rows, audits


def margin(vector, reference_fg, reference_bg):
    positive = float(vector @ reference_fg)
    return positive - max((float(vector @ bg) for bg in reference_bg), default=0.)


def fuse(base, delta, cfg=Config()):
    base, delta = np.asarray(base, dtype=np.float32), np.asarray(delta, dtype=np.float64)
    if base.shape != delta.shape or not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError('Finite aligned host and correction fields required')
    # delta=0 returns exactly the same float32 host field before interpolation.
    return base + np.tanh(delta / cfg.margin_tau).astype(np.float32)


def render(field, original_hw):
    field = np.asarray(field, dtype=np.float32)
    if len(original_hw) != 2 or any(int(v) != v or v < 1 for v in original_hw):
        raise ValueError('Positive original query H/W required')
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError('Finite grid field required')
    work = F.interpolate(torch.from_numpy(field.copy())[None, None], (1024, 1024),
                         mode='bilinear', align_corners=False) > .5
    original = F.interpolate(work.float(), tuple(map(int, original_hw)),
                             mode='bilinear', align_corners=False) > .5
    return work[0, 0].numpy(), original[0, 0].numpy()


@torch.inference_mode()
def predict(query_cache, reference_cache, query_features, reference_features, coverage, base,
            original_hw, *, reference_has_foreground, cfg=Config()):
    """Complete M5/control fields and original-size masks, no query GT inputs."""
    _config(cfg)
    started = time.perf_counter()
    base, coverage = np.asarray(base, dtype=np.float32), np.asarray(coverage, dtype=np.float64)
    if (base.ndim != 2 or coverage.shape != base.shape or not np.isfinite(base).all()
            or not np.isfinite(coverage).all() or coverage.min() < 0 or coverage.max() > 1
            or len(query_cache.patch_ids) != base.size or len(reference_cache.patch_ids) != base.size):
        raise ValueError('Aligned native patches, coverage and MEAN host required')
    if not reference_has_foreground:
        fields = {arm: np.zeros_like(base) for arm in ARMS}
        outputs = {arm: render(field, original_hw) for arm, field in fields.items()}
        return dict(fields=fields, work_masks={k:v[0] for k,v in outputs.items()},
                    original_masks={k:v[1] for k,v in outputs.items()},
                    info=dict(empty_reference=True,real_segmentation_benefit='unmeasured'))
    foreground = coverage.ravel() >= .5
    if not foreground.any():
        foreground = coverage.ravel() == coverage.max()
    partitions, query_ward = ward_partitions(query_features, base.shape, cfg.partition_counts)
    bg, reference_ward = ward_partitions(reference_features, base.shape, (cfg.reference_bg_regions,), ~foreground)
    reference_regions = [np.flatnonzero(foreground)] + bg[cfg.reference_bg_regions]
    reference_rows, reference_audits = region_bank(reference_cache, reference_regions, cfg)
    deltas = {arm: np.zeros(base.size, dtype=np.float64) for arm in ARMS}
    query_audits = []
    for count in cfg.partition_counts:
        regions = partitions[count]
        query_rows, audits = region_bank(query_cache, regions, cfg)
        query_audits.append(dict(partition_count=count, actual_regions=len(regions), regions=audits))
        for region, row in zip(regions, query_rows):
            old = margin(row['old'], reference_rows[0]['old'], [x['old'] for x in reference_rows[1:]])
            for arm in ('extrapolate', 'endpoint', 'hard_channel'):
                correction = margin(row[arm], reference_rows[0][arm], [x[arm] for x in reference_rows[1:]]) - old
                # Any invalid native query/FG descriptor means no correction.
                if row['zero_native'] or reference_rows[0]['zero_native']:
                    correction = 0.
                deltas[arm][region] += correction / len(cfg.partition_counts)
            deltas['native_region'][region] += old / len(cfg.partition_counts)
            deltas['response_objectness'][region] += -cfg.margin_tau * row['relative_response'] / len(cfg.partition_counts)
            deltas['mass_objectness'][region] += -cfg.margin_tau * row['mean_outside_mass'] / len(cfg.partition_counts)
    fields = {arm: fuse(base, delta.reshape(base.shape), cfg) for arm, delta in deltas.items()}
    outputs = {arm: render(field, original_hw) for arm, field in fields.items()}
    return dict(fields=fields,deltas={arm:delta.reshape(base.shape) for arm,delta in deltas.items()},
                work_masks={k:v[0] for k,v in outputs.items()},original_masks={k:v[1] for k,v in outputs.items()},
                info=dict(config=asdict(cfg),query_ward=query_ward,reference_ward=reference_ward,
                          query_audits=query_audits,reference_audits=reference_audits,
                          query_processed_area=sum(sum(len(r) for r in p) for p in partitions.values()),
                          reference_processed_area=sum(map(len,reference_regions)),
                          primary_additional_suffix_token_layers=4*(4*base.size+base.size),
                          executed_replay_multiplier=3,native_cache_build_token_layers=8*len(query_cache.h20),
                          cache_producers=dict(query=query_cache.producer,reference=reference_cache.producer),
                          wall_seconds=time.perf_counter()-started,query_gt_used=False,
                          real_segmentation_benefit='unmeasured',native_asset_validation='adapter_audit_only'))
