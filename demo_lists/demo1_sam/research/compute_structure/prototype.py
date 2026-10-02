"""CPU research prototypes. Exact-real graph rewrites; no GPU speed claims.

Independent switches isolate implicit attention, companion-free LayerNorm, and
phase-layout upscaling. This file does not alter the measured four-method suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import torch
from torch.nn import functional as F

EXEC = Path(__file__).resolve().parents[2] / "sam_shared_decoder/execution_baselines"
sys.path.insert(0, str(EXEC))
from baselines import heads, unheads, core, plain_attention, shape_plan, ImageCache
from baselines import read_attention as dense_read, write_attention as dense_write
from factor_candidate import FactorCache, FactorState, read_attention, write_factors


def center(x):
    return x - x.mean(-1, keepdim=True)


@dataclass
class ResearchCache:
    factor: FactorCache
    base_variance: tuple
    cross: tuple
    second_matrix: torch.Tensor | None
    second_bias: torch.Tensor | None

    def __getattr__(self, name):
        return getattr(self.factor, name)

    @classmethod
    def build(cls, model, image, dense, position, include_statistics=True, include_phase=True):
        factor = FactorCache.build(model, image, dense, position)
        layers = model.transformer.layers
        bases = [factor.first.base]
        if include_statistics:
            for block in layers:
                bases.append(center(bases[-1]) * block.norm4.weight)
        crosses, variances = [], []
        # A symbolic fixed channel dictionary is transformed by each preceding
        # LN affine map. Crosses are stored per head, preserving coefficient
        # sparsity rather than multiplying an N x K dense zero-padded dictionary.
        for stage in range(len(layers) if include_statistics else 0):
            bc = center(bases[stage])[0]
            variances.append(bc.square().mean(-1)[None, :, None])
            entries = {}
            for origin in range(stage + 1):
                attn = layers[origin].cross_attn_image_to_token
                h, dh = attn.num_heads, attn.internal_dim // attn.num_heads
                matrix = attn.out_proj.weight.T
                bias = attn.out_proj.bias[None]
                for after in range(origin, stage):
                    matrix = center(matrix) * layers[after].norm4.weight
                    bias = center(bias) * layers[after].norm4.weight
                entries[f"write{origin}.all"] = bc @ center(matrix).T
                for head in range(h):
                    entries[f"write{origin}.head{head}"] = entries[f"write{origin}.all"][:, head*dh:(head+1)*dh]
                entries[f"write{origin}.bias"] = bc @ center(bias).T
            for origin in range(stage):
                beta = layers[origin].norm4.bias[None]
                for after in range(origin + 1, stage):
                    beta = center(beta) * layers[after].norm4.weight
                entries[f"beta{origin}"] = bc @ center(beta).T
            crosses.append(entries)
        if include_phase:
            second = model.output_upscaling[3]
            assert second.kernel_size == second.stride == (2, 2)
            assert second.padding == second.output_padding == (0, 0)
            assert second.dilation == (1, 1) and second.groups == 1
            matrix = second.weight.permute(0, 2, 3, 1).reshape(second.in_channels, -1)
            bias = second.bias.repeat(4)
        else:
            matrix, bias = None, None
        return cls(factor, tuple(variances), tuple(crosses), matrix, bias)


@dataclass
class DensePhaseCache:
    first: ImageCache
    conv_matrix: torch.Tensor
    conv_bias: torch.Tensor
    second_matrix: torch.Tensor
    second_bias: torch.Tensor

    @classmethod
    def build(cls, model, image, dense, position):
        # No factor projections or sufficient-statistic caches are constructed
        # for this control. PositionCache should use the dense method's plans.
        first = ImageCache.build(model, image, dense, position)
        conv1, conv2 = model.output_upscaling[0], model.output_upscaling[3]
        for conv in (conv1, conv2):
            assert conv.kernel_size == conv.stride == (2, 2)
            assert conv.padding == conv.output_padding == (0, 0)
            assert conv.dilation == (1, 1) and conv.groups == 1
        matrix1 = conv1.weight.permute(0, 2, 3, 1).reshape(conv1.in_channels, -1)
        matrix2 = conv2.weight.permute(0, 2, 3, 1).reshape(conv2.in_channels, -1)
        return cls(first, matrix1, conv1.bias.repeat(4), matrix2, conv2.bias.repeat(4))


def shared_contract(coeff, image_cross):
    """[B,R,K] and common [N,K] -> [B,N,R], without image batch copies."""
    b, r, k = coeff.shape
    return (coeff.reshape(b*r, k) @ image_cross.T).reshape(b, r, -1).transpose(1, 2)


class StatisticState(FactorState):
    def __init__(self, cache, batch):
        super().__init__(cache, batch)
        del self.dense
        self.cache = cache
        self.segments = []
        self.stage = 0

    def update(self, update_u, update_v, constant, norm, raw_values):
        b, h, t, dh = raw_values.shape
        one = self.s.new_ones(b, 1, 1)
        origin = self.stage
        updates = [[(f"write{origin}.head{head}",
                     raw_values[:, head, :-1] - raw_values[:, head, -1:])]
                   for head in range(h)]
        constant_parts = [(f"write{origin}.all", raw_values[:, :, -1].reshape(b, 1, h*dh)),
                          (f"write{origin}.bias", one)]
        if self.has_constant:
            old_v = torch.cat((self.v[:, :-1], self.v[:, -1:] + constant[:, None]), 1)
            u = torch.cat((self.u, update_u), -1)
            v = torch.cat((old_v, update_v), 1)
            segments = self.segments[:-1] + [self.segments[-1] + constant_parts] + updates
        else:
            u = torch.cat((update_u, torch.ones_like(self.s)), -1)
            v = torch.cat((update_v, constant[:, None]), 1)
            segments = updates + [constant_parts]
        vc = center(v)
        cross = torch.cat([sum(shared_contract(coeff, self.cache.cross[origin][key])
                                  for key, coeff in parts) for parts in segments], -1)
        gram = vc @ vc.transpose(-1, -2) / v.shape[-1]
        # Centered second moment, never E[x^2]-E[x]^2. Near-exact cancellation
        # between the shared base and factors is still numerically hazardous.
        variance = (self.s.square() * self.cache.base_variance[origin]
                    + 2 * self.s * (u * cross).sum(-1, keepdim=True) / v.shape[-1]
                    + ((u @ gram) * u).sum(-1, keepdim=True))
        rstd = torch.rsqrt(variance + norm.eps)
        self.s = self.s * rstd
        self.u = torch.cat((u * rstd, torch.ones_like(rstd)), -1)
        self.v = torch.cat((vc * norm.weight, norm.bias[None, None].expand(b, 1, -1)), 1)
        self.segments = segments + [[(f"beta{origin}", one)]]
        self.has_constant = True
        self.stage += 1


def implicit_read(a, q_input, state, cache, index, backend):
    if index == 0:
        return read_attention(a, q_input, state, cache, index, backend)
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    q = heads(a.q_proj(q_input), h)
    kb, vb = cache.read_base[index]
    # Shared base contractions flatten batch and sparse tokens per head, so a
    # broadcast [B,H,N,Dh] image is never materialized.
    def base_dot(qhead, base):
        return torch.einsum("bhtd,hnd->bhtn", qhead, heads(base, h)[0])
    vk = heads(F.linear(state.v, a.k_proj.weight, None), h)
    vv = heads(F.linear(state.v, a.v_proj.weight, None), h)
    logits = base_dot(q, kb) * state.s[:, None, :, 0].unsqueeze(-2)
    logits = logits + torch.einsum("bhtr,bnr->bhtn", q @ vk.transpose(-1, -2), state.u)
    logits = logits + base_dot(q, cache.first.position.read_k[index])
    # k_proj.bias is spatially constant and cancels in the read softmax.
    probability = torch.softmax(logits / math.sqrt(dh), -1)
    weighted = probability * state.s[:, None, :, 0].unsqueeze(-2)
    output = torch.einsum("bhtn,hnd->bhtd", weighted, heads(vb, h)[0])
    output = output + torch.einsum("bhtn,bnr->bhtr", probability, state.u) @ vv
    output = output + a.v_proj.bias.reshape(1, h, 1, dh)
    return a.out_proj(unheads(output))


def research_write(a, sparse, tokens, state, cache, index, implicit):
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    kh, vh = heads(a.k_proj(sparse + tokens), h), heads(a.v_proj(sparse), h)
    if index == 0 or not implicit:
        q = cache.first.first_q if index == 0 else state.project(
            a.q_proj, cache.write_base[index], cache.first.position.write_q[index])
        logits = heads(q, h) @ kh.transpose(-1, -2)
    else:
        qb = heads(cache.write_base[index], h)[0]
        qpe = heads(cache.first.position.write_q[index], h)[0]
        qv = heads(F.linear(state.v, a.q_proj.weight, None), h)
        logits = torch.einsum("hnd,bhtd->bhnt", qb, kh) * state.s[:, None]
        logits = logits + torch.einsum("bnr,bhrt->bhnt", state.u, qv @ kh.transpose(-1, -2))
        logits = logits + torch.einsum("hnd,bhtd->bhnt", qpe, kh)
        logits = logits + (kh * a.q_proj.bias.reshape(1, h, 1, dh)).sum(-1).unsqueeze(-2)
    probabilities = torch.softmax(logits / math.sqrt(dh), -1)
    output_weight = a.out_proj.weight.reshape(-1, h, dh).permute(1, 2, 0)
    values = vh @ output_weight
    b, _, n, t = probabilities.shape
    u = probabilities[..., :-1].permute(0, 2, 1, 3).reshape(b, n, h*(t-1))
    v = (values[..., :-1, :] - values[..., -1:, :]).reshape(b, h*(t-1), -1)
    constant = values[..., -1, :].sum(1) + a.out_proj.bias
    return u, v, constant, vh


def phase_outputs(model, cache, pre, q):
    """Both nonoverlapping transposed convolutions remain in parent/phase layout.

    Only the four final masks undergo spatial reordering. No 32-channel dense
    upscaling state is copied into NCHW. All LN, GELU, bias and masks remain.
    """
    batch = q.shape[0]
    first = model.output_upscaling[0]
    pre = pre.reshape(batch, -1, 4, first.out_channels)
    norm = model.output_upscaling[1]
    mean = pre.mean(-1, keepdim=True)
    variance = (pre - mean).square().mean(-1, keepdim=True)
    child = (pre - mean) / torch.sqrt(variance + norm.eps)
    child = child * norm.weight + norm.bias
    child = model.output_upscaling[2](child)
    second = child @ cache.second_matrix + cache.second_bias
    second = model.output_upscaling[4](second)
    channels = model.output_upscaling[3].out_channels
    second = second.reshape(batch, -1, 4, 4, channels)
    hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    mask_phases = torch.einsum("bmc,bnuvc->bmnuv", hyper, second)
    height, width = cache.first.height, cache.first.width
    masks = mask_phases.reshape(batch, model.num_mask_tokens, height, width, 2, 2, 2, 2)
    masks = masks.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(batch, model.num_mask_tokens, 4*height, 4*width)
    return masks, model.iou_prediction_head(q[:, 0])


def phase_head(model, cache, state, q):
    pre = state.s * cache.conv_base + state.u @ (state.v @ cache.conv_matrix) + cache.conv_bias
    return phase_outputs(model, cache, pre, q)


def dense_phase_predict(model, cache, sparse, backend="explicit", method="dense_assoc",
                        order="auto", write_output="auto", plans=None):
    """Give the same phase upscaler to the strong dense baseline for fairness.

    This research adapter executes the original dense baseline transformer
    schedule. Use DensePhaseCache so unused factor/variance caches are absent.
    """
    b = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), 0)
    tokens = torch.cat((learned[None].expand(b, -1, -1), sparse), 1)
    q, x = tokens, cache.first.base.expand(b, -1, -1)
    if plans is None:
        plans = shape_plan(model, x.shape[1], tokens.shape[1], method, order, write_output)
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        q = block.norm2(q + dense_read(block.cross_attn_token_to_image, q + tokens,
                                     x, cache.first, i, plans[2*i], backend))
        q = block.norm3(q + block.mlp(q))
        x = block.norm4(x + dense_write(block.cross_attn_image_to_token, q + tokens,
                                       q, x, cache.first, i, plans[2*i+1], backend))
    q = model.transformer.norm_final_attn(q + dense_read(model.transformer.final_attn_token_to_image,
                            q + tokens, x, cache.first, len(model.transformer.layers), plans[-1], backend))
    pre = x @ cache.conv_matrix + cache.conv_bias
    return phase_outputs(model, cache, pre, q)


def predict(model, cache, sparse, backend="explicit", implicit_attention=False,
            statistic_ln=False, phase_layout=False):
    batch = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), 0)
    tokens = torch.cat((learned[None].expand(batch, -1, -1), sparse), 1)
    q = tokens
    state = StatisticState(cache, batch) if statistic_ln else FactorState(cache, batch)
    read = implicit_read if implicit_attention else read_attention
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        q = block.norm2(q + read(block.cross_attn_token_to_image, q + tokens, state, cache, i, backend))
        q = block.norm3(q + block.mlp(q))
        u, v, constant, vh = research_write(block.cross_attn_image_to_token, q, tokens, state, cache, i, implicit_attention)
        if statistic_ln:
            state.update(u, v, constant, block.norm4, vh)
        else:
            state.update(u, v, constant, block.norm4)
    q = model.transformer.norm_final_attn(q + read(model.transformer.final_attn_token_to_image,
                q + tokens, state, cache, len(model.transformer.layers), backend))
    if phase_layout:
        return phase_head(model, cache, state, q)
    pre = state.s * cache.conv_base + state.u @ (state.v @ cache.conv_matrix) + cache.conv_bias
    h, w = cache.first.height, cache.first.width
    channels = model.output_upscaling[0].out_channels
    upscaled = pre.reshape(batch, h, w, 2, 2, channels).permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, 2*h, 2*w)
    for module in model.output_upscaling[1:]:
        upscaled = module(upscaled)
    hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    masks = (hyper @ upscaled.flatten(2)).reshape(batch, model.num_mask_tokens, *upscaled.shape[-2:])
    return masks, model.iou_prediction_head(q[:, 0])
