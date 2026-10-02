"""Inference-only projected-factor SAM candidate with a dense LN companion.

No diagnostic reconstructions, error reductions, duplicate write GEMMs or
LayerNorm calls. Its optional state trace is used only by the separate validator.
This is the projected-factor schedule, not an implicit-factor attention kernel.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from baselines import ImageCache, core, heads, plain_attention, unheads


@dataclass
class FactorCache:
    first: ImageCache
    read_base: tuple
    write_base: tuple
    conv_matrix: torch.Tensor
    conv_base: torch.Tensor
    conv_bias: torch.Tensor

    @classmethod
    def build(cls, model, image, dense, position):
        first = ImageCache.build(model, image, dense, position)
        bases = [first.base]
        for block in model.transformer.layers:
            old = bases[-1]
            bases.append((old - old.mean(-1, keepdim=True)) * block.norm4.weight)
        reads = [block.cross_attn_token_to_image for block in model.transformer.layers]
        reads.append(model.transformer.final_attn_token_to_image)
        read_base = tuple(None if i == 0 else (
            F.linear(bases[i], a.k_proj.weight, None),
            F.linear(bases[i], a.v_proj.weight, None)) for i, a in enumerate(reads))
        write_base = tuple(None if i == 0 else F.linear(bases[i],
            block.cross_attn_image_to_token.q_proj.weight, None)
            for i, block in enumerate(model.transformer.layers))
        conv = model.output_upscaling[0]
        if not (conv.kernel_size == (2, 2) and conv.stride == (2, 2) and
                conv.padding == (0, 0) and conv.output_padding == (0, 0) and
                conv.dilation == (1, 1) and conv.groups == 1):
            raise ValueError("Candidate requires official nonoverlapping first upscaler")
        matrix = conv.weight.permute(0, 2, 3, 1).reshape(conv.in_channels, -1)
        return cls(first, read_base, write_base, matrix, bases[-1] @ matrix,
                   conv.bias.repeat(4))


class FactorState:
    def __init__(self, cache, batch):
        base = cache.first.base
        self.s = base.new_ones(batch, base.shape[1], 1)
        self.u = base.new_empty(batch, base.shape[1], 0)
        self.v = base.new_empty(batch, 0, base.shape[-1])
        self.dense = base.expand(batch, -1, -1)
        self.has_constant = False

    def project(self, linear, base_projection, pe_projection=None):
        # Called only after the first LN; rank0 uses native-order joint caches.
        value = self.s * base_projection + self.u @ F.linear(self.v, linear.weight, None)
        if pe_projection is not None:
            value = value + pe_projection  # PE is independent of the row scale.
        return value + linear.bias

    def update(self, update_u, update_v, constant, norm):
        # Exactly one new-write GEMM and one native LN, whose normalized output
        # and rstd are both consumed. No complete factor reconstruction occurs.
        update = update_u @ update_v + constant[:, None, :]
        normalized, _mean, rstd = torch.native_layer_norm(self.dense + update,
            norm.normalized_shape, norm.weight, norm.bias, norm.eps)
        if self.has_constant:
            old_v = torch.cat((self.v[:, :-1], self.v[:, -1:] + constant[:, None]), dim=1)
            u = torch.cat((self.u, update_u), dim=-1)
            v = torch.cat((old_v, update_v), dim=1)
        else:
            u = torch.cat((update_u, torch.ones_like(self.s)), dim=-1)
            v = torch.cat((update_v, constant[:, None]), dim=1)
        self.s = self.s * rstd
        self.u = torch.cat((u * rstd, torch.ones_like(rstd)), dim=-1)
        centered = (v - v.mean(-1, keepdim=True)) * norm.weight
        self.v = torch.cat((centered, norm.bias[None, None].expand(v.shape[0], 1, -1)), dim=1)
        self.has_constant = True
        self.dense = normalized


def read_attention(a, q_input, state, cache, index, backend):
    if index == 0:
        k, v = cache.first.first_k, cache.first.first_v
    else:
        kb, vb = cache.read_base[index]
        k = state.project(a.k_proj, kb, cache.first.position.read_k[index])
        v = state.project(a.v_proj, vb)
    qh = heads(a.q_proj(q_input), a.num_heads)
    return a.out_proj(unheads(core(qh, heads(k, a.num_heads), heads(v, a.num_heads), backend)))


def write_factors(a, sparse, tokens, state, cache, index):
    q = cache.first.first_q if index == 0 else state.project(
        a.q_proj, cache.write_base[index], cache.first.position.write_q[index])
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    qh, kh, vh = heads(q, h), heads(a.k_proj(sparse + tokens), h), heads(a.v_proj(sparse), h)
    probabilities = torch.softmax((qh @ kh.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
    output_weight = a.out_proj.weight.reshape(-1, h, dh).permute(1, 2, 0)
    values = vh @ output_weight
    b, _, n, t = probabilities.shape
    update_u = probabilities[..., :-1].permute(0, 2, 1, 3).reshape(b, n, h*(t-1))
    update_v = (values[..., :-1, :] - values[..., -1:, :]).reshape(b, h*(t-1), -1)
    constant = values[..., -1, :].sum(1) + a.out_proj.bias
    return update_u, update_v, constant


def factor_predict(model, cache, sparse, backend="explicit", trace=None):
    """All-four masks/IoU, with no timings or diagnostic computations inside."""
    batch = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), dim=0)
    tokens = torch.cat((learned[None].expand(batch, -1, -1), sparse), dim=1)
    state = FactorState(cache, batch)
    q = tokens
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        q = block.norm2(q + read_attention(block.cross_attn_token_to_image,
            q + tokens, state, cache, i, backend))
        q = block.norm3(q + block.mlp(q))
        update_u, update_v, constant = write_factors(block.cross_attn_image_to_token,
            q, tokens, state, cache, i)
        state.update(update_u, update_v, constant, block.norm4)
        if trace is not None:
            trace[f"layer{i}.sparse"] = q.detach()
            trace[f"layer{i}.dense"] = state.dense.detach()
            trace[f"layer{i}.rank"] = state.u.shape[-1]
    q = model.transformer.norm_final_attn(q + read_attention(
        model.transformer.final_attn_token_to_image, q + tokens, state, cache,
        len(model.transformer.layers), backend))
    if trace is not None:
        trace["final.sparse"] = q.detach()
    # First deconvolution is a phase-aware shared linear projection. Bias is
    # added once, then ordinary per-child LN/GELU and the second deconv execute.
    pre = state.s * cache.conv_base + state.u @ (state.v @ cache.conv_matrix) + cache.conv_bias
    child_channels = model.output_upscaling[0].out_channels
    h, w = cache.first.height, cache.first.width
    upscaled = pre.reshape(batch, h, w, 2, 2, child_channels)
    upscaled = upscaled.permute(0, 5, 1, 3, 2, 4).reshape(batch, child_channels, 2*h, 2*w)
    for module in model.output_upscaling[1:]:
        upscaled = module(upscaled)
    hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    masks = (hyper @ upscaled.flatten(2)).reshape(batch, model.num_mask_tokens, *upscaled.shape[-2:])
    iou = model.iou_prediction_head(q[:, 0])
    return masks, iou
