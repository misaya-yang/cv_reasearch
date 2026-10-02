"""Two post-LN projection groups for SAM; inference only, complete outputs.

Import the execution_baselines directory before this module. Cache construction
is separate from inference; it never constructs unused independent factor bases.
The original associated-attention strong baseline keeps its contraction order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from baselines import (ImageCache, cached_predict, core, heads, plain_attention,
                       read_attention, shape_plan, unheads, write_attention)
from factor_candidate import FactorState


@dataclass
class ProjectionGroup:
    weight: torch.Tensor
    bias: torch.Tensor
    pe: torch.Tensor
    names: tuple
    widths: tuple
    base: torch.Tensor | None

    @classmethod
    def build(cls, entries, base=None):
        # Each entry is (name, [out,D] weight, [out] bias, [1,N,out] PE).
        weight = torch.cat([e[1] for e in entries], dim=0)
        bias = torch.cat([e[2] for e in entries])
        pe = torch.cat([e[3] for e in entries], dim=-1)
        projection = None if base is None else F.linear(base, weight, None)
        return cls(weight, bias, pe, tuple(e[0] for e in entries),
                   tuple(e[1].shape[0] for e in entries), projection)

    def dense_project(self, x):
        # Match ordinary projected baselines: linear bias, then separate PE.
        return (F.linear(x, self.weight, self.bias) + self.pe).split(self.widths, dim=-1)

    def factor_project(self, state):
        # One factor expansion U @ (V W^T) for every member of this group.
        merged = state.s * self.base + state.u @ F.linear(state.v, self.weight, None)
        # Match factor candidate's order: separate PE, then projection bias.
        return (merged + self.pe + self.bias).split(self.widths, dim=-1)


@dataclass
class MergeCache:
    first: ImageCache
    groups: tuple
    method: str
    plans: list | None

    @classmethod
    def build(cls, model, image, dense, position, method="factor", plans=None):
        if len(model.transformer.layers) != 2:
            raise ValueError("This two-group schedule requires the official depth-2 SAM decoder")
        if method not in ("factor", "cached", "dense_assoc"):
            raise ValueError(method)
        if method == "dense_assoc" and plans is None:
            raise ValueError("Strong-baseline cache needs the actual N/T execution plans")
        first = ImageCache.build(model, image, dense, position)
        layers = model.transformer.layers
        factor = method == "factor"
        if method == "cached" and plans is None:
            # All ordinary cached post-LN projections use the projected route.
            plans = shape_plan(model, first.base.shape[1], 5, "cached")
        bases = [first.base]
        if factor:
            for block in layers:
                old = bases[-1]
                bases.append((old - old.mean(-1, keepdim=True)) * block.norm4.weight)
        groups = []
        for index in (1, 2):
            entries = []
            read = (layers[index].cross_attn_token_to_image if index == 1
                    else model.transformer.final_attn_token_to_image)
            read_plan = None if factor else plans[2*index if index == 1 else -1]
            if not factor and read_plan["qk"] != read_plan["value"]:
                raise ValueError("Mixed read plans need a separate schedule; do not weaken their route")
            if factor or read_plan["qk"] == "projected":
                entries.extend([
                    ("k", read.k_proj.weight, read.k_proj.bias, position.read_k[index]),
                    ("v", read.v_proj.weight, read.v_proj.bias,
                     first.base.new_zeros(1, first.base.shape[1], read.v_proj.out_features))])
            if index == 1:
                write = layers[1].cross_attn_image_to_token.q_proj
                if factor or plans[3]["qk"] == "projected":
                    entries.append(("q", write.weight, write.bias, position.write_q[1]))
            elif entries or factor:
                # A lone upscaler is left native in the associated strong route.
                conv = model.output_upscaling[0]
                if not (conv.kernel_size == (2, 2) and conv.stride == (2, 2)
                        and conv.padding == (0, 0) and conv.output_padding == (0, 0)
                        and conv.dilation == (1, 1) and conv.groups == 1):
                    raise ValueError("Merge requires the official nonoverlapping first upscaler")
                matrix = conv.weight.permute(0, 2, 3, 1).reshape(conv.in_channels, -1)
                entries.append(("conv", matrix.T, conv.bias.repeat(4),
                                first.base.new_zeros(1, first.base.shape[1], matrix.shape[1])))
            groups.append(ProjectionGroup.build(entries, bases[index] if factor else None)
                          if entries else None)
        return cls(first, tuple(groups), method, plans)


def projected_read(a, q_input, k, v, backend):
    qh = heads(a.q_proj(q_input), a.num_heads)
    return a.out_proj(unheads(core(qh, heads(k, a.num_heads), heads(v, a.num_heads), backend)))


def factor_write(a, sparse, tokens, q):
    # Original write_factors contraction, accepting the already-grouped Q.
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    qh = heads(q, h)
    kh = heads(a.k_proj(sparse + tokens), h)
    vh = heads(a.v_proj(sparse), h)
    probabilities = torch.softmax((qh @ kh.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
    output_weight = a.out_proj.weight.reshape(-1, h, dh).permute(1, 2, 0)
    values = vh @ output_weight
    b, _, n, t = probabilities.shape
    update_u = probabilities[..., :-1].permute(0, 2, 1, 3).reshape(b, n, h*(t-1))
    update_v = (values[..., :-1, :] - values[..., -1:, :]).reshape(b, h*(t-1), -1)
    constant = values[..., -1, :].sum(1) + a.out_proj.bias
    return update_u, update_v, constant


def phase_upscale(model, pre, cache, batch):
    child_channels = model.output_upscaling[0].out_channels
    h, w = cache.first.height, cache.first.width
    upscaled = pre.reshape(batch, h, w, 2, 2, child_channels)
    upscaled = upscaled.permute(0, 5, 1, 3, 2, 4).reshape(batch, child_channels, 2*h, 2*w)
    for module in model.output_upscaling[1:]:
        upscaled = module(upscaled)
    return upscaled


def mask_iou(model, q, upscaled):
    hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    masks = (hyper @ upscaled.flatten(2)).reshape(q.shape[0], model.num_mask_tokens, *upscaled.shape[-2:])
    return masks, model.iou_prediction_head(q[:, 0])


def factor_merge_predict(model, cache, sparse, backend="explicit", trace=None):
    """Two factor expansions, same dense LayerNorm companion and complete head."""
    if cache.method != "factor":
        raise ValueError("factor_merge_predict requires a factor MergeCache")
    batch = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), dim=0)
    tokens = torch.cat((learned[None].expand(batch, -1, -1), sparse), dim=1)
    state = FactorState(cache, batch)
    q = tokens
    grouped = None
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        k, v = ((cache.first.first_k, cache.first.first_v) if i == 0 else grouped[:2])
        q = block.norm2(q + projected_read(block.cross_attn_token_to_image, q + tokens, k, v, backend))
        q = block.norm3(q + block.mlp(q))
        write_q = cache.first.first_q if i == 0 else grouped[2]
        state.update(*factor_write(block.cross_attn_image_to_token, q, tokens, write_q), block.norm4)
        if trace is not None:
            trace[f"layer{i}.sparse"] = q.detach()
            trace[f"layer{i}.dense"] = state.dense.detach()
            trace[f"layer{i}.rank"] = state.u.shape[-1]
        grouped = cache.groups[i].factor_project(state)
    k, v, pre = grouped
    q = model.transformer.norm_final_attn(q + projected_read(
        model.transformer.final_attn_token_to_image, q + tokens, k, v, backend))
    if trace is not None:
        trace["final.sparse"] = q.detach()
    return mask_iou(model, q, phase_upscale(model, pre, cache, batch))


def cached_merge_predict(model, cache, sparse, backend="explicit", trace=None):
    """Same grouped projections for eligible dense routes; associated routes stay."""
    if cache.method not in ("cached", "dense_assoc"):
        raise ValueError("cached_merge_predict requires a dense MergeCache")
    if all(group is None for group in cache.groups):
        return cached_predict(model, cache.first, sparse, cache.method, backend,
                              trace=trace, plans=cache.plans)
    batch = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), dim=0)
    tokens = torch.cat((learned[None].expand(batch, -1, -1), sparse), dim=1)
    q, x = tokens, cache.first.base.expand(batch, -1, -1)
    projected_values = {}
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        read = (projected_read(block.cross_attn_token_to_image, q + tokens,
                              projected_values["k"], projected_values["v"], backend)
                if "k" in projected_values else read_attention(block.cross_attn_token_to_image,
                    q + tokens, x, cache.first, i, cache.plans[2*i], backend))
        q = block.norm2(q + read)
        q = block.norm3(q + block.mlp(q))
        if "q" in projected_values:
            a = block.cross_attn_image_to_token
            # A projected strong write may still choose sparse output. Reuse
            # that exact route and cached Q, without projecting it a second time.
            if cache.plans[2*i+1]["output"] == "sparse":
                u, v, constant = factor_write(a, q, tokens, projected_values["q"])
                update = u @ v + constant[:, None, :]
            else:
                update = a.out_proj(unheads(core(
                    heads(projected_values["q"], a.num_heads), heads(a.k_proj(q + tokens), a.num_heads),
                    heads(a.v_proj(q), a.num_heads), backend)))
        else:
            update = write_attention(block.cross_attn_image_to_token, q + tokens, q,
                                     x, cache.first, i, cache.plans[2*i+1], backend)
        x = block.norm4(x + update)
        if trace is not None:
            trace[f"layer{i}.sparse"] = q.detach()
            trace[f"layer{i}.dense"] = x.detach()
        group = cache.groups[i]
        projected_values = {} if group is None else dict(zip(group.names, group.dense_project(x)))
    a = model.transformer.final_attn_token_to_image
    read = (projected_read(a, q + tokens, projected_values["k"], projected_values["v"], backend)
            if "k" in projected_values else read_attention(a, q + tokens, x, cache.first, 2, cache.plans[-1], backend))
    q = model.transformer.norm_final_attn(q + read)
    if trace is not None:
        trace["final.sparse"] = q.detach()
    upscaled = (phase_upscale(model, projected_values["conv"], cache, batch)
                if "conv" in projected_values else model.output_upscaling(
                    x.transpose(1, 2).reshape(batch, -1, cache.first.height, cache.first.width)))
    return mask_iou(model, q, upscaled)
