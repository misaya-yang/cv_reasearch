"""Exact-real-arithmetic SAM execution baselines; no persistent image factors.

Only pinned, inspected official modules are imported. This file does not copy,
modify, download, or monkey-patch official sources. All masks/tokens are retained.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIN = "dca509fe793f601edb92606367a655c15ac00fdf"


def official_modules():
    """Verify the already-inspected source copies before any official import."""
    records = json.loads((ROOT / "source_audit/source_provenance.json").read_text())["files"]
    folder = ROOT / "official_cpu_validation/official_sam"
    provenance = []
    for stem in ("common", "transformer", "mask_decoder"):
        record = next(r for r in records if r["project"] == "sam" and
                      r["source_path"] == f"segment_anything/modeling/{stem}.py")
        path = folder / f"{stem}.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"] or record["revision_commit"] != PIN:
            raise RuntimeError(f"Official source provenance mismatch: {path}")
        provenance.append({"file": str(path.relative_to(ROOT)), "sha256": digest,
                           "source_url": record["immutable_url"],
                           "revision": PIN})
    init = (folder / "__init__.py").read_text()
    if init != "# Local package scaffolding only.\n":
        raise RuntimeError("Unexpected package initialization code")
    sys.path.insert(0, str(folder.parent))
    # Disabling bytecode avoids any writes outside this task's own folder.
    sys.dont_write_bytecode = True
    return (importlib.import_module("official_sam.transformer"),
            importlib.import_module("official_sam.mask_decoder"),
            importlib.import_module("official_sam.common"), provenance)


def make_fixture(seed=0, grid=8, tokens=7, batch=2, device="cpu", dtype=torch.float32):
    transformer, decoder, common, provenance = official_modules()
    if tokens < 5 or grid < 1 or batch < 1:
        raise ValueError("Need T >= 5, positive grid and batch")
    torch.manual_seed(seed)
    model = decoder.MaskDecoder(transformer_dim=256,
        transformer=transformer.TwoWayTransformer(depth=2, embedding_dim=256,
            num_heads=8, mlp_dim=2048)).eval()
    # Nontrivial LN affine parameters exercise trained-shape semantics.
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, (nn.LayerNorm, common.LayerNorm2d)):
                mod.weight.copy_(1 + .15 * torch.randn_like(mod.weight))
                mod.bias.copy_(.15 * torch.randn_like(mod.bias))
    image = torch.randn(1, 256, grid, grid)
    pe = torch.randn_like(image)
    dense = torch.randn(1, 256, 1, 1).expand(1, 256, grid, grid)
    sparse = torch.randn(batch, tokens - 5, 256)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.contiguous().numpy().tobytes())
    model = model.to(device=device, dtype=dtype)
    image, pe, dense, sparse = [x.to(device=device, dtype=dtype) for x in (image, pe, dense, sparse)]
    return model, image, pe, dense, sparse, {"model_fp32_state_sha256": digest.hexdigest(),
        "provenance": provenance, "seed": seed, "T": tokens, "N": grid * grid}


def heads(x, h):
    return x.reshape(x.shape[0], x.shape[1], h, -1).transpose(1, 2)


def unheads(x):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def core(q, k, v, backend="explicit", scale=None):
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)
    # Division, rather than multiplication by the reciprocal, matches official
    # SAM's reduction order for the ordinary-width explicit baseline.
    if backend != "explicit":
        raise ValueError(backend)
    a = torch.softmax((q @ k.transpose(-1, -2)) / (1 / scale), dim=-1)
    return a @ v


def plain_attention(attn, q, k, v, backend):
    if backend == "explicit":
        return attn(q=q, k=k, v=v)
    return attn.out_proj(unheads(core(heads(attn.q_proj(q), attn.num_heads),
        heads(attn.k_proj(k), attn.num_heads), heads(attn.v_proj(v), attn.num_heads), backend)))


@dataclass
class PositionCache:
    """Fixed-grid/model/dtype/device cache, separable from per-image cache."""
    pe: torch.Tensor
    read_k: tuple
    write_q: tuple

    @classmethod
    def build(cls, model, pe, plans=None):
        if pe.shape[0] != 1:
            raise ValueError("One common positional grid required")
        p = pe.flatten(2).transpose(1, 2)
        layers = model.transformer.layers
        reads = [b.cross_attn_token_to_image for b in layers]
        reads += [model.transformer.final_attn_token_to_image]
        # First attention projections already cache E+PE jointly. Associated
        # later routes use joint X+PE too, so avoid unused PE projections when
        # a shape plan is supplied. This cache must match that execution plan.
        read_plans = None if plans is None else [plans[2*i] for i in range(len(layers))] + [plans[-1]]
        read_pe = tuple(None if i == 0 or (read_plans is not None and read_plans[i]["qk"] == "associated")
                        else F.linear(p, a.k_proj.weight, None) for i, a in enumerate(reads))
        write_pe = tuple(None if i == 0 or (plans is not None and plans[2*i+1]["qk"] == "associated")
                         else F.linear(p, b.cross_attn_image_to_token.q_proj.weight, None) for i, b in enumerate(layers))
        return cls(p, read_pe, write_pe)


@dataclass
class ImageCache:
    """First unchanging image K/V and write Q; no later content projections."""
    base: torch.Tensor
    height: int
    width: int
    first_k: torch.Tensor
    first_v: torch.Tensor
    first_q: torch.Tensor
    position: PositionCache

    @classmethod
    def build(cls, model, image, dense, position):
        if image.shape[0] != 1 or dense.shape[0] != 1:
            raise ValueError("Caches require one shared image AND identical dense prompt")
        if image.shape != dense.shape or position.pe.shape != (1, image.shape[-2] * image.shape[-1], image.shape[1]):
            raise ValueError("Image, dense prompt and PE shapes must match")
        x = (image + dense).flatten(2).transpose(1, 2)
        first = model.transformer.layers[0]
        # Joint E+PE projection matches ordinary first-layer arithmetic. PE is
        # absent from first_v. Later position-only terms are cached separately.
        xp = x + position.pe
        return cls(x, image.shape[-2], image.shape[-1],
            first.cross_attn_token_to_image.k_proj(xp),
            first.cross_attn_token_to_image.v_proj(x),
            first.cross_attn_image_to_token.q_proj(xp), position)


def shape_plan(model, n, t, method="dense_assoc", order="auto", write_output="auto"):
    """Choose contraction orders, not timings. Costs count multiply-add as 2.

    Model/PE-only terms are not per-prompt selection costs. Common terms cancel.
    The list is deliberately explicit so benchmark reports record exact routes.
    """
    result = []
    for i, block in enumerate(model.transformer.layers):
        result.append(attention_plan(block.cross_attn_token_to_image, n, t,
            read=True, cached=i == 0, method=method, order=order, write_output=write_output))
        result.append(attention_plan(block.cross_attn_image_to_token, n, t,
            read=False, cached=i == 0, method=method, order=order, write_output=write_output))
    result.append(attention_plan(model.transformer.final_attn_token_to_image, n, t,
        read=True, cached=False, method=method, order=order, write_output=write_output))
    return result


def attention_plan(a, n, t, read, cached, method, order, write_output):
    d, m, h = a.embedding_dim, a.internal_dim, a.num_heads
    # Associated q/k use joint X+PE, so no separate per-prompt PE dot.
    dense_kq = 2*n*d*m + 2*n*t*m
    assoc_kq = 2*t*d*m + 2*h*n*t*d
    dense_v = 2*n*d*m + 2*n*t*m
    assoc_v = 2*h*n*t*d + 2*t*d*m
    direct_out = 2*n*t*m + 2*n*m*d
    narrow_out = 2*t*m*d + 2*n*h*(t-1)*d
    def pick(dense, assoc):
        if method != "dense_assoc": return "projected"
        if order != "auto": return order
        return "associated" if assoc < dense else "projected"
    result = {"role": "read" if read else "write", "first_image_cache": cached,
        "qk": "cached" if cached else pick(dense_kq, assoc_kq),
        "value": "cached" if read and cached else (pick(dense_v, assoc_v) if read else "sparse"),
        "comparison_flops": {"qk_projected": dense_kq, "qk_associated": assoc_kq}}
    if read:
        result["comparison_flops"].update(value_projected=dense_v, value_associated=assoc_v)
    else:
        out = "dense"
        if method == "dense_assoc":
            out = ("sparse" if narrow_out < direct_out else "dense") if write_output == "auto" else write_output
        result["output"] = out
        result["comparison_flops"].update(write_dense=direct_out, write_sparse=narrow_out)
    return result


def projected(x, linear, pe_projection=None):
    out = F.linear(x, linear.weight, linear.bias)
    return out if pe_projection is None else out + pe_projection


def read_attention(a, q_input, x, cache, index, plan, backend):
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    q = heads(a.q_proj(q_input), h)
    if plan["qk"] == "cached":
        k = heads(cache.first_k, h)
        v = heads(cache.first_v, h)
        return a.out_proj(unheads(core(q, k, v, backend)))
    if plan["qk"] == "projected" and plan["value"] == "projected":
        k = heads(projected(x, a.k_proj, cache.position.read_k[index]), h)
        v = heads(a.v_proj(x), h)
        return a.out_proj(unheads(core(q, k, v, backend)))
    wk = a.k_proj.weight.reshape(h, dh, -1)
    wv = a.v_proj.weight.reshape(h, dh, -1)
    if backend == "sdpa" and plan["qk"] == "associated" and plan["value"] == "associated":
        # Shared N x D key/value stream, head-broadcast views, H x T x D read
        # result. Keep original 1/sqrt(dh), despite transformed query width D.
        # k_proj bias produces one constant per query/head over all N logits,
        # so softmax cancels it exactly in real arithmetic. Avoid an unnecessary
        # mask which could disable efficient backends. The bias is accounted
        # for by that identity, not forgotten; finite-precision gates still apply.
        qwide = q @ wk
        aggregated = F.scaled_dot_product_attention(qwide,
            (x + cache.position.pe).unsqueeze(1), x.unsqueeze(1),
            dropout_p=0.0, scale=1/math.sqrt(dh))
        read = aggregated @ wv.transpose(-1, -2) + a.v_proj.bias.reshape(1, h, 1, dh)
    else:
        if plan["qk"] == "associated":
            qwide = q @ wk
            batch, _, t, d = qwide.shape
            # Flatten sparse heads rather than broadcasting the dense image
            # across H; torch.matmul's broadcast lowering can otherwise clone
            # a [B,H,N,D] dense input before bmm.
            logits = torch.bmm(qwide.reshape(batch, h*t, d),
                               (x + cache.position.pe).transpose(-1, -2)).reshape(batch, h, t, x.shape[1])
            logits = logits + (q * a.k_proj.bias.reshape(1, h, 1, dh)).sum(-1, keepdim=True)
        else:
            k = heads(projected(x, a.k_proj, cache.position.read_k[index]), h)
            logits = q @ k.transpose(-1, -2)
        prob = torch.softmax(logits / math.sqrt(dh), dim=-1)
        if plan["value"] == "associated":
            batch, _, t, n = prob.shape
            aggregated = torch.bmm(prob.reshape(batch, h*t, n), x).reshape(batch, h, t, x.shape[-1])
            read = aggregated @ wv.transpose(-1, -2)
            read = read + a.v_proj.bias.reshape(1, h, 1, dh)
        else:
            read = prob @ heads(a.v_proj(x), h)
    return a.out_proj(unheads(read))


def write_attention(a, q_input, sparse, x, cache, index, plan, backend):
    h, dh = a.num_heads, a.internal_dim // a.num_heads
    k, v = heads(a.k_proj(q_input), h), heads(a.v_proj(sparse), h)
    if plan["qk"] == "cached":
        q = heads(cache.first_q, h)
        if plan["output"] == "dense":
            return a.out_proj(unheads(core(q, k, v, backend)))
        logits = q @ k.transpose(-1, -2)
    elif plan["qk"] == "projected":
        q = heads(projected(x, a.q_proj, cache.position.write_q[index]), h)
        if plan["output"] == "dense":
            return a.out_proj(unheads(core(q, k, v, backend)))
        logits = q @ k.transpose(-1, -2)
    else:
        wq = a.q_proj.weight.reshape(h, dh, -1)
        kwide = k @ wq
        batch, _, t, d = kwide.shape
        logits = torch.bmm(x + cache.position.pe, kwide.reshape(batch, h*t, d).transpose(-1, -2))
        logits = logits.reshape(batch, x.shape[1], h, t).permute(0, 2, 1, 3)
        logits = logits + (k * a.q_proj.bias.reshape(1, h, 1, dh)).sum(-1).unsqueeze(-2)
    if plan["output"] == "dense":
        prob = torch.softmax(logits / math.sqrt(dh), dim=-1)
        return a.out_proj(unheads(prob @ v))
    prob = torch.softmax(logits / math.sqrt(dh), dim=-1)
    # Move output projection onto T sparse values, then remove each reference
    # attention column using row stochasticity. Values already contain v bias.
    # No [B,H,N,D] output is created: one [B,N,H*(T-1)] @ [B,H*(T-1),D].
    out_w = a.out_proj.weight.reshape(-1, h, dh).permute(1, 2, 0)
    c = v @ out_w
    b, _, n, t = prob.shape
    u = prob[..., :-1].permute(0, 2, 1, 3).reshape(b, n, h*(t-1))
    difference = (c[..., :-1, :] - c[..., -1:, :]).reshape(b, h*(t-1), -1)
    constant = c[..., -1, :].sum(1) + a.out_proj.bias
    return u @ difference + constant[:, None, :]


def official_predict(model, image, pe, dense, sparse):
    """Baseline 1: original complete MaskDecoder.predict_masks, unmodified."""
    return model.predict_masks(image_embeddings=image, image_pe=pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense.expand(sparse.shape[0], -1, -1, -1))


def cached_predict(model, cache, sparse, method="cached", backend="explicit",
                   order="auto", write_output="auto", trace=None, plans=None):
    """Baseline 2/3, returning every mask and IoU. trace is correctness-only."""
    if method not in ("cached", "dense_assoc"):
        raise ValueError(method)
    b = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), dim=0)
    tokens = torch.cat((learned[None].expand(b, -1, -1), sparse), dim=1)
    q, x = tokens, cache.base.expand(b, -1, -1)
    if plans is None:
        plans = shape_plan(model, x.shape[1], tokens.shape[1], method, order, write_output)
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        q = block.norm2(q + read_attention(block.cross_attn_token_to_image,
            q + tokens, x, cache, i, plans[2*i], backend))
        q = block.norm3(q + block.mlp(q))
        x = block.norm4(x + write_attention(block.cross_attn_image_to_token,
            q + tokens, q, x, cache, i, plans[2*i+1], backend))
        if trace is not None:
            trace[f"layer{i}.sparse"] = q.detach()
            trace[f"layer{i}.dense"] = x.detach()
    q = model.transformer.norm_final_attn(q + read_attention(
        model.transformer.final_attn_token_to_image, q + tokens, x, cache,
        len(model.transformer.layers), plans[-1], backend))
    if trace is not None:
        trace["final.sparse"] = q.detach()
    upscaled = model.output_upscaling(x.transpose(1, 2).reshape(b, -1, cache.height, cache.width))
    hyper = torch.stack([mlp(q[:, 1+i]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    masks = (hyper @ upscaled.flatten(2)).reshape(b, model.num_mask_tokens, *upscaled.shape[-2:])
    iou = model.iou_prediction_head(q[:, 0])
    return masks, iou


def tensor_bytes(value):
    """Logical persistent tensors (excluding shared model weights)."""
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(x) for x in value)
    if hasattr(value, "__dict__"): return sum(tensor_bytes(x) for x in vars(value).values())
    return 0
