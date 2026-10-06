"""Reference FG-minus-BG value messages, with the native attention path retained.

One frozen paired RGB forward; no query labels or fitted parameters. The scoped
SDPA interception reads actual values and adds a message after native attention,
before the untouched attention norm/gate/projection and block residual/MLP.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .intervention import _SDPA_LOCK


@dataclass(frozen=True)
class ReferenceValueConfig:
    block_from_end: int = 6
    temperature: float = 0.1
    residual_ratio: float = 0.1
    chunk_size: int = 256


CONFIG = asdict(ReferenceValueConfig())
ARMS = ("identity", "control", "reference_value", "foreground_only")


def reference_messages(reference, query, values, coverage, config=None):
    """Conditional FG/BG value means from pre-RoPE cosine correspondence.

    Values are real projected V, flattened over heads. The uniform control runs
    this identical retrieval and discards its conditional means, so its resource
    budget and reference information match the candidate.
    """
    cfg = config or ReferenceValueConfig()
    if (reference.ndim != 2 or query.ndim != 2 or values.ndim != 2
            or reference.shape[1] != query.shape[1] or len(values) != len(reference)
            or len(reference) == 0 or len(query) == 0):
        raise ValueError("Expected nonempty matching reference/query descriptors and reference values")
    if cfg.temperature <= 0 or cfg.chunk_size < 1 or not 0 <= cfg.residual_ratio <= 1:
        raise ValueError("Invalid frozen retrieval configuration")
    cov = torch.as_tensor(coverage, device=reference.device).float().reshape(-1)
    if (cov.numel() != len(reference) or not torch.isfinite(cov).all()
            or ((cov < 0) | (cov > 1)).any()):
        raise ValueError("Reference coverage must be finite fractions in [0,1]")
    fg_mass, bg_mass = cov.sum(), (1 - cov).sum()
    if min(float(fg_mass), float(bg_mass)) <= 0:
        raise ValueError("Reference must supply foreground and background")
    if any(not torch.isfinite(x).all() for x in (reference, query, values)):
        raise ValueError("Descriptors and values must be finite")
    ref, qry = F.normalize(reference.float(), dim=-1), F.normalize(query.float(), dim=-1)
    val = values.float()
    log_fg = torch.where(cov > 0, cov.log(), -torch.inf)
    log_bg = torch.where(cov < 1, (1 - cov).log(), -torch.inf)
    fg, bg, margin, entropy_fg, entropy_bg = [], [], [], [], []
    for chunk in qry.split(cfg.chunk_size):
        similarity = (chunk @ ref.T) / cfg.temperature
        f_logits, b_logits = similarity + log_fg, similarity + log_bg
        f_weight, b_weight = f_logits.softmax(-1), b_logits.softmax(-1)
        fg.append(f_weight @ val)
        bg.append(b_weight @ val)
        margin.append(torch.logsumexp(f_logits, -1) - fg_mass.log()
                      - torch.logsumexp(b_logits, -1) + bg_mass.log())
        entropy_fg.append(-(f_weight * f_weight.clamp_min(1e-30).log()).sum(-1))
        entropy_bg.append(-(b_weight * b_weight.clamp_min(1e-30).log()).sum(-1))
    uniform_fg = (cov @ val) / fg_mass
    uniform_bg = ((1 - cov) @ val) / bg_mass
    return dict(foreground=torch.cat(fg), background=torch.cat(bg),
                uniform=uniform_fg - uniform_bg,
                density_margin=torch.cat(margin), entropy_fg=torch.cat(entropy_fg),
                entropy_bg=torch.cat(entropy_bg))


def bounded_message(messages, native, arm, ratio=0.1):
    """Equal per-query-token L2 norm budget, measured before norm/projection.

    The control has one shared reference direction; the candidate can retrieve
    different part directions. A zero direction receives zero update and is
    reported as degenerate, rather than inventing another direction.
    """
    if arm == "reference_value":
        direction = messages["foreground"] - messages["background"]
    elif arm == "control":
        direction = messages["uniform"].expand_as(native)
    elif arm == "foreground_only":
        direction = messages["foreground"]
    else:
        raise ValueError(f"Unknown message arm: {arm}")
    if direction.shape != native.shape or not 0 <= ratio <= 1:
        raise ValueError("Message/native shape or budget mismatch")
    norm = direction.norm(dim=-1, keepdim=True)
    budget = ratio * native.float().norm(dim=-1, keepdim=True)
    delta = direction * (budget / norm.clamp_min(1e-12))
    delta = torch.where(norm > 1e-12, delta, torch.zeros_like(delta))
    return delta


@contextmanager
def reference_value(vit, reference_mask, *, arm="reference_value", config=None, capture=None):
    """Intercept one actual fused SDPA call and restore it even after failure.

    `capture`, if supplied, receives detached tensors for label-free attribution.
    It is the caller's responsibility to release those tensors after the episode.
    Neither the reference batch item nor any prefix output is directly modified.
    """
    cfg = config or ReferenceValueConfig()
    if arm not in ARMS or not 1 <= cfg.block_from_end <= len(vit.blocks):
        raise ValueError("Unknown arm or missing target block")
    attn = vit.blocks[-cfg.block_from_end].attn
    if vit.training or attn.training or not getattr(attn, "fused_attn", False):
        raise RuntimeError("Requires the real frozen evaluation encoder with fused SDPA")
    prefix = int(vit.num_prefix_tokens)
    had_forward, saved_forward = "forward" in attn.__dict__, attn.__dict__.get("forward")
    original = attn.forward
    receipt = dict(arm=arm, config=asdict(cfg), attention_calls=0, sdpa_calls=0,
                   paired_encoder_forwards=1, added_encoder_forwards=0,
                   native_logits_unchanged=True, native_sdpa_mask_unchanged=True,
                   reference_directly_modified=False, prefix_directly_modified=False,
                   injection="native SDPA output + reference value residual before attention norm/gate/projection")

    def forward(*args, **kwargs):
        x = args[0] if args else kwargs["x"]
        if x.ndim != 3 or x.shape[0] != 2:
            raise RuntimeError("Requires an actual paired [reference, query] batch of size 2")
        patches = x.shape[1] - prefix
        side = math.isqrt(patches)
        if patches < 1 or side * side != patches:
            raise RuntimeError("Requires a square reference/query patch grid")
        receipt["attention_calls"] += 1
        if not _SDPA_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent SDPA interception is not supported")
        native_sdpa = F.scaled_dot_product_attention
        count = 0

        def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            nonlocal count
            count += 1
            if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
                    or q.shape[0] != 2 or q.shape[-2] != x.shape[1]
                    or is_causal or dropout_p != 0.0):
                raise RuntimeError("Expected deterministic equal-shape paired native Q/K/V")
            native = native_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, **kwargs)
            native_query = native[1, :, prefix:, :].transpose(0, 1).reshape(patches, -1)
            if capture is not None:
                capture["native_sdpa"] = native_query.detach().float()
            receipt.update(patch_grid=[side, side], prefix_tokens=prefix,
                           heads=int(v.shape[1]), value_width=int(v.shape[-1]))
            if arm == "identity":
                return native
            mask = reference_mask() if callable(reference_mask) else reference_mask
            mask = torch.as_tensor(mask, device=x.device).squeeze()
            if mask.ndim != 2 or not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
                raise ValueError("Reference mask must be finite H x W fractions in [0,1]")
            cov = F.interpolate(mask[None, None].float(), (side, side), mode="area").flatten()
            values = v[0, :, prefix:, :].transpose(0, 1).reshape(patches, -1)
            messages = reference_messages(x[0, prefix:], x[1, prefix:], values, cov, cfg)
            delta = bounded_message(messages, native_query, arm, cfg.residual_ratio).to(native.dtype)
            out = native.clone()
            out[1, :, prefix:, :] += delta.reshape(patches, v.shape[1], v.shape[-1]).transpose(0, 1)
            actual = (out[1, :, prefix:, :] - native[1, :, prefix:, :]).transpose(0, 1).reshape(patches, -1).float()
            ratios = actual.norm(dim=-1) / native_query.float().norm(dim=-1).clamp_min(1e-12)
            receipt.update(injected_norm_ratio_mean=float(ratios.mean()),
                           injected_norm_ratio_max=float(ratios.max()),
                           zero_message_tokens=int((delta.float().norm(dim=-1) <= 1e-12).sum()),
                           uniform_direction_norm=float(messages["uniform"].norm()),
                           density_margin_mean=float(messages["density_margin"].mean()),
                           foreground_entropy_mean=float(messages["entropy_fg"].mean()),
                           background_entropy_mean=float(messages["entropy_bg"].mean()))
            if capture is not None:
                capture.update(injected_delta=actual.detach(),
                               density_margin=messages["density_margin"].detach(),
                               entropy_fg=messages["entropy_fg"].detach(),
                               entropy_bg=messages["entropy_bg"].detach(),
                               contrast_uniform_cosine=F.cosine_similarity(
                                   messages["foreground"] - messages["background"],
                                   messages["uniform"][None].expand_as(native_query), dim=-1).detach())
            return out

        try:
            F.scaled_dot_product_attention = sdpa
            result = original(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = native_sdpa
            _SDPA_LOCK.release()
        if count != 1:
            raise RuntimeError(f"Expected one native SDPA call, got {count}")
        receipt["sdpa_calls"] += count
        if capture is not None:
            capture["attention_output"] = result[1, prefix:].detach().float()
        return result

    attn.forward = forward
    try:
        yield receipt
    finally:
        if had_forward:
            attn.forward = saved_forward
        else:
            delattr(attn, "forward")
