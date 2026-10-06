"""Scoped reference-conditioned routing in one frozen transformer attention block.

No query labels, image edits, parameters fitted across episodes, extra model, or
extra encoder forward. Requires an actual paired [reference, query] forward.
The context is intentionally single-threaded and must not span concurrent model
calls: PyTorch SDPA is patched only while the selected attention module executes.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import threading

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InterventionConfig:
    block_from_end: int = 6
    temperature: float = 0.1
    max_bias: float = math.log(2.0)
    chunk_size: int = 512


CONFIG = asdict(InterventionConfig())
_SDPA_LOCK = threading.Lock()


def soft_roles(reference, query, coverage, temperature=0.1, chunk_size=512):
    """Role-balanced cosine kernel densities; return signed soft query roles."""
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Expected matching token-by-channel feature matrices")
    cov = coverage.reshape(-1).to(reference.device, torch.float32)
    if cov.numel() != len(reference) or not torch.isfinite(cov).all() or ((cov < 0) | (cov > 1)).any():
        raise ValueError("Reference coverage must be finite fractions in [0,1]")
    if temperature <= 0 or chunk_size < 1:
        raise ValueError("Invalid kernel temperature/chunk size")
    fg, bg = cov.sum(), (1 - cov).sum()
    if min(float(fg), float(bg)) <= 0:
        raise ValueError("Reference requires positive foreground and background mass")
    ref = F.normalize(reference.float(), dim=-1)
    qry = F.normalize(query.float(), dim=-1)
    log_f = torch.where(cov > 0, cov.log(), -torch.inf) - fg.log()
    log_b = torch.where(cov < 1, (1 - cov).log(), -torch.inf) - bg.log()
    parts = []
    for q in qry.split(chunk_size):
        sim = (q @ ref.T) / temperature
        margin = torch.logsumexp(sim + log_f, -1) - torch.logsumexp(sim + log_b, -1)
        parts.append(torch.tanh(margin / 2))
    return torch.cat(parts)


def routing_bias(signed_roles, arm, max_bias=math.log(2.0)):
    """Pair role compatibility or equally powered, key-only reference prior.

    Both have exactly the same RMS bias. The native attention score remains
    present; this is an additive bounded perturbation, not attention replacement.
    """
    s = signed_roles.float()
    if s.ndim != 1 or not torch.isfinite(s).all() or (s.abs() > 1.000001).any():
        raise ValueError("Expected finite signed soft roles in [-1,1]")
    if arm == "intervention":
        return max_bias * s[:, None] * s[None, :]
    if arm == "control":
        return (max_bias * s.square().mean().sqrt() * s[None, :]).expand(len(s), -1)
    if arm in ("identity", "masked_identity"):
        return torch.zeros((len(s), len(s)), device=s.device)
    raise ValueError(f"Unknown arm: {arm}")


@contextmanager
def attention_intervention(vit, reference_mask, *, arm="intervention", config=None):
    """Alter one actual SDPA call, restoring methods even after an exception.

    `vit` is the timm encoder itself (the `.m` of TimmDINOv3), and reference_mask
    is either a model-space mask or a zero-argument getter available at forward.
    Prefix tokens are never directly biased. Only batch index 1 is changed.
    The caller must establish and audit the host's [reference, query] batch order.
    """
    cfg = config or InterventionConfig()
    if arm not in ("intervention", "control", "identity", "masked_identity"):
        raise ValueError(f"Unknown arm: {arm}")
    if cfg.block_from_end < 1 or cfg.block_from_end > len(vit.blocks):
        raise ValueError("Target block does not exist")
    attn = vit.blocks[-cfg.block_from_end].attn
    if not getattr(attn, "fused_attn", False):
        raise RuntimeError("This adapter requires fused SDPA; do not silently replace the encoder")
    prefix = int(vit.num_prefix_tokens)
    had_forward = "forward" in attn.__dict__
    saved_forward = attn.__dict__.get("forward")
    original = attn.forward
    receipt = dict(arm=arm, config=asdict(cfg), attention_calls=0, sdpa_calls=0,
                   added_encoder_forwards=0, batch_order="reference,query",
                   sdpa_mask_path="native" if arm == "identity" else "dense_additive",
                   reference_directly_biased=False,
                   reference_kernel_can_change=arm != "identity")

    def forward(*args, **kwargs):
        x = args[0] if args else kwargs["x"]
        if x.ndim != 3 or x.shape[0] != 2:
            raise RuntimeError("Expected actual paired [reference, query] batch of size 2")
        patches = x.shape[1] - prefix
        side = math.isqrt(patches)
        if side * side != patches:
            raise RuntimeError("Only square 1024-model grids are supported")
        mask = reference_mask() if callable(reference_mask) else reference_mask
        mask = torch.as_tensor(mask, device=x.device).squeeze()
        if mask.ndim != 2:
            raise ValueError("Reference mask must reduce to H x W")
        cov = F.interpolate(mask[None, None].float(), (side, side), mode="area").flatten()
        roles = soft_roles(x[0, prefix:], x[1, prefix:], cov, cfg.temperature, cfg.chunk_size)
        bias = routing_bias(roles, arm, cfg.max_bias)
        receipt.update(role_mean=float(roles.mean()), role_rms=float(roles.square().mean().sqrt()),
                       bias_rms=float(bias.square().mean().sqrt()), bias_max_abs=float(bias.abs().max()),
                       patch_grid=[side, side], prefix_tokens=prefix)
        receipt["attention_calls"] += 1
        if not _SDPA_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent attention intervention is not supported")
        native_sdpa = F.scaled_dot_product_attention
        count = 0

        def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **sdpa_kwargs):
            nonlocal count
            count += 1
            if q.shape[0] != 2 or q.shape[-2] != x.shape[1] or k.shape[-2] != x.shape[1]:
                raise RuntimeError("SDPA layout is incompatible with paired token bias")
            if is_causal or dropout_p != 0.0:
                raise RuntimeError("Expected deterministic noncausal evaluation attention")
            if arm == "identity":
                return native_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, **sdpa_kwargs)
            extra = torch.zeros((2, 1, x.shape[1], x.shape[1]), device=q.device, dtype=q.dtype)
            extra[1, 0, prefix:, prefix:] = bias.to(q.dtype)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    extra = extra.masked_fill(~attn_mask, -torch.inf)
                else:
                    extra = extra + attn_mask
            return native_sdpa(q, k, v, attn_mask=extra, dropout_p=dropout_p,
                               is_causal=is_causal, **sdpa_kwargs)
        try:
            F.scaled_dot_product_attention = sdpa
            result = original(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = native_sdpa
            _SDPA_LOCK.release()
        if count != 1:
            raise RuntimeError(f"Expected exactly one observed SDPA call, got {count}")
        receipt["sdpa_calls"] += count
        return result

    attn.forward = forward
    try:
        yield receipt
    finally:
        if had_forward:
            attn.forward = saved_forward
        else:
            delattr(attn, "forward")


def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Fail closed: final-layer caches cannot simulate this mechanism."""
    raise RuntimeError("Intervention requires real RGB and encoder forward; use scripts/run_intervention.py")


def control(q, r, cov, score, *, device="cpu", extras=None):
    raise RuntimeError("Key-prior control requires real RGB and encoder forward; use scripts/run_intervention.py")
