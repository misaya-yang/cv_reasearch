"""Fixed, untrained task-context intervention in a frozen final EVA block.

Cross-image keys deliberately have no RoPE correspondence. Native query
attention retains its original RoPE. This is an inference intervention, not
a claim that the pretrained model has native inter-image attention.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MIX = 0.5


def qkv(block, x):
    attn = block.attn
    assert attn.qkv is not None and attn.q_bias is None and attn.gate is None
    z = block.norm1(x)
    batch, length, _ = z.shape
    packed = attn.qkv(z).reshape(batch, length, 3, attn.num_heads, -1)
    q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
    return attn.q_norm(q), attn.k_norm(k), v


def role_context(logits, values, coverage):
    """All reference locations, exact soft Mask mass; no role subsampling."""
    assert coverage.ndim == 1 and values.shape[-2] == coverage.numel()
    assert bool(coverage.sum() > 0) and bool((1 - coverage).sum() > 0)
    contexts, evidence = [], []
    for mass in (coverage, 1 - coverage):
        log_mass = torch.where(mass > 0, mass.log(), -torch.inf)
        weighted = logits + log_mass
        contexts.append(weighted.softmax(dim=-1) @ values)
        evidence.append(torch.logsumexp(weighted, dim=-1) - mass.sum().log())
    return contexts[0], contexts[1], (evidence[0] - evidence[1]).mean(dim=0)


def cosine_role_density(logits, coverage):
    result = []
    for mass in (coverage, 1 - coverage):
        log_mass = torch.where(mass > 0, mass.log(), -torch.inf)
        result.append(torch.logsumexp(logits + log_mass, dim=-1) - mass.sum().log())
    return result[0] - result[1]


def replay_with_context(block, final_norm, x, context):
    """Exact EVA residual/MLP formula with supplied pre-attention-norm context."""
    attn = block.attn
    assert attn.gate is None and not block.training
    attention = attn.proj_drop(attn.proj(attn.norm(context)))
    if block.gamma_1 is not None:
        attention = block.gamma_1 * attention
    hidden = x + block.drop_path1(attention)
    update = block.mlp(block.norm2(hidden))
    if block.gamma_2 is not None:
        update = block.gamma_2 * update
    return final_norm(hidden + block.drop_path2(update))


def compatibility(native, foreground, background):
    native = F.normalize(native, dim=-1)
    foreground = F.normalize(foreground, dim=-1)
    background = F.normalize(background, dim=-1)
    return ((background - native).square() - (foreground - native).square()).mean(dim=-1)


def blended_context(native_context, memory_context, prefix=5):
    assert native_context.ndim == 3 and native_context.shape[0] == 1
    assert memory_context.shape == native_context[0, prefix:].shape
    return torch.cat((native_context[:, :prefix],
                      ((1 - MIX) * native_context[0, prefix:] + MIX * memory_context)[None]), dim=1)


def check_fixtures(device):
    """Null-task, role symmetry and mass accounting checks, independent of GT."""
    logits = torch.tensor([[[1., 0., -1.], [0., 2., 1.]],
                           [[0., 1., 0.], [1., -1., 2.]]], device=device)
    values = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]],
                           [[0., 1.], [1., 0.], [-1., 1.]]], device=device)
    mass = torch.tensor([1., .5, 0.], device=device)
    f, b, evidence = role_context(logits, values, mass)
    bf, fb, flipped = role_context(logits, values, 1 - mass)
    assert torch.equal(f, fb) and torch.equal(b, bf)
    assert torch.equal(evidence, -flipped)
    same_f, same_b, same_evidence = role_context(logits, values, torch.full_like(mass, .5))
    assert torch.equal(same_f, same_b) and torch.equal(same_evidence, torch.zeros_like(same_evidence))
    assert torch.equal(compatibility(f, same_f, same_b), torch.zeros(f.shape[:-1], device=device))
    order = torch.tensor([2, 0, 1], device=device)
    pf, pb, pe = role_context(logits[..., order], values[:, order], mass[order])
    assert torch.allclose(f, pf, atol=2e-6) and torch.allclose(b, pb, atol=2e-6)
    assert torch.allclose(evidence, pe, atol=2e-6)
    return {"role_complement_symmetry": True, "uniform_role_zero_signal": True,
            "joint_key_value_mass_permutation": True}
