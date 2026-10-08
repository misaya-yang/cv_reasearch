"""Read-only DINOv3 O/Q/K/V taps and each branch's own black-image basis.

O is the same norm=True block output used by get_intermediate_layers. Q/K
are captured from the actual q_norm/k_norm outputs, before RoPE. V is the
projection branch, before attention mixes positions. Prefix tokens are removed.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def observe_branches(encoder, layers=None, branches='OQKV', callback=None):
    model = encoder.m
    layers = tuple(range(1, len(model.blocks)+1)) if layers is None else tuple(layers)
    if len(set(layers)) != len(layers) or any(l < 1 or l > len(model.blocks) for l in layers):
        raise ValueError('Layers are unique and one-based')
    if not branches or set(branches)-set('OQKV'):
        raise ValueError('Unknown representation branch')
    captured, handles = {}, []

    def save(key, value, heads=False):
        if heads:
            value = value.transpose(1, 2).flatten(2)
        value = value[:, model.num_prefix_tokens:].detach().float().cpu().contiguous()
        if callback is None:
            captured[key] = value
        else:
            callback(key, value)

    try:
        for layer in layers:
            block = model.blocks[layer-1]
            attn = block.attn
            if 'O' in branches:
                def output_hook(module, inputs, output, layer=layer):
                    save(f'O/{layer}', model.norm(output))
                handles.append(block.register_forward_hook(output_hook))
            for kind, module in [('Q', attn.q_norm), ('K', attn.k_norm)]:
                if kind in branches:
                    def head_hook(module, inputs, output, key=f'{kind}/{layer}'):
                        save(key, output, heads=True)
                    handles.append(module.register_forward_hook(head_hook))
            if 'V' in branches:
                if attn.qkv is not None:
                    # The installed vit_large_patch16_dinov3 uses qkv_bias=False.
                    # F.linear with a separate bias bypasses the qkv module;
                    # reject that different architecture rather than miss V.
                    if attn.q_bias is not None:
                        raise ValueError('This observer requires the current bias-free fused DINOv3 QKV')
                    def value_hook(module, inputs, output, layer=layer, heads=attn.num_heads):
                        b, n, triple = output.shape
                        v = output.reshape(b, n, 3, heads, triple//(3*heads))[:, :, 2].flatten(2)
                        save(f'V/{layer}', v)
                    handles.append(attn.qkv.register_forward_hook(value_hook))
                else:
                    def value_hook(module, inputs, output, layer=layer):
                        save(f'V/{layer}', output)
                    handles.append(attn.v_proj.register_forward_hook(value_hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def positional_basis(black_tokens, components=500):
    """Original FoRIS unit-token/centered-SVD convention, for this branch only.

    Exactly constant black outputs have no positional subspace. Detect them
    before a float32 mean can manufacture a constant rounding residual. Other
    branches keep the source's top-500 convention without fitting a rank.
    """
    x = torch.as_tensor(black_tokens).float().cpu().reshape(-1, black_tokens.shape[-1])
    if x.ndim != 2 or not torch.isfinite(x).all() or components < 1:
        raise ValueError('Invalid black-image tokens')
    x = F.normalize(x, dim=1)
    if torch.equal(x, x[:1].expand_as(x)):
        return torch.empty(x.shape[1], 0), dict(rank_used=0, zero_rank=True)
    e = x.T.contiguous()
    e = e-e.mean(dim=1, keepdim=True)
    u, singular, _ = torch.linalg.svd(e, full_matrices=False)
    k = min(components, len(singular))
    return u[:, :k].contiguous(), dict(rank_used=k, zero_rank=False,
                                      largest_singular=float(singular[0]))


@torch.inference_mode()
def unit_tokens(tokens, basis=None):
    x = F.normalize(torch.as_tensor(tokens).float(), dim=-1)
    if basis is not None and basis.shape[1]:
        u = basis.to(x.device, x.dtype)
        if u.shape[0] != x.shape[-1]:
            raise ValueError('A branch must use its own matching positional basis')
        x = F.normalize(x-(x@u)@u.T, dim=-1)
    return x


@torch.inference_mode()
def source_reference_roles(features, foreground):
    """Exact single-reference FoRIS prototype membership, reused for all taps.

    Source mean weights are one per selected token, not area weights. RCG has
    no hard-BG definition; this explicitly uses FoRIS's existing 20% rule.
    """
    r = torch.as_tensor(features).float().flatten(1).T.contiguous()
    fg = torch.as_tensor(foreground).bool().flatten()
    if r.shape[0] != len(fg) or not fg.any():
        raise ValueError('Source reference has no valid FG tokens')
    fi, bi = torch.where(fg)[0], torch.where(~fg)[0]
    mu = F.normalize(r[fi].mean(0), dim=0)
    if len(bi):
        hard = torch.topk(r[bi]@mu, max(1, int(.2*len(bi)))).indices
        bi = bi[hard]
    return fi, bi


@torch.inference_mode()
def reference_margin(pair_tokens, foreground, background):
    """Unit reference mean difference. Takes R/Q features and reference roles only."""
    if pair_tokens.ndim != 3 or pair_tokens.shape[0] != 2:
        raise ValueError('Require exactly one reference and one query')
    r, q = pair_tokens
    fi, bi = foreground.to(r.device), background.to(r.device)
    if not len(fi) or not len(bi):
        return torch.zeros(q.shape[0], device=q.device), False
    f, b = r[fi].mean(0), r[bi].mean(0)
    if not f.norm() or not b.norm():
        return torch.zeros(q.shape[0], device=q.device), False
    f, b = F.normalize(f, dim=0), F.normalize(b, dim=0)
    return q@(f-b), bool((f-b).norm() > 0)


@torch.inference_mode()
def branch_margins(bank, bases, foreground, background, device='mps'):
    """Reduce the 96 tapped pairs to compact margins; release full feature banks.

    Raw and own-basis variants are prescribed representation choices. The
    four fixed Q/K parts of the C2 direct control have equal unit-token weight.
    """
    fields, validity, concat = {}, {}, {False: [], True: []}
    for key in sorted(list(bank), key=lambda k: (int(k.split('/')[1]), k[0])):
        raw = bank.pop(key).to(device)
        for debias in (False, True):
            x = unit_tokens(raw, bases[key] if debias else None)
            name = key+('/deb' if debias else '/raw')
            d, valid = reference_margin(x, foreground, background)
            fields[name] = d.cpu().numpy()
            validity[name] = valid
            if key in ('Q/16', 'K/16', 'Q/24', 'K/24'):
                concat[debias].append(x)
        del raw, x
    for debias, pieces in concat.items():
        x = F.normalize(torch.cat(pieces, dim=-1), dim=-1)
        d, valid = reference_margin(x, foreground, background)
        name = 'C2concat/'+('deb' if debias else 'raw')
        fields[name], validity[name] = d.cpu().numpy(), valid
    return fields, validity
