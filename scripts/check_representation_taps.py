#!/usr/bin/env python3
"""Check taps against the actual EVA forward, before running branch diagnostics."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from timm.models.eva import Eva

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.representations import observe_branches, positional_basis, unit_tokens


def main():
    torch.manual_seed(21)
    torch.set_num_threads(2)
    model = Eva(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4,
                num_reg_tokens=4, dynamic_img_size=True, use_abs_pos_emb=False,
                use_rot_pos_emb=True, rope_type='dinov3', qkv_bias=False,
                qk_norm=True).eval()
    encoder = SimpleNamespace(m=model)
    x = torch.randn(2, 3, 32, 32)
    incoming = {}
    handle = model.blocks[0].attn.register_forward_pre_hook(
        lambda module, args: incoming.update(x=args[0].clone()))
    with torch.inference_mode():
        plain = model.forward_intermediates(x, indices=1, norm=True,
                                           output_fmt='NLC', intermediates_only=True)[0]
        with observe_branches(encoder) as taps:
            observed = model.forward_intermediates(x, indices=1, norm=True,
                                                   output_fmt='NLC', intermediates_only=True)[0]
        assert torch.equal(plain, observed)
        assert len(taps) == 8 and all(v.shape == (2, 4, 32) for v in taps.values())
        assert torch.equal(taps['O/2'], observed)
        a = model.blocks[0].attn
        qkv = a.qkv(incoming['x']).reshape(2, 9, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        for name, value in [('Q', a.q_norm(q)), ('K', a.k_norm(k)), ('V', v)]:
            expected = value.transpose(1, 2).flatten(2)[:, 5:].contiguous()
            assert torch.equal(taps[name+'/1'], expected)
        handle.remove()
        assert not any(b._forward_hooks or b.attn.q_norm._forward_hooks or
                       b.attn.k_norm._forward_hooks or b.attn.qkv._forward_hooks for b in model.blocks)
        constant = torch.ones(1, 8, 4)*.1
        basis, info = positional_basis(constant, components=2)
        assert basis.shape == (4, 0) and info['zero_rank']
        assert torch.equal(unit_tokens(constant, basis), F.normalize(constant, dim=-1))
        source = torch.randn(1, 20, 4)
        basis, info = positional_basis(source, components=2)
        assert basis.shape == (4, 2)
        expected = F.normalize(source, dim=-1)
        expected = F.normalize(expected-(expected@basis)@basis.T, dim=-1)
        assert torch.equal(unit_tokens(source, basis), expected)
    print('PASS: unchanged EVA outputs, normalized O, Q/K before RoPE, actual V, hook removal, per-branch SVD and zero-rank identity')


if __name__ == '__main__':
    main()
