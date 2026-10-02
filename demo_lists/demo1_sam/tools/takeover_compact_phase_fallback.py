"""Original local head in phase layout; same learned coarse head and selector."""
import math
import torch
from takeover_head_fallback import grid


def build_patch_cache(model):
    from types import SimpleNamespace
    first, second = model.output_upscaling[0], model.output_upscaling[3]
    for conv in (first, second):
        assert conv.kernel_size == conv.stride == (2, 2)
        assert conv.padding == conv.output_padding == (0, 0)
        assert conv.dilation == (1, 1) and conv.groups == 1
    return SimpleNamespace(conv_matrix=first.weight.permute(0, 2, 3, 1).reshape(first.in_channels, -1),
                           conv_bias=first.bias.repeat(4),
                           second_matrix=second.weight.permute(0, 2, 3, 1).reshape(second.in_channels, -1),
                           second_bias=second.bias.repeat(4))


def phase_patches(model, cache, states):
    # Identical nonoverlapping deconvolution weights; only tensor layout changes.
    pre = (states @ cache.conv_matrix + cache.conv_bias).reshape(-1, 4, 64)
    norm = model.output_upscaling[1]
    mean = pre.mean(-1, keepdim=True)
    variance = (pre-mean).square().mean(-1, keepdim=True)
    child = (pre-mean)/torch.sqrt(variance+norm.eps)
    child = model.output_upscaling[2](child*norm.weight+norm.bias)
    second = model.output_upscaling[4](child @ cache.second_matrix + cache.second_bias)
    # Phase axes (first row,col,second row,col) -> final 4x4 raster pixels.
    return second.reshape(-1, 2, 2, 2, 2, 32).permute(0, 5, 1, 3, 2, 4).reshape(-1, 32, 16)


def head(model, cache, student, x, hyper, center, basis, budget):
    p = len(x)
    coeff = student(x)
    weights = torch.einsum('pmc,ucr->pmur', hyper, basis.reshape(16, 32, 64)).reshape(p, 64, 64)
    bias = torch.einsum('pmc,uc->pmu', hyper, center.reshape(16, 32)).reshape(p, 64)
    ph = (torch.bmm(coeff, weights.transpose(1, 2))+bias[:, None]).reshape(p, 4096, 4, 16)
    if budget:
        margin = (ph.abs()/hyper.norm(dim=-1)[:, None, :, None].clamp_min(1e-6)).amin((-2, -1))
        indices = margin.topk(math.ceil(4096*budget), dim=-1, largest=False).indices
        batch = torch.arange(p, device=x.device)[:, None]
        patches = phase_patches(model, cache, x[batch, indices].reshape(-1, 256))
        h = hyper[:, None].expand(-1, indices.shape[1], -1, -1).reshape(-1, 4, 32)
        ph[batch, indices] = torch.bmm(h, patches).reshape(p, indices.shape[1], 4, 16)
    return grid(ph)
