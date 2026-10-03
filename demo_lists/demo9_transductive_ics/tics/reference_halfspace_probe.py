"""Reference-label-defined value probes; mathematical interface only.

No encoder hooks, fitting, posterior, segmentation head or resource guard.
The outputs are computational interventions, never labels for physical objects.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ReferenceAxis:
    center: torch.Tensor  # [heads, head_dimension]
    direction: torch.Tensor
    contrast_norm: torch.Tensor  # [heads]
    enabled: torch.Tensor  # [heads], mathematical nondegeneracy only
    foreground_count: int
    background_count: int


def reference_axis(values, foreground, background):
    """Values [heads, patches, dimension], disjoint legal support masks.

    Callers select pure patches from SUPPORT annotation; query labels must
    never enter here. Tiny nonzero contrasts remain a stated failure risk.
    No unvalidated stability threshold is introduced by this math helper.
    """
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError('Expected floating [heads, patches, dimension] values')
    heads, patches, dimension = values.shape
    if heads == 0 or dimension == 0 or patches == 0:
        raise ValueError('Nonempty attention dimensions required')
    for mask in (foreground, background):
        if mask.dtype != torch.bool or mask.shape != (patches,) or mask.device != values.device:
            raise ValueError('Boolean support masks must match patches and device')
    if (foreground & background).any() or not torch.isfinite(values).all():
        raise ValueError('Disjoint labels and finite values required')
    nf, nb = int(foreground.sum()), int(background.sum())
    if not nf or not nb:
        zero = values.new_zeros(heads, dimension)
        return ReferenceAxis(zero, zero.clone(), values.new_zeros(heads),
                             torch.zeros(heads, dtype=torch.bool, device=values.device), nf, nb)
    fg = values[:, foreground].mean(dim=1)
    bg = values[:, background].mean(dim=1)
    difference = fg-bg
    norm = torch.linalg.vector_norm(difference, dim=-1)
    enabled = norm > 0
    denominator = torch.where(enabled, norm, torch.ones_like(norm))
    direction = difference/denominator[:, None]
    return ReferenceAxis((fg+bg)/2, direction, norm, enabled, nf, nb)


def project_query_values(values, axis, branch, *, query_index, prefix_tokens):
    """Return [batch, heads, tokens, dimension] with only query patches changed.

    plus projects to alpha<=0; minus projects to alpha>=0. There is no
    new scale parameter. Support samples/prefix values are copied unchanged.
    Q/K are absent from this interface and cannot be changed by it.
    """
    if values.ndim != 4 or not values.is_floating_point():
        raise ValueError('Expected floating [batch, heads, tokens, dimension]')
    batch, heads, tokens, dimension = values.shape
    if not 0 <= query_index < batch or not 0 <= prefix_tokens <= tokens:
        raise ValueError('Invalid query sample or prefix size')
    if branch not in ('plus', 'minus'):
        raise ValueError('Unknown signed probe')
    if axis.center.shape != (heads, dimension) or axis.direction.shape != (heads, dimension):
        raise ValueError('Reference/query head coordinates must coincide')
    if any(x.device != values.device or x.dtype != values.dtype
           for x in (axis.center, axis.direction)):
        raise ValueError('Reference/query dtype and device must coincide')
    if not torch.isfinite(values).all():
        raise ValueError('Finite query values required')
    if not axis.enabled.any() or prefix_tokens == tokens:
        return values
    patches = values[query_index, :, prefix_tokens:]
    alpha = ((patches-axis.center[:, None])*axis.direction[:, None]).sum(dim=-1)
    amount = alpha.clamp_min(0) if branch == 'plus' else alpha.clamp_max(0)
    change = amount[..., None]*axis.direction[:, None]
    result = values.clone()
    result[query_index, :, prefix_tokens:] = patches-change
    return result
