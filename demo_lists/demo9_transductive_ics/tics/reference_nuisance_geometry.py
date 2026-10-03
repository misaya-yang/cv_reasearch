"""Full-rank angular nuisance geometry; a prepared next operator, not a gain.

Unlike projected Gaussian RICE, similarities are explicitly Mahalanobis
cosines. The native cosine scoring function can therefore remain unchanged.
Foreground intervention covariance determines which directions to suppress;
native hard-background contrast still supplies the actual class evidence.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_angular_geometry(covariance, ridge):
    """L=sqrt(ridge)*(C+ridge I)^(-1/2), full rank for ridge>0.

PSD C implies 0<L<=I, and L is exactly identity on null(C), in ideal
arithmetic. Linear pre-normalization contraction is NOT a guarantee of
smaller angular variance, cross-image error, or semantic invariance.
"""
    c = covariance.detach().double()
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not torch.isfinite(c).all():
        raise ValueError('Finite square reference covariance required')
    if not ridge > 0 or not __import__('math').isfinite(ridge):
        raise ValueError('Positive reference-only ridge required')
    if not torch.allclose(c, c.T, atol=1e-12, rtol=1e-10):
        raise ValueError('Symmetric reference covariance required')
    if not torch.count_nonzero(c):
        return torch.eye(len(c), dtype=c.dtype, device=c.device), dict(state='EXACT_IDENTITY', rank=len(c))
    values, u = torch.linalg.eigh((c + c.T) / 2)
    error = 32 * torch.finfo(c.dtype).eps * len(c) * max(float(values.abs().max()), torch.finfo(c.dtype).tiny)
    if float(values.min()) < -error:
        raise ArithmeticError('Materially indefinite nuisance covariance')
    corrected = values.clamp_min(0)  # Only previously bounded negative roundoff.
    factors = (float(ridge) / (corrected + float(ridge))).sqrt()
    # I-minus-low-rank form preserves unpenalized directions numerically better.
    l = torch.eye(len(c), device=c.device, dtype=c.dtype) - (u * (1 - factors)[None]) @ u.T
    return l, dict(state='ANGULAR_GEOMETRY_READY', full_rank=True, rank=len(c),
        minimum_factor=float(factors.min()), maximum_factor=float(factors.max()),
        negative_roundoff_eigenvalues=int((values < 0).sum()),
        ridge=float(ridge), preserves_nullspace_in_ideal_arithmetic=True,
        query_GT_used=False, angular_variance_or_task_gain_guaranteed=False)


@torch.no_grad()
def transform_tokens(tokens, geometry):
    """Unit transformed descriptors implement cos_M with M=L^T L.

This declared angular metric is different from the earlier unnormalized
Euclidean/KDE metric. Exact identity bypass returns the SAME native object,
avoiding a second normalization's rounding changes.
"""
    if tokens.shape[-1] != geometry.shape[0] or geometry.shape[0] != geometry.shape[1]:
        raise ValueError('Common descriptor coordinates required')
    if not torch.isfinite(tokens).all() or not torch.isfinite(geometry).all():
        raise ValueError('Finite tokens and geometry required')
    identity = torch.eye(len(geometry), device=geometry.device, dtype=geometry.dtype)
    if torch.equal(geometry, identity): return tokens
    value = tokens.double() @ geometry.double()
    if ((value.norm(dim=-1) == 0) & (tokens.norm(dim=-1) > 0)).any():
        raise ArithmeticError('Full-rank mapping numerically annihilated a nonzero descriptor')
    # Native FoRIS uses F.normalize's zero->zero convention. Keep that input
    # domain instead of introducing an exception for an admitted native token;
    # Mahalanobis-cosine claims apply only to nonzero descriptors.
    return F.normalize(value, dim=-1).to(tokens.dtype)


@torch.no_grad()
def transform_native_maps(maps, geometry):
    if maps.ndim != 5:
        raise ValueError('Native [B,T,C,H,W] descriptors required')
    identity = torch.eye(len(geometry), device=geometry.device, dtype=geometry.dtype)
    if torch.equal(geometry, identity): return maps
    return transform_tokens(maps.movedim(2, -1), geometry).movedim(-1, 2)
