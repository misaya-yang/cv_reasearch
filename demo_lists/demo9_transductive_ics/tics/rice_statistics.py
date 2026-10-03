"""Reference-only statistics for the frozen RICE covariance comparison.

The caller supplies three aligned, already debiased feature views: original
and two lawful background interventions. No query, target labels, model,
training, transport, calibration or image transformation is performed here.
Projection coordinates are NOT normalized after the discriminant transform.
"""
import math
import torch
import torch.nn.functional as F


def _sample(indices, cap):
    """Evenly spaced original-grid indices, independent of features/quality."""
    if len(indices) <= cap:
        return indices
    offsets = torch.arange(cap, device=indices.device) * len(indices) // cap
    return indices[offsets]


def _modes(x, requested, iterations=30):
    """Farthest-first Euclidean initialization; deterministic Lloyd updates."""
    centers = [x[0]]
    closest = (x - centers[0]).square().sum(1)
    for _ in range(1, min(requested, len(x))):
        index = int(closest.argmax())
        if float(closest[index]) == 0:
            break
        centers.append(x[index])
        closest = torch.minimum(closest, (x - x[index]).square().sum(1))
    centers = torch.stack(centers)
    labels = None
    for _ in range(iterations):
        distances = (x[:, None] - centers[None]).square().sum(-1)
        assigned = distances.argmin(1)
        occupied = torch.unique(assigned, sorted=True)
        if len(occupied) < len(centers):
            centers = centers[occupied]
            continue
        updated = torch.stack([x[assigned == k].mean(0) for k in range(len(centers))])
        if labels is not None and torch.equal(assigned, labels):
            centers = updated
            break
        labels, centers = assigned, updated
    # Returned labels and means always describe the same partition, including
    # an iteration-limit exit; remove empty modes without inventing anchors.
    labels = ((x[:, None] - centers[None]).square().sum(-1)).argmin(1)
    occupied = torch.unique(labels, sorted=True)
    labels = torch.searchsorted(occupied, labels)
    centers = torch.stack([x[labels == k].mean(0) for k in range(len(occupied))])
    return labels, centers


def _scatter(values, labels):
    """Unnormalized scatter across views, centered separately in each mode."""
    dimension = values.shape[-1]
    scatter = values.new_zeros((dimension, dimension))
    for label in torch.unique(labels, sorted=True):
        block = values[:, labels == label].reshape(-1, dimension)
        residual = block - block.mean(0)
        scatter += residual.T @ residual
    return (scatter + scatter.T) / 2


def _context(values):
    residual = values - values.mean(0, keepdim=True)
    flat = residual.reshape(-1, values.shape[-1])
    return flat.T @ flat / len(flat)


@torch.no_grad()
def build_reference_statistics(views, coverage, grid):
    """Return fixed contrasts, five covariance controls and reference audits.

    views: [3,N,D], original first, common native feature coordinates.
    coverage: [N], lawful original-reference foreground area fractions.
    grid: (height,width), with N=height*width and row-major patch indexing.
    Every matrix is FP64 on the input device. Sampling limits are FG128/BG256.
    All arms add the SAME ridge=0.1*trace(pooled_fisher)/D at projection time.
    A zero pooled trace is explicitly DEGENERATE_ZERO_POOLED_TRACE; no ridge
    or direction fallback is manufactured. Empty pure classes are explicit.
    """
    if not isinstance(views, torch.Tensor) or views.ndim != 3 or views.shape[0] != 3:
        raise ValueError('views must be [3,N,D]: original plus two aligned interventions')
    if len(grid) != 2 or any(int(v) != v or v <= 0 for v in grid):
        raise ValueError('grid must contain two positive integers')
    if views.shape[1] != int(grid[0]) * int(grid[1]) or views.shape[2] < 1:
        raise ValueError('grid/feature shape mismatch')
    x = views.detach().to(dtype=torch.float64)
    c = torch.as_tensor(coverage, device=x.device, dtype=torch.float64)
    if c.shape != (x.shape[1],) or not torch.isfinite(c).all() or not torch.isfinite(x).all():
        raise ValueError('finite views and coverage[N] required')
    if ((c < 0) | (c > 1)).any():
        raise ValueError('coverage must be lawful fractions in [0,1]')
    fg = _sample(torch.nonzero(c >= .9).flatten(), 128)
    bg = _sample(torch.nonzero(c <= .1).flatten(), 256)
    audit = dict(grid=[int(v) for v in grid], views=3, dimension=x.shape[-1],
                 foreground_count=len(fg), background_count=len(bg),
                 query_used=False, query_ground_truth_used=False,
                 clustering_view='original', sampling='evenly_spaced_row_major',
                 shuffle_seed=2052, covariance_dtype='float64',
                 ridge_rule='0.1*trace(pooled_fisher)/dimension')
    if not len(fg) or not len(bg):
        return dict(state='DEGENERATE_MISSING_PURE_CLASS', contrasts=x.new_empty((x.shape[-1], 0)),
                    covariances={}, ridge=0., foreground_indices=fg, background_indices=bg, audit=audit)
    requested = min(8, max(1, len(fg) // 16))
    fg_labels, fg_means = _modes(x[0, fg], requested)
    native_cosine = F.normalize(x[0, bg], dim=1) @ F.normalize(fg_means, dim=1).T
    hard_count = max(1, math.ceil(.2 * len(bg)))
    top = torch.argsort(native_cosine, dim=0, descending=True, stable=True)[:hard_count]
    hard_local = torch.unique(top.flatten(), sorted=True)
    hard_bg = bg[hard_local]
    bg_labels, bg_means = _modes(x[0, hard_bg], min(4, len(hard_bg)))
    k, j = len(fg_means), len(bg_means)
    contrasts = (fg_means[:, None] - bg_means[None]).reshape(k * j, -1).T / math.sqrt(k * j)
    fg_views = x[:, fg]
    ctx = _context(fg_views)
    within = _scatter(fg_views.mean(0, keepdim=True), fg_labels) / len(fg)
    fg_aug = _scatter(fg_views, fg_labels) / (3 * len(fg))
    bg_views = x[:, bg]
    bg_residual = bg_views.reshape(-1, x.shape[-1])
    bg_residual = bg_residual - bg_residual.mean(0)
    pooled = (_scatter(fg_views, fg_labels) + bg_residual.T @ bg_residual) / (3 * (len(fg) + len(bg)))
    # Each view has independent, fixed per-mode permutation. Permuting ONLY
    # view order would leave paired covariance unchanged and is not a control.
    generator = torch.Generator(device='cpu').manual_seed(2052)
    shuffled = fg_views.clone()
    for a in range(3):
        for mode in range(k):
            positions = torch.nonzero(fg_labels == mode).flatten()
            perm = torch.randperm(len(positions), generator=generator).to(x.device)
            shuffled[a, positions] = fg_views[a, positions[perm]]
    shuffled_ctx = _context(shuffled)
    shuffled_aug = _scatter(shuffled, fg_labels) / (3 * len(fg))
    ridge = .1 * float(torch.trace(pooled)) / x.shape[-1]
    audit.update(foreground_modes=k, requested_foreground_modes=requested,
                 background_modes=j, hard_background_count=len(hard_bg),
                 hard_background_top_fraction=.2,
                 covariance_identity_max_error=float((fg_aug - ctx - within).abs().max()),
                 augmented_shuffle_max_error=float((fg_aug - shuffled_aug).abs().max()),
                 paired_shuffle_max_difference=float((ctx - shuffled_ctx).abs().max()),
                 pooled_trace=float(torch.trace(pooled)))
    state = 'REFERENCE_STATISTICS_BUILT' if ridge > 0 else 'DEGENERATE_ZERO_POOLED_TRACE'
    return dict(state=state, contrasts=contrasts,
                covariances=dict(identity=torch.eye(x.shape[-1], device=x.device, dtype=x.dtype),
                                 fg_aug=fg_aug, pooled_fisher=pooled,
                                 paired_only=ctx, paired_shuffle=shuffled_ctx),
                context_covariance=ctx, mean_within_covariance=within,
                ridge=ridge, foreground_indices=fg, background_indices=bg,
                hard_background_indices=hard_bg, foreground_labels=fg_labels,
                background_labels=bg_labels, foreground_means=fg_means,
                background_means=bg_means, audit=audit)


@torch.no_grad()
def discriminant_projection(contrasts, covariance, ridge, rank_cap=16):
    """Exact FP64 generalized discriminant with all eligible top<=16 axes.

    Sd=D D^T; Sn=C+ridge I; W=Sn^-1 D R Lambda^-1/2.
    Return (W,audit). Empty contrasts return rank zero; zero/nonpositive ridge
    yields an explicit degenerate state rather than an arbitrary stabilizer.
    Repeated eigenvalues can make the selected basis nonunique; its metric is
    what comparisons should examine. No post-W normalization is performed.
    """
    if not isinstance(contrasts, torch.Tensor) or contrasts.ndim != 2:
        raise ValueError('contrasts must be a torch tensor [D,M]')
    d = contrasts.detach().double()
    c = torch.as_tensor(covariance, device=d.device, dtype=torch.float64)
    if c.shape != (len(d), len(d)) or not torch.isfinite(c).all() or not torch.isfinite(d).all():
        raise ValueError('finite compatible covariance/contrasts required')
    if rank_cap < 1 or rank_cap > 16 or int(rank_cap) != rank_cap or not math.isfinite(float(ridge)):
        raise ValueError('rank_cap must be an integer in [1,16], ridge finite')
    if not torch.allclose(c, c.T, atol=1e-12, rtol=1e-10):
        raise ValueError('covariance must be symmetric')
    if ridge <= 0:
        return d.new_empty((len(d), 0)), dict(state='DEGENERATE_NONPOSITIVE_RIDGE', rank=0)
    sn = (c + c.T) / 2 + float(ridge) * torch.eye(len(d), device=d.device, dtype=d.dtype)
    factor, info = torch.linalg.cholesky_ex(sn)
    if int(info) != 0:
        raise ValueError('regularized covariance must be positive definite')
    inverse_d = torch.cholesky_solve(d, factor)
    gram = d.T @ inverse_d
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    order = torch.argsort(values, descending=True, stable=True)
    values, vectors = values[order], vectors[:, order]
    scale = max(float(values.abs().max()) if len(values) else 0., torch.finfo(d.dtype).tiny)
    cutoff = torch.finfo(d.dtype).eps * max(1, gram.shape[0]) * scale
    if len(values) and float(values.min()) < -32 * cutoff:
        raise ArithmeticError('materially negative generalized Gram eigenvalue')
    eligible = values > cutoff
    positive, vectors = values[eligible], vectors[:, eligible]
    rank = min(int(rank_cap), len(positive))
    if not rank:
        return d.new_empty((len(d), 0)), dict(state='NO_REFERENCE_DISCRIMINANT', rank=0,
                                            numerical_zero_cutoff=cutoff, eigenvalues=[])
    w = inverse_d @ (vectors[:, :rank] / positive[:rank].sqrt()[None])
    orth_error = float((w.T @ sn @ w - torch.eye(rank, device=d.device, dtype=d.dtype)).abs().max())
    residual = d @ (d.T @ w) - (sn @ w) * positive[:rank][None]
    relative_error = float(residual.norm() / max(float((d @ (d.T @ w)).norm()), torch.finfo(d.dtype).tiny))
    if orth_error > 1e-7 or relative_error > 1e-7:
        raise ArithmeticError('generalized projection failed numerical verification')
    tied = rank < len(positive) and float((positive[rank - 1] - positive[rank]).abs()) <= 32 * cutoff
    return w, dict(state='REFERENCE_SUBSPACE_BUILT', rank=rank,
                   eigenvalues=positive.tolist(), numerical_zero_cutoff=cutoff,
                   truncates_numerically_tied_eigenspace=tied,
                   whitening_max_error=orth_error, generalized_relative_error=relative_error,
                   rank_rule='all_numerically_positive_top16', post_projection_normalization=False)
