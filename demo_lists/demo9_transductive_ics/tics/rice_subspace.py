"""Reference-conditioned generalized discriminant, CPU preparation kernel.

This is the RICE proposal's core operator, not a complete segmentation method.
Inputs are already declared reference contrast/nuisance columns. No model,
query annotations, learned dictionary, host blending, transport or calibration.
No normalization after the discriminant transform. Classical algebra alone
does not establish novelty or cross-image segmentation benefit.
"""
import numpy as np


def inverse_nuisance_action(nuisance, values, ridge):
    """Exact (ridge I + V V^T)^-1 values, choosing the smaller solve.

The dual dimension is V.shape[1], NOT the number of contrast columns. When V
has more columns than feature dimensions, the primal solve is smaller.
"""
    v, x = np.asarray(nuisance, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or x.ndim != 2 or len(v) != len(x):
        raise ValueError('Expected V[D,m], values[D,n]')
    if not np.isfinite(v).all() or not np.isfinite(x).all() or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Finite inputs and positive declared ridge required')
    d, m = v.shape
    if m < d:
        return (x - v @ np.linalg.solve(ridge * np.eye(m) + v.T @ v, v.T @ x)) / ridge
    return np.linalg.solve(ridge * np.eye(d) + v @ v.T, x)


def reference_subspace(contrasts, nuisance, ridge, rank_cap=16, retained_energy=.9):
    """Sd=D D^T; Sn=ridge I+V V^T; return W with W^T Sn W=I.

Discard only numerical zero eigenvalues using a declared floating-point
roundoff criterion. Report repeated eigenvalues at the truncation boundary;
do not claim a uniquely identifiable basis or use query GT to select rank.
"""
    d = np.asarray(contrasts, dtype=np.float64)
    v = np.asarray(nuisance, dtype=np.float64)
    if d.ndim != 2 or v.ndim != 2 or len(d) != len(v) or rank_cap < 1 or not 0 < retained_energy <= 1:
        raise ValueError('Compatible reference matrices, positive cap and energy in (0,1] required')
    invd = inverse_nuisance_action(v, d, ridge)
    gram = d.T @ invd
    gram = (gram + gram.T) / 2
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    scale = max(float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0., np.finfo(float).tiny)
    cutoff = np.finfo(float).eps * max(1, *gram.shape) * scale
    if len(eigenvalues) and eigenvalues.min() < -32 * cutoff:
        raise ArithmeticError('PSD Gram has a materially negative eigenvalue')
    valid = eigenvalues > cutoff
    values, vectors = eigenvalues[valid], vectors[:, valid]
    if not len(values):
        return np.empty((len(d), 0)), dict(state='NO_REFERENCE_DISCRIMINANT', rank=0,
                                          eigenvalues=[], numerical_zero_cutoff=cutoff)
    energy_rank = int(np.searchsorted(np.cumsum(values) / values.sum(), retained_energy)) + 1
    rank = min(rank_cap, energy_rank)
    w = invd @ (vectors[:, :rank] / np.sqrt(values[:rank])[None, :])
    repeated_boundary = rank < len(values) and abs(values[rank - 1] - values[rank]) <= 32 * cutoff
    return w, dict(state='REFERENCE_SUBSPACE_BUILT', rank=rank, eigenvalues=values.tolist(),
        retained_energy=float(values[:rank].sum() / values.sum()), numerical_zero_cutoff=cutoff,
        truncates_numerically_tied_eigenspace=bool(repeated_boundary),
        inverse_solve_dimension=min(v.shape), contrast_columns=d.shape[1], nuisance_columns=v.shape[1],
        task_gain_measured=False)


def signed_kernel_evidence(query, foreground, background, directions, temperature,
                           query_chunk=128, anchor_chunk=512):
    """Complete, equally weighted FG-versus-BG kernel score; no H0 residual.

tau(logmeanexp_FG - logmeanexp_BG) with distances in the transformed
coordinates. Query features contain no labels. A positive score favours FG
under this declared equal-prior kernel model; it is not a calibrated posterior.
"""
    q, fg, bg, w = [np.asarray(a, dtype=np.float64) for a in (query, foreground, background, directions)]
    if any(a.ndim != 2 for a in (q, fg, bg, w)) or not len(fg) or not len(bg) or w.shape[1] == 0:
        raise ValueError('Nonempty foreground/background and discriminant directions required')
    if any(a.shape[1] != w.shape[0] for a in (q, fg, bg)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Common coordinates and positive reference-fixed temperature required')
    if not all(np.isfinite(a).all() for a in (q, fg, bg, w)) or min(query_chunk, anchor_chunk) < 1:
        raise ValueError('Finite inputs and positive chunks required')
    # Deliberately no L2 normalization after W: it would change the metric.
    q, fg, bg = q @ w, fg @ w, bg @ w
    output = []
    for start in range(0, len(q), query_chunk):
        block = q[start:start + query_chunk]
        ratios = []
        for anchors in (fg, bg):
            aggregate = np.full(len(block), -np.inf)
            for offset in range(0, len(anchors), anchor_chunk):
                a = anchors[offset:offset + anchor_chunk]
                distances = np.maximum((block ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * block @ a.T, 0)
                logits = -distances / (2 * temperature)
                peak = logits.max(1)
                local = peak + np.log(np.exp(logits - peak[:, None]).sum(1))
                aggregate = np.logaddexp(aggregate, local)
            ratios.append(aggregate - np.log(len(anchors)))
        output.append(temperature * (ratios[0] - ratios[1]))
    return np.concatenate(output) if output else np.empty(0)
