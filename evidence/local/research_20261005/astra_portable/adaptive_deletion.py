"""Retained exploratory deletion rule and fixed, same-input controls.

No query ground truth, native/pre mask, or prior run is an input. The canonical
budget uses the packet's original Part1 foreground/background maxima. Recomputed
maxima from quantized features are an explicit alternative, never a silent swap.
"""
import numpy as np
import torch
import torch.nn.functional as F
from rcg_readout import unpack
from statistics_randomstate import episode_key, photograph_components

ARMS = ['explicit_BG_delete', 'FG_count_matched_delete',
        'RCG_count_matched_delete', 'global_RCG_same_budget']
FIXED_ARM = 'crossfit_fraction_RCG_rank'

def render(field):
    a = np.asarray(field, np.float32).reshape(64, 64)
    return F.interpolate(torch.from_numpy(a)[None, None], (1024, 1024),
                         mode='bilinear', align_corners=False)[0, 0].numpy()

def lowest(domain, field, count):
    indices = np.flatnonzero(domain.ravel())
    if not 0 <= count <= len(indices):
        raise ValueError('Deletion count is outside its allowed domain')
    order = np.argsort(field.ravel()[indices], kind='stable')
    mask = np.zeros(domain.size, bool)
    mask[indices[order[:count]]] = True
    return mask.reshape(domain.shape)

def feature_maxima(q, r, cov):
    """Replay cached Part1 dot products without renormalizing FP16 tokens.

    Original packets were generated before feature storage quantization; the
    cached feature vectors are already approximately unit norm. Re-normalizing
    them would add a second numerical change to this explicit replay.
    """
    q = np.asarray(q, np.float32); r = np.asarray(r, np.float32)
    fg = np.asarray(cov).ravel() >= .5
    if not fg.any() or fg.all():
        raise ValueError('Both reference roles are required by the locked rule')
    norm_error = float(max(np.abs(np.linalg.norm(q, axis=1)-1).max(),
                           np.abs(np.linalg.norm(r, axis=1)-1).max()))
    if norm_error >= .001:
        raise ValueError('Cache is not in the expected approximately unit Part1 space')
    sim = q @ r.T
    return sim[:, fg].max(1), sim[:, ~fg].max(1), norm_error

def predict(masks, fields, fg_max, bg_max):
    R = unpack(masks['RCG']); D = unpack(masks['conservative_delete'])
    domain = R & ~D
    fg = np.asarray(fg_max, np.float64).reshape(4096)
    bg = np.asarray(bg_max, np.float64).reshape(4096)
    if not (np.isfinite(fg).all() and np.isfinite(bg).all()):
        raise ValueError('Nonfinite source-role maxima')
    host = render(fields['RCG_field'])
    if not np.array_equal(host > .5, R):
        raise AssertionError('RCG mask/field renderer parity failed')
    # Subtract in float64, then cast to float32 before interpolation, exactly
    # as in the original frozen composition, not after separate interpolation.
    margin = render(bg-fg)
    explicit = domain & (margin > 0)
    K = int(explicit.sum())
    delete_fg = lowest(domain, render(fg), K)
    delete_restricted = lowest(domain, host, K)
    delete_global = lowest(R, host, K)
    result = dict(explicit_BG_delete=R & ~explicit,
                  FG_count_matched_delete=R & ~delete_fg,
                  RCG_count_matched_delete=R & ~delete_restricted,
                  global_RCG_same_budget=R & ~delete_global)
    for mask in result.values():
        assert not (mask & ~R).any()
        assert int((R & ~mask).sum()) == K
    audit = dict(eligible_deletions=int(domain.sum()), explicit_deletions=K,
                 zero_margin_in_domain=int((domain & (margin == 0)).sum()),
                 global_deleted_outside_domain=int((delete_global & ~domain).sum()))
    return {name: np.packbits(mask) for name, mask in result.items()}, audit

def crossfit_fractions(rows, budgets, smoke=False):
    """No labels: other-fold sum(K)/sum(domain), excluding shared-photo groups."""
    groups = photograph_components(rows)
    group_for = {i: g for g, indices in enumerate(groups) for i in indices}
    result = {}
    for fold in sorted({int(row['fold']) for row in rows}):
        test = [i for i, row in enumerate(rows) if int(row['fold']) == fold]
        blocked = {group_for[i] for i in test}
        train = [i for i, row in enumerate(rows)
                 if int(row['fold']) != fold and group_for[i] not in blocked]
        K = sum(budgets[episode_key(rows[i])]['explicit_deletions'] for i in train)
        N = sum(budgets[episode_key(rows[i])]['eligible_deletions'] for i in train)
        if N == 0 and not smoke:
            raise ValueError('No other-fold eligible pixels for fixed-fraction control')
        excluded = [episode_key(rows[i]) for i, row in enumerate(rows)
                    if int(row['fold']) != fold and group_for[i] in blocked]
        result[str(fold)] = dict(fraction=K/N if N else 0.,
            train_BG_gate_deletions=K, train_domain_pixels=N,
            train_episodes=len(train), test_episodes=len(test),
            train_keys=[episode_key(rows[i]) for i in train],
            excluded_other_fold_shared_photo_keys=excluded, no_shared_photos=True,
            smoke_zero_training_domain=(N == 0))
    return result

def fixed_fraction_mask(masks, fields, fraction):
    R = unpack(masks['RCG']); D = unpack(masks['conservative_delete'])
    domain = R & ~D
    K = int(np.floor(float(fraction)*int(domain.sum()) + .5))
    return np.packbits(R & ~lowest(domain, render(fields['RCG_field']), K)), K
