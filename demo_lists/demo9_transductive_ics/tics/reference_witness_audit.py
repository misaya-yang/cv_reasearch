"""Two read-only reference NN audits, never a segmentation replacement.

The source branch preserves FoRIS's actual height-axis reference normalisation;
it is NOT a cosine similarity. The cosine branch normalises both channel axes.
Both use the source's per-reference nearest-label majority, not margin >= 0.
No query labels, feature payload, learned parameters or resampled anchors.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def reference_witness_audit(ref_feats, tgt_feat, ref_masks_downsampled, chunk=256):
    """Audit [R,D,H,W] references against [D,H,W] query on the input device.

    ``nearest_index`` indexes the concatenated R,H,W reference grid;
    ``per_ref_nearest_index`` indexes each H,W grid separately. Missing classes
    have None maxima/margins plus explicit states, never artificial evidence.
    Every result key is prefixed ``source_`` or ``cosine_``. A prefixed
    ``per_ref_margin`` is [R,H,W] only when every reference contains both labels;
    otherwise inspect ``per_ref`` and ``per_ref_margin_valid``. Floating scores
    can have backend-level rounding differences across chunk sizes; flatten
    argmax ordering and exact score ties are preserved, not margin tie-broken.
    Source reference normalisation is dim=2 of [R,D,H,W] (height); target
    normalisation is dim=0 of [D,H,W] (channels). Cosine uses dim=1 and dim=0.
    Both call F.normalize with its default eps=1e-12; neither changes the host.
    """
    inputs = (ref_feats, tgt_feat, ref_masks_downsampled)
    if not all(isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError('Inputs must be torch tensors')
    if ref_feats.ndim != 4 or tgt_feat.ndim != 3:
        raise ValueError('Expected ref_feats[R,D,H,W] and tgt_feat[D,H,W]')
    r, d, h, w = ref_feats.shape
    if min(r, d, h, w) < 1 or tuple(tgt_feat.shape) != (d, h, w):
        raise ValueError('Nonempty matching feature grids are required')
    if tuple(ref_masks_downsampled.shape) != (r, h, w):
        raise ValueError('Expected reference masks[R,H,W]')
    if ref_masks_downsampled.dtype != torch.bool:
        raise TypeError('Reference masks must be bool, not soft coverage')
    if not ref_feats.is_floating_point() or ref_feats.dtype != tgt_feat.dtype:
        raise TypeError('Features must share a floating dtype')
    if len({x.device for x in inputs}) != 1:
        raise ValueError('All inputs must share a device')
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('chunk must be a positive integer')
    if not torch.isfinite(ref_feats).all() or not torch.isfinite(tgt_feat).all():
        raise ValueError('Nonfinite input features')
    query = F.normalize(tgt_feat, p=2, dim=0).reshape(d, h*w)
    if not torch.isfinite(query).all():
        raise ValueError('F.normalize produced nonfinite scores in input dtype')
    labels = ref_masks_downsampled.reshape(r, h*w)
    result = {}
    for branch, axis in (('source', 2), ('cosine', 1)):
        refs = F.normalize(ref_feats, p=2, dim=axis).reshape(r, d, h*w)
        if not torch.isfinite(refs).all():
            raise ValueError(f'{branch} normalisation is nonfinite in input dtype')
        outputs = _normalised_audit(refs, query, labels, h, w, chunk)
        result.update({f'{branch}_{key}': value for key, value in outputs.items()})
    result.update(source_ref_normalization_axis='height', source_ref_normalization_dim=2,
                  cosine_ref_normalization_axis='channels', cosine_ref_normalization_dim=1,
                  target_normalization_axis='channels', normalization_eps=1e-12,
                  query_GT_used=False, segmentation_replacement=False,
                  semantics='Both per-reference nearest-label majority; source is NOT cosine')
    return result


def _normalised_audit(refs, query, labels, h, w, chunk):
    r = refs.shape[0]
    p = h*w
    per_ref = []
    all_scores, all_indices, all_labels = [], [], []
    for j in range(r):
        has_fg, has_bg = bool(labels[j].any()), bool((~labels[j]).any())
        state = 'BOTH_LABELS' if has_fg and has_bg else ('NO_FOREGROUND' if has_bg else 'NO_BACKGROUND')
        fg_parts, bg_parts, score_parts, index_parts = [], [], [], []
        for start in range(0, p, chunk):
            # Reference rows remain in source row-major order at every chunk.
            sim = refs[j].T @ query[:, start:start+chunk]
            maxima, indices = sim.max(dim=0)
            score_parts.append(maxima)
            index_parts.append(indices)
            if has_fg:
                fg_parts.append(sim[labels[j]].max(dim=0).values)
            if has_bg:
                bg_parts.append(sim[~labels[j]].max(dim=0).values)
        nearest = torch.cat(index_parts).reshape(h, w)
        nearest_score = torch.cat(score_parts).reshape(h, w)
        nearest_label = labels[j][nearest]
        fg_max = torch.cat(fg_parts).reshape(h, w) if has_fg else None
        bg_max = torch.cat(bg_parts).reshape(h, w) if has_bg else None
        per_ref.append(dict(state=state, fg_max=fg_max, bg_max=bg_max,
                            margin=fg_max-bg_max if has_fg and has_bg else None))
        all_scores.append(nearest_score)
        all_indices.append(nearest)
        all_labels.append(nearest_label)
    per_ref_nearest_score = torch.stack(all_scores)
    per_ref_nearest_index = torch.stack(all_indices)
    per_ref_nearest_label = torch.stack(all_labels)
    nearest_score, nearest_ref_index = per_ref_nearest_score.max(dim=0)
    local_index = per_ref_nearest_index.gather(0, nearest_ref_index[None])[0]
    nearest_label = per_ref_nearest_label.gather(0, nearest_ref_index[None])[0]
    fg_maps = [item['fg_max'] for item in per_ref if item['fg_max'] is not None]
    bg_maps = [item['bg_max'] for item in per_ref if item['bg_max'] is not None]
    fg_max = torch.stack(fg_maps).max(dim=0).values if fg_maps else None
    bg_max = torch.stack(bg_maps).max(dim=0).values if bg_maps else None
    valid = torch.tensor([item['margin'] is not None for item in per_ref], device=query.device)
    votes = per_ref_nearest_label.to(torch.int32).sum(dim=0, dtype=torch.int32)
    majority = math.ceil(r/2)
    state = 'BOTH_LABELS' if fg_maps and bg_maps else ('NO_FOREGROUND' if bg_maps else 'NO_BACKGROUND')
    return dict(state=state, fg_max=fg_max, bg_max=bg_max,
                margin=fg_max-bg_max if fg_maps and bg_maps else None,
                nearest_index=nearest_ref_index*p+local_index,
                nearest_ref_index=nearest_ref_index, nearest_score=nearest_score,
                nearest_label=nearest_label, per_ref=per_ref,
                per_ref_margin=torch.stack([item['margin'] for item in per_ref]) if bool(valid.all()) else None,
                per_ref_margin_valid=valid, per_ref_nearest_index=per_ref_nearest_index,
                per_ref_nearest_label=per_ref_nearest_label, votes=votes,
                vote_soft=votes.to(query.dtype)/r, majority=majority,
                candidate=votes >= majority,
                state_class='COMPLETE' if fg_maps and bg_maps else 'DEGENERATE')
