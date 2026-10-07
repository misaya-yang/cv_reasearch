"""DINO patch features + the known reference mask -> a complete query mask.

This is a direct prototype baseline, not a new scientific mechanism. It has
no FoRIS score, positional debiasing, MEAN field or native-mask input.
"""
from __future__ import annotations

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12)


def sample_grid(array, y, x):
    """Bilinear sampling at array-index coordinates, with border replication."""
    array = np.asarray(array)
    y, x = np.clip(y, 0, array.shape[0]-1), np.clip(x, 0, array.shape[1]-1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0+1, array.shape[0]-1), np.minimum(x0+1, array.shape[1]-1)
    fy, fx = y-y0, x-x0
    if array.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    return ((1-fy)*((1-fx)*array[y0, x0]+fx*array[y0, x1])
            + fy*((1-fx)*array[y1, x0]+fx*array[y1, x1]))


def resize(array, shape):
    """Half-pixel bilinear, no antialiasing; do not use PIL downsampling here."""
    y = (np.arange(shape[0])+.5)*array.shape[0]/shape[0]-.5
    x = (np.arange(shape[1])+.5)*array.shape[1]/shape[1]-.5
    xx, yy = np.meshgrid(x, y)
    return sample_grid(array, yy, xx)


def query_on_work_grid(q, geometry):
    side = int(round(np.sqrt(len(q))))
    if side*side != len(q):
        raise ValueError('Square native patch grid required')
    view = int(geometry['view_side'])
    sh, sw = geometry['resized_hw']
    oy, ox = geometry['padding_top_left']
    if min(sh, sw) <= 0 or min(oy, ox) < 0 or oy+sh > view or ox+sw > view:
        raise ValueError('Invalid recorded physical image transform')
    y = (oy+(np.arange(64)+.5)*sh/64)*side/view-.5
    x = (ox+(np.arange(64)+.5)*sw/64)*side/view-.5
    xx, yy = np.meshgrid(x, y)
    return unit(sample_grid(unit(q).reshape(side, side, -1), yy, xx)).reshape(4096, -1)


def predict(q, r, foreground_weight, valid_weight, query_geometry, original_shape):
    q, r = np.asarray(q), np.asarray(r)
    wf, valid = np.asarray(foreground_weight, float), np.asarray(valid_weight, float)
    if (q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]
            or wf.shape != (len(r),) or valid.shape != wf.shape
            or not all(np.isfinite(v).all() for v in (q, r, wf, valid))
            or np.any(wf < 0) or np.any(valid < wf) or np.any(valid > 1)
            or valid.sum() <= 0):
        raise ValueError('Finite DINO features and aligned known-reference area weights required')
    original_shape = tuple(map(int, original_shape))
    if len(original_shape) != 2 or min(original_shape) < 1:
        raise ValueError('Actual original query H/W required')
    wb = valid-wf
    if wf.sum() == 0:
        margin = np.full((64, 64), -2.)
        state = 'empty_known_reference_foreground'
    elif wb.sum() == 0:
        margin = np.full((64, 64), 2.)
        state = 'reference_has_only_foreground_no_negative_class'
    else:
        rr = unit(r)
        fg = unit(np.sum(rr*wf[:, None], axis=0)/wf.sum())
        bg = unit(np.sum(rr*wb[:, None], axis=0)/wb.sum())
        qq = query_on_work_grid(q, query_geometry)
        margin = np.einsum('nd,d->n', qq, fg-bg, optimize=False).reshape(64, 64)
        state = 'two_known_reference_prototypes'
    # A signed similarity decision; this field is not a calibrated probability.
    field = (.5+.25*margin).astype(np.float32)
    work = resize(field, (1024, 1024)) > .5
    original = resize(work.astype(np.float32), original_shape) > .5
    return dict(field=field, margin=margin, work=work, original=original,
                info=dict(state=state, query_GT_read=False,
                          inputs='DINO q/r plus known R-mask area weights and image geometry only',
                          ForIS_scores_or_masks_used=False, MEAN_used=False,
                          positional_debiasing_applied=False,
                          probability_calibration_claim=False,
                          native_query_tokens=len(q), native_reference_tokens=len(r),
                          reference_FG_area_weight=float(wf.sum()),
                          reference_BG_area_weight=float(wb.sum())))
