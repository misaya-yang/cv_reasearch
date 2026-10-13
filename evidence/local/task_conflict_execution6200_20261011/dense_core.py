"""Pure frozen task cue mathematics; no IO, labels, masks or predictions.

MPS/float32 is the production path. CPU/float64 is permitted only for the
protocol's fixed first-three numerical reference, never as another output arm.
Coverage/quantiles and returned scalar decisions use CPU float64.
"""
import hashlib
import math

import numpy as np
import torch


def role_indices(weights):
    w = np.asarray(weights, dtype=np.float64).reshape(4096)
    mass = float(w.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Unavailable reference role')
    thresholds = (np.arange(128, dtype=np.float64) + 0.5) * mass / 128
    return np.searchsorted(np.cumsum(w, dtype=np.float64), thresholds,
                           side='right').astype(np.int64)


def compute_dense(raw_r, raw_q, coverage_r, coverage_a, *,
                  device='mps', dtype=torch.float32, chunk_rows=1024):
    """Return two transient float64 CPU h64 arrays plus compact diagnostics.

Caller synchronizes MPS for cost/memory accounting and implements actual-error
CPU/float32 retry. No numeric-invalid-to-device retry is performed here.
"""
    if not 1 <= chunk_rows <= 1024:
        raise ValueError('Invalid chunk scheduling')
    p = np.asarray(coverage_r, dtype=np.float64).reshape(4096)
    a = np.asarray(coverage_a, dtype=np.float64).reshape(4096)
    if (not np.isfinite(p).all() or not np.isfinite(a).all()
            or np.any((p < 0) | (p > 1)) or np.any((a < 0) | (a > 1))):
        raise ValueError('Invalid area coverage')
    mf, mb, ma = (float(x.sum(dtype=np.float64)) for x in (p, 1-p, a))
    r = torch.as_tensor(np.asarray(raw_r).reshape(4096, 1024),
                        dtype=dtype, device=device)
    q = torch.as_tensor(np.asarray(raw_q).reshape(4096, 1024),
                        dtype=dtype, device=device)
    rn, qn = torch.linalg.vector_norm(r, dim=1), torch.linalg.vector_norm(q, dim=1)
    if not bool((torch.isfinite(rn).all() & torch.isfinite(qn).all()
                 & (rn > 0).all() & (qn > 0).all()).item()):
        raise ValueError('Zero/nonfinite raw token is an input error')
    r, q = r / rn[:, None], q / qn[:, None]
    stats = dict(MF=mf, MB=mb, MA=ma, device=str(device),
                 dtype=str(dtype), reference_reason=None, anchor_reason=None)
    result = dict(reference_h=None, anchor_h=None, stats=stats)
    if mf <= 0 or mb <= 0:
        stats.update(reference_reason='reference_role_unavailable',
                     anchor_reason='reference_role_unavailable')
        return result
    wp = torch.as_tensor(p, dtype=dtype, device=device)
    mu_rf = (wp[:, None] * r).sum(dim=0) / mf
    rr = r - mu_rf
    second = (rr * rr).sum(dim=1)
    scale2 = 0.5 * ((wp * second).sum() / mf
                    + ((1-wp) * second).sum() / mb)
    s2 = float(scale2.item())
    stats['source_scale_squared'] = s2
    if not math.isfinite(s2) or s2 <= 0:
        stats.update(reference_reason='reference_scale_invalid',
                     anchor_reason='reference_scale_invalid')
        return result
    scale = torch.sqrt(scale2)
    fi, bi = role_indices(p), role_indices(1-p)
    stats['FG_bank_index_sha256'] = hashlib.sha256(fi.tobytes()).hexdigest()
    stats['BG_bank_index_sha256'] = hashlib.sha256(bi.tobytes()).hexdigest()
    fg = rr[torch.as_tensor(fi, device=device)] / scale
    bg = rr[torch.as_tensor(bi, device=device)] / scale
    fg_norm = (fg*fg).sum(dim=1)
    bg_norm = (bg*bg).sum(dim=1)

    def memory_sample():
        if str(device) == 'mps':
            torch.mps.synchronize()
            for key, value in (
                    ('sampled_MPS_allocated_memory_bytes', torch.mps.current_allocated_memory()),
                    ('sampled_MPS_driver_memory_bytes', torch.mps.driver_allocated_memory())):
                stats[key] = max(stats.get(key, 0), int(value))

    memory_sample()

    def cue(centre):
        pieces = []
        valid = True
        for start in range(0, 4096, chunk_rows):
            u = (q[start:start+chunk_rows] - centre) / scale
            u2 = (u*u).sum(dim=1, keepdim=True)
            df = (u2 + fg_norm[None, :] - 2*(u @ fg.T)).clamp_min(0)
            db = (u2 + bg_norm[None, :] - 2*(u @ bg.T)).clamp_min(0)
            lf = torch.logsumexp(-df/(2*0.07), dim=1) - math.log(128)
            lb = torch.logsumexp(-db/(2*0.07), dim=1) - math.log(128)
            contrast = lf - lb
            h = torch.tanh(contrast/2)
            memory_sample()
            valid = valid and bool((torch.isfinite(contrast).all()
                                    & torch.isfinite(h).all()).item())
            pieces.append(h.detach().to(device='cpu').to(dtype=torch.float64).numpy())
        return np.concatenate(pieces).reshape(64, 64) if valid else None

    result['reference_h'] = cue(mu_rf)
    if result['reference_h'] is None:
        stats['reference_reason'] = 'numeric_cue_invalid'
    if ma <= 0:
        stats['anchor_reason'] = 'anchor_empty'
        return result
    wa = torch.as_tensor(a, dtype=dtype, device=device)
    mu_qa = (wa[:, None] * q).sum(dim=0) / ma
    stats['anchor_centre_delta_norm'] = float(torch.linalg.vector_norm(mu_qa-mu_rf).item())
    result['anchor_h'] = cue(mu_qa)
    if result['anchor_h'] is None:
        stats['anchor_reason'] = 'numeric_cue_invalid'
    else:
        sa = float(np.dot(a, result['anchor_h'].reshape(4096))/ma)
        stats['S_A'] = sa
        if not math.isfinite(sa) or sa <= 0:
            stats['anchor_reason'] = 'anchor_incompatible'
            result['anchor_h'] = None
    return result
