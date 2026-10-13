"""Shared source FG/BG and query FG densities; pure mathematics, no file IO."""
import hashlib
import math

import numpy as np
import torch


def indices(weights):
    w = np.asarray(weights, dtype=np.float64).reshape(4096)
    mass = float(w.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Unavailable role bank')
    return np.searchsorted(np.cumsum(w), (np.arange(128)+0.5)*mass/128,
                           side='right').astype(np.int64)


def compute(raw_r, raw_q, p, a, *, device='mps', dtype=torch.float32):
    p, a = (np.asarray(x, dtype=np.float64).reshape(4096) for x in (p, a))
    if (not np.isfinite(p).all() or not np.isfinite(a).all()
            or np.any((p < 0) | (p > 1)) or np.any((a < 0) | (a > 1))):
        raise ValueError('Invalid coverage')
    mf, mb, ma = map(float, (p.sum(), (1-p).sum(), a.sum()))
    stats = dict(MF=mf, MB=mb, MA_unvetted=ma, MA_vetted=None,
                 reference_reason=None, vetted_reason=None, unvetted_reason=None,
                 dtype=str(dtype), device=str(device))
    result = dict(reference_h=None, vetted_h=None, unvetted_h=None, stats=stats)
    r, q = (torch.as_tensor(np.asarray(x).reshape(4096, 1024), device=device,
                            dtype=dtype) for x in (raw_r, raw_q))
    rn, qn = torch.linalg.vector_norm(r, dim=1), torch.linalg.vector_norm(q, dim=1)
    if not bool((torch.isfinite(rn).all() & torch.isfinite(qn).all()
                 & (rn > 0).all() & (qn > 0).all()).item()):
        raise ValueError('Zero/nonfinite raw token is an input error')
    r, q = r/rn[:, None], q/qn[:, None]
    if mf <= 0 or mb <= 0:
        stats.update(reference_reason='reference_role_unavailable',
                     vetted_reason='reference_role_unavailable',
                     unvetted_reason='reference_role_unavailable')
        return result
    wp = torch.as_tensor(p, dtype=dtype, device=device)
    centre = (wp[:, None]*r).sum(0)/mf
    rr = r-centre
    distances = (rr*rr).sum(1)
    s2_tensor = 0.5*((wp*distances).sum()/mf+((1-wp)*distances).sum()/mb)
    s2 = float(s2_tensor.item())
    stats['source_scale_squared'] = s2
    if not math.isfinite(s2) or s2 <= 0:
        stats.update(reference_reason='reference_scale_invalid',
                     vetted_reason='reference_scale_invalid',
                     unvetted_reason='reference_scale_invalid')
        return result
    scale = torch.sqrt(s2_tensor)
    qr, rr = (q-centre)/scale, rr/scale
    rf_indices, rb_indices = indices(p), indices(1-p)
    rf, rb = (rr[torch.as_tensor(i, device=device)] for i in (rf_indices, rb_indices))
    for name, ix in [('reference_FG', rf_indices), ('reference_BG', rb_indices)]:
        stats[name+'_index_sha256'] = hashlib.sha256(ix.tobytes()).hexdigest()

    def memory_sample():
        if str(device) == 'mps':
            torch.mps.synchronize()
            for key, value in (
                    ('sampled_MPS_allocated_memory_bytes', torch.mps.current_allocated_memory()),
                    ('sampled_MPS_driver_memory_bytes', torch.mps.driver_allocated_memory())):
                stats[key] = max(stats.get(key, 0), int(value))

    def log_likelihood(bank):
        bank_norm = (bank*bank).sum(1)
        pieces = []
        for begin in range(0, 4096, 1024):
            u = qr[begin:begin+1024]
            d2 = ((u*u).sum(1, keepdim=True)+bank_norm[None, :]
                  - 2*(u @ bank.T)).clamp_min(0)
            pieces.append(torch.logsumexp(-d2/(2*0.07), dim=1)-math.log(128))
            memory_sample()
        return torch.cat(pieces)

    def cpu64(tensor):
        return tensor.detach().to(device='cpu').to(dtype=torch.float64).numpy()

    def contrasted(lf, lb):
        difference = lf-lb
        h = torch.tanh(difference/2)
        if not bool((torch.isfinite(difference).all() & torch.isfinite(h).all()).item()):
            return None, None
        return cpu64(h).reshape(64, 64), cpu64(difference)

    lb = log_likelihood(rb)
    h, reference_lr = contrasted(log_likelihood(rf), lb)
    result['reference_h'] = h
    if h is None:
        stats.update(reference_reason='numeric_cue_invalid',
                     vetted_reason='numeric_cue_invalid',
                     unvetted_reason='numeric_cue_invalid')
        return result
    # Compare the original two log likelihoods, before tanh rounding.
    av = a*(reference_lr > 0)
    mav = float(av.sum())
    stats.update(MA_vetted=mav, reference_rejected_anchor_mass=ma-mav,
                 reference_positive_native_tokens=int(np.count_nonzero(reference_lr > 0)))
    for name, weights, mass in [('vetted', av, mav), ('unvetted', a, ma)]:
        if mass <= 0:
            stats[name+'_reason'] = 'query_anchor_empty_or_rejected'
            continue
        ix = indices(weights)
        stats[name+'_query_FG_index_sha256'] = hashlib.sha256(ix.tobytes()).hexdigest()
        qf = qr[torch.as_tensor(ix, device=device)]
        hq, _ = contrasted(log_likelihood(qf), lb)
        result[name+'_h'] = hq
        if hq is None:
            stats[name+'_reason'] = 'numeric_cue_invalid'
    memory_sample()
    return result
