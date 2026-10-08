"""Existing RCG fine readout, with device and memory adaptation only.

Source: 82df916:scripts/run_rcg2_stream.py. Constants are bound by the existing
rcg2_frozen.json. The learned size/cut rule is excluded; the cut is always .5.
Four fixed shifted query forwards use the same frozen encoder and native basis.
"""
import itertools
import numpy as np
import torch
import torch.nn.functional as F

CONFIG = dict(sigma=1.25, tau=0.15, window=2, shifts=[-4, 4], threshold=0.5,
              source='82df916:scripts/run_rcg2_stream.py',
              constants='evidence/local/research_20261005/rcg2_frozen.json:readout',
              exposure='historical fresh600 development-selected readout; no size/cut fitting')


@torch.inference_mode()
def shifted_features(host, target, debiased):
    pad = F.pad(target[None], (4, 4, 4, 4), mode='reflect')[0]
    fine = torch.empty(128, 128, 1024, dtype=torch.float32)
    for ay, ax in itertools.product((0, 1), repeat=2):
        sy, sx = CONFIG['shifts'][ay], CONFIG['shifts'][ax]
        crop = pad[:, 4+sy:4+sy+1024, 4+sx:4+sx+1024]
        features = F.normalize(host._extract_features(crop[None, None]), p=2, dim=2)
        if debiased:
            features = host._debias_features(features)
        fine[ay::2, ax::2] = F.normalize(features[0, 0].float(), dim=0).permute(1, 2, 0).cpu()
    return fine.reshape(16384, 1024)


@torch.inference_mode()
def field(fine, q_rounded, coarse, chunk=512):
    """Same feature/spatial weights as the historical stream; no query GT."""
    fine, q = torch.as_tensor(fine).float().cpu(), torch.as_tensor(q_rounded).float().cpu()
    z = torch.as_tensor(coarse).float().cpu().flatten()
    if fine.shape != (16384, 1024) or q.shape != (4096, 1024) or z.shape != (4096,):
        raise ValueError('Invalid fine/coarse geometry')
    if chunk < 1:
        raise ValueError('chunk must be positive')
    fi = torch.arange(128)
    base = ((fi*8+4-8).float()/16).round().long()
    off = torch.arange(-CONFIG['window'], CONFIG['window']+1)
    ti = base[:, None]+off[None]
    dist = ((ti*16+8)-(fi*8+4)[:, None]).float()/16
    ok = (ti >= 0) & (ti < 64)
    ti = ti.clamp(0, 63)
    k = len(off)
    nb = (ti[:, None, :, None].expand(128, 128, k, k)*64
          + ti[None, :, None, :].expand(128, 128, k, k)).reshape(-1, k*k)
    d2 = (dist[:, None, :, None].square()+dist[None, :, None, :].square()).reshape(-1, k*k)
    valid = (ok[:, None, :, None] & ok[None, :, None, :]).reshape(-1, k*k)
    spatial = torch.exp(-d2/(2*CONFIG['sigma']**2))*valid
    pieces = []
    for start in range(0, len(fine), chunk):
        end = min(start+chunk, len(fine))
        ids = nb[start:end]
        cosine = torch.einsum('pc,pkc->pk', fine[start:end], q[ids])
        weights = spatial[start:end]*torch.exp((cosine-1)/CONFIG['tau'])
        pieces.append((weights*z[ids]).sum(1)/weights.sum(1).clamp_min(1e-12))
    return torch.cat(pieces).reshape(128, 128).numpy()


def mask(field128):
    x = torch.as_tensor(np.ascontiguousarray(field128))[None, None]
    return (F.interpolate(x, (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > .5).numpy()
