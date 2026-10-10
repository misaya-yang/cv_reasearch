"""Nine local readouts from one raw whole-query feature map.

Only the representation changes from native-window encoding: sample raw O24
before normalization for both the source FoRIS frontend and region descriptors.
The whole-image FoRIS pseudo FG/BG guide, region rule and voting stay fixed.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.sparse import coo_matrix

from ics.foris import run_foris

WINDOWS = tuple((x, y, x+512, y+512) for y in (0, 256, 512) for x in (0, 256, 512))
_AXIS = np.repeat(np.arange(64, dtype=np.int64), 8)
_PATCH = (_AXIS[:, None]*64+_AXIS[None, :]).reshape(-1)


def feature_map(raw):
    raw = torch.as_tensor(raw)
    if raw.dtype != torch.float32 or tuple(raw.shape) != (4096, 1024) or raw.device.type != 'cpu':
        raise ValueError('Require the original CPU FP32 O24 4096x1024 array')
    return raw.reshape(64, 64, 1024).permute(2, 0, 1).contiguous()


class PairFeatures(torch.nn.Module):
    """Explicit two-image replay; absent or mismatched inputs cannot encode."""
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.expected = self.maps = None
        self.calls = 0

    def bind(self, reference_input, reference_map, target_input, target_map):
        self.expected = torch.stack((reference_input, target_input))
        self.maps = torch.stack((reference_map, target_map))

    def get_intermediate_layers(self, inputs, n=1, reshape=True):
        if n != 1 or not reshape or self.expected is None or not torch.equal(inputs, self.expected):
            raise ValueError('The frontend requested an unbound input pair')
        self.calls += 1
        return [self.maps]


def sample_window(full_map, box):
    """Same 64-grid patch centers as the pre-existing interpolation control."""
    x0, y0, _, _ = box
    xx = x0+(torch.arange(64, dtype=torch.float32)+.5)*8
    yy = y0+(torch.arange(64, dtype=torch.float32)+.5)*8
    y, x = torch.meshgrid(2*yy/1024-1, 2*xx/1024-1, indexing='ij')
    return F.grid_sample(full_map[None], torch.stack((x, y), -1)[None],
        mode='bilinear', padding_mode='border', align_corners=False)[0].contiguous()


def region_descriptors(features, ids, n):
    flat = ids.reshape(-1)
    fg = flat > 0
    weights = coo_matrix((np.full(int(fg.sum()), 1/64, dtype=np.float32),
        (flat[fg]-1, _PATCH[fg])), shape=(n, 4096)).tocsr()
    mean = weights@features.numpy()/np.asarray(weights.sum(1), dtype=np.float32)
    return F.normalize(torch.from_numpy(mean), dim=1)


@torch.inference_mode()
def segment(host, adapter, reference, reference_mask, query, reference_raw,
            query_raw, global_guide, transform, *, tau=.07):
    """Return primary, same-K geometry and unfiltered masks, without DINO/GT."""
    if tuple(np.asarray(global_guide).shape) != (1024, 1024):
        raise ValueError('Require the frozen whole-query FoRIS guide on its1024 canvas')
    reference_map, full_map = feature_map(reference_raw), feature_map(query_raw)
    reference_input = transform(reference)
    canonical = transform.transforms[0](query)
    guide = np.asarray(global_guide, dtype=bool)
    coverage = F.adaptive_avg_pool2d(torch.from_numpy(guide).float()[None, None], (64, 64)).reshape(-1)
    mass = [coverage.sum(), (1-coverage).sum()]
    missing_role = any(float(s) <= 0 for s in mass)
    full = F.normalize(torch.as_tensor(query_raw), dim=1)
    log_weights = [] if missing_role else [torch.where(c > 0, (c/s).log(), -torch.inf)
        for c, s in zip((coverage, 1-coverage), mass)]
    denominator = np.zeros((1024, 1024), dtype=np.uint8)
    votes = {name: np.zeros((1024, 1024), dtype=np.uint8) for name in ('region', 'geometry', 'unfiltered')}
    local_masks, details, times = [], [], []
    previous_refiner = host.mask_refiner
    host.mask_refiner = 'bilinear'
    try:
        for i, box in enumerate(WINDOWS):
            x0, y0, x1, y1 = box
            image = canonical.crop(box)
            sampled = sample_window(full_map, box)
            adapter.bind(reference_input, reference_map, transform(image), sampled)
            t = time.perf_counter()
            prediction, _, _, _ = run_foris(host, reference, reference_mask, image)
            # Preserve the original bilinear Boolean readback and strict>.5.
            local = F.interpolate(prediction.float()[None, None], size=(512, 512),
                mode='bilinear', align_corners=False)[0, 0].numpy() > .5
            head_seconds = time.perf_counter()-t
            t = time.perf_counter()
            ids, n = label(local, np.ones((3, 3), dtype=bool))
            areas = np.bincount(ids.reshape(-1), minlength=n+1)[1:]
            overlap = np.bincount(ids.reshape(-1), weights=guide[y0:y1, x0:x1].reshape(-1), minlength=n+1)[1:]
            union = areas+int(guide[y0:y1, x0:x1].sum())-overlap
            agreement = (overlap/np.maximum(union, 1)).tolist()
            if n == 0:
                margins, accept = [], []
            elif missing_role:
                margins, accept = [None]*n, [True]*n
            else:
                # Derived features serve descriptors too; no native crop cache.
                raw = sampled.permute(1, 2, 0).reshape(4096, 1024).contiguous()
                pooled = region_descriptors(F.normalize(raw, dim=1), ids, n)
                logits = pooled@full.T/tau
                fg, bg = [tau*torch.logsumexp(logits+w[None], dim=1) for w in log_weights]
                margins = (fg-bg).tolist()
                accept = [m > 0 for m in margins]
            order = sorted(range(n), key=lambda j: (-agreement[j], j))
            selected = set(order[:sum(accept)])
            budget = [j in selected for j in range(n)]
            denominator[y0:y1, x0:x1] += 1
            for name, take in [('region', accept), ('geometry', budget)]:
                lookup = np.asarray([False]+take, dtype=bool)
                votes[name][y0:y1, x0:x1] += lookup[ids]
            votes['unfiltered'][y0:y1, x0:x1] += local
            local_masks.append(local)
            details.append(dict(window=i, box=box, components=n, component_sizes=areas.tolist(),
                context_margin=margins, global_agreement=agreement, accepted=accept, geometry_accepted=budget))
            times.append(dict(head_seconds=head_seconds, region_seconds=time.perf_counter()-t))
    finally:
        host.mask_refiner = previous_refiner
        adapter.expected = adapter.maps = None
    masks = {name: 2*vote > denominator for name, vote in votes.items()}
    return dict(masks=masks, local_masks=local_masks, regions=details, times=times,
        missing_query_role=missing_role, model_forward=0, frontend_calls=9, local_CRF_calls=0,
        native_window_raw_reads=0)
