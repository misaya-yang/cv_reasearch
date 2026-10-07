"""Lazy, source-bound observations shared by methods on the same episode.

Native final-LN packs and legacy processed MEAN remain distinct resources.
This provider never opens annotations, legacy packets, or query labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .common import ArtifactUnavailable, as_episode, continuous_original, readonly
from ics.cpu100.common import sha


class EpisodeProvider:
    def __init__(self, row, *, host=None, model_dir=None, threads=1):
        self.row = dict(row)
        self.host = host
        self.model_dir, self.threads = model_dir, threads
        self.cache = {}
        self.receipt = {'query_GT_read': False, 'assets': {}, 'extra_encoder_forwards': 0}

    def _bound(self, spec):
        p = Path(spec['path'])
        if not spec.get('sha256') or sha(p) != spec['sha256']:
            raise ValueError('Source-bound artifact missing or changed: ' + str(p))
        self.receipt['assets'][str(p)] = spec['sha256']
        return p

    def _raw(self):
        p = self._bound({'path': self.row['feature_pack'], 'sha256': self.row['sha256']})
        with np.load(p, allow_pickle=False) as z:
            q, r = z['q'].copy(), z['r'].copy()
            producer = json.loads(z['producer_json'].item())
        if producer.get('kind') != 'frozen_DINOv3_FP32_native_final_LN_patches':
            raise ArtifactUnavailable('Raw final-LN magnitude was not retained by this producer')
        self.cache.update(q_raw=readonly(q), r_raw=readonly(r),
                          q_norm=readonly(np.linalg.norm(q, axis=1)),
                          r_norm=readonly(np.linalg.norm(r, axis=1)))

    def _mean(self, ep):
        if not isinstance(self.host, dict) or self.host.get('kind') != 'source_bound_MEAN':
            raise ArtifactUnavailable('Actual MEAN host not bound for this episode')
        producer = self.host['producer']
        if producer.get('source_image_hashes') != ep.producer.get('source_image_hashes'):
            raise ValueError('MEAN and native features have different R/Q images')
        fpath = self._bound(self.host['field'])
        mpath = self._bound(self.host['mask'])
        with np.load(fpath, allow_pickle=False) as z:
            field = z[self.host['field']['key']].copy()
        with np.load(mpath, allow_pickle=False) as z:
            packed = z[self.host['mask']['key']].copy()
        # The recorded MEAN source retained complete binary1024 masks. Preserve
        # that host; the direct continuous renderer is a separate zero arm.
        work = np.unpackbits(packed, count=1024 * 1024).reshape(1024, 1024)
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(self.threads)
        with torch.inference_mode():
            mask = F.interpolate(torch.from_numpy(work.astype(np.float32))[None, None],
                                 tuple(ep.original_shape), mode='bilinear',
                                 align_corners=False)[0, 0].numpy() > .5
        self.cache.update(mean_field=readonly(field), p0=readonly(field), mask0=readonly(mask),
                          mean_original_field=readonly(continuous_original(ep, field)),
                          host_producer=producer, host_renderer=self.host['renderer'])

    def require(self, ep, name):
        aliases = {'raw_q': 'q_raw', 'raw_r': 'r_raw', 'raw_q_norm': 'q_norm',
                   'raw_r_norm': 'r_norm', 'query_raw_norm': 'q_norm',
                   'reference_raw_norm': 'r_norm'}
        name = aliases.get(name, name)
        if name in self.cache:
            return self.cache[name]
        if name in ('q_raw', 'r_raw', 'q_norm', 'r_norm'):
            self._raw()
        elif name in ('mean_field', 'p0', 'mask0', 'host_producer', 'host_renderer', 'mean_original_field'):
            self._mean(ep)
        elif name in ('encoder', 'encoder_binding', 'frozen_model'):
            if self.model_dir is None:
                raise ArtifactUnavailable('Actual frozen CPU encoder requires a model directory')
            from ics.cpu100.encoder import get_cpu_encoder
            enc = get_cpu_encoder(self.model_dir, ep.producer, threads=self.threads, max_cached_views=16)
            self.cache.update(encoder=enc, encoder_binding=enc.binding,
                              frozen_model=enc._real_forward._model)
        else:
            if self.model_dir is None:
                raise ArtifactUnavailable('Actual observation requires a bound frozen CPU model: ' + name)
            from .internal_encoder import get_internal_encoder
            engine = get_internal_encoder(self.model_dir, ep.producer, threads=self.threads)
            self.cache['encoder'] = engine.encoder
            self.receipt['internal_encoder'] = engine.receipt
            value = engine if name == 'internal_encoder' else engine.require(ep, name)
            self.cache[name] = value
        return self.cache[name]


def bind_episode(ep, row, *, host=None, model_dir=None, threads=1):
    return as_episode(ep, provider=EpisodeProvider(row, host=host, model_dir=model_dir, threads=threads))
