"""DINO raw-cache adapter for complete FoRIS and its shifted fine readout.

Recording keeps the encoder's original batch and FP32 outputs. Replay has no
encoder and raises on absent/corrupt branches. The same transformed image has
one key across reference, query and repeated episodes; masks are not keys.
"""
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from raw_feature_cache import RawFeatureCache, MissingBranches, file_hash


def dino_profile(assets, device):
    assets=Path(assets)
    repo=Path(__file__).resolve().parents[1]
    return dict(model='DINOv3-L/16',architecture='vit_large_patch16_dinov3',
        weights_sha256=file_hash(assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
        model_config_sha256=file_hash(assets/'demo4_cache/models/dinov3-vitl16-timm/config.json'),
        encoder_source_sha256=file_hash(repo/'src/ics/data.py'),
        observer_source_sha256=file_hash(repo/'src/ics/representations.py'),
        torch_version=str(torch.__version__),timm_version=importlib.metadata.version('timm'),
        producer_device=device,encoder_dtype='float32',storage_dtype='float32',
        preprocessing=dict(source_sha256=file_hash(assets/'third_party/foris_official/utils/data.py'),
            rgb=True,resize=[1024,1024],tensor='float32 CHW',
            mean=[.485,.456,.406],std=[.229,.224,.225]),
        features=dict(patch_grid=[64,64],channels=1024,prefix_tokens_removed=True,
            O='model.norm block output; no token L2 or positional projection',
            QK='actual QK norm output before RoPE; no token L2 or positional projection',
            V='actual value projection before attention mixing'))


class CachedDINO(torch.nn.Module):
    def __init__(self, cache, encoder=None, device='cpu', pair_branches=('O/24',)):
        super().__init__()
        self.cache,self.encoder,self.producer_device=cache,encoder,device
        self.pair_branches=tuple(dict.fromkeys(('O/24',)+tuple(pair_branches)))
        self.encoder_calls=0
        self.cache_reads=0
        self.provenance={}
        self.used_entries={}
        self.last_live_maps=None

    @torch.inference_mode()
    def raw(self, inputs, branches, provenance=None):
        from ics.representations import observe_branches
        branches=tuple(dict.fromkeys(('O/24',)+tuple(branches)))
        if inputs.dtype!=torch.float32 or inputs.ndim!=4 or tuple(inputs.shape[1:])!=(3,1024,1024):
            raise ValueError('Require the recorded FP32 1024px transformed inputs')
        arrays=inputs.detach().cpu().numpy()
        self.last_live_maps=None
        missing=False
        for x in arrays:
            path=self.cache.folder/self.cache.key(x)/'entry.json'
            if not path.exists():
                missing=True
                continue
            try:
                self.cache.read(x,branches)
            except MissingBranches:
                missing=True
        if missing:
            if self.encoder is None:
                raise KeyError('Raw cache coverage incomplete; replay has no encoder fallback')
            layers=sorted({int(key.split('/')[1]) for key in branches})
            kinds=''.join(sorted({key.split('/')[0] for key in branches}))
            with observe_branches(self.encoder,layers=layers,branches=kinds) as bank:
                self.last_live_maps=self.encoder.get_intermediate_layers(inputs.to(self.producer_device),n=1,reshape=True)[0].cpu()
            observed=bank['O/24'].reshape(len(inputs),64,64,1024).permute(0,3,1,2)
            if not torch.equal(observed,self.last_live_maps):
                raise ValueError('Raw O24 observer differs from the native encoder output')
            self.encoder_calls+=1
            for i,x in enumerate(arrays):
                self.cache.write(x,{key:bank[key][i].numpy() for key in branches},
                                 (provenance or self.provenance)|{'batch_index':i,'batch_size':len(inputs)})
        values=[]
        for x in arrays:
            raw=self.cache.read(x,branches)
            self.cache_reads+=1
            key=self.cache.key(x)
            self.used_entries[key]=str(self.cache.folder/key/'entry.json')
            values.append(raw)
        return {key:torch.from_numpy(np.stack([v[key] for v in values])) for key in branches}

    @torch.inference_mode()
    def get_intermediate_layers(self, inputs, n=1, reshape=True):
        if n!=1 or not reshape:
            raise ValueError('Cache adapter exposes the native final O24 map only; use raw for other branches')
        branches=self.pair_branches if len(inputs)==2 else ('O/24',)
        raw=self.raw(inputs,branches)['O/24']
        if self.last_live_maps is not None:
            return [self.last_live_maps]
        return [raw.reshape(len(inputs),64,64,1024).permute(0,3,1,2).contiguous()]


def cache_host(assets, adapter, mask_refiner='crf', *, prepared_crf=True):
    """Construct the original CPU FoRIS without constructing DINO for replay."""
    import sys
    from ics.native_basis import reuse_native_basis
    from ics.m4_crf import install
    assets=Path(assets)
    sys.path.insert(0,str(assets/'third_party/foris_official'))
    from models.foris import FoRIS
    if mask_refiner == 'crf':
        install(assets/'third_party/crf_source',assets/'runtime/macos/crf')
    with reuse_native_basis(FoRIS,assets/'native_assets/positional_basis.pt'):
        host=FoRIS(encoder=adapter,image_size=1024,svd_components=500,tau=.6,
                   mask_refiner=mask_refiner,resize_to_orig_size=False,device='cpu').eval().requires_grad_(False)
    if mask_refiner == 'crf' and prepared_crf:
        from ics.m4_crf_cached import build_backend, enable_on_crf
        backend = build_backend(assets/'third_party/crf_source', assets/'runtime/macos/crf/prepared_v1')
        enable_on_crf(host._crf, backend)
    # FoRIS moves all registered children to its host device during init.
    if adapter.encoder is not None:
        adapter.encoder.to(adapter.producer_device)
    return host
