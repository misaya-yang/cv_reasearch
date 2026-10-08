#!/usr/bin/env python3
"""One fixed real R/Q pair: raw vectors, exact readback and CPU FoRIS replay."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from ics.data import TimmDINOv3
from ics.native_basis import reuse_native_basis
from ics.official_data import array_hash, file_hash, load_inputs
from ics.representations import observe_branches
from ics.m4_crf import install
from raw_feature_cache import RawFeatureCache, tensor_hash


class EncoderForbidden(torch.nn.Module):
    def get_intermediate_layers(self,*args,**kwargs):
        raise AssertionError('CPU replay attempted DINO encoding')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=REPO.parent/'cv_data')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'report.json').exists():
        print('Existing fixed raw-cache probe retained');return
    row=json.loads(a.manifest.read_text())[0]
    image,mask,query=load_inputs(dict(row,query_mask_path='/query-label-must-not-be-opened.png'),a.assets)
    sys.path.insert(0,str(a.assets/'third_party/foris_official'))
    from utils.data import build_transform
    from models.foris import FoRIS
    transform=build_transform(1024)
    inputs=torch.stack([transform(image),transform(query)])
    profile=dict(model='DINOv3-L/16',architecture='vit_large_patch16_dinov3',
        weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
        torch_version=str(torch.__version__),timm_version=importlib.metadata.version('timm'),
        producer_device=a.device,encoder_dtype='float32',storage_dtype='float32',
        preprocessing={'source_sha256':file_hash(a.assets/'third_party/foris_official/utils/data.py'),
                       'rgb':True,'resize':[1024,1024],'tensor':'float32 CHW',
                       'mean':[.485,.456,.406],'std':[.229,.224,.225]},
        features={'patch_grid':[64,64],'channels':1024,'prefix_tokens_removed':True,
                  'O':'model.norm block output; no token L2 or positional projection',
                  'QK':'actual QK norm output before RoPE; no token L2 or positional projection'})
    cache=RawFeatureCache(a.out/'cache',profile)
    encoder=TimmDINOv3(a.assets/'demo4_cache/models/dinov3-vitl16-timm').to(a.device).eval().requires_grad_(False)
    began=time.monotonic()
    with torch.inference_mode(),observe_branches(encoder,layers=(16,24),branches='OQK') as bank:
        live=encoder.get_intermediate_layers(inputs.to(a.device),n=1,reshape=True)[0].cpu()
    encoding_seconds=time.monotonic()-began
    keys=('O/24','Q/16','K/16','Q/24','K/24')
    receipts=[]
    for index,role in enumerate(('reference','query')):
        raw={key:bank[key][index].numpy() for key in keys}
        key,receipt=cache.write(inputs[index].numpy(),raw,
            dict(role=role,episode_id=row['episode_id'],source_rgb_hash=row[role+'_rgb_hash'],
                 source_path=row[role+'_path'],source_crop=row.get(role+'_crop')))
        read=cache.read(inputs[index].numpy(),keys)
        assert all(np.array_equal(read[k],raw[k]) for k in keys)
        assert np.array_equal(read['O/24'].reshape(64,64,1024).transpose(2,0,1),live[index].numpy())
        receipts.append(dict(role=role,input_key=key,entry=str(cache.folder/key/'entry.json'),
                             file_bytes=(cache.folder/key/receipt['file']).stat().st_size,
                             branches=keys,shapes={k:list(raw[k].shape) for k in keys}))
    del encoder,bank
    if a.device=='mps':torch.mps.empty_cache()
    install(a.assets/'third_party/crf_source',a.assets/'runtime/macos/crf')
    with reuse_native_basis(FoRIS,a.assets/'native_assets/positional_basis.pt'):
        host=FoRIS(encoder=EncoderForbidden(),image_size=1024,svd_components=500,tau=.6,
                   mask_refiner='crf',resize_to_orig_size=False,device='cpu').eval().requires_grad_(False)
    hashes=[tensor_hash(x.numpy()) for x in inputs]
    calls={'cached':0,'live':0}
    def from_live(imgs):
        flat=imgs.reshape(-1,*imgs.shape[-3:])
        assert [tensor_hash(x.numpy()) for x in flat]==hashes
        calls['live']+=1;return live.reshape(1,2,1024,64,64)
    def from_cache(imgs):
        shape=imgs.shape;maps=[]
        for x in imgs.reshape(-1,*shape[-3:]):
            raw=cache.read(x.numpy(),['O/24'])['O/24']
            maps.append(torch.from_numpy(raw.reshape(64,64,1024).transpose(2,0,1).copy()))
        calls['cached']+=1;return torch.stack(maps).reshape(shape[0],shape[1],1024,64,64)
    with torch.inference_mode():
        host._extract_features=from_live;host.set_reference(image,mask);host.set_target(query);first=host.segment().clone()
        host._extract_features=from_cache;host.set_reference(image,mask);host.set_target(query);replayed=host.segment().clone()
    assert torch.equal(first,replayed)
    report=dict(state='PASSED',episode_id=row['episode_id'],producer_device=a.device,
        profile_id=cache.profile_id,profile=profile,entries=receipts,raw_array_readback_exact=True,
        CPU_replay='complete FoRIS/CRF from stored O24, encoder raises if called',
        cpu_final_mask_exact=True,replay_encoder_calls=0,extract_calls=calls,query_labels_opened=False,
        encoding_seconds=encoding_seconds,scope='one fixed real implementation probe, not full cache or candidate gain',
        cached_coverage=list(keys),not_cached=['all other layer branches','four shifted query O24 features'])
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
