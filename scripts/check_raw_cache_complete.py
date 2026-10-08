#!/usr/bin/env python3
"""Five fixed development probes: complete baselines replay without encoding.

Only the first frozen draw of each dataset is used. Query annotations are never
opened. This implementation check neither scores a candidate nor runs a pool.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from ics.data import TimmDINOv3
from ics.foris import run_foris
from ics.official_data import load_inputs
from ics.methods import rcg,mean_control
from ics import fine_readout
from cached_dino import CachedDINO,cache_host,dino_profile
from raw_feature_cache import RawFeatureCache,file_hash
from frozen_dino_plan import locked_run,write
from run_m4_baselines import render


def complete(host,row,assets):
    rgb,mask,query=load_inputs(dict(row,query_mask_path='/query-annotation-forbidden'),assets)
    native,got,transformed,target=run_foris(host,rgb,mask,query)
    processed=F.normalize(got['deb'][0].float(),dim=1)
    q,r=(processed[i].flatten(1).T.half().contiguous() for i in (1,0))
    cov=F.interpolate(transformed[None,None].float(),(64,64),mode='area')[0,0].numpy()
    score=got['score'].float().numpy()
    z,_=rcg.predict(q,r,cov,score)
    mf,_=mean_control.predict(q,r,cov,score)
    debiased=bool((F.normalize(got['raw'][0].float(),dim=1)-got['deb'][0]).abs().max()>1e-4)
    fine=fine_readout.field(fine_readout.shifted_features(host,target,debiased),q,z)
    fields=dict(score=score,rcg=z,mean=mf,rcg_fine=fine)
    masks={'foris.crf':native.numpy(),'rcg':rcg.mask_from_field(z),
           'mean':rcg.mask_from_field(mf),'rcg.fine':fine_readout.mask(fine)}
    shape=(query.height,query.width)
    masks={kind+'/'+key:m if kind=='cli' else render(m,shape)
           for kind in ('original','cli') for key,m in masks.items()}
    return fields,masks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=REPO.parent/'cv_data')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    a=p.parse_args()
    prepared=json.loads((a.manifest.parent/'prepared.json').read_text())
    if prepared.get('seed')!=1 or prepared.get('split_role')!='dev':
        raise ValueError('Probe only uses fresh development seed1')
    first={}
    for row in json.loads(a.manifest.read_text()):first.setdefault(row['dataset'],row)
    if set(first)!=set(('coco','lvis','pascal_part','paco_part','suim')):
        raise ValueError('Require the five development datasets')
    torch.set_num_threads(2)
    with locked_run(a.out):
        if (a.out/'report.json').exists():
            raise FileExistsError('Completed probe is immutable')
        rows=list(first.values())
        profile=dino_profile(a.assets,a.device)
        write(a.out/'config.json',dict(role='five fixed implementation probes, not gain evidence',
              manifest_sha256=file_hash(a.manifest),profile=profile,
              source_sha256={str(x):file_hash(x) for x in (Path(__file__),REPO/'scripts/cached_dino.py',REPO/'scripts/raw_feature_cache.py')},
              commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        write(a.out/'manifest.json',rows)
        cache=RawFeatureCache(a.out/'cache',profile)
        encoder=TimmDINOv3(a.assets/'demo4_cache/models/dinov3-vitl16-timm').to(a.device).eval().requires_grad_(False)
        recording=CachedDINO(cache,encoder,a.device,pair_branches=('Q/16','K/16','Q/24','K/24'))
        host=cache_host(a.assets,recording)
        expected={};timings={}
        with torch.inference_mode():
            for row in rows:
                began=time.monotonic()
                recording.provenance=dict(episode_id=row['episode_id'],
                    reference_rgb_hash=row['reference_rgb_hash'],query_rgb_hash=row['query_rgb_hash'],
                    reference_crop=row.get('reference_crop'),query_crop=row.get('query_crop'))
                expected[row['dataset']]=complete(host,row,a.assets)
                timings[row['dataset']]=dict(record_seconds=time.monotonic()-began)
                print(json.dumps(dict(dataset=row['dataset'],phase='record',encoder_calls=recording.encoder_calls)),flush=True)
        count=recording.encoder_calls;entries=recording.used_entries.copy()
        # Destroy the actual encoder before the independent CPU replay phase.
        del host,recording,encoder
        if a.device=='mps':torch.mps.empty_cache()
        replay=CachedDINO(cache)
        host=cache_host(a.assets,replay)
        report=dict(state='PASSED',profile_id=cache.profile_id,query_labels_opened=False,
                    encoder_record_calls=count,replay_encoder_calls=0,datasets={},
                    scope='five fixed development probes; complete baseline readback, not full coverage or candidate gain')
        (a.out/'predictions').mkdir(exist_ok=True)
        with torch.inference_mode():
            for row in rows:
                name=row['dataset'];began=time.monotonic()
                actual=complete(host,row,a.assets)
                timings[name]['cpu_replay_seconds']=time.monotonic()-began
                fields,masks=expected[name]
                exact_fields={k:np.array_equal(v,actual[0][k]) for k,v in fields.items()}
                exact_masks={k:np.array_equal(v,actual[1][k]) for k,v in masks.items()}
                assert all(exact_fields.values()) and all(exact_masks.values()),(name,exact_fields,exact_masks)
                payload={phase+'/field/'+k:v for phase,pair in (('live',expected[name]),('replay',actual)) for k,v in pair[0].items()}
                payload.update({phase+'/mask/'+k:np.packbits(v) for phase,pair in (('live',expected[name]),('replay',actual)) for k,v in pair[1].items()})
                destination=a.out/'predictions'/(name+'.npz');np.savez_compressed(destination,**payload)
                report['datasets'][name]=dict(episode_id=row['episode_id'],fields_exact=exact_fields,
                                              masks_exact=exact_masks,prediction_sha256=file_hash(destination),**timings[name])
                print(json.dumps(dict(dataset=name,phase='cpu_replay',all_exact=True)),flush=True)
        assert replay.encoder is None and replay.encoder_calls==0
        report['entries']=entries
        report['n_unique_transformed_inputs']=len(entries)
        report['payload_bytes']=sum((Path(p).parent/json.loads(Path(p).read_text())['file']).stat().st_size for p in entries.values())
        write(a.out/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('datasets','entries')}),flush=True)


if __name__=='__main__':main()
