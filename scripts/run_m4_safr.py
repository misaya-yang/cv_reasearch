#!/usr/bin/env python3
"""Fixed SAFR guide composition vs the sealed complete M4 baselines.

Primary: SAFR foreground-minus-background replaces the MEAN guide.
Controls: SAFR foreground-only and raw final-layer foreground-minus-background.
Same FoRIS score, post-Part1 graph, constants, full renderer and paired inputs.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
spec=importlib.util.spec_from_file_location('m4_base',REPO/'scripts/run_m4_baselines.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.experiment import sha
    from ics.methods import rcg
    from ics.safr import capture,fuse,selector,solve_guide

    parent=json.loads((a.base/'sealed.json').read_text());config=json.loads((a.base/'config.json').read_text())
    if parent['state']!='ALL_PREDICTIONS_SEALED' or sha(a.base/'manifest.json')!=parent['manifest_sha256'] or sha(a.base/'config.json')!=parent['config_sha256']:
        raise ValueError('Baseline is not sealed')
    if a.out.exists():raise FileExistsError('Fresh output required')
    rows=json.loads((a.base/'manifest.json').read_text())
    assets=Path(config['assets']);torch.set_num_threads(config['threads']);torch.manual_seed(0)
    model_dir=assets/'demo4_cache/models/dinov3-vitl16-timm'
    if sha(model_dir/'model.safetensors')!=config['weights_sha256']:raise ValueError('Changed backbone weights')
    net=TimmDINOv3(str(model_dir)).to(config['encoder_device']).eval().requires_grad_(False)
    select,source=selector(assets/'third_party/RSRM/model/rsrm.py')
    config=dict(config,candidate='SAFR FG-BG guide, fixed alpha .25 and original MEAN graph',
        primary='mean.safr_fg_bg',controls=['mean.safr_fg','mean.raw_fg_bg'],rsrm_source=source,
        parent_seal_sha256=sha(a.base/'sealed.json'),
        selection='RSRM reference-only original methods; explicit MEAN fallback when undefined',
        added_encoder_forwards='one paired forward in this composition experiment; same-backbone observational hooks',
        source_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'src/ics/safr.py']})
    a.out.mkdir(parents=True);(a.out/'predictions').mkdir();(a.out/'features').mkdir()
    base.write(a.out/'manifest.json',rows);base.write(a.out/'config.json',config)
    transform=transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    hashes,inputs,timings={},{},{};began=time.monotonic()
    with torch.inference_mode():
        for n,row in enumerate(rows,1):
            ident=row['pack']+'__'+row['key'];started=time.monotonic();paths=base.paths(assets,row)
            for k,receipt in parent['inputs'][ident].items():
                if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Changed image/reference mask: '+k)
            packet=a.base/'predictions'/(ident+'.npz');feature=a.base/'features'/(ident+'.npz')
            if sha(packet)!=parent['hashes'][ident]['predictions'] or sha(feature)!=parent['hashes'][ident]['features']:
                raise ValueError('Changed baseline outputs')
            with np.load(feature,allow_pickle=False) as z:
                q,r,cov,score=(z[k].copy() for k in ('q','r','cov','score'));raw_q,raw_r=(z[k].copy() for k in ('q_raw','r_raw'));old_mean=z['mean'].copy()
            with np.load(packet,allow_pickle=False) as z:predictions={k:z[k].copy() for k in z.files};hw=tuple(predictions['original_hw'])
            with Image.open(paths['support_mask']) as im:mask=torch.from_numpy((np.asarray(im.convert('L'))>0).copy())
            mask=F.interpolate(mask[None,None].float(),(1024,1024),mode='nearest')[0,0]
            imgs=[]
            for k in ('support','query'):
                with Image.open(paths[k]) as im:imgs.append(transform(im.convert('RGB')))
            x=torch.stack(imgs).to(config['encoder_device']);encode_start=time.monotonic()
            with capture(net.m) as branches:
                last=net.get_intermediate_layers(x,n=1,reshape=True)[0].cpu()
            if config['encoder_device']=='mps':torch.mps.synchronize()
            encode_seconds=time.monotonic()-encode_start
            q_check=F.normalize(last[1].flatten(1).T,dim=1).numpy()
            reproduction_error=float(np.abs(q_check-raw_q).max())
            if reproduction_error>1e-5:raise ValueError('Paired encoder output drift: '+str(reproduction_error))
            guides,info=fuse(branches,last[0],mask,select)
            del branches,last,x
            raw_q,raw_r=(F.normalize(torch.from_numpy(v),dim=1) for v in (raw_q,raw_r))
            fi=torch.from_numpy(cov.ravel()>=.5)
            raw_guide=(raw_q@(F.normalize(raw_r[fi].mean(0),dim=0)-F.normalize(raw_r[~fi].mean(0),dim=0))).numpy()
            names={'mean.raw_fg_bg':raw_guide}
            if guides is not None:names.update({'mean.safr_fg':guides['fg'],'mean.safr_fg_bg':guides['fg_bg']})
            fields={};solvers={}
            for arm in ('mean.safr_fg_bg','mean.safr_fg','mean.raw_fg_bg'):
                if arm in names:
                    field,solvers[arm]=solve_guide(q,cov,score,names[arm])
                else:field=old_mean.copy();solvers[arm]={'fallback':'exact stored MEAN field'}
                fields[arm]=field
                predictions[arm]=np.packbits(base.render(rcg.mask_from_field(field),hw))
            dest=a.out/'predictions'/(ident+'.npz');np.savez_compressed(dest,**predictions)
            fp=a.out/'features'/(ident+'.npz');np.savez_compressed(fp,**fields)
            hashes[ident]=dict(predictions=sha(dest),features=sha(fp));inputs[ident]=parent['inputs'][ident]
            timings[ident]=dict(seconds=time.monotonic()-started,extra_paired_encoder_seconds=encode_seconds,
                reproduction_max_error=reproduction_error,safr=info,solvers=solvers)
            base.write(a.out/'progress.json',dict(n=n,total=len(rows),seconds=time.monotonic()-began,last=ident))
            print(json.dumps(dict(n=n,total=len(rows),key=ident,seconds=timings[ident]['seconds'],safr=info)),flush=True)
    base.write(a.out/'timings.json',timings)
    base.write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',
        manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),
        hashes=hashes,inputs=inputs,query_labels_opened=False,seconds=time.monotonic()-began))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['infer','score']);p.add_argument('--base',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.mode=='infer':
        if a.base is None:p.error('--base required for inference')
        infer(a)
    else:base.score(a)


if __name__=='__main__':main()
