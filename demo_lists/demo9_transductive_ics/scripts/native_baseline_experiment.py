#!/usr/bin/env python3
"""E8 real source baseline + source-faithful small shared basis artifact.

One host per process prevents `models`/`utils` namespace collisions. Uses only
the already installed encoder, existing RGB/official masks and original host.
No downloaded assets. Query annotations enter only after native prediction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from tics.native_assets import reuse_native_basis
from global_representation_probe import fixed_selection,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    p.add_argument('--foris-root',default='/root/autodl-tmp/demo8_local_verification/foris_source')
    p.add_argument('--prepared-manifest');p.add_argument('--out')
    p.add_argument('--host',choices=['insid3','foris'],default='insid3')
    p.add_argument('--episode-mode',choices=['standard','explicit_metric_cohort'],default='standard')
    p.add_argument('--refiner',choices=['bilinear','crf'],default='bilinear')
    p.add_argument('--projection-basis');p.add_argument('--save-basis')
    p.add_argument('--query-brightness',type=float,choices=[1.,.75],default=1.)
    p.add_argument('--limit',type=int,default=10);p.add_argument('--self-check',action='store_true')
    a=p.parse_args()
    if a.self_check:
        assert fixed_selection([(4,'q',['s']),(0,'a',['b'])],2)[0][1][0]==0
        assert summary([dict(c=0,iu={'native':[2,4]})])=={'native':50.}
        rgb=np.array([0,128,255],dtype=np.uint8)
        assert np.rint(rgb.astype(np.float32)*.75).astype(np.uint8).tolist()==[0,96,191]
        print(json.dumps(dict(state='CPU_BASELINE_INTERFACE_PASSED',no_model_or_GPU=True)));return
    if not a.out or not a.prepared_manifest:p.error('--out and --prepared-manifest required')
    if os.environ.get('DEMO9_CUDA_GUARD')!='1':raise RuntimeError('Only the prepared resource guard may launch native baseline')
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(a.prepared_manifest).read_text())
    if manifest.get('state')!='PREPARED_ASSETS':raise ValueError('CPU assets must already be prepared')
    for asset in manifest['assets']:
        actual=Path(asset['path']).stat()
        if actual.st_size!=asset['size'] or actual.st_mtime_ns!=asset['mtime_ns']:raise RuntimeError('Prepared asset drift')
    report=dict(state='PREPARING',host=a.host,refiner=a.refiner,records=[],args=vars(a),
        input_contract='Same existing DINO weights; native released host; official masks; no queried labels before frozen prediction',
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def save():
        report['class_miou']=summary(report['records'])
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(out/'report.json')
    save();start=time.monotonic()
    try:
        if not torch.cuda.is_available():raise RuntimeError('No CUDA; no model was loaded')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
        if a.host=='foris':
            sys.path.insert(0,a.foris_root)
            from models.foris import FoRIS as Host
            from utils.data import build_transform
        else:
            sys.path.insert(0,str(Path(_paths.DEMO4)/'INSID3'))
            from models.insid3 import INSID3 as Host
            from utils.data import build_transform
        sys.path.insert(0,_paths.DEMO4)
        from icx.common import TimmDINOv3,coco_episodes
        if a.episode_mode=='standard':
            rows,_,base=coco_episodes(manifest['fold'],400,shot=1,seed=manifest['seed'])
            selected=fixed_selection(rows,a.limit)
        else:
            if manifest.get('episode_protocol')!='frozen_metric_inference_cohort':raise ValueError('Explicit held-out metric cohort required')
            base=manifest['data_root']
            selected=[(r['e'],(r['c'],r['query'],[r['support']])) for r in manifest['frozen_episodes']]
            if len(selected)!=a.limit:raise ValueError('Explicit cohort count mismatch')
        expected=[dict(e=e,c=c,support=s[0],query=q) for e,(c,q,s) in selected]
        if expected!=manifest['frozen_episodes']:raise RuntimeError('Frozen episode identity mismatch')
        torch.manual_seed(0)
        with torch.no_grad():
            encoder=TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(Host,a.projection_basis) as basis_receipt:
                model=Host(encoder=encoder,image_size=1024,svd_components=500,tau=.6,
                           mask_refiner=a.refiner,resize_to_orig_size=False,device='cuda').eval().requires_grad_(False)
        report['projection_basis']=basis_receipt
        if a.save_basis:
            dest=Path(a.save_basis);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise RuntimeError('Do not overwrite a frozen basis')
            torch.save(dict(state='NATIVE_BASIS_FROZEN',source_input='normalized_black_image',
                source='original_INSID3_normalized_torch_zeros_black_image_FP32_positional_basis',
                basis=model.positional_basis.detach().float().cpu(),rank=500,dimension=1024),dest)
            report['saved_basis']=dict(path=str(dest),source_input='normalized_black_image')
        report['state']='RUNNING';save()
        transform=build_transform(1024)
        with torch.inference_mode():
            for e,(c,q,refs) in selected:
                si=Image.open(Path(base)/refs[0]).convert('RGB');qi=Image.open(Path(base)/q).convert('RGB')
                if a.query_brightness!=1.:
                    qi=Image.fromarray(np.rint(np.asarray(qi,dtype=np.float32)*a.query_brightness).clip(0,255).astype(np.uint8))
                sm=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/Path(refs[0]).with_suffix('.png')))==c+1).copy())
                s=transform(si).cuda()[None];t=transform(qi).cuda()[None]
                mask=F.interpolate(sm[None,None].float().cuda(),(1024,1024),mode='nearest')[0]
                began=time.monotonic()
                if mask.any():
                    pred=model.predict_mask(s,mask,t) if a.host=='insid3' else model.predict(s,mask,t[0])
                    pred=pred.reshape(1024,1024).bool().detach().clone()
                else:pred=torch.zeros(1024,1024,dtype=torch.bool,device='cuda')
                torch.cuda.synchronize();elapsed=time.monotonic()-began
                # This is the only query-annotation access, after prediction freezes.
                gt=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/Path(q).with_suffix('.png')))==c+1).copy()).cuda()
                native=F.interpolate(gt[None,None].float(),(1024,1024),mode='nearest')[0,0].bool()
                original=F.interpolate(pred[None,None].float(),gt.shape,mode='bilinear',align_corners=False)[0,0]>.5
                name=f'{a.host}_{a.refiner}'
                if a.query_brightness!=1.:name+='_brightness075'
                row=dict(e=e,c=c,support=refs[0],query=q,allrole_photo_ids=[refs[0],q],
                    iu={name:[int((pred&native).sum()),int((pred|native).sum())]},
                    original_iu={name:[int((original&gt).sum()),int((original|gt).sum())]},
                    armstate={name:'COMPLETED'},native_query_truth_opened_after_prediction=True,
                    cost=dict(request_s=elapsed))
                report['records'].append(row);report['elapsed_s']=time.monotonic()-start
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved();save()
                print(json.dumps(dict(count=len(report['records']),scores=report['class_miou'],elapsed_s=report['elapsed_s'])),flush=True)
                del s,t,mask,pred,gt,native,original
        report['state']='COMPLETED';save()
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc),elapsed_s=time.monotonic()-start);save();raise


if __name__=='__main__':main()
