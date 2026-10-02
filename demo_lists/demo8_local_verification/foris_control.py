"""Official FoRIS core + its paper CRF on the same 1200 old episodes.

Existing timm-converted DINOv3 weights adapter and reconstructed masks reused.
Author source/defaults are preserved; results are matched development controls,
not an exact reproduction of the published benchmark number. No downloads.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path[:0]=['/root/demo4_cache/env',str(ROOT/'foris_source'),'/root/autodl-tmp/demo4']
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.foris import FoRIS
from utils.refinement import init_crf,crf_refine,upsample_mask
from icx.common import TimmDINOv3,coco_episodes
from crf_runtime import load_crf


def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


@torch.inference_mode()
def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    status={'state':'WAITING_FOR_CRF_BUILD','pid':os.getpid(),'completed':0,'total':1200,
            'source_commit':'1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e',
            'crf_source_commit':'13a123f7cd3ea1f975e6c483b1e7c52112d7c151',
            'precision':'author source FP32 encoder and inference; TF32off',
            'weights':'existing timm DINOv3-L adapter; no model download',
            'dataset':'same reconstructed COCO masks and seed0 old development episodes',
            'resolution':args.resolution,'timing':'concurrent, not latency evidence'}
    save(out/'status.json',status)
    while True:
        p=ROOT/'results/crf_runtime.json'
        if p.exists():
            s=json.loads(p.read_text())
            if s['state']=='ERROR':raise RuntimeError(s['traceback'])
            if s['state']=='COMPLETED':break
        time.sleep(10)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    load_crf()
    model=FoRIS(TimmDINOv3().eval(),image_size=args.resolution,mask_refiner='bilinear',resize_to_orig_size=False)
    for p in model.parameters():p.requires_grad=False
    crf,band,core=init_crf(args.resolution,'cuda');records=[];start=time.monotonic()
    for fold in range(4):
        eps,_,base=coco_episodes(fold,300)
        for e,(c,q,refs) in enumerate(eps):
            dest=out/f'f{fold}_e{e:04d}.json'
            if dest.exists():records.append(json.loads(dest.read_text()));continue
            ri=Image.open(f'{base}/{refs[0]}').convert('RGB');qi=Image.open(f'{base}/{q}').convert('RGB')
            rm=torch.from_numpy((np.asarray(Image.open(f'{base}/annotations/{refs[0][:-4]}.png'))==c+1).copy())
            model.set_reference(ri,rm);model.set_target(qi)
            pred=model.predict(model._ref_images,model._ref_masks,model._tgt_image)
            refined=crf_refine(crf,band,core,model._tgt_image[None],pred)
            model._ref_images=None;model._ref_masks=None
            # Query annotation loaded only after both predictions are finalized.
            gt=torch.from_numpy((np.asarray(Image.open(f'{base}/annotations/{q[:-4]}.png'))==c+1).copy()).cuda()
            methods={}
            for name,mask in [(f'official_core{args.resolution}_fp32',pred),(f'official_crf{args.resolution}_fp32',refined)]:
                z=upsample_mask(mask,*gt.shape)
                methods[name]=[int((z&gt).sum()),int((z|gt).sum())]
            r={'fold':fold,'e':e,'c':c,'query':q,'reference':refs[0],'counts':methods}
            save(dest,r);records.append(r)
            status.update(state='RUNNING',completed=len(records),elapsed_seconds=time.monotonic()-start)
            save(out/'status.json',status)
            if len(records)%10==0:print(json.dumps(status),flush=True)
    result={}
    for name in records[0]['counts']:
        dd={}
        for r in records:
            a=dd.setdefault(r['c'],[0,0]);i,u=r['counts'][name];a[0]+=i;a[1]+=u
        result[name]={'class_miou':100*float(np.mean([i/max(u,1) for i,u in dd.values()]))}
    save(out/'report.json',{'state':'COMPLETED','metadata':status,'summary':result,'records':records})
    status.update(state='COMPLETED');save(out/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--resolution',type=int,default=512);args=ap.parse_args()
    try:run(args)
    except Exception:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json',{'state':'ERROR','traceback':traceback.format_exc()});raise
