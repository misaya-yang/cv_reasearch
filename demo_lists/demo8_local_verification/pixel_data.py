"""Legal dense supervision for the existing, frozen candidate observations.

Old development images only. Query masks produce labels/oracle diagnostics after
all observations were extracted; no test mask enters the inference interface.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4']
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from icx.offline2 import leaves_of
from extract import box,mask_up,resize_crop,save


def run(root):
    root=Path(root);out=root/'pixel_data_v1';out.mkdir(exist_ok=True)
    files=sorted((root/'dense_observation_v1').glob('f*_e*.npz'));assert len(files)==1200
    torch.set_num_threads(4)
    targets=[];weights=[];masks=[];boxes=[];shapes=[];meta=[];oracle=[]
    cache=Path('/root/demo8_dependency_pins')
    for fold in range(4):
        trees=torch.load(cache/f'results/l3b_f{fold}.l3.pt',map_location='cpu',weights_only=False)
        for f in files:
            with np.load(f) as d:
                if int(d['fold'])!=fold:continue
                r={k:d[k].item() for k in ['fold','e','c','query','reference']}
                t=trees[r['e']];assert t['c']==r['c']
                top=d['top'];geom=d['maps'][0,:,6].astype(np.float32)
            cm=[t['insid3'].reshape(64,64)]
            for node in top:
                a=np.zeros(4096,bool);a[leaves_of(t['children'],4096,int(node))]=True;cm.append(a.reshape(64,64))
            cm=torch.from_numpy(np.stack(cm));cc=torch.stack([mask_up(x,(1024,1024)) for x in cm])
            bb=np.array([box(x) for x in cc],np.int32)
            gt=torch.from_numpy((np.asarray(Image.open(cache/f"data/COCO2014/annotations/{r['query'][:-4]}.png"))==r['c']+1).copy())
            gs=F.interpolate(gt[None,None].float(),(1024,1024),mode='nearest')[0,0]
            ys=[];ww=[]
            for j,b in enumerate(bb):
                w=F.adaptive_avg_pool2d(resize_crop(cc[j][None,None].float(),b),(28,28))[0,0]
                fg=F.adaptive_avg_pool2d(resize_crop((gs*cc[j])[None,None],b),(28,28))[0,0]
                ys.append((fg/w.clamp_min(1e-6)).clamp(0,1));ww.append(w)
            w=torch.stack(ww).numpy();y=torch.stack(ys).numpy()
            assert np.max(np.abs(w-geom))<.002,'ROI contract differs from extracted observation'
            targets.append(y.astype(np.float16));weights.append(w.astype(np.float16))
            masks.append(np.packbits(cm.numpy(),axis=None));boxes.append(bb);shapes.append(list(gt.shape));meta.append(r)
            native=mask_up(cm[0],gt.shape);coverage=torch.zeros_like(gt)
            for x in cm:coverage|=mask_up(x,gt.shape)
            # Explicit pixel oracle, never a deployment result.
            pure=coverage&gt;aug=native|pure
            oracle.append({'c':r['c'],'pure_pixel_oracle':[int((pure&gt).sum()),int((pure|gt).sum())],
                           'add_pixel_oracle':[int((aug&gt).sum()),int((aug|gt).sum())]})
            if len(meta)%100==0:print('labels',len(meta),flush=True)
        del trees
    np.savez_compressed(out/'labels.npz',targets=np.stack(targets),weights=np.stack(weights),masks=np.stack(masks),
                        boxes=np.stack(boxes),shapes=np.array(shapes))
    metrics={}
    for key in ['pure_pixel_oracle','add_pixel_oracle']:
        sums={}
        for r in oracle:
            z=sums.setdefault(r['c'],np.zeros(2,np.int64));z+=r[key]
        metrics[key]=100*float(np.mean([i/max(u,1) for i,u in sums.values()]))
    save(out/'metadata.json',{'state':'COMPLETED','n':len(meta),'rows':meta,
                             'labels':'foreground fraction CONDITIONAL on candidate membership in each 28x28 ROI token',
                             'oracle_only':metrics,'gt_role':'train/validation labels or offline scoring only; not inference input'})
    print(json.dumps({'state':'COMPLETED','n':len(meta),'oracle_only':metrics}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();run(Path(args.root))
