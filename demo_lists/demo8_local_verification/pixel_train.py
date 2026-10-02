"""Can a support-conditioned pixel verifier unmix fixed candidate regions?

Equal pixel supervision, approximately matched capacity, shared fold/image
splits. Global/query/paired and joint observations; pointwise MLP is a strong
same-input control for spatial CNN. No proposal-ranking or budget controller.
"""
import argparse
import json
from pathlib import Path
import os
import sys
import time
import traceback

sys.path.insert(0,'/root/demo4_cache/env')
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from train import split_rows,save


class PixelHead(nn.Module):
    def __init__(self,nc,spatial):
        super().__init__()
        if spatial:
            h=16 if nc==8 else 14
            self.net=nn.Sequential(nn.Conv2d(nc,h,3,padding=1),nn.GELU(),nn.Conv2d(h,h,3,padding=1),nn.GELU(),nn.Conv2d(h,1,1))
        else:
            h=61 if nc==8 else 55
            self.net=nn.Sequential(nn.Conv2d(nc,h,1),nn.GELU(),nn.Conv2d(h,48,1),nn.GELU(),nn.Conv2d(48,1,1))
    def forward(self,x):
        n,k,c,h,w=x.shape
        return self.net(x.reshape(n*k,c,h,w)).reshape(n,k,h,w)


def render(pred,masks,boxes,shape,gt):
    """Warp probabilities to original pixels; restrict each to its fixed mask.

    RGB was square-resized to1024, cropped by these boxes then resized448.
    Coordinate mapping below preserves that exact align_corners=False chain.
    """
    h,w=shape
    xs=(torch.arange(w,device='cuda')+.5)*1024/w
    ys=(torch.arange(h,device='cuda')+.5)*1024/h
    b=torch.as_tensor(boxes,device='cuda',dtype=torch.float32)
    gy=2*(ys[None,:,None]-b[:,0,None,None])/(b[:,1]-b[:,0])[:,None,None]-1
    gx=2*(xs[None,None,:]-b[:,2,None,None])/(b[:,3]-b[:,2])[:,None,None]-1
    grid=torch.stack([gx.expand(-1,h,-1),gy.expand(-1,-1,w)],-1)
    probs=F.grid_sample(pred[:,None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[:,0]
    cm=F.interpolate(torch.as_tensor(masks,device='cuda',dtype=torch.float32)[:,None],(h,w),mode='bilinear',align_corners=False)[:,0]>.5
    cover=cm.sum(0);mean=(probs*cm).sum(0)/cover.clamp_min(1)
    maximum=torch.where(cm,probs,0.).max(0).values
    native=cm[0];gt=torch.as_tensor(gt,device='cuda',dtype=torch.bool)
    results={}
    for name,p in [('mean',mean),('max',maximum)]:
        z=(p>.5)&(cover>0)
        for action,mask in [('replace',z),('add',z|native)]:
            i=int((mask&gt).sum());u=int((mask|gt).sum())
            add=mask&~native
            results[f'{action}_{name}']=[i,u,int((add&gt).sum()),int((add&~gt).sum())]
    return results


def summarize(rows):
    out={}
    for name in rows[0]['counts']:
        sums={}
        for r in rows:
            sums.setdefault(r['c'],np.zeros(4,np.int64))[:] += r['counts'][name]
        out[name]={'class_miou':100*float(np.mean([v[0]/max(v[1],1) for v in sums.values()])),
                   'true_foreground_added':int(sum(v[2] for v in sums.values())),
                   'false_foreground_added':int(sum(v[3] for v in sums.values()))}
    return out


def run(args):
    root=Path(args.root);out=root/'pixel_verifier_v1';out.mkdir(exist_ok=True)
    status={'state':'WAITING_FOR_LABELS','pid':os.getpid(),'completed_models':0,'total_models':80,
            'task':'direct foreground pixels inside frozen candidates, not another quality ranker',
            'split':'same class/image-disjoint development folds; future800 remains unopened',
            'threshold':.5,'fusion':['mean','max'],'precision':'FP32 head, TF32off; fixed existing encoder maps',
            'epoch_selection':'original-resolution validation class-mIoU averaged over four predefined fusion/action outputs'}
    save(out/'status.json',status)
    while not (root/'pixel_data_v1/metadata.json').exists():time.sleep(10)
    info=json.loads((root/'pixel_data_v1/metadata.json').read_text());meta=info['rows'];assert len(meta)==1200
    with np.load(root/'pixel_data_v1/labels.npz') as d:
        labels=d['targets'].astype(np.float32);weights=d['weights'].astype(np.float32)
        masks=np.unpackbits(d['masks'],axis=1).reshape(1200,13,64,64).astype(bool)
        boxes=d['boxes'];shapes=d['shapes']
    maps=[];gts=[]
    for r in meta:
        with np.load(root/f"dense_observation_v1/f{r['fold']}_e{r['e']:04d}.npz") as d:maps.append(d['maps'])
        gt=np.asarray(Image.open(f"/root/demo8_dependency_pins/data/COCO2014/annotations/{r['query'][:-4]}.png"))==r['c']+1
        assert list(gt.shape)==shapes[len(gts)].tolist();gts.append(gt)
    maps=np.stack(maps);classes=np.array([r['c'] for r in meta])
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    yg=torch.from_numpy(labels).cuda();wg=torch.from_numpy(weights).cuda()
    views=['global','query_crop','paired_crop','global_query_joint','global_paired_joint']
    results={};start=time.monotonic()
    for fold in range(4):
        tr,val,test=split_rows(meta,fold)
        _,nc=np.unique(classes[tr],return_counts=True)
        class_counts={int(c):int((classes[tr]==c).sum()) for c in np.unique(classes[tr])}
        ew=np.array([1/class_counts[int(c)] for c in classes[tr]],np.float32)
        for vi,view in enumerate(views):
            if vi<3:x=maps[:,vi].astype(np.float32)
            else:x=np.concatenate([maps[:,0,:,:6],maps[:,vi-2,:,:6],maps[:,0,:,6:]],axis=2).astype(np.float32)
            xg=torch.from_numpy(x).cuda();del x
            for spatial,arch in [(False,'point_mlp'),(True,'spatial_cnn')]:
                for seed in [2037,2038]:
                    tag=f'{view}_{arch}_seed{seed}';receipt=out/f'f{fold}_{tag}.json'
                    if receipt.exists():
                        results.setdefault(tag,[]).extend(json.loads(receipt.read_text())['rows']);status['completed_models']+=1;continue
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
                    model=PixelHead(xg.shape[2],spatial).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    best=-np.inf;bestepoch=-1;beststate=None;curves=[]
                    for epoch in range(160):
                        model.train();losses=[]
                        order=rng.permutation(len(tr))
                        for chunk in np.array_split(order,max(1,int(np.ceil(len(tr)/128)))):
                            idx=tr[chunk];opt.zero_grad(set_to_none=True)
                            logits=model(xg[idx]);p=torch.sigmoid(logits);w=wg[idx];y=yg[idx]
                            bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum((2,3))/w.sum((2,3)).clamp_min(1e-6)
                            inter=(p*y*w).sum((2,3));den=((p+y)*w).sum((2,3))
                            dice=1-(2*inter+1e-6)/(den+1e-6)
                            per_episode=(bce+dice).mean(1)
                            a=torch.from_numpy(ew[chunk]).cuda();loss=(per_episode*a).sum()/a.sum()
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                        if epoch%10==0:
                            model.eval();vr=[]
                            with torch.no_grad():
                                pp=[]
                                for chunk in np.array_split(val,max(1,int(np.ceil(len(val)/64)))):pp.append(torch.sigmoid(model(xg[chunk])))
                                prob=torch.cat(pp)
                                for j,i in enumerate(val):vr.append({'c':int(classes[i]),'counts':render(prob[j],masks[i],boxes[i],tuple(shapes[i]),gts[i])})
                            metric=summarize(vr);score=np.mean([a['class_miou'] for a in metric.values()])
                            curves.append({'epoch':epoch,'train_bce_plus_dice':float(np.mean(losses)),'validation':metric})
                            if score>best+1e-5:best=score;bestepoch=epoch;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            save(out/'progress.json',{'fold':fold,'view':view,'arch':arch,'seed':seed,'epoch':epoch,'validation':metric})
                            if epoch>=40 and epoch-bestepoch>=40:break
                    model.load_state_dict(beststate);model.eval();rows=[]
                    with torch.no_grad():
                        pp=[]
                        for chunk in np.array_split(test,max(1,int(np.ceil(len(test)/64)))):pp.append(torch.sigmoid(model(xg[chunk])))
                        prob=torch.cat(pp)
                        for j,i in enumerate(test):
                            rows.append({'fold':fold,'e':meta[i]['e'],'c':int(classes[i]),'counts':render(prob[j],masks[i],boxes[i],tuple(shapes[i]),gts[i])})
                    r={'fold':fold,'view':view,'arch':arch,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                       'train_episodes':len(tr),'validation_episodes':len(val),'test_episodes':len(test),
                       'selected_epoch':bestepoch,'last_epoch':epoch,'metrics':summarize(rows),'rows':rows,'curves':curves}
                    save(receipt,r);torch.save({'state':beststate,'view':view,'arch':arch},out/f'f{fold}_{tag}.pt')
                    results.setdefault(tag,[]).extend(rows)
                    status.update(state='TRAINING',completed_models=status['completed_models']+1,fold=fold,view=view,arch=arch,seed=seed,elapsed_seconds=time.monotonic()-start)
                    save(out/'status.json',status);print(json.dumps({k:v for k,v in r.items() if k not in ['rows','curves']}),flush=True)
                    del model,opt,beststate
            del xg
    save(out/'report.json',{'state':'COMPLETED','metadata':status,'summary':{k:summarize(v) for k,v in results.items()},'rows':results})
    status.update(state='COMPLETED');save(out/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    try:run(args)
    except Exception:
        out=Path(args.root)/'pixel_verifier_v1';out.mkdir(exist_ok=True)
        save(out/'error.json',{'state':'ERROR','traceback':traceback.format_exc()});raise
