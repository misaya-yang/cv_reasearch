"""Fixed-candidate observation test; existing trees/models, no downloads.

All candidate geometry is chosen before reading query labels. Save dense matching
maps, not multi-GB raw token caches. This is a development information experiment,
not a FoRIS/FROST reproduction or a new-method benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path[:0] = ['/root/demo4_cache/env', '/root/autodl-tmp/demo4']
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from icx.common import build_model, coco_episodes
from icx.offline2 import accumulate, leaves_of, f1
from utils.data import load_image, load_mask, downsample_mask


def save(path, value):
    p = Path(path); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(p)


def box(mask, margin=.15):
    ys, xs = torch.where(mask)
    if not len(ys): return (0, mask.shape[0], 0, mask.shape[1])
    y0,y1,x0,x1 = int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
    my,mx = int(margin*(y1-y0))+4,int(margin*(x1-x0))+4
    return max(0,y0-my),min(mask.shape[0],y1+my),max(0,x0-mx),min(mask.shape[1],x1+mx)


def resize_crop(x, b, size=448):
    y0,y1,x0,x1 = b
    return F.interpolate(x[...,y0:y1,x0:x1], (size,size), mode='bilinear', align_corners=False)


def mask_up(mask, shape):
    return F.interpolate(mask[None,None].float(),shape,mode='bilinear',align_corners=False)[0,0]>.5


def response(q, r, fg):
    """Complete FG/BG token banks: max similarities and normalized KDE log ratio.

    Fixed beta=.07; no tuning on query labels. Both classes have equal priors.
    This established readout is a controlled observation transform, not novelty.
    """
    if not fg.any(): raise ValueError('empty support foreground')
    if fg.all(): raise ValueError('empty support background')
    out=[]
    for z in q.split(256):
        s=z@r.T; a,b=s[:,fg],s[:,~fg]
        ratio=.07*(torch.logsumexp(a/.07,1)-math.log(a.shape[1])
                   -torch.logsumexp(b/.07,1)+math.log(b.shape[1]))
        out.append(torch.stack([a.max(1).values,b.max(1).values,ratio],1))
    return torch.cat(out)


@torch.inference_mode()
def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    status={'state':'STARTING','pid':os.getpid(),'completed':0,'total':4*args.n,
            'query_gt_role':'labels/counts only, after candidate and observation construction',
            'split':'existing development episodes; future untouched evaluation required',
            'weights':'existing DINOv3-L16, frozen; no download',
            'views':['global_plus_old_crop_scalars','query_crop_plus_old_scalars','paired_crop_plus_old_scalars'],
            'maps':'six raw/debiased dense complete-FG/BG similarity channels + candidate/base geometry',
            'candidate_contract':'native plus top12 GT-free F1 tree regions; also native union each candidate',
            'precision':'official full-image bf16 encoder; crop bf16; matching FP32 TF32off; maps stored FP16',
            'timing':'concurrent diagnostic, no latency claim'}
    save(out/'status.json',status)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=build_model(); enc=model.encoder.m; U=model.positional_basis.float()
    def deb(z): return F.normalize(z-(z@U)@U.T,dim=-1)
    def embedding(ims):
        zz=[]
        for batch in torch.cat(ims).split(4):
            with torch.autocast('cuda',dtype=torch.bfloat16): z=enc.forward_features(batch)
            zz.append(z.float())
        return torch.cat(zz)
    start=time.monotonic(); calls=0; max_cache_diff=0.
    for fold in range(4):
        trees=torch.load(f'/root/demo4_cache/results/l3b_f{fold}.l3.pt',map_location='cpu',weights_only=False)
        eps,_,base=coco_episodes(fold,len(trees))
        for e in range(args.n):
            dest=out/f'f{fold}_e{e:04d}.npz'
            if dest.exists():
                status['completed']+=1; continue
            t=trees[e]; c,tn,rn=eps[e]; assert c==t['c'] and e==t['e']
            # The support annotation is legal input. Do not read query annotation yet.
            ti,_=load_image(Image.open(f'{base}/{tn}').convert('RGB'),model._transform,'cuda')
            ri,_=load_image(Image.open(f'{base}/{rn[0]}').convert('RGB'),model._transform,'cuda')
            ra=np.asarray(Image.open(f'{base}/annotations/{rn[0][:-4]}.png'))==c+1
            rm=load_mask(torch.from_numpy(ra.copy()),1024,'cuda')[0]
            n=len(t['g']);h=int(math.sqrt(n)); assert h==64
            ch=t['children'].astype(np.int64); m=t['r_mask']
            back=m[t['nn_t'].astype(np.int64)].astype(np.float32)
            fwd=np.bincount(t['nn_r'].astype(np.int64)[m],minlength=n).astype(np.float32)
            sums=accumulate(ch,np.stack([np.ones(n,np.float32),back,fwd]))
            fs=f1(sums[1]/sums[0],sums[2]/max(int(m.sum()),1))
            top=np.argsort(-fs)[:12]
            masks=[torch.from_numpy(t['insid3'].reshape(h,h).copy()).cuda()]
            for node in top:
                x=np.zeros(n,bool);x[leaves_of(ch,n,int(node))]=True
                masks.append(torch.from_numpy(x.reshape(h,h)).cuda())
            ms1024=torch.stack([mask_up(x,(1024,1024)) for x in masks]);bs=[box(x) for x in ms1024]
            sb=box(rm); crop_masks=F.interpolate(torch.stack([resize_crop(ms1024[j][None,None].float(),b)[0,0] for j,b in enumerate(bs)])[:,None],(28,28),mode='area')[:,0]
            base_masks=torch.stack([F.adaptive_avg_pool2d(resize_crop(ms1024[0][None,None].float(),b),(28,28))[0,0] for b in bs])
            support_crop=resize_crop(ri,sb);support_mask=resize_crop(rm[None,None].float(),sb)[0,0]>.5
            ims=[support_crop]+[resize_crop(ti,b) for b in bs]
            greys=[support_crop*support_mask[None,None]]+[im*resize_crop(ms1024[j][None,None].float(),bs[j]) for j,im in enumerate(ims[1:])]
            z=embedding(ims+greys);calls+=len(ims)+len(greys)
            raw=F.normalize(z[:14,enc.num_prefix_tokens:],dim=-1); zd=deb(raw)
            # Corresponding support FG/BG bank, same crop resolution and geometry.
            sf=downsample_mask(support_mask[None,None],28,28).flatten()
            if sf.all(): sf=F.adaptive_avg_pool2d(support_mask[None,None].float(),(28,28)).flatten()>.99
            if sf.all(): raise ValueError('support crop has no observable background')
            f=F.normalize(model._extract_features(torch.cat([ri,ti])[None]).float(),dim=2)
            calls+=2
            R,T=f[0,0].flatten(1).T,f[0,1].flatten(1).T; Rd,Td=deb(R),deb(T)
            mf=downsample_mask(rm[None,None],64,64).flatten()
            proto=F.normalize(Rd[mf].mean(0),dim=0)
            diff=float((Td@proto-torch.from_numpy(t['sim'].astype(np.float32)).cuda()).abs().max())
            max_cache_diff=max(max_cache_diff,diff)
            if diff>.003: raise ValueError(f'cached feature contract drift {diff}')
            whole=torch.cat([response(T,R,mf),response(Td,Rd,mf)],1).T.reshape(1,6,64,64)
            global_maps=F.interpolate(whole,(1024,1024),mode='bilinear',align_corners=False)
            fields=[]
            for j,b in enumerate(bs):
                global_j=F.adaptive_avg_pool2d(resize_crop(global_maps,b),(28,28))[0]
                query_j=torch.cat([response(raw[j+1],R,mf),response(zd[j+1],Rd,mf)],1).T.reshape(6,28,28)
                paired_j=torch.cat([response(raw[j+1],raw[0],sf),response(zd[j+1],zd[0],sf)],1).T.reshape(6,28,28)
                geom=torch.stack([crop_masks[j],base_masks[j]])
                fields.append(torch.stack([torch.cat([x,geom]) for x in (global_j,query_j,paired_j)]))
            maps=torch.stack(fields).permute(1,0,2,3,4).half().cpu().numpy()
            # Established CLS/pool/debiased-pool second-look controls, including grey view.
            extra=[]
            weights=torch.cat([F.adaptive_avg_pool2d(support_mask[None,None].float(),(28,28)).flatten()[None],crop_masks.flatten(1)])
            for offset in (0,14):
                v=z[offset:offset+14];cls=F.normalize(v[:,0],dim=1)
                pp=v[:,enc.num_prefix_tokens:]
                pool=F.normalize((pp*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-6),dim=1)
                pd=deb(pool)
                extra.extend([(x[1:]@x[0]) for x in (cls,pool,pd)])
            old=torch.stack(extra,1)
            stats=[]
            basearea=float(masks[0].float().mean());reffrac=float(mf.float().mean())
            for j,x in enumerate(masks):
                s=whole[0].permute(1,2,0)[x]
                if not len(s):s=whole[0].permute(1,2,0).reshape(-1,6)[:1]*0
                stats.append([0. if j==0 else float(fs[top[j-1]]),float(x.float().mean()),basearea,reffrac,
                              float((x&~masks[0]).float().mean()),*s.mean(0).tolist(),*s.std(0,unbiased=False).tolist(),*old[j].tolist()])
            # Query GT appears only after the observations above have been finalized.
            gt=torch.from_numpy((np.asarray(Image.open(f'{base}/annotations/{tn[:-4]}.png'))==c+1).copy()).cuda()
            shape=tuple(gt.shape);native=mask_up(masks[0],shape);counts=[];purity=[];delta=[]
            for x in masks:
                full=mask_up(x,shape);add=full&~native;union=native|full
                counts.append([[int((full&gt).sum()),int((full|gt).sum())],[int((union&gt).sum()),int((union|gt).sum())]])
                purity.append(float((add&gt).sum()/add.sum().clamp_min(1)))
                delta.append([int((add&gt).sum()),int((add&~gt).sum())])
            basic={
                'native': [int((native&gt).sum()),int((native|gt).sum())],
                'full_support_kde': [int((mask_up((whole[0,5]>0),(shape))&gt).sum()),int((mask_up((whole[0,5]>0),(shape))|gt).sum())]}
            np.savez_compressed(dest,maps=maps,stats=np.asarray(stats,np.float32),counts=np.asarray(counts,np.int64),
                                purity=np.asarray(purity,np.float32),added_counts=np.asarray(delta,np.int64),
                                basic=np.asarray(list(basic.values()),np.int64),top=top,fold=fold,e=e,c=c,
                                query=tn,reference=rn[0],cache_sim_max_diff=diff)
            status.update(state='EXTRACTING',completed=status['completed']+1,encoder_image_forwards=calls,
                          elapsed_seconds=time.monotonic()-start,cache_sim_max_diff=max_cache_diff)
            save(out/'status.json',status)
            if status['completed']%10==0:print(json.dumps(status),flush=True)
            del z,f,R,T,Rd,Td,raw,zd,whole,global_maps,maps
        del trees
    status.update(state='COMPLETED');save(out/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--n',type=int,default=300)
    args=ap.parse_args()
    try:run(args)
    except Exception:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json',{'state':'ERROR','traceback':traceback.format_exc()});raise
