#!/usr/bin/env python3
"""Native FoRIS full-pipeline causal smoke: replace only the single-seed intra-similarity factor by one.

Baseline feature forward is native; ablation/replay reuse exactly those features. No downloads or disk feature cache.
Later normalization, score boost, disagreement and reweighting all run unchanged. GT pixel union is diagnostic only.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
import argparse, hashlib, inspect, json, sys, textwrap, time
from pathlib import Path
sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo8_local_verification/foris_source','/root/autodl-tmp/demo4']
import torch, numpy as np, torch.nn.functional as F
import models.foris as fm
from icx.common import TimmDINOv3, coco_episodes
from PIL import Image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--limit',type=int,default=10);a=ap.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.45')))
    torch.backends.cuda.matmul.allow_tf32=False
    source=textwrap.dedent(inspect.getsource(fm.FoRIS._build_seed_cluster_prior))
    needle='torch.einsum("c,kc->k", protos[seed_cluster], protos).clamp_min(0.0)'
    if source.count(needle)!=1: raise RuntimeError('Unrecognized prior implementation; no silent patch')
    changed=source.replace(needle,'torch.ones(protos.shape[0], device=protos.device, dtype=protos.dtype)')
    namespace={};exec(changed,dict(vars(fm)),namespace);ablate_prior=namespace['_build_seed_cluster_prior']
    class Native(fm.FoRIS):
        ablate=False
        cached=None
        def _extract_features(self,imgs):
            if self.cached is None: self.cached=super()._extract_features(imgs)
            return self.cached
        def _build_seed_cluster_prior(self,*args,**kwargs):
            return ablate_prior(self,*args,**kwargs) if self.ablate else super()._build_seed_cluster_prior(*args,**kwargs)
    # Hash the entire feature tensor: no rounded-sum cache key collisions.
    original_cluster=fm.agglomerative_clustering;memo={}
    def clustered(x,tau=.6):
        key=(tuple(x.shape),str(x.dtype),tau,hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        if key not in memo:memo[key]=original_cluster(x,tau=tau)
        return memo[key]
    fm.agglomerative_clustering=clustered
    model=Native(TimmDINOv3().eval(),image_size=1024,mask_refiner='bilinear',resize_to_orig_size=False).cuda().eval()
    for p in model.parameters():p.requires_grad=False
    eps,_,base=coco_episodes(0,400,shot=1,seed=0)
    annotations=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
    records=[];t0=time.time()
    report=dict(state='RUNNING',fold=0,records=records,args=vars(a),prior_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                mask_protocol=str(annotations),intervention='Only intra-to-one-seed factor replaced by one; full downstream FoRIS unchanged.',
                scope='Small development causal smoke, not new method/FoRIS transfer success. GT union is a generous pixel ceiling, not attainable inference.')
    def save():
        acc={}
        for r in records:
            for key,(i,u) in r['iu'].items():
                row=acc.setdefault(key,{}).setdefault(r['c'],[0.,0.]);row[0]+=i;row[1]+=u
        report['miou']={key:100*float(np.mean([i/max(u,1) for i,u in rows.values()])) for key,rows in acc.items()}
        report['episodes']=len(records);report['elapsed_s']=time.time()-t0
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report));tmp.replace(out)
    def mask(name,c):
        ann=np.array(Image.open(annotations/(name[:-4]+'.png')))
        return F.interpolate(torch.from_numpy(ann==c+1)[None,None].float().cuda(),size=(1024,1024),mode='nearest')[0,0]>.5
    save()
    try:
        for e,(c,qn,rn) in enumerate(eps[:a.limit]):
            x=torch.stack([model._transform(Image.open(Path(base)/name).convert('RGB')) for name in (rn[0],qn)]).cuda()
            gold=mask(rn[0],c);model.cached=None;memo.clear()
            with torch.inference_mode():
                model.ablate=False;b=model.predict(x[:1],gold[None],x[1])
                model.ablate=True;p=model.predict(x[:1],gold[None],x[1])
                model.ablate=False;replay=model.predict(x[:1],gold[None],x[1])
            if not torch.equal(b,replay):raise RuntimeError(f'Nonexact native replay e={e}')
            truth=mask(qn,c);delta=p&~b
            preds={'native':b,'no_intra':p,'GT_pixel_unblock':b|(delta&truth)}
            iu={k:[float((v&truth).sum()),float((v|truth).sum())] for k,v in preds.items()}
            records.append(dict(e=e,c=c,query=qn,support=rn[0],iu=iu,native_replay_exact=True,
                                added_tp=float((delta&truth).sum()),added_fp=float((delta&~truth).sum()),
                                removed_tp=float((b&~p&truth).sum()),removed_fp=float((b&~p&~truth).sum())))
            model.cached=None;memo.clear();del x,p,b,replay,preds;torch.cuda.empty_cache();save()
            print(e+1,report['miou'],flush=True)
        report['state']='COMPLETED';save()
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex));save();raise

if __name__=='__main__':main()
