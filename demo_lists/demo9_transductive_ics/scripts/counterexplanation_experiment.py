"""Bounded complete-source negative-reference observation; no auto start/shutdown.

Source reference grouping is a local semantic/spatial HYPOTHESIS, never a known
background concept. Real DINO interface/mask equality are runtime gates.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata

from counterexplanation_core import (cache_public_extractor, counterexplanation_cut,
                                    _connected_exact, positive_only_cut)


def proposals(host, got, gold, support_model):
    """Exactly public FoRIS feature/RGB/position grouping, on support not query.

    Intersect each connected group with original known BG; never the whole BG.
    Priority is mean ORIGINAL positive-query score routed by reverse NN. At
    most2 groups; same-area random BG controls use identical connectivity.
    """
    from utils.clustering import agglomerative_clustering
    from utils.data import denormalize
    raw=F.normalize(got['raw'][0,0],dim=0);c,h,w=raw.shape
    x=raw.flatten(1).T
    rgb=denormalize(F.interpolate(support_model[None],(h,w),mode='bilinear',align_corners=False)[0]).clamp(0,1)
    colour=F.normalize(rgb.flatten(1).T,dim=1)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
    pos=F.normalize(torch.stack((yy,xx),-1).reshape(-1,2),dim=1)
    groups=agglomerative_clustering(F.normalize(torch.cat((x,.35*colour,.20*pos),1),dim=1),tau=host.tau)
    groups=groups.reshape(h,w).cpu().numpy()
    deb=got['deb'][0];gq=F.normalize(deb[-1].flatten(1).T,dim=1);gs=F.normalize(deb[0].flatten(1).T,dim=1)
    sim=gq@gs.T;back=sim.argmax(0)
    score=got['score'].float();priority=score.flatten()[back].reshape(h,w).cpu().numpy()
    fg=np.asarray(gold.cpu(),bool);bg=~fg;bg_count=int(bg.sum());candidates=[]
    for label in np.unique(groups):
        grid=groups==label
        if grid.sum()<4:
            continue
        up=np.asarray(Image.fromarray(grid.astype('uint8')).resize((fg.shape[1],fg.shape[0]),Image.Resampling.NEAREST),bool)&bg
        components,n=ndimage.label(up)
        for k in range(1,n+1):
            mask=components==k;area=int(mask.sum())
            if area<4*fg.size/(h*w) or area==bg_count:
                continue
            # One original BG-connected piece of one public semantic group.
            low=np.asarray(Image.fromarray(mask.astype('uint8')).resize((w,h),Image.Resampling.NEAREST),bool)
            if not low.any():
                continue
            candidates.append((float(priority[low].mean()),int(label),k,mask))
    candidates.sort(key=lambda v:(-v[0],v[1],v[2]))
    hard=[v[3] for v in candidates[:2]];random=[];labs,n=ndimage.label(bg);sizes=np.bincount(labs.ravel());rng=np.random.default_rng(0)
    for mask in hard:
        area=int(mask.sum());allowed=np.flatnonzero(bg&(sizes[labs]>=area))
        if not len(allowed):
            raise RuntimeError('random matched-BG component unavailable')
        point=np.unravel_index(int(rng.choice(allowed)),bg.shape)
        random.append(_connected_exact(bg,point,area))
    assert all(not (m&fg).any() for m in hard+random)
    assert all(a.sum()==b.sum() for a,b in zip(hard,random))
    return hard,random,dict(count=len(hard),areas=[int(m.sum()) for m in hard],whole_BG_rejected=True),sim


def auc(value,truth,valid):
    v=np.asarray(value)[valid];y=np.asarray(truth)[valid];n=int(y.sum());m=int((~y).sum())
    return None if min(n,m)<4 else float((rankdata(v)[y].sum()-n*(n+1)/2)/(n*m))


def fixture_host(man):
    sys.path.insert(0,man['foris_root']);import models.foris as source
    class Encoder(torch.nn.Module):
        def get_intermediate_layers(self,x,n=1,reshape=True):
            v=F.avg_pool2d(x,16);e=F.avg_pool2d((x-F.interpolate(v,scale_factor=16)).abs(),16)
            return [torch.cat((v,v.square(),v[:,:2]-v[:,1:3],e),1)]
    original=source.FoRIS._build_positional_basis
    # Fixed synthetic orthonormal fixture basis, no SVD/model weights loaded.
    source.FoRIS._build_positional_basis=lambda self,device:torch.eye(11,2,device=device)
    try:
        return source.FoRIS(encoder=Encoder(),image_size=256,svd_components=2,tau=.6,
                           mask_refiner='bilinear',resize_to_orig_size=False,device='cpu').eval()
    finally:
        source.FoRIS._build_positional_basis=original


def run(a):
    torch.set_num_threads(1 if a.fixture else 2)
    sys.path.insert(0,a.public_scripts);import extent_experiment as public
    from tics.extent_cut import LEVELS,binarise,choose,normalise,level_statistics,neighbour_affinity
    man=json.loads(Path(a.manifest).read_text());out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise RuntimeError('fresh own output directory required')
    out.mkdir(parents=True)
    if not a.fixture and os.environ.get('DEMO9_CUDA_GUARD')!='1':raise RuntimeError('finite guard required')
    host=fixture_host(man) if a.fixture else public.build_host(SimpleNamespace(foris_root=None,demo4_root=a.demo4_root,fixture=False),man,'cuda')
    dev='cpu' if a.fixture else 'cuda';records=[];report=dict(state='RUNNING',episodes=0,premise='UNTESTED',mask_before_queryGT=True,feature_files_written=False)
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2))
    start=time.monotonic();save()
    try:
        for row in man['episodes'][:10]:
            if time.monotonic()-start>600:raise RuntimeError('bounded600s exceeded')
            sp=Image.open(Path(man['data_root'])/row['support']).convert('RGB');qp=Image.open(Path(man['data_root'])/row['query']).convert('RGB')
            gold=torch.from_numpy((np.asarray(Image.open(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')))==row['c']+1).copy())
            torch.manual_seed(0)
            with cache_public_extractor(host) as cache:
                native,got,model_ref,tgt=public.run_foris(host,sp,gold,qp)
                raw_support=host._transform(sp).to(dev)
                torch.manual_seed(0)
                replay=public.run_foris(host,sp,gold,qp)[0]
                if not torch.equal(native,replay):raise RuntimeError('public cached/uncached native mask differs')
                hard,random,proposal_info,sim=proposals(host,got,gold,raw_support)
                negfields=[];randomfields=[];negative_meta=[]
                for kind,masks in [('hard',hard),('random',random)]:
                    fields=negfields if kind=='hard' else randomfields
                    for mask in masks:
                        if (mask&gold.numpy()).any():raise RuntimeError('negative mask contains labelled FG pixel')
                        torch.manual_seed(0)
                        _,ng,nref,_=public.run_foris(host,sp,torch.from_numpy(mask.copy()),qp)
                        # Same source nearest preprocessing must preserve subset.
                        if (nref&model_ref).any():raise RuntimeError('preprocessed negative overlaps known FG')
                        fields.append(ng['score'].float().cpu().numpy())
                        negative_meta.append(dict(kind=kind,original_area=int(mask.sum()),model_area=int(nref.sum())))
                source=got['score'].float().cpu().numpy();hw=tuple(native.shape)
                null=np.ones_like(source)/source.size
                cuts={}
                cuts['complete_negative']=counterexplanation_cut(source,negfields,LEVELS)
                cuts['random_BG']=counterexplanation_cut(source,randomfields,LEVELS)
                cuts['positive_only']=positive_only_cut(source,LEVELS)
                cuts['uniform_response']=counterexplanation_cut(source,[null],LEVELS,uniform_control=True)
                masks=dict(native=native)
                for name,(cut,reason,evidence) in cuts.items():
                    masks[name]=native if cut==.5 else public.finalise(host,binarise(got['score'].float(),cut,hw),tgt)
                fg=F.interpolate(model_ref[None,None].float(),got['score'].shape,mode='area')[0,0].flatten()>=.5
                margin=(sim[:,fg].amax(1)-sim[:,~fg].amax(1)).reshape(source.shape).cpu().numpy()
                if cache['calls']!=1:raise RuntimeError('encoder cache budget violated')
                encoding_calls,cache_reuses=cache['calls'],cache['reuses']
            # First query-label opening: all legal proposals and predictions frozen.
            truth=torch.from_numpy((np.asarray(Image.open(Path(man['annotation_root'])/Path(row['query']).with_suffix('.png')))==row['c']+1).copy()).to(dev)
            tm=F.interpolate(truth[None,None].float(),hw,mode='nearest')[0,0].bool()
            sn=(source-source.min())/max(float(source.max()-source.min()),1e-6)
            coverage=F.interpolate(tm[None,None].float(),source.shape,mode='area')[0,0].cpu().numpy();y=coverage>=.5;band=(sn>=.35)&(sn<=.65)
            native_coverage=F.interpolate(native[None,None].float(),source.shape,mode='area')[0,0].cpu().numpy();true_tp=y&(native_coverage>=.5)
            def iu(pred):
                original=F.interpolate(pred[None,None].float(),truth.shape,mode='bilinear',align_corners=False)[0,0]>.5
                return [int((original&truth).sum()),int((original|truth).sum())]
            iurows={k:iu(v) for k,v in masks.items()}
            if not a.fixture and 'expect_original_iu' in row and iurows['native']!=row['expect_original_iu']:raise RuntimeError('per-case native original IU gate')
            info=dict(score_auc=auc(sn,y,band),NN_margin_auc=auc(margin,y,band),negative_fields=[])
            for name,fields in [('hard',negfields),('random',randomfields)]:
                evidence=cuts['complete_negative' if name=='hard' else 'random_BG'][2]
                info[name+'_excess_auc']=None if evidence is None else auc(evidence,y,band)
                for field in fields:
                    nz=(field-field.min())/max(float(field.max()-field.min()),1e-6);corr=None if np.std(nz)==0 else float(np.corrcoef(sn.ravel(),nz.ravel())[0,1])
                    info['negative_fields'].append(dict(kind=name,correlation=corr,top10_overlap=float(((sn>=np.quantile(sn,.9))&(nz>=np.quantile(nz,.9))).sum()/max(1,(sn>=np.quantile(sn,.9)).sum())),TP_high_negative=float((nz[true_tp]>.5).mean()) if true_tp.any() else None))
            per= dict(fold=row['fold'],e=row['e'],c=row['c'],support=row['support'],query=row['query'],proposals=proposal_info,negative_masks=negative_meta,encoding_calls=encoding_calls,cache_reuses=cache_reuses,native_exact=True,original_iu=iurows,information=info,cut={k:dict(level=v[0],reason=v[1]) for k,v in cuts.items()},TP_injury={k:int((native&tm&~v).sum()) for k,v in masks.items()},FN_repair={k:int((~native&tm&v).sum()) for k,v in masks.items()})
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(per)+'\n')
            records.append(per);report['episodes']=len(records);save();print(json.dumps(per),flush=True)
        report.update(state='COMPLETED',scope='10 already-seen observation tasks; no method score',seconds=time.monotonic()-start,source_cpu_fixture=bool(a.fixture));save()
    except BaseException as err:
        report.update(state='ERROR',error=repr(err));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--fixture',action='store_true');p.add_argument('--public-scripts',default='/root/autodl-tmp/demo9_extent/scripts');p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4');run(p.parse_args())
