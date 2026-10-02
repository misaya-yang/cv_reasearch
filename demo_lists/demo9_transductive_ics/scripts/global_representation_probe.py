"""Frozen encoder intervention, native geometry, and matched compute controls.

No training/pool/new assets. Original/native model path is the reference. All
predictions precede query-GT diagnostics. No raw feature cache is written.
Run only via the prepared resource guard on an actually available GPU.
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
from global_content_attention import global_content_attention, cuda_flash_probe


def fixed_selection(episodes, limit):
    """One first standard draw per class, then rounds; no GT/performance access."""
    by_class={c:[] for c in sorted({int(r[0]) for r in episodes})}
    for e,row in enumerate(episodes):by_class[int(row[0])].append((e,row))
    rows=[]
    for turn in range(max(map(len,by_class.values()))):
        for c,values in by_class.items():
            if turn<len(values):rows.append(values[turn])
    return rows[:limit]


def auc(values, labels):
    values=np.asarray(values);labels=np.asarray(labels,dtype=bool)
    positive=int(labels.sum());negative=len(labels)-positive
    if not positive or not negative:return None
    order=np.argsort(values,kind='stable');ranks=np.empty(len(values),dtype=float)
    lo=0
    while lo<len(values):
        hi=lo+1
        while hi<len(values) and values[order[hi]]==values[order[lo]]:hi+=1
        ranks[order[lo:hi]]=(lo+hi+1)/2;lo=hi
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def pixel_ledger(base, changed, truth):
    return dict(recovered_fn=int((changed & truth & ~base).sum()),
        lost_tp=int((base & truth & ~changed).sum()),
        added_fp=int((changed & ~truth & ~base).sum()),
        removed_fp=int((base & ~truth & ~changed).sum()))


def summary(records):
    totals={}
    for row in records:
        for name,counts in row['iu'].items():
            values=totals.setdefault(name,{}).setdefault(row['c'],[0.,0.])
            values[0]+=counts[0];values[1]+=counts[1]
    return {name:100*float(np.mean([i/max(u,1) for i,u in classes.values()]))
            for name,classes in totals.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    p.add_argument('--out',required=False);p.add_argument('--limit',type=int,default=10)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--self-check',action='store_true')
    a=p.parse_args()
    if a.self_check:self_check();return
    if not a.out:p.error('--out is required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Use fresh output; do not overwrite completed/failed evidence')
    report=dict(state='PREPARING',args=vars(a),records=[],seed=0,
        method_claim='Fixed development probe, no overall gain/novelty claim.',
        contract='One-shot/native pair1024; fixed original U500/tau/merge/background/coverage/seed rules. No query-GT route or layer selection. No feature caches.',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),Path(__file__).with_name('global_content_attention.py'))})
    def save():
        report['class_miou']=summary(report['records'])
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(out/'report.json')
    save()
    if not torch.cuda.is_available():
        report.update(state='NO_GPU',reason='No CUDA; no model load/CPU fallback');save();return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
    sys.path.insert(0,_paths.DEMO4)
    from icx.common import build_model,coco_episodes,DEV
    from utils.data import load_image,load_mask,downsample_mask
    from utils.clustering import agglomerative_clustering
    from tics.imageset import ImageSet,cluster_protos
    from PIL import Image
    started=time.time()
    try:
        report['flash_backend_check']=cuda_flash_probe();save()
        model=build_model();backbone=model.encoder.m;U=model.positional_basis.float()
        episodes,_,base=coco_episodes(a.fold,400,shot=1,seed=0)
        chosen=fixed_selection(episodes,a.limit)
        report['frozen_episodes']=[dict(e=e,c=c,support=refs[0],query=q) for e,(c,q,refs) in chosen]
        report['native_attention_classes']=sorted({type(b.attn).__name__ for b in backbone.blocks})
        report['state']='RUNNING';save()
        def encode(images,enabled=False,mode='global_content'):
            with global_content_attention(backbone,enabled=enabled,mode=mode) as audit:
                raw=model._extract_features(torch.cat(images).unsqueeze(0)).float()
                features=F.normalize(raw,p=2,dim=2)[0].flatten(2).transpose(1,2).contiguous()
            if not audit['rollback_complete']:raise RuntimeError('Attention rollback incomplete')
            return features,audit
        def geometry(features):
            labels=[];protos=[]
            for f in features:
                lab=agglomerative_clustering(f,model.tau)
                labels.append(lab);protos.append(cluster_protos(f,lab,int(lab.max())+1)[0])
            return torch.stack(labels),protos
        def debias(features):return F.normalize(features-(features@U)@U.T,dim=-1)
        def decode(match,geom,c,names,masks):
            labels,protos=geom
            bits=np.stack([np.packbits(g[0].cpu().numpy().reshape(-1)) for g in masks])
            small=torch.stack([downsample_mask(g.unsqueeze(1),64,64).reshape(64,64) if g.any()
                              else torch.zeros((64,64),dtype=torch.bool,device=DEV) for g in masks])
            s=ImageSet(dict(c=c,names=names,fq=debias(match),lab=labels,Po=protos,
                gt64=small,gt_bits=bits,S=model.image_size))
            result=s.up(s.predict(1,[0],[s.gt64[0]]))
            return result,s.fd,small
        with torch.inference_mode():
            for e,(c,query,refs) in chosen:
                names=[refs[0],query];images=[];masks=[]
                for name in names:
                    image=Image.open(Path(base)/name).convert('RGB')
                    annotation=torch.from_numpy(np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(name).with_suffix('.png')))))
                    images.append(load_image(image,model._transform,DEV)[0]);masks.append(load_mask(annotation==c+1,model.image_size,DEV))
                native,na=encode(images)
                noop,noa=encode(images,enabled=False)
                if not torch.equal(native,noop):raise RuntimeError(f'Native no-op feature drift at {e}')
                native_geom=geometry(native)
                baseline,fd,small=decode(native,native_geom,c,names,masks)
                official=model.predict_mask(images[0],masks[0],images[1]).reshape(model.image_size,model.image_size).bool() if masks[0].any() else torch.zeros_like(baseline)
                if not torch.equal(baseline,official):
                    report['renderer_failure']=dict(e=e,changed_pixels=int((baseline!=official).sum()));save()
                    raise RuntimeError('Native renderer not exact; stop mechanism attribution')
                semantic,sa=encode(images,enabled=True)
                kernel,ka=encode(images,enabled=True,mode='padded_native')
                flipped,fa=encode([torch.flip(x,[-1]) for x in images])
                flipped=flipped.transpose(1,2).reshape(2,1024,64,64).flip(-1).flatten(2).transpose(1,2)
                averaged=F.normalize(native+flipped,dim=-1)
                semantic_geom=geometry(semantic)
                predictions={'native_native':baseline};spaces={'native':fd}
                for name,matching,grouping in [('content_native',semantic,native_geom),
                     ('native_content',native,semantic_geom),('content_content',semantic,semantic_geom),
                     ('kernel_native',kernel,native_geom),('flip_average_native',averaged,native_geom)]:
                    prediction,space,_=decode(matching,grouping,c,names,masks)
                    predictions[name]=prediction
                    if name=='content_native':spaces['content']=space
                # Predictions fully frozen before query-label diagnostic calculations.
                truth=masks[1][0].bool();row=dict(e=e,c=c,names=names,iu={},pixels={},diagnostics={},
                    native_noop_exact=True,native_renderer_exact=True,
                    kernel_feature_max_abs=float((kernel-native).abs().max()),
                    kernel_changed_pixels=int((predictions['kernel_native']!=baseline).sum()),
                    attention_audit=sa,kernel_audit=ka)
                for name,prediction in predictions.items():
                    row['iu'][name]=[int((prediction&truth).sum()),int((prediction|truth).sum())]
                    row['pixels'][name]=pixel_ledger(baseline,prediction,truth)
                support=small[0].flatten().bool();target=small[1].flatten().bool()
                if support.any() and (~support).any():
                    for name,space in spaces.items():
                        sims=space[1]@space[0].T
                        fg=sims[:,support].max(1).values;bg=sims[:,~support].max(1).values
                        margin=fg-bg;values=margin.cpu().numpy();labels=target.cpu().numpy()
                        accepted=margin>0
                        row['diagnostics'][name]=dict(auc=auc(values,labels),
                            fg_margin=float(margin[target].mean()) if target.any() else None,
                            bg_margin=float(margin[~target].mean()) if (~target).any() else None,
                            fg_nn_recall=float(accepted[target].float().mean()) if target.any() else None,
                            bg_nn_specificity=float((~accepted[~target]).float().mean()) if (~target).any() else None,
                            all_patch_cross_cosine=float(sims.mean()))
                report['records'].append(row);report['elapsed_concurrent_s']=time.time()-started
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
                print(json.dumps(dict(e=e,c=c,count=len(report['records']),scores=report['class_miou'],elapsed_s=report['elapsed_concurrent_s'])),flush=True)
                del predictions,spaces,native,noop,semantic,kernel,flipped,averaged,images,masks,native_geom,semantic_geom
        report.update(state='COMPLETED',elapsed_concurrent_s=time.time()-started);save()
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex),elapsed_concurrent_s=time.time()-started);save();raise


def self_check():
    eps=[(c,'q'+str(i),['s'+str(i)]) for i,c in enumerate([4,0,8,0,4,8])]
    assert [e for e,_ in fixed_selection(eps,6)]==[1,0,2,3,4,5]
    assert auc([1,2,3,4],[0,0,1,1])==1
    assert auc([1,1,1,1],[0,0,1,1])==.5
    base=torch.tensor([1,0,0,1],dtype=torch.bool);new=~base;truth=torch.tensor([1,1,0,0],dtype=torch.bool)
    assert pixel_ledger(base,new,truth)==dict(recovered_fn=1,lost_tp=1,added_fp=1,removed_fp=1)
    assert summary([dict(c=0,iu={'x':[1,2]}),dict(c=0,iu={'x':[1,2]}),dict(c=4,iu={'x':[2,2]})])=={'x':75.}
    print('CPU checks passed: fixed class sampling, tie-aware AUC, complete error ledger, class-IU aggregation; no CUDA/model access.')


if __name__=='__main__':main()
