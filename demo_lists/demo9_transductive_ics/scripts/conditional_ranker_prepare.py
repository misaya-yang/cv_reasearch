#!/usr/bin/env python3
"""Bounded acquisition of full original final DINO tokens in native PAIR context.

Stage pairs is immediately usable: CPU gallery/split first, then full reference
and query token tensors. All role-photo test exposure is purged before extraction.
Pair contexts are never deduplicated by image name. Existing local weights/data
only. No downloading. Per-episode atomic files, resumable under the same manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import numpy as np


def photo_identity(name):
    stem=Path(name).stem
    match=re.fullmatch(r'COCO_(?:train|val)2014_(\d+)',stem)
    return 'COCO:'+str(int(match.group(1))) if match else stem


def gallery(meta,fold=0,n=400,seed=0):
    rng=np.random.RandomState(seed);classes=[fold+4*i for i in range(20)];episodes=[]
    for e in range(n):
        c=int(rng.choice(classes,1,replace=False)[0]);q=str(rng.choice(meta[c],1,replace=False)[0])
        while True:
            ref=str(rng.choice(meta[c],1,replace=False)[0])
            if ref!=q:break
        episodes.append(dict(e=e,c=c,support=ref,query=q))
    return episodes


def pilot_manifest(episodes,classes=(0,4,8,12,16,20),per_class=10,M=3,held_count=2,dev_count=1):
    rng=np.random.default_rng(0);expanded=[]
    for r in episodes:
        candidates=[o['query'] for o in episodes if o['e']!=r['e'] and o['c']==r['c']
                    and o['query'] not in (r['support'],r['query'])]
        candidates=list(dict.fromkeys(candidates));donors=[str(x) for x in rng.permutation(candidates)[:M]]
        expanded.append(dict(r,donors=donors,image_ids=[r['support'],r['query']]+donors))
    if held_count+dev_count>=len(classes):raise ValueError('Need distinct train/dev/test classes')
    held_classes=set(classes[-held_count:]);dev_classes=set(classes[-held_count-dev_count:-held_count]);held=[];count={c:0 for c in classes}
    for r in expanded:
        if r['c'] in held_classes and count[r['c']]<per_class and len(r['donors'])==M:
            held.append(dict(r,split='test'));count[r['c']]+=1
    test_ids=set(photo_identity(p) for r in held for p in r['image_ids']);dev=[]
    for r in expanded:
        if r['c'] in dev_classes and count[r['c']]<per_class and len(r['donors'])==M and not set(map(photo_identity,r['image_ids']))&test_ids:
            dev.append(dict(r,split='dev'));count[r['c']]+=1
    held=dev+held
    held_ids=set(photo_identity(p) for r in held for p in r['image_ids']);train=[];removed=[]
    for r in expanded:
        if r['c'] not in classes or (r['c'] in held_classes or r['c'] in dev_classes) or count[r['c']]>=per_class or len(r['donors'])!=M:continue
        overlap=sorted(set(map(photo_identity,r['image_ids']))&held_ids)
        if overlap:removed.append(dict(e=r['e'],c=r['c'],shared_photo_ids=overlap));continue
        train.append(dict(r,split='train'));count[r['c']]+=1
    chosen=train+held
    # Interleave classes so first acquisition includes training and held categories.
    ordered=[]
    for i in range(per_class):
        for c in classes:
            rows=[r for r in chosen if r['c']==c]
            if i<len(rows):ordered.append(rows[i])
    train_ids=set(photo_identity(p) for r in train for p in r['image_ids'])
    if train_ids&held_ids:raise AssertionError('Role-photo split leakage')
    aliases={}
    for r in expanded:
        for name in r['image_ids']:aliases.setdefault(photo_identity(name),set()).add(name)
    alias_audit={key:sorted(names) for key,names in aliases.items() if len(names)>1}
    return dict(records=ordered,photo_identity='COCO image numeric ID; non-COCO basename stem',alias_audit=alias_audit,classes=list(classes),held_classes=sorted(held_classes),
                dev_classes=sorted(dev_classes),requested_per_class=per_class,actual_per_class=count,M=M,
                removed_train_candidates=removed,photo_disjoint=True,
                shortfall={str(c):per_class-count[c] for c in classes if count[c]<per_class})


def attach_candidates(payload,row,model,base,annotation_root,DEV):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from utils.data import load_image,load_mask,downsample_mask
    names=row['image_ids'];images=[];gold=[];original_truth=[]
    for name in names:
        image=Image.open(base/name).convert('RGB')
        ann=torch.from_numpy(np.array(Image.open(Path(annotation_root)/str(Path(name).with_suffix('.png')))))
        images.append(load_image(image,model._transform,DEV)[0])
        original_truth.append((ann==row['c']+1).to(DEV))
        gold.append(load_mask(ann==row['c']+1,model.image_size,DEV))
    # Prediction calls receive only the sole support annotation. Query/donor truth
    # below is evaluator-only; no extra-label oracle is in the candidate bank.
    candidates=[];provenance=[];donor_tokens=[];donor_masks=[];contexts=[]
    with torch.inference_mode():
        direct=model.predict_mask(images[0],gold[0],images[1]).reshape(model.image_size,model.image_size).bool()
        candidates.append(direct);provenance.append(dict(kind='direct_1shot',index=0))
        original_extract=model._extract_features
        for j in range(2,len(names)):
            captured=[]
            def capture(inp):
                raw=original_extract(inp)
                pair=F.normalize(raw.float(),p=2,dim=2)[0]
                if tuple(pair.shape)!=(2,1024,64,64):raise RuntimeError('Unexpected donor native-pair features')
                captured.append(pair[-1].flatten(1).T.half().cpu())
                return raw
            model._extract_features=capture
            try: p1=model.predict_mask(images[0],gold[0],images[j]).reshape(model.image_size,model.image_size).bool()
            finally:model._extract_features=original_extract
            if len(captured)!=1:raise RuntimeError('Expected exactly one native feature call per donor P1')
            donor_tokens.append(captured[0]);contexts.append(dict(ordered_photos=[names[0],names[j]],pair_specific=True))
            donor_masks.append(downsample_mask(p1.reshape(1,1,model.image_size,model.image_size),64,64).reshape(64,64).cpu().bool() if p1.any() else torch.zeros((64,64),dtype=torch.bool))
            if p1.any(): second=model.predict_mask(images[j],p1[None],images[1]).reshape(model.image_size,model.image_size).bool()
            else: second=torch.zeros_like(direct)
            candidates.append(second);provenance.append(dict(kind='donor_two_hop',donor_photo_id=names[j],
                donor_slot=j-2,empty_donor_p1=not bool(p1.any()),
                first_pair=[names[0],names[j]],second_pair=[names[j],names[1]],
                caveat='Empty P1 is not semantic absence'))
        native_iu=[];model_iu=[];patch_masks=[];packed=[]
        query_gt=original_truth[1]
        for prediction in candidates:
            pred_original=F.interpolate(prediction.reshape(1,1,model.image_size,model.image_size).float(),size=query_gt.shape,
                                        mode='bilinear',align_corners=False)[0,0]>.5
            native_iu.append([int((pred_original & query_gt).sum()),int((pred_original | query_gt).sum())])
            query_host_gold=gold[1][0].bool()
            model_iu.append([int((prediction & query_host_gold).sum()),int((prediction | query_host_gold).sum())])
            patch_masks.append(downsample_mask(prediction.reshape(1,1,model.image_size,model.image_size),64,64).reshape(64,64).cpu().bool() if prediction.any() else torch.zeros((64,64),dtype=torch.bool))
            packed.append(np.packbits(prediction.cpu().numpy().reshape(-1)))
    original_iu=torch.tensor(native_iu,dtype=torch.long)
    iu=torch.tensor(model_iu,dtype=torch.long)
    payload.update(candidate_state='COMPLETE',donor_token_state='COMPLETE',
        candidate_masks=torch.stack(patch_masks),candidate_iu=iu,
        candidate_iou=(iu[:,0].double()/iu[:,1].clamp_min(1)).float(),
        candidate_model_iu=iu,candidate_original_iu=original_iu,
        candidate_original_iou=(original_iu[:,0].double()/original_iu[:,1].clamp_min(1)).float(),candidate_native_mask_bits=np.stack(packed),
        candidate_native_mask_size=model.image_size,candidate_provenance=provenance,
        candidate_label_resolution='Native INSID3/common.iu model-size1024-square with official load_mask; original-HxW IU separately recorded; no patch-IoU label',
        query_original_shape=list(query_gt.shape),donor_tokens=torch.stack(donor_tokens),
        donor_masks=torch.stack(donor_masks),donor_token_context=contexts,
        current_naive_state='Not synthesized: only direct and native per-donor two-hop candidates in fixed bank')
    return payload


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    ap.add_argument('--out',required=False);ap.add_argument('--stage',choices=['pairs','candidates'],default='pairs')
    ap.add_argument('--stage-limit',type=int,default=10);ap.add_argument('--per-class',type=int,default=10)
    ap.add_argument('--classes',default='0,4,8,12,16,20');ap.add_argument('--M',type=int,default=3)
    ap.add_argument('--held-count',type=int,default=2);ap.add_argument('--dev-count',type=int,default=1)
    ap.add_argument('--max-bytes',type=int,default=2_000_000_000);ap.add_argument('--self-check',action='store_true')
    a=ap.parse_args()
    if a.self_check:self_check();return
    if not a.out:ap.error('--out required')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'))
    import _paths
    sys.path.insert(0,_paths.DEMO4)
    base=Path(os.environ.get('DEMO4_CACHE','/root/demo4_cache'))/'data/COCO2014'
    with open(base/'splits/val/fold0.pkl','rb') as f:meta=pickle.load(f)
    selected=pilot_manifest(gallery(meta),tuple(int(c) for c in a.classes.split(',')),a.per_class,a.M,a.held_count,a.dev_count)
    manifest_path=out/'manifest.json';manifest_text=json.dumps(selected,sort_keys=True)
    if manifest_path.exists() and json.loads(manifest_path.read_text())['records']!=selected['records']:
        raise RuntimeError('Existing output manifest role/split records differ; use a new own pilot directory')
    manifest_path.write_text(manifest_text)
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from icx.common import build_model,DEV
    from utils.data import load_image,load_mask,downsample_mask
    report_path=out/'report.json';report=dict(state='RUNNING',stage=a.stage,args=vars(a),
        manifest=str(manifest_path),manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
        schema='pair-token-stage-v2-full-donor-candidates',records=[],
        contract='Full normalized ORIGINAL final4096x1024 FP16, no pooling/debias compression. Ordered native [reference,query] context. Different pairs never photo-deduped.',
        photo_purge=selected['photo_disjoint'],shortfall=selected['shortfall'])
    def save():
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report));tmp.replace(report_path)
    save();model=None;start=time.time()
    try:
        for row in selected['records'][:a.stage_limit]:
            path=out/f'episode_{row["e"]:04d}.pt'
            if path.exists() and (a.stage=='pairs' or torch.load(path,weights_only=False,map_location='cpu').get('candidate_state')=='COMPLETE'):
                report['records'].append(dict(e=row['e'],c=row['c'],split=row['split'],path=str(path),bytes=path.stat().st_size,resumed=True));save();continue
            expected_bytes=(2*4096*1024*2+200_000) if a.stage=='pairs' else (a.M*4096*1024*2+1_000_000)
            existing_bytes=sum(p.stat().st_size for p in out.glob('episode_*.pt'))
            if existing_bytes+expected_bytes>a.max_bytes:
                report['state']='BUDGET_STOP';break
            if model is None:model=build_model()
            if a.stage=='candidates':
                if not path.exists():raise RuntimeError(f'Missing first pair stage for {path}; acquire pairs before candidates')
                payload=torch.load(path,weights_only=False,map_location='cpu')
                payload=attach_candidates(payload,row,model,base,_paths.COCO_ANN,DEV)
                tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp)
                if existing_bytes-path.stat().st_size+tmp.stat().st_size>a.max_bytes:
                    tmp.unlink();report['state']='BUDGET_STOP';break
                tmp.replace(path)
                report['records'].append(dict(e=row['e'],c=row['c'],split=row['split'],path=str(path),bytes=path.stat().st_size,candidates=1+a.M))
                report.update(elapsed_s=time.time()-start,bytes=sum(p.stat().st_size for p in out.glob('episode_*.pt')));save()
                print('candidate',len(report['records']),row['e'],row['c'],round(report['elapsed_s'],1),report['bytes'],flush=True)
                del payload
                continue
            imgs=[];masks=[]
            for name in (row['support'],row['query']):
                img=Image.open(base/name).convert('RGB')
                ann=torch.from_numpy(np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(name).with_suffix('.png')))))
                imgs.append(load_image(img,model._transform,DEV)[0]);masks.append(load_mask(ann==row['c']+1,model.image_size,DEV))
            with torch.inference_mode():
                pair=F.normalize(model._extract_features(torch.cat(imgs).unsqueeze(0)).float(),p=2,dim=2)[0]
                if tuple(pair.shape)!=(2,1024,64,64):raise RuntimeError(f'Unexpected full final token shape {tuple(pair.shape)}')
                tokens=pair.flatten(2).transpose(1,2).half().cpu()
                refmask=downsample_mask(masks[0].unsqueeze(1),64,64).reshape(64,64).cpu().bool()
            payload=dict(e=row['e'],c=row['c'],split=row['split'],photo_ids=row['image_ids'],
                reference_tokens=tokens[0],query_tokens=tokens[1],reference_mask=refmask,
                token_context=dict(ordered_photos=[row['support'],row['query']],pair_specific=True,
                                   source='Unmodified native INSID3 _extract_features, final normalized original features'),
                candidate_state='PENDING',donor_token_state='PENDING',donor_photo_ids=row['donors'])
            tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path)
            report['records'].append(dict(e=row['e'],c=row['c'],split=row['split'],path=str(path),bytes=path.stat().st_size,resumed=False))
            report.update(elapsed_s=time.time()-start,bytes=sum(p.stat().st_size for p in out.glob('episode_*.pt')));save()
            print(len(report['records']),row['e'],row['c'],row['split'],round(report['elapsed_s'],1),report['bytes'],flush=True)
            del pair,tokens,imgs,masks
        if report['state']=='RUNNING':report['state']='PAIR_STAGE_COMPLETED' if a.stage=='pairs' else 'CANDIDATE_STAGE_COMPLETED'
        report['elapsed_s']=time.time()-start;save()
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc),elapsed_s=time.time()-start);save();raise


def self_check():
    eps=[]
    for c in (0,4,8,12,16,20):
        for i in range(20):eps.append(dict(e=len(eps),c=c,support=f's{c}_{i}',query=f'q{c}_{i}'))
    d=pilot_manifest(eps)
    assert len(d['records'])==60 and len(set(r['c'] for r in d['records'][:10]))==6
    tr=set(p for r in d['records'] if r['split']=='train' for p in r['image_ids'])
    te=set(p for r in d['records'] if r['split']=='test' for p in r['image_ids'])
    assert not tr&te and not d['shortfall']
    dev=set(photo_identity(p) for r in d['records'] if r['split']=='dev' for p in r['image_ids'])
    test=set(photo_identity(p) for r in d['records'] if r['split']=='test' for p in r['image_ids'])
    assert dev and not dev&test
    assert photo_identity('COCO_train2014_000000001234.jpg')==photo_identity('COCO_val2014_000000001234.png')
    assert all(len(r['donors'])==3 and r['support'] not in r['donors'] and r['query'] not in r['donors'] for r in d['records'])
    print('CPU smoke passed: deterministic 60-episode/6-class gallery, interleaved first10, pre-extraction all-role photo purge, fixed donor pool.')


if __name__=='__main__':main()
