#!/usr/bin/env python3
"""Metadata-only seed0 continuation with conservative photograph exclusion."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import time
import numpy as np

PHOTO=re.compile(r'(?:COCO_(?:val|train)(?:2014|2017)_)?(\d{12})\.(?:jpg|jpeg|png)',re.I)
ID_FIELD=re.compile(r'"(?:query|support|image|img|target|reference|photo)_id"\s*:\s*(\d+)')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def photo_id(name):
    match=PHOTO.search(str(name))
    if not match:raise ValueError('Unrecognized COCO photograph '+str(name))
    return int(match.group(1))


def collect(roots,exclude):
    photos=set();receipts=[]
    for root in roots:
        root=Path(root)
        if not root.exists():continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json','.jsonl','.csv','.tsv'):continue
            if path.is_relative_to(exclude):continue
            raw=path.read_bytes()
            try:s=raw.decode('utf-8')
            except UnicodeDecodeError:continue
            ids={int(x) for x in PHOTO.findall(s)}|{int(x) for x in ID_FIELD.findall(s)}
            if ids:
                photos.update(ids);receipts.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),photo_ids=sorted(ids)))
    return photos,receipts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scan-root',action='append',default=[])
    p.add_argument('--extra-exposure',type=Path);p.add_argument('--local-exposure-only',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--host-manifest',type=Path);p.add_argument('--prior-public',type=Path)
    p.add_argument('--per-fold',type=int,default=300);p.add_argument('--max-source-draws',type=int,default=100000)
    a=p.parse_args();started=time.monotonic();a.out.mkdir(parents=True,exist_ok=True)
    blocked,receipts=collect(a.scan_root,a.out.resolve())
    if a.extra_exposure:
        extra=json.loads(a.extra_exposure.read_text());blocked.update(extra['photo_ids']);receipts.append(dict(path=str(a.extra_exposure),sha256=digest(a.extra_exposure),source_receipts=extra['sources']))
    exposure=dict(photo_ids=sorted(blocked),sources=receipts,rule='conservative union of all COCO photo IDs referenced in stored experimental JSON/JSONL/CSV/TSV, including prepared manifests; either role and numeric COCO aliases merge',query_mask_pixels_opened=False)
    (a.out/'exposure.json').write_text(json.dumps(exposure,indent=2)+'\n')
    if a.local_exposure_only:print(json.dumps(dict(blocked_photos=len(blocked),source_files=len(receipts))));return
    if (a.out/'manifest.json').exists():raise FileExistsError('Confirmation manifest is immutable')
    host=json.loads(a.host_manifest.read_text());base=Path(host['data_root']);ann=Path(host['annotation_root'])
    prior=json.loads(a.prior_public.read_text());prior=prior if isinstance(prior,list) else prior['episodes'];lookup={(r['fold'],r['e']):r for r in prior}
    rows=[];audit={};metadata=[];pool=[];parity=0
    for fold in range(4):
        path=base/'splits'/'val'/('fold%d.pkl'%fold);meta=pickle.loads(path.read_bytes());metadata.append(dict(fold=fold,path=str(path),sha256=digest(path)))
        classes=[fold+4*v for v in range(20)];available={}
        for c in classes:
            values=[]
            for name in meta[c]:
                name=str(name);exists=(base/name).is_file() and (ann/Path(name).with_suffix('.png')).is_file()
                eligible=exists and photo_id(name) not in blocked
                pool.append([fold,c,name,exists,eligible])
                if eligible:values.append(name)
            available[str(c)]=dict(metadata_images=len(meta[c]),unseen_legal_images=len(set(values)),distinct_ordered_pairs=len(set(values))*max(len(set(values))-1,0))
        rng=np.random.RandomState(0);selected=[];reasons=Counter();end=-1
        for e in range(a.max_source_draws):
            c=int(rng.choice(classes,1,replace=False)[0]);q=str(rng.choice(meta[c],1,replace=False)[0])
            while True:
                s=str(rng.choice(meta[c],1,replace=False)[0])
                if q!=s:break
            if e<1000:
                r=lookup[(fold,e)]
                if (c,q,s)!=(r['c'],r['query'],r['support']):raise ValueError('Official prefix sampling parity failed')
                parity+=1;continue
            end=e
            if photo_id(q) in blocked or photo_id(s) in blocked:reasons['previously_referenced_photo_either_role']+=1;continue
            if not all((base/n).is_file() and (ann/Path(n).with_suffix('.png')).is_file() for n in (q,s)):reasons['missing_existing_RGB_or_annotation_file']+=1;continue
            selected.append(dict(key='confirm_photo1200:%d_%d_%d'%(fold,e,c),source_key='%d_%d_%d'%(fold,e,c),fold=fold,e=e,c=c,query=q,support=s,batch='confirm_photo_disjoint_v1',query_photo_id=photo_id(q),support_photo_id=photo_id(s)))
            # Selected photographs are deliberately not added to blocked: preserve natural repeats.
            if len(selected)==a.per_fold:break
        rows.extend(selected);audit[str(fold)]=dict(selected=len(selected),last_source_draw=end,rejections=dict(reasons),class_counts=dict(sorted(Counter(str(r['c']) for r in selected).items(),key=lambda x:int(x[0]))),available_by_class=available)
    (a.out/'image_pool.json').write_text(json.dumps(pool,separators=(',',':'))+'\n')
    common={k:host[k] for k in ['data_root','annotation_root','foris_root','projection_basis']}
    manifest=dict(state='PHOTO_DISJOINT_CONFIRMATION_PREPARED' if len(rows)==4*a.per_fold else 'INSUFFICIENT_SELECTED_CONFIRMATION_DRAWS',episodes=rows,**common,protocol=dict(name='metadata-only seed0 continuation, rejected previously referenced photographs; separate confirmation, not official first1000/fold SOTA',source_seed=0,start_source_draw=1000,target_per_fold=a.per_fold,worker_rng='serial numpy RandomState(0), exact choice calls and metadata order',natural_repeats_preserved=True,parameter_or_recipe_selection_on_confirmation=False))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    unique={(r['c'],r['query'],r['support']) for r in rows};selected_photos={r[k] for r in rows for k in ['query_photo_id','support_photo_id']}
    if selected_photos&blocked:raise ValueError('Exposed photograph leaked into confirmation')
    report=dict(state=manifest['state'],n=len(rows),target=4*a.per_fold,metadata=metadata,official_prefix_parity_draws=parity,blocked_photo_count=len(blocked),confirmation_unique_photos=len(selected_photos),overlap_count=0,natural_repeated_episode_identities=len(rows)-len(unique),classes=len({r['c'] for r in rows}),absent_classes=sorted(set(range(80))-{r['c'] for r in rows}),folds=audit,freeze_sha256=digest(a.freeze),manifest_sha256=digest(a.out/'manifest.json'),image_pool_sha256=digest(a.out/'image_pool.json'),exposure_sha256=digest(a.out/'exposure.json'),selection_reads='metadata class membership, photograph IDs and file existence only; no query mask/RGB pixels, GT area/error/model scores',source_scan_limit=a.max_source_draws,remaining_pool_defined_by='per-class unseen legal photo counts and distinct ordered pairs; candidate selection retains replacement sampling',seconds=time.monotonic()-started,source_code_sha256=digest(__file__))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['state','n','classes','absent_classes','blocked_photo_count','confirmation_unique_photos','overlap_count','official_prefix_parity_draws','seconds']}),flush=True)


if __name__=='__main__':main()
