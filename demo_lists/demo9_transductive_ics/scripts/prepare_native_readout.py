#!/usr/bin/env python3
"""Prepare fixed source/asset contract in CPU-only mode. No torch/model load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np
from PIL import Image


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--shared-root',type=Path,default=Path('/root/demo4_cache'))
    p.add_argument('--annotation-root',type=Path,default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES empty required; no paid GPU preparation')
    if not 0<=a.fold<4 or not 1<=a.limit<=20:raise ValueError('Only a bounded first-draw interface probe is prepared')
    data=a.shared_root/'data/COCO2014'
    split=data/f'splits/val/fold{a.fold}.pkl'
    with split.open('rb') as stream:meta=pickle.load(stream)
    classes=[a.fold+4*v for v in range(20)]
    np.random.seed(0);episodes=[]
    for e in range(400):
        c=int(np.random.choice(classes,1,replace=False)[0]);q=str(np.random.choice(meta[c],1,replace=False)[0])
        while True:
            s=str(np.random.choice(meta[c],1,replace=False)[0])
            if q!=s:break
        episodes.append(dict(e=e,c=c,support=s,query=q))
    first={}
    for row in episodes:first.setdefault(row['c'],row)
    selected=[first[c] for c in sorted(first)][:a.limit]
    assets=[split,a.shared_root/'models/dinov3-vitl16-timm/config.json',a.shared_root/'models/dinov3-vitl16-timm/model.safetensors']
    sizes=[]
    for row in selected:
        sample={}
        for role in ('support','query'):
            image=data/row[role];mask=a.annotation_root/Path(row[role]).with_suffix('.png')
            assets.extend([image,mask])
            with Image.open(image) as value:image_shape=value.size;value.verify()
            with Image.open(mask) as value:mask_shape=value.size;value.verify()
            if image_shape!=mask_shape:raise RuntimeError('Official mask/image dimensions disagree')
            sample[role]=dict(shape=list(image_shape))
        sizes.append(sample)
    config=json.loads(assets[1].read_text())
    if config['architecture']!='vit_large_patch16_dinov3':raise ValueError('Expected existing ViT-L/16 DINOv3')
    unique=sorted(set(assets));stat=[]
    for path in unique:
        value=path.stat()
        if not value.st_size:raise RuntimeError('Empty prepared asset')
        stat.append(dict(path=str(path),size=value.st_size,mtime_ns=value.st_mtime_ns))
    report=dict(state='PREPARED_ASSETS',seed=0,fold=a.fold,limit=a.limit,
        frozen_episodes=selected,dimensions=sizes,assets=stat,official_mask_root=str(a.annotation_root),
        split_sha256=file_sha(split),scope='Existing assets/interface preparation only, no GT efficacy selection/model load/GPU allocation/download')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report))
    print(json.dumps(dict(state=report['state'],episodes=len(selected),existing_assets=len(stat),architecture=config['architecture'],torch_imported=False)))


if __name__=='__main__':main()
