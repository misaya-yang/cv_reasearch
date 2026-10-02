"""Freeze an independent image cohort from EXISTING shared COCO archives.

Only category presence metadata is used. No images, masks, model predictions or
test quality are opened. Actual evaluation waits for a development-frozen method.
This is a new image-disjoint COCO cohort, not the official COCO-20i episode list.
"""
import argparse
import json
from pathlib import Path
import random
import zipfile
from collections import defaultdict


def run(out):
    cached=Path('/root/demo4_cache/data/COCO2014/val2014')
    blocked={int(p.stem.split('_')[-1]) for p in cached.glob('*.jpg')}
    zpath=Path('/root/autodl-pub/COCO2017/annotations_trainval2017.zip')
    with zipfile.ZipFile(zpath) as z:
        data=json.load(z.open('annotations/instances_train2017.json'))
    cats=sorted(c['id'] for c in data['categories']);assert len(cats)==80
    names={c['id']:c['name'] for c in data['categories']}
    presence=defaultdict(set)
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['area']>=64 and a['image_id'] not in blocked:
            presence[a['category_id']].add(a['image_id'])
    rng=random.Random(2040);used=set();records=[]
    # Reserve scarce classes first; all query/support roles globally photo-disjoint.
    for cid in sorted(cats,key=lambda c:(len(presence[c]),c)):
        available=sorted(presence[cid]-used);assert len(available)>=20,(names[cid],len(available))
        sampled=rng.sample(available,20);used.update(sampled);c=cats.index(cid)
        for e in range(10):
            q,r=sampled[2*e:2*e+2]
            records.append({'fold':c%4,'class':c,'category_id':cid,'class_name':names[cid],
                            'within_class_episode':e,'query_image_id':q,'support_image_id':r,
                            'query_member':f'train2017/{q:012d}.jpg','support_member':f'train2017/{r:012d}.jpg'})
    records.sort(key=lambda r:(r['fold'],r['class'],r['within_class_episode']))
    assert len(records)==800 and len(used)==1600 and not used&blocked
    result={'state':'MANIFEST_FROZEN_UNEVALUATED','seed':2040,'episodes':800,'classes':80,
            'source_annotation_archive':str(zpath),'source_image_archive':'/root/autodl-pub/COCO2017/train2017.zip',
            'excluded_all_cached_COCO2014_val_image_ids':len(blocked),
            'sampling':'10 episodes/class, category presence/noncrowd annotation area>=64; scarce classes reserved first; all 1600 support/query images distinct',
            'boundary':'no test pixels or model outputs read; eligibility masks must be checked deterministically before any predictions; no method tuning on this cohort',
            'benchmark':'independent image-disjoint COCO cohort, NOT official published COCO-20i episodes',
            'records':records}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Never silently replace a frozen manifest')
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();run(args.out)
