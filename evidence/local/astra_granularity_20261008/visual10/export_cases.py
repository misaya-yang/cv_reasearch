"""Export ten existing cases for qualitative LOCAL review; no encoder or new inference.

Use only PACO development folds f0/f1 plus the already-evaluated COCO cohort.
Selection intentionally spans deletion failures, remaining overreach, successes.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/astra_granularity_20261008')
O=Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
DEST=ROOT/'visual10'


def main():
    DEST.mkdir(exist_ok=False)
    pools={'PACO':[],'COCO':[]}
    for pack in ['paco_part_f0','paco_part_f1','coco_fresh600']:
        cache=ROOT/'cache'/pack
        rows=json.loads((cache/'rows.json').read_text())
        counts=[json.loads(x) for x in (ROOT/'candidate_C1'/(pack+'.jsonl')).read_text().splitlines()]
        with np.load(cache/'inference.npz') as z:fields=z['field'];cov=z['cov']
        with np.load(cache/'evaluation.npz') as z:truth=z['truth']
        with np.load(ROOT/'candidate_C1'/(pack+'_masks.npz')) as z:
            index=z['names'].tolist().index('joint_mask');pred=np.unpackbits(z['masks'][:,index],axis=-1).astype(bool)
        for i,(row,rec) in enumerate(zip(rows,counts)):
            b=fields[i]>.5;t=truth[i];T=int(t.sum());FP=int((b&~t).sum())
            bi=rec['iu']['B'][0]/max(rec['iu']['B'][1],1)
            ci=rec['iu']['joint_mask'][0]/max(rec['iu']['joint_mask'][1],1)
            dto=rec['edits']['joint_mask'][2];dfo=rec['edits']['joint_mask'][3]
            r=dict(row,pack=pack,record=rec,truth_tokens=T,base_fp=FP,base_iou=bi,candidate_iou=ci,
                   delta=ci-bi,pred_to_truth=float(b.sum()/max(T,1)),delete_true_fraction=dto/max(T,1),
                   remaining_fp_ratio=(FP-dfo)/max(T,1),field=fields[i],cov=cov[i],truth=t,pred=pred[i])
            pools['COCO' if pack=='coco_fresh600' else 'PACO'].append(r)
    picked=[];used=set();used_classes=set()
    def choose(pool,kind,number,condition,score):
        candidates=sorted([r for r in pools[pool] if condition(r)],key=score,reverse=True)
        for _ in range(number):
            available=[r for r in candidates if (r['pack'],r['key']) not in used and (pool,r['record']['class']) not in used_classes]
            if not available:raise RuntimeError('Insufficient distinct-class cases: '+kind)
            r=available[0];r['reason']=kind;r['dataset']=pool;picked.append(r)
            used.add((r['pack'],r['key']));used_classes.add((pool,r['record']['class']))
    choose('PACO','PACO: severe true-target deletion',2,lambda r:r['base_iou']>=.4,lambda r:r['delete_true_fraction'])
    choose('PACO','PACO: substantial overreach remains',2,lambda r:r['pred_to_truth']>=2 and r['base_fp']>=16,lambda r:r['remaining_fp_ratio'])
    choose('PACO','PACO: successful deletion control',2,lambda r:r['delta']>=.05 and r['pred_to_truth']>1,lambda r:r['delta'])
    choose('COCO','COCO: severe true-target deletion',2,lambda r:r['base_iou']>=.4,lambda r:r['delete_true_fraction'])
    choose('COCO','COCO: substantial overreach remains',1,lambda r:r['pred_to_truth']>=2 and r['base_fp']>=16,lambda r:r['remaining_fp_ratio'])
    choose('COCO','COCO: successful deletion control',1,lambda r:r['delta']>=.05 and r['pred_to_truth']>1,lambda r:r['delta'])
    manifest=[]
    for number,r in enumerate(picked,1):
        dest=DEST/f'case{number:02d}';dest.mkdir()
        if r['dataset']=='COCO':
            data=Path('/root/demo4_cache/data/COCO2014')
            rp,qp=data/r['support'],data/r['query']
            rm=np.asarray(Image.open(data/'annotations'/Path(r['support']).with_suffix('.png')))==r['c']+1
            # Original query annotation is supplied for visual context; metric truth stays sealed tokens.
            tm=np.asarray(Image.open(data/'annotations'/Path(r['query']).with_suffix('.png')))==r['c']+1
        else:
            data=O/'claude_packs'/r['pack']
            rp,qp=data/'data'/r['support'],data/'data'/r['query']
            rm=np.asarray(Image.open(data/'ann'/r['support']).convert('L'))>0
            tm=np.asarray(Image.open(data/'ann'/r['query']).convert('L'))>0
        Image.open(rp).convert('RGB').save(dest/'reference.png')
        Image.open(qp).convert('RGB').save(dest/'query.png')
        Image.fromarray(rm.astype(np.uint8)*255).save(dest/'reference_mask.png')
        Image.fromarray(tm.astype(np.uint8)*255).save(dest/'query_mask.png')
        np.savez_compressed(dest/'masks.npz',field=r['field'].reshape(64,64),base=(r['field']>.5).reshape(64,64),
                            candidate=r['pred'].reshape(64,64),truth=r['truth'].reshape(64,64),reference_coverage=r['cov'].reshape(64,64))
        meta={k:v for k,v in r.items() if k not in ['field','cov','truth','pred']}
        meta['case']=number;meta['source_query']=str(qp);meta['source_reference']=str(rp)
        meta['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()}
        (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');manifest.append(meta)
    (DEST/'manifest.json').write_text(json.dumps({'selection':'6 PACO f0/f1 and 4 COCO; intentional failure/success examples, not representative statistics',
        'new_inference':False,'held_folds_read':False,'cases':manifest},indent=2)+'\n')
    print(json.dumps([dict(case=r['case'],dataset=r['dataset'],key=r['key'],reason=r['reason'],B=r['base_iou'],C1=r['candidate_iou']) for r in manifest],indent=2))


if __name__=='__main__':main()
