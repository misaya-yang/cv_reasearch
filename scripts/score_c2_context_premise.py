#!/usr/bin/env python3
"""Sealed C2 minimum vs direct/wrong AUC on the fixed development photo halves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from frozen_dino_plan import write
from score_representation_premise import macro,photo_half,photo_key
from ics.official_data import array_hash,file_hash

DATASETS=('coco','pascal_part','paco_part')
FIELDS={'minimum':'probability','direct':'direct_probability','wrong':'wrong_probability'}


def episode_auc(row,parent_mask,fields):
    """Label-only scoring; geometry is raw GT -> CLI frame ->64 area."""
    from sklearn.metrics import roc_auc_score
    with Image.open(row['query_mask_path']) as image:truth=(np.asarray(image.convert('L'))>0).astype(np.uint8)
    if array_hash(truth)!=row['query_mask_hash']:raise ValueError('Frozen query annotation changed')
    truth=F.interpolate(torch.from_numpy(truth)[None,None].float(),(1024,1024),mode='nearest')
    target=(F.interpolate(truth,(64,64),mode='area')[0,0].numpy()>=.5).ravel()
    circle=(F.interpolate(torch.from_numpy(parent_mask)[None,None].float(),(64,64),mode='area')[0,0].numpy()>=.5).ravel()
    if len(np.unique(target[circle]))!=2:return {}
    result={}
    for label,key in FIELDS.items():
        score=fields[key]
        if score.shape!=(4096,) or not np.isfinite(score).all():raise ValueError('Invalid C2 score field')
        result[label]=float(roc_auc_score(target[circle],score[circle]))
    return result


def summarize(records):
    table=[]
    for half in (0,1):
        for name in DATASETS:
            paired=[r for r in records if r['half']==half and r['dataset']==name and set(FIELDS)<=set(r['auc'])]
            values={field:macro(paired,field) for field in FIELDS}
            minimum,direct,wrong=(values[k]['auc'] for k in FIELDS)
            table.append(dict(half=half,dataset=name,minimum_auc=minimum,direct_auc=direct,wrong_auc=wrong,
                delta_direct=None if minimum is None else minimum-direct,
                delta_wrong=None if minimum is None else minimum-wrong,
                valid_episodes=values['minimum']['valid_episodes'],valid_classes=values['minimum']['valid_classes']))
    passed=all(row['delta_direct'] is not None and row['delta_direct']>=.01 for row in table)
    return table,passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    seal=json.loads((a.run/'sealed.json').read_text())
    config=json.loads((a.run/'config.json').read_text())
    if seal['state']!='ALL_CONTEXT_FIELDS_SEALED' or seal.get('query_labels_opened') is not False:
        raise ValueError('All context fields must be sealed before query labels')
    for name,key in (('config.json','config_sha256'),('manifest.json','manifest_sha256'),('inference.jsonl','inference_index_sha256')):
        if file_hash(a.run/name)!=seal[key]:raise ValueError('Sealed C2 run changed')
    rows=json.loads((a.run/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(a.run/'inference.jsonl').read_text().splitlines())}
    if (len(rows)!=6000 or set(index)!={r['episode_id'] for r in rows} or set(r['dataset'] for r in rows)!=set(DATASETS)
            or any(sum(r['dataset']==name for r in rows)!=2000 for name in DATASETS)):
        raise ValueError('Require the full original C2 minimum cohort')
    parent=Path(config['parent_run'])
    parent_seal=json.loads((parent/'sealed.json').read_text())
    if file_hash(parent/'sealed.json')!=config['parent_seal_sha256'] or file_hash(parent/'manifest.json')!=config['parent_manifest_sha256']:
        raise ValueError('Parent predictions/manifest changed')
    if file_hash(parent/'inference.jsonl')!=parent_seal['inference_index_sha256']:
        raise ValueError('Parent index changed')
    if json.loads((parent/'manifest.json').read_text())!=rows:
        raise ValueError('Context and parent use different episodes')
    parent_index={r['episode_id']:r for r in map(json.loads,(parent/'inference.jsonl').read_text().splitlines())}
    if set(parent_index)!=set(index):raise ValueError('Incomplete parent cohort')
    score_config=dict(run_seal_sha256=file_hash(a.run/'sealed.json'),scorer_sha256=file_hash(Path(__file__)),
                      parent_seal_sha256=config['parent_seal_sha256'],representation=config['choice'],
                      photo_macro_rules_sha256=file_hash(Path(__file__).with_name('score_representation_premise.py')))
    if (a.out/'config.json').exists() and json.loads((a.out/'config.json').read_text())!=score_config:
        raise ValueError('Score output belongs to different fields or scorer')
    if (a.out/'decision.json').exists():
        print('Existing fixed C2 minimum decision retained');return
    write(a.out/'config.json',score_config)
    records=[]
    for row in rows:
        item=index[row['episode_id']];baseline=parent_index[row['episode_id']]
        if file_hash(a.run/'fields'/item['field_file'])!=item['field_sha256'] or file_hash(parent/'predictions'/baseline['prediction_file'])!=baseline['prediction_sha256']:
            raise ValueError('Sealed field or parent mask changed')
        record=dict(episode_id=row['episode_id'],dataset=row['dataset'],fold=row['fold'],class_id=row['loader_class_id'],
                    photo_key=photo_key(row),half=photo_half(row),auc={})
        records.append(record)
        if not item['reference_valid']:continue
        with np.load(parent/'predictions'/baseline['prediction_file']) as saved:
            mask=np.unpackbits(saved['cli/rcg.fine'],count=1024**2).reshape(1024,1024)
        with np.load(a.run/'fields'/item['field_file'],allow_pickle=False) as saved:
            record['auc']=episode_auc(row,mask,saved)
    table,passed=summarize(records)
    with (a.out/'context.csv').open('w') as output:
        writer=csv.DictWriter(output,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (a.out/'episode_auc.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    decision=dict(state='COMPLETE',passed=passed,representation=config['choice'],datasets=table,
                  role='minimum context AUC prerequisite; does not admit a segmentation candidate',
                  gate='minimum - same-role direct >=0.01 on all three datasets and both fixed photo halves',
                  labels='GT nearest to1024 then area to64>=0.5; actual parent fine mask area to64>=0.5',
                  aggregation='paired episode AUC -> class mean -> fold mean; no two-label circle is N/A',
                  full_C2_implementation_allowed=passed)
    write(a.out/'decision.json',decision);print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
