"""Actual region-method additional reference crop; only stored masks/metadata."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from PIL import Image
import analyze as a

profiles={r['episode_id']:r for r in a.lines(a.OUT/'mask_profiles.jsonl')}
mapping=a.read(a.MECHANISM/'cache_research_map.json')
maprows={r['episode_id']:r for k in ('LVIS1400','PACO600') for r in mapping['datasets'][k]['records']}
out=[]
for cohort,names in [('LVIS600',('lvis_fixedwindows200_20261009','lvis_component_replication200_20261009','lvis_component_completion600_20261009')),
                    ('PACO600',('paco_fast9_600_20261009',))]:
    for name in names:
        folder=a.DATA/'a'/name
        infer={r['episode_id']:r for r in a.lines(folder/'inference.jsonl')}
        for row in a.read(folder/'manifest.json'):
            eid=row['episode_id'];box=infer[eid]['reference_box'];old=profiles[eid]['reference']
            stat={k:old[k] for k in ('fg_fraction','fg_mass_purity','mixed_fg_mass','tokens_ge90','coverage_max')}
            if box:
                mr=maprows[eid]['legal_reference_crop'];assert mr['box_xyxy']==box and mr['oracle_input'] is False
                p=Path(row['reference_mask_path']);a.receipts[str(p)]=a.sha(p)
                with Image.open(p) as im:arr=(np.asarray(im.convert('L'))>0).astype(np.uint8)
                assert a.array_sha(arr)==row['reference_mask_hash']
                x0,y0,x1,y1=box;h,w=arr.shape;assert 0<=x0<x1<=w and 0<=y0<y1<=h
                arr=arr[y0:y1,x0:x1];h,w=arr.shape
                canon=arr[(np.arange(1024)*h//1024)[:,None],(np.arange(1024)*w//1024)[None,:]]
                c=canon.reshape(64,16,64,16).mean(axis=(1,3)).ravel();mass=c.sum()
                stat=dict(fg_fraction=float(c.mean()),fg_mass_purity=float((c*c).sum()/mass),
                    mixed_fg_mass=float(c[(c>0)&(c<1)].sum()/mass),tokens_ge90=int((c>=.9).sum()),coverage_max=float(c.max()))
            out.append(dict(episode_id=eid,cohort=cohort,reference_box=box,original_reference={k:old[k] for k in stat},effective_region_reference=stat))

(a.OUT/'region_reference_crop_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out))
summary={}
main={r['episode_id'] for r in a.read(a.PILOT/'manifest.json')}
for cohort in ('LVIS600','PACO600'):
    rows=[r for r in out if r['cohort']==cohort];cropped=[r for r in rows if r['reference_box']]
    summary[cohort]=dict(n=len(rows),crop_n=len(cropped),main200_crop_n=sum(r['episode_id'] in main for r in cropped),
        all={field:{k:a.summary([r[field][k] for r in rows]) for k in rows[0][field]} for field in ('original_reference','effective_region_reference')},
        cropped_only={field:{k:a.summary([r[field][k] for r in cropped]) for k in cropped[0][field]} for field in ('original_reference','effective_region_reference')})
a.write(a.OUT/'region_reference_crop_summary.json',dict(summary=summary,source_sha256=a.receipts,script_sha256=a.sha(__file__),
    query_GT_reads=0,raw_payload_reads=0,new_encoder_calls=0,warning='These effective R views belong to existing region methods; current APD-ridge main200 uses whole R, including four PACO cases where region method additionally crops R.'))
print(json.dumps(summary),flush=True)
