#!/usr/bin/env python3
"""Expand previously frozen delete-p comparisons from sealed public blocks.

No encoder, feature restoration or search. Preserve naturally repeated draws.
Independently verify full RCG/MEAN/Astra masks against the existing annotations.
Native/delete-p arms retain their count-only provenance and missing diagnostics.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.experiment import sha,unpack,summarize
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-n',type=int,required=True)
    p.add_argument('--annotations',type=Path,default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    p.add_argument('--prior',type=Path,required=True,help='Previously verified1200episodes; fixes recipe/count parity')
    a=p.parse_args();torch.set_num_threads(1)
    if a.out.exists():raise FileExistsError('Fresh output required')
    banks=('astra','p','all','far')+tuple(f'top{f:g}' for f in (.02,.05,.1,.2))+('split:astra','split:p','split:top0.05','split:top0.1')+('p0.5','p2','p4')
    index=list(itertools.product(banks,(0,5),(-.1,-.05,-.02,0.,.02,.05))).index(('p',0,0.))
    prior=[json.loads(line) for line in a.prior.read_text().splitlines()]
    prior_by_identity={(r['c'],Path(r['support']).name,Path(r['query']).name):r for r in prior}
    arrays={};details=[];receipts=[];verified_prior=set()
    source_names={'RCG':'rcg','MEAN_CONTROL':'mean.control','external_mean__delete':'astra.control'}
    for block,run in enumerate(a.source):
        seal=json.loads((run/'sealed.json').read_text());rows=json.loads((run/'manifest.json').read_text())
        if seal['state']!='ALL_PREDICTIONS_SEALED' or sha(run/'manifest.json')!=seal['manifest_sha256']:
            raise ValueError('Unsealed or changed source')
        statistics=run/'sweep_counts_wide.npz';digest=sha(statistics)
        with np.load(statistics,allow_pickle=False) as z: counts={k:z[k].copy() for k in ['iu_native','iu_rcg','iu_mean','iu_astra','iu_c','d_rem','d_ctl']}
        if len(rows) not in (400,600) or counts['d_rem'].shape!=(180,len(rows),2):raise ValueError('Unexpected fixed library/cohort')
        count_arms={'native':counts['iu_native'],'rcg':counts['iu_rcg'],'mean.control':counts['iu_mean'],
                    'astra.control':counts['iu_astra'],'frozen.delete_p':counts['iu_c']-counts['d_rem'][index],
                    'frozen.delete_p.same_count.control':counts['iu_c']-counts['d_ctl'][index]}
        inputs={}
        for num,row in enumerate(rows):
            key=row['key'];prediction=run/'predictions'/(key+'.npz')
            if sha(prediction)!=seal['predictions'][key]:raise ValueError('Changed source mask '+key)
            gt=a.annotations/Path(row['query']).with_suffix('.png')
            with Image.open(gt) as image: truth=np.asarray(image)==row['c']+1
            truth=F.interpolate(torch.as_tensor(truth.copy())[None,None].float(),(1024,1024),mode='nearest')[0,0].bool().numpy()
            with np.load(prediction,allow_pickle=False) as z:
                for source,name in source_names.items():
                    mask=unpack(z[source]);iu=np.array([int((mask&truth).sum()),int((mask|truth).sum())])
                    if not np.array_equal(iu,count_arms[name][num]):raise ValueError('Mask/annotation/count parity: '+name+' '+key)
            identity=(row['c'],Path(row['support']).name,Path(row['query']).name)
            if identity in prior_by_identity:
                previous=prior_by_identity[identity]
                for name,values in count_arms.items():
                    if not np.array_equal(values[num],previous['iu'][name]):raise ValueError('Previously frozen1200 arm changed: '+name)
                verified_prior.add(identity)
            record=dict(row,key=f'public{block}:{key}',public_batch=block,iu={})
            for name,values in count_arms.items():
                iu=values[num].tolist();arrays.setdefault(name,[]).append(iu);record['iu'][name]=iu
            details.append(record);inputs[key]=dict(prediction_sha256=sha(prediction),annotation_sha256=sha(gt))
        if sha(statistics)!=digest:raise ValueError('Statistics changed during read')
        receipts.append(dict(run=str(run),seal_sha256=sha(run/'sealed.json'),sweep_sha256=digest,inputs=inputs))
        print(json.dumps(dict(completed_blocks=block+1,n=len(details))),flush=True)
    if len(details)!=a.expected_n or len(verified_prior)!=1200:raise ValueError('Incomplete target or prior parity')
    result,_=summarize(details,{name:np.array(values) for name,values in arrays.items()}, {})
    result.update(exposure='Public benchmark reuse including prior development; not independent confirmation',
        primary='frozen.delete_p',frozen_recipe=['del[p,w0,t0]','none'],parameter_updates=False,
        missing_fields={'native':'per-episode I/U from exporter/sweep; native full masks not independently replayed here',
                        'frozen.delete_p':'count-only arm; full-mask edits unavailable',
                        'frozen.delete_p.same_count.control':'count-only arm; full-mask edits unavailable',
                        'raw_DINO':'not available beyond original DEV accounting',
                        'INSID3':'complete implementation comparison remains separate DEV241'},
        prior1200_parity=len(verified_prior),prior_sha256=sha(a.prior),source_code_sha256=sha(__file__),
        draws_preserved=True)
    a.out.mkdir();(a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
    (a.out/'receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(dict(n=result['n'],scores=result['scores'],primary=result['contrasts']['frozen.delete_p'])),flush=True)


if __name__=='__main__':main()
