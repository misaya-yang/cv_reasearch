#!/usr/bin/env python3
"""Encode only uncached canonical episode pairs; query truth never read."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 import numpy as np
 import torch
 import torch.nn.functional as F
 from PIL import Image
 sys.path.insert(0,'/root/autodl-tmp/demo9_extent');sys.path.insert(0,'/root/autodl-tmp/demo9_extent/scripts')
 from extent_experiment import build_host,run_foris
 man=json.loads(a.manifest.read_text());rows=man['episodes']
 if a.out.exists():raise FileExistsError('Fresh gap directory required')
 a.out.mkdir(parents=True);begin=time.monotonic();records={}
 if rows:
  with torch.inference_mode():
   host=build_host(SimpleNamespace(fixture=None,foris_root=None,demo4_root='/root/autodl-tmp/demo4'),man,'cuda');torch.set_num_threads(1)
   for n,row in enumerate(rows,1):
    data,ann=Path(man['data_root']),Path(man['annotation_root'])
    with Image.open(data/row['support']) as im:support=im.convert('RGB')
    with Image.open(data/row['query']) as im:query=im.convert('RGB')
    with Image.open(ann/Path(row['support']).with_suffix('.png')) as im:reference=torch.from_numpy((np.asarray(im)==row['c']+1).copy())
    native,got,mask,_=run_foris(host,support,reference,query)
    with np.load(row['packet_export'],allow_pickle=False) as z:stored={name:z[name].copy() for name in ['native','score','cov']}
    covariance=F.interpolate(mask[None,None].float(),(64,64),mode='area')[0,0]
    differing=int(np.unpackbits(np.packbits(native.cpu().numpy())^stored['native']).sum())
    score_difference=float(np.abs(got['score'].float().cpu().numpy()-stored['score']).max());covariance_difference=float(np.abs(covariance.cpu().numpy()-stored['cov']).max())
    if differing>100 or score_difference>1e-6 or covariance_difference:raise ValueError('Canonical gap replay differs: '+row['key'])
    deb=got['deb'][0];raw=F.normalize(got['raw'][0],dim=1)
    feature=dict(q=deb[-1].flatten(1).T.half().cpu(),r=deb[0].flatten(1).T.half().cpu(),debiased=bool((raw-deb).abs().max()>1e-4))
    torch.save(feature,a.out/(row['key']+'.pt'))
    records[row['key']]=dict(native_pixels=differing,score_max=score_difference,cov_max=covariance_difference)
    print(json.dumps(dict(n=n,total=len(rows),seconds=round(time.monotonic()-begin,1))),flush=True)
 report=dict(state='COMPLETED',n=len(rows),records=records,seconds=time.monotonic()-begin,query_labels_read=False,role='necessary input extraction for frozen1200evaluation; no old600restoration')
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(state=report['state'],n=report['n'],seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
