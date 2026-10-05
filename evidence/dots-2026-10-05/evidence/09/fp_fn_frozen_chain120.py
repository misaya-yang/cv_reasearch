#!/usr/bin/env python3
"""Read existing old120 masks, measure corrections/damage; no new prediction."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=json.load(open(a.root/'cpu_method/cohort_inputs/index120.json'))['episodes'];assert len(rows)==120
names=['mean_FG_BG','insid3_cached','foris_pre','foris_native'];pairs=[('mean_FG_BG','insid3_cached'),('insid3_cached','foris_pre'),('insid3_cached','foris_native'),('foris_pre','foris_native')];records=[]
for r in sorted(rows,key=lambda x:(x['fold'],x['e'],x['c'])):
 k='%d_%d_%d'%(r['fold'],r['e'],r['c'])
 with np.load(a.root/'cpu_method/object_set120/frozen'/(k+'.npz'),allow_pickle=False) as z:
  hw=tuple(z['work_hw']);m={'mean_FG_BG':np.unpackbits(z['reference_mean']).reshape(hw).astype(bool),'insid3_cached':np.unpackbits(z['insid3_cached']).reshape(hw).astype(bool)}
 with np.load(r['packet_path'],allow_pickle=False) as z:
  y=np.unpackbits(z['truth']).reshape(hw).astype(bool);m['foris_pre']=np.unpackbits(z['pre']).reshape(hw).astype(bool);m['foris_native']=np.unpackbits(z['native']).reshape(hw).astype(bool)
 rec={x:r[x] for x in ['fold','e','c','support','query','cohort']};rec.update(key=k,iu={n:[int((z&y).sum()),int((z|y).sum())] for n,z in m.items()},edits={})
 for old,new in pairs:
  add=m[new]&~m[old];delete=m[old]&~m[new]
  rec['edits'][old+'__to__'+new]=dict(add_TP=int((add&y).sum()),add_FP=int((add&~y).sum()),delete_TP=int((delete&y).sum()),delete_FP=int((delete&~y).sum()))
 records.append(rec)
report={'scope':'Old120 exposed development; frozen 1024 masks only. mean_FG_BG is raw source-mean FG-BG zero cut, token binary then bilinear. cached INSID3 is one-space adaptation, not published dual-space pipeline. All changes include final normalization/renderer. No module-causal attribution.','cohorts':{},'episodes':records}
for co in ['old20','new40','new60','pooled120']:
 rs=[r for r in records if co=='pooled120' or r['cohort']==co];cs=sorted({r['c'] for r in rs});points={}
 for n in names:
  total=np.asarray([np.sum([r['iu'][n] for r in rs if r['c']==c],axis=0) for c in cs]);points[n]=float(100*np.mean(total[:,0]/total[:,1]))
 edits={}
 for old,new in pairs:
  link=old+'__to__'+new;v={x:sum(r['edits'][link][x] for r in rs) for x in ['add_TP','add_FP','delete_TP','delete_FP']};v.update(net_TP=v['add_TP']-v['delete_TP'],net_FP=v['add_FP']-v['delete_FP'],corrected=v['add_TP']+v['delete_FP'],damaged=v['add_FP']+v['delete_TP']);edits[link]=v
 report['cohorts'][co]={'episodes':len(rs),'classes':len(cs),'class_mIoU':points,'raw_pixel_edits':edits}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report['cohorts']['pooled120'],indent=2))
