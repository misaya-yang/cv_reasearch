import argparse,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('results/pilot_v1'));a=p.parse_args();reports={}
for f in a.root.glob('*/report.json'):
 d=json.loads(f.read_text())
 if d.get('phase')=='train':reports[d['mode']]=d
if 'original' not in reports:raise SystemExit('Original matched-training report missing')
def values(d,t):return {r['image_id']:r['velocity_mse'] for r in d['heldout_paired_scores'] if r['t']==t}
rows=[]
for name,d in reports.items():
 for t in (.2,.5,.8):
  ref=values(reports['original'],t);cur=values(d,t)
  if ref.keys()!=cur.keys():raise ValueError('Image pairing mismatch')
  ids=sorted(ref);diff=np.array([cur[i]-ref[i] for i in ids]);g=np.random.default_rng(2027);boot=diff[g.integers(len(ids),size=(2000,len(ids)))].mean(1)
  rows.append({'mode':name,'t':t,'tau':(1-t)/t,'images':len(ids),'paired_velocity_mse_delta':float(diff.mean()),'image_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'mean_velocity_mse':float(np.mean([cur[i] for i in ids]))})
out={'status':'SHORT_PILOT_ONLY','training_seeds':1,'rows':rows,'interpretation':'Negative delta favors branch. Image-level paired intervals do not cover training-seed uncertainty. Upsampled COCO diagnostic; no native-1024/SOTA/FID claim.'};(a.root/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
