#!/usr/bin/env python3
"""Portable class-macro IoU bootstrap over connected photograph groups.
Requires Python3 and NumPy only. No images, ground truth, features, or model code.

Example:
 python photo_group_bootstrap.py --root /path/to/run \
   --episodes episodes.jsonl --manifest manifest.json --baseline native_pre \
   --contrast mixture:native_pre --output statistics.json --verify

Repeat --episodes to pool disjoint episode cohorts. Arms default to the
intersection of available arm names across all rows; --arms chooses explicitly.
Each episode needs fold,e,c,support,query and field (default 'iu') mapping arm
names to [intersection,union]. Metadata may be supplied by JSON/JSONL/CSV
manifest(s). Relative input/output paths resolve under --root, default cwd.
"""
from pathlib import Path
import argparse,collections,csv,hashlib,json
import numpy as np

def read_rows(path):
 p=Path(path)
 if p.suffix.lower()=='.csv':return list(csv.DictReader(p.open()))
 if p.suffix.lower()=='.jsonl':return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
 d=json.loads(p.read_text())
 if isinstance(d,list):return d
 for k in ['episodes','records','rows']:
  if isinstance(d.get(k),list):return d[k]
 raise ValueError('Expected JSON list or episodes/records/rows list: '+str(p))

def episode_key(r):return '%d_%d_%d'%(int(r['fold']),int(r['e']),int(r['c']))

def photograph_components(rows):
 n=len(rows);parent=list(range(n));seen={}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for i,r in enumerate(rows):
  for role in ['support','query']:
   photo=Path(str(r[role])).name
   if photo in seen:parent[find(i)]=find(seen[photo])
   else:seen[photo]=i
 groups=collections.defaultdict(list)
 for i in range(n):groups[find(i)].append(i)
 return sorted(groups.values(),key=lambda ids:ids[0])

def analyze(rows,field='iu',baseline='native_cached_CRF',arms=None,contrasts=(),draws=2000,seed=0,verify=False):
 if not rows:raise ValueError('No episodes')
 rows=sorted(rows,key=lambda r:(int(r['fold']),int(r['e']),int(r['c'])))
 keys=[episode_key(r) for r in rows]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate episode key in pooled input; resolve before pooling')
 for r in rows:
  for k in ['support','query','fold','e','c']:assert k in r,(episode_key(r),k)
 common=set.intersection(*(set(r[field]) for r in rows))
 if arms is None:arms=[a for a in rows[0][field] if a in common]
 if baseline not in arms:raise ValueError('Baseline absent from common requested arms')
 if not set(arms)<=common:raise ValueError('An arm is missing from at least one episode')
 classes=sorted({int(r['c']) for r in rows});class_id={c:i for i,c in enumerate(classes)}
 n=len(rows);A=len(arms);C=len(classes);groups=photograph_components(rows);G=len(groups)
 x=np.asarray([[r[field][a] for a in arms] for r in rows],dtype=np.float64)
 if x.shape!=(n,A,2) or not np.isfinite(x).all() or not (x[:,:,1]>0).all() or not (x[:,:,0]>=0).all() or not (x[:,:,0]<=x[:,:,1]+1e-7).all():raise ValueError('Invalid I/U counts')
 group_iu=np.zeros((G,C,A,2))
 for g,ids in enumerate(groups):
  for i in ids:group_iu[g,class_id[int(rows[i]['c'])]]+=x[i]
 full=group_iu.sum(0);class_iou=100*full[:,:,0]/full[:,:,1];point=class_iou.mean(0)
 rng=np.random.RandomState(seed);indices=rng.randint(G,size=(draws,G))
 multiplicity=np.stack([np.bincount(z,minlength=G) for z in indices])
 sampled=(multiplicity@group_iu.reshape(G,-1)).reshape(draws,C,A,2)
 present=sampled[:,:,0,1]>0
 if not np.array_equal(sampled[:,:,:,1]>0,np.broadcast_to(present[:,:,None],sampled[:,:,:,1].shape)):raise ValueError('Class presence differs by arm')
 ratios=np.divide(100*sampled[:,:,:,0],sampled[:,:,:,1],out=np.zeros((draws,C,A)),where=sampled[:,:,:,1]>0)
 replicate=ratios.sum(1)/present.sum(1)[:,None]
 verification=None
 if verify:
  # Independent literal expansion: append actual repeated episodes, then group by class.
  direct=[]
  for selected in indices:
   by_class=collections.defaultdict(list)
   for g in selected:
    for i in groups[g]:by_class[int(rows[i]['c'])].append(i)
   vals=[]
   for c,ids in by_class.items():
    total=np.sum(x[ids],axis=0);vals.append(100*total[:,0]/total[:,1])
   direct.append(np.mean(vals,axis=0))
  discrepancy=float(np.max(np.abs(replicate-np.asarray(direct))))
  if discrepancy>1e-10:raise AssertionError('Vectorized/literal bootstrap mismatch: '+str(discrepancy))
  verification=dict(all_draws_independently_expanded=draws,max_score_difference_pp=discrepancy)
 pairs=list(dict.fromkeys([(a,baseline) for a in arms if a!=baseline]+[tuple(p) for p in contrasts]))
 comparisons={}
 for a,b in pairs:
  ia=arms.index(a);ib=arms.index(b);delta=class_iou[:,ia]-class_iou[:,ib];episode_delta=100*(x[:,ia,0]/x[:,ia,1]-x[:,ib,0]/x[:,ib,1]);ci=np.quantile(replicate[:,ia]-replicate[:,ib],[.025,.975]).tolist();folds={}
  for f in sorted({int(r['fold']) for r in rows}):
   cf={int(r['c']) for r in rows if int(r['fold'])==f};folds[str(f)]=float(np.mean([delta[class_id[c]] for c in sorted(cf)]))
  comparisons[a+'__minus__'+b]=dict(gain_pp=float(point[ia]-point[ib]),ci95_pp=ci,per_fold_gain_pp=folds,episodes_up=int((episode_delta>1e-10).sum()),episodes_down=int((episode_delta<-1e-10).sum()),episodes_tie=int((abs(episode_delta)<=1e-10).sum()),ci_relation='above_zero' if ci[0]>0 else 'below_zero' if ci[1]<0 else 'spans_zero')
 return dict(episodes=n,classes=C,photograph_groups=G,keys=keys,group_episode_keys=[[keys[i] for i in g] for g in groups],point_estimates_pp=dict(zip(arms,map(float,point))),comparisons=comparisons,per_class={str(c):{a:dict(I=float(full[class_id[c],j,0]),U=float(full[class_id[c],j,1]),IoU_pp=float(class_iou[class_id[c],j])) for j,a in enumerate(arms)} for c in classes},omitted_noncommon_arms=sorted(set.union(*(set(r[field]) for r in rows))-set(arms)),bootstrap_represented_classes=dict(minimum=int(present.sum(1).min()),mean=float(present.sum(1).mean()),maximum=int(present.sum(1).max())),independent_verification=verification)

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--episodes',action='append',required=True);p.add_argument('--manifest',action='append',default=[]);p.add_argument('--field',default='iu');p.add_argument('--baseline',default='native_cached_CRF');p.add_argument('--arms',nargs='+');p.add_argument('--contrast',action='append',default=[],help='candidate:control');p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=0);p.add_argument('--verify',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 resolve=lambda x:Path(x) if Path(x).is_absolute() else a.root/Path(x)
 meta={};source=[]
 for path in a.manifest:
  pp=resolve(path);source.append(pp)
  for r in read_rows(pp):
   k=episode_key(r)
   if k in meta:
    for f in ['support','query','fold','e','c']:assert str(meta[k][f])==str(r[f]),('Conflicting manifest metadata',k,f)
   meta[k]=r
 rows=[]
 for path in a.episodes:
  pp=resolve(path);source.append(pp)
  for r in read_rows(pp):
   m=meta.get(episode_key(r),{})
   for f in ['support','query','fold','e','c']:
    if f in m and f in r:assert str(m[f])==str(r[f]),('Episode/manifest mismatch',episode_key(r),f)
   rows.append(dict(m,**r))
 contrasts=[x.split(':') for x in a.contrast]
 if any(len(x)!=2 for x in contrasts):p.error('--contrast requires candidate:control')
 stats=analyze(rows,a.field,a.baseline,a.arms,contrasts,a.draws,a.seed,a.verify)
 result=dict(protocol='Photograph-connected EPISODE components; sample G groups G times with replacement; reaggregate weighted I/U within each represented class, then macro mean. Paired draws across arms.',missing_class_rule='Omit classes absent from that replicate. Do not set to zero, retain original score, or resample class groups.',seed=a.seed,bootstrap_draws=a.draws,metric_field=a.field,source_sha256={str(pp):hashlib.sha256(pp.read_bytes()).hexdigest() for pp in source},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),statistics=stats)
 out=resolve(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print('Saved',out,'episodes',stats['episodes'],'classes',stats['classes'],'photograph groups',stats['photograph_groups'])
 for name,r in stats['comparisons'].items():print(name,r['gain_pp'],r['ci95_pp'])
if __name__=='__main__':main()
