"""New seed/images for the provided frozen 2x2 weight-path experiment."""
import os,sys,json,argparse
from pathlib import Path
sys.path[:0]=['/root/demo4_cache/env','/root/autodl-tmp/demo4']
from icx.common import *

def main(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():print('Existing frozen manifest reused');return
 excluded=set()
 for f in range(4):excluded.update(json.load(open(Path(args.previous)/f'f{f}/manifest.json'))['names'])
 fold=0;base=f'{CACHE}/data/COCO2014';meta=pickle.load(open(f'{base}/splits/val/fold{fold}.pkl','rb'));ids=[4*i for i in range(20)]
 rng=np.random.default_rng(2052);galleries={};supports={};rows=[]
 for c in ids:
  available=sorted(set(map(str,meta[c]))-excluded)
  if len(available)<7:raise RuntimeError(f'Class{c} has too few unseen images; do not silently drop classes')
  count=min(20,len(available)-1);gallery=list(rng.choice(available,count,replace=False));galleries[c]=gallery
  remainder=sorted(set(available)-set(gallery));supports[c]=list(rng.choice(remainder,5,replace=len(remainder)<5))
 all_gallery=sorted({n for vv in galleries.values() for n in vv})
 for c in ids:
  for i,q in enumerate(galleries[c][:5]):
   support=supports[c][i];pool=list(rng.permutation([x for x in galleries[c] if x!=q])[:15]);k=len(pool)//2
   negative_candidates=[x for x in all_gallery if x not in set(map(str,meta[c])) and x not in (support,q) and x not in pool]
   negative=list(rng.choice(negative_candidates,k,replace=False));mixed=pool[:len(pool)-k]+negative
   rows.append(dict(episode=len(rows),class_id=c,support=support,query=q,clean=pool,mixed=mixed,negative=negative))
 names=sorted({n for r in rows for n in [r['support'],r['query']]+r['clean']+r['mixed']})
 assert not set(names)&excluded
 doc=dict(fold=0,seed=2052,data=base,episodes=rows,names=names,excluded_prior_openpool_images=len(excluded),
  protocol='100 queries, five/class. All actual encoder image identities disjoint from the 400-query open_pool_v1 study. Not claimed disjoint from every historical project experiment. New seed/custom gallery, not the exact standard seed0 episode list. All query/donor labels hidden in inference.',
  experiment='All four weight paths use one fixed P3 snapshot and raw agreement vector per condition; weights cannot change reference identities/appearance top-k. Separate top-half practical control. GT query only after inference.',
  feature_contract='Native BF16 encoder; common FP32 post-encoder readout like prior compact demo9, compressed FP16 complement coordinates. No new weights/data.')
 (out/'manifest.json').write_text(json.dumps(doc,indent=2));print(json.dumps(dict(images=len(names),episodes=len(rows),excluded=len(excluded))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--previous',default='results/open_pool_v1');p.add_argument('--out',required=True);a=p.parse_args();main(a)
