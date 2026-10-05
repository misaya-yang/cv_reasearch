#!/usr/bin/env python3
"""Reproduce candidate statistics and influence from bundled saved integer I/U."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from statistics_randomstate import analyze, episode_key, photograph_components
CAND='RCG_count_matched_delete'
def sensitivity(rows,baseline):
 rows=sorted(rows,key=lambda r:(r['fold'],r['e'],r['c']));classes=sorted({int(r['c'])for r in rows});cid={c:i for i,c in enumerate(classes)};ids=np.array([cid[int(r['c'])]for r in rows]);X=np.array([[r['iu'][a]for a in [CAND,baseline]]for r in rows],float);N=len(rows);C=len(classes)
 def evaluate(keep):
  totals=np.stack([np.stack([np.bincount(ids,weights=X[:,a,j]*keep,minlength=C)for j in [0,1]],axis=1)for a in [0,1]],axis=1);present=totals[:,0,1]>0;assert np.array_equal(present,totals[:,1,1]>0);rat=100*totals[present,:,0]/totals[present,:,1];return float((rat[:,0]-rat[:,1]).mean()),int(present.sum())
 total=np.zeros((C,2,2));np.add.at(total,ids,X);delta=100*(total[:,0,0]/total[:,0,1]-total[:,1,0]/total[:,1,1]);gain=float(delta.mean());classes_out=[]
 for i,c in enumerate(classes):
  lo=float((delta.sum()-delta[i])/(C-1));classes_out.append(dict(class_id=c,episodes=int((ids==i).sum()),candidate_IU=total[i,0].astype(int).tolist(),baseline_IU=total[i,1].astype(int).tolist(),class_gain_pp=float(delta[i]),contribution_to_full_gain_pp=float(delta[i]/C),gain_without_class_pp=lo))
 positive=sorted(classes_out,key=lambda r:r['class_gain_pp'],reverse=True);negative=sorted(classes_out,key=lambda r:r['class_gain_pp']);pos=delta[delta>0];class_remove={}
 for k in [1,3,5]:
  removed=[r['class_id']for r in positive[:k]];keep=np.array([r['c']not in removed for r in rows]);g,nc=evaluate(keep);class_remove[str(k)]=dict(removed_classes=removed,gain_pp=g,remaining_classes=nc,remaining_episodes=int(keep.sum()),positive_gain_mass_fraction=float(sum(max(r['class_gain_pp'],0)for r in positive[:k])/max(float(pos.sum()),1e-12)))
 per=[]
 for i,r in enumerate(rows):
  keep=np.ones(N);keep[i]=0;g,nc=evaluate(keep);per.append(dict(key=episode_key(r),class_id=int(r['c']),fold=int(r['fold']),without_gain_pp=g,influence_pp=gain-g,without_classes=nc,candidate_episode_IoU_pp=float(100*X[i,0,0]/X[i,0,1]),baseline_episode_IoU_pp=float(100*X[i,1,0]/X[i,1,1])))
 byinf=sorted(per,key=lambda r:r['influence_pp'],reverse=True);episode_remove={}
 for k in [1,3,5]:
  keys={r['key']for r in byinf[:k]};keep=np.array([episode_key(r)not in keys for r in rows]);g,nc=evaluate(keep);episode_remove[str(k)]=dict(removed_keys=sorted(keys),gain_pp=g,remaining_classes=nc,remaining_episodes=int(keep.sum()))
 pg=[]
 for group in photograph_components(rows):
  keep=np.ones(N);keep[group]=0;g,nc=evaluate(keep);pg.append(dict(keys=[episode_key(rows[i])for i in group],without_gain_pp=g,influence_pp=gain-g,without_classes=nc))
 count=lambda x:[int((x>1e-10).sum()),int((x< -1e-10).sum()),int((abs(x)<=1e-10).sum())]
 return dict(contrast=CAND+' minus '+baseline,gain_pp=gain,classes_up_down_tie=count(delta),positive_class_gain_sum_pp=float(pos.sum()),negative_class_gain_sum_pp=float(delta[delta<0].sum()),effective_positive_class_count=float(pos.sum()**2/np.sum(pos**2))if len(pos) else 0,leave_one_class_out_range_pp=[min(r['gain_without_class_pp']for r in classes_out),max(r['gain_without_class_pp']for r in classes_out)],leave_one_episode_out_range_pp=[min(r['without_gain_pp']for r in per),max(r['without_gain_pp']for r in per)],leave_one_photo_group_out_range_pp=[min(r['without_gain_pp']for r in pg),max(r['without_gain_pp']for r in pg)],remove_top_positive_classes=class_remove,remove_top_positive_influence_episodes=episode_remove,largest_benefit_classes=positive[:10],largest_harm_classes=negative[:10],largest_positive_influence_episodes=byinf[:15],largest_negative_influence_episodes=sorted(per,key=lambda r:r['influence_pp'])[:10],largest_positive_influence_photo_groups=sorted(pg,key=lambda r:r['influence_pp'],reverse=True)[:10],all_class_contributions=classes_out,note='GT-derived diagnostic stress tests, not querycase gates, parameter selection or newvalidation. Classremoval removeswholeclass; episode/groupremoval reaggregates allclassI/U and drops emptyclasses. Topk episode rankings fixedonce from leave-one-out fullcohort influence, not greedy optimization.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=Path,default=Path(__file__).parent/'evidence/scored220_compact.jsonl');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Refuse to overwrite an existing analysis')
 rows=[json.loads(line) for line in a.episodes.read_text().splitlines() if line.strip()]
 controls=['native_cached_CRF','RCG','MEAN_CONTROL','crossfit_fraction_RCG_rank','global_RCG_same_budget']
 groups={'combined220':rows}
 for batch in sorted({r['batch'] for r in rows}):groups[batch]=[r for r in rows if r['batch']==batch]
 result={'scope':'Exposed DEV; fixed exploratory candidate. Influence exclusions are diagnostics, never inference filters.','input_sha256':hashlib.sha256(a.episodes.read_bytes()).hexdigest(),'statistics':{},'sensitivity':{}}
 for label,rr in groups.items():
  result['statistics'][label]=analyze(rr,baseline='native_cached_CRF',contrasts=[(CAND,b) for b in controls if b!='native_cached_CRF'])
  result['sensitivity'][label]={b:sensitivity(rr,b) for b in controls}
 a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result['statistics']['combined220']['point_estimates_pp']))
if __name__=='__main__':main()
