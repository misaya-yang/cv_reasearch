"""Fixed P3 snapshot: prototype/vote 2x2 on a new image-disjoint cohort."""
import os,sys,json,hashlib,time,argparse,importlib.util,shutil
from pathlib import Path
from open_pool_eval import prepare,score,atomic
from icx.common import *
from tics.propagate import propagate,reliability,choose
from utils.data import load_mask
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vote_separated',ROOT/'isolated_copy/demo_lists/demo4_incontext_seg/icx/fast.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main(args):
 out=Path(args.out);doc=json.load(open(out/'manifest.json'));cache=out/'cache';torch.backends.cuda.matmul.allow_tf32=False
 state=dict(state='RUNNING',pid=os.getpid(),completed=0,total=len(doc['episodes']));atomic(out/'eval_status.json',state);records=[]
 for row in doc['episodes']:
  path=out/f"episode{row['episode']:04d}.json"
  if path.exists():records.append(json.load(open(path)));state['completed']+=1;continue
  predictions={};details={};scoring_sets={}
  for condition in ['clean','mixed']:
   s,d,g0=prepare(row,condition,doc,cache);J=list(range(1,s.n));cs=module.ClassSet(d)
   assert torch.equal(s.fd,cs.fd), 'Reader feature mismatch'
   # All four variants get precisely the same independently-generated P3.
   snapshots=propagate(s,J,rounds=3,k=5,trust='agree',rule='tophalf')
   P=snapshots[-1];ag=reliability(s,P,J,'agree');allowed=choose(ag,'tophalf')
   cs._nn=s._nn
   donors=[j for j in J if j!=1];refs=[0]+donors;masks=[P[0]]+[P[j] for j in donors];weights=[1.]+[ag[j] for j in donors]
   if not np.isfinite(weights).all() or min(weights)<0:raise ValueError('Invalid shared agreement vector')
   key=lambda arm:condition+'_'+arm
   predictions[key('1shot')]=snapshots[0][1]
   for arm,wp,wv in [('none',None,None),('prototype_only',weights,None),('vote_only',None,weights),('both',weights,weights)]:
    result=cs.predict(1,refs,masks,weights=wp,vote_weights=wv,backward='pooled',k=5,bonus0=0.,return_parts=True)
    pred,parts=result if isinstance(result,tuple) else (result,None);predictions[key(arm)]=pred
    details[key(arm)]=dict(candidate_pixels=int(parts['cand'].sum()) if parts else None)
    if arm=='none':none_parts=parts
    if parts and none_parts:
     if arm=='prototype_only':assert torch.equal(parts['votes'],none_parts['votes']), 'Prototype intervention changed backward votes'
     if arm=='vote_only':assert torch.equal(parts['sim_fwd'],none_parts['sim_fwd']), 'Vote intervention changed prototype similarity'
   selected=[j for j in donors if allowed[j] and P[j].any()]
   predictions[key('tophalf_control')]=s.predict(1,[0]+selected,[P[0]]+[P[j] for j in selected],backward='pooled',k=5)
   # Old no-opt-in path remains equal to ImageSet for these identical inputs.
   old_none=s.predict(1,refs,masks,backward='pooled',k=5)
   assert torch.equal(old_none,predictions[key('none')]), 'Opt-in patch changed unchanged reader path'
   details[condition+'_snapshot']=dict(round=3,refs=[s.names[j] for j in refs],agreement=weights,
    mask_sha256=[hashlib.sha256(np.packbits(m.cpu().numpy()).tobytes()).hexdigest() for m in masks],
    no_branch_feedback=True,reference_selection_same=True,appearance_topk_unweighted=True)
   scoring_sets[condition]=s
  # Query truth accessed only after both conditions/all branches finish.
  _,qm=coco_load(doc['data'],row['query'],row['class_id']);truth=load_mask(qm,1024,DEV)[0]
  metrics={name:score(p,truth,scoring_sets[name.split('_')[0]]) for name,p in predictions.items()}
  r=dict(episode=row['episode'],class_id=row['class_id'],metrics=metrics,details=details);atomic(path,r);records.append(r)
  state['completed']+=1;atomic(out/'eval_status.json',state)
  del scoring_sets,predictions,s,cs;torch.cuda.empty_cache()
  if state['completed']%10==0:print(json.dumps(state),flush=True)
 sums={}
 for r in records:
  for name,v in r['metrics'].items():
   x=sums.setdefault(name,{}).setdefault(str(r['class_id']),[0.,0.]);x[0]+=v['intersection'];x[1]+=v['union']
 names=list(sums);classes=sorted(sums[names[0]],key=int);V=np.array([[100*sums[k][c][0]/max(sums[k][c][1],1) for k in names] for c in classes])
 rng=np.random.default_rng(2053);ix=rng.integers(len(V),size=(5000,len(V)));contrasts={}
 for condition in ['clean','mixed']:
  for a,b in [('prototype_only','none'),('vote_only','none'),('both','none'),('vote_only','prototype_only'),('vote_only','tophalf_control')]:
   delta=V[:,names.index(condition+'_'+a)]-V[:,names.index(condition+'_'+b)]
   contrasts[condition+'_'+a+' minus '+b]=dict(mean_pp=float(delta.mean()),paired_class_ci95=np.percentile(delta[ix].mean(1),[2.5,97.5]).tolist())
 report=dict(state='COMPLETED',episodes=len(records),classes=len(classes),class_sums=sums,miou={k:float(V[:,i].mean()) for i,k in enumerate(names)},contrasts=contrasts,records=records,
  scope='One frozen P3 snapshot per condition, same refs/masks/raw agreement/appearance topk in four arms. No feedback between branches. Identifies last-readout paths, not full weighted iterative method or publication success. New seed2052 images disjoint from preceding open-pool study, not all historical repository work; only20 classes/fold0.')
 atomic(out/'report.json',report);state['state']='COMPLETED';atomic(out/'eval_status.json',state);print(json.dumps(report['miou']),flush=True)
 size=sum(p.stat().st_size for p in cache.glob('*.pt'));free=shutil.disk_usage(out).free;shutil.rmtree(cache)
 atomic(out/'cache_cleanup_receipt.json',dict(reason='Complete matched snapshot task results; preserve small masks/weights/manifest only',feature_bytes_removed=size,free_before=free,free_after=shutil.disk_usage(out).free))
 cs=json.load(open(out/'cache_status.json'));cs['state']='RETIRED_FEATURES_REMOVED';atomic(out/'cache_status.json',cs)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  import traceback
  atomic(Path(a.out)/'eval_status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
