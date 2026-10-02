"""Only the missing matched-positive control; pin active features by hardlink.

Does not restart or edit the ongoing main experiment, nor re-encode images.
"""
import argparse,os,json,time,hashlib,shutil,sys
from pathlib import Path
from open_pool_eval import prepare,score,atomic
from icx.common import *
from utils.data import load_mask
from tics.propagate import one_shot,propagate

def main(out):
 out=Path(out);doc=json.load(open(out/'manifest.json'));cache=out/'cache';pin=out/'matched_control_pin';pin.mkdir(exist_ok=True)
 names={n for r in doc['episodes'] for n in [r['support'],r['query']]+[x for x in r['mixed'] if x not in r['negative']]}
 for n in names:
  name=hashlib.sha256(n.encode()).hexdigest()[:20]+'.pt';dest=pin/name
  if not dest.exists():os.link(cache/name,dest)
 torch.backends.cuda.matmul.allow_tf32=False;records=[]
 state=dict(state='RUNNING',pid=os.getpid(),completed=0,total=len(doc['episodes']),scope='Matched positive identities control only, no feature/other-arm recomputation')
 atomic(out/'matched_status.json',state)
 for row in doc['episodes']:
  path=out/f"matched_episode{row['episode']:04d}.json"
  if path.exists():records.append(json.load(open(path)));state['completed']+=1;continue
  s,d,g0=prepare(row,'present_matched',doc,pin);J=list(range(1,s.n));P1=one_shot(s,J);others=[j for j in J if j!=1]
  predictions={'present_matched_naive':s.predict(1,[0]+others,[P1[0]]+[P1[j] for j in others])}
  for trust,rule in [('none','all'),('agree','tophalf')]:predictions['present_matched_'+trust]=propagate(s,J,rounds=4,k=5,trust=trust,rule=rule,P1=P1)[-1][1]
  # The method completed before querying the evaluation annotation.
  _,qm=coco_load(doc['data'],row['query'],row['class_id']);truth=load_mask(qm,1024,DEV)[0]
  r=dict(episode=row['episode'],class_id=row['class_id'],metrics={k:score(m,truth,s) for k,m in predictions.items()},positive_donors=[n for n in row['mixed'] if n not in row['negative']])
  atomic(path,r);records.append(r);state['completed']+=1;atomic(out/'matched_status.json',state)
  del s,d;torch.cuda.empty_cache()
  if state['completed']%10==0:print(json.dumps(state),flush=True)
 sums={}
 for r in records:
  for k,v in r['metrics'].items():
   x=sums.setdefault(k,{}).setdefault(str(r['class_id']),[0.,0.]);x[0]+=v['intersection'];x[1]+=v['union']
 report=dict(state='COMPLETED',fold=doc['fold'],episodes=len(records),class_sums=sums,records=records,miou={k:100*float(np.mean([i/max(u,1) for i,u in cc.values()])) for k,cc in sums.items()},scope='Absent labels used only to construct this environmental control, never provided to propagation; not oracle-filtering deployment method')
 atomic(out/'matched_control.json',report);state['state']='COMPLETED';atomic(out/'matched_status.json',state)
 size=sum(p.stat().st_size for p in pin.glob('*.pt'));free=shutil.disk_usage(out).free;shutil.rmtree(pin)
 atomic(out/'matched_pin_cleanup.json',dict(released_linked_feature_bytes=size,free_before=free,free_after=shutil.disk_usage(out).free,reason='Completed extra control; original main-cache may still be active, so unlink frees payload only when no other link remains'))
 print(json.dumps(report['miou']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
 try:main(a.out)
 except Exception:
  import traceback
  atomic(Path(a.out)/'matched_status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
