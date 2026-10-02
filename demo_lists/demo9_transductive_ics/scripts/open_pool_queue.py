"""Sequential four-fold cache/eval/cleanup; resumes completed scientific stages."""
import os,sys,json,time,subprocess,shutil,argparse
from pathlib import Path

def atomic(path,d):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(path)

def read(p):return json.load(open(p)) if p.exists() else {}

def main(args):
 root=Path(args.root);root.mkdir(parents=True,exist_ok=True);state=dict(state='RUNNING',pid=os.getpid(),fold=0,stage='prepare')
 env=dict(os.environ,DEMO4_GPU_FRAC='0.55',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 def execute(script,fold,stage):
  logfile=open(root/f'f{fold}_{stage}.log','a');cmd=[sys.executable,'-u',f'scripts/{script}','--out',str(root/f'f{fold}')]
  if stage=='cache':cmd+=['--fold',str(fold)]
  if stage=='eval' and fold>=1:cmd+=['--matched-positives']
  child=subprocess.Popen(cmd,env=env,stdout=logfile,stderr=subprocess.STDOUT)
  state.update(fold=fold,stage=stage,child_pid=child.pid);atomic(root/'queue_status.json',state)
  code=child.wait();logfile.close()
  if code:raise RuntimeError(f'{stage} fold{fold} exited{code}; preserve completed files, inspect log')
 for fold in range(4):
  d=root/f'f{fold}';d.mkdir(exist_ok=True)
  if read(d/'report.json').get('state')=='COMPLETED':continue
  # Fold0 extraction was started while this continuation was implemented.
  old=read(d/'cache_status.json')
  while old.get('state')=='EXTRACTING' and Path(f"/proc/{old.get('pid',0)}").exists():
   state.update(fold=fold,stage='existing_healthy_cache',child_pid=old['pid']);atomic(root/'queue_status.json',state);time.sleep(5);old=read(d/'cache_status.json')
  if old.get('state')!='COMPLETED':execute('open_pool_cache.py',fold,'cache')
  execute('open_pool_eval.py',fold,'eval')
  report=read(d/'report.json')
  if report.get('state')!='COMPLETED' or report.get('episodes')!=100:raise RuntimeError('Missing full evaluation receipt; protect cache')
  cache=d/'cache';size=sum(f.stat().st_size for f in cache.glob('*.pt'));free=shutil.disk_usage(d).free
  shutil.rmtree(cache)
  atomic(d/'cache_cleanup_receipt.json',dict(reason='Finished fold interfaces and all fixed clean/mixed controls; compact metrics/patch masks/manifest retained',removed_feature_bytes=size,free_before=free,free_after=shutil.disk_usage(d).free))
  cs=read(d/'cache_status.json');cs.update(state='RETIRED_FEATURES_REMOVED');atomic(d/'cache_status.json',cs)
 # Only small paired task results enter CPU summary; no feature rebuild.
 state.update(stage='paired_analysis');atomic(root/'queue_status.json',state)
 subprocess.run([sys.executable,'-u','scripts/open_pool_analyze.py','--root',str(root)],env=env,check=True)
 state.update(state='COMPLETED',stage='done');atomic(root/'queue_status.json',state)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/open_pool_v1');a=ap.parse_args()
 try:main(a)
 except Exception:
  import traceback
  p=Path(a.root);p.mkdir(parents=True,exist_ok=True);atomic(p/'queue_status.json',dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc()));raise
