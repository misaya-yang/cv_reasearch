"""Author DMS backend with on-the-fly 4D KMeans; no duplicate HDF dataset.

Same cached RoMa matches as GIC. Report matched 1px and author 2.5px indoor
threshold separately. One seed, not a claimed ten-trial paper reproduction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,hashlib,concurrent.futures,traceback,sys
from pathlib import Path
from pose_eval import camera,intrinsic,error,auc
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'sources/DMS/src'))
import h5py,numpy as np
from sklearn.cluster import KMeans
import sklearn

def evaluate(task):
 from dms import geometry
 from dms.clusters import ClusterCorrespondence
 from dms.estimation import estimate_relative_pose
 cache,key,scene=task
 with h5py.File(cache) as f:
  g=f[key];a=g['x1'][:].astype(float);b=g['x2'][:].astype(float)
  ka,kb=intrinsic(camera(g['camera1'])),intrinsic(camera(g['camera2']))
  rg=g['R'][:].astype(float);tg=g['t'][:].astype(float)
 start=time.perf_counter();ac,bc=geometry.calibrate_pts(a,ka),geometry.calibrate_pts(b,kb)
 labels=KMeans(n_clusters=128,random_state=0,n_init='auto').fit_predict(np.column_stack([ac,bc]))
 clusters=[ClusterCorrespondence({'x1':a[labels==i],'x2':b[labels==i]},ka,kb) for i in np.unique(labels)]
 clustering=time.perf_counter()-start
 meanfocal=float(np.mean([ka[0,0],ka[1,1],kb[0,0],kb[1,1]]))
 data=dict(x1=a,x2=b,x1c=ac,x2c=bc,ecorrs=clusters,filter_outliers=False)
 policies=[('dms_dense_matched1',1.,'dense','dense','dense'),('dms_cca_matched1',1.,'center','center','approx')]
 if 'scannet' in cache:policies+=[('dms_dense_author2p5',2.5,'dense','dense','dense'),('dms_cca_author2p5',2.5,'center','center','approx')]
 results={}
 for name,threshold,sampling,scoring,refinement in policies:
  start=time.perf_counter()
  try:
   infer=dict(data,threshold_E=threshold/meanfocal)
   # No truth, scene labels, quality-dependent threshold or subsampling enters.
   p,backend_seconds=estimate_relative_pose(infer,sampling,scoring,refinement,seed=2033)
   r,t=np.asarray(p['R']),np.asarray(p['t'])
   if not np.isfinite(r).all() or not np.isfinite(t).all() or np.linalg.norm(t)<1e-12:raise ValueError('invalid native DMS pose')
   result=dict(R=r.tolist(),t=t.tolist(),seconds=time.perf_counter()-start,backend_reported_seconds=backend_seconds,status='COMPLETED')
  except Exception as exc:result=dict(seconds=time.perf_counter()-start,status='ERROR',error_message=str(exc))
  results[name]=result
 # Score only after all backend arms have completed.
 for result in results.values():result['error']=error(np.array(result['R']),np.array(result['t']),rg,tg) if result['status']=='COMPLETED' else {'pose':180.}
 return dict(key=key,scene=scene,arms=results,clustering_seconds=clustering,cluster_count=len(clusters),input_matches=len(a))

def run(args):
 ref=Path(args.reference);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tasks=json.loads((ref/'manifest.json').read_text())['pairs']
 if args.limit:tasks=tasks[:args.limit]
 (out/'manifest.json').write_text(json.dumps(dict(pairs=tasks,protocol='Official DMS source7d5b228, unchanged cluster summaries/backend, KMeans4d calibrated/128/random_state0/n_init auto; seed2033; matched1px and separate author2.5px indoor. Same cache, no ten-trial reproduction',sklearn=sklearn.__version__),indent=2))
 status=dict(state='WAITING_DMS_BUILD',pid=os.getpid(),total=len(tasks),completed=0)
 (out/'status.json').write_text(json.dumps(status))
 start=time.monotonic()
 while not list((ROOT/'env').glob('pysumopt*.so')):
  if time.monotonic()-start>1800:raise RuntimeError('DMS build timeout; preserve stage and inspect build_dms.log')
  time.sleep(10)
 import pysumopt
 records={};pending=[]
 for t in tasks:
  pp=out/(hashlib.sha256(t[1].encode()).hexdigest()[:16]+'.json')
  if pp.exists():records[t[1]]=json.loads(pp.read_text())
  else:pending.append(t)
 status.update(state='RUNNING',completed=len(records));(out/'status.json').write_text(json.dumps(status))
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
  fs={pool.submit(evaluate,t):t for t in pending}
  for fu in concurrent.futures.as_completed(fs):
   t=fs[fu]
   try:r=fu.result()
   except Exception:r=dict(key=t[1],scene=t[2],fatal_error=traceback.format_exc(),arms={a:dict(error={'pose':180.},status='FATAL') for a in (['dms_dense_matched1','dms_cca_matched1']+(['dms_dense_author2p5','dms_cca_author2p5'] if 'scannet' in t[0] else []))})
   records[t[1]]=r;(out/(hashlib.sha256(t[1].encode()).hexdigest()[:16]+'.json')).write_text(json.dumps(r))
   status['completed']=len(records);(out/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
 rows=[records[t[1]] for t in tasks];arms=list(rows[0]['arms'])
 summary={a:{f'AUC@{t}':auc([r['arms'][a]['error']['pose'] for r in rows],t) for t in [5,10,20]} for a in arms}
 (out/'report.json').write_text(json.dumps(dict(state='COMPLETED',results=rows,summary=summary,pairs=len(rows),scenes=len({r['scene'] for r in rows}),source='official DMS full source backend, Ceres2.2/Eigen3.4; no image encoder',runtime_scope='includes separately recorded clustering; concurrent CPU times not formal isolated latency'),indent=2));status['state']='COMPLETED';(out/'status.json').write_text(json.dumps(status));print(json.dumps(summary),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True);ap.add_argument('--limit',type=int,default=8);ap.add_argument('--jobs',type=int,default=2);args=ap.parse_args()
 try:run(args)
 except Exception:
  pp=Path(args.out);pp.mkdir(parents=True,exist_ok=True);(pp/'error.txt').write_text(traceback.format_exc());(pp/'status.json').write_text(json.dumps(dict(state='ERROR',pid=os.getpid(),error=traceback.format_exc())));raise
