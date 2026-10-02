"""Unmodified supplied core/pose on identical cached native initializations.

Author caches omit image dimensions: use the same pre-inlier full-match bbox
grid as the independent implementation. Never invent dimensions or call the
package's NPZ cache loader with fake native uncertainty/image dimensions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,time,argparse,concurrent.futures,traceback
from pathlib import Path
from pose_eval import camera,intrinsic,extent,error,auc
sys.path.insert(0,str(Path(__file__).parent/'sources'))
import h5py,numpy as np,poselib
from provided_extension.gic import core,pose
from gic import grid_basis

ARMS=['extension_diagonal','extension_gic','extension_shuffled']
def evaluate(task):
 cache,key,scene,reference,*tail=task
 iterations=tail[0] if tail else 8
 old=json.loads(Path(reference).read_text());initial=old['arms']['native_poselib']
 r0=np.array(initial['R']);t0=np.array(initial['t']);t0/=np.linalg.norm(t0)
 with h5py.File(cache) as f:
  g=f[key];a=g['x1'][:].astype(float);b=g['x2'][:].astype(float)
  ca,cb=camera(g['camera1']),camera(g['camera2']);ka,kb=intrinsic(ca),intrinsic(cb)
  rg=g['R'][:].astype(float);tg=g['t'][:].astype(float)
 xa,domaina=extent(a,ca);xb,domainb=extent(b,cb)
 # The frozen initial report did not save the native Boolean mask. Replay
 # ONLY native inference with identical seed/options, assert pose identity.
 pn,info=poselib.estimate_relative_pose(a,b,ca,cb,{'max_epipolar_error':1.,'max_iterations':10000,'seed':2033},{})
 rt=np.array(pn.t);rt/=np.linalg.norm(rt)
 if not np.allclose(pn.R,r0,rtol=0,atol=1e-12) or not np.allclose(rt,t0,rtol=0,atol=1e-12):raise ValueError('native replay differs from frozen initial')
 keep=np.asarray(info['inliers'],bool)
 if int(keep.sum())!=old['fixed_inliers']:raise ValueError('native replay inlier count mismatch')
 a,b,xa,xb=a[keep],b[keep],xa[keep],xb[keep]
 A,B=grid_basis(xa),grid_basis(xb);cov=np.broadcast_to(np.eye(2),(len(a),2,2))
 results={};rho=0.;fit={};start=time.perf_counter()
 if len(a)>=20:
  e,ga,gb,J=pose.linearize_dual_endpoint(r0,t0,a,b,ka,kb)
  rho,fit=core.estimate_rho_dual_endpoint(J,e,cov,cov,ga,gb,A,B)
 calibration=time.perf_counter()-start
 perm=np.random.default_rng(2033).permutation(len(a))
 for arm in ARMS:
  start=time.perf_counter()
  try:
   if len(a)<20:r,t,status=r0,t0,'FALLBACK_FEW_INLIERS';steps=0
   else:
    aa,bb=(A[perm],B[perm]) if arm=='extension_shuffled' else (A,B)
    pp=pose.refine_pose_dual_endpoint(a,b,ka,kb,r0,t0,cov,cov,aa,bb,0. if arm=='extension_diagonal' else rho,max_iterations=iterations)
    r,t,status,steps=pp.R,pp.t,pp.status,pp.steps
   results[arm]=dict(R=r.tolist(),t=t.tolist(),seconds=time.perf_counter()-start+(calibration if arm!='extension_diagonal' else 0),status=status,steps=steps)
  except Exception as exc:
   results[arm]=dict(R=r0.tolist(),t=t0.tolist(),seconds=time.perf_counter()-start,status='FALLBACK_ERROR',reason=str(exc))
 # No GT argument has been passed to any refinement or correlation fitting.
 for v in results.values():v['error']=error(np.array(v['R']),np.array(v['t']),rg,tg)
 return dict(key=key,scene=scene,arms=results,rho=rho,fit=fit,calibration_seconds=calibration,fixed_inliers=int(keep.sum()),grid_domain=[domaina,domainb],native_replay_verified=True)

def run(args):
 parent=Path(args.reference);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tasks=[]
 for cache,key,scene in json.loads((parent/'manifest.json').read_text())['pairs']:
  reference=parent/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.json')
  tasks.append([cache,key,scene,str(reference),args.iterations])
 if args.limit:tasks=tasks[:args.limit]
 (out/'manifest.json').write_text(json.dumps(dict(pairs=tasks,source='User-provided gic_extension core/pose unchanged, explicit bbox adapter because author camera dimensions are zero',iterations=args.iterations,protocol='frozen native matches/init/inliers replay, g5, package rho/BIC/default LM; no GT fitting',diagnostic_scope='One step is a post-negative causal diagnostic, not new held-out method validation' if args.iterations==1 else 'frozen implementation validation'),indent=2))
 records={};pending=[]
 for task in tasks:
  pp=out/(hashlib.sha256(task[1].encode()).hexdigest()[:16]+'.json')
  if pp.exists():records[task[1]]=json.loads(pp.read_text())
  else:pending.append(task)
 status=dict(state='RUNNING',pid=os.getpid(),total=len(tasks),completed=len(records))
 (out/'status.json').write_text(json.dumps(status))
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
  futures={pool.submit(evaluate,t):t for t in pending}
  for fu in concurrent.futures.as_completed(futures):
   task=futures[fu]
   try:r=fu.result()
   except Exception:r=dict(key=task[1],scene=task[2],fatal_error=traceback.format_exc(),arms={a:dict(error={'pose':180.},status='FATAL') for a in ARMS})
   records[task[1]]=r;(out/(hashlib.sha256(task[1].encode()).hexdigest()[:16]+'.json')).write_text(json.dumps(r))
   status['completed']=len(records);(out/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
 rows=[records[t[1]] for t in tasks]
 summary={a:{f'AUC@{t}':auc([r['arms'][a]['error']['pose'] for r in rows],t) for t in [5,10,20]} for a in ARMS}
 (out/'report.json').write_text(json.dumps(dict(state='COMPLETED',summary=summary,results=rows,pairs=len(rows),fatal_failures=sum('fatal_error' in r for r in rows)),indent=2));status['state']='COMPLETED';(out/'status.json').write_text(json.dumps(status));print(json.dumps(summary),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True);ap.add_argument('--limit',type=int,default=8);ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--iterations',type=int,default=8);run(ap.parse_args())
