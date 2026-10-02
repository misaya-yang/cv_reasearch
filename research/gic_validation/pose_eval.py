"""Matched-input CPU relative-pose test on official DMS RoMa caches.

These caches are ORIGINAL RoMa, no anisotropic precision. Omega_A=Omega_B=I
in original pixel units. No images, model weights, new sampling, test-label
tuning or claimed RoMaV2 reproduction. Native PoseLib remains a separate arm.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,concurrent.futures,hashlib,json,sys,time,traceback
from pathlib import Path
sys.path[:0]=['/root/autodl-tmp/gic_validation/env','/root/demo4_cache/env']
import h5py,numpy as np,poselib
from scipy.spatial.transform import Rotation
from gic import two_end_field,SpectralREML,inverse_product


def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def tangent(t):
    axis=np.eye(3)[np.argmin(np.abs(t))];a=np.cross(t,axis);a/=np.linalg.norm(a)
    return np.column_stack([a,np.cross(t,a)])


def update(r,t,delta):
    rn=Rotation.from_rotvec(delta[:3]).as_matrix()@r
    tn=t+tangent(t)@delta[3:];tn/=np.linalg.norm(tn)
    return rn,tn


def residual(r,t,x1,x2,k1,k2,derivatives=False):
    f=np.linalg.inv(k2).T@skew(t)@r@np.linalg.inv(k1)
    a=np.column_stack([x1,np.ones(len(x1))]);b=np.column_stack([x2,np.ones(len(x2))])
    lb=a@f.T;la=b@f;e=np.einsum('ni,ni->n',b,lb)
    den=np.sqrt((lb[:,:2]**2).sum(1)+(la[:,:2]**2).sum(1)).clip(1e-15)
    result=e/den
    if not derivatives:return result
    ga=la[:,:2]/den[:,None]-e[:,None]*(lb[:,:2]@f[:2,:2])/(den**3)[:,None]
    gb=lb[:,:2]/den[:,None]-e[:,None]*(la[:,:2]@f[:2,:2].T)/(den**3)[:,None]
    return result,ga,gb


def jacobian(r,t,x1,x2,k1,k2):
    columns=[];eps=1e-5
    for i in range(5):
        d=np.zeros(5);d[i]=eps;rp,tp=update(r,t,d);rm,tm=update(r,t,-d)
        columns.append((residual(rp,tp,x1,x2,k1,k2)-residual(rm,tm,x1,x2,k1,k2))/(2*eps))
    return np.column_stack(columns)


def extent(x,cam):
    w,h=cam['width'],cam['height']
    if w>0 and h>0:return (x/np.array([w,h])).clip(0,1),'camera_dimensions'
    # Author cache stores zero image sizes for ScanNet. Freeze an observable
    # match-bounding-box domain; do not pretend to know uncached image size.
    lo=x.min(0);span=(x.max(0)-lo).clip(1e-6)
    return ((x-lo)/span).clip(0,1),'all_input_match_bbox_no_image_dimensions'


def refine(r0,t0,x1,x2,k1,k2,xy1,xy2,mode,confidence):
    r=r0.copy();t=t0/np.linalg.norm(t0);rho=0.;gate={};trace=[]
    cov=np.broadcast_to(np.eye(2),(len(x1),2,2))
    raw,ga,gb=residual(r,t,x1,x2,k1,k2,True);j=jacobian(r,t,x1,x2,k1,k2)
    v,field=two_end_field(xy1,xy2,cov,cov,ga,gb)
    if mode in ['gic','shuffled']:
        rho,gate=SpectralREML(raw,j,v,field).select()
    if mode=='one_per_cell':
        cell=np.minimum((xy1*5).astype(int),4);keys=cell[:,1]*5+cell[:,0]
        ix=np.array([np.where(keys==c)[0][np.argmax(confidence[keys==c])] for c in np.unique(keys)])
        if len(ix)>=8:
            x1,x2,xy1,xy2,confidence=x1[ix],x2[ix],xy1[ix],xy2[ix],confidence[ix]
            cov=cov[ix]
    rng=np.random.default_rng(2033);perm=rng.permutation(len(x1))
    damp_ratio=1e-3
    for it in range(8):
        rr,ga,gb=residual(r,t,x1,x2,k1,k2,True);j=jacobian(r,t,x1,x2,k1,k2)
        v,field=two_end_field(xy1,xy2,cov,cov,ga,gb)
        if mode=='shuffled':
            # Permute each image's spatial bases, not residual sensitivities,
            # so every row marginal remains EXACTLY unchanged.
            v,field=two_end_field(xy1[perm],xy2[perm],cov,cov,ga,gb)
        d=(1-rho)*v;u=np.sqrt(rho)*field
        if mode=='inverse_occupancy':
            cells=np.minimum((xy1*5).astype(int),4);key=cells[:,1]*5+cells[:,0]
            counts=np.bincount(key,minlength=25)[key];weight=1/counts.astype(float)
            weight/=weight.mean();d=v/weight
        stack=inverse_product(d,u,np.column_stack([j,rr]));h=j.T@stack[:,:5];b=j.T@stack[:,5]
        damping=damp_ratio*np.trace(h)/5
        delta=-np.linalg.solve(h+damping*np.eye(5),b)
        # Common angular step constraint and fixed-covariance acceptance.
        delta*=min(1.,.05/max(np.linalg.norm(delta),1e-15))
        old=float(rr@stack[:,5]);accepted=False
        for half in range(6):
            rn,tn=update(r,t,delta/(2**half));candidate=residual(rn,tn,x1,x2,k1,k2)
            cost=float(candidate@inverse_product(d,u,candidate))
            if cost<old:
                r,t=rn,tn;accepted=True;damp_ratio=max(damp_ratio/2,1e-8);break
        if not accepted:damp_ratio=min(damp_ratio*10,1e4)
        trace.append(dict(iteration=it,cost=old,accepted=accepted,step_norm=float(np.linalg.norm(delta))))
        if np.linalg.norm(delta)<1e-9:break
    return r,t,dict(rho=rho,gate=gate,iterations=len(trace),accepted_steps=sum(z['accepted'] for z in trace),points=len(x1))


def camera(g):
    return dict(model=g['model'].asstr()[0],width=int(g['width'][0]),height=int(g['height'][0]),params=g['params'][:].astype(float).tolist())


def intrinsic(cam):
    if cam['model']=='PINHOLE':fx,fy,cx,cy=cam['params']
    elif cam['model']=='SIMPLE_PINHOLE':fx,cx,cy=cam['params'];fy=fx
    else:raise ValueError('Unsupported cached camera model; do not silently ignore distortion')
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])


def error(r,t,rg,tg):
    a=np.linalg.svd(rg);rg=a[0]@a[2]
    re=np.degrees(np.arccos(np.clip((np.trace(r@rg.T)-1)/2,-1,1)))
    te=np.degrees(np.arccos(np.clip(abs(t@tg)/(np.linalg.norm(t)*np.linalg.norm(tg)),-1,1)))
    return dict(rotation=float(re),translation=float(te),pose=float(max(re,te)))


ARMS=['native_poselib','native_extra8','diagonal','gic','shuffled','inverse_occupancy','one_per_cell']


def evaluate_pair(task):
    cache,key,scene=task;started=time.monotonic()
    with h5py.File(cache,'r') as f:
        g=f[key];x1=g['x1'][:].astype(float);x2=g['x2'][:].astype(float)
        c1,c2=camera(g['camera1']),camera(g['camera2']);k1,k2=intrinsic(c1),intrinsic(c2)
        confidence=g['confidence'][:].astype(float) if 'confidence' in g else np.ones(len(x1))
        # GT is withheld from every inference/calibration function below.
        rg=g['R'][:].astype(float);tg=g['t'][:].astype(float)
    xy1,domain1=extent(x1,c1);xy2,domain2=extent(x2,c2)
    before=time.perf_counter()
    pose,info=poselib.estimate_relative_pose(x1,x2,c1,c2,{'max_epipolar_error':1.,'max_iterations':10000,'seed':2033},{})
    native_time=time.perf_counter()-before;r0=np.asarray(pose.R).copy();t0=np.asarray(pose.t).copy()
    ix=np.asarray(info['inliers'],bool);inputs=(x1[ix],x2[ix],k1,k2,xy1[ix],xy2[ix])
    outputs={'native_poselib':dict(R=r0,t=t0,seconds=native_time,details={'inliers':int(ix.sum())})}
    for arm in ARMS[1:]:
        timer=time.perf_counter()
        try:
            if ix.sum()<8:r,t,detail=r0,t0,{'fallback':'less_than_8_fixed_inliers'}
            elif arm=='native_extra8':
                initial=poselib.CameraPose();initial.R=r0.copy();initial.t=t0.copy()
                p,meta=poselib.refine_relative_pose(x1[ix],x2[ix],initial,c1,c2,{'max_iterations':8,'loss_type':'TRIVIAL'})
                r,t,detail=np.asarray(p.R),np.asarray(p.t),{'iterations_budget':8}
            else:r,t,detail=refine(r0,t0,*inputs,arm,confidence[ix])
            if not np.isfinite(r).all() or not np.isfinite(t).all():raise ValueError('nonfinite pose')
        except Exception as exc:r,t,detail=r0,t0,{'fallback':'native_init','error':type(exc).__name__+': '+str(exc)}
        outputs[arm]=dict(R=r,t=t,seconds=time.perf_counter()-timer,details=detail)
    # Score GT only after all arms have finished; inference never receives it.
    result=dict(key=key,scene=scene,input_matches=len(x1),fixed_inliers=int(ix.sum()),coordinate_units='original pixels',
                grid_domain=[domain1,domain2],matcher='author precomputed original RoMa; NOT RoMaV2',
                uncertainty='Omega_A=Omega_B=I pixel^2; no predicted anisotropic precision in this cache',
                arms={a:dict(error=error(o['R'],o['t'],rg,tg),seconds=o['seconds'],details=o['details'],R=o['R'].tolist(),t=o['t'].tolist()) for a,o in outputs.items()},
                elapsed_seconds=time.monotonic()-started)
    return result


def auc(values,threshold):
    x=np.r_[0.,np.sort(values)];y=np.arange(len(x))/len(values);k=np.searchsorted(x,threshold)
    xx=np.r_[x[:k],threshold];yy=np.r_[y[:k],y[max(k-1,0)]]
    return float(np.trapezoid(yy,xx)/threshold*100)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():tasks=json.loads(manifest.read_text())['pairs']
    else:
        tasks=[]
        with h5py.File(args.cache) as f:
            for key in sorted(f.keys()):
                name=f[key]['name1'].asstr()[0]
                scene=name.split('/')[0] if '/' in name else '_'.join(name.split('_')[:2])
                dev=int(hashlib.sha256(scene.encode()).hexdigest()[:8],16)%5==0
                if args.split=='all' or (args.split=='dev')==dev:tasks.append([args.cache,key,scene])
        if args.limit:tasks=tasks[:args.limit]
        manifest.write_text(json.dumps(dict(pairs=tasks,split=args.split,limit=args.limit,selection='sorted keys, whole scene hash mod5; no GT/predictions inspected for selection',
             protocol='g5, one rho, 8 common LM steps, lambda/trace scale, .05 angular norm clip, fixed native inliers, 1px native threshold',
             uncertainty='two-end isotropic original-pixel covariance; legacy RoMa, no RoMaV2 claim'),indent=2))
    results={};pending=[]
    for task in tasks:
        p=out/(hashlib.sha256(task[1].encode()).hexdigest()[:16]+'.json')
        if p.exists():results[task[1]]=json.loads(p.read_text())
        else:pending.append(task)
    status=dict(state='RUNNING',pid=os.getpid(),completed=len(results),total=len(tasks),arms=ARMS,scope='real legacy-RoMa isotropic pilot; not full claimed strong-baseline matrix')
    (out/'status.json').write_text(json.dumps(status,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures={executor.submit(evaluate_pair,task):task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task=futures[future]
            try:r=future.result()
            except Exception:
                r=dict(key=task[1],scene=task[2],fatal_error=traceback.format_exc(),arms={a:dict(error={'pose':180.},seconds=0.,details={'failure':'all-arm initialization/input failure'}) for a in ARMS})
            p=out/(hashlib.sha256(task[1].encode()).hexdigest()[:16]+'.json');p.write_text(json.dumps(r,indent=2));results[task[1]]=r
            status['completed']=len(results);(out/'status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps({'completed':len(results),'total':len(tasks),'last':task[1]}),flush=True)
    ordered=[results[t[1]] for t in tasks]
    summary={a:{f'AUC@{t}':auc([r['arms'][a]['error']['pose'] for r in ordered],t) for t in [5,10,20]} for a in ARMS}
    report=dict(state='COMPLETED',summary=summary,pairs=len(ordered),scenes=len({r['scene'] for r in ordered}),
                fatal_failures=sum('fatal_error' in r for r in ordered),results=ordered,
                boundaries='not DMS full backend, no RoMaV2 anisotropy or scene confidence intervals yet; runtime includes CPU inference/calibration/refinement but no image encoder',
                native='PoseLib2.0.4 full estimate_relative_pose, not own diagonal implementation',
                source_cache=args.cache)
    (out/'report.json').write_text(json.dumps(report,indent=2));status['state']='COMPLETED';(out/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--split',choices=['dev','test','all'],default='dev');ap.add_argument('--limit',type=int,default=8)
    ap.add_argument('--jobs',type=int,default=2);run(ap.parse_args())
