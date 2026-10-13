"""Does true physical view pairing identify a better source-compatible task?

Full-mask area regression gives scores a defined source target. Query-label
oracle fits only whole-query training blocks; actual crop features are never
used to fit those heads. They test cross-view generalization, not self-fit.
Prediction consistency is only a diagnostic; no segmentation method is claimed.
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_n,'2')
import argparse
from pathlib import Path
from collections import defaultdict
import json
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task_transfer_identifiability as common
import run_reference_focus200 as binding

np,torch,F,base=common.np,common.torch,common.F,common.base
ROOT=base.ASSETS/'a/reference_focus_head200_20261010'
OUT=base.ASSETS/'a/task_view_compatibility40_20261010'


def prepare(out):
    if (out/'config.json').exists():return
    rows=common.read(ROOT/'manifest.json');sources=common.read(ROOT/'tasks.json')
    counts=defaultdict(int);selected=[]
    for row,task in zip(rows,sources):
        key=row['dataset'],row['fold'];limit=20 if row['dataset']=='deepglobe_road' else 5
        if counts[key]<limit:
            counts[key]+=1;selected.append(dict(row=row,task=task))
    assert len(selected)==40
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('source','oracle','frozen'):(out/folder).mkdir(exist_ok=True)
    shutil.copyfile(__file__,out/'frozen'/Path(__file__).name)
    base.write(out/'tasks.json',selected)
    base.write(out/'config.json',dict(n=40,source_sha256=base.sha(__file__),
        tasks_sha256=base.sha(out/'tasks.json'),parent_tasks_sha256=base.sha(ROOT/'tasks.json'),
        helper_sha256=base.sha(common.__file__),
        hypothesis='True physical multiview pairing prefers the more transferable source-compatible classifier over an equally lawful reference-only fit',
        target='unbalanced physical mask area coverage, not balanced-role margin or a calibrated-query probability claim',
        heads=['reference_only','query_whole_oracle','joint_whole_oracle'],
        fit='all reference64 tokens; oracle train on alternating8x8-token whole-Q blocks; never train on nativeQ',
        regularizer='lambda=trace(reference-centered covariance)/1024, common across all heads for one episode',
        inputs='bound original FP32 wholeR,wholeQ,four nativeQ corners; unit normalize each actual observation',
        query_views=['whole64','actual_native128','derived_whole128'],
        primary_criterion='mean((whole-pool2(actual_fine))^2) / same error with physical pairing permuted within heldout blocks',
        null='seed0 permutation preserves the heldout whole/fine marginal feature and field bags; same permutation for all heads',
        no_FoRIS=True,new_encodings=0,
        limits='Physical consistency cannot identify a common query offset; oracle labels diagnostic only; same-image global context remains shared'))


def verify(out):
    cfg=common.read(out/'config.json')
    assert base.sha(__file__)==cfg['source_sha256']==base.sha(out/'frozen'/Path(__file__).name)
    assert base.sha(common.__file__)==cfg['helper_sha256']
    assert base.sha(out/'tasks.json')==cfg['tasks_sha256'] and base.sha(ROOT/'tasks.json')==cfg['parent_tasks_sha256']
    return cfg


def raw_inputs(task):
    result=[]
    for request in task['raw']:
        assert base.sha(request['entry_path'])==request['entry_sha256']
        entry=common.read(request['entry_path'])
        assert entry['input_tensor_hash']==request['input_tensor_hash']
        path=Path(request['entry_path']).parent/entry['file']
        assert base.sha(path)==request['payload_sha256']==entry['file_sha256']
        with np.load(path) as z:raw=z['O/24'].copy()
        assert common.tensor_hash(raw)==request['tensor_sha256'] and raw.shape==(4096,1024) and raw.dtype==np.float32
        result.append(torch.from_numpy(raw))
    assert len(result)==6
    r,q=map(lambda x:F.normalize(x,dim=1),result[:2])
    fine=torch.empty(128,128,1024)
    for x,(y0,x0) in zip(result[2:],[(0,0),(0,64),(64,0),(64,64)]):
        fine[y0:y0+64,x0:x0+64]=F.normalize(x,dim=1).reshape(64,64,1024)
    derived=F.interpolate(result[1].reshape(64,64,1024).permute(2,0,1)[None],(128,128),mode='bilinear',align_corners=False)[0].permute(1,2,0).reshape(16384,1024)
    return r,q,fine.reshape(16384,1024),F.normalize(derived,dim=1)


def coverage(path,grid):
    mask=base.read_mask(path).astype('float32')
    canon=F.interpolate(torch.from_numpy(mask)[None,None],(1024,1024),mode='nearest')
    return F.avg_pool2d(canon,1024//grid)[0,0].numpy().reshape(-1)


@torch.inference_mode()
def fit(samples,regularizer=None,check=False):
    x=torch.cat([s[0] for s in samples]).double();y=torch.from_numpy(np.concatenate([s[1] for s in samples])).double()
    a=torch.cat([torch.full((len(s[0]),),1/(len(samples)*len(s[0])),dtype=torch.float64) for s in samples])
    mx=a@x;my=a@y;xc=x-mx;yc=y-my
    gx=xc.float().to('mps');ga=a.float().to('mps');gy=yc.float().to('mps')
    covariance=(gx.T@(ga[:,None]*gx)).cpu().double();rhs=(gx.T@(ga*gy)).cpu().double()
    if regularizer is None:regularizer=float(covariance.trace()/x.shape[1])
    assert regularizer>0
    system=covariance+regularizer*torch.eye(x.shape[1],dtype=torch.float64)
    w=torch.linalg.solve(system,rhs);b=my-mx@w
    info=dict(lambda_=regularizer,normal_residual=float((system@w-rhs).norm()/rhs.norm().clamp_min(1e-15)))
    if check:
        cov64=xc.T@(a[:,None]*xc);rhs64=xc.T@(a*yc)
        w64=torch.linalg.solve(cov64+regularizer*torch.eye(x.shape[1],dtype=torch.float64),rhs64);b64=my-mx@w64
        error=float(((x@w+b)-(x@w64+b64)).abs().max());assert error<1e-4,error
        info['cpu_FP64_prediction_max_error']=error
    return w,b,info


def predict(model,x):
    w,b,_=model
    s=(x.to('mps')@w.float().to('mps')+b.float().to('mps')).cpu().numpy()
    assert np.isfinite(s).all()
    return s


def source(out):
    prepare(out);verify(out)
    if (out/'source_sealed.json').exists():return
    tasks=common.read(out/'tasks.json');denied=[t['row']['query_mask_path'] for t in tasks]
    records=[];tick=time.monotonic()
    for i,t in enumerate(tasks):
        row,task=t['row'],t['task'];record=out/'source'/f'{i:03d}.json'
        if record.exists():records.append(common.read(record));continue
        with binding.deny_paths(denied) as guard:
            r,q,fine,derived=raw_inputs(task);cr=coverage(row['reference_mask_path'],64)
            model=fit([(r,cr)],check=i in (0,20))
            fields={name:predict(model,x) for name,x in [('reference',r),('whole',q),('native',fine),('derived',derived)]}
            fields.update(coefficient=model[0].numpy(),bias=model[1].numpy())
            path=out/'source'/f'{i:03d}.npz';np.savez_compressed(path,**fields)
            assert guard['attempts']==0
        value=dict(index=i,id=row['episode_id'],dataset=row['dataset'],field_sha256=base.sha(path),fit=model[2],query_GT_reads=0,new_encodings=0)
        base.write(record,value);records.append(value)
        if (i+1)%10==0:print(json.dumps(dict(stage='SOURCE',n=i+1,seconds=time.monotonic()-tick)),flush=True)
    base.write(out/'source_sealed.json',dict(n=40,records=records,config_sha256=base.sha(out/'config.json'),seconds=time.monotonic()-tick))


def metrics(score,c,keep):
    y=c[keep];s=score[keep]
    auc=base.weighted_auc(s,y,1-y) if y.sum()>0 and (1-y).sum()>0 else None
    return dict(auc=auc,area_mse=float(np.mean((s-y)**2)))


def physical(whole,fine,keep,permutation,source_contrast):
    a=whole[keep].astype(float);b=fine.reshape(64,2,64,2).mean((1,3)).ravel()[keep].astype(float)
    error=float(np.mean((a-b)**2));null=float(np.mean((a-b[permutation])**2))
    return dict(paired_mse=error,shuffled_mse=null,ratio=None if null==0 else error/null,
        common_view_shift=float(a.mean()-b.mean()),
        contrast_normalized_mse=None if abs(source_contrast)<1e-10 else error/source_contrast**2)


def oracle(out):
    verify(out);tasks=common.read(out/'tasks.json');seal=common.read(out/'source_sealed.json')
    assert seal['config_sha256']==base.sha(out/'config.json')
    split=common.split_grid();records=[];tick=time.monotonic()
    for i,t in enumerate(tasks):
        row,task=t['row'],t['task'];record=out/'oracle'/f'{i:03d}.json'
        if record.exists():records.append(common.read(record));continue
        sr=seal['records'][i];assert base.sha(out/'source'/f'{i:03d}.npz')==sr['field_sha256']
        r,q,fine,derived=raw_inputs(task);cr=coverage(row['reference_mask_path'],64)
        cq=coverage(row['query_mask_path'],64);cq_fine=coverage(row['query_mask_path'],128)
        assert np.array_equal(cq,cq_fine.reshape(64,2,64,2).mean((1,3)).ravel())
        rows=[];arrays={}
        with np.load(out/'source'/f'{i:03d}.npz') as source:
            for fold in (0,1):
                train=split==fold;test=~train;fine_test=np.repeat(np.repeat(test.reshape(64,64),2,0),2,1).ravel()
                if cq[train].sum()==0 or (1-cq[train]).sum()==0 or cq[test].sum()==0 or (1-cq[test]).sum()==0:
                    rows.append(dict(fold=fold,valid=False,reason='query_train_or_test_missing_role'));continue
                models=dict(query_whole_oracle=fit([(q[train],cq[train])],sr['fit']['lambda_'],check=i in (0,20)),
                            joint_whole_oracle=fit([(r,cr),(q[train],cq[train])],sr['fit']['lambda_'],check=i in (0,20)))
                permutation=np.random.default_rng(0).permutation(int(test.sum()));measures={}
                for name in ('reference_only',*models):
                    fields={v:source[v].copy() for v in ('reference','whole','native','derived')} if name=='reference_only' else {
                        v:predict(models[name],x) for v,x in [('reference',r),('whole',q),('native',fine),('derived',derived)]}
                    ref=fields['reference'];contrast=float(np.sum(ref*cr)/cr.sum()-np.sum(ref*(1-cr))/(1-cr).sum())
                    m=dict(reference=metrics(ref,cr,np.ones(len(cr),bool)),whole=metrics(fields['whole'],cq,test),
                        native=metrics(fields['native'],cq_fine,fine_test),derived=metrics(fields['derived'],cq_fine,fine_test),
                        native_physical=physical(fields['whole'],fields['native'],test,permutation,contrast),
                        derived_physical=physical(fields['whole'],fields['derived'],test,permutation,contrast),reference_contrast=contrast)
                    if name!='reference_only':m['fit']=models[name][2]
                    measures[name]=m
                    for field,s in fields.items():arrays[f'{fold}/{name}/{field}']=s
                rows.append(dict(fold=fold,valid=True,methods=measures))
        path=out/'oracle'/f'{i:03d}.npz';np.savez_compressed(path,**arrays)
        value=dict(index=i,id=row['episode_id'],dataset=row['dataset'],folds=rows,field_sha256=base.sha(path),
            query_GT_used_for_oracles=True,new_encodings=0)
        base.write(record,value);records.append(value)
        if (i+1)%10==0:print(json.dumps(dict(stage='ORACLE',n=i+1,seconds=time.monotonic()-tick)),flush=True)
    summary={}
    for dataset in sorted({r['dataset'] for r in records}):
        episodes=[]
        for row in records:
            if row['dataset']!=dataset:continue
            valid=[f for f in row['folds'] if f['valid']]
            if not valid:continue
            arms={}
            for name in ('reference_only','query_whole_oracle','joint_whole_oracle'):
                arms[name]={}
                for metric in ('reference','whole','native','derived'):
                    arms[name][metric]=float(np.mean([f['methods'][name][metric]['auc'] for f in valid]))
                for metric in ('native_physical','derived_physical'):
                    arms[name][metric]=float(np.mean([f['methods'][name][metric]['ratio'] for f in valid]))
                    arms[name][metric+'_raw']=float(np.mean([f['methods'][name][metric]['paired_mse'] for f in valid]))
                    arms[name][metric+'_null']=float(np.mean([f['methods'][name][metric]['shuffled_mse'] for f in valid]))
            episodes.append(dict(id=row['id'],arms=arms))
        names=episodes[0]['arms'];means={a:{m:float(np.mean([e['arms'][a][m] for e in episodes])) for m in names[a]} for a in names}
        delta={metric:base.bootstrap([e['arms']['joint_whole_oracle'][metric]-e['arms']['reference_only'][metric] for e in episodes])
               for metric in ('reference','whole','native','derived','native_physical','derived_physical')}
        summary[dataset]=dict(n=len(episodes),means=means,paired_joint_minus_reference=delta,
            better_query_and_lower_physical_cost=sum(e['arms']['joint_whole_oracle']['whole']>e['arms']['reference_only']['whole'] and e['arms']['joint_whole_oracle']['native_physical']<e['arms']['reference_only']['native_physical'] for e in episodes),
            episodes=episodes)
    base.write(out/'report.json',dict(state='PHYSICAL_PAIRING_IDENTIFIABILITY_DIAGNOSTIC',datasets=summary,
        oracle_seconds=time.monotonic()-tick,new_encodings=0,warning='Query-label oracle; not a deployable segmentation result',
        source_seal_sha256=base.sha(out/'source_sealed.json'),config_sha256=base.sha(out/'config.json')))
    print(json.dumps({d:{k:v for k,v in s.items() if k!='episodes'} for d,s in summary.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['source','oracle','all']);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    if args.mode in ('source','all'):source(args.out)
    if args.mode in ('oracle','all'):oracle(args.out)


if __name__=='__main__':main()
