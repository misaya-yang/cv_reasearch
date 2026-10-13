#!/usr/bin/env python3
"""Independent single-reference rank update in sampled-reference nullspace.

No query labels, oracle coefficients or FoRIS fields enter prediction.
The source-only update learns from cached true reference detail, with matched
interpolation and unconstrained controls. This is an exposed development test.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task_transfer_identifiability as d
import task_transfer_source_span_audit as span
import task_transfer_focus_observer as focus

np,torch,F,base=d.np,d.torch,d.F,d.base
OUT=base.ASSETS/'a/task_transfer_null_rank_probe17_20261010'
ARMS=['reference_only','actual_unconstrained','actual_projected','actual_projected_safe','interpolated_projected_safe']


@contextmanager
def guard_oracles(attempts):
    import builtins,io,os
    forbidden=[span.OUT/'oracle_fields',focus.OUT]
    original=builtins.open,io.open
    def wrapped(fn):
        def read(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).resolve()
                if any(root==p or root in p.parents for root in forbidden):
                    attempts.append(str(p));raise RuntimeError('Oracle inputs prohibited in prediction')
            return fn(path,*args,**kwargs)
        return read
    builtins.open,io.open=map(wrapped,original)
    try:yield
    finally:builtins.open,io.open=original


def prepare():
    if (OUT/'config.json').exists():return
    # Only copy the cache/input manifest; no observer outcomes enter the input.
    tasks=d.read(focus.OUT/'tasks.json');assert len(tasks)==17
    base.write(OUT/'tasks.json',tasks)
    base.write(OUT/'config.json',dict(n_pairs=17,arms=ARMS,lambda_=d.LAMBDA,pairwise_target_margin=2.,
        formula='E_F,B[(theta dot (zF-zB)-2)^2]+.01||delta||^2, at theta=theta_reference+delta; projected arms impose Z_reference_quadrature delta=0',
        safety='min(1, half earliest source token sign crossing), all4096 reference predictions preserved in FP64',
        source_sha256={str(p):base.sha(p) for p in [Path(__file__),Path(d.__file__),Path(span.__file__),Path(focus.__file__)]},
        input_manifest_sha256=base.sha(OUT/'tasks.json'),parent_input_manifest_sha256=base.sha(focus.OUT/'tasks.json'),
        no_query_labels_in_prediction=True,no_oracle_fields_in_prediction=True,no_FoRIS_dependency=True,new_encodings=0,
        exposure='Same17 PACO development pairs. A uses its legal mask-generated focus; B fixed-A-view diagnostic, not official B view policy.',
        output='FP64 signed64 bilinear1024 align_corners=False then strict>0; no CRF or fitted threshold',
        novelty='Mechanistic prototype, not a verified originality or generalization claim'))


def rank_moments(sample,validate=False):
    x,y,a=sample[:3];x=x.double();y=y.double();a=a.double()
    x=torch.cat([x,torch.ones((len(x),1),dtype=torch.float64)],dim=1)
    fg=a*(1+y);bg=a*(1-y)
    mu_f=x.T@fg;mu_b=x.T@bg;contrast=mu_f-mu_b
    gx=(x*(2*a).sqrt()[:,None]).float().to('mps')
    h=(gx.T@gx).cpu().double()-torch.outer(mu_f,mu_b)-torch.outer(mu_b,mu_f)
    h=(h+h.T)/2;h[-1,:]=0;h[:,-1]=0;contrast[-1]=0
    error=None
    if validate:
        exact=x.T@((2*a)[:,None]*x)-torch.outer(mu_f,mu_b)-torch.outer(mu_b,mu_f)
        error=float((h-exact).abs().max());assert error<2e-6
    return h.numpy(),contrast.numpy(),error


def projected_update(h,contrast,theta,basis=None):
    from scipy.linalg import solve
    gradient=2*contrast-h@theta
    if basis is None:
        return solve(h+d.LAMBDA*np.eye(len(h)),gradient,assume_a='pos',check_finite=False)
    reduced=basis.T@h@basis+d.LAMBDA*np.eye(basis.shape[1])
    return basis@solve(reduced,basis.T@gradient,assume_a='pos',check_finite=False)


def infer():
    from scipy.linalg import svd
    from raw_feature_cache import tensor_hash
    prepare();cfg=d.read(OUT/'config.json')
    for p,h in cfg['source_sha256'].items():assert base.sha(p)==h
    assert base.sha(OUT/'tasks.json')==cfg['input_manifest_sha256']
    if (OUT/'sealed.json').exists():return
    for folder in ['fields','predictions','records']:(OUT/folder).mkdir(exist_ok=True)
    tasks=d.read(OUT/'tasks.json');records=[];start=time.monotonic()
    for i,task in enumerate(tasks):
        p=task['pair'];foc=task['focus'];rp=OUT/'records'/f'{i:03d}.json'
        if rp.exists():
            rec=d.read(rp);assert base.sha(OUT/'fields'/rec['file'])==rec['fields_sha256']
            assert base.sha(OUT/'predictions'/rec['file'])==rec['prediction_sha256'];records.append(rec);continue
        attempts=[];fields={};predictions={};info={};tick=time.monotonic()
        with base.deny_labels(base.OUT,attempts),guard_oracles(attempts):
            (r,q),raw=d.load_raw(p)
            assert base.sha(foc['entry_path'])==foc['entry_sha256']
            assert base.sha(foc['payload_path'])==foc['payload_sha256']
            entry=d.read(foc['entry_path'])
            with np.load(foc['payload_path']) as z:actual_raw=z['O/24'].copy()
            assert tensor_hash(actual_raw)==foc['O24']['tensor_sha256']==entry['features']['O/24']['tensor_sha256']
            actual=F.normalize(torch.from_numpy(actual_raw),dim=1)
            x0,y0,x1,y1=foc['box_xyxy'];cx=x0+(np.arange(64)+.5)*8;cy=y0+(np.arange(64)+.5)*8
            gx,gy=np.meshgrid(2*cx/1024-1,2*cy/1024-1)
            grid=torch.from_numpy(np.stack([gx,gy],-1).astype('float32'))[None]
            view=F.grid_sample(r.T.reshape(1,1024,64,64),grid,mode='bilinear',padding_mode='border',align_corners=False)
            derived=F.normalize(view[0].reshape(1024,-1).T,dim=1)
            zr=np.column_stack([r.numpy().astype(float),np.ones(4096)]);zq=np.column_stack([q.numpy().astype(float),np.ones(4096)])
            for v in ['A','B']:
                binding=p['variants'][v];c=d.coverage(binding['path'],binding['file_sha256'])
                source=d.sample(r,c,np.ones(4096,bool));fitted=d.fit([source]);assert fitted is not None
                theta=np.r_[fitted['w'].numpy(),fitted['b'].numpy()];reference=zr@theta
                train=zr[source[3]];_,s,vh=svd(train,full_matrices=True,check_finite=False)
                tolerance=np.finfo('float64').eps*max(train.shape)*s[0];rank=int((s>tolerance).sum())
                null_basis=vh[rank:].T
                cf=focus.crop_coverage(p,v,foc['box_xyxy'])
                af=d.sample(actual,cf,np.ones(4096,bool));df=d.sample(derived,cf,np.ones(4096,bool))
                updates={arm:np.zeros_like(theta) for arm in ARMS};metadata=dict(source_sample_rank=rank,focus_has_both_roles=af is not None)
                if af is not None:
                    h,cvec,error=rank_moments(af,validate=i==0)
                    raw_delta=projected_update(h,cvec,theta)
                    projected=projected_update(h,cvec,theta,null_basis)
                    ih,icvec,ierror=rank_moments(df,validate=i==0)
                    interpolated=projected_update(ih,icvec,theta,null_basis)
                    identity_error=float(np.max(np.abs(train@projected)))
                    assert identity_error<1e-10
                    scale=span.safe_scale(reference,zr@projected);iscale=span.safe_scale(reference,zr@interpolated)
                    updates.update(actual_unconstrained=raw_delta,actual_projected=projected,
                        actual_projected_safe=scale*projected,interpolated_projected_safe=iscale*interpolated)
                    assert np.array_equal(reference>0,(reference+zr@updates['actual_projected_safe'])>0)
                    assert np.array_equal(reference>0,(reference+zr@updates['interpolated_projected_safe'])>0)
                    metadata.update(projected_training_logit_error=identity_error,safe_scale=scale,interpolated_safe_scale=iscale,
                        actual_delta_norm=float(np.linalg.norm(projected)),interpolated_delta_norm=float(np.linalg.norm(interpolated)),
                        actual_moment_numerical_error=error,interpolated_moment_numerical_error=ierror)
                for arm,delta in updates.items():
                    score=zq@(theta+delta);key=v+'/'+arm
                    fields[key]=score.reshape(64,64);fields[key+'/reference']=zr@(theta+delta)
                    pre=F.interpolate(torch.from_numpy(score).reshape(1,1,64,64),(1024,1024),mode='bilinear',align_corners=False)[0,0]>0
                    predictions[key]=np.packbits(pre.numpy())
                info[v]=metadata
            assert not attempts
            name=f'{i:03d}.npz';np.savez_compressed(OUT/'fields'/name,**fields)
            np.savez_compressed(OUT/'predictions'/name,**predictions)
        rec=dict(id=p['id'],dataset=p['dataset'],index=i,file=name,fields_sha256=base.sha(OUT/'fields'/name),
            prediction_sha256=base.sha(OUT/'predictions'/name),metadata=info,query_or_oracle_access_attempts=attempts,
            raw=raw,focus_payload_sha256=foc['payload_sha256'],seconds=time.monotonic()-tick,new_encodings=0)
        base.write(rp,rec);records.append(rec);torch.mps.empty_cache()
        base.write(OUT/'activity.json',dict(state='INDEPENDENT_REFERENCE_RANK_PROBE',n=len(records),total=17,seconds=time.monotonic()-start,new_encodings=0))
    base.write(OUT/'sealed.json',dict(n_pairs=17,arms=ARMS,records=records,seconds=time.monotonic()-start,
        config_sha256=base.sha(OUT/'config.json'),tasks_sha256=base.sha(OUT/'tasks.json'),query_or_oracle_reads=0,new_encodings=0))


def score():
    seal=d.read(OUT/'sealed.json');assert base.sha(OUT/'config.json')==seal['config_sha256']
    truth=d.read(base.OUT/'query_truth.json');tasks=d.read(OUT/'tasks.json');records=[]
    native={r['id']:r for r in d.read(base.OUT/'diagnosis.json')['records']}
    for task,rec in zip(tasks,seal['records']):
        p=task['pair'];assert p['id']==rec['id']
        assert base.sha(OUT/'fields'/rec['file'])==rec['fields_sha256'] and base.sha(OUT/'predictions'/rec['file'])==rec['prediction_sha256']
        masks=[]
        for v in ['A','B']:
            qb=truth[p['id']]['masks'][v];assert base.sha(qb['path'])==qb['sha256']
            gt=F.interpolate(torch.from_numpy(base.read_mask(qb['path']).astype('float32'))[None,None],(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
            masks.append(gt)
        exclusive=[masks[0]&~masks[1],masks[1]&~masks[0]];mass=[x.reshape(64,16,64,16).sum((1,3)) for x in exclusive]
        measured={}
        with np.load(OUT/'fields'/rec['file']) as fields,np.load(OUT/'predictions'/rec['file']) as preds:
            for arm in ARMS:
                values=[];switch=[]
                for j,v in enumerate(['A','B']):
                    pred=np.unpackbits(preds[v+'/'+arm],count=1024**2).reshape(1024,1024).astype(bool);gt=masks[j]
                    hit=int((pred&gt).sum());union=int((pred|gt).sum());c=gt.reshape(64,16,64,16).mean((1,3))
                    values.append(dict(I=hit,U=union,FP=int((pred&~gt).sum()),FN=int((~pred&gt).sum()),iou=100*hit/max(union,1),
                        full_role_auc=base.weighted_auc(fields[v+'/'+arm],c,1-c)))
                    switch.append(base.weighted_auc(fields[v+'/'+arm],mass[j],mass[1-j]))
                measured[arm]=dict(tasks=values,task_mean_iou=float(np.mean([x['iou'] for x in values])),switch_auc=float(np.mean(switch)))
        measured['foris_full']={v:100*native[p['id']]['full_mask_IU']['crf'][j]['I']/max(native[p['id']]['full_mask_IU']['crf'][j]['U'],1) for j,v in enumerate(['A','B'])}
        records.append(rec|dict(measures=measured))
    summary={}
    for arm in ARMS:
        summary[arm]=dict(task_mean_iou=float(np.mean([r['measures'][arm]['task_mean_iou'] for r in records])),
            switch_auc=float(np.mean([r['measures'][arm]['switch_auc'] for r in records])),roles={})
        for j,v in enumerate(['A','B']):
            summary[arm]['roles'][v]=dict(mean_iou=float(np.mean([r['measures'][arm]['tasks'][j]['iou'] for r in records])),
                full_role_auc=float(np.mean([r['measures'][arm]['tasks'][j]['full_role_auc'] for r in records])),
                delta_reference=base.bootstrap([r['measures'][arm]['tasks'][j]['iou']-r['measures']['reference_only']['tasks'][j]['iou'] for r in records]),
                delta_foris=base.bootstrap([r['measures'][arm]['tasks'][j]['iou']-r['measures']['foris_full'][v] for r in records]))
    base.write(OUT/'report.json',dict(state='COMPLETE_INDEPENDENT_NULL_RANK_PROBE',arms=summary,records=records,n_pairs=17,
        seal_sha256=base.sha(OUT/'sealed.json'),new_encodings=0,query_labels_in_prediction=False,
        warning='Primary A uses its legal mask-generated view; fixed-A-view B is mechanism diagnostic. Exposed PACO17, not official mIoU or cross-domain confirmation.'))
    lines=['# 独立单参考nullspace排序更新','',
        '全部预测封存后才读query GT。只有合法R mask与已有focus及unit O24进入读出，无FoRIS、oracle、全局训练或新增编码。A为本view-policy合法单参考输出；B固定原A view只作反任务诊断。','',
        '|臂|原任务A完整IoU|A−原R读出|A−完整FoRIS|A全图角色AUC|','|---|---:|---:|---:|---:|']
    for arm,s in summary.items():
        a=s['roles']['A'];lines.append(f"|{arm}|{a['mean_iou']:.3f}|{a['delta_reference']['mean']:+.3f}|{a['delta_foris']['mean']:+.3f}|{a['full_role_auc']:.5f}|")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    base.write(OUT/'activity.json',dict(state='DIAGNOSED',n_pairs=17,new_encodings=0))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    infer();score()
