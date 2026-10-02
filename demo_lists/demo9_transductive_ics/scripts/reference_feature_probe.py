#!/usr/bin/env python3
"""Standalone prepared-feature executor for E3/E6 and E8 cheap controls only.

Trusted episode.pt: reference_features[Nr,C] UNIT, reference_coverage[Nr] in
[0,1], query_features[Nq,C] UNIT, query_geometry[Nq,D] native unit geometry,
grid={reference:[Hr,Wr],query:[Hq,Wq]} (or shared [H,W]), host_signed_field[H,W],
feature_contract (native diffusion requires stream='native'). No query GT here.

--evaluator-labels is a DIFFERENT file {native_mask:bool[H,W], optional
original_mask:bool[Ho,Wo]}. It is opened only after ALL method fields freeze.
Output is compact prediction bits and evaluator I/U/pixel ledgers, never raw
features. Metric requires a frozen base-trained checkpoint; no random dictionary
performance. Diffusion is the native closed-graph control, not an all-stream
method adapter. Numerical errors stay ERROR with only explicitly named safe
host fallback statistics; support degeneracies preserve the host exactly.

CPU by default. CUDA requires --allow-gpu AND a real parent resource-guard gate
whose own_child_pid equals this process and records CUDA readiness. No automatic
GPU discovery, encoder/model downloads, datasets or actual experiment orchestration.
"""
import argparse
import base64
import contextlib
import io
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

METHODS=('transport','metric','diffusion','kde','balanced_ot')
TICS=Path(__file__).resolve().parents[1]/'tics'


def module(name):
    key='_feature_probe_'+name
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,TICS/(name+'.py'))
        value=importlib.util.module_from_spec(spec);sys.modules[key]=value;spec.loader.exec_module(value)
    return sys.modules[key]


def file_sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def load_trusted(path):
    try:return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError as ex:
        if 'mmap' not in str(ex):raise
        return torch.load(path,map_location='cpu',weights_only=False)


def json_audit(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:json_audit(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_audit(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def canonical_episode(d):
    aliases={'reference_features':'Fs','reference_coverage':'coverage','query_features':'Fq','host_signed_field':'host_field'}
    out=dict(d)
    for long,short in aliases.items():
        if long in out and short in out:
            a,b=out[long],out[short]
            if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or not torch.equal(a,b):
                raise ValueError('Conflicting explicit long/short feature alias: '+long+'/'+short)
        if long not in out and short in out:out[long]=out[short]
        out.pop(short,None)
    if isinstance(out.get('grid'),dict):
        grid=dict(out['grid'])
        if 'support' in grid and 'reference' in grid and list(grid['support'])!=list(grid['reference']):
            raise ValueError('Conflicting support/reference grid aliases')
        if 'reference' not in grid and 'support' in grid:grid['reference']=grid['support']
        out['grid']=grid
    return out


def metric_loader(cpu_only=True):
    name='_feature_probe_metric_trainer'
    if name not in sys.modules:
        path=Path(__file__).resolve().parent/'train_reference_metric.py'
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec);sys.modules[name]=value
        if cpu_only:
            # Suppress unused package DEV auto-discovery on explicit CPU import.
            from unittest.mock import patch
            with patch.object(torch.cuda,'is_available',return_value=False):spec.loader.exec_module(value)
        else:spec.loader.exec_module(value)
    return sys.modules[name]


def validate_episode(d,method_count,max_ram_bytes):
    required=('reference_features','reference_coverage','query_features','query_geometry','grid','host_signed_field','feature_contract')
    if any(k not in d for k in required):raise ValueError('Missing explicit prepared feature/geometry/host contract')
    forbidden={'query_gt','query_truth','query_mask','query_label','query_labels','native_mask','original_mask','gt64','gt_bits','candidate_iou'}
    if forbidden & set(d):raise ValueError('Query evaluator supervision must be in a separate file')
    r,c,q,g=d['reference_features'],d['reference_coverage'],d['query_features'],d['query_geometry']
    grid=d['grid'];rg=tuple(grid['reference']) if isinstance(grid,dict) else tuple(grid)
    qg=tuple(grid['query']) if isinstance(grid,dict) else tuple(grid)
    if len(rg)!=2 or len(qg)!=2 or min(*rg,*qg)<1:raise ValueError('Positive 2D grids required')
    if r.ndim!=2 or q.ndim!=2 or r.shape[1]!=q.shape[1] or len(r)!=math.prod(rg) or len(q)!=math.prod(qg):
        raise ValueError('Full reference/query native token dimensions or grids disagree')
    if c.numel()!=len(r) or g.ndim!=2 or len(g)!=len(q):raise ValueError('Coverage/geometry length mismatch')
    host=d['host_signed_field']
    if host.ndim!=2 or min(host.shape)<1 or not host.is_floating_point() or not torch.isfinite(host).all():raise ValueError('Finite native 2D signed host field required')
    if not torch.isfinite(c).all() or (c<0).any() or (c>1).any():raise ValueError('Reference coverage must be finite in [0,1]')
    # Tensor working estimate includes casts, outputs, blocks and worst-case metric triplets.
    tensors=(r,c,q,g,host);inputs=sum(v.numel()*v.element_size() for v in tensors)
    estimate=inputs+4*(r.numel()+q.numel()+g.numel())+method_count*host.numel()*5
    estimate+=8*min(256,len(q))*len(r)*8+5*4096*r.shape[1]*8
    if estimate>max_ram_bytes:raise MemoryError(f'tensor estimate {estimate} > budget {max_ram_bytes}')
    return rg,qg,estimate


def device_gate(device,allow_gpu,guard_path):
    if device=='cpu':return torch.device('cpu')
    if device!='cuda' or not allow_gpu or guard_path is None:raise ValueError('CUDA requires explicit --allow-gpu and --resource-guard-state')
    guard=json.loads(guard_path.read_text())
    if guard.get('state')!='GPU_RUNNING' or guard.get('last_event',{}).get('own_child_pid')!=os.getpid() or not any(
        e.get('state')=='GPU_CUDA_READINESS_CONFIRMED' for e in guard.get('events',[])):
        raise ValueError('CUDA resource guard must own THIS executor and confirm real readiness')
    if not sys.platform.startswith('linux') or not torch.cuda.is_available():raise ValueError('Verified live Linux CUDA required')
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda')


def lift_signed(field,shape):
    if tuple(field.shape)==tuple(shape):return field
    return F.interpolate(field.reshape(1,1,*field.shape).float(),size=shape,mode='bilinear',align_corners=False)[0,0]


def metric_field(d,rg,checkpoint):
    if checkpoint is None:raise ValueError('metric requires frozen base-trained checkpoint; random performance forbidden')
    loader=metric_loader(cpu_only=d['query_features'].device.type=='cpu')
    model,ck=loader.load_deployment(checkpoint,d['feature_contract'],device=d['query_features'].device)
    delta,audit=model(d['reference_features'].float(),d['reference_coverage'].flatten().float(),d['query_features'].float(),rg)
    field=loader.correct_host_field(d['host_signed_field'],delta,tuple(d['query_grid']))
    audit.update(deployment_variant=ck['variant'],deployment_trained_steps=ck['trained_steps'],
        deployment_feature_contract_exact=True,deployment_frozen=all(not p.requires_grad for p in model.parameters()))
    return field,audit


@torch.no_grad()
def freeze_outputs(d,methods,*,metric_checkpoint=None,max_ram_bytes=512*1024**2):
    d=canonical_episode(d)
    rg,qg,ram=validate_episode(d,len(methods),max_ram_bytes)
    work=dict(d,query_grid=qg)
    host=d['host_signed_field'];ref=d['reference_features'];qry=d['query_features'];coverage=d['reference_coverage'].flatten()
    labels=coverage>=.5;outputs={'host':host.detach().clone()};records={}
    for name in methods:
        tick=time.monotonic();audit={};state='SOLVED';field=None
        try:
            if any(not torch.isfinite(v).all() or (v.norm(dim=-1)<=1e-12).any() for v in (ref,qry,d['query_geometry'])):
                raise ValueError('nonfinite/zero native descriptor is a numerical input error')
            elif not labels.any() or labels.all():
                field=host;state='FALLBACK_DEGENERATE_SUPPORT';audit={'preserved_host_exact':True}
            elif name in ('transport','kde','balanced_ot'):
                m=module('unbalanced_transport');mode={'transport':'unbalanced','kde':'kde','balanced_ot':'balanced'}[name]
                scores,audit=m.transport_scores(ref,labels,qry,mode=mode,max_ram_bytes=max_ram_bytes)
                field=lift_signed((scores-.5).reshape(qg),host.shape)
            elif name=='metric':
                field,audit=metric_field(work,rg,metric_checkpoint)
                if str(audit.get('state','')).startswith('FALLBACK'):
                    state='FALLBACK_DEGENERATE_SUPPORT';field=host;audit['preserved_host_exact']=True
            elif name=='diffusion':
                contract=d['feature_contract'];native=contract=='native' or (isinstance(contract,dict) and contract.get('stream')=='native')
                if not native:raise ValueError('diffusion here is native closed-graph control only')
                m=module('reference_diffusion')
                host_patch=lift_signed(host,qg).reshape(-1)
                scores,audit=m.reference_diffusion(ref,labels,qry,qg,d['query_geometry'],fallback_score=host_patch,variant='diffusion')
                if audit.get('state') in ('FALLBACK_NONFINITE_FEATURES','FALLBACK_ZERO_FEATURES','FALLBACK_NONCONVERGED'):
                    raise RuntimeError('diffusion numerical fallback promoted to ERROR: '+audit['state'])
                if audit.get('state')=='FALLBACK_MISSING_SUPPORT_CLASS':
                    state='FALLBACK_DEGENERATE_SUPPORT';field=host
                else:field=lift_signed(scores.reshape(qg),host.shape)
            else:raise ValueError('unknown requested method')
            if not torch.isfinite(field).all():raise RuntimeError('method produced nonfinite field')
            # Clone before evaluator file opens, including host fallback values.
            outputs[name]=field.detach().clone()
            record=dict(state=state,audit=json_audit(audit),wall_s=time.monotonic()-tick,used_host_exactly=bool(torch.equal(field,host)))
        except Exception as ex:
            # This output is a clearly named safe host fallback, NEVER method success.
            outputs[name]=host.detach().clone()
            record=dict(state='ERROR',error=repr(ex),wall_s=time.monotonic()-tick,error_safe_host_only=True)
        records[name]=record
    return outputs,records,dict(reference_grid=list(rg),query_grid=list(qg),estimated_tensor_bytes=ram,max_ram_bytes=max_ram_bytes,
        reference_FG_anchors=int(labels.sum()),reference_BG_anchors=int((~labels).sum()),feature_contract=d['feature_contract'])


def bits(mask):
    array=mask.detach().cpu().numpy().astype(bool)
    return dict(shape=list(array.shape),encoding='numpy.packbits-big/base64',data=base64.b64encode(np.packbits(array.reshape(-1)).tobytes()).decode())


def pixel_ledger(pred,truth,host):
    if pred.shape!=truth.shape:raise ValueError('Frozen native prediction and evaluator GT shapes differ')
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
    return dict(intersection=tp,union=tp+fp+fn,tp=tp,fp=fp,fn=fn,tn=tn,pred_fg=tp+fp,pred_bg=fn+tn,
        gt_fg=tp+fn,gt_bg=fp+tn,removed_host_FP=int((host&~truth&~pred).sum()),added_FP=int((~host&~truth&pred).sum()),
        recovered_host_FN=int((~host&truth&pred).sum()),removed_host_TP=int((host&truth&~pred).sum()))


def evaluate_frozen(outputs,records,labels_path):
    # Sole evaluator-label opening site: every field already detached/cloned.
    labels=load_trusted(labels_path)
    raw_truth=torch.as_tensor(labels['native_mask'])
    if not torch.isfinite(raw_truth).all() or not ((raw_truth==0)|(raw_truth==1)).all():raise ValueError('native evaluator mask must be binary; no implicit ignore-label conversion')
    truth=raw_truth.bool()
    if tuple(truth.shape)!=tuple(outputs['host'].shape):raise ValueError('native GT must match host scoring resolution')
    host=(outputs['host']>0).cpu();results={}
    for name,field in outputs.items():
        pred=(field>0).cpu();entry=dict(prediction_bits=bits(pred),pixel_counts=dict(FG=int(pred.sum()),BG=int((~pred).sum())))
        ledger=pixel_ledger(pred,truth,host)
        if name!='host' and records[name]['state']=='ERROR':entry['ERROR_safe_host_ledger']=ledger
        else:entry['native_ledger']=ledger
        if 'original_mask' in labels:
            raw_original=torch.as_tensor(labels['original_mask'])
            if not torch.isfinite(raw_original).all() or not ((raw_original==0)|(raw_original==1)).all():raise ValueError('original evaluator mask must be binary')
            original=raw_original.bool()
            # Same native binary prediction, then bilinear binary resize >.5.
            po=lift_signed(pred.float(),original.shape)>.5;ho=lift_signed(host.float(),original.shape)>.5
            key='ERROR_safe_host_original_ledger' if name!='host' and records[name]['state']=='ERROR' else 'original_ledger'
            entry[key]=pixel_ledger(po,original,ho)
        results[name]=entry
    return results


def self_check():
    torch.set_num_threads(1);torch.manual_seed(2044)
    fg=torch.arange(16)%2==0;ref=torch.zeros(16,3);ref[:,0]=torch.where(fg,1.,-1.)
    qry=ref.clone();host=torch.linspace(-1,1,16).reshape(4,4)
    d=dict(reference_features=ref,reference_coverage=fg.float(),query_features=qry,query_geometry=qry.clone(),
        grid=[4,4],host_signed_field=host,feature_contract={'stream':'native','space':'toy_unit'})
    aliases=dict(Fs=ref,coverage=fg.float(),Fq=qry,query_geometry=qry.clone(),grid={'support':[4,4],'query':[4,4]},host_field=host,feature_contract=d['feature_contract'])
    assert torch.equal(canonical_episode(aliases)['reference_features'],ref)
    try:canonical_episode(dict(aliases,reference_features=ref+1));raise AssertionError('Conflicting aliases accepted')
    except ValueError:pass
    outputs,records,_=freeze_outputs(d,['transport','kde','balanced_ot','diffusion','metric'])
    assert all(records[k]['state']=='SOLVED' for k in ('transport','kde','balanced_ot','diffusion'))
    assert records['metric']['state']=='ERROR' and 'checkpoint' in records['metric']['error']
    original={k:v.clone() for k,v in outputs.items()}
    with tempfile.TemporaryDirectory(prefix='reference_feature_probe_cpu_') as folder:
        saved=Path(folder)/'input.pt';torch.save(aliases,saved)
        reloaded=canonical_episode(load_trusted(saved));assert torch.equal(reloaded['query_features'],qry)
        path=Path(folder)/'eval.pt';torch.save(dict(native_mask=fg.reshape(4,4),original_mask=fg.reshape(4,4)),path)
        scored=evaluate_frozen(outputs,records,path)
        assert scored['transport']['native_ledger']['fp']==0 and scored['transport']['native_ledger']['fn']==0
        assert 'native_ledger' not in scored['metric'] and 'ERROR_safe_host_ledger' in scored['metric']
        assert all(torch.equal(outputs[k],original[k]) for k in outputs)
    with tempfile.TemporaryDirectory(prefix='reference_feature_metric_export_cpu_') as folder:
        root=Path(folder);trainer=metric_loader();contract={'coordinates':'synthetic_unit6','projection_id':'toy_identity','stream':'native'}
        split_rows={'train':[],'development':[]};first_episode=None
        for split,cid in (('train',0),('development',20)):
            for j in range(2):
                cov=(torch.arange(64)%8<4).float();truth_patch=cov.reshape(8,8).bool()
                fs=torch.randn(64,6);fq=torch.randn(64,6);fs[:,0]+=(cov*2-1)*.8;fq[:,0]+=(cov*2-1)*.7
                fs=F.normalize(fs,dim=1);fq=F.normalize(fq,dim=1)
                truth=F.interpolate(truth_patch.float()[None,None],(16,16),mode='nearest')[0,0]
                native_host=.08*(truth*2-1)+torch.randn(16,16)*.14
                roles={'support':[split+'-s-'+str(j)],'query':[split+'-q-'+str(j)]}
                payload=dict(Fs=fs,coverage=cov,Fq=fq,host_field=native_host,query_truth=truth,grid=[8,8],classid=cid,roles_photo_ids=roles,feature_contract=contract)
                path=root/(split+str(j)+'.pt');torch.save(payload,path)
                split_rows[split].append(dict(path=path.name,classid=cid,roles_photo_ids=roles))
                if first_episode is None:first_episode=payload
        manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema='demo9_metric_training_v1',authorization='AUTHORIZED_FUTURE',tensor_root=str(root),feature_contract=contract,splits=split_rows)))
        config=replace(trainer.MetricConfig(),rank=2,temperature=.2,regularization=1.,bandwidth=.4,max_anchors_per_class=8,positives=1,negatives=1,exclusion_radius=1)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(manifest,root/'trained','fixed_global',config,epochs=1,learning_rate=.02,device=torch.device('cpu'))
            trainer.export_deployment(root/'trained/best.pt',root/'frozen.pt')
        inference={k:first_episode[k] for k in ('Fs','coverage','Fq','host_field','grid','feature_contract')}
        inference['query_geometry']=inference['Fq'].clone()
        field,record,_=freeze_outputs(inference,['metric'],metric_checkpoint=root/'frozen.pt')
        assert record['metric']['state']=='SOLVED' and record['metric']['audit']['deployment_variant']=='fixed_global'
        assert record['metric']['audit']['state']=='FIXED_GLOBAL' and record['metric']['audit']['deployment_frozen']
        invalid=load_trusted(root/'frozen.pt');invalid['variant']='unknown_control';torch.save(invalid,root/'bad_variant.pt')
        _,rejected,_=freeze_outputs(inference,['metric'],metric_checkpoint=root/'bad_variant.pt')
        assert rejected['metric']['state']=='ERROR'
    bad=dict(d,reference_coverage=torch.zeros(16));preserved,fallback,_=freeze_outputs(bad,['transport','metric','diffusion'])
    assert all(torch.equal(v,host) for v in preserved.values())
    assert all(r['state']=='FALLBACK_DEGENERATE_SUPPORT' for r in fallback.values())
    invalid=dict(d,query_features=qry.clone());invalid['query_features'][0,0]=float('nan')
    _,errors,_=freeze_outputs(invalid,['transport','diffusion']);assert all(r['state']=='ERROR' for r in errors.values())
    failed=False
    try:device_gate('cuda',False,None)
    except ValueError:failed=True
    assert failed
    print(json.dumps(dict(state='CPU_FAKE_FEATURE_SELF_CHECK_PASSED',methods_frozen_before_evaluator=True,
        metric_random_dictionary_forbidden=True,fresh_trained_checkpoint_variant_preserved=True,unknown_metric_variant_ERROR=True,long_short_aliases_checked=True,saved_tensor_input_roundtrip=True,degenerate_support_host_exact=True,numerical_errors_explicit=True,
        ledger_and_bits=True,CUDA_without_guard_refused=True,real_DINO_or_task_gain_claim=False)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path)
    ap.add_argument('--evaluator-labels',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--methods',default='transport,kde,balanced_ot,diffusion')
    ap.add_argument('--metric-checkpoint',type=Path)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--allow-gpu',action='store_true')
    ap.add_argument('--resource-guard-state',type=Path)
    ap.add_argument('--max-ram-bytes',type=int,default=512*1024**2)
    ap.add_argument('--self-check',action='store_true')
    a=ap.parse_args()
    if a.self_check:self_check();return
    if not all((a.input,a.evaluator_labels,a.out)) or a.max_ram_bytes<=0:ap.error('--input --evaluator-labels --out and positive RAM budget required')
    methods=a.methods.split(',')
    if len(set(methods))!=len(methods) or not methods or any(k not in METHODS for k in methods):ap.error('unique supported methods required')
    if a.input.resolve()==a.evaluator_labels.resolve():ap.error('evaluator labels MUST be a separate file')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out.parent).free<5*1024**3:raise RuntimeError('Disk guard: need >=5GiB free')
    device=device_gate(a.device,a.allow_gpu,a.resource_guard_state)
    d=canonical_episode(load_trusted(a.input));validate_episode(d,len(methods),a.max_ram_bytes)
    for key in ('reference_features','reference_coverage','query_features','query_geometry','host_signed_field'):d[key]=d[key].to(device)
    started=time.monotonic();outputs,records,contract=freeze_outputs(d,methods,metric_checkpoint=a.metric_checkpoint,max_ram_bytes=a.max_ram_bytes)
    report=dict(state='METHOD_OUTPUTS_FROZEN',input=str(a.input),device=str(device),contract=contract,methods=records,
        episode_identity={k:json_audit(d[k]) for k in ('dataset','fold','seed','e','c','classid','photo_ids','roles_photo_ids') if k in d},
        metric_checkpoint=str(a.metric_checkpoint) if a.metric_checkpoint else None,query_gt_passed_to_methods=False,
        encoder_cost_excluded=True,scope='Prepared feature executor, native closed diffusion control; no claim of all-method adaptation or new real gain',
        source_sha256={str(a.input):file_sha(a.input),str(Path(__file__)):file_sha(Path(__file__))})
    if a.metric_checkpoint:report['source_sha256'][str(a.metric_checkpoint)]=file_sha(a.metric_checkpoint)
    if 'metric' in methods:
        source=Path(__file__).resolve().parent/'train_reference_metric.py'
        report['source_sha256'][str(source)]=file_sha(source)
    for name in methods:
        source=TICS/({'transport':'unbalanced_transport','kde':'unbalanced_transport','balanced_ot':'unbalanced_transport','metric':'reference_metric','diffusion':'reference_diffusion'}[name]+'.py')
        report['source_sha256'][str(source)]=file_sha(source)
    # Evaluator is opened only NOW, after even failed outputs have frozen explicit states.
    report['predictions']=evaluate_frozen(outputs,records,a.evaluator_labels)
    report['evaluator_sha256']=file_sha(a.evaluator_labels)
    report['failure_count']=sum(v['state']=='ERROR' for v in records.values())
    report['fallback_count']=sum(v['state'].startswith('FALLBACK') for v in records.values())
    report['state']='COMPLETED_WITH_ERRORS' if report['failure_count'] else 'COMPLETED'
    report['elapsed_s']=time.monotonic()-started
    tmp=a.out.with_suffix(a.out.suffix+'.tmp');tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(a.out)
    print(json.dumps(dict(state=report['state'],failure_count=report['failure_count'],fallback_count=report['fallback_count'],
        native_IU={k:[v['native_ledger']['intersection'],v['native_ledger']['union']] for k,v in report['predictions'].items() if 'native_ledger' in v})))
    if report['failure_count']:raise SystemExit(2)


if __name__=='__main__':main()
