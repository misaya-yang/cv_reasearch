#!/usr/bin/env python3
"""E6 fixed native COCO one-shot UOT with full-anchor KDE/balanced-OT controls.

No pools/training/downloads/cache replay. Requires a CPU-prepared fixed manifest,
source receipt and an owning CUDA resource guard. Unit native final descriptors
are FP32-normalized and debiased by the ACTUAL unchanged INSID3 U500 basis;
all three readers share them. Their spatial reader is NONE: bilinear posterior
lift and fixed .5 threshold only. Actual INSID3 remains a distinct host baseline.
Every mask freezes before the evaluator opens official query annotations.
Numerical errors are implementation ERROR (fatal), never scientific negatives.
"""
import argparse
import base64
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
from reference_feature_probe import device_gate,file_sha,json_audit,module

NAMES={'unbalanced':'uot','kde':'fullanchor_kde','balanced':'balanced_ot'}


def fixed_selection(episodes,limit):
    classes=sorted({int(r[0]) for r in episodes});by={c:[] for c in classes}
    for e,row in enumerate(episodes):by[int(row[0])].append((e,row))
    return [(e,row) for turn in range(max(map(len,by.values()))) for c in classes
            for e,row in by[c][turn:turn+1]][:limit]


def source_receipt(path):
    value=json.loads(path.read_text())
    if value.get('state')!='PREPARED_SOURCE' or not value.get('files'):
        raise ValueError('CPU-prepared source guard with nonempty files is required')
    verified={}
    for name,sha in value['files'].items():
        f=Path(name)
        expected=sha['sha256'] if isinstance(sha,dict) else sha
        if not f.is_absolute() or not f.is_file() or file_sha(f)!=expected:
            raise RuntimeError('Prepared source differs: '+name)
        verified[str(f.resolve())]=expected
    return verified


def require_sources(verified,objects):
    for obj in objects:
        path=Path(inspect.getsourcefile(obj)).resolve()
        if str(path) not in verified:raise RuntimeError('Live core source missing from source guard: '+str(path))


def prepared_assets(path,fold,limit):
    value=json.loads(path.read_text())
    if value.get('state')!='PREPARED_ASSETS' or value.get('seed')!=0 or value.get('fold')!=fold or value.get('limit')!=limit:
        raise ValueError('Frozen prepared manifest seed/fold/limit/state differs')
    if len(value.get('frozen_episodes',[]))!=limit or not value.get('assets'):
        raise ValueError('Prepared fixed episode/assets coverage missing')
    for item in value['assets']:
        info=Path(item['path']).stat()
        if info.st_size!=item['size'] or info.st_mtime_ns!=item['mtime_ns']:
            raise RuntimeError('Prepared immutable asset changed: '+item['path'])
    return value


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def freeze_readers(reference,foreground,query,grid,shape,host,options):
    """No evaluator argument; numerical failure aborts before any query GT read."""
    predictions={'insid3_native':host.detach().clone().bool()};infos={}
    if not foreground.any() or foreground.all():
        for name in NAMES.values():
            predictions[name]=host.detach().clone().bool()
            infos[name]=dict(state='SOURCE_DEGENERATE_HOST_FALLBACK',
                            reason='No both annotated support foreground/background anchors')
    else:
        transport=module('unbalanced_transport')
        for mode,name in NAMES.items():
            reader_options=dict(options)
            balanced_cap=reader_options.pop('balanced_max_iterations',2000)
            if mode=='balanced':reader_options['max_iterations']=balanced_cap
            synchronize(query.device);started=time.monotonic()
            try:
                score,audit=transport.transport_scores(reference,foreground,query,mode=mode,**reader_options)
                field=F.interpolate((score-.5).reshape(1,1,*grid),shape,mode='bilinear',align_corners=False)[0,0]
                if not torch.isfinite(field).all():raise RuntimeError('Nonfinite dense posterior field')
                predictions[name]=(field>0).detach().clone()
                synchronize(query.device)
                infos[name]=dict(state='COMPLETED',wall_seconds=time.monotonic()-started,
                    iteration_cap=reader_options['max_iterations'],
                    foreground_patch_count=int(foreground.sum()),background_patch_count=int((~foreground).sum()),
                    patch_fg_count=int((score>.5).sum()),spatial_reader='NONE',
                    dense_reader='bilinear signed(score-.5), align_corners=False; >0',audit=json_audit(audit))
            except Exception as exc:
                raise RuntimeError('E6 reader '+name+' failed: '+repr(exc)) from exc
    return predictions,infos


def evaluate_frozen(predictions,truth,original_truth):
    """Evaluator receives already frozen masks; no evaluator values return to reader."""
    row=dict(iu={},original_iu={},pixels={},confusion={},prediction_bits={})
    base=predictions['insid3_native']
    for name,pred in predictions.items():
        if pred.dtype!=torch.bool or pred.shape!=truth.shape:raise ValueError('Frozen prediction shape/type mismatch')
        row['iu'][name]=[int((pred&truth).sum()),int((pred|truth).sum())]
        original=F.interpolate(pred.reshape(1,1,*pred.shape).float(),tuple(original_truth.shape),mode='bilinear',align_corners=False)[0,0]>.5
        row['original_iu'][name]=[int((original&original_truth).sum()),int((original|original_truth).sum())]
        row['pixels'][name]=dict(recovered_fn=int((pred&truth&~base).sum()),lost_tp=int((base&truth&~pred).sum()),
            added_fp=int((pred&~truth&~base).sum()),removed_fp=int((base&~truth&~pred).sum()))
        row['confusion'][name]=dict(tp=int((pred&truth).sum()),fp=int((pred&~truth).sum()),
                                    fn=int((~pred&truth).sum()),tn=int((~pred&~truth).sum()))
        packed=np.packbits(pred.cpu().numpy().reshape(-1),bitorder='big')
        row['prediction_bits'][name]=dict(shape=list(pred.shape),bitorder='big',base64=base64.b64encode(packed).decode())
    return row


def summarize(records,key='iu'):
    sums={}
    for row in records:
        for name,(i,u) in row[key].items():
            pair=sums.setdefault(name,{}).setdefault(row['c'],[0,0]);pair[0]+=i;pair[1]+=u
    return {name:100*float(np.mean([i/max(u,1) for i,u in cs.values()])) for name,cs in sums.items()}


def self_check():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(2044)
    for case in range(10):
        ref=F.normalize(torch.randn(16,8,generator=g,dtype=torch.float64),dim=1)
        query=F.normalize(torch.randn(16,8,generator=g,dtype=torch.float64),dim=1);fg=torch.arange(16)<8;host=fg.reshape(4,4).repeat_interleave(2,0).repeat_interleave(2,1)
        opts=dict(epsilon=.1,rho_query=.1,rho_reference=.1,chunk_size=4,max_iterations=200,
                  tolerance=1e-5,max_ram_bytes=1024**2,require_convergence=True,balanced_max_iterations=2000)
        pred,info=freeze_readers(ref,fg,query,(4,4),(8,8),host,opts)
        assert len(pred)==4 and all(p.dtype==torch.bool and p.shape==(8,8) for p in pred.values())
        assert all(x['state']=='COMPLETED' and x['audit']['converged'] for x in info.values())
        frozen={k:v.clone() for k,v in pred.items()}
        evaluated=evaluate_frozen(pred,host,host)
        assert all(torch.equal(pred[k],frozen[k]) for k in pred)
        assert evaluated['iu']['insid3_native']==[32,32]
        assert evaluated['pixels']['insid3_native']==dict(recovered_fn=0,lost_tp=0,added_fp=0,removed_fp=0)
        fallback,audit=freeze_readers(ref,torch.zeros_like(fg),query,(4,4),(8,8),host,opts)
        assert all(torch.equal(x,host) for x in fallback.values())
    # A sharper numerical control: fixed identity geometry, FP32, no task GT.
    sharp=torch.Generator().manual_seed(2044);sharp_iterations=[]
    for case in range(10):
        r=F.normalize(torch.randn(16,8,generator=sharp,dtype=torch.float64),dim=1).float()
        _,audit=module('unbalanced_transport').transport_scores(r,fg,r,mode='balanced',epsilon=.1,
            max_iterations=2000,chunk_size=4,tolerance=1e-5,require_convergence=True)
        sharp_iterations.append(audit['iterations'])
    try:
        freeze_readers(ref,fg,query,(4,4),(8,8),host,{**opts,'max_ram_bytes':1})
    except RuntimeError as exc:assert 'failed' in str(exc)
    else:raise AssertionError('Numerical/working-memory failure silently skipped')
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);f=root/'source.py';f.write_text('pass\n')
        r=root/'receipt.json';r.write_text(json.dumps(dict(state='PREPARED_SOURCE',files={str(f):file_sha(f)})))
        source_receipt(r);f.write_text('raise ValueError\n')
        try:source_receipt(r)
        except RuntimeError:pass
        else:raise AssertionError('Source mutation not rejected')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'synthetic_basis.pt';basis=torch.eye(1024)[:,:500]
        torch.save(dict(state='NATIVE_BASIS_FROZEN',source_input='normalized_black_image',basis=basis),path)
        class FakeHost:
            svd_components=500
            def _build_positional_basis(self,device):return torch.zeros(1024,500)
        original=FakeHost._build_positional_basis
        with torch.no_grad(),module('native_assets').reuse_native_basis(FakeHost,path) as receipt:
            assert torch.equal(FakeHost()._build_positional_basis('cpu'),basis)
            assert receipt['source_input']=='normalized_black_image'
        assert FakeHost._build_positional_basis is original
        assert FakeHost()._build_positional_basis('cpu').sum()==0
    assert [e for e,_ in fixed_selection([(4,'q',['s']),(0,'q',['s']),(4,'q',['s'])],3)]==[1,0,2]
    print(json.dumps(dict(state='CPU_CHECKS_PASSED',fake_feature_episodes=10,numerical_error_fatal=True,
                         degenerate_support_exact_host=True,predictions_frozen_before_evaluator=True,
                         source_mutation_rejected=True,shared_basis_scoped_restoration=True,balanced_sharp_FP32_cases=10,balanced_sharp_max_actual_iterations=max(sharp_iterations),
                         balanced_iteration_cap=2000,UOT_iteration_cap=200,GPU_calls=0,dataset_or_model_loads=0,real_gain_claim=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true');p.add_argument('--prepared-root',type=Path,default=Path('/root/autodl-tmp/demo9'))
    p.add_argument('--prepared-manifest',type=Path);p.add_argument('--source-guard',type=Path)
    p.add_argument('--projection-basis',type=Path,help='Optional source-faithful NATIVE_BASIS_FROZEN normalized-black U500')
    p.add_argument('--out',type=Path);p.add_argument('--fold',type=int,default=0);p.add_argument('--limit',type=int,default=10)
    p.add_argument('--allow-gpu',action='store_true');p.add_argument('--resource-guard-state',type=Path)
    p.add_argument('--epsilon',type=float,default=.1);p.add_argument('--rho',type=float,default=.1)
    p.add_argument('--balanced-max-iterations',type=int,default=2000,help='Separately preregistered strong-control cap; UOT retains200')
    p.add_argument('--max-iterations',type=int,default=200);p.add_argument('--tolerance',type=float,default=1e-5)
    p.add_argument('--chunk-size',type=int,default=256);p.add_argument('--max-workspace-mib',type=int,default=256)
    a=p.parse_args()
    if a.self_check:self_check();return
    if not all((a.out,a.prepared_manifest,a.source_guard)):p.error('--out --prepared-manifest --source-guard required')
    if not 0<=a.fold<4 or not 1<=a.limit<=20:p.error('Bounded frozen fold0..3/limit1..20 only')
    if not math.isfinite(a.epsilon) or a.epsilon<=0 or not math.isfinite(a.rho) or a.rho<0:
        p.error('Finite positive epsilon and nonnegative rho required')
    if not 1<=a.max_iterations<=200 or not 0<a.tolerance<1 or not 1<=a.chunk_size<=4096 or a.max_workspace_mib<=0 or not 1<=a.balanced_max_iterations<=2000:
        p.error('Prepared bounds: UOTiterations1..200, balanced1..2000, tolerance(0,1), chunk1..4096, positive workspace')
    if a.out.exists() and any(a.out.iterdir()):raise RuntimeError('Fresh output directory required')
    a.out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out).free<5*1024**3:raise RuntimeError('At least5GiB disk headroom required')
    started=time.monotonic();report=dict(state='PREPARING',records=[],seed=0,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        contract='Actual native encoder/U500; shared full anchors/cosine; no spatial reader; query GT evaluator only',
        scope='Development E6 pilot, not all-fold method score; no oracle-GT method or task-gain guarantee',
        source_sha256={str(f):file_sha(f) for f in (Path(__file__),HERE/'reference_feature_probe.py',
            HERE.parent/'tics/unbalanced_transport.py',HERE.parent/'tics/native_assets.py')})
    def save():
        report['class_miou']=summarize(report['records']);report['original_class_miou']=summarize(report['records'],'original_iu')
        report['elapsed_concurrent_seconds']=time.monotonic()-started
        tmp=a.out/'report.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(a.out/'report.json')
    try:
        # No torch GPU discovery or model import before all receipts pass.
        prepared=prepared_assets(a.prepared_manifest,a.fold,a.limit);verified=source_receipt(a.source_guard)
        if a.projection_basis and str(a.projection_basis.resolve()) not in verified:
            raise RuntimeError('Shared basis not included in immutable source/asset guard')
        report['prepared_manifest_sha256']=file_sha(a.prepared_manifest);report['source_guard_sha256']=file_sha(a.source_guard)
        save();device=device_gate('cuda',a.allow_gpu,a.resource_guard_state)
        torch.set_num_threads(2);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(a.prepared_root/'scripts'));import _paths
        sys.path.insert(0,_paths.DEMO4)
        from icx.common import build_model,coco_episodes,TimmDINOv3
        from models.insid3 import INSID3
        from utils.data import load_image,load_mask,downsample_mask
        from PIL import Image
        from timm.models.eva import Eva,EvaAttention
        require_sources(verified,[build_model,TimmDINOv3,INSID3,load_image,load_mask,downsample_mask,Eva,EvaAttention])
        for path in (Path(__file__),HERE/'reference_feature_probe.py',HERE.parent/'tics/unbalanced_transport.py',HERE.parent/'tics/native_assets.py'):
            if str(path.resolve()) not in verified:raise RuntimeError('Executor/solver missing from source guard: '+str(path))
        episodes,_,base=coco_episodes(a.fold,400,shot=1,seed=0);selected=fixed_selection(episodes,a.limit)
        frozen=[dict(e=e,c=c,support=refs[0],query=q) for e,(c,q,refs) in selected]
        if frozen!=prepared['frozen_episodes']:raise RuntimeError('Live official COCO draws differ from frozen CPU manifest')
        if Path(_paths.COCO_ANN).resolve()!=Path(prepared['official_mask_root']).resolve():raise RuntimeError('Official mask root differs')
        report['frozen_episodes']=frozen;report['verified_core_sources']=verified;save()
        reuse_native_basis=module('native_assets').reuse_native_basis
        with torch.no_grad(),reuse_native_basis(INSID3,a.projection_basis) as basis_receipt:
            host=build_model()
        host.eval().requires_grad_(False)
        report['shared_native_basis_receipt']=basis_receipt
        report['native_basis_source_input']='normalized_black_image, actual source zeros; never Gaussian'
        if host.positional_basis.shape!=(1024,500):raise RuntimeError('Actual native U500 dimensions differ')
        import timm
        if not isinstance(host.encoder.m,Eva) or not isinstance(host.encoder.m.blocks[0].attn,EvaAttention):raise RuntimeError('Native Eva implementation differs')
        report.update(state='RUNNING',timm_version=timm.__version__,basis_shape=list(host.positional_basis.shape),
            descriptor_contract='Actual host extraction BF16 autocast; FP32 normalized final features; actual host._debias_features(U500) in FP32; same for all readers',
            host_contract='Unchanged official host.predict_mask full native recipe, actual host native precision; no claim reader reproduces host clustering')
        save();opts=dict(epsilon=a.epsilon,rho_query=a.rho,rho_reference=a.rho,chunk_size=a.chunk_size,
            max_iterations=a.max_iterations,balanced_max_iterations=a.balanced_max_iterations,tolerance=a.tolerance,max_ram_bytes=a.max_workspace_mib*1024**2,require_convergence=True)
        with torch.inference_mode():
            for e,(c,q,refs) in selected:
                episode_started=time.monotonic();report['active_episode']=dict(e=e,c=c);save()
                si=Image.open(Path(base)/refs[0]).convert('RGB');qi=Image.open(Path(base)/q).convert('RGB')
                gold=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/Path(refs[0]).with_suffix('.png')))==c+1).copy())
                s=load_image(si,host._transform,device)[0];t=load_image(qi,host._transform,device)[0]
                sm=load_mask(gold,host.image_size,device)
                extraction_started=time.monotonic()
                raw=host._extract_features(torch.cat([s,t]).unsqueeze(0)).float()
                norm=F.normalize(raw,p=2,dim=2);debiased=host._debias_features(norm)
                grid=tuple(raw.shape[-2:]);features=debiased[0].flatten(2).transpose(1,2).contiguous()
                fg=downsample_mask(sm.unsqueeze(1),*grid).reshape(-1).bool() if sm.any() else torch.zeros(grid[0]*grid[1],device=device,dtype=torch.bool)
                synchronize(device);feature_seconds=time.monotonic()-extraction_started
                baseline_started=time.monotonic()
                baseline=host.predict_mask(s,sm,t).reshape(host.image_size,host.image_size).bool() if sm.any() else torch.zeros((host.image_size,host.image_size),device=device,dtype=torch.bool)
                synchronize(device);baseline_seconds=time.monotonic()-baseline_started
                predictions,infos=freeze_readers(features[0],fg,features[1],grid,tuple(baseline.shape),baseline,opts)
                # Freeze EVERY arm before query annotation is first opened.
                predictions={k:v.detach().clone() for k,v in predictions.items()}
                qgt=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/Path(q).with_suffix('.png')))==c+1).copy()).to(device)
                truth=load_mask(qgt,host.image_size,device)[0].bool()
                row=evaluate_frozen(predictions,truth,qgt)
                row.update(e=e,c=c,support=refs[0],query=q,readers=infos,feature_seconds=feature_seconds,
                    host_seconds=baseline_seconds,episode_seconds=time.monotonic()-episode_started,
                    support_foreground_patches=int(fg.sum()),support_background_patches=int((~fg).sum()),grid=list(grid),
                    predictions_frozen_before_query_gt=True)
                report['records'].append(row);report.pop('active_episode',None)
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved();save()
                print(json.dumps(dict(state='RUNNING',e=e,c=c,completed=len(report['records']),class_miou=report['class_miou'],episode_seconds=row['episode_seconds'])),flush=True)
                del si,qi,gold,s,t,sm,raw,norm,debiased,features,fg,baseline,predictions,infos,qgt,truth
        report.update(state='COMPLETED',failure_count=0,scientific_outcome='Measured controls only; negative gains are healthy completed experiments');save()
    except BaseException as exc:
        report.update(state='ERROR',failure_count=1,error=repr(exc),traceback=traceback.format_exc(),stage_fatal=True);save()
        raise


if __name__=='__main__':main()
