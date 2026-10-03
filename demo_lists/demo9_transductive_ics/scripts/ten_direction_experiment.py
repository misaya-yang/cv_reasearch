#!/usr/bin/env python3
"""One finite eight-DEV-pair batch of the user's ten independent proposals.

Only root starts CUDA. Source replay always uses the public FoRIS pipeline.
Every method and its naive controls freezes before query labels are opened.
Per-method engineering errors are preserved, never replaced by native scores.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULES = ('ten_correspondence','ten_identity_competition','ten_grouping',
           'ten_observation','ten_crossfit_layers')
COST_LEDGER = {
    'scope':'Static operation counts, not measured 1024-DINO wall time',
    'common':dict(native_B2_per_pair=1,native_full_decodes_per_pair=3,
                  middle_layers_extra_encoder_calls=0),
    'm01':dict(extra_B2=0,full_cached_decodes=2,
               operation='same-role conditional three-point retrieval and independent-bank control'),
    'm02':dict(extra_B2=0,full_cached_decodes_at_most=6,
               operation='three shared BG-hypothesis full replays, method/point-naive/refund-only controls'),
    'm03':dict(extra_B2=0,full_cached_decodes=2,
               operation='reference-only split-cost CV and query full average-linkage tree DP; N squared transient distance'),
    'm04':dict(extra_B2=0,full_cached_decodes=2,
               operation='bounded part hypotheses, occlusion/tree messages, source seed-prior replay'),
    'm05':dict(extra_B2_at_most=2,child_full_decodes_at_most=6,additional_common_refinements_at_most=2,
               operation='aligned/native-box new paired views; method/naive share their view contexts'),
    'm06':dict(extra_B2=0,full_cached_decodes=3,
               operation='up to three source clusters, independent prior maxima, naive and K1 identity'),
    'm07':dict(extra_B2=0,full_cached_decodes=2,
               operation='four spatially heldout BG banks, deterministic four-mode fitting and no-holdout control'),
    'm08':dict(extra_B2=0,full_cached_decodes=3,
               operation='three-layer chunked shared/late/concat correspondence; no N squared disk cache'),
    'm09':dict(extra_B2=0,full_cached_decodes=3,
               operation='fixed foreground/background modes, conditional/symmetric/mean-background competitors'),
    'm10':dict(extra_B2_at_most=3,child_full_decodes_at_most=9,additional_common_refinements_at_most=2,
               operation='three extra phases, support-only monotone calibration, shift-average/inverse share observations'),
    'planned_all_active_8_pairs':dict(B2_calls_at_most=48,encoded_images_at_most=96,
                                    full_decodes_at_most=328),
    'hard_guard_only_not_expected':dict(extra_B2_per_pair=8,total_B2_8_pairs=72),
    'time_policy':'No real timing exists. After explicit GPU authorization, use first-case per-arm encoder/decode/solver clocks to judge whether remaining cases fit 1800s; do not use query GT to change budget or parameters.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False,indent=1)+'\n')
    temporary.replace(path)


def registry():
    sys.path.insert(0,str(ROOT))
    found = []
    for module_name in MODULES:
        mod = importlib.import_module('tics.'+module_name)
        entries = mod.METHODS
        if not isinstance(entries,list):
            raise ValueError('Each worker exports a literal METHODS list')
        for item in entries:
            item=dict(item)
            if 'card' not in item and 'card4lines' in item:
                item['card']=item['card4lines']
            if not all(k in item for k in ('id','function','naive','card')):
                raise ValueError('Method, same-information naive, and four-line card required')
            if len(item['card']) != 4 or any(not isinstance(s,str) or not s for s in item['card']):
                raise ValueError('Four nonempty pre-registered lines required')
            if not callable(item['function']) or not callable(item['naive']):
                raise ValueError('Actual executable method/control required')
            found.append({**item,'module':module_name})
    found.sort(key=lambda r:r['id'])
    if [r['id'] for r in found] != ['m%02d'%i for i in range(1,11)]:
        raise ValueError('Exactly ten distinct independent user proposals required')
    return found


def prepare(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError('CPU preflight initialized CUDA')
    parent = json.loads(a.from_manifest.read_text())
    rows = parent.get('frozen_episodes',parent.get('episodes',[]))
    selected = []
    for j in range(2):
        for f in range(4):
            fold = [r for r in rows if r['fold']==f]
            if len(fold)<2: raise ValueError('Two old DEV episodes per fold required')
            selected.append({k:fold[j][k] for k in ('fold','e','c','support','query')})
    source = Path(parent['foris_root'])
    sys.path.insert(0,str(source))
    from models.foris import FoRIS
    from utils.data import build_transform
    # Verify real production public preprocessing without constructing a model.
    host=FoRIS.__new__(FoRIS);torch.nn.Module.__init__(host)
    host.device=torch.device('cpu');host.image_size=1024;host._transform=build_transform(1024)
    host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
    assets = []
    for row in selected:
        sp=Path(parent['data_root'])/row['support']
        qp=Path(parent['data_root'])/row['query']
        mp=Path(parent['annotation_root'])/Path(row['support']).with_suffix('.png')
        with Image.open(sp) as im: support=im.convert('RGB')
        with Image.open(qp) as im: query=im.convert('RGB')
        reference=torch.from_numpy((np.asarray(Image.open(mp))==row['c']+1).copy())
        host.set_reference(support,reference);host.set_target(query)
        if not torch.equal(host._ref_images,host._transform(support)[None]):
            raise RuntimeError('Actual public reference transform drift')
        if not torch.equal(host._tgt_image,host._transform(query)):
            raise RuntimeError('Actual public target transform drift')
        if not host._ref_masks.any(): raise ValueError('No legal reference foreground')
        row['support_original_hw']=[support.height,support.width]
        row['query_original_hw']=[query.height,query.width]
        host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
        for path in (sp,qp,mp):
            st=path.stat();assets.append(dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns))
    methods=registry()
    receipt=json.loads(a.cpu_receipt.read_text())
    if receipt.get('state')!='CPU_TEN_CONTEXT_PASSED' or receipt.get('cases')!=10 or receipt.get('CUDA_initialized'):
        raise ValueError('Actual public-source CPU ten-case context receipt required')
    integrated=json.loads(a.integration_receipt.read_text())
    if (integrated.get('state')!='CPU_ALL_TEN_METHODS_PASSED' or
        integrated.get('methods')!=10 or integrated.get('CUDA_initialized') is not False):
        raise ValueError('Every method/control must compose with the actual public source CPU fixture')
    refreshed={}
    refresh_receipts=[]
    for path in a.source_refresh_receipt:
        refresh=json.loads(path.read_text())
        if (refresh.get('state')!='CPU_SOURCE_REFRESH_PASSED' or
            refresh.get('actual_source_execution') is not True or
            refresh.get('CUDA_initialized') is not False or refresh.get('no_query_GT') is not True):
            raise ValueError('Actual CPU-only source composition proof required for changed runtime module')
        passed={(r['id'],r['role']) for r in refresh.get('records',[]) if r.get('state')=='PASSED'}
        for source_path,current_hash in refresh.get('source_hashes',{}).items():
            module=Path(source_path).stem
            affected=[method for method in methods if method['module']==module]
            if not affected or sha(source_path)!=current_hash:
                raise ValueError('Refresh must identify a current runtime module in this batch')
            required={(method['id'],role) for method in affected for role in ('naive','method')}
            required|={(method['id'],name) for method in affected for name in method.get('controls',{})}
            if not required<=passed:raise ValueError('Changed module lacks every method/naive/control composition proof')
            refreshed[source_path]=current_hash
        refresh_receipts.append(dict(path=str(path.resolve()),sha256=sha(path)))
    runtime_sources={str((ROOT/'tics'/(name+'.py')).resolve()) for name in MODULES}
    runtime_sources.add(str((ROOT/'tics/ten_direction_context.py').resolve()))
    runtime_sources.add(str((HERE/'ten_direction_all_methods_cpu.py').resolve()))
    for path,digest in integrated['source_hashes'].items():
        if path not in runtime_sources:continue
        if sha(path)!=digest and refreshed.get(path)!=sha(path):
            raise ValueError('Method source changed since actual public composition smoke without affected-role proof')
    files=[Path(__file__),ROOT/'tics/ten_direction_context.py',HERE/'ten_direction_context_cpu.py',
           HERE/'ten_direction_all_methods_cpu.py',ROOT/'tics/native_assets.py',Path('/root/autodl-tmp/demo4/icx/common.py')]
    files += [ROOT/'tics'/(m+'.py') for m in MODULES]
    files += [source/'models/foris.py',source/'utils/clustering.py',source/'utils/data.py',source/'utils/refinement.py']
    hashes={str(p.resolve()):sha(p) for p in files}
    if receipt.get('context_sha256')!=hashes[str((ROOT/'tics/ten_direction_context.py').resolve())]:
        raise ValueError('Context receipt predates actual executable context')
    for p in files: compile(p.read_text(),str(p),'exec')
    # Reuse metadata for the already-existing model; no weight hashing/loading.
    assets.extend(item for item in parent.get('assets',[]) if '/models/dinov3-vitl16-timm/' in item['path'])
    basis_stat=a.projection_basis.stat()
    assets.append(dict(path=str(a.projection_basis.resolve()),size=basis_stat.st_size,mtime_ns=basis_stat.st_mtime_ns))
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Keep at least 5GiB disk reserve')
    manifest=dict(schema='ten_direction_assets_v1',state='PREPARED',
        frozen_episodes=selected,data_root=parent['data_root'],annotation_root=parent['annotation_root'],
        foris_root=str(source),seed=parent.get('seed',0),source_hashes=hashes,assets=assets,
        cpu_receipt=str(a.cpu_receipt),cpu_receipt_sha256=sha(a.cpu_receipt),
        integration_receipt=str(a.integration_receipt),integration_receipt_sha256=sha(a.integration_receipt),
        source_refresh_receipts=refresh_receipts,
        methods=[dict(id=m['id'],module=m['module'],card=m['card']) for m in methods],
        max_extra_b2=8,fixed_layers=[6,12,'final'],
        scope='Eight reused old development pairs, two per fold; interface/decision signal, not an independent method score',
        public_rgb=True,reference_only=True,query_GT_opened_after_all_predictions=True,
        original_pipeline_and_CRF=True,no_model_or_data_download=True,
        readiness_limits=['CPU synthetic source/API composition is not a real DINO segmentation result',
                          'Production 1024-DINO cached/full-public identity and CRF remain unexecuted',
                          'Some tiny-source observation cases legitimately abstain; active GPU view paths remain unmeasured',
                          'No real runtime, gain, cross-fold method score or independent confirmation exists'],
        method6_control_limit='Primary and naive can retain different actual seed counts; success alone does not identify semantic gating benefit',
        source_context_cache='Actual original n=1 B2 final plus hooks6/12, each replacement-mask Part1 gate recomputed',
        CPU_preprocess_cases=len(selected),CUDA_initialized=False)
    if a.manifest.exists(): raise ValueError('Fresh prepared manifest required')
    write_json(a.manifest,manifest)
    # Generous finite wall cap, not a five-minute throughput promise. Root alone
    # starts this command; this manifest neither leases nor powers off a GPU.
    write_json(a.plan,dict(state='CPU_PREPARED_NOT_GPU_AUTHORIZED',platform='shared_server',
        cuda_launch_owner='root_only',foreign_jobs_protected=True,shutdown_armed=False,
        autostart=False,shutdown=False,executable=False,
        later_user_GPU_authorization_required=True,
        one_scientific_process=True,max_wall_seconds=a.budget_seconds,
        maximum_native_b2_calls=8,maximum_extra_b2_calls=64,
        GPU_command_omitted_until_authorized=True,
        minimum_disk_free_bytes=5*1024**3,prepared_manifest_sha256=sha(a.manifest),
        cost_ledger=COST_LEDGER,
        methods=[r['id'] for r in methods],
        stopping='A shared native/cache identity failure stops the stage; each mechanism error is saved and does not cancel independent methods. No automatic retries/sweeps.'))
    print(json.dumps(dict(state='CPU_PREPARED_NOT_GPU_AUTHORIZED',methods=10,pairs=8,CUDA_initialized=False,
                          autostart=False,shutdown=False,executable=False)),flush=True)


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import traceback
    if not a.allow_gpu or os.environ.get('DEMO9_TEN_ROOT_LAUNCH')!='1':
        raise RuntimeError('Only root may start the prepared finite GPU batch')
    m=json.loads(a.manifest.read_text())
    if m.get('state')!='PREPARED' or m.get('schema')!='ten_direction_assets_v1':
        raise ValueError('Frozen CPU-prepared complete ten-method manifest required')
    for p,h in m['source_hashes'].items():
        if sha(p)!=h: raise RuntimeError('Prepared source drift: '+p)
    for item in m['assets']:
        st=Path(item['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(item['size'],item['mtime_ns']):
            raise RuntimeError('Prepared existing asset drift')
    if a.out.exists() and any(a.out.iterdir()): raise ValueError('Fresh finite output required')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'predictions').mkdir()
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,m['foris_root'])
    import models.foris as source
    sys.path.insert(0,'/root/autodl-tmp/demo4')
    from icx.common import TimmDINOv3
    from tics.native_assets import reuse_native_basis
    from tics.ten_direction_context import TenDirectionContext,ViewBudget,capture_exact_native_and_middle
    methods=registry();start=time.monotonic();records=[]
    report=dict(state='RUNNING',records=records,manifest_sha256=sha(a.manifest),
                no_query_GT_in_context=True,own_GPU_stage=True,methods=[r['id'] for r in methods],
                design_DEV=True,source_hashes=m['source_hashes'])
    def save():
        report['elapsed_s']=time.monotonic()-start;write_json(a.out/'report.json',report)
    save()
    try:
        with torch.inference_mode():
            encoder=TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(source.FoRIS,a.projection_basis):
                host=source.FoRIS(encoder=encoder,image_size=1024,svd_components=500,tau=.6,
                    mask_refiner='crf',resize_to_orig_size=False,device='cuda').eval().requires_grad_(False)
            for row in m['frozen_episodes']:
                if time.monotonic()-start>a.budget_seconds:
                    report['state']='COMPLETED_FINITE_BUDGET';break
                unit_start=time.monotonic()
                support=Image.open(Path(m['data_root'])/row['support']).convert('RGB')
                query=Image.open(Path(m['data_root'])/row['query']).convert('RGB')
                ref_mask=torch.from_numpy((np.asarray(Image.open(Path(m['annotation_root'])/Path(row['support']).with_suffix('.png')))==row['c']+1).copy())
                budget=ViewBudget(max_extra_b2=m['max_extra_b2'])
                def construct(support_rgb,reference_mask_original,query_rgb,shared_budget,metadata):
                    captured={}
                    try:
                        host.set_reference(support_rgb,reference_mask_original);host.set_target(query_rgb)
                        with capture_exact_native_and_middle(host,encoder) as captured:
                            uncached=host.segment().reshape(1024,1024).bool().clone()
                    finally:
                        shared_budget.actual_b2_calls+=captured.get('encoder_forward_calls',0)
                        host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
                    ctx=TenDirectionContext(host,encoder,support_rgb,query_rgb,reference_mask_original,
                        raw_native=captured['raw_native'],layer_maps={6:captured[6],12:captured[12]},
                        budget=shared_budget,new_view_callback=construct,coordinate_metadata=metadata)
                    replay=ctx.initialize_native()
                    if not torch.equal(uncached,replay['mask']):
                        raise RuntimeError('Actual unobserved public B2 vs cached public pipeline is not pixel exact')
                    ctx.native['uncached_public_exact']=True
                    return ctx
                budget.native_b2_calls=1
                ctx=construct(support,ref_mask,query,budget,None)
                predictions={'native':ctx.native['mask'].detach().cpu().numpy()}
                scores={'native':ctx.native['score'].detach().cpu().numpy()}
                audits={};errors={};timing={}
                def small_json(value):
                    if isinstance(value,torch.Tensor):
                        if value.numel()>2048:raise ValueError('Do not serialize large features in method audit')
                        value=value.detach().cpu().tolist()
                    if isinstance(value,np.ndarray):
                        if value.size>2048:raise ValueError('Do not serialize large arrays in method audit')
                        value=value.tolist()
                    if isinstance(value,np.generic):value=value.item()
                    if isinstance(value,dict):return {str(k):small_json(v) for k,v in value.items()}
                    if isinstance(value,(list,tuple)):return [small_json(v) for v in value]
                    json.dumps(value,allow_nan=False)
                    return value
                def freeze_result(name,result):
                    if not isinstance(result,dict) or 'mask' not in result:
                        raise ValueError('Every method must return a full working-grid prediction dict')
                    mask=result['mask']
                    if not isinstance(mask,torch.Tensor) or mask.shape!=(1024,1024) or mask.dtype!=torch.bool:
                        raise ValueError('Method did not preserve full-grid bool mask contract')
                    audit=small_json({**result.get('audit',{}),
                                      **({'method_audit':result['method_audit']} if 'method_audit' in result else {})})
                    predictions[name]=mask.detach().cpu().numpy().copy()
                    if isinstance(result.get('score'),torch.Tensor):
                        score=result['score']
                        if not torch.isfinite(score).all():raise ValueError('Nonfinite continuous method response')
                        if score.numel()<=4096:scores[name]=score.detach().cpu().numpy().copy()
                    audits[name]=audit
                    for control,value in result.get('controls',{}).items():
                        freeze_result(name+'__'+control,value)
                for entry in methods:
                    for role in ('naive','function'):
                        name=entry['id']+('_naive' if role=='naive' else '')
                        tick=time.monotonic()
                        try:
                            freeze_result(name,entry[role](ctx))
                        except Exception as exc:
                            errors[name]=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                        timing[name]=time.monotonic()-tick
                        print(json.dumps(dict(pair=[row['fold'],row['e'],row['c']],arm=name,
                                              state='ERROR' if name in errors else 'PREDICTION_FROZEN',
                                              elapsed_s=timing[name],extra_b2_calls=budget.extra_b2_calls)),flush=True)
                    for control, fn in entry.get('controls',{}).items():
                        name=entry['id']+'__'+control;tick=time.monotonic()
                        try:
                            freeze_result(name,fn(ctx))
                        except Exception as exc:
                            errors[name]=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                        timing[name]=time.monotonic()-tick
                # Freeze full predictions and small scores BEFORE any query annotation IO.
                stem='%d_%d_%d'%(row['fold'],row['e'],row['c']);path=a.out/'predictions'/(stem+'.npz')
                arrays={name:np.packbits(mask.reshape(-1)) for name,mask in predictions.items()}
                arrays.update({'score__'+name:value for name,value in scores.items()})
                np.savez_compressed(path,**arrays);frozen_hash=sha(path)
                truth=torch.from_numpy((np.asarray(Image.open(Path(m['annotation_root'])/Path(row['query']).with_suffix('.png')))==row['c']+1).copy())
                truth_model=F.interpolate(truth[None,None].float(),(1024,1024),mode='nearest')[0,0].bool().numpy()
                iu={};ledger={};oracle={}
                for name,mask in predictions.items():
                    original=F.interpolate(torch.from_numpy(mask)[None,None].float(),truth.shape,mode='nearest')[0,0].bool()
                    iu[name]=[int((original&truth).sum()),int((original|truth).sum())]
                    fp=mask&~truth_model;fn=~mask&truth_model
                    ledger[name]=dict(FP=int(fp.sum()),FN=int(fn.sum()),TP=int((mask&truth_model).sum()))
                    if name!='native':
                        baseline=predictions['native']
                        ledger[name].update(deleted_FP=int((baseline&~mask&~truth_model).sum()),lost_TP=int((baseline&~mask&truth_model).sum()),
                                            added_TP=int((~baseline&mask&truth_model).sum()),added_FP=int((~baseline&mask&~truth_model).sum()))
                        ideal=(baseline&mask)|(truth_model&(baseline^mask))
                        original_ideal=F.interpolate(torch.from_numpy(ideal)[None,None].float(),truth.shape,mode='nearest')[0,0].bool()
                        oracle[name]=[int((original_ideal&truth).sum()),int((original_ideal|truth).sum())]
                rec=dict(row=row,original_iu=iu,model_error_ledger=ledger,changed_pixel_oracle_iu=oracle,
                    method_errors=errors,method_audit=audits,arm_seconds=timing,
                    prediction_path=str(path),predictions_sha256=frozen_hash,
                    predictions_frozen_before_query_GT=True,full_native_exact=True,
                    encoder_b2_calls=budget.actual_b2_calls,
                    extra_b2_calls=budget.extra_b2_calls,extra_view_requests=budget.extra_b2_calls,
                    elapsed_s=time.monotonic()-unit_start)
                if sha(path)!=frozen_hash:raise RuntimeError('Frozen prediction artifact changed during scoring')
                records.append(rec);save()
                with open(a.out/'episodes.jsonl','a') as stream:stream.write(json.dumps(rec,allow_nan=False)+'\n')
                budget.cache.clear();ctx._cluster_cache.clear();del ctx
            else:
                report['state']='COMPLETED_WITH_METHOD_ERRORS' if any(r['method_errors'] for r in records) else 'COMPLETED'
        save()
    except BaseException as exc:
        report.update(state='ERROR_SHARED_PIPELINE',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());save();raise


def analyze(a):
    """Pair-preserving pilot comparisons, no GPU or feature access."""
    import numpy as np
    report=json.loads((a.out/'report.json').read_text())
    records=report['records']
    if not records:raise ValueError('No frozen episodes to analyze')
    if not all(r['predictions_frozen_before_query_GT'] and r['full_native_exact'] for r in records):
        raise ValueError('Frozen prediction/source identity evidence required')
    for r in records:
        if sha(r['prediction_path'])!=r['predictions_sha256']:
            raise ValueError('Frozen scientific output drift')
    names=sorted({n for r in records for n in r['original_iu']})
    def score(rows,name,oracle=False):
        by={}
        for r in rows:
            table=r['changed_pixel_oracle_iu'] if oracle else r['original_iu']
            i,u=table[name];c=r['row']['c'];old=by.get(c,(0,0));by[c]=(old[0]+i,old[1]+u)
        return 100*float(np.mean([i/max(u,1) for i,u in by.values()]))
    # Any repeated image role connects resampling units. These are old DEV
    # examples and uncertainty is descriptive; it is not a confirmatory test.
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(records):
        for role in ('support','query'):
            photo=r['row'][role]
            if photo in seen:parent[find(i)]=find(seen[photo])
            else:seen[photo]=i
    groups={}
    for i,r in enumerate(records):groups.setdefault(find(i),[]).append(r)
    groups=list(groups.values());rng=np.random.default_rng(2060)
    samples=[[r for j in rng.integers(0,len(groups),len(groups)) for r in groups[j]] for _ in range(2000)]
    result=dict(state='CPU_TEN_DIRECTION_ANALYZED',pairs=len(records),expected_pairs=8,
                source_report_state=report['state'],scope='Exploratory eight old DEV; no independent method-score claim',
                complete_all_folds=all(any(r['row']['fold']==f for r in records) for f in range(4)),
                photo_connected_groups=len(groups),methods={},bootstrap_seed=2060)
    for name in names:
        missing=[r['row'] for r in records if name not in r['original_iu']]
        if missing:
            result['methods'][name]=dict(state='INCOMPLETE',missing=missing)
            continue
        native=score(records,'native');value=score(records,name)
        deltas=[score(sample,name)-score(sample,'native') for sample in samples]
        item=dict(state='SCORED_ALL_COMPLETED_PAIRS',original_class_sum_mIoU_pp=value,
                  native_mIoU_pp=native,delta_pp=value-native,
                  image_connected_paired_CI95=np.quantile(deltas,[.025,.975]).tolist(),
                  folds={str(f):score([r for r in records if r['row']['fold']==f],name)-score([r for r in records if r['row']['fold']==f],'native')
                         for f in range(4) if any(r['row']['fold']==f for r in records)},
                  model_error_ledger={k:sum(r['model_error_ledger'][name].get(k,0) for r in records)
                                      for k in ('FP','FN','TP','deleted_FP','lost_TP','added_TP','added_FP')})
        if name!='native':item['GT_changed_pixel_oracle_mIoU_pp']=score(records,name,oracle=True)
        naive=name+'_naive'
        if naive in names and all(naive in r['original_iu'] for r in records):
            paired=[score(sample,name)-score(sample,naive) for sample in samples]
            item['same_information_naive_mIoU_pp']=score(records,naive)
            item['method_minus_naive_pp']=value-score(records,naive)
            item['method_minus_naive_CI95']=np.quantile(paired,[.025,.975]).tolist()
        result['methods'][name]=item
    result['method_errors']=[dict(row=r['row'],errors=r['method_errors']) for r in records if r['method_errors']]
    result['total_B2_encoder_calls']=sum(r['encoder_b2_calls'] for r in records)
    write_json(a.out/'analysis.json',result)
    print(json.dumps(dict(state=result['state'],pairs=len(records),errors=len(result['method_errors']))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--analyze',action='store_true')
    p.add_argument('--allow-gpu',action='store_true');p.add_argument('--from-manifest',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cpu-receipt',type=Path);p.add_argument('--integration-receipt',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--source-refresh-receipt',type=Path,action='append',default=[])
    p.add_argument('--projection-basis',type=Path,required=True)
    p.add_argument('--budget-seconds',type=int,default=1800)
    p.add_argument('--memory-fraction',type=float,default=.35)
    a=p.parse_args()
    if sum((a.prepare,a.run,a.analyze))!=1:p.error('Exactly one of prepare/run/analyze is required')
    if a.prepare:prepare(a)
    elif a.run:run(a)
    else:analyze(a)


if __name__=='__main__':main()
