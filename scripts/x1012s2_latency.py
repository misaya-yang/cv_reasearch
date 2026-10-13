"""Serial CPU2 cache-replay latency evidence; no scoring or saved masks.

Ten preselected domains, one task each. Timers return every official operation
unchanged. Inclusive and exclusive spans prevent charging nested/shared work
twice. Existing raw O24 replaces the encoder, so none of these measurements is
an RGB-to-mask end-to-end latency measurement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import types

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
sys.dont_write_bytecode=True

REPO=Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS=REPO.parent/'cv_data'
OUT=ASSETS/'a/x1012s2_latency'
RUNTIME=OUT/'runtime'
R5=ASSETS/'a/x1012_R5_validation'
R6=ASSETS/'a/x1012_R6_insid3'
DOMAINS=('coco','lvis','pascal_part','paco_part','suim','isaid','isic','lung','fundus','deepglobe_road')


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare():
    if (OUT/'config.json').exists(): return check()
    source=read(R5/'manifest.baselines.json')
    rows=[copy.deepcopy(next(b for b in source if b['dataset']==d)) for d in DOMAINS]
    insid={r['index']:r for r in read(R6/'r5_500/sealed.json')['records']}
    for b in rows: b['insid3_archive']=insid[b['index']]
    write(OUT/'manifest.json',rows)
    frozen=[]
    names=('lab.py','plan_common.py','p0_manifest.py','raw_feature_cache.py','ics/__init__.py',
           'ics/official_data.py','ics/native_basis.py','ics/m4_crf_cached.py')
    sources={name:R5/'runtime'/name for name in names}
    for model,root in [('foris',ASSETS/'third_party/foris_official'),('insid3',ASSETS/'third_party/INSID3')]:
        for folder in ('models','utils'):
            for p in sorted((root/folder).glob('*.py')):
                sources[str(Path(model)/folder/p.name)]=p
    for name,source in sources.items():
        target=RUNTIME/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        frozen.append(dict(source=str(source),snapshot=str(target),sha256=sha(target)))
    for model in ('foris','insid3'):
        for folder in ('models','utils'):
            p=RUNTIME/model/folder/'__init__.py'
            if not p.exists():
                p.write_text('"""Isolated official module namespace."""\n')
                frozen.append(dict(source='namespace marker',snapshot=str(p),sha256=sha(p)))
    target=OUT/'frozen/x1012s2_latency.py';target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),target)
    frozen.append(dict(source=str(Path(__file__).resolve()),snapshot=str(target),sha256=sha(target)))
    write(OUT/'runtime_sources.json',frozen)
    config=dict(state='FROZEN_COST_ONLY',n=10,selection='First manifest task in each named domain; fixed before latency measurement',
                cpu_threads=2,torch_interop_threads=1,workers=1,device='cpu',model_order=['foris','insid3'],
                warmup='One untimed first selected task per model; spatial CRF geometry remains warm; bilateral geometry cleared before timed task1',
                timed_passes=1,scope='Serial official complete output from existing O24; RGB/reference-mask decoding and cache I/O separately reported',
                query_GT_reads=0,new_encoder_calls=0,parameter_fitting=False,new_method_masks_saved=0,
                manifest_sha256=sha(OUT/'manifest.json'),runtime_sources_sha256=sha(OUT/'runtime_sources.json'),
                host=dict(platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count()),
                reference_basis_sha256=sha(ASSETS/'native_assets/positional_basis.pt'),
                source_manifests={str(R5/'manifest.baselines.json'):sha(R5/'manifest.baselines.json'),
                                  str(R6/'r5_500/sealed.json'):sha(R6/'r5_500/sealed.json')})
    write(OUT/'config.json',config)
    return config


def check():
    config=read(OUT/'config.json')
    assert sha(OUT/'manifest.json')==config['manifest_sha256']
    assert sha(OUT/'runtime_sources.json')==config['runtime_sources_sha256']
    for row in read(OUT/'runtime_sources.json'):assert sha(row['snapshot'])==row['sha256']
    return config


class Spans:
    def __init__(self): self.reset()
    def reset(self): self.stack=[];self.values=defaultdict(lambda:dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
    def wrap(self,name,fn):
        def call(*a,**kw):
            path='/'.join([f['name'] for f in self.stack]+[name]); frame=dict(name=name,child=0.)
            self.stack.append(frame);start=time.perf_counter()
            try:return fn(*a,**kw)
            finally:
                elapsed=time.perf_counter()-start;self.stack.pop()
                if self.stack:self.stack[-1]['child']+=elapsed
                v=self.values[path];v['calls']+=1;v['inclusive_seconds']+=elapsed;v['exclusive_seconds']+=elapsed-frame['child']
        return call


def setup(model):
    sys.path.insert(0,str(RUNTIME))
    import numpy as np
    import torch
    import lab
    lab.ROOT,lab.ASSETS=REPO,ASSETS;lab.BASIS=ASSETS/'native_assets/positional_basis.pt';lab.DEV=torch.device('cpu')
    import plan_common as pc
    sys.path.insert(0,str(RUNTIME))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    native=pc.extension('ics_permutohedral_m4',ASSETS/'runtime/macos/crf/ics_permutohedral_m4.so')
    prepared=pc.extension('ics_permutohedral_m4_prepared_v1',ASSETS/'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')
    class Layer(torch.nn.Module):
        def __init__(self,bilateral,theta_alpha,theta_beta,theta_gamma,nhwc=True):
            super().__init__();self.bilateral=bilateral;self.parameters_=(theta_alpha,theta_beta,theta_gamma)
        def forward(self,x,image):
            a=x.detach().float().permute(0,2,3,1).contiguous();b=image.detach().float().permute(0,2,3,1).contiguous()
            return native.forward(a,b,self.bilateral,*self.parameters_).permute(0,3,1,2).contiguous()
    replacement=types.ModuleType('PermutohedralFiltering');replacement.PermutohedralLayer=Layer
    sys.modules['PermutohedralFiltering']=replacement
    sys.path.insert(0,str(ASSETS/'third_party/crf_source/src'));sys.path.insert(0,str(RUNTIME/model))
    official=importlib.import_module('models.'+model)
    from ics.native_basis import reuse_native_basis
    from ics.m4_crf_cached import enable_on_crf
    from ics.official_data import load_inputs
    adapter=pc.Replay();Class=official.FoRIS if model=='foris' else official.INSID3
    kwargs=dict(encoder=adapter,image_size=1024,svd_components=500,tau=.6,mask_refiner='crf',resize_to_orig_size=False,device='cpu')
    if model=='insid3':kwargs.update(merge_threshold=.2,crf_size=640)
    began=time.perf_counter()
    with reuse_native_basis(Class,lab.BASIS):host=Class(**kwargs).eval().requires_grad_(False)
    enable_on_crf(host._crf,prepared);setup_seconds=time.perf_counter()-began
    spans=Spans()
    names=('_extract_features','_part1_positional_debias','_debias_features','_part2_background_suppression',
           '_part2_stage1_feature_gating','_part2_stage2_contrastive_score','_reference_contrastive_prototypes',
           '_part3_clustering','_locate_candidates','_build_seed_cluster_prior','_part4_semantic_consistency_correction',
           '_semantic_disagreement_penalty','_semantic_cluster_reweight_map','_binarize_response','_seed_and_aggregate',
           '_finalize_mask','predict' if model=='foris' else 'predict_mask')
    for name in names:
        if hasattr(host,name):setattr(host,name,spans.wrap(name,getattr(host,name)))
    for name in ('agglomerative_clustering','compute_cluster_prototypes','crf_refine'):
        setattr(official,name,spans.wrap(name,getattr(official,name)))
    einsum=torch.einsum
    def timed_einsum(equation,*operands):
        if equation.replace(' ','')=='bchw,bcxy->bhwxy':
            return spans.wrap('cross_image_similarity',einsum)(equation,*operands)
        return einsum(equation,*operands)
    torch.einsum=timed_einsum
    return pc,host,adapter,load_inputs,spans,setup_seconds


def measure(model):
    config=check()
    if (OUT/(model+'.json')).exists():return
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info
    pc,host,adapter,loader,spans,setup_seconds=setup(model)
    rows=read(OUT/'manifest.json')
    def run(b,record):
        spans.reset();start=time.perf_counter();adapter.bind(b)
        with pc.inference_only(b):
            tick=time.perf_counter();r,m,q=loader(b['row'],ASSETS);load=time.perf_counter()-tick
            tick=time.perf_counter();host.set_reference(r,m);host.set_target(q);preprocess=time.perf_counter()-tick
            tick=time.perf_counter()
            try:pred=host.segment()
            finally:
                if hasattr(host,'reset_state'):host.reset_state()
                else:host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None
            segment=time.perf_counter()-tick
        complete=time.perf_counter()-start
        assert pred.dtype==torch.bool and tuple(pred.shape)==(1024,1024)
        assert adapter.encoder is None and adapter.encoder_calls==0
        root='predict' if model=='foris' else 'predict_mask'
        extract=spans.values[root+'/_extract_features']['inclusive_seconds']
        post_o24=segment-extract
        # Archive parity is outside the timing region and never opens labels.
        if record:
            source=b['old_arms']['B'] if model=='foris' else b['insid3_archive']
            path=source['path'] if model=='foris' else source['prediction_path']
            key=source['cli_key'] if model=='foris' else 'cli/insid3.crf640'
            with np.load(path,allow_pickle=False) as z:old=pc.unpack(z[key])
            xor=int(np.count_nonzero(pred.numpy()!=old))
            result=dict(index=b['index'],dataset=b['dataset'],complete_cached_replay_seconds=complete,
                        RGB_and_reference_mask_load_seconds=load,preprocess_seconds=preprocess,segment_seconds=segment,
                        verified_O24_replay_seconds=extract,post_O24_compute_seconds=post_o24,
                        archive_CLI_mask_xor=xor,spans=dict(spans.values),new_encoder_calls=0,query_GT_reads=0,
                        masks_saved=0,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())
        else:result=None
        adapter.memo.clear();del pred
        return result
    began=time.perf_counter();run(rows[0],False);warmup=time.perf_counter()-began
    # Keep reusable spatial geometry, but do not let task1 reuse warmup RGB geometry.
    bilateral=host._crf.bilateral_filter
    bilateral._prepared_geometry=None;bilateral._prepared_key=None;bilateral._prepared_colours=None
    results=[];started=time.perf_counter()
    for i,b in enumerate(rows,1):
        result=run(b,True);results.append(result)
        print(dict(model=model,completed=i,total=10,dataset=b['dataset'],cached_s=round(result['complete_cached_replay_seconds'],4),
                   post_O24_s=round(result['post_O24_compute_seconds'],4),xor=result['archive_CLI_mask_xor']),flush=True)
    metrics=('complete_cached_replay_seconds','RGB_and_reference_mask_load_seconds','preprocess_seconds','segment_seconds',
             'verified_O24_replay_seconds','post_O24_compute_seconds')
    summary={key:dict(mean=float(np.mean([r[key] for r in results])),median=float(np.median([r[key] for r in results])),
                      min=float(min(r[key] for r in results)),max=float(max(r[key] for r in results)),
                      p95=float(np.quantile([r[key] for r in results],.95))) for key in metrics}
    stage_names=sorted({p for r in results for p in r['spans']})
    stage_summary={p:{name:float(np.mean([r['spans'].get(p,{}).get(name,0) for r in results]))
                      for name in ('calls','inclusive_seconds','exclusive_seconds')} for p in stage_names}
    result=dict(state='COST_MEASUREMENT_COMPLETE_NO_SCORING',model=model,n=10,config_sha256=sha(OUT/'config.json'),
                setup_seconds=setup_seconds,warmup_seconds=warmup,measurement_wall_seconds=time.perf_counter()-started,
                threadpools=threadpool_info(),torch_version=str(torch.__version__),summary=summary,stage_summary=stage_summary,
                records=results,new_encoder_calls=0,query_GT_reads=0,masks_saved=0,parameter_fitting=False,
                source_mask_parity_changed_tasks=sum(r['archive_CLI_mask_xor']!=0 for r in results))
    write(OUT/(model+'.json'),result)


def report():
    import numpy as np
    f=read(OUT/'foris.json');i=read(OUT/'insid3.json')
    fm=f['summary']['post_O24_compute_seconds']['mean'];im=i['summary']['post_O24_compute_seconds']['mean']
    historical_paths=[ASSETS/'a/deepglobe_component100_20261009/performance/live_cost_compare/result.json',
                      ASSETS/'a/deepglobe_fast3_vertical100_20261009/performance/full_latency.json',
                      ASSETS/'a/paco_mean200_20261008/performance_snapshot.json']
    live=read(historical_paths[0]);fast=read(historical_paths[1]);prepared=next(r for r in live['results'] if r['arm']=='foris.prepared')
    encoding=dict(state='HISTORICAL_CONTEXT_ONLY_NOT_ADDED_TO_CURRENT_CPU_MEASUREMENTS',
                  source_sha256={str(p):sha(p) for p in historical_paths},
                  single_case_prepared_foris_RGB_total_seconds=prepared['complete_pipeline_seconds'],
                  single_case_two_input_MPS_encoding_seconds=prepared['common_RQ_encoder_seconds'],
                  five_case_mean_two_input_MPS_encoding_seconds=float(np.mean([r['common_RQ_encoder_seconds'] for r in fast['results']])),
                  conditions='Historical MPS FP32 batch1; CPU2 postprocessing; one whole-FoRIS case and five window-pipeline cases, source records from 2026-10-09',
                  limits='Different date/tasks/device split and no current encoder rerun; timing implementation synchronization not reverified from surviving source. Do not synthesize a current RGB end-to-end latency.',
                  rejected_as_pure_encoder='PACO performance_snapshot explicitly includes raw-cache compression, hashes and readback in its paired_encoding_seconds')
    write(OUT/'encoding_evidence.json',encoding)
    result=dict(state='LATENCY_REFERENCE_COMPLETE',n=10,CPU_threads=2,workers=1,device='cpu',
                FoRIS_mean_post_O24_seconds=fm,INSID3_mean_post_O24_seconds=im,INSID3_to_FoRIS_post_O24_ratio=im/fm,
                illustrative_2p5x_post_O24_budget_seconds=2.5*fm,additional_post_O24_budget_seconds=1.5*fm,
                budget_rule='Use one complete composed execution with shared prefixes. Do not add inclusive parent spans to their children or rerun a shared prefix per operator.',
                encoding_argument='If the same encoder/preprocess cost E is shared, C<=2.5*F implies (E+C)/(E+F)<=2.5 for E>=0. This is a conservative sufficient ratio bound, not a measured RGB latency.',
                sharing_constraints=['Raw O24 can be shared once per legal input.',
                    'FoRIS positional suppression is conditional, INSID3 debias is unconditional; equal names do not prove equal input tensors.',
                    'FoRIS query clustering augments features with RGB/XY; INSID3 uses raw query features. Those cluster results are not interchangeable.',
                    'A final chosen mask needs one final CRF; multiple intermediate masks should not each pay a fresh whole CRF unless the tested composition explicitly requires it.',
                    'The prepared spatial CRF geometry is shared across images; image-dependent bilateral geometry is charged for each distinct query.'],
                new_encoder_calls=0,query_GT_reads=0,parameter_fitting=False,masks_saved=0,
                foris_sha256=sha(OUT/'foris.json'),insid3_sha256=sha(OUT/'insid3.json'),encoding_evidence_sha256=sha(OUT/'encoding_evidence.json'))
    write(OUT/'report.json',result)
    lines=['# Same-machine operator cost reference','',
           'Ten fixed tasks, one per domain. Serial CPU2, same existing FP32 O24, official complete outputs. No query-label reads, fitting, encoder calls, or mask files.',
           '','|Measured interval|FoRIS mean s|INSID3 CRF640 mean s|','|---|---:|---:|']
    for key in ('complete_cached_replay_seconds','verified_O24_replay_seconds','post_O24_compute_seconds'):
        lines.append(f"|{key}|{f['summary'][key]['mean']:.6f}|{i['summary'][key]['mean']:.6f}|")
    lines+=['',f'At a 2.5x ratio, the measured mean FoRIS post-O24 cost gives a provisional composed-head budget of {2.5*fm:.3f}s, or {1.5*fm:.3f}s additional work.',
            'This is cache-after computation, not RGB end-to-end latency. The actual final composition must be timed as a whole; ten single samples are descriptive, not a latency SLA.',
            '','## FoRIS top-level stages','', '|Stage|Mean inclusive s|','|---|---:|']
    for name,value in f['stage_summary'].items():
        if name.count('/')==1:lines.append(f"|{name}|{value['inclusive_seconds']:.6f}|")
    lines+=['','## Reuse without double counting','']+[f'- {s}' for s in result['sharing_constraints']]
    lines+=['','Do not sum parent and child spans: part3 already includes correspondence, clustering and seed-prior work. Exclusive timings are supplied for disjoint accounting.',
            '', '## Existing encoder evidence','',
            f"Historical MPS records report {prepared['common_RQ_encoder_seconds']:.3f}s for two sequential encoded inputs and {prepared['complete_pipeline_seconds']:.3f}s for that prepared FoRIS RGB pipeline; five other tasks average {encoding['five_case_mean_two_input_MPS_encoding_seconds']:.3f}s for common R/Q encoding.",
            encoding['limits'],encoding['rejected_as_pure_encoder']]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','foris','insid3','report','run'))
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode in ('foris','insid3'):measure(args.mode)
    elif args.mode=='report':report()
    else:
        prepare()
        script=OUT/'frozen/x1012s2_latency.py'
        for model in ('foris','insid3'):
            subprocess.run([sys.executable,str(script),model],check=True)
        report()
