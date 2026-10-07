#!/usr/bin/env python3
"""Single fixed locked-RCG+RGB-disagreement composition; sealed CPU inference.

Inference reuses bound q/r/coverage/source-score/RGB128 only, recomputes locked
RCG and the current MEAN control, then seals all masks before separate GT score.
This composition adds neither new information nor an independent mechanism.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
PRIMARY = 'rcg_rgb_proposal_v2'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def initialize(threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if sys.platform == 'darwin' else 1024))


def episode(row, out, memory_limit, numerical_audit=False):
    import numpy as np
    from ics.experiment import load_inputs, render, sha
    from ics.methods.rcg_rgb_proposal import predict
    started, cpu = time.perf_counter(), time.process_time()
    (q,r,cov,score), source = load_inputs('/root', row)
    # Directly open the already bound RGB128 packet; no image/annotation fallback.
    rgb_path = Path(row['query_rgb_export'])
    rgb_hash = sha(rgb_path)
    with np.load(rgb_path,allow_pickle=False) as packet:
        rgb = packet['rgb'].copy()
    result = predict(q,r,cov,score,rgb,original_shape=row['query_image_hw'])
    audit={}
    if numerical_audit:
        from ics.methods.rcg import predict as direct_rcg
        direct, direct_info=direct_rcg(q,r,cov,score,device='cpu')
        error=float(np.max(np.abs(direct-result['rcg_field'])))
        xor=int(np.count_nonzero(render(direct)!=result['rcg_mask_work']))
        if error!=0 or xor!=0:raise RuntimeError('Current RCG direct field/mask parity failed')
        audit=dict(current_rcg_direct_max_field_error=error,current_rcg_direct_mask_xor=xor,
                   field_shapes={key:list(result[key].shape) for key in ('rcg_field','mean_field','field','full_cut_field','same_unary_field')},
                   direct_rcg_solver=direct_info,query_gt_opened=False)
        if row.get('audit_history_rcg'):
            from ics.experiment import unpack
            spec=row['audit_history_rcg']
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Historical RCG prediction changed')
            with np.load(spec['path'],allow_pickle=False) as archive:historical=unpack(archive[spec['key']])
            audit.update(historical_rcg_mask_xor=int(np.count_nonzero(historical!=result['rcg_mask_work'])),
                         historical_rcg_prediction=spec,
                         historical_mask_parity_required=False)
    if sha(rgb_path) != rgb_hash:
        raise ValueError('Bound RGB packet changed during inference')
    fields = {'rcg_current.control':result['rcg_field'], 'mean.control':result['mean_field'],
              PRIMARY:result['field'].astype(np.uint8),
              'rcg_rgb_full_cut.control':result['full_cut_field'],
              'rcg_rgb_same_unary.control':result['same_unary_field']}
    masks = {'rcg_current.control':result['rcg_mask_work'], 'mean.control':result['mean_mask_work'],
             PRIMARY:result['mask_work'], 'rcg_rgb_full_cut.control':result['full_cut_mask_work'],
             'rcg_rgb_same_unary.control':result['same_unary_mask_work']}
    out = Path(out)
    pp = out/'predictions'/(row['occurrence_id']+'.npz')
    fp = out/'fields'/(row['occurrence_id']+'.npz')
    np.savez_compressed(pp,**{key:np.packbits(value) for key,value in masks.items()})
    np.savez_compressed(fp,**fields)
    # Original masks are an extra output, never substituted into the legacy
    # 1024 evaluation convention used by the generic paired scorer.
    op = out/'original_predictions'/(row['occurrence_id']+'.npz')
    np.savez_compressed(op,mask=result['mask_original'])
    peak = rss()
    if peak > memory_limit:
        raise RuntimeError('Worker peak RSS exceeds its share of the declared budget')
    receipt = dict(occurrence_id=row['occurrence_id'],source_key=row['key'],inputs=source,
                   numerical_audit=audit,rcg_recompute_seconds=result['info']['rcg_recompute_seconds'],
                   mean_recompute_seconds=result['info']['mean_recompute_seconds'],
                   rgb=dict(path=str(rgb_path),sha256=rgb_hash,keys_read=['rgb']),
                   simple_control=result['info'],prediction_sha256=sha(pp),field_sha256=sha(fp),
                   original_prediction_sha256=sha(op),wall_seconds=time.perf_counter()-started,
                   cpu_seconds=time.process_time()-cpu,process_peak_rss_bytes=peak,
                   query_gt_opened=False,new_encoder_forwards=0,independent_candidate_methods=0,
                   pid=os.getpid())
    dump(out/'receipts'/(row['occurrence_id']+'.json'),receipt)
    return receipt


def infer(args):
    from ics.experiment import sha
    from ics.methods.mean_rgb_potts import Config as RGBConfig
    from ics.methods.rcg_rgb_proposal import CONFIG
    from ics.methods.rcg import CONFIG as RCG_CONFIG
    from dataclasses import asdict
    if not 1 <= args.workers <= 6 or not 1 <= args.threads <= 2:
        raise ValueError('Composition budget is at most six workers with two threads each')
    if not 0 < args.memory_gb <= 12:
        raise ValueError('Memory budget must be positive and at most 12GiB')
    source = json.loads(args.manifest.read_text())
    source = source if isinstance(source,list) else source['episodes']
    source = source[:args.limit] if args.limit is not None else source
    if args.expected is not None and len(source)!=args.expected:raise ValueError('Manifest count differs from fixed expected count')
    if args.numerical_audit and len(source)>4:raise ValueError('Numerical smoke audit is limited to four bound cases')
    if args.workers*args.threads>12:raise ValueError('Total worker thread budget exceeds 12 CPU')
    if not source:
        raise ValueError('Empty manifest')
    inference,evaluation = [],[]
    for i,original in enumerate(source):
        row = dict(original)
        for key in ('c','fold','support','query','feature_export','packet_export','query_rgb_export','query_image_hw'):
            if key not in row:
                raise ValueError('Required bound manifest field missing: '+key)
        row['occurrence_id'] = f'{i:06d}'
        row.setdefault('key',f"{row['fold']}_{row.get('e',i)}_{row['c']}")
        for key in ('feature_export','packet_export','query_rgb_export'):
            path = Path(row[key])
            row[key] = str((path if path.is_absolute() else args.root/path).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if (len(row['query_image_hw']) != 2 or any(int(v) != v or v <= 0 for v in row['query_image_hw'])):
            raise ValueError('Original positive integer query H/W metadata required')
        for spec in row.get('evaluation_controls',{}).values():
            path=Path(spec['path']);spec['path']=str((path if path.is_absolute() else args.root/path).resolve())
            spec['sha256']=sha(spec['path'])
        if args.numerical_audit and 'rcg.control' in row.get('evaluation_controls',{}):
            row['audit_history_rcg']=dict(row['evaluation_controls']['rcg.control'])
        evaluation.append(row)
        inference.append({key:row[key] for key in ('occurrence_id','key','feature_export','packet_export',
                                                  'query_rgb_export','query_image_hw','audit_history_rcg') if key in row})
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('predictions','original_predictions','fields','receipts'):
        (args.out/name).mkdir()
    paths = (Path(__file__),REPO/'src/ics/methods/rcg_rgb_proposal.py',
             REPO/'src/ics/methods/mean_rgb_proposal.py',REPO/'src/ics/methods/mean_rgb_potts.py',
             REPO/'src/ics/methods/pro_paired_environment.py',
             REPO/'src/ics/methods/huber_graph.py',REPO/'src/ics/methods/prepared_cpu_bundle.py',
             REPO/'src/ics/methods/rcg.py',REPO/'src/ics/experiment.py')
    config = dict(root=str(args.root.resolve()),backend='fixed_rcg_rgb_exploratory_composition',
                  primary=PRIMARY,recipe=CONFIG,rcg_config=RCG_CONFIG,rgb_config=asdict(RGBConfig()),
                  numerical_audit=args.numerical_audit,
                  workers=args.workers,threads=args.threads,cpu_budget=args.workers*args.threads,
                  memory_budget_bytes=int(args.memory_gb*1024**3),exposure=args.exposure,
                  source_manifest_sha256=sha(args.manifest),
                  code_sha256={str(path.relative_to(REPO)):sha(path) for path in paths},
                  independent_methods=0,novelty_claim=False,new_encoder_forwards=0,
                  evaluation_convention='legacy paired 1024 masks; original masks also saved separately')
    dump(args.out/'config.json',config)
    dump(args.out/'inference_manifest.json',inference)
    dump(args.out/'evaluation_manifest.json',evaluation)
    began=time.perf_counter(); receipts={}; peaks={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(args.threads,)) as pool:
        futures={pool.submit(episode,row,str(args.out),int(args.memory_gb*1024**3/args.workers),args.numerical_audit):row for row in inference}
        for future in as_completed(futures):
            record=future.result();key=record['occurrence_id']
            receipts[key]=sha(args.out/'receipts'/(key+'.json'))
            peaks[record['pid']]=max(peaks.get(record['pid'],0),record['process_peak_rss_bytes'])
            print(json.dumps(dict(occurrence_id=key,completed=len(receipts),total=len(inference),
                                  wall_seconds=record['wall_seconds'],
                                  mean_seconds=record['mean_recompute_seconds'],
                                  rcg_seconds=record['rcg_recompute_seconds'],rgb_cut_seconds=record['simple_control']['rgb']['cut_seconds'])),flush=True)
    seal=dict(state='ALL_PREDICTIONS_SEALED',receipts=receipts,n=len(receipts),
              elapsed_seconds=time.perf_counter()-began,
              peak_owned_rss_bytes=sum(peaks.values())+rss(),
              owned_rss_semantics='conservative sum of each worker lifetime peak and parent peak',
              config_sha256=sha(args.out/'config.json'),
              inference_manifest_sha256=sha(args.out/'inference_manifest.json'),
              evaluation_manifest_sha256=sha(args.out/'evaluation_manifest.json'),
              query_gt_opened=False,new_encoder_forwards=0,independent_methods=0)
    if seal['peak_owned_rss_bytes']>config['memory_budget_bytes']:
        raise RuntimeError('Conservative sum of worker/parent peaks exceeds collective memory budget')
    dump(args.out/'sealed.json',seal)


def score(args):
    # Reuse the established separately invoked paired evaluator, without editing
    # that shared source or loading query truth during inference.
    from run_cpu_feature_candidates import score as existing_score
    existing_score(args)
    path=args.out/'score'/'report.json';report=json.loads(path.read_text())
    report.update(independent_methods=0,novelty_claim=False,
                  interpretation='One fixed RCG + existing RGB disagreement composition on reused public600; actual quality measured, previous gains not added; no new information or independent mechanism.')
    dump(path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('infer')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--memory-gb',type=float,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--expected',type=int)
    p.add_argument('--numerical-audit',action='store_true')
    p.add_argument('--exposure',default='reused public600 exploratory composition; no independent confirmation')
    p=sub.add_parser('score');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(args.threads if args.command=='infer' else 1)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.command=='infer':infer(args)
    else:score(args)


if __name__=='__main__':
    main()
