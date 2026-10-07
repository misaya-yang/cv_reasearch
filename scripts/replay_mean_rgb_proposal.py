#!/usr/bin/env python3
"""Derive a single fixed RGB readout revision from sealed 600 mask predictions.

Validation/derive and GT score are separate commands. Derive reads only three
named source prediction masks after validating the source seal, receipt,
prediction, field and packet hashes. No target labels or feature inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
PRIMARY='mean_rgb_proposal_v2'
SOURCE_KEYS=('mean.control','mean_rgb_potts','mean_rgb_unary.control')


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def derive_one(item,source,out):
    import numpy as np
    from ics.experiment import sha,unpack
    from ics.methods.mean_rgb_proposal import compose
    key=item['occurrence_id'];source=Path(source);out=Path(out)
    rp=source/'receipts'/(key+'.json');pp=source/'predictions'/(key+'.npz')
    receipt=json.loads(rp.read_text());started=time.perf_counter()
    # Recheck files at consumption time, after the full prevalidation pass.
    if sha(rp)!=item['source_receipt_sha256'] or sha(pp)!=receipt['prediction_sha256']:
        raise ValueError('Source receipt/prediction changed during replay')
    with np.load(pp,allow_pickle=False) as archive:
        values=[unpack(archive[name]) for name in SOURCE_KEYS]
    if sha(pp)!=receipt['prediction_sha256']:
        raise ValueError('Source prediction changed while reading its masks')
    mean,cut,unary=values
    final,region,info=compose(mean,cut,unary)
    masks={PRIMARY:final,'rgb_full_cut_v1.control':cut,'rgb_same128_unary.control':unary,'mean.control':mean}
    op=out/'predictions'/(key+'.npz');fp=out/'fields'/(key+'.npz')
    np.savez_compressed(op,**{name:np.packbits(value) for name,value in masks.items()})
    # All replay fields are explicitly binary1024, not reconstructed continuous
    # MEAN fields or recovered RGB128 fields. Prediction identity is authoritative.
    np.savez_compressed(fp,**{name:value.astype(np.uint8) for name,value in masks.items()})
    result=dict(occurrence_id=key,source_key=receipt['source_key'],inputs=receipt['inputs'],
                source_prediction_sha256=receipt['prediction_sha256'],
                source_receipt_sha256=item['source_receipt_sha256'],
                source_field_sha256=receipt['field_sha256'],
                prediction_sha256=sha(op),field_sha256=sha(fp),proposal=info,
                field_semantics='binary1024 replay masks; not continuous probabilities',
                query_gt_opened=False,new_encoder_forwards=0,independent_candidate_methods=0,
                derived_seconds=time.perf_counter()-started,
                inherited_source_episode_seconds=receipt['wall_seconds'],
                inherited_shared_MEAN_seconds=receipt['shared_pregraph_and_base_seconds'],
                inherited_RGB_component_seconds=receipt['methods']['mean_rgb_potts']['candidate_and_controls_seconds'],
                inherited_other_selected_component_cost_is_not_new_method_cost=True)
    dump(out/'receipts'/(key+'.json'),result)
    return result


def derive(args):
    from ics.experiment import sha
    from ics.methods.mean_rgb_proposal import CONFIG
    source=args.source.resolve()
    source_seal=json.loads((source/'sealed.json').read_text())
    if source_seal['state']!='ALL_PREDICTIONS_SEALED' or source_seal['n']!=600:
        raise ValueError('Require the already sealed exact public600 source')
    for name in ('config','inference_manifest','evaluation_manifest'):
        if sha(source/(name+'.json'))!=source_seal[name+'_sha256']:
            raise ValueError('Source sealed metadata changed: '+name)
    source_config=json.loads((source/'config.json').read_text())
    rows=json.loads((source/'evaluation_manifest.json').read_text())
    if len(rows)!=600 or len({row['occurrence_id'] for row in rows})!=600:
        raise ValueError('Source public600 occurrence mapping differs')
    bound=[];validation_started=time.perf_counter()
    for row in rows:
        key=row['occurrence_id'];rp=source/'receipts'/(key+'.json')
        digest=sha(rp)
        if digest!=source_seal['receipts'][key]:
            raise ValueError('Source receipt hash mismatch: '+key)
        receipt=json.loads(rp.read_text())
        if (sha(source/'predictions'/(key+'.npz'))!=receipt['prediction_sha256']
                or sha(source/'fields'/(key+'.npz'))!=receipt['field_sha256']
                or sha(row['packet_export'])!=receipt['inputs']['packet_sha256']):
            raise ValueError('Source prediction/field/evaluation packet hash mismatch: '+key)
        # Query labels are never decompressed; packet hashing is byte-only.
        if receipt.get('query_gt_opened') is not False:
            raise ValueError('Source inference truth-isolation binding failed')
        bound.append(dict(occurrence_id=key,source_prediction_sha256=receipt['prediction_sha256'],
                          source_receipt_sha256=digest))
    validation_seconds=time.perf_counter()-validation_started
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('predictions','fields','receipts'):(args.out/name).mkdir()
    exposure='posthoc exploratory readout revision on reused public600; source quality was already exposed; not independent confirmation'
    config=dict(root='/root',backend='sealed_prediction_readout_revision',primary=PRIMARY,
                recipe=CONFIG,independent_methods=0,novelty_claim=False,new_encoder_forwards=0,
                source_run=str(source),source_seal_sha256=sha(source/'sealed.json'),
                source_config_sha256=sha(source/'config.json'),source_elapsed_seconds=source_seal['elapsed_seconds'],
                inherited_source_peak_owned_rss_bytes=source_seal['peak_owned_rss_bytes'],
                source_code_sha256=source_config['code_sha256'],exposure=exposure,
                threads=2,memory_budget_bytes=4*1024**3,
                code_sha256={str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'src/ics/methods/mean_rgb_proposal.py')},
                inference_contract='exact same source three masks; no RGB/DINO/MEAN recomputation',
                proposal_optimality_claim=False,
                inherited_cost_semantics='source full pipeline cost retained; replay is not a substitute for RGB+MEAN inference cost')
    dump(args.out/'config.json',config)
    dump(args.out/'evaluation_manifest.json',rows)
    dump(args.out/'inference_manifest.json',bound)
    dump(args.out/'source_validation.json',dict(state='ALL_600_SOURCE_RECEIPTS_PREDICTIONS_FIELDS_PACKETS_VERIFIED',
                                              source_seal_sha256=config['source_seal_sha256'],
                                              n=600,byte_hash_only_packets=True,query_gt_opened=False,
                                              validation_seconds=validation_seconds))
    started=time.perf_counter()
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index,result in enumerate(pool.map(lambda row:derive_one(row,source,args.out),bound),1):
            results.append(result)
            if index%100==0:
                print(json.dumps(dict(state='DERIVING_LABEL_FREE_MASKS',completed=index,total=600,
                                      seconds=time.perf_counter()-started)),flush=True)
    derived_seconds=time.perf_counter()-started
    if peak_rss()>4*1024**3:
        raise RuntimeError('Derived inference peak RSS exceeded 4GiB')
    seal=dict(state='ALL_PREDICTIONS_SEALED',n=600,
              receipts={x['occurrence_id']:sha(args.out/'receipts'/(x['occurrence_id']+'.json')) for x in results},
              config_sha256=sha(args.out/'config.json'),
              inference_manifest_sha256=sha(args.out/'inference_manifest.json'),
              evaluation_manifest_sha256=sha(args.out/'evaluation_manifest.json'),
              elapsed_seconds=derived_seconds,derived_seconds=derived_seconds,
              source_validation_seconds=validation_seconds,
              inherited_full_pipeline_seconds=source_seal['elapsed_seconds'],
              inherited_MEAN_RGB_serial_episode_seconds=sum(x['inherited_shared_MEAN_seconds']+x['inherited_RGB_component_seconds'] for x in results),
              inherited_all_source_serial_episode_seconds=sum(x['inherited_source_episode_seconds'] for x in results),
              peak_owned_rss_bytes=peak_rss(),query_gt_opened=False,encoder_forwards=0,
              independent_methods=0,exploratory_posthoc_replay=True)
    dump(args.out/'sealed.json',seal)
    print(json.dumps(dict(state=seal['state'],n=600,derived_seconds=derived_seconds,validation_seconds=validation_seconds,
                          peak_rss_bytes=peak_rss(),changed_total=sum(x['proposal']['changed_vs_mean_pixels'] for x in results))),flush=True)


def score(args):
    from ics.experiment import sha
    from run_cpu_feature_candidates import score as existing_score
    config=json.loads((args.out/'config.json').read_text())
    if sha(Path(config['source_run'])/'sealed.json')!=config['source_seal_sha256']:
        raise ValueError('Source seal changed before evaluation')
    existing_score(args)
    path=args.out/'score'/'report.json';report=json.loads(path.read_text())
    seal=json.loads((args.out/'sealed.json').read_text())
    report.update(exposure=config['exposure'],exploratory_posthoc_replay=True,independent_methods=0,
                  independent_confirmation=False,novelty_claim=False,
                  inherited_full_pipeline_seconds=seal['inherited_full_pipeline_seconds'],
                  inherited_MEAN_RGB_serial_episode_seconds=seal['inherited_MEAN_RGB_serial_episode_seconds'],
                  derived_seconds=seal['derived_seconds'],source_validation_seconds=seal['source_validation_seconds'],
                  interpretation='Single specified posthoc readout revision. Complete mIoU measured directly; historical improvements are not added. Final local substitution is not a Potts optimum.')
    dump(path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('derive');p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('score');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='derive':derive(args)
    else:score(args)


if __name__=='__main__':main()
