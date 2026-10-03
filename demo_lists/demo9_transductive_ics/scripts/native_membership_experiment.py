#!/usr/bin/env python3
"""Finite reference-only full-reader comparison and native decision tracing.

Complete source controls, not a new claimed method. CPU preparation uses only
existing assets. Query annotations open only after all six predictions freeze.
"""
import argparse
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import zlib
from native_rice_core_experiment import digest, write_json

HERE=Path(__file__).resolve().parent
ARMS=('foris_crf','native_identity','part2_native_direct','kernel_svm','complete_frost','foris_stateful','foris_channel_nn')


@contextmanager
def same_input_raw_cache(host, expected, raw):
    """Source encoder bypass only for a bitwise-identical native B2 input."""
    import torch
    existed='_extract_features' in host.__dict__;previous=host.__dict__.get('_extract_features')
    calls=[]
    def cached(imgs):
        if imgs.shape!=expected.shape or imgs.dtype!=expected.dtype or imgs.stride()!=expected.stride() or not torch.equal(imgs,expected):
            raise RuntimeError('Native public-context raw cache input is not exact')
        calls.append(True);return raw
    try:
        host._extract_features=cached;yield calls
    finally:
        if existed:host._extract_features=previous
        else:delattr(host,'_extract_features')


def prepare(a):
    parent=json.loads(Path(a.from_manifest).read_text())
    if parent.get('schema')!='native_angular_assets_v1' or parent.get('state')!='PREPARED_ASSETS':
        raise ValueError('Previously validated actual native-angular assets required')
    if Path(a.prepared_manifest).exists():raise ValueError('Fresh membership manifest required')
    for asset in parent['assets']:
        st=Path(asset['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(asset['size'],asset['mtime_ns']):raise ValueError('Asset drift')
    base=HERE.parent
    source=[Path(__file__),HERE/'analyze_native_membership.py',HERE/'prepare_native_membership_queue.py',
        HERE/'native_membership_source_cpu.py',HERE/'reference_kernel_svm_cpu.py',
        HERE/'native_candidate_axis_cpu.py',HERE/'reference_witness_audit_cpu.py',
        HERE/'native_rice_core_experiment.py',HERE/'native_angular_experiment.py',
        HERE/'analyze_native_angular.py',HERE/'analyze_rice_core.py',HERE/'experiment_resource_guard.py',
        HERE/'stream_owned_results.py',
        base/'tics/native_decision_trace.py',base/'tics/reference_kernel_svm.py',
        base/'tics/native_candidate_axis.py',base/'tics/reference_witness_audit.py',
        base/'tics/frost_existing_adapter.py',base/'tics/native_assets.py',
        base/'tics/__init__.py',base/'tics/imageset.py',base/'tics/propagate.py']
    inherited=[Path(p) for p in parent['source_hashes'] if '/foris_source/' in p or p.endswith('/icx/common.py') or p.endswith('/scripts/_paths.py')]
    for p in inherited:
        if digest(p)!=parent['source_hashes'][str(p)]:raise ValueError('Existing source drift')
    source+=inherited+[Path(a.frost_root)/'frost'/n for n in ['model.py','density.py','data.py','encoder.py']]
    package_init=Path(a.foris_root)/'models/__init__.py'
    if package_init.is_file() and package_init not in source:source.append(package_init)
    for p in source:compile(p.read_text(),str(p),'exec')
    manifest={**parent,'schema':'native_membership_assets_v1','arms':list(ARMS),
        'parent_manifest_sha256':digest(a.from_manifest),'source_hashes':{str(p.resolve()):digest(p) for p in source},
        'frost_root':str(Path(a.frost_root).resolve()),'fixed_config':dict(
            scope='Same40 official seed0 tasks; reused development, not independent confirmation',
            backbone='same frozen FP32 local timm DINOv3, no AMP/TF32',
            historical_foris_context='direct tensor predict; source _tgt_image None',
            strong_foris_context='source set_reference/set_target/segment with RGB context',
            native_trace='public stateful context, unchanged source stages and one CRF',
            candidate_axis_control='complete public FoRIS, ONLY ref_m normalization dim2(H) to dim1(D); source dim2 replica exact required; not a novel method',
            full_frost='complete unchanged source head/injected timm adaptation, native B3/SVD250/all anchors/bilateral/continuous0, no CRF',
            kernel_svm='native Part1 full coordinates, pureFG>=.9/BG<=.1 deterministic caps128/256, C1, balanced weights, reference-pair median bandwidth',
            kernel_output='continuous signed margin upsample/zero threshold/native single CRF; no minmax/area/gate',
            naive_control='original Part2 score with source minmax/bilinear/singleCRF, no Part3/4',
            GT_use='support legal, query annotations after ALL outputs immutable',
            data_storage='small trace score maps/masks only, no feature bank',
            no_new_method_or_gain_claim=True)}
    write_json(a.prepared_manifest,manifest)
    print(json.dumps(dict(state='PREPARED_ASSETS',tasks=len(parent['frozen_episodes']),source_files=len(source),CUDA_imported=False)))


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from native_rice_core_experiment import capture_native, frozen_native_replay, exact_clustering_cache
    if not a.allow_gpu or os.environ.get('DEMO9_CUDA_GUARD')!='1':raise RuntimeError('Guarded GPU stage required')
    guard=json.loads(Path(a.resource_guard_state).read_text())
    if guard['state']!='GPU_RUNNING' or guard['last_event'].get('stage')!='native_membership40':raise RuntimeError('Wrong owned stage')
    manifest=json.loads(Path(a.prepared_manifest).read_text())
    if manifest.get('schema')!='native_membership_assets_v1' or manifest.get('arms')!=list(ARMS):raise ValueError('Frozen manifest mismatch')
    for p,h in manifest['source_hashes'].items():
        if digest(p)!=h:raise ValueError('Source drift: '+p)
    for asset in manifest['assets']:
        st=Path(asset['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(asset['size'],asset['mtime_ns']):raise ValueError('Existing asset drift')
    # Missing solver is a preparation defect, never silent native fallback.
    from sklearn.svm import SVC
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output directory required')
    out.mkdir(parents=True,exist_ok=True)
    report=dict(state='RUNNING',records=[],arms=list(ARMS),manifest_sha256=digest(a.prepared_manifest),
                source_hashes=manifest['source_hashes'],fixed_config=manifest['fixed_config'],
                query_GT_used_for_prediction=False,query_GT_used_in_prediction=False,feature_cache_written=False)
    start=time.monotonic()
    def save():report['elapsed_s']=time.monotonic()-start;write_json(out/'report.json',report)
    def iu(x,y):return [int((x&y).sum()),int((x|y).sum())]
    def packed(x):return dict(shape=list(x.shape),codec='zlib_np_packbits_big',data=base64.b64encode(zlib.compress(np.packbits(x.reshape(-1)).tobytes(),6)).decode())
    save()
    try:
        if not torch.cuda.is_available():raise RuntimeError('No CUDA; no model allocated')
        torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
        sys.path.insert(0,a.foris_root);import models.foris as foris_module
        from utils.data import build_transform
        sys.path.insert(0,_paths.DEMO4);from icx.common import TimmDINOv3
        sys.path.insert(0,str(HERE.parent))
        from tics.native_assets import reuse_native_basis
        from tics.native_decision_trace import trace_native_decisions
        from tics.native_candidate_axis import candidate_normalization_axis
        from tics.reference_kernel_svm import fit_reference_kernel_svm
        from tics.frost_existing_adapter import construct_complete_frost,capture_frost_finalization,verify_frost_public_tensors
        with torch.inference_mode():
            encoder=TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(foris_module.FoRIS,a.projection_basis):
                host=foris_module.FoRIS(encoder=encoder,image_size=1024,svd_components=500,tau=.6,mask_refiner='crf',resize_to_orig_size=False,device='cuda').eval().requires_grad_(False)
            init=time.monotonic();frost,frost_contract=construct_complete_frost(manifest['frost_root'],encoder,device='cuda')
            report['full_frost_contract']={**frost_contract,'constructor_seconds':time.monotonic()-init}
            transform=build_transform(1024)
            for row in manifest['frozen_episodes']:
                begin=time.monotonic();c=row['c']
                sp=Image.open(Path(manifest['data_root'])/row['support']).convert('RGB')
                qp=Image.open(Path(manifest['data_root'])/row['query']).convert('RGB')
                s=transform(sp).cuda()[None];q=transform(qp).cuda()[None]
                gold_cpu=torch.from_numpy((np.asarray(Image.open(Path(manifest['annotation_root'])/Path(row['support']).with_suffix('.png')))==c+1).copy())
                gold=gold_cpu.cuda();mask=F.interpolate(gold[None,None].float(),(1024,1024),mode='nearest')[0].bool()
                with exact_clustering_cache(foris_module) as cache:
                    with capture_native(host) as packet:native=host.predict(s,mask,q[0]).reshape(1024,1024).bool().clone()
                    raw,part1,part2=(packet[k] for k in ['_extract_features','_part1_positional_debias','_part2_background_suppression'])
                    with frozen_native_replay(host,raw,part1,part2):identity=host.predict(s,mask,q[0]).reshape(1024,1024).bool().clone()
                    if not torch.equal(native,identity):raise RuntimeError('Historical native identity drift')
                    expected=torch.cat([s,q],dim=0)[None]
                    def set_public():
                        host.set_reference(sp,gold_cpu);host.set_target(qp)
                        if not torch.equal(host._ref_images,s) or not torch.equal(host._ref_masks,mask) or not torch.equal(host._tgt_image,q[0]):raise RuntimeError('Public source preprocessing differs from frozen tensors')
                    set_public()
                    with same_input_raw_cache(host,expected,raw) as raw_calls:stateful=host.segment().reshape(1024,1024).bool().clone()
                    set_public()
                    with same_input_raw_cache(host,expected,raw),trace_native_decisions(host) as trace:stateful_replay=host.segment().reshape(1024,1024).bool().clone()
                    if not torch.equal(stateful,stateful_replay):raise RuntimeError('Public-context observer changes source mask')
                    set_public()
                    with same_input_raw_cache(host,expected,raw),candidate_normalization_axis(host,2):
                        axis_replica=host.segment().reshape(1024,1024).bool().clone()
                    if not torch.equal(stateful,axis_replica):raise RuntimeError('Whole-source candidate-axis2 replica changes public mask')
                    set_public()
                    with same_input_raw_cache(host,expected,raw),candidate_normalization_axis(host,1):
                        channel_prediction=host.segment().reshape(1024,1024).bool().clone()
                    # Naive direct original-score reader, source scalar/refiner.
                    direct_mask=host._binarize_response(part2[0],target_hw=(1024,1024))
                    direct=host._finalize_mask(direct_mask,q[0]).reshape(1024,1024).bool().clone()
                    grid=tuple(part1.shape[-2:]);coverage=F.interpolate(gold[None,None].float(),grid,mode='area')[0,0].reshape(-1)
                    tokens=part1[0,0].flatten(1).T.cpu().numpy();qtokens=part1[0,1].flatten(1).T.cpu().numpy()
                    def indices(label,cap):
                        value=torch.nonzero(label).flatten().cpu().numpy()
                        return value if len(value)<=cap else value[np.arange(cap)*len(value)//cap]
                    fg=indices(coverage>=.9,128);bg=indices(coverage<=.1,256)
                    learner,svm_audit=fit_reference_kernel_svm(tokens[fg],tokens[bg],C=1.)
                    svm_margin=None
                    if learner is None:
                        if svm_audit['state'] in ['SKLEARN_UNAVAILABLE','SVM_NOT_CONVERGED']:raise RuntimeError('SVM solver/preflight failure: '+svm_audit['state'])
                        svm_prediction=native.clone();svm_audit['exact_native_fallback']=True
                    else:
                        svm_margin=learner.decision_function(qtokens).reshape(grid)
                        margin=torch.from_numpy(svm_margin).cuda().float()
                        binary=F.interpolate(margin[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>0
                        svm_prediction=host._finalize_mask(binary,q[0]).reshape(1024,1024).bool().clone()
                    frost.set_reference(sp,gold_cpu);frost.set_target(qp)
                    verify_frost_public_tensors(frost,s,mask,q)
                    with capture_frost_finalization(frost) as frost_trace:
                        frost_prediction=frost.segment().reshape(1024,1024).bool().clone()
                    predictions=dict(foris_crf=native,native_identity=identity,foris_stateful=stateful,
                        part2_native_direct=direct,kernel_svm=svm_prediction,complete_frost=frost_prediction,foris_channel_nn=channel_prediction)
                    torch.cuda.synchronize();frozen={k:v.cpu().numpy().astype(bool) for k,v in predictions.items()}
                    inference_s=time.monotonic()-begin
                    # FIRST access to query labels occurs after every method.
                    truth_original=np.asarray(Image.open(Path(manifest['annotation_root'])/Path(row['query']).with_suffix('.png')))==c+1
                    truth=torch.from_numpy(truth_original.copy()).cuda()
                    truth_model=F.interpolate(truth[None,None].float(),(1024,1024),mode='nearest')[0,0].bool().cpu().numpy()
                    query_coverage=F.interpolate(truth[None,None].float(),grid,mode='area')[0,0].cpu().numpy()
                    original={k:(F.interpolate(v[None,None].float(),truth_original.shape,mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy() for k,v in predictions.items()}
                    native_np=frozen['foris_crf'];ledger={k:dict(recovered_fn=int((~native_np&v&truth_model).sum()),lost_tp=int((native_np&~v&truth_model).sum()),added_fp=int((~native_np&v&~truth_model).sum()),removed_fp=int((native_np&~v&~truth_model).sum())) for k,v in frozen.items()}
                    maps={k:v.numpy() for k,v in trace['maps'].items()}
                    maps.update(pre_refinement=maps.pop('pre_refinement_mask'),post_refinement=maps.pop('post_refinement_mask'),coverage=query_coverage,truth_model=truth_model)
                    if svm_margin is not None:maps['svm_grid_margin']=svm_margin
                    if frost_trace.get('continuous_density_available'):
                        maps['frost_grid_posterior']=frost_trace['continuous_posterior'].numpy()
                    if 'dilated_candidate' in frost_trace:maps['frost_candidate']=frost_trace['dilated_candidate'].numpy()
                    trace_name=f"trace_{row['fold']}_{row['e']}_{c}.npz"
                    np.savez_compressed(out/trace_name,**maps)
                    report['records'].append(dict(**row,iu={k:iu(v,truth_model) for k,v in frozen.items()},original_iu={k:iu(v,truth_original) for k,v in original.items()},ledger=ledger,
                        prediction_bits={k:packed(v) for k,v in frozen.items()},evaluator_bits=packed(truth_model),
                        replay_identity_exact=True,stateful_replay_exact=True,candidate_axis2_replica_exact=True,trace_source_arm='foris_stateful',
                        query_GT_opened_after_all_predictions=True,trace_archive=trace_name,trace_sha256=digest(out/trace_name),
                        native_binarization=trace['binarization'],svm_audit=svm_audit,
                        witness_audit=trace.get('witness_audit'),
                        frost_audit=dict(state='SOURCE_CONTINUOUS_DENSITY' if frost_trace.get('continuous_density_available') else 'SOURCE_FALLBACK_NO_CONTINUOUS_DENSITY',continuous_density_available=frost_trace.get('continuous_density_available',False),threshold=frost_trace.get('density_tau'),source_pipeline=True,timm_adaptation=True),
                        cost=dict(inference_s=inference_s,clustering_cache=cache.copy(),stateful_raw_cache_input_exact=True,stateful_baseline_raw_cache_calls=len(raw_calls),native_batch='S,Q',frost_batch='S,flip(S),Q')))
                    save();print(json.dumps(dict(completed=len(report['records']),fold=row['fold'],c=c,inference_s=inference_s,native_IU=report['records'][-1]['original_iu']['foris_crf'],public_IU=report['records'][-1]['original_iu']['foris_stateful'])),flush=True)
                del packet,raw,part1,part2,predictions,frozen,trace,maps
            report['state']='COMPLETED';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
    except BaseException as error:
        report.update(state='ERROR',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--from-manifest');p.add_argument('--prepared-manifest');p.add_argument('--projection-basis')
    p.add_argument('--frost-root',default='/root/autodl-tmp/demo9_transductive_ics/runtime/frost_source')
    p.add_argument('--foris-root',default='/root/autodl-tmp/demo8_local_verification/foris_source')
    p.add_argument('--prepared-root',default='/root/autodl-tmp/demo9');p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--resource-guard-state');p.add_argument('--out');a=p.parse_args()
    if a.prepare_only:prepare(a)
    else:run(a)


if __name__=='__main__':main()
