#!/usr/bin/env python3
"""Native-scoring full-rank angular covariance comparison; finite, no cache.

The original score, hard-negative construction and downstream native geometry
stay intact. The same forty development tasks isolate the previous projected
Gaussian/score-replacement ambiguity; this is not independent validation.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
ARMS = ('foris_crf', 'native_identity', 'paired_geometry', 'pooled_geometry',
        'fg_aug_geometry', 'shuffled_geometry', 'frost95_geometry')


def query_score_audit(score, coverage):
    """Evaluation only: pure query patches, exact tie-aware ranking, no tuning.

    FPR@95TPR uses query labels as a diagnostic ROC operating point. It is
    never an inference threshold, a calibration rule or a saved new model.
    """
    import numpy as np
    s = np.asarray(score, dtype=np.float64)
    c = np.asarray(coverage, dtype=np.float64)
    if s.shape != c.shape or not np.isfinite(s).all() or not np.isfinite(c).all():
        raise ValueError('Finite matching native score and query coverage grid required')
    p, n = s[c >= .9], s[c <= .1]
    result = dict(positive_patches=len(p), negative_patches=len(n),
                  mixed_patches=int(((c > .1) & (c < .9)).sum()),
                  scope='Query GT evaluation after every prediction freezes; ROC is not a deployable threshold')
    if not len(p) or not len(n):
        return dict(**result, state='NO_PURE_QUERY_CLASSES')
    ordered = np.sort(n)
    less = np.searchsorted(ordered, p, side='left')
    equal = np.searchsorted(ordered, p, side='right') - less
    threshold = np.sort(p)[int(np.floor(.05 * len(p)))]
    return dict(**result, state='SCORED', auc=float((less + .5*equal).mean()/len(n)),
                fpr_at_at_least_95tpr=float((n >= threshold).mean()),
                actual_tpr=float((p >= threshold).mean()),
                mean_fg_minus_bg=float(p.mean()-n.mean()),
                fg_quantiles=np.quantile(p,[.05,.5,.95]).tolist(),
                bg_quantiles=np.quantile(n,[.05,.5,.95]).tolist())


def reference_orbit_audit(views, indices, geometry):
    """Reference-only angular scatter, not a guarantee of semantic invariance."""
    import torch.nn.functional as F
    x = F.normalize(views[:,indices].double() @ geometry.double(), dim=-1)
    if not x.shape[1]: return dict(state='NO_PURE_REFERENCE_FOREGROUND')
    return dict(state='SCORED', foreground_positions=x.shape[1],
                mean_squared_chord_to_view_mean=float((x-x.mean(0,keepdim=True)).square().sum(-1).mean()),
                scope='Same-position reference foreground across declared background views; not cross-instance evidence')


def prepare(args):
    from native_rice_core_experiment import digest, write_json
    old = json.loads(Path(args.from_manifest).read_text())
    if old.get('schema') != 'rice_core_assets_v1' or old.get('state') != 'PREPARED_ASSETS':
        raise ValueError('Previously validated exact core40 assets required')
    if len(old['frozen_episodes']) != 40:
        raise ValueError('Frozen forty development tasks required')
    if Path(args.prepared_manifest).exists():
        raise ValueError('Fresh angular manifest required')
    for asset in old['assets']:
        actual = Path(asset['path']).stat()
        if (actual.st_size, actual.st_mtime_ns) != (asset['size'], asset['mtime_ns']):
            raise RuntimeError('Previously verified asset changed')
    # No torch, image decode, model, CUDA or repeated weight download/checksum.
    base = HERE.parent
    code = [Path(__file__), HERE/'analyze_native_angular.py', HERE/'prepare_native_angular_queue.py',
            HERE/'native_geometry_cpu.py', HERE/'native_rice_core_experiment.py',
            HERE/'analyze_rice_core.py', HERE/'experiment_resource_guard.py',
            HERE/'global_representation_probe.py', base/'tics/native_geometry_adapter.py',
            base/'tics/reference_nuisance_geometry.py', base/'tics/rice_statistics.py',
            base/'tics/reference_views.py', base/'tics/readout_baselines.py', base/'tics/native_assets.py',
            base/'tics/rice_subspace.py', base/'tics/__init__.py', base/'tics/imageset.py', base/'tics/propagate.py']
    inherited = [Path(p) for p in old['source_hashes'] if '/foris_source/' in p or p.endswith('/icx/common.py') or p.endswith('/scripts/_paths.py')]
    for source in inherited:
        if digest(source) != old['source_hashes'][str(source)]:
            raise RuntimeError('Original native source drift')
    code += inherited
    for source in code: compile(source.read_text(), str(source), 'exec')
    manifest = {**old, 'schema':'native_angular_assets_v1', 'state':'PREPARED_ASSETS', 'arms':list(ARMS),
                'parent_manifest_sha256':digest(args.from_manifest),
                'source_hashes':{str(p.resolve()):digest(p) for p in code},
                'scope':'Same forty development tasks; geometry/scoring attribution, not a fresh test',
                'fixed_config':dict(views=['original','global_mean_BG','half_cycle_BG_permutation'],
                    query_encoder_calls=1, support_encoder_calls=3, full_descriptor_rank=1024,
                    covariance_sampling='same original FG128/BG256 pure patch indices and fixed modes as core40',
                    ridge='0.1*trace(pooled_augmented_within)/D',
                    frost_control='same view anchors, unchanged frost_whitening .95 shrink/eigfloor1e-6; component not fullFROST',
                    metric='Mahalanobis cosine in full original feature coordinates',
                    scoring='Original FoRIS Part2, no posterior/KDE/threshold/gain replacement',
                    downstream='Original mu_fg and denoised target objects; original Stage3/4 and single CRF',
                    baseline_identity='zero covariance exact same feature/Part2 tuple bypass',
                    APD='Native original pair choice kept fixed across reference views')}
    manifest['fixed_config']['evidence_chain'] = 'Shared encodings: reference angular scatter -> native query score ROC on pure patches -> final mask error ledger; diagnostic only, no GT-based selection or extra stages'
    write_json(args.prepared_manifest, manifest)
    print(json.dumps(dict(state=manifest['state'], tasks=40, code_files=len(code),
                         assets_reused=len(old['assets']), CUDA_or_model_imported=False)))


def self_check(args):
    import importlib.util
    import torch
    import torch.nn.functional as F
    from unittest.mock import patch
    from native_rice_core_experiment import write_json
    torch.set_num_threads(1)
    import numpy as np
    assert query_score_audit(np.array([0,1,2,3]),np.array([0,0,1,1]))['auc'] == 1.
    assert query_score_audit(np.array([3,2,1,0]),np.array([0,0,1,1]))['auc'] == 0.
    assert query_score_audit(np.ones(4),np.array([0,0,1,1]))['auc'] == .5
    assert query_score_audit(np.ones(4),np.ones(4))['state'] == 'NO_PURE_QUERY_CLASSES'
    with patch('torch.cuda.is_available', return_value=False):
        path = HERE.parent/'tics/reference_nuisance_geometry.py'
        spec = importlib.util.spec_from_file_location('angular_runtime_cpu_geometry', path)
        geometry = importlib.util.module_from_spec(spec); spec.loader.exec_module(geometry)
        rows = []
        for index in range(10):
            torch.manual_seed(2057+index)
            a = torch.randn(12,4,dtype=torch.float64); c = a@a.T
            epsilon = .1*float(torch.trace(c))/12
            l, audit = geometry.fit_angular_geometry(c,epsilon)
            x = F.normalize(torch.randn(8,12,dtype=torch.float64),dim=1)
            y = geometry.transform_tokens(x,l)
            m = l.T@l
            reference = (x@m@x.T)/torch.sqrt((x@m*x).sum(1)[:,None]*(x@m*x).sum(1)[None])
            assert torch.allclose(y@y.T,reference,atol=1e-12,rtol=1e-12)
            zero, _ = geometry.fit_angular_geometry(torch.zeros_like(c),epsilon)
            assert geometry.transform_tokens(x,zero) is x
            original_zero = torch.zeros(1,12,dtype=torch.float64)
            assert torch.equal(geometry.transform_tokens(original_zero,l),original_zero)
            assert torch.linalg.matrix_rank(l) == len(l)
            rows.append(dict(index=index,full_rank=True,angular_error=float((y@y.T-reference).abs().max()),
                             identity_object_exact=True, native_zero_convention_preserved=True))
    receipt = dict(state='CPU_NATIVE_ANGULAR_RUNTIME_PASSED',records=rows, CUDA_initialized=torch.cuda.is_initialized(),
                   pretrained_DINO=False, real_image_gain=False)
    if args.out:
        if Path(args.out).exists(): raise ValueError('Preserve existing CPU checks')
        write_json(args.out,receipt)
    print(json.dumps(dict(state=receipt['state'],cases=10,CUDA_initialized=receipt['CUDA_initialized'])))


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from native_rice_core_experiment import (digest,write_json,capture_native,frozen_native_replay,exact_clustering_cache)
    if not args.allow_gpu or os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise RuntimeError('Only the prepared guarded CUDA stage may execute')
    guard = json.loads(Path(args.resource_guard_state).read_text())
    if guard['state'] != 'GPU_RUNNING' or guard['last_event'].get('stage') != 'native_angular40':
        raise RuntimeError('Current owned native-angular GPU stage required')
    manifest = json.loads(Path(args.prepared_manifest).read_text())
    if manifest.get('schema') != 'native_angular_assets_v1' or manifest.get('arms') != list(ARMS):
        raise ValueError('Frozen angular protocol required')
    for asset in manifest['assets']:
        st = Path(asset['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(asset['size'],asset['mtime_ns']): raise RuntimeError('Prepared asset drift')
    for source, expected in manifest['source_hashes'].items():
        if digest(source)!=expected: raise RuntimeError('Prepared source drift')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh owned output required')
    out.mkdir(parents=True,exist_ok=True)
    report = dict(state='RUNNING',records=[],arms=list(ARMS),manifest_sha256=digest(args.prepared_manifest),
                  feature_cache_written=False,query_GT_used_in_prediction=False,config=manifest['fixed_config'])
    started = time.monotonic()
    def save():
        report['elapsed_s']=time.monotonic()-started;write_json(out/'report.json',report)
    save()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA; model not loaded')
        torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.3)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(Path(args.prepared_root)/'scripts'));import _paths
        sys.path.insert(0,args.foris_root);import models.foris as foris_module
        from utils.data import build_transform
        sys.path.insert(0,_paths.DEMO4);from icx.common import TimmDINOv3
        sys.path.insert(0,str(HERE.parent))
        from tics.native_assets import reuse_native_basis
        from tics.reference_views import support_background_views
        from tics.rice_statistics import build_reference_statistics
        from tics.reference_nuisance_geometry import fit_angular_geometry
        from tics.native_geometry_adapter import native_geometry_part2
        from tics.readout_baselines import frost_whitening
        def iu(a,b):return [int((a&b).sum()),int((a|b).sum())]
        def packed(a):return dict(shape=list(a.shape),codec='zlib_np_packbits_big',data=base64.b64encode(zlib.compress(np.packbits(a.flatten()).tobytes(),6)).decode())
        with torch.inference_mode():
            encoder=TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(foris_module.FoRIS,args.projection_basis) as receipt:
                host=foris_module.FoRIS(encoder=encoder,image_size=1024,svd_components=500,tau=.6,
                    mask_refiner='crf',resize_to_orig_size=False,device='cuda').eval().requires_grad_(False)
            transform=build_transform(1024);report['basis_receipt']=receipt
            for row in manifest['frozen_episodes']:
                begin=time.monotonic();c=row['c']
                s=transform(Image.open(Path(manifest['data_root'])/row['support']).convert('RGB')).cuda()[None]
                q=transform(Image.open(Path(manifest['data_root'])/row['query']).convert('RGB')).cuda()[None]
                gold=torch.from_numpy((np.asarray(Image.open(Path(manifest['annotation_root'])/Path(row['support']).with_suffix('.png')))==c+1).copy()).cuda()
                mask=F.interpolate(gold[None,None].float(),(1024,1024),mode='nearest')[0].bool()
                with exact_clustering_cache(foris_module) as cache:
                    with capture_native(host) as packet:native=host.predict(s,mask,q[0]).reshape(1024,1024).bool().clone()
                    raw,part1,native_tuple=[packet[k] for k in ('_extract_features','_part1_positional_debias','_part2_background_suppression')]
                    grid=tuple(part1.shape[-2:])
                    def encode_view(rgb):
                        if torch.equal(rgb,s):return part1[:,0].clone()
                        f=F.normalize(host._extract_features(rgb[:,None]),dim=2)
                        if packet['apd_applied']:f=host._debias_features(f)
                        return f[:,0]
                    views_bank=support_background_views(s,mask,encode_view)
                    views=torch.stack([f[0].flatten(1).T.float() for f in views_bank.maps])
                    coverage=F.interpolate(gold[None,None].float(),grid,mode='area')[0,0].flatten()
                    stats=build_reference_statistics(views,coverage,grid)
                    predictions={'foris_crf':native};methods={}
                    stage_scores={k:native_tuple[0].detach().cpu().numpy().copy() for k in ARMS[:2]}
                    zero=torch.zeros(part1.shape[2],part1.shape[2],device='cuda',dtype=torch.float64)
                    identity, _ =fit_angular_geometry(zero,1.)
                    identity_tuple=native_geometry_part2(host,part1,native_tuple,identity,mask,grid)
                    if identity_tuple is not native_tuple:raise RuntimeError('Native identity tuple bypass failed')
                    with frozen_native_replay(host,raw,part1,identity_tuple):identity_prediction=host.predict(s,mask,q[0]).reshape(1024,1024).bool().clone()
                    if not torch.equal(native,identity_prediction):raise RuntimeError('Real native identity is not pixel-exact')
                    predictions['native_identity']=identity_prediction
                    methods['native_identity']=dict(state='EXACT_NATIVE_IDENTITY',same_part2_tuple=True)
                    reference_audits={k:reference_orbit_audit(views,stats['foreground_indices'],identity) for k in ARMS[:2]}
                    if stats['state'].startswith('DEGENERATE'):
                        for arm in ARMS[2:]:
                            predictions[arm]=native.clone();methods[arm]=dict(state=stats['state'],exact_native_fallback=True)
                            stage_scores[arm]=stage_scores['foris_crf'].copy()
                            reference_audits[arm]=reference_audits['foris_crf'].copy()
                    else:
                        for arm,key in [('paired_geometry','paired_only'),('pooled_geometry','pooled_fisher'),
                                        ('fg_aug_geometry','fg_aug'),('shuffled_geometry','paired_shuffle'),('frost95_geometry',None)]:
                            if key is not None:
                                geometry,audit=fit_angular_geometry(stats['covariances'][key],stats['ridge'])
                            else:
                                fg=views[:,stats['foreground_indices']].reshape(-1,views.shape[-1])
                                bg=views[:,stats['background_indices']].reshape(-1,views.shape[-1])
                                geometry=frost_whitening(fg,bg).double()
                                # Positive global scale leaves the angular metric unchanged.
                                geometry=geometry/geometry.norm()
                                audit=dict(state='FROST95_COMPONENT',shrinkage=.95,eigenfloor=1e-6,
                                    full_rank=True,component_not_full_FROST=True,source_function='readout_baselines.frost_whitening')
                            new_tuple=native_geometry_part2(host,part1,native_tuple,geometry,mask,grid)
                            stage_scores[arm]=new_tuple[0].detach().cpu().numpy().copy()
                            reference_audits[arm]=reference_orbit_audit(views,stats['foreground_indices'],geometry)
                            if new_tuple[3] is not native_tuple[3] or new_tuple[4] is not native_tuple[4]:raise RuntimeError('Original downstream geometry was replaced')
                            with frozen_native_replay(host,raw,part1,new_tuple):pred=host.predict(s,mask,q[0]).reshape(1024,1024).bool().clone()
                            predictions[arm]=pred;methods[arm]=dict(state='COMPLETED',geometry=audit,
                                changed_model_pixels=int((pred!=native).sum()),native_score_functions_preserved=True,
                                original_downstream_geometry_objects_preserved=True)
                    if set(predictions)!=set(ARMS):raise RuntimeError('Missing declared output')
                    torch.cuda.synchronize();frozen={k:v.cpu().numpy().astype(bool) for k,v in predictions.items()}
                    inference_s=time.monotonic()-begin
                    # Every branch is immutable before opening this query's annotation.
                    truth_original=np.asarray(Image.open(Path(manifest['annotation_root'])/Path(row['query']).with_suffix('.png')))==c+1
                    truth_tensor=torch.from_numpy(truth_original.copy()).cuda()
                    truth_model=F.interpolate(truth_tensor[None,None].float(),(1024,1024),mode='nearest')[0,0].bool().cpu().numpy()
                    query_coverage=F.interpolate(truth_tensor[None,None].float(),grid,mode='area')[0,0].cpu().numpy()
                    query_audits={k:query_score_audit(stage_scores[k],query_coverage) for k in ARMS}
                    original={k:(F.interpolate(v[None,None].float(),truth_original.shape,mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy() for k,v in predictions.items()}
                    base=frozen['foris_crf']
                    ledger={k:dict(recovered_fn=int((~base&a&truth_model).sum()),lost_tp=int((base&~a&truth_model).sum()),
                        added_fp=int((~base&a&~truth_model).sum()),removed_fp=int((base&~a&~truth_model).sum())) for k,a in frozen.items()}
                    report['records'].append(dict(**row,iu={k:iu(a,truth_model) for k,a in frozen.items()},
                        original_iu={k:iu(a,truth_original) for k,a in original.items()},ledger=ledger,methods=methods,
                        replay_identity_exact=True,query_GT_opened_after_all_predictions=True,
                        stats=stats['audit'],statistics_state=stats['state'],native_APD=packet['apd_applied'],
                        evidence_chain=dict(reference_orbit=reference_audits,query_score=query_audits,
                            query_GT_used_only_after_all_predictions=True,diagnostic_scores_never_drive_selection=True),
                        prediction_bits={k:packed(a) for k,a in frozen.items()},evaluator_bits=packed(truth_model),
                        cost=dict(inference_s=inference_s,clustering_cache=cache.copy(),support_encoder_calls=3,query_encoder_calls=1)))
                    save();print(json.dumps(dict(completed=len(report['records']),fold=row['fold'],c=c,inference_s=inference_s,
                        native_IU=report['records'][-1]['original_iu']['foris_crf'],paired_IU=report['records'][-1]['original_iu']['paired_geometry'])),flush=True)
                del raw,part1,native_tuple,packet,views,views_bank,stats,predictions,frozen
        report['state']='COMPLETED';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
    except BaseException as error:
        report.update(state='ERROR',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--self-check',action='store_true')
    p.add_argument('--from-manifest');p.add_argument('--prepared-manifest');p.add_argument('--projection-basis')
    p.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    p.add_argument('--foris-root',default='/root/autodl-tmp/demo8_local_verification/foris_source')
    p.add_argument('--resource-guard-state');p.add_argument('--allow-gpu',action='store_true');p.add_argument('--out')
    a=p.parse_args()
    if a.prepare_only:prepare(a)
    elif a.self_check:self_check(a)
    else:run(a)


if __name__=='__main__':main()
