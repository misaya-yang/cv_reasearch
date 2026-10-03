#!/usr/bin/env python3
"""Joint CPU readiness of ACTUAL installed FoRIS/FROST source with fake encoder.

Only constructor positional-basis calculation is scoped to a D8/rank2 fixture.
All prediction/scoring/clustering/density/postprocess functions are source code.
No hub factory, pretrained weights, CUDA, CRF, network, or task-quality claim.
Missing source/import dependency returns 2 with an explicit pending receipt.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

# Process-local CPU boundary, before importing torch or source dependencies.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def source_files(foris, frost):
    paths = [foris/'models/foris.py']
    paths += sorted(path for path in (foris/'utils').rglob('*.py')
                    if not any(part.startswith('._') or part == '__MACOSX' for part in path.parts))
    if (foris/'models/__init__.py').is_file(): paths.append(foris/'models/__init__.py')
    paths += [frost/'frost'/name for name in ('model.py', 'density.py', 'data.py', 'encoder.py')]
    return paths


@contextmanager
def isolated_foris(root):
    """Execute actual packages while preserving other models/utils aliases."""
    old_modules = {name: value for name, value in sys.modules.items()
                   if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.')}
    old_path = list(sys.path)
    for name in old_modules: del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        source = importlib.import_module('models.foris')
        if Path(source.__file__).resolve() != (root/'models/foris.py').resolve():
            raise RuntimeError('Imported a different FoRIS source tree')
        yield source
    finally:
        for name in list(sys.modules):
            if name == 'models' or name.startswith('models.') or name == 'utils' or name.startswith('utils.'):
                del sys.modules[name]
        sys.modules.update(old_modules)
        sys.path[:] = old_path


@contextmanager
def observe_actual_clustering(source):
    # Observes original function inputs; returns the exact source result unchanged.
    original = source.agglomerative_clustering
    calls = []
    def observed(values, *args, **kwargs):
        calls.append(dict(shape=list(values.shape), descriptor_dimension=int(values.shape[-1])))
        return original(values, *args, **kwargs)
    source.agglomerative_clustering = observed
    try: yield calls
    finally: source.agglomerative_clustering = original


def methods_snapshot(host, names):
    return {name: (name in host.__dict__, host.__dict__.get(name)) for name in names}


def methods_restored(host, snapshot):
    return all((name in host.__dict__) == existed and
               (not existed or host.__dict__.get(name) is value)
               for name, (existed, value) in snapshot.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foris-root', type=Path, required=True)
    parser.add_argument('--frost-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('Fresh receipt required; preserve prior evidence')
    foris, frost = args.foris_root.resolve(), args.frost_root.resolve()
    base = Path(__file__).resolve().parents[1]
    required = [foris/'models/foris.py', foris/'utils/clustering.py',
                foris/'utils/data.py', foris/'utils/refinement.py']
    required += [frost/'frost'/name for name in ('model.py', 'density.py', 'data.py', 'encoder.py')]
    own = [Path(__file__), base/'tics/native_decision_trace.py', base/'tics/frost_existing_adapter.py']
    files = source_files(foris, frost) + own
    report = dict(state='PREPARING_CPU_SOURCE', records=[], cases_requested=10,
        source_hashes={str(path):sha(path) for path in files if path.is_file()},
        scope='Actual installed source functions on D8 synthetic features, no pretrained or task-gain claim',
        fixture=dict(image_size=64, grid=[4,4], descriptor_dimension=8,
            positional_basis_rank=2, source_basis_constructor_only_scoped_patch=True,
            NOT_production_APD250=True, NOT_real_DINO=True, FoRIS_refiner='bilinear', CRF_executed=False),
        no_downloads=True, hub_factory_called=False, CUDA_initialized=False)
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.out)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        report.update(state='PENDING_SOURCE_DEPENDENCY', missing_source=missing,
                      actual_source_execution=False, cases_completed=0, exit_code=2)
        save();print(json.dumps(dict(state=report['state'],missing_source=missing)),flush=True)
        return 2
    save()
    torch = None
    stage = 'import_torch'
    try:
        import torch
        import torch.nn.functional as F
        from unittest.mock import patch
        from PIL import Image
        if torch.cuda.is_initialized(): raise RuntimeError('CUDA was initialized before this CPU audit')
        torch.set_num_threads(1)
        def forbidden(*args, **kwargs):
            raise RuntimeError('CPU source audit attempted CUDA initialization or hub loading')
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.cuda.init', side_effect=forbidden), \
             patch('torch.cuda._lazy_init', side_effect=forbidden), \
             patch('torch.hub.load', side_effect=forbidden):
            stage = 'import_own_trace_and_adapter'
            trace = load_module(base/'tics/native_decision_trace.py', '_demo9_membership_cpu_trace')
            adapter = load_module(base/'tics/frost_existing_adapter.py', '_demo9_membership_cpu_frost_adapter')
            stage = 'import_actual_FROST_source'
            frost_source, frost_encoder, frost_hashes = adapter.existing_frost_modules(frost)
            report['FROST_namespace'] = frost_source.__name__
            report['source_hashes'].update(frost_hashes)
            class SyntheticEncoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.ones((),dtype=torch.float32), requires_grad=False)
                    self.calls = []
                def get_intermediate_layers(self, rgb, n=1, reshape=True):
                    assert rgb.device.type == 'cpu' and rgb.ndim == 4
                    assert rgb.shape[1:] == (3,64,64) and n == 1 and reshape is True
                    values = F.avg_pool2d(rgb,16)
                    maps = torch.cat((values,values.square(),values[:,:2]-values[:,1:3]),dim=1) * self.scale
                    self.calls.append(dict(batch=int(rgb.shape[0]),shape=list(rgb.shape),dtype=str(rgb.dtype),
                        requires_grad=bool(rgb.requires_grad),grad_enabled=torch.is_grad_enabled(),
                        input_sha256=hashlib.sha256(rgb.detach().contiguous().numpy().tobytes()).hexdigest()))
                    return [maps]
            def basis(this, device):
                assert str(device) == 'cpu'
                return torch.eye(8,dtype=torch.float32)[:,:2].clone()
            stage = 'import_actual_FoRIS_source'
            with isolated_foris(foris) as foris_source:
                report['FoRIS_source_module'] = str(Path(foris_source.__file__).resolve())
                for index in range(10):
                    stage = 'construct_actual_source_models_case_' + str(index)
                    seed = 2081 + index
                    generator = torch.Generator(device='cpu').manual_seed(seed)
                    support_pixels = torch.randint(0,256,(64,64,3),generator=generator,dtype=torch.uint8)
                    query_pixels = torch.randint(0,256,(64,64,3),generator=generator,dtype=torch.uint8)
                    support_pil = Image.fromarray(support_pixels.numpy())
                    query_pil = Image.fromarray(query_pixels.numpy())
                    mask = torch.zeros(1,64,64,dtype=torch.bool);mask[:,16:48,16:48]=True
                    foris_raw = SyntheticEncoder().eval()
                    frost_raw = SyntheticEncoder().eval()
                    foris_builder = foris_source.FoRIS._build_positional_basis
                    frost_builder = frost_source.FROST._build_positional_basis
                    with torch.no_grad(), patch.object(foris_source.FoRIS,'_build_positional_basis',basis):
                        foris_host = foris_source.FoRIS(foris_raw,image_size=64,svd_components=2,
                            mask_refiner='bilinear',resize_to_orig_size=False,device='cpu').eval().requires_grad_(False)
                    with torch.no_grad(), patch.object(frost_source.FROST,'_build_positional_basis',basis):
                        extractor = frost_encoder.DINOv3FeatureExtractor(frost_raw,image_size=64,patch_size=16)
                        frost_host = frost_source.FROST(extractor,raw_encoder=frost_raw,image_size=64,
                            device='cpu',resize_to_orig_size=False).eval().requires_grad_(False)
                    support = foris_host._transform(support_pil).unsqueeze(0)
                    query = foris_host._transform(query_pil).unsqueeze(0)
                    assert torch.equal(frost_host._transform(support_pil).unsqueeze(0),support)
                    assert torch.equal(frost_host._transform(query_pil).unsqueeze(0),query)
                    assert foris_source.FoRIS._build_positional_basis is foris_builder
                    assert frost_source.FROST._build_positional_basis is frost_builder
                    assert foris_host.positional_basis.shape == frost_host.positional_basis.shape == (8,2)
                    assert not foris_raw.calls and not frost_raw.calls  # fixture constructor intentionally avoids BLACK/SVD
                    # Ensure pre-existing instance overrides survive the trace context.
                    foris_host._binarize_response = foris_host._binarize_response
                    foris_snapshot = methods_snapshot(foris_host, trace.STAGES)
                    frost_host._finalize_mask = frost_host._finalize_mask
                    frost_snapshot = methods_snapshot(frost_host, ['_finalize_mask'])
                    state_f = {key:value.clone() for key,value in foris_host.state_dict().items()}
                    state_r = {key:value.clone() for key,value in frost_host.state_dict().items()}
                    with torch.no_grad():
                        stage = 'actual_FoRIS_prediction_case_' + str(index)
                        assert foris_host._tgt_image is None
                        cluster_original = foris_source.agglomerative_clustering
                        with observe_actual_clustering(foris_source) as direct_cluster_calls:
                            original_f = foris_host.predict(support,mask,query[0]).clone()
                            with trace.trace_native_decisions(foris_host) as packet_f:
                                traced_f = foris_host.predict(support,mask,query[0])
                        assert foris_source.agglomerative_clustering is cluster_original
                        assert all(call['descriptor_dimension']==8 for call in direct_cluster_calls)
                        assert torch.equal(original_f,traced_f)
                        assert original_f.dtype == torch.bool and tuple(original_f.shape) == (64,64)
                        assert packet_f['state'] == 'CAPTURED' and packet_f['original_outputs_returned']
                        assert all(value.device.type == 'cpu' for value in packet_f['maps'].values())
                        assert tuple(packet_f['maps']['pre_refinement_mask'].shape) == (64,64)
                        assert tuple(packet_f['maps']['post_refinement_mask'].shape) == (64,64)
                        assert torch.equal(packet_f['maps']['post_refinement_mask'],traced_f)
                        for name in ['part2_score','part2_sf','part2_sbn','candidate_hard','candidate_vote',
                                     'seed_prior','part3_score','part4_penalty','part4_cluster_delta','part4_score']:
                            assert tuple(packet_f['maps'][name].shape) == (4,4)
                        assert methods_restored(foris_host,foris_snapshot)
                        stage = 'actual_FROST_prediction_case_' + str(index)
                        original_r = frost_host.predict_mask(support,mask,query).clone()
                        with adapter.capture_frost_finalization(frost_host) as packet_r:
                            traced_r = frost_host.predict_mask(support,mask,query)
                        assert torch.equal(original_r,traced_r)
                        assert original_r.dtype == torch.bool and tuple(original_r.shape) == (64,64)
                        assert packet_r['state'] == 'CAPTURED'
                        assert torch.equal(packet_r['final_mask'],traced_r)
                        assert tuple(packet_r['pre_refinement_mask'].shape) == (4,4)
                        if packet_r['continuous_density_available']:
                            ell = packet_r['continuous_posterior']
                            assert ell.device.type == 'cpu' and ell.shape == (4,4) and torch.isfinite(ell).all()
                            assert packet_r['density_tau'] == 0.
                            assert getattr(frost_host,'_post_ell',None) is None
                        assert methods_restored(frost_host,frost_snapshot)
                        # PRIMARY real public stateful API: preserve target RGB/position context.
                        stage = 'actual_FoRIS_PUBLIC_stateful_case_' + str(index)
                        def prepare_foris_public():
                            foris_host.set_reference(support_pil,mask)
                            foris_host.set_target(query_pil)
                            assert torch.equal(foris_host._ref_images,support)
                            assert torch.equal(foris_host._ref_masks,mask)
                            assert torch.equal(foris_host._tgt_image,query[0])
                        prepare_foris_public()
                        with observe_actual_clustering(foris_source) as public_cluster_calls:
                            original_public_f = foris_host.segment().clone()
                            assert foris_host._tgt_image is None  # source public reset
                            prepare_foris_public()
                            with trace.trace_native_decisions(foris_host) as public_packet_f:
                                traced_public_f = foris_host.segment()
                        assert foris_source.agglomerative_clustering is cluster_original
                        assert torch.equal(original_public_f,traced_public_f)
                        assert original_public_f.shape==(64,64) and original_public_f.dtype==torch.bool
                        assert torch.equal(public_packet_f['maps']['post_refinement_mask'],traced_public_f)
                        assert public_packet_f['maps']['pre_refinement_mask'].shape==(64,64)
                        assert any(call['descriptor_dimension']==13 for call in public_cluster_calls)
                        assert methods_restored(foris_host,foris_snapshot)
                        stage = 'actual_FROST_PUBLIC_stateful_case_' + str(index)
                        def prepare_frost_public():
                            frost_host.set_reference(support_pil,mask)
                            frost_host.set_target(query_pil)
                            assert torch.equal(frost_host._ref_images,support)
                            assert torch.equal(frost_host._ref_masks,mask)
                            assert torch.equal(frost_host._tgt_image,query)
                        prepare_frost_public()
                        original_public_r = frost_host.segment().clone()
                        assert frost_host._tgt_image is None
                        prepare_frost_public()
                        with adapter.capture_frost_finalization(frost_host) as public_packet_r:
                            traced_public_r = frost_host.segment()
                        assert torch.equal(original_public_r,traced_public_r)
                        assert original_public_r.shape==(64,64) and original_public_r.dtype==torch.bool
                        assert torch.equal(public_packet_r['final_mask'],traced_public_r)
                        if public_packet_r['continuous_density_available']:
                            assert public_packet_r['continuous_posterior'].shape==(4,4)
                            assert torch.isfinite(public_packet_r['continuous_posterior']).all()
                        assert methods_restored(frost_host,frost_snapshot)
                    # Exceptions need to restore only these contexts' own instance changes.
                    for manager, host, snapshot in [(trace.trace_native_decisions,foris_host,foris_snapshot),
                                                    (adapter.capture_frost_finalization,frost_host,frost_snapshot)]:
                        try:
                            with manager(host): raise RuntimeError('intentional source context restoration fixture')
                        except RuntimeError as error:
                            assert str(error) == 'intentional source context restoration fixture'
                        assert methods_restored(host,snapshot)
                    assert all(torch.equal(value,state_f[key]) for key,value in foris_host.state_dict().items())
                    assert all(torch.equal(value,state_r[key]) for key,value in frost_host.state_dict().items())
                    assert all(parameter.grad is None for parameter in list(foris_host.parameters())+list(frost_host.parameters()))
                    assert [call['batch'] for call in foris_raw.calls] == [2,2,2,2]
                    assert [call['batch'] for call in frost_raw.calls] == [3,3,3,3]
                    assert foris_raw.calls[0]['input_sha256'] == foris_raw.calls[1]['input_sha256']
                    assert frost_raw.calls[0]['input_sha256'] == frost_raw.calls[1]['input_sha256']
                    assert len({call['input_sha256'] for call in foris_raw.calls})==1
                    assert len({call['input_sha256'] for call in frost_raw.calls})==1
                    assert not torch.cuda.is_initialized()
                    row = dict(index=index,seed=seed,FoRIS_actual_source_prediction_exact=True,
                        FROST_actual_source_prediction_exact=True,
                        FoRIS_actual_public_stateful_trace_exact=True,FROST_actual_public_stateful_trace_exact=True,
                        prepared_tensors_equal_source_public_transform=True,
                        FoRIS_direct_prior_clustering_calls=direct_cluster_calls,
                        FoRIS_PUBLIC_prior_clustering_calls=public_cluster_calls,
                        FoRIS_PUBLIC_RGB_position_branch_executed=True,
                        PUBLIC_not_required_equal_direct=True,
                        PUBLIC_changed_from_direct_pixels=int((traced_public_f!=traced_f).sum()),
                        FROST_PUBLIC_density_available=public_packet_r['continuous_density_available'],
                        FoRIS_stage_shapes={name:list(value.shape) for name,value in packet_f['maps'].items()},
                        FROST_continuous_density_available=packet_r['continuous_density_available'],
                        FROST_source_candidate_or_anchor_fallback=not packet_r['continuous_density_available'],
                        FROST_final_shape=list(traced_r.shape),FoRIS_encoder_calls=foris_raw.calls,
                        FROST_encoder_calls=frost_raw.calls,hook_restore_success_and_exception=True,
                        constructor_methods_restored=True,backbone_unchanged=True)
                    report['records'].append(row);report['cases_completed']=len(report['records']);save()
                    print(json.dumps(dict(case=index,FoRIS_source=True,FROST_source=True,
                        FROST_density=packet_r['continuous_density_available'],FoRIS_B=2,FROST_B=3)),flush=True)
            # Source hashes are immutable across all actual imports/predictions.
            for name, expected in report['source_hashes'].items():
                if sha(name) != expected: raise RuntimeError('CPU test source drift: '+name)
            report.update(state='CPU_INSTALLED_MEMBERSHIP_SOURCE_PASSED',cases_completed=10,
                actual_source_execution=True,synthetic_encoder=True,pretrained_DINO=False,
                task_gain_measured=False,CUDA_initialized=False,CUDA_discovery_suppressed=True,
                density_cases=sum(row['FROST_continuous_density_available'] for row in report['records']),
                source_fallback_cases=sum(row['FROST_source_candidate_or_anchor_fallback'] for row in report['records']),
                public_density_cases=sum(row['FROST_PUBLIC_density_available'] for row in report['records']),
                public_FoRIS_RGB_position_cases=10, public_same_mode_noop_gate_only=True, exit_code=0)
            save();return 0
    except (ImportError,ModuleNotFoundError,OSError) as error:
        report.update(state='PENDING_SOURCE_DEPENDENCY',failure_stage=stage,error=repr(error),
                      cases_completed=len(report['records']),exit_code=2)
        if torch is not None: report['CUDA_initialized']=torch.cuda.is_initialized()
        save();print(json.dumps(dict(state=report['state'],stage=stage,error=repr(error))),flush=True)
        return 2
    except BaseException as error:
        # Installed-extension/ABI errors can be RuntimeError/ValueError during
        # import, not only ImportError. They remain pending dependency readiness.
        import_pending = stage.startswith('import_') and isinstance(error, Exception)
        code = 2 if import_pending else 1
        report.update(state='PENDING_SOURCE_DEPENDENCY' if import_pending else 'CPU_INSTALLED_MEMBERSHIP_SOURCE_FAILED',
                      failure_stage=stage,error=repr(error),cases_completed=len(report['records']),exit_code=code)
        if torch is not None: report['CUDA_initialized']=torch.cuda.is_initialized()
        save();print(json.dumps(dict(state=report['state'],stage=stage,error=repr(error))),flush=True)
        return code


if __name__ == '__main__': raise SystemExit(main())
