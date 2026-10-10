"""Read-only frozen actual-pair replay with explicit forbidden-input guards."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
DATA = REPO.parent / 'cv_data'
OUT = Path(__file__).resolve().parent
RUN = DATA / 'a/autonomous_scene_reconstruction600_20261010'
INDEX = 400


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_explicit(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    began = time.perf_counter()
    cfg = read(RUN/'config.json')
    seal = read(RUN/'sealed.json')
    assert seal['n'] == cfg['n'] == 600
    assert sha(RUN/'config.json') == seal['config_sha256']
    assert sha(RUN/'manifest.json') == cfg['manifest_sha256']
    assert sha(RUN/'tasks.json') == cfg['tasks_sha256']
    manifest, tasks = read(RUN/'manifest.json'), read(RUN/'tasks.json')
    row, task, receipt = manifest[INDEX], tasks[INDEX], seal['receipts'][INDEX]
    assert row['episode_id'] == task['episode_id'] == receipt['episode_id']
    prediction_path, fields_path = RUN/'predictions'/receipt['filename'], RUN/'fields'/receipt['filename']
    assert sha(prediction_path) == receipt['prediction_sha256']
    assert sha(fields_path) == receipt['fields_sha256']
    protected = [RUN/'sealed.json', RUN/'config.json', RUN/'manifest.json', RUN/'tasks.json',
                 RUN/'records'/f'{INDEX:06d}.json', prediction_path, fields_path]
    before = {str(path): sha(path) for path in protected}

    forbidden = {str(Path(row['query_mask_path']).resolve()),
                 *(str(Path(b['path']).resolve()) for b in task['baselines'].values()),
                 str((RUN/'scored_episodes.jsonl').resolve()), str((RUN/'report.json').resolve())}
    assert str(Path(row['reference_mask_path']).resolve()) not in forbidden
    attempts, protected_write_attempts = [], []
    root = str(RUN.resolve())+os.sep
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path in forbidden:
            attempts.append(path)
            raise RuntimeError('Forbidden query GT/baseline/scoring read')
        mode, flags = args[1], args[2]
        writes = ((isinstance(mode, str) and any(c in mode for c in 'wa+'))
                  or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))))
        if path.startswith(root) and writes:
            protected_write_attempts.append(path)
            raise RuntimeError('Forbidden sealed-experiment write')
    sys.addaudithook(guard)

    for relative, digest in cfg['source_sha256'].items():
        assert sha(RUN/'frozen'/relative) == digest
    for path, digest in cfg['external_source_sha256'].items():
        assert sha(path) == digest
    assert sha(cfg['profile_path']) == cfg['profile_sha256']
    assert sha(cfg['basis_path']) == cfg['basis_sha256']
    sys.path[:0] = [str(RUN/'frozen/src'), str(RUN/'frozen/scripts'),
                   str(DATA/'third_party/foris_official'), str(DATA/'third_party/crf_source/src')]
    method_path = RUN/'frozen/src/ics/methods/autonomous_context_transport.py'
    method = load_explicit('independent_frozen_scene_replay', method_path)
    from ics.official_data import load_inputs, array_hash
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from utils.data import build_transform
    torch.set_num_threads(cfg['threads_per_worker'])
    torch.manual_seed(0)
    profile_path = Path(cfg['profile_path'])
    cache = RawFeatureCache(profile_path.parent.parent, read(profile_path))
    transform = build_transform(1024)
    reference_image, reference_mask, query_image = load_inputs(row, DATA)
    tensors, raw_checks = [], []
    for image, request in zip((reference_image, query_image), task['raw']):
        model_input = transform(image).numpy()
        assert cache.key(model_input) == request['key']
        assert tensor_hash(model_input) == request['input_sha256']
        entry_path = cache.folder/request['key']/'entry.json'
        assert sha(entry_path) == request['entry_sha256']
        entry = read(entry_path)
        assert entry['file_sha256'] == request['payload_sha256']
        raw = cache.read(model_input, ('O/24',))['O/24']
        assert raw.dtype == np.float32 and raw.shape == (4096, 1024)
        assert tensor_hash(raw) == request['tensor_sha256']
        tensors.append(torch.from_numpy(raw))
        raw_checks.append(dict(role=request['role'], input_sha256=request['input_sha256'],
                               tensor_sha256=request['tensor_sha256'], payload_sha256=request['payload_sha256'],
                               reader_verifies_complete_payload=True, shape=list(raw.shape), dtype=str(raw.dtype)))
    r, q = (F.normalize(x, dim=1) for x in tensors)
    binary = F.interpolate(reference_mask.float()[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
    mu = q.mean(0)
    semantic = float(F.normalize(r[binary].mean(0), dim=0) @ mu/(mu.norm()+1e-6)) if binary.any() else None
    apply_apd = semantic is None or semantic < .8
    basis = torch.load(cfg['basis_path'], map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis @ basis.T
    if apply_apd:
        r, q = (F.normalize(x @ projection.T, dim=1) for x in (r, q))
    mask1024 = F.interpolate(reference_mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
    coverage = mask1024.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1).numpy()
    assert apply_apd == receipt['apd_applied'] and semantic == receipt['semantic']
    result = method.fit_predict(r, coverage, q, (64, 64))
    repeat = method.fit_predict(r, coverage, q, (64, 64))
    field_checks = {}
    with np.load(fields_path, allow_pickle=False) as saved:
        for arm, field in result['fields'].items():
            promoted = field.astype(np.float64).reshape(64, 64)
            field_checks[arm] = dict(saved_dtype=str(saved[arm].dtype), returned_dtype=str(field.dtype),
                exact_saved=np.array_equal(promoted, saved[arm]), exact_repeat=np.array_equal(field, repeat['fields'][arm]),
                maximum_absolute_error=float(np.abs(promoted-saved[arm]).max()),
                positive_tokens=int((field > 0).sum()), field_tensor_hash=array_hash(promoted))
            assert field_checks[arm]['exact_saved'] and field_checks[arm]['exact_repeat'], field_checks[arm]

    # Import the EXISTING binaries directly; no compilation or build-receipt writes.
    runtime = DATA/'runtime/macos/crf'
    native_paths = {'ics_permutohedral_m4': runtime/'ics_permutohedral_m4.so',
                    'ics_permutohedral_m4_prepared_v1': runtime/'prepared_v1/ics_permutohedral_m4_prepared_v1.so'}
    native = {name: load_explicit(name, path) for name, path in native_paths.items()}
    # Execute the exact frozen unprepared layer class from install(), without calling install().
    wrapper_path = RUN/'frozen/src/ics/m4_crf.py'
    tree = ast.parse(wrapper_path.read_text())
    install_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    wrapper_class = next(n for n in install_node.body if isinstance(n, ast.ClassDef) and n.name == 'PermutohedralLayer')
    namespace = {'torch': torch, 'backend': native['ics_permutohedral_m4']}
    exec(compile(ast.Module(body=[wrapper_class], type_ignores=[]), str(wrapper_path), 'exec'), namespace)
    filtering_module = types.ModuleType('PermutohedralFiltering')
    filtering_module.PermutohedralLayer = namespace['PermutohedralLayer']
    sys.modules['PermutohedralFiltering'] = filtering_module
    from models.foris import FoRIS
    from ics.native_basis import reuse_native_basis
    from ics.m4_crf_cached import enable_on_crf
    class ForbiddenEncoder(torch.nn.Module):
        def get_intermediate_layers(self, *args, **kwargs):
            raise RuntimeError('Validator must never construct/call DINO')
    with reuse_native_basis(FoRIS, cfg['basis_path']):
        host = FoRIS(encoder=ForbiddenEncoder(), image_size=1024, svd_components=500, tau=.6,
                     mask_refiner='crf', resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
    enable_on_crf(host._crf, native['ics_permutohedral_m4_prepared_v1'])
    image = transform(query_image)[None]
    shape = tuple(row['query_size_hw'])
    mask_checks = {}
    with np.load(prediction_path, allow_pickle=False) as saved:
        for arm, values in result['fields'].items():
            field = torch.as_tensor(values, dtype=torch.float64).reshape(1, 1, 64, 64)
            direct = F.interpolate(field, (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > 0
            final = host._finalize_mask(direct, image)
            for suffix, predicted in (('direct', direct), ('crf', final)):
                for frame, hw in (('cli1024', (1024, 1024)), ('original', shape)):
                    rendered = predicted if hw == (1024, 1024) else F.interpolate(
                        predicted.float()[None, None], hw, mode='bilinear', align_corners=False)[0, 0] > .5
                    label = frame+'/'+arm+'.'+suffix
                    actual = rendered.numpy()
                    original = np.unpackbits(saved[label], count=actual.size).reshape(hw).astype(bool)
                    check = dict(exact_saved=np.array_equal(actual, original),
                                 differing_pixels=int(np.count_nonzero(actual != original)), shape=list(hw),
                                 prediction_tensor_hash=array_hash(actual))
                    mask_checks[label] = check
                    assert check['exact_saved'], (label, check)
    after = {str(path): sha(path) for path in protected}
    assert before == after and not attempts and not protected_write_attempts
    smoke = read(OUT/'real_smoke4.json')
    crf_sources = {path: sha(path) for path in smoke['CRF_sources_sha256']}
    assert crf_sources == smoke['CRF_sources_sha256']
    receipt_doc = dict(status='PASSED_ACTUAL_SEALED_REPEAT_NO_GT', index=INDEX, episode_id=row['episode_id'], dataset=row['dataset'],
                       frozen_method_path=str(method_path), frozen_method_sha256=sha(method_path),
                       frozen_runner_sha256=cfg['source_sha256']['scripts/autonomous_scene_reconstruction.py'],
                       validator_sha256=sha(__file__), source_manifest_sha256=cfg['manifest_sha256'],
                       sealed_sha256=before[str(RUN/'sealed.json')], raw_checks=raw_checks,
                       common_apd=dict(applied=apply_apd, semantic=semantic,
                                       coverage_tensor_hash=array_hash(coverage), mask_pixel_role='reference only'),
                       fields=field_checks, masks=mask_checks, saved_complete_mask_count=len(mask_checks),
                       forbidden_input_open_attempts=attempts, sealed_write_attempts=protected_write_attempts,
                       protected_file_hashes_unchanged=before == after, query_GT_reads=0, baseline_mask_reads=0,
                       encoder_calls=0, raw_feature_writes=0,
                       native_binaries={str(path): sha(path) for path in native_paths.values()},
                       crf_sources_sha256=crf_sources, frozen_wrapper_sha256=sha(wrapper_path),
                       backend_note='Existing native binaries loaded directly; frozen wrapper extracted by AST; original FoRIS CRF settings; prepared geometry enabled.',
                       runtime=dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(), python=sys.version),
                       elapsed_seconds=time.perf_counter()-began,
                       limitations=['One actual LVIS sample; no segmentation scoring or generalization/novelty claim.',
                                    'CRF source/binary hashes were recorded at replay; config itself remains incompletely bound to the external CRF tree.'])
    (OUT/'actual_repeat_index400_no_gt.json').write_text(json.dumps(receipt_doc, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt_doc['status'], index=INDEX, episode_id=row['episode_id'],
                         all_fields_exact=all(x['exact_saved'] and x['exact_repeat'] for x in field_checks.values()),
                         complete_masks_exact=len(mask_checks), query_GT_reads=0, baseline_mask_reads=0,
                         sealed_unchanged=before == after, seconds=receipt_doc['elapsed_seconds'])))


if __name__ == '__main__':
    main()
