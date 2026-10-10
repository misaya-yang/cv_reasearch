"""Read-only frozen reference-focus replay after all200 predictions seal."""
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
DATA = REPO.parent/'cv_data'
OUT = Path(__file__).resolve().parent
RUN = DATA/'a/reference_focus_head200_20261010'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def explicit(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    began = time.perf_counter()
    cfg, seal = read(RUN/'config.json'), read(RUN/'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and seal['n'] == cfg['n'] == 200
    for name, key in (('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('tasks.json', 'tasks_sha256'), ('inference.jsonl', 'inference_sha256')):
        assert sha(RUN/name) == seal[key]
    for rel, digest in cfg['source_sha256'].items():
        assert sha(RUN/'frozen'/rel) == digest
    for original, digest in cfg['external_source_sha256'].items():
        path = Path(original)
        rel = path.relative_to(DATA/'third_party/foris_official') if path.is_relative_to(DATA/'third_party/foris_official') else path.relative_to(DATA/'third_party/crf_source/src')
        frozen = RUN/'frozen/external'/('foris' if path.is_relative_to(DATA/'third_party/foris_official') else 'crf')/rel
        assert sha(path) == sha(frozen) == digest
    assert sha(RUN/'positional_basis.pt') == cfg['basis_sha256']
    assert sha(RUN/'profile.json') == cfg['profile_sha256']
    manifest, tasks = read(RUN/'manifest.json'), read(RUN/'tasks.json')
    records = [json.loads(line) for line in (RUN/'inference.jsonl').read_text().splitlines()]
    indices = (0, 100)
    forbidden = {str(Path(manifest[i]['query_mask_path']).resolve()) for i in indices}
    forbidden.update(str(Path(t['baseline']['path']).resolve()) for t in tasks)
    forbidden.update(str((RUN/name).resolve()) for name in ('report.json', 'scored_episodes.jsonl', 'COMPLETE.json'))
    assert all(str(Path(manifest[i]['reference_mask_path']).resolve()) not in forbidden for i in indices)
    attempts, writes = [], []
    prefix = str(RUN.resolve())+os.sep
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path in forbidden:
            attempts.append(path)
            raise RuntimeError('Forbidden query GT, baseline mask or scoring output')
        mode, flags = args[1:3]
        write = ((isinstance(mode, str) and any(c in mode for c in 'wa+'))
                 or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))))
        if path.startswith(prefix) and write:
            writes.append(path)
            raise RuntimeError('Forbidden sealed-experiment write')
    sys.addaudithook(guard)
    protected = [RUN/name for name in ('config.json', 'sealed.json', 'manifest.json', 'tasks.json', 'inference.jsonl')]
    protected += [RUN/folder/records[i]['filename'] for i in indices for folder in ('fields', 'predictions')]
    protected += [RUN/'records'/f'{i:06d}.json' for i in indices]
    before = {str(path): sha(path) for path in protected}
    sys.path[:0] = [str(RUN/'frozen/src'), str(RUN/'frozen/scripts'), str(RUN/'frozen/external/foris'),
                   str(DATA/'third_party/crf_source/src')]
    torch.set_num_threads(cfg['threads_per_worker'])
    torch.manual_seed(0)
    method_path = RUN/'frozen/src/ics/methods/reference_focus_head.py'
    head = explicit('independent_frozen_reference_focus', method_path)
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from ics.official_data import load_inputs, array_hash
    from utils.data import build_transform
    profile = Path(cfg['profile_path'])
    cache = RawFeatureCache(profile.parent.parent, read(RUN/'profile.json'))
    transform = build_transform(1024)
    basis = torch.load(RUN/'positional_basis.pt', map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis @ basis.T
    runtime = DATA/'runtime/macos/crf'
    native_paths = {'ics_permutohedral_m4': runtime/'ics_permutohedral_m4.so',
                    'ics_permutohedral_m4_prepared_v1': runtime/'prepared_v1/ics_permutohedral_m4_prepared_v1.so'}
    native = {name: explicit(name, path) for name, path in native_paths.items()}
    wrapper_path = RUN/'frozen/src/ics/m4_crf.py'
    tree = ast.parse(wrapper_path.read_text())
    installer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    wrapper = next(n for n in installer.body if isinstance(n, ast.ClassDef) and n.name == 'PermutohedralLayer')
    namespace = {'torch': torch, 'backend': native['ics_permutohedral_m4']}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(wrapper_path), 'exec'), namespace)
    filtering = types.ModuleType('PermutohedralFiltering')
    filtering.PermutohedralLayer = namespace['PermutohedralLayer']
    sys.modules['PermutohedralFiltering'] = filtering
    from models.foris import FoRIS
    from ics.native_basis import reuse_native_basis
    from ics.m4_crf_cached import enable_on_crf
    class ForbiddenEncoder(torch.nn.Module):
        def get_intermediate_layers(self, *args, **kwargs):
            raise RuntimeError('No encoder may run during independent replay')
    with reuse_native_basis(FoRIS, RUN/'positional_basis.pt'):
        host = FoRIS(encoder=ForbiddenEncoder(), image_size=1024, svd_components=500, tau=.6,
                     mask_refiner='crf', resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
    assert torch.equal(host.positional_basis, basis)
    enable_on_crf(host._crf, native['ics_permutohedral_m4_prepared_v1'])
    replays = []
    for index in indices:
        row, task, rec = manifest[index], tasks[index], records[index]
        assert row['episode_id'] == task['episode_id'] == rec['episode_id']
        assert sha(RUN/'fields'/rec['filename']) == rec['fields_sha256']
        assert sha(RUN/'predictions'/rec['filename']) == rec['prediction_sha256']
        reference, mask, query = load_inputs(row, DATA)
        canonical = transform.transforms[0](query)
        images = [reference, query]+[canonical.crop(box) for box in head.CORNER_BOXES]
        focus = task['focus']
        focus_image = transform.transforms[0](reference).crop(tuple(focus['box_xyxy']))
        assert array_hash(np.asarray(focus_image)) == focus['focus_rgb_hash']
        requests = task['raw']+[dict(role='reference_focus', key=focus['key'], entry_path=focus['entry_path'],
            entry_sha256=focus['entry_sha256'], payload_sha256=focus['payload_sha256'],
            input_tensor_hash=focus['input_tensor_hash'], tensor_sha256=focus['O24']['tensor_sha256'])]
        arrays, raw_checks = [], []
        for image, request in zip(images+[focus_image], requests):
            model_input = transform(image).numpy()
            assert cache.key(model_input) == request['key'] and tensor_hash(model_input) == request['input_tensor_hash']
            assert sha(request['entry_path']) == request['entry_sha256']
            entry = read(request['entry_path'])
            assert entry['file_sha256'] == request['payload_sha256']
            raw = cache.read(model_input, ('O/24',))['O/24']
            assert raw.shape == (4096, 1024) and raw.dtype == np.float32 and tensor_hash(raw) == request['tensor_sha256']
            arrays.append(torch.from_numpy(raw))
            raw_checks.append(dict(role=request['role'], key=request['key'], payload_sha256=request['payload_sha256'],
                                   tensor_sha256=request['tensor_sha256'], complete_payload_verified=True))
        mask1024 = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
        assert array_hash(mask1024.numpy().astype(np.uint8)) == focus['canonical_mask_uint8_array_sha256']
        box = tuple(focus['box_xyxy'])
        x0, y0, x1, y1 = box
        expected_c = mask1024[y0:y1, x0:x1].reshape(64, 8, 64, 8).mean((1, 3)).numpy()
        assert tensor_hash(expected_c) == focus['coverage64_tensor_sha256']
        maps = torch.stack([value.reshape(64, 64, 1024).permute(2, 0, 1).contiguous() for value in arrays[:2]])[None]
        normalized_maps = F.normalize(maps, dim=2)
        apd = bool(host._should_apply_positional_debias(normalized_maps, mask1024[None, None], 1))
        assert apd == rec['apd_applied']
        whole_r, whole_q = (F.normalize(value, dim=1) for value in arrays[:2])
        old_binary = F.interpolate(mask.float()[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
        mu = whole_q.mean(0)
        old_semantic = float(F.normalize(whole_r[old_binary].mean(0), dim=0) @ mu/(mu.norm()+1e-6)) if old_binary.any() else None
        old_gate = old_semantic is None or old_semantic < .8
        result = head.fit_predict(arrays[0], arrays[6], mask1024, box, arrays[1], arrays[2:6], apply_apd=apd, projection=projection)
        repeat = head.fit_predict(arrays[0], arrays[6], mask1024, box, arrays[1], arrays[2:6], apply_apd=apd, projection=projection)
        assert result['diagnostics']['physical_measure'] == rec['diagnostics']['physical_measure']
        assert result['diagnostics']['quadrature'] == rec['diagnostics']['quadrature']
        field_checks, mask_checks = {}, {}
        with np.load(RUN/'fields'/rec['filename'], allow_pickle=False) as saved:
            for arm, field in result['fields'].items():
                check = dict(saved_exact=np.array_equal(field, saved[arm]), repeat_exact=np.array_equal(field, repeat['fields'][arm]),
                             maximum_absolute_error=float(np.abs(field-saved[arm]).max()), tensor_sha256=array_hash(field))
                assert check['saved_exact'] and check['repeat_exact'], (arm, check)
                field_checks[arm] = check
        image = transform(query)[None]
        with np.load(RUN/'predictions'/rec['filename'], allow_pickle=False) as saved:
            for arm, values in result['fields'].items():
                field = torch.from_numpy(values).reshape(1, 1, 128, 128)
                direct = F.interpolate(field, (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > 0
                final = host._finalize_mask(direct, image)
                for suffix, prediction in (('direct', direct), ('crf', final)):
                    for frame, hw in (('cli1024', (1024, 1024)), ('original', tuple(row['query_size_hw']))):
                        rendered = prediction if hw == (1024, 1024) else F.interpolate(
                            prediction.float()[None, None], hw, mode='bilinear', align_corners=False)[0, 0] > .5
                        label = frame+'/'+arm+'.'+suffix
                        pixels = rendered.numpy()
                        stored = np.unpackbits(saved[label], count=pixels.size).reshape(hw).astype(bool)
                        check = dict(saved_exact=np.array_equal(pixels, stored), differing_pixels=int(np.count_nonzero(pixels != stored)))
                        assert check['saved_exact'], (label, check)
                        mask_checks[label] = check
        replays.append(dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'], raw=raw_checks,
                            focus_box=box, common_apd=apd, prior_row_normalized_gate=old_gate,
                            prior_gate_matches=old_gate == apd, prior_semantic=old_semantic,
                            physical_measure=result['diagnostics']['physical_measure'], quadrature=result['diagnostics']['quadrature'],
                            fields=field_checks, masks=mask_checks, full_fields_exact=len(field_checks), complete_masks_exact=len(mask_checks)))
    after = {str(path): sha(path) for path in protected}
    assert before == after and not attempts and not writes
    document = dict(status='PASSED_REFERENCE_FOCUS_TWO_ACTUAL_REPLAYS_NO_GT',
                    frozen_head_sha256=sha(method_path), frozen_runner_sha256=cfg['source_sha256']['scripts/run_reference_focus200.py'],
                    validator_sha256=sha(__file__), sealed_sha256=before[str(RUN/'sealed.json')],
                    all41_external_live_and_frozen_sources_verified=True, native_binary_sha256={str(path): sha(path) for path in native_paths.values()},
                    actual_replays=replays, query_GT_reads=0, baseline_mask_reads=0, encoder_calls=0,
                    new_raw_feature_writes=0, denied_input_open_attempts=attempts, sealed_write_attempts=writes,
                    protected_file_hashes_unchanged=before == after, elapsed_seconds=time.perf_counter()-began,
                    scope='Two actual Deep/PACO frozen replays; no scoring or scientific performance/novelty conclusion.')
    (OUT/'reference_focus_actual_replay.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=document['status'], actual=[dict(index=r['index'], episode_id=r['episode_id'],
        fields_exact=r['full_fields_exact'], masks_exact=r['complete_masks_exact'], prior_gate_matches=r['prior_gate_matches']) for r in replays],
        query_GT_reads=0, baseline_mask_reads=0, sealed_unchanged=True, seconds=document['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
