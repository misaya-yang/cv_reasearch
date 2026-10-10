"""Paired real/derived reference-focus heads on frozen Deep100/PACO100.

prepare binds seven actual raw inputs and freezes code/configuration. run has
no encoder, raw-cache writer, query-label or archived-mask access. score opens
labels and the complete archived controls only after all predictions are sealed.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import builtins
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent/'cv_data'
DEFAULT = DATA/'a/reference_focus_head200_20261010'
HEAD_SHA = '37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413'
AUDIT_SHA = 'd809c7c19b5dc09239ebfb152f79ecc798be0785642e1d7acd1931a9aabc47f1'
FIELDS = ('actual.global', 'actual.local4', 'actual.equal',
          'derived.global', 'derived.local4', 'derived.equal')
ARMS = tuple(a+'.'+s for a in FIELDS for s in ('direct', 'crf'))
CONTROLS = ('foris.crf', 'mean', 'region.fast')
RAW_ROLES = ('reference_full', 'query_full', 'window_0', 'window_2', 'window_6', 'window_8')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def deny_paths(paths):
    """Deny labels and archived prediction files even to indirect file opens."""
    denied = {str(Path(p).resolve()) for p in paths}
    state = {'attempts': 0}
    original_builtin, original_io = builtins.open, io.open
    def guarded(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and str(Path(os.fsdecode(file)).resolve()) in denied:
                state['attempts'] += 1
                raise PermissionError('Prediction phase attempted a denied label/control file')
            return original(file, *args, **kwargs)
        return open_file
    try:
        builtins.open, io.open = guarded(original_builtin), guarded(original_io)
        yield state
    finally:
        builtins.open, io.open = original_builtin, original_io


def imports(code_root, external):
    sys.path[:0] = [str(code_root/'src'), str(code_root/'scripts'), str(external)]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import load_inputs, array_hash
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from utils.data import build_transform
    return np, torch, F, load_inputs, array_hash, RawFeatureCache, tensor_hash, build_transform


def prepare(out, assets, producer, workers):
    assert not (out/'config.json').exists(), 'Refuse to replace a prepared experiment'
    pilot = assets/'a/joint_role_pilot200_20261010'
    producer_complete = read(producer/'COMPLETE.json')
    assert producer_complete['state'] == 'REFERENCE_FOCUS_O24_COMPLETE' and producer_complete['n'] == 200
    pcfg = read(producer/'config.json')
    for name, key in (('config.json', 'config_sha256'), ('tasks.json', 'tasks_sha256'),
                      ('input_bindings.json', 'input_bindings_sha256')):
        assert sha(producer/name) == producer_complete[key]
    assert producer_complete['selection_audit_sha256'] == pcfg['selection_audit_sha256'] == AUDIT_SHA
    assert sha(REPO/'src/ics/methods/reference_focus_head.py') == HEAD_SHA
    manifest = read(pilot/'manifest.json')
    assert len(manifest) == 200 and sha(pilot/'manifest.json') == producer_complete['source_manifest_sha256']
    assert [r['dataset'] for r in manifest] == ['deepglobe_road']*100+['paco_part']*100
    old_cfg, old_seal = read(pilot/'config.json'), read(pilot/'sealed.json')
    assert old_seal['state'] == 'ALL_PREDICTIONS_SEALED' and old_seal['n'] == 200
    for file, key in (('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('inference.jsonl', 'inference_sha256')):
        assert sha(pilot/file) == old_seal[key]
    previous = {r['episode_id']: r for r in lines(pilot/'inference.jsonl')}
    focus_rows = read(producer/'input_bindings.json')
    assert [r['episode_id'] for r in focus_rows] == [r['episode_id'] for r in manifest]
    np, torch, F, load_inputs, array_hash, RawFeatureCache, tensor_hash, build_transform = imports(
        REPO, assets/'third_party/foris_official')
    torch.set_num_threads(2)
    profile = Path(pcfg['profile_path'])
    assert sha(profile) == pcfg['profile_sha256'] == old_cfg['raw_profile_sha256']
    cache, transform = RawFeatureCache(profile.parent.parent, read(profile)), build_transform(1024)
    assert cache.profile_id == pcfg['profile_id']
    tasks, baseline_paths = [], []
    for i, (row, focus) in enumerate(zip(manifest, focus_rows)):
        old = previous[row['episode_id']]
        baseline_path = pilot/'predictions'/old['filename']
        assert sha(baseline_path) == old['prediction_sha256']
        baseline_paths.append(str(baseline_path))
        assert [r['role'] for r in old['raw_inputs']] == list(RAW_ROLES)
        for key in ('reference_path', 'reference_crop', 'reference_rgb_hash', 'reference_mask_path', 'reference_mask_hash'):
            assert focus[key] == row.get(key), (i, key)
        assert focus['index'] == i and focus['actual_readback_verified']
        with deny_paths([row['query_mask_path'], baseline_path]):
            reference, mask, query = load_inputs(row, assets)
            canonical_query = transform.transforms[0](query)
            images = [reference, query]+[canonical_query.crop(box) for box in
                ((0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024))]
            whole_requests = []
            for image, prior in zip(images, old['raw_inputs']):
                x = transform(image).numpy()
                assert cache.key(x) == prior['key']
                entry = cache.folder/prior['key']/'entry.json'
                info = read(entry)
                assert info['file_sha256'] == prior['payload_sha256']
                assert info['features']['O/24']['tensor_sha256'] == prior['tensor_sha256']
                assert info['profile_id'] == cache.profile_id and info['input_tensor_hash'] == tensor_hash(x)
                assert info['features']['O/24']['shape'] == [4096, 1024]
                assert info['features']['O/24']['dtype'] == 'float32'
                whole_requests.append(dict(**prior, entry_path=str(entry), entry_sha256=sha(entry),
                    payload_path=str(entry.parent/info['file']), input_tensor_hash=info['input_tensor_hash']))
            crop = transform.transforms[0](reference).crop(tuple(focus['box_xyxy']))
            assert array_hash(np.asarray(crop)) == focus['focus_rgb_hash']
            x = transform(crop).numpy()
            assert cache.key(x) == focus['key'] and tensor_hash(x) == focus['input_tensor_hash']
            mask1024 = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
            assert array_hash(mask1024.numpy().astype(np.uint8)) == focus['canonical_mask_uint8_array_sha256']
        tasks.append(dict(index=i, episode_id=row['episode_id'], raw=whole_requests, focus=focus,
            baseline=dict(path=str(baseline_path), sha256=old['prediction_sha256']),
            input_binding='same complete original pilot manifest and six actual transformed cache requests'))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pilot/'manifest.json', out/'manifest.json')
    write(out/'tasks.json', tasks)
    local_files = ['scripts/run_reference_focus200.py', 'scripts/raw_feature_cache.py', 'scripts/cached_dino.py',
        'src/ics/__init__.py', 'src/ics/official_data.py', 'src/ics/metrics.py', 'src/ics/native_basis.py',
        'src/ics/m4_crf.py', 'src/ics/m4_crf_cached.py', 'src/ics/representations.py',
        'src/ics/methods/__init__.py', 'src/ics/methods/reference_focus_head.py']
    frozen = {}
    for rel in local_files:
        destination = out/'frozen'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/rel, destination)
        frozen[rel] = sha(destination)
    external = {}
    for p in (assets/'third_party/foris_official').rglob('*.py'):
        rel = p.relative_to(assets/'third_party/foris_official')
        dest = out/'frozen/external/foris'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        external[str(p)] = sha(p)
    for p in (assets/'third_party/crf_source/src').rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.hpp'):
            dest = out/'frozen/external/crf'/p.relative_to(assets/'third_party/crf_source/src')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            external[str(p)] = sha(p)
    basis = assets/'native_assets/positional_basis.pt'
    shutil.copyfile(basis, out/'positional_basis.pt')
    shutil.copyfile(profile, out/'profile.json')
    audit = REPO/'evidence/local/autonomous_20261010/mechanism_research/reference_focus_mask_audit_torch_nearest.json'
    assert sha(audit) == AUDIT_SHA
    shutil.copyfile(audit, out/'selection_audit.json')
    for name in ('config.json', 'COMPLETE.json', 'input_bindings.json'):
        shutil.copyfile(producer/name, out/('producer_'+name))
    paco_classes = old_cfg['paco_expected_classes']
    observed = {str(f): sorted({r['loader_class_id'] for r in manifest if r['dataset'] == 'paco_part' and r['fold'] == f}) for f in range(4)}
    assert sum(map(len, observed.values())) == 87 and sum(map(len, paco_classes.values())) == 303
    for folder in ('predictions', 'fields', 'records'):
        (out/folder).mkdir(exist_ok=True)
    write(out/'config.json', dict(n=200, method='paired actual versus derived reference focus',
        created_utc=datetime.now(timezone.utc).isoformat(), assets=str(assets), workers=workers, threads_per_worker=2,
        manifest_sha256=sha(out/'manifest.json'), tasks_sha256=sha(out/'tasks.json'),
        source_sha256=frozen, external_source_sha256=external,
        profile_path=str(profile), profile_sha256=sha(profile), basis_sha256=sha(basis),
        producer_complete_sha256=sha(out/'producer_COMPLETE.json'),
        producer_bindings_sha256=sha(out/'producer_input_bindings.json'), selection_audit_sha256=AUDIT_SHA,
        producer_root=str(producer), source_pilot_root=str(pilot), source_pilot_seal_sha256=sha(pilot/'sealed.json'),
        fields=list(FIELDS), candidate_arms=list(ARMS), complete_controls=list(CONTROLS),
        fixed_default='actual.equal.crf', matched_control='derived.equal.crf',
        source_gate='exact frozen FoRIS._should_apply_positional_debias on normalized actual whole R/Q maps and lawful mask',
        representation='seven actual CPU FP32 O24 arrays; all views share whole-query source APD; module preserves Q layout/order',
        renderer='signed FP32 field128 bilinear1024 align_corners=False strict>0; original boundary CRF; binary bilinear original frame >.5',
        primary_metric='CLI1024; Deep custom total foreground I/U; PACO87 observed classes fold/class pooled I/U',
        paco_observed_classes=observed, paco_expected_classes=paco_classes,
        additional_metric='original frame retained separately; PACO fixed303 class slots appendix',
        comparisons_fixed_before_score=['actual.equal.crf versus derived.equal.crf', 'actual.equal.crf versus complete FoRIS/MEAN/region.fast'],
        bootstrap=dict(repetitions=10000, unit='query_photo', seed=0, scope='conditional on exposed development images'),
        no_parameter_search=True, threshold=0, query_GT_in_inference=False, baseline_masks_in_inference=False,
        cost_scope='fourCPU workers/two threads: verified raw IO, paired six-field head, rendering and six CRFs; encoder cost reported separately',
        observations_per_episode=7, cached_existing_observations=6, added_actual_reference_observations=1,
        exposure='Previously exposed Deep custom100/PACO first25 per fold; exploration only; not untouched confirmation or ten-dataset result',
        denied_paths=[r['query_mask_path'] for r in manifest]+baseline_paths,
        runtime=dict(python=sys.version, packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'torchvision', 'pillow')})))
    write(out/'activity.json', dict(state='PREPARED', n=200, query_GT_reads=0, baseline_mask_reads=0))
    print(json.dumps(dict(state='PREPARED', root=str(out), n=200)), flush=True)


def verify_configuration(out):
    cfg = read(out/'config.json')
    for file, key in (('manifest.json', 'manifest_sha256'), ('tasks.json', 'tasks_sha256'),
                      ('producer_COMPLETE.json', 'producer_complete_sha256'),
                      ('producer_input_bindings.json', 'producer_bindings_sha256'),
                      ('selection_audit.json', 'selection_audit_sha256'), ('positional_basis.pt', 'basis_sha256'),
                      ('profile.json', 'profile_sha256')):
        assert sha(out/file) == cfg[key], ('Frozen input changed', file)
    assert sha(cfg['profile_path']) == cfg['profile_sha256']
    assert sha(Path(cfg['assets'])/'native_assets/positional_basis.pt') == cfg['basis_sha256']
    for rel, digest in cfg['source_sha256'].items():
        assert sha(out/'frozen'/rel) == digest
    assert sha(__file__) == cfg['source_sha256']['scripts/run_reference_focus200.py']
    for path, digest in cfg['external_source_sha256'].items():
        assert sha(path) == digest, ('External dependency changed', path)
    return cfg


def initialize(out):
    global OUT, CFG, CACHE, TRANSFORM, HOST, ADAPTER, PROJECTION, fit_predict
    global np, torch, F, load_inputs, array_hash, tensor_hash
    OUT = Path(out)
    CFG = verify_configuration(OUT)
    np, torch, F, load_inputs, array_hash, RawFeatureCache, tensor_hash, build_transform = imports(
        OUT/'frozen', OUT/'frozen/external/foris')
    torch.set_num_threads(CFG['threads_per_worker'])
    torch.manual_seed(0)
    class ReadOnlyCache(RawFeatureCache):
        def write(self, *args, **kwargs):
            raise PermissionError('Head experiment cannot write raw features')
        def _write(self, *args, **kwargs):
            raise PermissionError('Head experiment cannot write raw features')
    profile = Path(CFG['profile_path'])
    CACHE = ReadOnlyCache(profile.parent.parent, read(OUT/'profile.json'))
    TRANSFORM = build_transform(1024)
    import importlib.util
    module_path = OUT/'frozen/src/ics/methods/reference_focus_head.py'
    spec = importlib.util.spec_from_file_location('frozen_reference_focus_head', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fit_predict = module.fit_predict
    # Preload the frozen author module before cache_host adds the source root.
    from models.foris import FoRIS
    assert Path(sys.modules[FoRIS.__module__].__file__).resolve() == (OUT/'frozen/external/foris/models/foris.py').resolve()
    from cached_dino import CachedDINO, cache_host
    ADAPTER = CachedDINO(CACHE, encoder=None)
    HOST = cache_host(CFG['assets'], ADAPTER, mask_refiner='crf')
    basis = torch.load(OUT/'positional_basis.pt', map_location='cpu', weights_only=True)['basis'].float()
    assert torch.equal(HOST.positional_basis, basis)
    PROJECTION = torch.eye(1024)-basis@basis.T


def load_bound_inputs(row, task):
    """Replay every actual transform and verify payload, tensor and crop coverage."""
    reference, mask, query = load_inputs(row, CFG['assets'])
    canonical = TRANSFORM.transforms[0](query)
    images = [reference, query]+[canonical.crop(box) for box in
        ((0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024))]
    focus = task['focus']
    crop = TRANSFORM.transforms[0](reference).crop(tuple(focus['box_xyxy']))
    assert array_hash(np.asarray(crop)) == focus['focus_rgb_hash']
    requests = task['raw']+[dict(role='reference_focus', key=focus['key'], entry_path=focus['entry_path'],
        entry_sha256=focus['entry_sha256'], payload_path=focus['payload_path'], payload_sha256=focus['payload_sha256'],
        input_tensor_hash=focus['input_tensor_hash'], tensor_sha256=focus['O24']['tensor_sha256'])]
    arrays, checked = [], []
    for image, request in zip(images+[crop], requests):
        x = TRANSFORM(image).numpy()
        assert CACHE.key(x) == request['key'] and tensor_hash(x) == request['input_tensor_hash']
        assert sha(request['entry_path']) == request['entry_sha256']
        info = read(request['entry_path'])
        assert info['file_sha256'] == request['payload_sha256'] and info['features']['O/24']['tensor_sha256'] == request['tensor_sha256']
        assert Path(request['payload_path']).resolve() == (Path(request['entry_path']).parent/info['file']).resolve()
        raw = CACHE.read(x, ('O/24',))['O/24']
        assert raw.dtype == np.float32 and raw.shape == (4096, 1024) and tensor_hash(raw) == request['tensor_sha256']
        arrays.append(torch.from_numpy(raw))
        checked.append(request)
    mask1024 = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
    assert array_hash(mask1024.numpy().astype(np.uint8)) == focus['canonical_mask_uint8_array_sha256']
    x0, y0, x1, y1 = focus['box_xyxy']
    mask512 = mask1024[y0:y1, x0:x1]
    assert array_hash(mask512.numpy().astype(np.uint8)) == focus['focus_mask512_hash']
    expected_coverage = mask512.reshape(64, 8, 64, 8).mean((1, 3)).numpy()
    assert sha(focus['coverage64_path']) == focus['coverage64_file_sha256']
    coverage = np.load(focus['coverage64_path'], allow_pickle=False)
    assert tensor_hash(coverage) == focus['coverage64_tensor_sha256'] and np.array_equal(coverage, expected_coverage)
    return arrays, mask1024, TRANSFORM(query)[None], checked


def infer(task):
    index, row, source = task
    filename = f'{index:06d}.npz'
    record_path = OUT/'records'/f'{index:06d}.json'
    if record_path.exists():
        rec = read(record_path)
        assert rec['episode_id'] == row['episode_id']
        assert sha(OUT/'predictions'/filename) == rec['prediction_sha256']
        assert sha(OUT/'fields'/filename) == rec['fields_sha256']
        return rec
    begun = time.monotonic()
    with deny_paths(CFG['denied_paths']) as guard, torch.inference_mode():
        arrays, mask1024, image, checked = load_bound_inputs(row, source)
        io_seconds = time.monotonic()-begun
        tick = time.monotonic()
        maps = torch.stack([x.reshape(64, 64, 1024).permute(2, 0, 1).contiguous() for x in arrays[:2]])[None]
        normalized = F.normalize(maps, dim=2)
        apd = bool(HOST._should_apply_positional_debias(normalized, mask1024[None, None], 1))
        binary = (F.interpolate(mask1024[None, None], (64, 64), mode='nearest')[0, 0] > .5).reshape(-1)
        semantic = None
        if binary.any():
            target = normalized[0, -1].reshape(1024, -1).mean(1)
            foreground = normalized[0, 0].reshape(1024, -1)[:, binary]
            semantic = float(torch.dot(F.normalize(foreground.mean(1), dim=0), target)/(target.norm()+1e-6))
            assert apd == (semantic < .8)
        gate_seconds = time.monotonic()-tick
        tick = time.monotonic()
        result = fit_predict(arrays[0], arrays[6], mask1024, source['focus']['box_xyxy'], arrays[1], arrays[2:6],
            apply_apd=apd, projection=PROJECTION)
        assert set(result['fields']) == set(FIELDS)
        head_seconds = time.monotonic()-tick
        predictions, crf_seconds, rendering_seconds = {}, {}, 0.
        shape = tuple(row['query_size_hw'])
        for arm in FIELDS:
            tick = time.monotonic()
            field = torch.from_numpy(result['fields'][arm]).reshape(1, 1, 128, 128)
            assert field.dtype == torch.float32 and bool(torch.isfinite(field).all())
            direct = F.interpolate(field, (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > 0
            rendering_seconds += time.monotonic()-tick
            tick = time.monotonic()
            final = HOST._finalize_mask(direct, image)
            crf_seconds[arm] = time.monotonic()-tick
            tick = time.monotonic()
            for suffix, prediction in (('direct', direct), ('crf', final)):
                for frame, hw in (('cli1024', (1024, 1024)), ('original', shape)):
                    rendered = prediction if hw == (1024, 1024) else F.interpolate(
                        prediction.float()[None, None], hw, mode='bilinear', align_corners=False)[0, 0] > .5
                    predictions[frame+'/'+arm+'.'+suffix] = np.packbits(rendered.numpy().reshape(-1))
            rendering_seconds += time.monotonic()-tick
        assert ADAPTER.encoder is None and ADAPTER.encoder_calls == ADAPTER.cache_reads == 0 and guard['attempts'] == 0
        tick = time.monotonic()
        np.savez_compressed(OUT/'predictions'/filename, original_hw=np.asarray(shape), **predictions)
        np.savez_compressed(OUT/'fields'/filename, **result['fields'])
        serialization_seconds = time.monotonic()-tick
    rec = dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'], filename=filename,
        prediction_sha256=sha(OUT/'predictions'/filename), fields_sha256=sha(OUT/'fields'/filename),
        raw_requests=checked, apd_applied=apd, semantic=semantic, diagnostics=result['diagnostics'],
        io_seconds=io_seconds, gate_seconds=gate_seconds, head_seconds=head_seconds,
        crf_seconds=crf_seconds, rendering_seconds=rendering_seconds, serialization_seconds=serialization_seconds,
        total_seconds=time.monotonic()-begun, encoder_constructions=0, encoder_forward=0,
        raw_cache_writes=0, query_GT_reads=0, baseline_mask_reads=0, denied_file_attempts=guard['attempts'], CRF_calls=6)
    write(record_path, rec)
    return rec


def run(out):
    cfg = verify_configuration(out)
    if (out/'sealed.json').exists():
        raise ValueError('Refuse to replace a sealed experiment')
    manifest, sources = read(out/'manifest.json'), read(out/'tasks.json')
    assert len(manifest) == len(sources) == cfg['n']
    tasks = [(i, row, source) for i, (row, source) in enumerate(zip(manifest, sources))]
    start, receipts = time.monotonic(), []
    context = multiprocessing.get_context('spawn')
    write(out/'activity.json', dict(state='INFERENCE', pid=os.getpid(), completed=0, n=200))
    with concurrent.futures.ProcessPoolExecutor(cfg['workers'], mp_context=context,
            initializer=initialize, initargs=(str(out),)) as executor:
        futures = {executor.submit(infer, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            receipts.append((futures[future], future.result()))
            if len(receipts) % 10 == 0:
                state = dict(state='INFERENCE', completed=len(receipts), n=200, seconds=time.monotonic()-start)
                write(out/'activity.json', state)
                print(json.dumps(state), flush=True)
    records = [r for _, r in sorted(receipts)]
    assert len(records) == 200 and all(r['query_GT_reads'] == r['baseline_mask_reads'] == r['denied_file_attempts'] == 0 for r in records)
    (out/'inference.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    verify_configuration(out)
    write(out/'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=200,
        manifest_sha256=sha(out/'manifest.json'), config_sha256=sha(out/'config.json'),
        tasks_sha256=sha(out/'tasks.json'), inference_sha256=sha(out/'inference.jsonl'),
        inference_wall_seconds=time.monotonic()-start, workers=cfg['workers'], threads_per_worker=2,
        encoder_forward=0, raw_cache_writes=0, query_GT_reads=0, baseline_mask_reads=0, CRF_calls=1200))
    write(out/'activity.json', dict(state='SEALED', n=200))
    print(json.dumps(dict(state='SEALED', n=200, root=str(out))), flush=True)


def score(out):
    cfg = verify_configuration(out)
    assert not (out/'report.json').exists(), 'Retain the completed score; no additional statistical look'
    seal = read(out/'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and seal['n'] == 200
    for f, k in (('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                 ('tasks.json', 'tasks_sha256'), ('inference.jsonl', 'inference_sha256')):
        assert sha(out/f) == seal[k]
    np, torch, F, _, array_hash, *_ = imports(out/'frozen', out/'frozen/external/foris')
    torch.set_num_threads(2)
    from PIL import Image
    from ics.metrics import summarize, counts, gross_edits
    manifests, tasks, records = read(out/'manifest.json'), read(out/'tasks.json'), lines(out/'inference.jsonl')
    scored, groups = [], {}
    for row, task, rec in zip(manifests, tasks, records):
        assert row['episode_id'] == task['episode_id'] == rec['episode_id']
        assert sha(out/'predictions'/rec['filename']) == rec['prediction_sha256']
        assert sha(out/'fields'/rec['filename']) == rec['fields_sha256']
        prior = task['baseline']
        assert sha(prior['path']) == prior['sha256']
        with Image.open(row['query_mask_path']) as im:
            raw = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        assert array_hash(raw) == row['query_mask_hash']
        with np.load(out/'predictions'/rec['filename'], allow_pickle=False) as new, np.load(prior['path'], allow_pickle=False) as old:
            for frame, hw in (('cli1024', (1024, 1024)), ('original', tuple(row['query_size_hw']))):
                truth = F.interpolate(torch.from_numpy(raw).float()[None, None], hw, mode='nearest')[0, 0].numpy() > .5
                masks = {}
                for arm in CONTROLS:
                    masks[arm] = np.unpackbits(old[frame+'/'+arm], count=truth.size).reshape(hw).astype(bool)
                for arm in ARMS:
                    masks[arm] = np.unpackbits(new[frame+'/'+arm], count=truth.size).reshape(hw).astype(bool)
                entry = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
                    class_id=row['loader_class_id'], query_photo_id=row['query_photo_id'], frame=frame,
                    truth_pixels=int(truth.sum()), iu={a: counts(p, truth) for a, p in masks.items()},
                    edits={a: {b: gross_edits(masks[a], masks[b], truth) for b in ('foris.crf', 'derived.equal.crf')} for a in ARMS})
                scored.append(entry)
                groups.setdefault((row['dataset'], frame), []).append(entry)
    (out/'scored_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    results = {}
    for (dataset, frame), group in groups.items():
        expected = cfg['paco_observed_classes'] if dataset == 'paco_part' else None
        result = summarize(group, baselines=CONTROLS+('derived.equal.crf',),
            expected_classes=expected, repetitions=cfg['bootstrap']['repetitions'], unit='query_photo')
        result['metric_label'] = 'Deep custom total foreground I/U' if dataset == 'deepglobe_road' else 'PACO87 observed slots, fold/class pooled I/U'
        if dataset == 'paco_part':
            result['fixed303_slots'] = summarize(group, baselines=CONTROLS+('derived.equal.crf',),
                expected_classes=cfg['paco_expected_classes'], repetitions=cfg['bootstrap']['repetitions'], unit='query_photo')
        result['gross_edits'] = {a: {b: np.sum([r['edits'][a][b] for r in group], axis=0).tolist()
                                    for b in ('foris.crf', 'derived.equal.crf')} for a in ARMS}
        result['default_episode_comparisons'] = {}
        for base in CONTROLS+('derived.equal.crf',):
            delta = [100*(r['iu']['actual.equal.crf'][0]/max(r['iu']['actual.equal.crf'][1], 1)
                         -r['iu'][base][0]/max(r['iu'][base][1], 1)) for r in group]
            result['default_episode_comparisons'][base] = dict(
                improved=sum(d > 0 for d in delta), tied=sum(d == 0 for d in delta), worse=sum(d < 0 for d in delta),
                complete_misses_default=sum(r['truth_pixels'] > 0 and r['iu']['actual.equal.crf'][0] == 0 for r in group),
                complete_misses_control=sum(r['truth_pixels'] > 0 and r['iu'][base][0] == 0 for r in group))
        results.setdefault(dataset, {})[frame] = result
    def distribution(values):
        return dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.quantile(values, .95)), sum=float(np.sum(values)))
    timing = {key: distribution([r[key] for r in records]) for key in
        ('io_seconds', 'gate_seconds', 'head_seconds', 'rendering_seconds', 'serialization_seconds', 'total_seconds')}
    timing['CRF_total_seconds'] = distribution([sum(r['crf_seconds'].values()) for r in records])
    timing['CRF_per_field'] = {arm: distribution([r['crf_seconds'][arm] for r in records]) for arm in FIELDS}
    producer = read(out/'producer_COMPLETE.json')
    report = dict(n=200, results=results, primary_frame='cli1024', fixed_default=cfg['fixed_default'],
        matched_control=cfg['matched_control'], paired_cached_timing=timing,
        actual_focus_producer_cost=producer, cost_scope=cfg['cost_scope'],
        observations_per_episode=7, cold_end_to_end_latency_measured=False,
        apd_applied_counts={d: sum(r['apd_applied'] for r in records if r['dataset'] == d) for d in results},
        exposure=cfg['exposure'], seal_sha256=sha(out/'sealed.json'), scored_sha256=sha(out/'scored_episodes.jsonl'),
        edit_order=['add_TP', 'add_FP', 'delete_TP', 'delete_FP'], parameter_search=False, statistical_look_count=1)
    write(out/'report.json', report)
    write(out/'COMPLETE.json', dict(state='COMPLETE', n=200, report_sha256=sha(out/'report.json'),
        seal_sha256=sha(out/'sealed.json'), scored_sha256=sha(out/'scored_episodes.jsonl'),
        primary_miou={d: r['cli1024']['miou'] for d, r in results.items()},
        encoder_forward=0, raw_cache_writes=0, query_GT_in_inference=0))
    write(out/'activity.json', dict(state='COMPLETE', n=200))
    print(json.dumps(dict(state='COMPLETE', primary_miou={d: r['cli1024']['miou'] for d, r in results.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'score'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--assets', type=Path, default=DATA)
    parser.add_argument('--producer', type=Path, default=DATA/'a/reference_focus200_20261010')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out, args.assets, args.producer = args.out.resolve(), args.assets.resolve(), args.producer.resolve()
    assert args.workers == 4, 'Frozen experiment uses four CPU workers and two threads each'
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'experiment.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare(args.out, args.assets, args.producer, args.workers)
        elif args.command == 'run':
            run(args.out)
        else:
            score(args.out)


if __name__ == '__main__':
    main()
