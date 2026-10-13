#!/usr/bin/env python3
"""One fixed FoRIS matching intervention; encoder-free, sealed five-domain output."""
from __future__ import annotations
import argparse
import builtins
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time
import types

for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[option] = '2'
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/foris_axis_contradictions_five_20261012'
PRIOR = ASSETS / 'a/task_semantic_hierarchy_five1000_20261012'
BINDINGS = ASSETS / 'a/consensus_reference_calibration6380_v1_20261011/source_bindings.json'
ATLAS = ROOT / 'evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl'
CASE_SPEC = ROOT / 'evidence/local/five_dataset_case_mechanisms_20261011/diagnostic_case_selection.frozen.json'
WITNESSES = ROOT / 'evidence/local/strong600_witness_provenance3_20261011/report.json'
SPEC = ROOT / 'docs/research/FORIS_AXIS_CONTRADICTIONS_FIVE_20261012.md'
DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim')
ARMS = ('source.pre', 'source.crf', 'axis.pre', 'axis.crf')
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(p)


def prepare():
    if (OUT / 'config.json').exists():
        return read(OUT / 'config.json')
    if OUT.exists():
        raise FileExistsError('Unconfigured output directory exists; preserve it.')
    tasks = read(PRIOR / 'manifest.json')
    assert len(tasks) == 1000 and Counter(t['dataset'] for t in tasks) == Counter({d: 200 for d in DOMAINS})
    for t in tasks:
        t['primary_cohort'] = True
        t['contradictions'] = []
    by_episode = {t['row']['episode_id']: t for t in tasks}
    bindings = {b['index']: b for b in read(BINDINGS)}
    atlas = {r['binding_index']: r for r in map(json.loads, ATLAS.read_text().splitlines())}
    wanted = [(r['episode_id'], '/'.join(r.get('roles', [r['role']]))) for r in read(CASE_SPEC)['cases']]
    wanted += [(r['case']['episode_id'], r['case']['role']) for r in read(WITNESSES)['records']]
    wanted += [('dev_s1/lvis/0/30', 'true_local_observation_rejected_by_whole_context')]
    for episode, role in wanted:
        if episode not in by_episode:
            candidates = [a for a in atlas.values() if a['episode_id'] == episode]
            assert len(candidates) == 1, (episode, len(candidates))
            item = candidates[0]
            b = bindings[item['binding_index']]
            t = {'id': f"case_{item['binding_index']:06d}", 'dataset': item['dataset'],
                 'atlas_index': item['binding_index'], 'row': b['row'],
                 'reference_mask_sha256': sha(b['row']['reference_mask_path']),
                 'raw': {'reference': b['fit']['reference_raw'], 'query': b['fit']['query_raw_identity']},
                 'controls': {k: item['prediction_sources'][k] for k in ('FoRIS', 'MEAN', 'R0', 'N')},
                 'foris_field': item['small_field_sources']['score64'],
                 'primary_cohort': False, 'contradictions': []}
            tasks.append(t)
            by_episode[episode] = t
        by_episode[episode]['contradictions'].append(role)
    OUT.mkdir()
    for name in ('predictions', 'fields', 'records', 'frozen'):
        (OUT / name).mkdir()
    source_paths = [Path(__file__), SPEC, ROOT / 'src/ics/foris.py', ROOT / 'src/ics/official_data.py',
                    ROOT / 'src/ics/native_basis.py', ROOT / 'src/ics/m4_crf_cached.py',
                    ASSETS / 'third_party/foris_official/models/foris.py',
                    ASSETS / 'third_party/foris_official/utils/data.py',
                    ASSETS / 'third_party/foris_official/utils/refinement.py',
                    ASSETS / 'third_party/foris_official/utils/clustering.py',
                    ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so',
                    ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
    for i, p in enumerate(source_paths):
        assert p.is_file(), p
        if p.suffix in ('.py', '.md'):
            shutil.copyfile(p, OUT / 'frozen' / f'{i:02d}_{p.name}')
    write(OUT / 'manifest.json', tasks)
    shutil.copyfile(PRIOR / 'expected_classes.json', OUT / 'expected_classes.json')
    config = {'state': 'FROZEN', 'primary_n': 1000, 'total_n': len(tasks), 'domains': list(DOMAINS),
              'arms': list(ARMS), 'primary_endpoint': 'original/axis.crf minus original/source.crf',
              'intervention': 'Only reference normalization dim=2 -> dim=1 in _locate_candidates',
              'source_sha256': {str(p): sha(p) for p in source_paths},
              'input_sources_sha256': {str(p): sha(p) for p in (PRIOR / 'manifest.json', BINDINGS, ATLAS, CASE_SPEC, WITNESSES)},
              'manifest_sha256': sha(OUT / 'manifest.json'),
              'expected_classes_sha256': sha(OUT / 'expected_classes.json'),
              'new_encoder_calls': 0, 'new_views': 0, 'training': False,
              'device': 'CPU', 'workers': 2, 'threads_per_worker': 2,
              'query_gt_in_inference': False, 'exposure': 'Previously exposed development tasks and fixed explanatory contradictions',
              'no_domain_or_case_parameters': True, 'frozen_before_new_query_gt': True}
    write(OUT / 'config.json', config)
    print(json.dumps({'state': 'PREPARED', 'primary': 1000, 'total': len(tasks), 'contradiction_tasks': sum(bool(t['contradictions']) for t in tasks)}), flush=True)
    return config


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def initialize():
    global NP, TORCH, F, HOST, ADAPTER, ORIGINAL, CORRECTED, RUN, LOAD, ARRAY_HASH, TENSOR_HASH
    import numpy as np
    import torch
    import torch.nn.functional as f
    from ics.official_data import load_inputs, array_hash
    from ics.native_basis import reuse_native_basis
    from ics.foris import run_foris
    from ics.m4_crf_cached import enable_on_crf
    from raw_feature_cache import tensor_hash
    config = read(OUT / 'config.json')
    assert all(sha(p) == h for p, h in config['source_sha256'].items())
    torch.set_num_threads(2)
    torch.manual_seed(0)
    native = load_extension('ics_permutohedral_m4', ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so')
    prepared = load_extension('ics_permutohedral_m4_prepared_v1', ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')

    class PermutohedralLayer(torch.nn.Module):
        def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
            super().__init__()
            self.bilateral = bilateral
            self.parameters_ = (theta_alpha, theta_beta, theta_gamma)

        def forward(self, x, image):
            values = x.detach().float().permute(0, 2, 3, 1).contiguous()
            colours = image.detach().float().permute(0, 2, 3, 1).contiguous()
            return native.forward(values, colours, self.bilateral, *self.parameters_).permute(0, 3, 1, 2).contiguous()

    module = types.ModuleType('PermutohedralFiltering')
    module.PermutohedralLayer = PermutohedralLayer
    sys.modules['PermutohedralFiltering'] = module
    sys.path[:0] = [str(ASSETS / 'third_party/foris_official'), str(ASSETS / 'third_party/crf_source/src')]
    from models.foris import FoRIS

    class Replay(torch.nn.Module):
        encoder = None
        encoder_calls = 0

        def bind(self, task):
            self.binding = {b['input_sha256']: b for b in task['raw'].values()}
            self.memo = {}

        def get_intermediate_layers(self, inputs, n=1, reshape=True):
            assert n == 1 and reshape and inputs.device.type == 'cpu' and inputs.dtype == torch.float32
            maps = []
            for value in inputs.numpy():
                key = tensor_hash(value)
                if key not in self.memo:
                    b = self.binding[key]
                    assert sha(b['entry_path']) == b['entry_sha256'] and sha(b['payload_path']) == b['payload_sha256']
                    info = read(b['entry_path'])
                    assert info['input_tensor_hash'] == key
                    with np.load(b['payload_path'], allow_pickle=False) as z:
                        raw = z['O/24'].copy()
                    assert raw.shape == (4096, 1024) and raw.dtype == np.float32
                    assert tensor_hash(raw) == b['tensor_sha256'] and np.isfinite(raw).all()
                    self.memo[key] = torch.from_numpy(raw).reshape(64, 64, 1024).permute(2, 0, 1).contiguous()
                maps.append(self.memo[key])
            return [torch.stack(maps)]

    adapter = Replay()
    with reuse_native_basis(FoRIS, ASSETS / 'native_assets/positional_basis.pt'):
        host = FoRIS(encoder=adapter, image_size=1024, svd_components=500, tau=.6,
                     mask_refiner='crf', resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
    enable_on_crf(host._crf, prepared)
    original = host._locate_candidates
    source = textwrap.dedent(inspect.getsource(original))
    old = 'ref_m = F.normalize(ref_feats[0:1, m], p=2, dim=2)'
    new = 'ref_m = F.normalize(ref_feats[0:1, m], p=2, dim=1)'
    assert source.count(old) == 1
    changed = source.replace(old, new)
    namespace = dict(original.__func__.__globals__)
    exec(compile(changed, '<frozen_axis_only_candidate>', 'exec'), namespace)
    corrected = types.MethodType(namespace['_locate_candidates'], host)
    NP, TORCH, F, HOST, ADAPTER, ORIGINAL, CORRECTED, RUN, LOAD, ARRAY_HASH, TENSOR_HASH = (
        np, torch, f, host, adapter, original, corrected, run_foris, load_inputs, array_hash, tensor_hash)


@contextmanager
def deny_query(task):
    from PIL import Image
    forbidden = Path(task['row']['query_mask_path']).resolve()
    assert forbidden != Path(task['row']['reference_mask_path']).resolve()
    saved = builtins.open, io.open, Image.open

    def wrap(fn):
        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == forbidden:
                raise RuntimeError('Query annotation access denied during inference')
            return fn(path, *args, **kwargs)
        return guarded

    builtins.open, io.open, Image.open = map(wrap, saved)
    try:
        yield
    finally:
        builtins.open, io.open, Image.open = saved


def render(mask, shape):
    return F.interpolate(mask.float()[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0] > .5


def infer(task):
    start = time.monotonic()
    ADAPTER.bind(task)
    arrays, fields, metrics = {}, {}, {}
    with deny_query(task), TORCH.inference_mode():
        reference, mask, query = LOAD(task['row'], ASSETS)
        assert sha(task['row']['reference_mask_path']) == task['reference_mask_sha256']
        for name, fn in (('source', ORIGINAL), ('axis', CORRECTED)):
            captured = {}

            def observe(*args, **kwargs):
                result = fn(*args, **kwargs)
                captured['vote'] = result[1].detach().clone()
                return result

            HOST._locate_candidates = observe
            t0 = time.monotonic()
            final, got, _, _ = RUN(HOST, reference, mask, query)
            metrics[name + '_seconds'] = time.monotonic() - t0
            fields[name + '_score'] = got['score'].numpy().copy()
            fields[name + '_vote'] = captured['vote'].numpy().copy()
            for endpoint, prediction in (('pre', got['pre']), ('crf', final)):
                arrays[f'cli/{name}.{endpoint}'] = NP.packbits(prediction.numpy())
                arrays[f'original/{name}.{endpoint}'] = NP.packbits(render(prediction, tuple(task['row']['query_size_hw'])).numpy())
            del got
        control = task['controls']['FoRIS']
        assert sha(control['path']) == control['sha256']
        with NP.load(control['path'], allow_pickle=False) as saved:
            metrics['archive_mask_xor'] = {frame: int(NP.unpackbits(NP.bitwise_xor(arrays[frame + '/source.crf'], saved[control[frame + '_key']])).sum())
                                           for frame in ('cli', 'original')}
        source_field = task['foris_field']
        assert sha(source_field['path']) == source_field['sha256']
        with NP.load(source_field['path'], allow_pickle=False) as saved:
            metrics['archive_score_max_abs'] = float(NP.abs(fields['source_score'] - saved[source_field['key']]).max())
        metrics['vote_changed_tokens'] = int(NP.count_nonzero(fields['source_vote'] != fields['axis_vote']))
        metrics['axis_source_mask_xor'] = {frame: int(NP.unpackbits(NP.bitwise_xor(arrays[frame + '/source.crf'], arrays[frame + '/axis.crf'])).sum())
                                          for frame in ('cli', 'original')}
    arrays['original_hw'] = NP.array(task['row']['query_size_hw'], dtype=NP.int32)
    prediction = OUT / 'predictions' / (task['id'] + '.npz')
    field = OUT / 'fields' / (task['id'] + '.npz')
    if prediction.exists() or field.exists():
        raise FileExistsError('Preserve existing complete or partial outputs for ' + task['id'])
    NP.savez_compressed(prediction, **arrays)
    NP.savez_compressed(field, **fields)
    record = {'id': task['id'], 'dataset': task['dataset'], 'episode_id': task['row']['episode_id'],
              'primary_cohort': task['primary_cohort'], 'contradictions': task['contradictions'],
              'prediction_path': str(prediction), 'prediction_sha256': sha(prediction),
              'field_path': str(field), 'field_sha256': sha(field), 'metrics': metrics,
              'seconds': time.monotonic() - start, 'new_encoder_calls': ADAPTER.encoder_calls,
              'query_gt_access_attempts': 0}
    write(OUT / 'records' / (task['id'] + '.json'), record)
    ADAPTER.memo.clear()
    return record


def produce(probe=False):
    config = prepare()
    tasks = read(OUT / 'manifest.json')
    if probe:
        tasks = [t for d in DOMAINS for t in [x for x in tasks if x['dataset'] == d and x['primary_cohort']][:2]]
    if (OUT / 'sealed.json').exists():
        print('Already sealed; production is complete.')
        return
    pending = [t for t in tasks if not (OUT / 'records' / (t['id'] + '.json')).exists()]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(infer, t): t['id'] for t in pending}
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if i <= 10 or i % 25 == 0:
                print(json.dumps({'completed_now': i, 'queued': len(pending), 'id': record['id'],
                                  'seconds': round(time.monotonic() - start, 2), **record['metrics']}), flush=True)
    records = [read(OUT / 'records' / (t['id'] + '.json')) for t in tasks]
    summary = {'n': len(tasks), 'wall_seconds': time.monotonic() - start,
               'archive_mismatch_tasks': sum(any(r['metrics']['archive_mask_xor'].values()) for r in records),
               'max_archive_score_abs': max(r['metrics']['archive_score_max_abs'] for r in records),
               'new_encoder_calls': sum(r['new_encoder_calls'] for r in records)}
    if probe:
        write(OUT / 'probe.json', summary)
        print(json.dumps({'PROBE': summary}), flush=True)
    else:
        for r in records:
            assert sha(r['prediction_path']) == r['prediction_sha256'] and sha(r['field_path']) == r['field_sha256']
        write(OUT / 'sealed.json', {'state': 'ALL_PREDICTIONS_SEALED_BEFORE_SCORING',
                                  'config_sha256': sha(OUT / 'config.json'), 'manifest_sha256': config['manifest_sha256'],
                                  'records': records, 'production_summary': summary})
        print(json.dumps({'SEALED': summary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'probe', 'produce'))
    phase = parser.parse_args().phase
    if phase == 'prepare':
        prepare()
    else:
        produce(phase == 'probe')


if __name__ == '__main__':
    main()
