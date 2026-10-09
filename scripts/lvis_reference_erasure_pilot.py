#!/usr/bin/env python3
"""Fixed100 mask-defined reference erasure experiment, prepared before launch.

prepare/probe never construct an encoder. infer alone may encode the two new
reference views on MPS, in a frozen FP32 batch of two when their cache is missing.
Full reference/query are read from their original raw cache, never re-encoded.
score opens query labels only after all100 masks/fields are sealed. There is no
larger cohort, checkpoint option, threshold search or automatic next experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '2')

from raw_feature_cache import RawFeatureCache, canonical_hash, file_hash, tensor_hash
from lvis_atomic_study import return_taps, point, write

ASSETS = REPO.parent / 'cv_data'
ATOMIC = ASSETS / 'a/lvis_atomic1400_20261009'
PILOT = ATOMIC / 'pilot100'
DEFAULT_ROOT = ASSETS / 'a/lvis_reference_erasure100_20261009'
BASELINES = ['foris.crf', 'mean']
NEW_ARMS = {'mean.reference_erasure': 'true',
            'mean.reference_erasure_shift': 'shift',
            'mean.raw_reference_contrast': 'raw'}
ARMS = BASELINES + list(NEW_ARMS)


def read_index(path):
    return {r['episode_id']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def verify_seal(root):
    seal = json.loads((root / 'sealed.json').read_text())
    for filename, key in [('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                          ('inference.jsonl', 'inference_index_sha256')]:
        if file_hash(root / filename) != seal[key]:
            raise ValueError('Parent seal changed: ' + str(root / filename))
    return seal


def source_paths():
    return [Path(__file__).resolve(),
            REPO / 'src/ics/methods/reference_erasure_guide.py',
            REPO / 'src/ics/methods/mean_control.py', REPO / 'src/ics/methods/rcg.py',
            REPO / 'src/ics/data.py', REPO / 'src/ics/representations.py',
            REPO / 'src/ics/native_basis.py', REPO / 'src/ics/official_data.py',
            REPO / 'scripts/raw_feature_cache.py', REPO / 'scripts/cached_dino.py',
            REPO / 'scripts/lvis_atomic_study.py', REPO / 'scripts/run_m4_baselines.py',
            ASSETS / 'third_party/foris_official/models/foris.py',
            ASSETS / 'third_party/foris_official/utils/data.py']


def setup_data():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    sys.path.insert(0, str(ASSETS / 'third_party/foris_official'))
    from utils.data import build_transform
    return build_transform(1024)


def original_profile():
    receipt = json.loads((ASSETS / 'a/lvis_mean200_20261008/run/raw_cache.json').read_text())
    path = Path(receipt['profile_path'])
    profile = json.loads(path.read_text())
    from cached_dino import dino_profile
    current = dino_profile(ASSETS, 'mps')
    if current != profile:
        raise ValueError('Live checkpoint/source/preprocessing/runtime differs from the original raw profile')
    if profile['producer_device'] != 'mps' or profile['storage_dtype'] != 'float32':
        raise ValueError('Expected the original MPS FP32 O24 profile')
    return path, profile


def prepare(root):
    verify_seal(ATOMIC)
    verify_seal(PILOT)
    profile_path, profile = original_profile()
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    if branch != 'codex_m4':
        raise ValueError('This experiment belongs to codex_m4')
    rows = json.loads((PILOT / 'manifest.json').read_text())
    parent_cfg = json.loads((ATOMIC / 'config.json').read_text())
    if len(rows) != 100 or {r['episode_id'] for r in rows} != set(parent_cfg['pilot_ids']):
        raise ValueError('Require exactly the already frozen100')
    index = read_index(ATOMIC / 'inference.jsonl')
    tasks = []
    for row in rows:
        rec = index[row['episode_id']]
        if not rec['pilot'] or not rec['feature_sha256']:
            raise ValueError('Missing the original processed feature input')
        name = rec['filename']
        paths = {'feature': ATOMIC / 'pilot_features' / name,
                 'source_field': ATOMIC / 'fields' / name,
                 'source_prediction': ATOMIC / 'predictions' / name}
        hashes = {'feature': rec['feature_sha256'], 'source_field': rec['field_sha256'],
                  'source_prediction': rec['prediction_sha256']}
        for key, path in paths.items():
            if file_hash(path) != hashes[key]:
                raise ValueError('Frozen input differs: ' + str(path))
        tasks.append(dict(episode_id=row['episode_id'], filename=name,
                          paths={key: str(value) for key, value in paths.items()}, sha256=hashes,
                          apd_applied=rec['apd_applied'], semantic_similarity=rec['semantic_similarity']))
    sources = {str(path): file_hash(path) for path in source_paths()}
    view_profile = dict(profile, counterfactual_builder=dict(
        name='normalized_reference_mask_erasure_and_half_roll_v1',
        source_sha256=sources[str(REPO / 'src/ics/methods/reference_erasure_guide.py')],
        reference_mask='original nearest1024; complete area64 weights',
        erased_value=0., shift_hw=[512, 512], view_order=['true_fg_erasure', 'shifted_mask_erasure'],
        producer_batch=2, encoder_source='same original TimmDINOv3 strict FP32 eval frozen',
        query_encoded=False, full_reference_encoded=False))
    config = dict(state='PREPARED_ONLY_FIXED100', n=100, arms=ARMS,
                  cohort='original exposed first10 per LVIS fold, frozen atomic pilot IDs',
                  branch=branch, preparation_commit=subprocess.check_output(
                      ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  parent_seal_sha256={str(p): file_hash(p / 'sealed.json') for p in (ATOMIC, PILOT)},
                  source_sha256=sources, original_raw_profile_path=str(profile_path),
                  original_raw_profile_sha256=file_hash(profile_path), original_raw_profile=profile,
                  counterfactual_raw_profile=view_profile,
                  counterfactual_raw_profile_id=canonical_hash(view_profile),
                  counterfactual_raw_cache=str(root / 'raw_cache'),
                  native_basis_path=str(ASSETS / 'native_assets/positional_basis.pt'),
                  native_basis_sha256=file_hash(ASSETS / 'native_assets/positional_basis.pt'),
                  image_builder='original build_transform1024; nearest1024 reference mask',
                  query_features='actual parent MEAN q from frozen processed FP16 cache; no query forward',
                  reference_processing='source channel normalize; frozen original APD flag; explicit P_perp; second source normalize; retain FP32',
                  prototype='complete c/(1-c) weighted means; unit full or full-minus-view contrast',
                  empty_reference_roles='reuse parent MEAN guide; explicit receipt',
                  zero_delta='exact raw FG-BG guide fallback, no epsilon-based event selection',
                  readout='actual parent H/A/s; alpha.25,lambda16, locked rank,CG rtol1e-7 atol1e-9 max300; no extra CRF',
                  rendering='bilinear1024 align_corners=False >.5; original from1024 bool using original render',
                  model='same checkpoint; no training or checkpoint options',
                  new_reference_view_batch_size=2, new_views_encoded_only_if_cache_missing=True,
                  full_reference_encoder_calls=0, query_encoder_calls=0,
                  actual_new_view_encoder_calls='recorded per case; missing pair uses one actual MPS B2 forward',
                  query_GT_in_inference=False, parameter_search=False, exposure='continued development on exposed100',
                  scope='Only this100; stopped6000/full/next600 and other datasets have no entrypoint here',
                  intermediate_storage='all guide/unary/solved fields, roles, prototype vectors/norms; raw FP32 O24 retained input-addressed; processed full/deltas reproducible with hashes')
    root.mkdir(parents=True, exist_ok=True)
    existing = root / 'config.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        if old['source_sha256'] == sources and old['parent_seal_sha256'] == config['parent_seal_sha256']:
            validate_config(root)
            return old
        if (root / 'inference.jsonl').exists() or (root / 'sealed.json').exists() or (root / 'raw_cache').exists():
            raise ValueError('Launched configuration/source changed; preserve the original run')
    write(root / 'manifest.json', rows)
    write(root / 'tasks.json', tasks)
    config.update(manifest_sha256=file_hash(root / 'manifest.json'), tasks_sha256=file_hash(root / 'tasks.json'))
    write(existing, config)
    source_root = root / 'source'
    source_root.mkdir(exist_ok=True)
    for path in source_paths():
        # Preserve hierarchy: official utils/data.py must not overwrite ics/data.py.
        relative = path.relative_to(REPO) if path.is_relative_to(REPO) else Path('assets') / path.relative_to(ASSETS)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(root / 'activity.json', dict(state='PREPARED_ONLY', n=100, completed=0,
                                     encoder_calls=0, worker_pids=[]))
    return config


def validate_config(root):
    config = json.loads((root / 'config.json').read_text())
    for path, digest in config['source_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    for path, digest in config['parent_seal_sha256'].items():
        if file_hash(Path(path) / 'sealed.json') != digest:
            raise ValueError('Parent identity changed')
        verify_seal(Path(path))
    for filename in ('manifest.json', 'tasks.json'):
        if file_hash(root / filename) != config[filename.split('.')[0] + '_sha256']:
            raise ValueError('Frozen cohort/input hashes changed')
    if file_hash(config['original_raw_profile_path']) != config['original_raw_profile_sha256']:
        raise ValueError('Original raw profile changed')
    if file_hash(config['native_basis_path']) != config['native_basis_sha256']:
        raise ValueError('Native basis changed')
    original_profile()
    return config


def legal_inputs(row, transform, feature_path):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from ics.official_data import load_inputs
    from ics.methods.reference_erasure_guide import build_reference_views
    reference, reference_mask, query = load_inputs(row, ASSETS)
    full = transform(reference)
    mask = f.interpolate(reference_mask[None, None].float(), (1024, 1024), mode='nearest')[0, 0] > .5
    cov = f.interpolate(mask[None, None].float(), (64, 64), mode='area')[0, 0].numpy()
    with np.load(feature_path) as source:
        old_cov = source['cov'].copy()
    if not np.array_equal(cov, old_cov):
        raise ValueError('Reference view coverage differs from original source')
    views, shifted, info = build_reference_views(full, mask)
    return full, mask, cov, views, shifted, info, query


def probe(root):
    """One fixed real case plus independent label-free algebra; no model."""
    import numpy as np
    import torch
    import torch.nn.functional as f
    from scipy import sparse
    from types import SimpleNamespace
    from ics.methods import reference_erasure_guide as method, mean_control
    from ics.native_basis import load_native_basis
    config = validate_config(root)
    transform = setup_data()
    from models.foris import FoRIS
    rows = json.loads((root / 'manifest.json').read_text())
    task = json.loads((root / 'tasks.json').read_text())[0]
    row = rows[0]
    assert task['episode_id'] == row['episode_id']
    full, mask, cov, views, shifted, view_info, query = legal_inputs(row, transform, task['paths']['feature'])
    profile_path = Path(config['original_raw_profile_path'])
    cache = RawFeatureCache(profile_path.parent.parent, config['original_raw_profile'])
    raw = cache.read(full.numpy(), ('O/24',))['O/24']
    basis, basis_info = load_native_basis(config['native_basis_path'])
    processed = method.process_reference(raw, basis, task['apd_applied'])
    with np.load(task['paths']['feature']) as source:
        q, r, saved_cov = (source[k].copy() for k in ('q', 'r', 'cov'))
    if not np.array_equal(processed.half().numpy(), r):
        raise ValueError('FP32 full reference source processing does not reproduce original stored FP16 R')
    # Check source matrix/reduction identity independently against the actual
    # official unbound function. This does not construct FoRIS or its encoder.
    maps = torch.from_numpy(raw).reshape(64, 64, 1024).permute(2, 0, 1).contiguous()[None, None]
    normalized = f.normalize(maps, p=2, dim=2)
    source_processed = FoRIS._debias_features(SimpleNamespace(positional_basis=basis), normalized) if task['apd_applied'] else normalized
    source_processed = f.normalize(source_processed[0].float(), dim=1)[0].flatten(1).T.contiguous()
    assert torch.equal(processed, source_processed)
    for applied in (False, True):
        actual = FoRIS._debias_features(SimpleNamespace(positional_basis=basis), normalized) if applied else normalized
        actual = f.normalize(actual[0].float(), dim=1)[0].flatten(1).T.contiguous()
        assert torch.equal(method.process_reference(raw, basis, applied), actual)
    with np.load(task['paths']['source_field']) as source:
        score, parent_field = source['score'].copy(), source['mean'].copy()
    with torch.inference_mode(), return_taps(dict(mean=(mean_control.mean_control, ['q', 'r', 's', 'a', 'H', 'guide']))) as taps:
        got, _ = mean_control.predict(q, r, cov, score)
    assert np.array_equal(got, parent_field)
    parent = taps['mean']
    z, y, _ = method.parent_graph_readout(parent['guide'], parent['s'], parent['a'], parent['H'])
    assert np.array_equal(z.reshape(64, 64), parent_field)
    zero = method.build_guides(parent['q'], processed, processed.clone(), processed.clone(), cov, parent['guide'])
    assert np.array_equal(zero['arrays']['guide_true'], zero['arrays']['guide_raw'])
    assert np.array_equal(zero['arrays']['guide_shift'], zero['arrays']['guide_raw'])
    assert zero['diagnostics']['guides']['true']['fallback'] == 'exactly_zero_delta_reuse_raw_contrast_guide'
    # Synthetic weighted algebra with complete coverage, no query labels.
    small_q = f.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]]), dim=1)
    small_r = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=torch.float32)
    small_cov = np.array([[1., .5], [0., .25]], dtype=np.float32)
    small = method.build_guides(small_q, small_r, small_r - 1, small_r - 2, small_cov, np.zeros(4, np.float32))
    c = torch.tensor(small_cov.ravel())
    manual = (small_r * c[:, None]).sum(0) / c.sum() - (small_r * (1 - c[:, None])).sum(0) / (1 - c).sum()
    assert np.array_equal(small['arrays']['contrast_raw'], manual.numpy())
    # A nonzero constant delta cancels under the matched FG/BG contrast and
    # must explicitly fall back; it must not fabricate a guide direction.
    assert np.array_equal(small['arrays']['guide_true'], small['arrays']['guide_raw'])
    empty = method.build_guides(small_q, small_r, small_r, small_r, np.zeros((2, 2), np.float32), np.arange(4, dtype=np.float32))
    assert np.array_equal(empty['arrays']['guide_true'], np.arange(4, dtype=np.float32))
    h = sparse.eye(4, dtype=np.float64).tocsr()
    z_identity, _, _ = method.parent_graph_readout(np.arange(4, dtype=np.float32), np.array([.1, .2, .3, .4], np.float32), np.ones(4, np.float64), h)
    assert np.isfinite(z_identity).all()
    assert torch.equal(views[0][:, ~mask], full[:, ~mask])
    assert torch.count_nonzero(views[0][:, mask]) == 0
    assert torch.equal(views[1][:, ~shifted], full[:, ~shifted])
    assert torch.count_nonzero(views[1][:, shifted]) == 0
    preparation = root.with_name(root.name + '_preparation')
    preparation.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preparation / 'first_case_views.npz', normalized_reference=full.numpy(),
                        original_reference_mask=mask.numpy(), shifted_reference_mask=shifted.numpy(),
                        reference_views=views.numpy(), coverage=cov)
    receipt = dict(state='PREPARATION_CHECKS_PASS_NO_ENCODER', episode_id=row['episode_id'], n_fixed=100,
                   checks={'reference_coverage_bit_exact': True, 'full_reference_FP16_cache_bit_exact': True,
                           'full_reference_FP32_actual_source_processing_bit_exact': True,
                           'actual_source_processing_both_APD_branches_bit_exact': True,
                           'parent_mean_field_bit_exact': True, 'parent_H_readout_bit_exact': True,
                           'zero_delta_reuses_raw_guide_bit_exact': True, 'weighted_means_match_independent_algebra': True,
                           'empty_role_explicit_parent_fallback': True,
                           'view_background_unchanged_bit_exact': True, 'view_erased_pixels_exactly_zero': True,
                           'shifted_mask_preserves_area': True},
                   native_basis=basis_info, view_builder=view_info, config_sha256=file_hash(root / 'config.json'),
                   source_sha256=config['source_sha256'], original_raw_profile_id=cache.profile_id,
                   original_reference_cache_key=cache.key(full.numpy()), original_reference_input_hash=tensor_hash(full.numpy()),
                   reference_view_input_hashes=[tensor_hash(v.numpy()) for v in views],
                   view_artifact_sha256=file_hash(preparation / 'first_case_views.npz'),
                   actual_new_view_encoder_calls=0, full_reference_encoder_calls=0, query_encoder_calls=0,
                   new_candidate_masks=0, new_miou=0, query_GT_opened=False,
                   scope='first fixed real input and synthetic algebra only; launch awaits parent review')
    write(preparation / 'receipt.json', receipt)
    print(json.dumps({'state': receipt['state'], 'receipt': str(preparation / 'receipt.json')}, ensure_ascii=False), flush=True)
    return receipt


def infer(root):
    import numpy as np
    import torch
    from cached_dino import CachedDINO
    from ics.data import TimmDINOv3
    from ics.native_basis import load_native_basis
    from ics.methods import mean_control, reference_erasure_guide as method, rcg
    from run_m4_baselines import render
    config = validate_config(root)
    if (root / 'sealed.json').exists():
        print('All100 already sealed; no restart', flush=True)
        return
    transform = setup_data()
    basis, basis_info = load_native_basis(config['native_basis_path'])
    profile_path = Path(config['original_raw_profile_path'])
    original = RawFeatureCache(profile_path.parent.parent, config['original_raw_profile'])
    counter_cache = RawFeatureCache(root / 'raw_cache', config['counterfactual_raw_profile'])
    adapter = CachedDINO(counter_cache, encoder=None, device='mps')
    rows = json.loads((root / 'manifest.json').read_text())
    tasks = {t['episode_id']: t for t in json.loads((root / 'tasks.json').read_text())}
    for directory in ('fields', 'predictions'):
        (root / directory).mkdir(exist_ok=True)
    done = {}
    if (root / 'inference.jsonl').exists():
        done = read_index(root / 'inference.jsonl')
        for item in done.values():
            for key, folder in [('fields_sha256', 'fields'), ('prediction_sha256', 'predictions')]:
                if file_hash(root / folder / item['filename']) != item[key]:
                    raise ValueError('Resume artifact identity differs')
    started = time.monotonic()
    with (root / 'inference.jsonl').open('a', buffering=1) as ledger:
        for row in rows:
            eid = row['episode_id']
            if eid in done:
                continue
            task = tasks[eid]
            for key, path in task['paths'].items():
                if file_hash(path) != task['sha256'][key]:
                    raise ValueError('Frozen input differs: ' + path)
            began = time.monotonic()
            full, mask, cov, views, shifted, view_info, _query = legal_inputs(row, transform, task['paths']['feature'])
            view_builder_seconds = time.monotonic() - began
            at = time.monotonic()
            raw_full = original.read(full.numpy(), ('O/24',))['O/24']
            original_raw_read_seconds = time.monotonic() - at
            with np.load(task['paths']['feature']) as saved:
                q, r = saved['q'].copy(), saved['r'].copy()
            with np.load(task['paths']['source_field']) as saved:
                score, parent_field = saved['score'].copy(), saved['mean'].copy()
            at = time.monotonic()
            with torch.inference_mode(), return_taps(dict(mean=(mean_control.mean_control, ['q', 'r', 's', 'a', 'H', 'guide', 'y']))) as taps:
                got, parent_info = mean_control.predict(q, r, cov, score)
            parent_replay_seconds = time.monotonic() - at
            if not np.array_equal(got, parent_field):
                raise ValueError('Original MEAN continuous replay differs')
            parent = taps['mean']
            at = time.monotonic()
            processed_full = method.process_reference(raw_full, basis, task['apd_applied'])
            full_reference_processing_seconds = time.monotonic() - at
            if not np.array_equal(processed_full.half().numpy(), r):
                raise ValueError('Full R processing differs from original processed FP16 cache')
            hits = []
            for view in views:
                entry = counter_cache.folder / counter_cache.key(view.numpy()) / 'entry.json'
                if entry.exists():
                    counter_cache.read(view.numpy(), ('O/24',))
                    hits.append(True)
                else:
                    hits.append(False)
            if not all(hits) and adapter.encoder is None:
                if not torch.backends.mps.is_available():
                    raise RuntimeError('Missing counterfactual features require the frozen MPS FP32 producer')
                adapter.encoder = TimmDINOv3(str(ASSETS / 'demo4_cache/models/dinov3-vitl16-timm')).to('mps').float().eval().requires_grad_(False)
            adapter.provenance = dict(episode_id=eid, view_builder=view_info, reference_mask_hash=row['reference_mask_hash'],
                                      full_reference_encoder_calls=0, query_encoder_calls=0,
                                      source_config_sha256=file_hash(root / 'config.json'))
            before_calls = adapter.encoder_calls
            at = time.monotonic()
            raw_views = adapter.raw(views, ('O/24',))['O/24']
            calls = adapter.encoder_calls - before_calls
            encode_seconds = time.monotonic() - at
            if calls and calls != 1:
                raise ValueError('Expected one MPS B2 forward for a missing view pair')
            at = time.monotonic()
            processed_views = [method.process_reference(raw_views[i], basis, task['apd_applied']) for i in range(2)]
            reference_view_processing_seconds = time.monotonic() - at
            at = time.monotonic()
            guides = method.build_guides(parent['q'], processed_full, processed_views[0], processed_views[1], cov, parent['guide'])
            fields = dict(guides['arrays'], source_score=score, source_s=parent['s'].reshape(64, 64),
                          parent_guide=parent['guide'].reshape(64, 64), parent_unary=parent['y'].reshape(64, 64), parent_mean=parent_field,
                          parent_a=parent['a'].reshape(64, 64), parent_H_indptr=parent['H'].indptr,
                          parent_H_indices=parent['H'].indices, parent_H_data=parent['H'].data,
                          parent_H_shape=np.asarray(parent['H'].shape))
            masks = {}
            readouts = {}
            for arm, kind in NEW_ARMS.items():
                z, y, info = method.parent_graph_readout(fields['guide_' + kind], parent['s'], parent['a'], parent['H'])
                fields['field_' + kind] = z.reshape(64, 64)
                fields['unary_' + kind] = y.reshape(64, 64)
                masks[arm] = rcg.mask_from_field(z.reshape(64, 64))
                readouts[arm] = info
            packed = dict(original_hw=np.asarray(row['query_size_hw']))
            with np.load(task['paths']['source_prediction']) as saved:
                for arm in BASELINES:
                    for frame in ('cli', 'original'):
                        packed[frame + '/' + arm] = saved[frame + '/' + arm].copy()
            if not np.array_equal(rcg.mask_from_field(parent_field), rcg.unpack(packed['cli/mean'])):
                raise ValueError('Original MEAN final mask differs')
            for arm, value in masks.items():
                packed['cli/' + arm] = np.packbits(value)
                packed['original/' + arm] = np.packbits(render(value, tuple(row['query_size_hw'])))
            name = task['filename']
            np.savez_compressed(root / 'fields' / name, **fields)
            np.savez_compressed(root / 'predictions' / name, **packed)
            view_entries = []
            for i, view in enumerate(views):
                key = counter_cache.key(view.numpy())
                entry = counter_cache.folder / key / 'entry.json'
                item = json.loads(entry.read_text())
                view_entries.append(dict(view_index=i, cache_key=key, entry_sha256=file_hash(entry),
                                         input_tensor_hash=item['input_tensor_hash'], payload_sha256=item['file_sha256'],
                                         raw_O24_tensor_sha256=item['features']['O/24']['tensor_sha256']))
            record = dict(episode_id=eid, filename=name, fields_sha256=file_hash(root / 'fields' / name),
                          prediction_sha256=file_hash(root / 'predictions' / name),
                          parent_mean_field_bit_exact=True, parent_mean_mask_bit_exact=True,
                          reference_coverage_bit_exact=True, full_reference_FP16_cache_bit_exact=True,
                          apd_applied=task['apd_applied'], apd_policy_recomputed=False,
                          actual_new_reference_view_encoder_calls=calls, actual_new_reference_view_encoder_inputs=2 * calls,
                          views_cached_before=hits, already_cached_view_inputs_reencoded_in_B2=sum(hits) if calls else 0,
                          full_reference_encoder_calls=0, query_encoder_calls=0, query_GT_in_inference=False,
                          view_cache_entries=view_entries, original_reference_cache_key=original.key(full.numpy()),
                          original_reference_raw_FP32_tensor_sha256=tensor_hash(raw_full),
                          processing_hashes={'full_processed_FP32': tensor_hash(processed_full.numpy()),
                                             'delta_true_FP32': tensor_hash(guides['delta_true'].numpy()),
                                             'delta_shift_FP32': tensor_hash(guides['delta_shift'].numpy()),
                                             'parent_q_FP32': tensor_hash(parent['q'].numpy())},
                          view_builder=view_info, role_guides=guides['diagnostics'], parent_graph=parent_info,
                          readouts=readouts, encoder_or_cache_seconds=encode_seconds,
                          view_builder_seconds=view_builder_seconds, original_raw_read_seconds=original_raw_read_seconds,
                          parent_replay_seconds=parent_replay_seconds,
                          full_reference_processing_seconds=full_reference_processing_seconds,
                          reference_view_processing_seconds=reference_view_processing_seconds,
                          postprocessing_seconds=time.monotonic() - at, seconds=time.monotonic() - began)
            ledger.write(json.dumps(record) + '\n')
            done[eid] = record
            write(root / 'activity.json', dict(state='INFERENCE', n=100, completed=len(done), pid=os.getpid(),
                                             elapsed_seconds=time.monotonic() - started,
                                             actual_reference_view_encoder_calls=sum(x['actual_new_reference_view_encoder_calls'] for x in done.values()),
                                             query_encoder_calls=0, full_reference_encoder_calls=0))
            if len(done) % 10 == 0:
                print(json.dumps(dict(completed=len(done), n=100, seconds=time.monotonic() - started)), flush=True)
    if len(done) != 100:
        raise ValueError('Do not seal an incomplete cohort')
    validate_config(root)
    seal = dict(state='ALL_PREDICTIONS_SEALED', n=100, config_sha256=file_hash(root / 'config.json'),
                manifest_sha256=file_hash(root / 'manifest.json'), tasks_sha256=file_hash(root / 'tasks.json'),
                inference_index_sha256=file_hash(root / 'inference.jsonl'),
                query_GT_in_inference=False, query_encoder_calls=0, full_reference_encoder_calls=0,
                actual_new_reference_view_encoder_calls=sum(x['actual_new_reference_view_encoder_calls'] for x in done.values()),
                actual_new_reference_view_encoder_inputs=sum(x['actual_new_reference_view_encoder_inputs'] for x in done.values()),
                all_parent_mean_replays_bit_exact=True, all_reference_coverages_bit_exact=True,
                native_basis=basis_info, counterfactual_raw_profile_id=counter_cache.profile_id)
    write(root / 'sealed.json', seal)
    write(root / 'activity.json', dict(state='INFERENCE_COMPLETE', n=100, completed=100, pid=None,
                                     elapsed_seconds=time.monotonic() - started, **{k: seal[k] for k in
                                     ('query_encoder_calls', 'full_reference_encoder_calls', 'actual_new_reference_view_encoder_calls')}))


def score(root):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.metrics import counts, gross_edits
    from ics.official_data import array_hash
    config = validate_config(root)
    seal = verify_seal(root)
    if seal['state'] != 'ALL_PREDICTIONS_SEALED' or seal['n'] != 100:
        raise ValueError('Query GT requires the completed100 prediction seal')
    setup_data()
    rows = json.loads((root / 'manifest.json').read_text())
    inference = read_index(root / 'inference.jsonl')
    previous = read_index(ATOMIC / 'episode_metrics.jsonl')
    records = []
    for row in rows:
        rec = inference[row['episode_id']]
        path = root / 'predictions' / rec['filename']
        if file_hash(path) != rec['prediction_sha256'] or file_hash(root / 'fields' / rec['filename']) != rec['fields_sha256']:
            raise ValueError('Candidate artifacts changed')
        with Image.open(row['query_mask_path']) as image:
            raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        if array_hash(raw) != row['query_mask_hash']:
            raise ValueError('Frozen evaluation labels changed')
        item = dict(episode_id=row['episode_id'], fold=row['fold'], class_id=row['loader_class_id'],
                    query_photo_id=row['query_photo_id'], frames={})
        with np.load(path) as saved:
            for frame, shape in [('cli', (1024, 1024)), ('original', tuple(row['query_size_hw']))]:
                truth = f.interpolate(torch.from_numpy(raw)[None, None].float(), shape, mode='nearest')[0, 0].numpy() > .5
                masks = {arm: np.unpackbits(saved[frame + '/' + arm], count=int(np.prod(shape))).reshape(shape).astype(bool)
                         for arm in ARMS}
                iu = {arm: counts(mask, truth) for arm, mask in masks.items()}
                for arm in BASELINES:
                    if iu[arm] != previous[row['episode_id']]['frames'][frame]['iu'][arm]:
                        raise ValueError('Reused baseline I/U differs')
                baselines = BASELINES + ['mean.reference_erasure_shift', 'mean.raw_reference_contrast']
                edits = {arm: {base: gross_edits(mask, masks[base], truth) for base in baselines}
                         for arm, mask in masks.items()}
                item['frames'][frame] = dict(iu=iu, edits=edits, truth_pixels=int(truth.sum()))
        records.append(item)
    (root / 'episode_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    report = dict(state='COMPLETE', n=100, frames={}, identity=seal, exposure=config['exposure'],
                  scope='same exposed100; independent scoring phase, not independent new-sample validation',
                  inference_sealed_before_candidate_labels=True, new_query_photos=0,
                  original_query_features_preserved=True, shifted_erasure_is_not_guaranteed_wrong_foreground=True)
    for frame in ('cli', 'original'):
        flat = [dict(r, **r['frames'][frame]) for r in records]
        points = point(flat, ARMS)
        pairs = {}
        for arm in ARMS:
            for base in BASELINES + ['mean.reference_erasure_shift', 'mean.raw_reference_contrast']:
                deltas = [100 * (r['iu'][arm][0] / max(r['iu'][arm][1], 1) - r['iu'][base][0] / max(r['iu'][base][1], 1))
                          for r in flat]
                pairs[arm + ' vs ' + base] = dict(delta_pp=points[arm] - points[base],
                    cases_up=sum(d > 1e-10 for d in deltas), cases_down=sum(d < -1e-10 for d in deltas),
                    cases_equal=sum(abs(d) <= 1e-10 for d in deltas),
                    edits=np.sum([r['edits'][arm][base] for r in flat], axis=0).tolist(),
                    edit_order=['add_TP', 'add_FP', 'delete_TP', 'delete_FP'])
        report['frames'][frame] = dict(miou=points, pairs=pairs)
    write(root / 'report.json', report)
    write(root / 'activity.json', dict(state='COMPLETE', n=100, completed=100, pid=None,
                                     actual_new_reference_view_encoder_calls=seal['actual_new_reference_view_encoder_calls'],
                                     actual_new_reference_view_encoder_inputs=seal['actual_new_reference_view_encoder_inputs'],
                                     query_encoder_calls=0, full_reference_encoder_calls=0))
    print(json.dumps({frame: report['frames'][frame]['miou'] for frame in ('cli', 'original')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'probe', 'infer', 'score', 'run'])
    parser.add_argument('--out', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--probe', action='store_true', help='prepare then check one fixed real input; never encode')
    args = parser.parse_args()
    if args.probe and args.mode != 'prepare':
        parser.error('--probe applies only to prepare')
    if args.mode == 'prepare':
        prepare(args.out)
        if args.probe:
            setup_data()
            probe(args.out)
    elif args.mode == 'probe':
        setup_data()
        probe(args.out)
    elif args.mode == 'infer':
        infer(args.out)
    elif args.mode == 'score':
        score(args.out)
    else:
        infer(args.out)
        score(args.out)


if __name__ == '__main__':
    main()
