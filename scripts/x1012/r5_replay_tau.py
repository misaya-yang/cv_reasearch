"""Freeze the development-selected scalar and replay sealed R5 natural fields.

No features, RGB, reference annotation, or query annotation are opened. The sole
primary is VP_UN at the selected global tau; rho>.9 keeps the original raw-T
mask. Fixed-.5 UN and the preceding VP regional method are declared controls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get('CV_RESEARCH_ROOT', Path.cwd())).resolve()
ASSETS = REPO.parent / 'cv_data'
PARENT = ASSETS / 'a/x1012_R5_candidate'
DEVELOPMENT = ASSETS / 'a/x1012_R4_global'
OUT = PARENT / 'final_tau0475'
ARMS = ('candidate', 'UN_midrange', 'VP_previous')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def initialize():
    global lab, pc, CONFIG
    runtime = PARENT / 'runtime'
    sys.path.insert(0, str(runtime))
    lab = importlib.import_module('lab')
    lab.ROOT, lab.ASSETS, lab.DEV = REPO, ASSETS, torch.device('cpu')
    pc = importlib.import_module('plan_common')
    torch.set_num_threads(2)
    CONFIG = read(OUT / 'config.json')


def freeze():
    development, completion = read(DEVELOPMENT / 'config.json'), read(DEVELOPMENT / 'completion.json')
    assert completion['state'] == 'COMPLETE'
    assert completion['selected_arm'] == 'UN_tau475' and completion['selected_tau'] == .475
    assert sha(DEVELOPMENT / 'config.json') == completion['config_sha256']
    parent = read(PARENT / 'sealed.json')
    assert parent['state'] == 'ALL_NATURAL_OUTPUTS_SEALED' and parent['n'] == 500
    assert sha(PARENT / 'manifest.json') == parent['manifest_sha256']
    assert sha(PARENT / 'config.json') == parent['config_sha256']
    selection = {'state': 'FINAL_SELECTION_FROZEN_BEFORE_VALIDATION_GT',
                 'criterion': development['selection'], 'selected_arm': completion['selected_arm'],
                 'selected_tau': .475, 'primary': 'candidate', 'pixel_refinement': False,
                 'development_n': completion['n'],
                 'development_worst_domain_delta_pp': completion['selected_worst_domain_delta_pp'],
                 'development_mean_domain_delta_pp': completion['selected_mean_domain_delta_pp'],
                 'development_config_path': str(DEVELOPMENT / 'config.json'),
                 'development_config_sha256': sha(DEVELOPMENT / 'config.json'),
                 'development_completion_path': str(DEVELOPMENT / 'completion.json'),
                 'development_completion_sha256': sha(DEVELOPMENT / 'completion.json')}
    config = {'state': 'FROZEN_BEFORE_FINAL_NATURAL_OUTPUTS', 'n': 500, 'arms': ARMS,
              'primary_arm': 'candidate', 'tau': .475, 'pixel_refinement': False,
              'candidate': 'VP_UN -> bilinear1024 align_corners=False -> minmax -> strict>.475',
              'fallback': 'rho>.9: packed parent UN_identity copied without a changed threshold',
              'UN_midrange': 'packed parent UN_identity copied bitwise; fixed strict>.5, no pixel',
              'VP_previous': 'FP32 onehot region(c_mod) + FP64-bincount region(addition_V+addition_P); bilinear1024,minmax,strict>.5; same rho>.9 raw-T fallback',
              'selection': selection, 'parent_seal_path': str(PARENT / 'sealed.json'),
              'parent_seal_sha256': sha(PARENT / 'sealed.json'),
              'parent_config_sha256': sha(PARENT / 'config.json'),
              'manifest_sha256': sha(PARENT / 'manifest.json'),
              'source_sha256': sha(__file__),
              'runtime_sources': {str(PARENT / 'runtime' / name): sha(PARENT / 'runtime' / name)
                                  for name in ('lab.py', 'plan_common.py')},
              'reference_count': 1, 'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
              'FoRIS_score_consumed': False, 'workers': 2, 'threads_per_worker': 2}
    if (OUT / 'config.json').exists():
        assert read(OUT / 'config.json') == json.loads(json.dumps(config)), 'Frozen replay config changed'
    else:
        (OUT / 'frozen').mkdir(parents=True)
        shutil.copyfile(__file__, OUT / 'frozen/r5_replay_tau.py')
        shutil.copyfile(PARENT / 'manifest.json', OUT / 'manifest.json')
        write(OUT / 'selection.json', selection)
        write(OUT / 'config.json', config)
    print({'state': config['state'], 'tau': config['tau'], 'primary_arm': 'candidate',
           'pixel_refinement': False, 'config_sha256': sha(OUT / 'config.json')}, flush=True)


def previous_field(fields):
    labels = fields['regions64'].astype(np.int64).reshape(-1)
    onehot = F.one_hot(torch.from_numpy(labels), 64).float()
    c_mod = torch.from_numpy(fields['c_mod'].astype(np.float32).reshape(-1))
    regional = (onehot @ ((onehot.T @ c_mod) / onehot.sum(0).clamp_min(1))).numpy().reshape(64, 64)
    correction = (fields['addition_V'] + fields['addition_P']).astype(np.float32).reshape(-1)
    counts = np.bincount(labels, minlength=64)
    sums = np.bincount(labels, weights=correction, minlength=64)
    correction_region = (sums / np.maximum(counts, 1))[labels].astype(np.float32).reshape(64, 64)
    return regional + correction_region


def produce_one(job):
    binding, source = job
    name = pc.key(binding)
    record_path = OUT / 'records' / (name + '.json')
    if record_path.exists():
        record = read(record_path)
        assert sha(record['prediction_path']) == record['prediction_sha256']
        return record
    started = time.monotonic()
    with pc.inference_only(binding):
        for kind in ('prediction', 'field'):
            assert sha(source[kind + '_path']) == source[kind + '_sha256']
        with np.load(source['prediction_path'], allow_pickle=False) as saved:
            fixed = saved['UN_identity'].copy()
        with np.load(source['field_path'], allow_pickle=False) as saved:
            field = saved['VP_UN'].copy()
            rho = float(saved['reference_rho'][0])
            assert np.array_equal(pc.pack(pc.readout(field)), fixed), (name, 'Parent fixed-.5 identity')
            if rho > .9:
                candidate, previous = fixed.copy(), fixed.copy()
            else:
                score = lab.minmax(lab.up(torch.from_numpy(field)))
                candidate = pc.pack((score > CONFIG['tau']).numpy())
                previous = pc.pack(pc.readout(previous_field(saved)))
        prediction_path = OUT / 'predictions' / (name + '.npz')
        assert not prediction_path.exists(), 'Preserve partial replay output'
        np.savez_compressed(prediction_path, candidate=candidate, UN_midrange=fixed, VP_previous=previous)
    record = {'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
              'prediction_path': str(prediction_path), 'prediction_sha256': sha(prediction_path),
              'parent_field_path': source['field_path'], 'parent_field_sha256': source['field_sha256'],
              'parent_prediction_sha256': source['prediction_sha256'], 'reference_rho': rho,
              'raw_T_fallback': rho > .9, 'fixed05_identity_xor': 0,
              'candidate_change_to_fixed05_pixels': int(np.unpackbits(np.bitwise_xor(candidate, fixed)).sum()),
              'seconds': time.monotonic() - started, 'reference_count': 1, 'new_encoder_calls': 0,
              'inference_query_GT_reads': 0, 'pixel_refinement': False}
    write(record_path, record)
    return record


def produce():
    config = read(OUT / 'config.json')
    assert config['source_sha256'] == sha(__file__)
    assert config['parent_seal_sha256'] == sha(PARENT / 'sealed.json')
    assert config['manifest_sha256'] == sha(OUT / 'manifest.json')
    assert all(sha(path) == digest for path, digest in config['runtime_sources'].items())
    selection = config['selection']
    assert sha(selection['development_config_path']) == selection['development_config_sha256']
    assert sha(selection['development_completion_path']) == selection['development_completion_sha256']
    rows = read(OUT / 'manifest.json')
    parent_records = {r['index']: r for r in read(PARENT / 'sealed.json')['records']}
    for name in ('predictions', 'records'):
        (OUT / name).mkdir(exist_ok=True)
    records, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        jobs = ((b, parent_records[str(b['index'])]) for b in rows)
        for i, record in enumerate(pool.map(produce_one, jobs, chunksize=1), 1):
            records.append(record)
            if i % 100 == 0:
                print({'completed': i, 'n': len(rows), 'seconds': round(time.monotonic() - started, 2)}, flush=True)
    seal = {'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(records), 'arms': ARMS, 'primary_arm': 'candidate',
            'records': records, 'config_sha256': sha(OUT / 'config.json'), 'selection_sha256': sha(OUT / 'selection.json'),
            'manifest_sha256': sha(OUT / 'manifest.json'), 'parent_seal_sha256': sha(PARENT / 'sealed.json'),
            'wall_seconds': time.monotonic() - started, 'case_seconds_sum': sum(r['seconds'] for r in records),
            'fixed05_identity_xor': 0, 'fallback_n': sum(r['raw_T_fallback'] for r in records),
            'reference_count': 1, 'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
            'pixel_refinement': False, 'workers': 2, 'threads_per_worker': 2}
    write(OUT / 'sealed.json', seal)
    print({'state': seal['state'], 'n': len(records), 'seconds': seal['wall_seconds'],
           'seal_sha256': sha(OUT / 'sealed.json')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce'))
    globals()[parser.parse_args().mode]()
