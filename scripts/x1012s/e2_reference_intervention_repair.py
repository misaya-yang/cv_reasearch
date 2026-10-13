"""Narrow E2 runtime repair: recover B1's original K64 partition if needed.

The v1 source/config/pilot seal remain unchanged. Some repeated MPS k-means
calls differ by a few tokens. A joint signature of the sealed B1 region-constant
natural fields identifies the original partition when it has exactly 64 unique
rows. No GT, score, parameter choice, new arm, or new model encoding is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import e2_reference_intervention as base

ORIGINAL_COMPUTE = base.compute
REPAIR = base.OUT / 'repair_runtime.json'
FIRST_CASE = 'k_10fd6b2f5286aac82d92'


def bind_runtime():
    cfg = base.check_config()
    original_path = str(Path(base.__file__).resolve())
    wrapper_path = str(Path(__file__).resolve())
    if not REPAIR.exists():
        base.pc.write(REPAIR, {
            'task': 'E2', 'repair': 'recover original B1 K64 from sealed natural field signatures',
            'first_case': FIRST_CASE, 'observed_v1_max_abs': 0.08088278770446777,
            'observed_v1_mask_xor': 1, 'recovered_case_control_max_abs': 0.0,
            'reason': 'Identical Q and frozen B1 region_labels returned two MPS partitions differing at five tokens in consecutive calls.',
            'authorization': 'Root accepted this necessary implementation repair; preserve v1/config/pilot seal and resume same full cohort.',
            'original_source': original_path, 'original_sha256': cfg['source_sha256'][original_path],
            'wrapper_source': wrapper_path, 'wrapper_sha256': base.pc.sha(wrapper_path),
            'original_config_sha256': base.pc.sha(base.OUT / 'config.json'),
            'pilot_seal_sha256': base.pc.sha(base.OUT / 'pilot/sealed.json'),
            'unchanged': ['all 4585 inputs', 'six intervention arms', 'temperature', 'role masses', 'centres', 'IND0 readout', 'scoring'],
            'recovery_requirement': 'exactly 64 joint natural signatures, then original K1 field tolerance and pixel-exact mask checks'})
    repair = base.pc.read(REPAIR)
    assert base.pc.sha(repair['original_source']) == repair['original_sha256']
    assert base.pc.sha(repair['wrapper_source']) == repair['wrapper_sha256']
    assert base.pc.sha(base.OUT / 'config.json') == repair['original_config_sha256']
    assert base.pc.sha(base.OUT / 'pilot/sealed.json') == repair['pilot_seal_sha256']


def recover_partition(binding, sources, error):
    key = base.pc.key(binding)
    root = base.OUT / 'recovered_labels'
    root.mkdir(exist_ok=True)
    record_path = root / (key + '.json')
    if record_path.exists():
        record = base.pc.read(record_path)
        assert base.pc.sha(record['path']) == record['sha256']
        assert record['B1_field_sha256'] == sources['B1']['field_sha256']
        return record
    with base.b1.legal_inference(binding, [r['view']['mask_path'] for r in binding['references'][:3]]):
        old = sources['B1']
        assert base.pc.sha(old['field_path']) == old['field_sha256']
        with np.load(old['field_path'], allow_pickle=False) as fields:
            names = [n for n in fields.files if n.endswith('/IND0') or n.endswith('/INDN')]
            signatures, labels = np.unique(np.stack([fields[n] for n in names], axis=1),
                                           axis=0, return_inverse=True)
        assert len(signatures) == 64, ('Cannot identify 64 original B1 regions', key, len(signatures))
        path = root / (key + '.npz')
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, regions64=labels.astype(np.uint16))
        temporary.replace(path)
        record = {'index': key, 'path': str(path), 'sha256': base.pc.sha(path), 'regions': 64,
                  'origin': 'sealed B1 joint natural constant-field signatures',
                  'B1_field_path': old['field_path'], 'B1_field_sha256': old['field_sha256'],
                  'signature_fields': names, 'P0_source_sha256': sources['labels']['sha256'],
                  'v1_control_failure': repr(error), 'query_GT_reads': 0, 'new_encoder_calls': 0}
        base.pc.write(record_path, record)
        return record


def compute(binding, sources):
    start = time.monotonic()
    key = base.pc.key(binding)
    recovered = base.OUT / 'recovered_labels' / (key + '.json')
    if recovered.exists():
        replacement = recover_partition(binding, sources, 'previously recorded')
    else:
        try:
            return ORIGINAL_COMPUTE(binding, sources)
        except AssertionError as error:
            detail = error.args[0] if error.args else None
            if not isinstance(detail, tuple) or detail[0] not in ('K1 field reproduction', 'K1 mask reproduction'):
                raise
            replacement = recover_partition(binding, sources, error)
    fixed = dict(sources, labels=replacement)
    payload, receipt = ORIGINAL_COMPUTE(binding, fixed)
    receipt['seconds'] = time.monotonic() - start
    receipt['diagnostics'].update(K64_recovered=True,
                                  K64_recovery_origin=replacement['origin'],
                                  K64_recovery_record=str(recovered))
    payload['receipt_json'] = np.frombuffer(json.dumps(receipt, allow_nan=False).encode(), dtype=np.uint8)
    print({'task': 'E2-K64-repair', 'index': key,
           'K1_max_abs': receipt['diagnostics']['K1_field_max_abs'],
           'K1_mask_xor': receipt['diagnostics']['K1_mask_xor_pixels']}, flush=True)
    return payload, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--cohort', choices=('full',), default='full')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    bind_runtime()
    if args.mode == 'produce':
        base.compute = compute
        base.produce(args.cohort)
        path = base.OUT / args.cohort / 'sealed.json'
        seal = base.pc.read(path)
        seal['runtime_repair_sha256'] = base.pc.sha(REPAIR)
        seal['K64_recovered_n'] = sum(r['diagnostics'].get('K64_recovered', False) for r in seal['records'])
        base.pc.write(path, seal)
    else:
        seal = base.pc.read(base.OUT / args.cohort / 'sealed.json')
        assert seal['runtime_repair_sha256'] == base.pc.sha(REPAIR)
        base.score(args.cohort, args.workers)
