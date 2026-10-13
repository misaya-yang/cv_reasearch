"""Independently score selected complete masks against the joint-count result."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import time
import numpy as np
import torch
import x1012s2_composition_library as lib


def main():
    started = time.monotonic()
    sys.path.insert(0, str(lib.ASSETS / 'a/x1012_R6_query_space/runtime'))
    import lab
    lab.ROOT, lab.ASSETS, lab.DEV = lib.REPO, lib.ASSETS, torch.device('cpu')
    import plan_common as pc
    torch.set_num_threads(1)
    rows = lib.read(lib.ASSETS / 'a/x1012_R1/manifest.json')
    hist = {r['index']: r for r in lib.read(lib.ASSETS / 'a/x1012s2_error_overlap/joint6_1360.json')['records']}
    v1_base = lib.OUT / 'materialized/r00097_r05681'
    v2_base = lib.OUT.parent / 'boolean_network_v2/materialized/fce0fce0fce0fce0_fee0fee0fee0fcc0'
    v1_records = {r['index']: r for r in lib.read(v1_base / 'sealed.json')['records']}
    v2_records = {r['index']: r for r in lib.read(v2_base / 'sealed.json')['records']}
    tables = {r['id']: int(r['truth_table_hex'], 16) for r in lib.read(lib.OUT / 'rules.json')}
    checks, all_records = 0, []
    for b in rows:
        name = pc.key(b)
        h = hist[name]
        natural = {}
        for record, path_key, hash_key, keys in ((v1_records[name], 'prediction_path', 'prediction_sha256', ['r00097', 'r05681']),
                (v2_records[name], 'path', 'sha256', ['fce0fce0fce0fce0', 'fee0fee0fee0fcc0'])):
            assert lib.sha(record[path_key]) == record[hash_key]
            with np.load(record[path_key], allow_pickle=False) as z:
                natural.update({k: pc.unpack(z[k]) for k in keys})
        gt, valid = pc.truth(b)
        row_metrics = {}
        for key, mask in natural.items():
            table = tables[key] if key.startswith('r') else int(key, 16)
            lookup = np.array([(table >> p) & 1 for p in range(64)], np.int64)
            metrics = pc.pixel_metrics(mask, gt, valid)
            intersection = int(lookup @ np.asarray(h['fg_count'], np.int64))
            fp = int(lookup @ np.asarray(h['bg_count'], np.int64))
            gt_area = sum(h['fg_count'])
            expected = {'I': intersection, 'U': gt_area + fp, 'TP': intersection,
                        'FP': fp, 'FN': gt_area - intersection, 'GT': gt_area,
                        'predicted': intersection + fp}
            assert all(metrics[k] == v for k, v in expected.items()), (name, key, metrics, expected)
            checks += len(expected)
            row_metrics[key] = expected
        all_records.append({'index': name, 'dataset': b['dataset'], 'fold': b['row']['fold'],
                            'class_id': b['row']['global_class_id'], 'metrics': row_metrics})
    destination = lib.OUT.parent / 'selected_mask_verification'
    lib.write(destination / 'metrics.json', all_records)
    result = {'state': 'ALL_SELECTED_COMPLETE_MASKS_MATCH_JOINT_COUNT_AGGREGATION',
              'n_cases': len(rows), 'n_rules': 4, 'integer_checks': checks,
              'new_encoder_calls': 0, 'source_sha256': lib.sha(__file__),
              'v1_seal_sha256': lib.sha(v1_base / 'sealed.json'),
              'v2_seal_sha256': lib.sha(v2_base / 'sealed.json'),
              'seconds': time.monotonic() - started}
    lib.write(destination / 'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
