"""Export exact GT-labelled 6-bit counts for existing development predictions.

This is an offline sufficient-statistic interface, not a selector or a new
prediction rule. Original exposed 1360 only; no R5 T is generated.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
import x1012s2_error_overlap as core

ROOT, OUT, pc = core.ROOT, core.OUT, core.pc
ORDER = ('B', 'M', 'U', 'T', 'I', 'J')
PATHS = {'B': 'FoRIS', 'M': 'MEAN', 'U': 'R1_VP_UN_midrange_05',
         'T': 'original_independent_T', 'I': 'INSID3_bilinear', 'J': 'INSID3_CRF640'}
CONTROLS = core.ASSETS/'a/x1012_B4/seed150/controls'
INS = core.INS/'development1360'


def main():
    read, write, sha = core.read, core.write, core.sha
    seal_paths = {'B4_complete_controls': CONTROLS/'sealed.json',
                  'R1': core.R1/'sealed.json', 'INSID3': INS/'sealed.json'}
    seals = {k: {'path': str(p), 'sha256': sha(p)} for k, p in seal_paths.items()}
    dependencies = [core.R1/'manifest.json', core.R1/'report.json', INS/'episodes.json', *seal_paths.values()]
    spec = {'state': 'FIXED_EXISTING_MASK_CONTINGENCY_EXPORT', 'n': 1360,
            'bit_order_LSB_first': ORDER, 'bit_weights': [1, 2, 4, 8, 16, 32],
            'path_meanings': PATHS, 'U_readout': 'original R1 VP_UN .5, not .475',
            'GT_role': 'already exposed offline count labels only; rule search/selection is a separate caller',
            'ignore_policy': 'pc.truth original valid mask; only valid pixels contribute',
            'new_predictions': 0, 'fitted_models': 0, 'new_encoder_calls': 0,
            'resources': {'CPU_workers': 1, 'CPU_threads': 1, 'MPS': False},
            'source_sha256': sha(__file__), 'helper_script_sha256': sha(core.__file__),
            'input_sha256': {str(p): sha(p) for p in dependencies}, 'prediction_seals': seals}
    config_path = OUT/'joint6_config.json'
    if config_path.exists():
        import json
        assert read(config_path) == json.loads(json.dumps(spec)), 'Joint6 source/config drift'
    else:
        write(config_path, spec)
        (OUT/'frozen').mkdir(exist_ok=True)
        shutil.copyfile(__file__, OUT/'frozen/x1012s2_joint6.py')
    rows = read(core.R1/'manifest.json')
    controls = {r['index']: r for r in read(seal_paths['B4_complete_controls'])['records']}
    un = {r['index']: r for r in read(seal_paths['R1'])['records']}
    ins = {r['index']: r for r in read(seal_paths['INSID3'])['records']}
    old = {r['index']: r for r in read(core.R1/'report.json')['records']}
    ins_scores = {r['index']: r for r in read(INS/'episodes.json')}
    began, records = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        index = pc.key(b)
        c, u, n = controls[index], un[index], ins[index]
        source = {
            'B': {'path': c['path'], 'mask_key': 'FoRIS', 'file_sha256': c['sha256'], 'seal_id': 'B4_complete_controls'},
            'M': {'path': c['path'], 'mask_key': 'MEAN', 'file_sha256': c['sha256'], 'seal_id': 'B4_complete_controls'},
            'U': {'path': u['prediction_path'], 'mask_key': 'VP_UN', 'file_sha256': u['prediction_sha256'], 'seal_id': 'R1'},
            'T': {'path': u['prediction_path'], 'mask_key': 'T', 'file_sha256': u['prediction_sha256'], 'seal_id': 'R1'},
            'I': {'path': n['prediction_path'], 'mask_key': 'cli/insid3.bilinear', 'file_sha256': n['prediction_sha256'], 'seal_id': 'INSID3'},
            'J': {'path': n['prediction_path'], 'mask_key': 'cli/insid3.crf640', 'file_sha256': n['prediction_sha256'], 'seal_id': 'INSID3'}}
        by_path = defaultdict(list)
        for j, arm in enumerate(ORDER):
            by_path[source[arm]['path']].append((j, arm))
        states = np.zeros((1024, 1024), dtype=np.uint8)
        for path, entries in by_path.items():
            assert sha(path) == source[entries[0][1]]['file_sha256']
            with np.load(path, allow_pickle=False) as z:
                for j, arm in entries:
                    states |= pc.unpack(z[source[arm]['mask_key']]).astype(np.uint8) << j
        gt, valid = pc.truth(b)
        code = states[valid].astype(np.uint16) + 64*gt[valid].astype(np.uint16)
        counts = np.bincount(code, minlength=128).reshape(2, 64)
        metrics = core.metrics_from_counts(counts, ORDER)
        expected = {'B': old[index]['metrics']['FoRIS'], 'M': old[index]['metrics']['MEAN'],
                    'U': old[index]['metrics']['VP_UN'], 'T': old[index]['metrics']['T'],
                    'I': ins_scores[index]['metrics']['insid3.bilinear'],
                    'J': ins_scores[index]['metrics']['insid3.crf640']}
        assert all(metrics[a][k] == expected[a][k] for a in ORDER for k in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted')), index
        records.append({'index': index, 'dataset': b['dataset'], 'fold': b['row']['fold'],
                        'class_id': b['row']['global_class_id'],
                        'query_photo': str(core.ASSETS/b['row']['query_path']),
                        'reference_photo': str(core.ASSETS/b['row']['reference_path']),
                        'query_mask_path': b['row']['query_mask_path'], 'query_crop': b['row'].get('query_crop'),
                        'valid_pixel_n': int(counts.sum()), 'gt_fg_pixel_n': int(counts[1].sum()),
                        'bg_count': counts[0].tolist(), 'fg_count': counts[1].tolist(),
                        'prediction_sources': source})
        if i <= 2 or i % 200 == 0:
            print({'phase': 'joint6-existing-mask-counts', 'completed': i, 'n': 1360,
                   'seconds': round(time.monotonic()-began, 1)}, flush=True)
    destination = OUT/'joint6_1360.json'
    result = {'schema': 'existing_binary_mask_joint6_v1', 'n': len(records),
              'bit_order_LSB_first': ORDER, 'bit_weights': [1, 2, 4, 8, 16, 32],
              'path_meanings': PATHS, 'state_code': 'B + 2*M + 4*U + 8*T + 16*I + 32*J',
              'bg_count': '64 counts at GT=0, only valid pixels',
              'fg_count': '64 counts at GT=1, only valid pixels',
              'prediction_seals': seals, 'records': records,
              'all_existing_path_integer_metrics_reproduced': True,
              'manifest_path': str(core.R1/'manifest.json'), 'manifest_sha256': sha(core.R1/'manifest.json'),
              'config_path': str(config_path), 'config_sha256': sha(config_path),
              'seconds': time.monotonic()-began, 'new_predictions': 0, 'new_encoder_calls': 0}
    write(destination, result)
    receipt = {'state': 'COMPLETE', 'n': len(records), 'path': str(destination), 'sha256': sha(destination),
               'bytes': destination.stat().st_size, 'config_sha256': sha(config_path),
               'bit_order_LSB_first': ORDER, 'all_existing_path_integer_metrics_reproduced': True,
               'seconds': time.monotonic()-began}
    write(OUT/'joint6_completion.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
