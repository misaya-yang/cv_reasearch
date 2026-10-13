"""B2: complete 1300 real-window independent controls, with two scoring phases.

produce never reads GT. score uses only the frozen P0 explore intersection.
score-after-confirm requires a completed C1 report and labels the other 192
LVIS tasks as confirmation-after diagnostics, excluded from component selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
import b1_kshot as core

OUT = lab.ASSETS / 'a/x1012_B2'
P0 = lab.ASSETS / 'a/x1012_P0'
WINDOW_ROOT = lab.ROOT / 'evidence/local/nine_window_execution_20261011'
DOMAINS = ('lvis', 'paco_part', 'deepglobe_road')
WINDOWS = [[x, y, x + 512, y + 512] for y in (0, 256, 512) for x in (0, 256, 512)]
ARMS = ('FoRIS', 'whole_T', 'window_T', 'gated_T', 'whole_IND0', 'window_IND0', 'gated_IND0')


def cover_canvas():
    cover = np.zeros((1024, 1024), np.uint8)
    for x0, y0, x1, y1 in WINDOWS:
        cover[y0:y1, x0:x1] += 1
    assert set(np.unique(cover)) == {1, 2, 4}
    return cover


def canvas_mask(field):
    return (lab.minmax(torch.from_numpy(field)) > .5).numpy()


def window_metadata(document, key):
    bound = document['entries'].get(key)
    if bound is None or not bound['available']:
        return None
    return {'entry_path': str(Path(document['profile_path']).parent / key / 'entry.json'),
            'entry_sha256': bound['entry_sha256'], 'payload_path': bound['payload'],
            'payload_sha256': bound['payload_sha256'], 'tensor_sha256': bound['branch']['tensor_sha256']}


def infer(binding, bound, document, block, coverage):
    """Same full-R and readout on whole and real windows; no component masks."""
    counters = {}
    with core.legal_inference(binding):
        y = lab.coverage64(lab.mask1024(binding['row']['reference_mask_path'])).flatten()
        area = float(y.mean())
        rho = float((y * (y < .9)).sum() / y.sum().clamp_min(1e-9))
        gate = rho > .5 or area < .02
        r = lab.debias(core.read_raw(binding['fit']['reference_raw'], counters))
        q = lab.debias(core.read_raw(binding['fit']['query_raw_identity'], counters))
        whole = core.independent_fields([r], q, [y], block)
        predictions = {'whole_T': pc.pack(pc.readout(whole['c'])),
                       'whole_IND0': pc.pack(pc.readout(whole['IND0'])),
                       'FoRIS': pc.pack(lab.foris_cli(binding).numpy())}
        unavailable = []
        if bound is None:
            unavailable = list(range(9))
        else:
            assert bound['role_keys']['reference_full'] == binding['fit']['reference_raw']['key']
            assert bound['role_keys']['query_full'] == binding['fit']['query_raw_identity']['key']
            for name in ('reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash', 'reference_crop', 'query_crop', 'query_size_hw'):
                assert bound['identity'].get(name) == binding['row'].get(name), ('Window identity mismatch', name)
            unavailable = [j for j in range(9) if window_metadata(document, bound['role_keys']['window_' + str(j)]) is None]
        if unavailable:
            for method in ('T', 'IND0'):
                predictions['window_' + method] = predictions['whole_' + method].copy()
        else:
            stitched = {'T': np.zeros((1024, 1024), np.float32), 'IND0': np.zeros((1024, 1024), np.float32)}
            for j, (x0, y0, x1, y1) in enumerate(WINDOWS):
                raw = core.read_raw(window_metadata(document, bound['role_keys']['window_' + str(j)]), counters)
                window_q = lab.debias(raw)
                local = core.independent_fields([r], window_q, [y], block)
                for method, key in (('T', 'c'), ('IND0', 'IND0')):
                    # Continuous probabilities/region fields, without per-window minmax.
                    view = lab.up(local[key].reshape(64, 64), size=512).numpy()
                    stitched[method][y0:y1, x0:x1] += view
                del raw, window_q, local, view
            for method in ('T', 'IND0'):
                stitched[method] /= coverage
                predictions['window_' + method] = pc.pack(canvas_mask(stitched[method]))
            del stitched
        for method in ('T', 'IND0'):
            predictions['gated_' + method] = predictions[('window_' if gate else 'whole_') + method].copy()
    return predictions, {'reference_area_fraction': area, 'rho': rho, 'gate_triggered': gate,
                         'missing_window_indices': unavailable, 'window_available': not unavailable,
                         'new_encoder_calls': 0, 'query_GT_reads': 0, **counters}


def prepare(args):
    full = [b for b in lab.bindings() + lab.ext_bindings() if b['dataset'] in DOMAINS]
    assert Counter(b['dataset'] for b in full) == {'lvis': 1400, 'paco_part': 600, 'deepglobe_road': 100}
    receipt = pc.read(WINDOW_ROOT / 'input_binding_receipt.json')
    assert receipt['window_boxes'] == WINDOWS and not receipt['missing_cache_requests']
    bindings_path = WINDOW_ROOT / 'input_bindings.jsonl'
    assert pc.sha(bindings_path) == receipt['input_bindings_sha256']
    bound_rows = [json.loads(line) for line in bindings_path.read_text().splitlines() if line]
    by_query = {}
    by_pair = {}
    for b in bound_rows:
        by_pair[b['episode_id']] = b
        key = b['role_keys']['query_full']
        if key in by_query:
            assert all(by_query[key]['role_keys']['window_' + str(j)] == b['role_keys']['window_' + str(j)] for j in range(9))
        by_query[key] = b
    selected, missing = [], []
    for b in full:
        query_key = b['fit']['query_raw_identity']['key']
        bound = by_query.get(query_key)
        if bound is None:
            missing.append({'index': pc.key(b), 'dataset': b['dataset'], 'episode_id': b['row']['episode_id'],
                            'query_raw_key': query_key, 'fallback': 'whole independent readout; no new encoding'})
            continue
        # The existing 1300 controls also bind the same reference view exactly.
        exact = by_pair[b['row']['episode_id']]
        assert exact['role_keys']['reference_full'] == b['fit']['reference_raw']['key']
        assert exact['role_keys']['query_full'] == query_key
        selected.append(dict(b, B2_window_binding=exact))
    assert Counter(b['dataset'] for b in selected) == {'lvis': 600, 'paco_part': 600, 'deepglobe_road': 100}
    explore = pc.read(P0 / 'explore.json')
    explore_ids = {(b['dataset'], b['row']['episode_id']) for b in explore}
    for b in selected:
        b['B2_score_phase'] = 'explore' if (b['dataset'], b['row']['episode_id']) in explore_ids else 'after_C1_diagnostic'
    metadata_path = WINDOW_ROOT / 'cache_metadata.json'
    assert pc.sha(metadata_path) == receipt['cache_metadata_sha256']
    pc.write(OUT / 'manifest.json', selected)
    pc.write(OUT / 'cache_boundary.json', {'state': 'COMPLETE_EXISTING_WINDOW_BINDING', 'full_task_n': len(full),
             'window_task_n': len(selected), 'missing_window_task_n': len(missing), 'missing_window_tasks': missing,
             'per_domain': {d: {'full_n': sum(b['dataset'] == d for b in full),
                              'window_n': sum(b['dataset'] == d for b in selected),
                              'missing_n': sum(b['dataset'] == d for b in missing),
                              'explore_score_n': sum(b['dataset'] == d and b['B2_score_phase'] == 'explore' for b in selected),
                              'after_C1_diagnostic_n': sum(b['dataset'] == d and b['B2_score_phase'] != 'explore' for b in selected)}
                            for d in DOMAINS},
             'cache_scope': 'Missing 800 LVIS tasks use whole fallback; they are outside this paired window-control cohort.',
             'observation_cost': 'Nine extra cached DINO observations per available query; cache replay is not free online inference.'})
    pc.write(OUT / 'config.json', {'state': 'FROZEN', 'task': 'B2', 'n': len(selected), 'cpu_threads': 4,
             'query_block': args.query_block, 'arms': ARMS, 'windows': WINDOWS,
             'gate': 'rho > .5 OR reference_foreground_fraction < .02; same rule in all domains',
             'reference': 'exact reference_full, never old component/local-R crop',
             'readouts': 'T token coverage transfer and IND0 K64 per-view query-region means; same whole/window rule',
             'composition': 'continuous64->512 bilinear align_corners=False; translate box; divide fixed cover1/2/4; one final minmax>.5',
             'missing_rule': 'whole fallback for an entirely unavailable or incomplete set of nine raw windows',
             'score_phases': dict(Counter(b['B2_score_phase'] for b in selected)),
             'input_bindings_sha256': pc.sha(bindings_path), 'cache_metadata_sha256': pc.sha(metadata_path),
             'explore_membership_source_sha256': pc.sha(P0 / 'explore.json'),
             'source_sha256': core.source_hashes(Path(__file__)),
             'new_encoder_calls': 0, 'query_GT_reads': 0})


def produce(args):
    pc.setup('cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'config.json').exists():
        prepare(args)
    config = pc.read(OUT / 'config.json')
    core.check_sources(config)
    if (OUT / 'sealed.json').exists():
        core.verify_seal(OUT, 'ALL_1300_WINDOW_PREDICTIONS_SEALED')
        print({'task': 'B2', 'state': 'ALREADY_SEALED', 'n': config['n']}, flush=True)
        return
    metadata_path = WINDOW_ROOT / 'cache_metadata.json'
    assert pc.sha(metadata_path) == config['cache_metadata_sha256']
    document = pc.read(metadata_path)
    profile_path = Path(document['profile_path'])
    assert pc.sha(profile_path) == document['profile_sha256']
    rows = pc.read(OUT / 'manifest.json')
    for name in ('predictions', 'records'):
        (OUT / name).mkdir(exist_ok=True)
    cover = cover_canvas()
    started, cpu = time.monotonic(), time.process_time()
    receipts = []
    for i, b in enumerate(rows, 1):
        name = pc.key(b)
        receipt_path = OUT / 'records' / (name + '.json')
        if receipt_path.exists():
            r = pc.read(receipt_path)
            assert pc.sha(r['prediction_path']) == r['prediction_sha256']
            receipts.append(r)
            continue
        path = OUT / 'predictions' / (name + '.npz')
        assert not path.exists(), 'Preserve partial B2 output'
        tick = time.monotonic()
        predictions, detail = infer(b, b['B2_window_binding'], document, config['query_block'], cover)
        np.savez_compressed(path, **predictions)
        r = {'index': name, 'dataset': b['dataset'], 'episode_id': b['row']['episode_id'],
             'prediction_path': str(path), 'prediction_sha256': pc.sha(path),
             'score_phase': b['B2_score_phase'], 'seconds': time.monotonic() - tick, **detail}
        pc.write(receipt_path, r)
        receipts.append(r)
        del predictions
        lab.free()
        if i <= 3 or i % 50 == 0:
            print({'task': 'B2', 'completed': i, 'n': len(rows), 'seconds': round(time.monotonic() - started, 1)}, flush=True)
    assert len(receipts) == 1300
    pc.write(OUT / 'sealed.json', {'state': 'ALL_1300_WINDOW_PREDICTIONS_SEALED', 'n': len(rows), 'records': receipts,
             'manifest_sha256': pc.sha(OUT / 'manifest.json'), 'config_sha256': pc.sha(OUT / 'config.json'),
             'invocation_wall_seconds': time.monotonic() - started, 'invocation_cpu_seconds': time.process_time() - cpu,
             'episode_seconds_sum': sum(r['seconds'] for r in receipts), 'cpu_threads': 4,
             'raw_requests': sum(r['raw_requests'] for r in receipts),
             'logical_payload_bytes': sum(r['logical_payload_bytes'] for r in receipts),
             'raw_io_seconds': sum(r['raw_io_seconds'] for r in receipts),
             'real_cached_query_window_applications': sum(9 if r['window_available'] else 0 for r in receipts),
             'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'new_encoder_calls': 0, 'query_GT_reads': 0})


def score(after_confirm=False, c1_report=None):
    pc.setup('cpu')
    seal, rows, receipts = core.verify_seal(OUT, 'ALL_1300_WINDOW_PREDICTIONS_SEALED')
    boundary = pc.read(OUT / 'cache_boundary.json')
    c1 = None
    if after_confirm:
        path = Path(c1_report or lab.ASSETS / 'a/x1012_C1/report.json')
        c1 = pc.read(path)
        assert c1.get('state') == 'COMPLETE' and c1.get('task') == 'C1', 'C1 must be scored/final before deferred diagnostics'
        c1 = {'path': str(path), 'sha256': pc.sha(path)}
        assert (OUT / 'component_selection.json').exists(), 'Explore selection must precede post-C1 diagnostics'
    chosen = rows if after_confirm else [b for b in rows if b['B2_score_phase'] == 'explore']
    records = []
    for b in chosen:
        receipt = receipts[pc.key(b)]
        with np.load(receipt['prediction_path'], allow_pickle=False) as f:
            predictions = {a: pc.unpack(f[a]) for a in ARMS}
        r = pr.score_record(b, predictions)
        r.update(score_phase=b['B2_score_phase'], rho=receipt['rho'],
                 gate_triggered=receipt['gate_triggered'], window_available=receipt['window_available'])
        records.append(r)
    notes = ['全部1300同参考真实九窗自然预测先seal；whole/window均用相同T或IND0读出，0FoRIS组件选择/分数依赖。',
             '固定门控rho>.5或参考前景面积<2%；全部连续窗口场按实际coverage1/2/4平均后统一minmax>.5。',
             '九窗是九次额外DINO观测；本次只读历史缓存的成本不能当在线免费视图或端到端冷启动延迟。',
             '完整窗口边界LVIS600/1400、PACO600/600、Deep100/100；缺窗800个LVIS任务固定whole回退，不混入本表的配对窗口分母。']
    if after_confirm:
        notes += ['确认后诊断：本表完整1300；原explore之外192个LVIS任务只在C1完成后评分，不参与组件选择。']
        report = pr.report('B2-after-C1', OUT / 'after_C1', records, ARMS, notes,
                           extra={'C1_report': c1, 'cache_boundary': boundary})
        deferred = [r for r in records if r['score_phase'] == 'after_C1_diagnostic']
        pr.report('B2-deferred-after-C1', OUT / 'after_C1/deferred', deferred, ARMS,
                  notes + ['本表只列保留到C1后的192个任务；不得回头更新组件选择。'])
        pc.write(OUT / 'completion.json', {'state': 'COMPLETE_ALL_1300_AFTER_C1', 'n': len(records),
                 'deferred_n': len(deferred), 'confirm_scored_by_B2': False, 'C1_report': c1,
                 'selection_unchanged_sha256': pc.sha(OUT / 'component_selection.json'), 'new_encoder_calls': 0})
        return report
    report = pr.report('B2', OUT, records, ARMS, notes + [
             '当前仅对固定P0 explore交集1108例评分；192个LVIS任务的GT保留到C1后，当前不声称完整1300评分。'],
             extra={'state': 'EXPLORE_SCORED_DEFERRED_DIAGNOSTICS_PENDING_C1',
                    'cache_boundary': boundary, 'production_cost': {k: v for k, v in seal.items() if k != 'records'}})
    candidates = {}
    for method in ('T', 'IND0'):
        arm, baseline = 'gated_' + method, 'whole_' + method
        checks = {}
        for d, table in report['summary'].items():
            small_a, small_b = table[arm]['size_bands']['<2%'], table[baseline]['size_bands']['<2%']
            gain = None if not small_a['n'] else small_a['episode_mean_iou'] - small_b['episode_mean_iou']
            other = {band: None if not table[arm]['size_bands'][band]['n'] else
                     table[arm]['size_bands'][band]['episode_mean_iou'] - table[baseline]['size_bands'][band]['episode_mean_iou']
                     for band in ('2-20%', '>20%')}
            checks[d] = {'small_target_gain_pp': gain, 'other_bands_gain_pp': other,
                         'nondecreasing_present_bands': (gain is None or gain >= 0) and all(v is None or v >= 0 for v in other.values())}
        small = [r for r in records if r['metrics'][baseline]['target_fraction'] < .02]
        small_gain = float(np.mean([r['metrics'][arm]['IoU'] - r['metrics'][baseline]['IoU'] for r in small])) if small else None
        candidates[method] = {'per_domain': checks, 'common_small_target_n': len(small),
                              'common_small_target_episode_gain_pp': small_gain,
                              'retain_uniform_window_component': small_gain is not None and small_gain >= 3 and
                                                                 all(v['nondecreasing_present_bands'] for v in checks.values())}
    pc.write(OUT / 'component_selection.json', {'state': 'FROZEN_FROM_EXPLORE_ONLY', 'n': len(records),
             'candidates': candidates, 'scope': 'Existing window domains only; no extrapolation to uncached domains',
             'deferred_diagnostic_n': len(rows) - len(records), 'deferred_used_for_selection': False,
             'selected_rule': 'Uniform rho>.5 OR reference area<.02 only if stated small/other-band criterion holds',
             'criterion': 'Common-query small-target episode-mean gain >=3pp; every present size band in each available domain is nondecreasing',
             'source_report_sha256': pc.sha(OUT / 'report.json'), 'new_encoder_calls': 0})
    pc.write(OUT / 'completion.json', {'state': 'EXPLORE_COMPLETE_DEFERRED_DIAGNOSTICS_PENDING_C1',
             'produced_n': len(rows), 'explore_scored_n': len(records), 'after_C1_pending_n': len(rows) - len(records),
             'new_encoder_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score', 'score-after-confirm'))
    parser.add_argument('--query-block', type=int, default=256)
    parser.add_argument('--c1-report', type=Path, default=lab.ASSETS / 'a/x1012_C1/report.json')
    args = parser.parse_args()
    if args.query_block <= 0:
        parser.error('--query-block must be positive')
    if args.mode == 'produce':
        produce(args)
    else:
        score(args.mode == 'score-after-confirm', args.c1_report)
