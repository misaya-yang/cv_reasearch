"""Fixed MFU boundary operators and optional reference-rho routing, 1360+500.

p=(M+U)/2 using the original continuous source frames. Produce all five fixed
operators and all five rho>.9?T:operator masks before opening query labels.
Both cohorts are exposed development; confirm1000 is never accessed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
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
import x1012s2_boundary_composition as boundary
import x1012s2_soft500 as five

pc, pr, lab, sc = boundary.pc, boundary.pr, boundary.lab, boundary.sc
ROOT, ASSETS = boundary.ROOT, lab.ASSETS
OUT = ASSETS/'a/x1012s2_boundary_mfu'
GATE = ASSETS/'a/x1012s2_reference_gate_composition'
DEV = ASSETS/'a/x1012s2_soft_composition'
FIVE = ASSETS/'a/x1012s2_soft500'
MFU = 'b0m2u2t0'
OLD_SOFT = 'b0m1u2t1'
COHORTS = ('development1360', 'development500')
BASE_ARMS = ('mfu', 'mfu_crf', 'mfu_or_crf', 'mfu_and_crf', 'mfu_area16')
ARMS = BASE_ARMS + tuple('G_'+a for a in BASE_ARMS)
CONTROL_KEYS = {'FoRIS': 'b4m0u0t0', 'MEAN': 'b0m4u0t0', 'U': 'b0m0u4t0', 'T': 'b0m0u0t4', 'old_soft': OLD_SOFT}
EDIT_BASELINES = ('FoRIS', 'MEAN', 'U', 'mfu', 'G_mfu')
EDIT_NAMES = ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')
TIMING_NAMES = ('source_seconds', 'query_preprocess_seconds', 'official_crf_seconds',
                'area_reorder_seconds', 'setop_seconds', 'gate_seconds', 'save_seconds', 'case_seconds')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'


def initialize_cpu():
    pc.setup('cpu')
    torch.set_num_threads(2)


def initialize_crf():
    boundary.initialize_crf()
    assert torch.get_num_threads() == 2


def source_files():
    return list(dict.fromkeys([Path(__file__), Path(five.__file__), Path(five.base.__file__),
                              *boundary.source_files()]))


def prepare():
    for folder in ('frozen', 'records', 'predictions'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    config_path = OUT/'config.json'
    if not config_path.exists():
        gate_bindings = pc.read(GATE/'bindings.json')
        five_sources = {r['index']: r for r in pc.read(FIVE/'bindings.json')}
        development_sources = {r['index']: r for r in pc.read(DEV/'sources_sealed.json')['records']}
        source_seals = {COHORTS[0]: pc.read(DEV/'sealed.json'), COHORTS[1]: pc.read(FIVE/'all35/sealed.json')}
        ref_roots = {COHORTS[0]: five.base.R1, COHORTS[1]: five.base.R5C}
        rows = []
        for cohort, expected in zip(COHORTS, (1360, 500)):
            old_predictions = {r['index']: r for r in source_seals[cohort]['records']}
            references = {r['index']: r for r in pc.read(ref_roots[cohort]/'sealed.json')['records']}
            assert len(gate_bindings[cohort]) == len(old_predictions) == expected
            for g in gate_bindings[cohort]:
                key = g['index']
                old = old_predictions[key]
                rho = float(references[key]['reference_rho'])
                assert rho == g['reference_rho'] and (rho > .9) == g['gate_true']
                assert old['prediction_sha256'] == g['source_prediction_sha256']
                assert old['prediction_path'] == g['source_prediction_path']
                r = {k: g[k] for k in ('index', 'dataset', 'binding', 'reference_rho', 'gate_true',
                                       'source_prediction_path', 'source_prediction_sha256')}
                r['cohort'] = cohort
                if cohort == COHORTS[0]:
                    r['source_record'] = development_sources[key]
                else:
                    r['source500'] = five_sources[key]
                rows.append(r)
        assert len(rows) == 1860 and len({(r['cohort'], r['index']) for r in rows}) == 1860
        assert all('confirm1000' not in str(r['binding'].get('split', '')) for r in rows)
        pc.write(OUT/'manifest.json', rows)
        dependencies = [GATE/'bindings.json', GATE/'config.json', DEV/'sources_sealed.json',
                        DEV/'sealed.json', DEV/'config.json', DEV/'manifest.json', DEV/'report.json',
                        FIVE/'bindings.json', FIVE/'sources_sealed.json', FIVE/'source_contract.json',
                        FIVE/'all35/sealed.json', FIVE/'all35/config.json', FIVE/'all35/report.json',
                        five.base.R1/'sealed.json', five.base.R5C/'sealed.json']
        binaries = [ASSETS/'runtime/macos/crf/ics_permutohedral_m4.so',
                    ASSETS/'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
        config = {
            'state': 'TEN_FIXED_MFU_BOUNDARY_AND_REFERENCE_GATE_ARMS_FROZEN_BEFORE_NEW_NATURAL_OUTPUTS',
            'n': 1860, 'cohort_n': dict(zip(COHORTS, (1360, 500))), 'arms': ARMS,
            'p': 'Original M and U float32 frames converted to FP64; (M+U)/2, strict >.5; exact original b0m2u2t0 identity; no normalization or clipping',
            'mfu': 'p > .5',
            'mfu_crf': 'once the unchanged official FoRIS1024 CRF(mfu, original query RGB and official preprocessing)',
            'mfu_or_crf': 'mfu OR mfu_crf; additions only; same CRF output',
            'mfu_and_crf': 'mfu AND mfu_crf; deletions only; same CRF output',
            'mfu_area16': 'preserve mfu foreground count in each16x16 block; descending CRF binary, descending continuous p, ascending row-major position; existing exact lexicographic algorithm',
            'gate': 'G_arm = original natural T if reference_rho > .9 else arm; strict fixed threshold, no tuning',
            'gate_source': 'existing R1/R5 reference statistics, checked against fixed reference-gate bindings; no query labels or scores',
            'T': 'retain each cohort original sealed T including preserved source-device numerical differences',
            'ungated_arms_preserved': True, 'gate_does_not_change_p_or_crf': True,
            'crf_parameters': pc.read(boundary.OUT/'config.json')['crf_parameters'],
            'source_sha256': {str(p): pc.sha(p) for p in source_files()},
            'dependency_sha256': {str(p): pc.sha(p) for p in dependencies},
            'native_binary_sha256': {str(p): pc.sha(p) for p in binaries},
            'manifest_sha256': pc.sha(OUT/'manifest.json'),
            'resources': {'workers': 2, 'threads_per_worker': 2, 'device': 'cpu'},
            'training': False, 'parameter_fitting': False, 'new_encoder_calls': 0,
            'all18600_natural_masks_sealed_before_GT': True, 'confirm1000_accessed': False,
            'ranking': 'Separate ten-domain class mIoU for each exposed development cohort and their arithmetic equal-cohort mean; one global formula, no per-domain choices',
            'paired_CI': '1000 paired query-photo cluster resamples per domain and cohort; seed20261013; aggregate I/U within observed fold/class slots each draw; descriptive development intervals, not selection-adjusted',
            'comparison_baselines': EDIT_BASELINES,
            'latency_scope': 'Boundary replay only; upstream encoder and M/U/T producer costs excluded. All ungated outputs require CRF; a deployed gated operator can skip CRF when rho>.9, logical savings separately reported.'}
        pc.write(config_path, config)
        for i, p in enumerate(source_files()):
            shutil.copyfile(p, OUT/'frozen'/f'{i}_{p.name}')
        (OUT/'SPEC.md').write_text(
            '# Frozen MFU boundary operators on two exposed development cohorts\n\n'
            'p=(M+U)/2 uses the original float32 continuous sources, accumulated in FP64. '
            'The five fixed outputs are p>.5, official CRF(p>.5), union, intersection, '
            'and the existing16x16 area-preserving reorder. For each, also produce the '
            'fixed reference_rho>.9?T:arm route. No new weights or rho thresholds. '
            'Generate and seal every natural mask for all1360+500 cases before scoring. '
            'Both cohorts are exposed development. confirm1000 is not accessed.\n\n'
            'Report both cohorts separately plus equal-cohort mean, four edit counts, '
            'photo-paired confidence intervals and measured boundary costs. '
            'All formulas are global; a domain or intermediate-step loss is not a selection gate.\n', encoding='utf-8')
    config = pc.read(config_path)
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'MFU boundary source drift'
    assert all(pc.sha(p) == h for p, h in config['dependency_sha256'].items()), 'MFU boundary dependency drift'
    assert all(pc.sha(p) == h for p, h in config['native_binary_sha256'].items()), 'CRF binary drift'
    assert pc.sha(OUT/'manifest.json') == config['manifest_sha256']
    return pc.read(OUT/'manifest.json'), config


def load_sources(row):
    assert pc.sha(row['source_prediction_path']) == row['source_prediction_sha256']
    with np.load(row['source_prediction_path'], allow_pickle=False) as z:
        a_bits = z[MFU].copy()
        t_bits = z[CONTROL_KEYS['T']].copy()
    if row['cohort'] == COHORTS[0]:
        source = row['source_record']
        for path, digest in ((source['mask_path'], source['mask_sha256']),
                             (source['M']['path'], source['M']['sha256']),
                             (source['R1_field_path'], source['R1_field_sha256']),
                             (source['P0_field_path'], source['P0_field_sha256'])):
            assert pc.sha(path) == digest
        if source['T_mode'] == 'original_MPS_actual_frame':
            assert pc.sha(source['T_frame_path']) == source['T_frame_sha256']
        frames, natural = sc.frames(source)
        m, u = frames[1], frames[2]
    else:
        maps, natural = five.frames(row['source500'])
        m, u = maps['M'], maps['U']
    assert np.array_equal(t_bits, natural['T'])
    p = (m.astype(np.float64)+u.astype(np.float64)).reshape(1024, 1024)/2
    a = pc.unpack(a_bits)
    assert np.array_equal(p > .5, a), (row['cohort'], row['index'], 'Original MFU identity')
    return a_bits, a, p, t_bits


def produce_one(row):
    key, cohort = row['index'], row['cohort']
    record_path = OUT/'records'/f'{cohort}_{key}.json'
    if record_path.exists():
        record = pc.read(record_path)
        assert pc.sha(record['prediction_path']) == record['prediction_sha256']
        assert record['config_sha256'] == pc.sha(OUT/'config.json')
        return record
    began = time.monotonic()
    with pc.inference_only(row['binding']):
        started = time.monotonic()
        a_bits, a, p, t_bits = load_sources(row)
        source_seconds = time.monotonic()-started
        started = time.monotonic()
        _, _, query = boundary.LOAD_INPUTS(row['binding']['row'], ASSETS)
        boundary.HOST.set_target(query)
        image = boundary.HOST._tgt_image.clone().unsqueeze(0)
        assert image.dtype == torch.float32 and tuple(image.shape) == (1, 3, 1024, 1024)
        preprocess_seconds = time.monotonic()-started
        started = time.monotonic()
        crf = boundary.CRF_REFINE(boundary.HOST._crf, boundary.HOST._crf_band_px,
                                  boundary.HOST._crf_p_core, image, torch.from_numpy(a)).numpy()
        crf_seconds = time.monotonic()-started
        started = time.monotonic()
        area, area_info = boundary.area_preserving_reorder(a, crf, p)
        area_seconds = time.monotonic()-started
        started = time.monotonic()
        masks = {'mfu': a_bits, 'mfu_crf': pc.pack(crf), 'mfu_or_crf': pc.pack(a | crf),
                 'mfu_and_crf': pc.pack(a & crf), 'mfu_area16': pc.pack(area)}
        setop_seconds = time.monotonic()-started
        started = time.monotonic()
        masks.update({'G_'+arm: t_bits if row['gate_true'] else masks[arm] for arm in BASE_ARMS})
        assert row['gate_true'] == (row['reference_rho'] > .9)
        gate_seconds = time.monotonic()-started
        assert boundary.ADAPTER.encoder is None and boundary.ADAPTER.encoder_calls == 0
        path = OUT/'predictions'/f'{cohort}_{key}.npz'
        assert not path.exists(), 'Preserve orphaned natural output'
        started = time.monotonic()
        np.savez_compressed(path, **masks)
        save_seconds = time.monotonic()-started
    boundary.HOST._ref_images = boundary.HOST._ref_masks = boundary.HOST._tgt_image = boundary.HOST._orig_tgt_size = None
    record = {'index': key, 'cohort': cohort, 'dataset': row['dataset'],
              'prediction_path': str(path), 'prediction_sha256': pc.sha(path),
              'source_prediction_sha256': row['source_prediction_sha256'],
              'config_sha256': pc.sha(OUT/'config.json'), 'reference_rho': row['reference_rho'],
              'gate_true': row['gate_true'], 'mfu_identity_xor': 0, 'area_reorder': area_info,
              'timing': {'source_seconds': source_seconds, 'query_preprocess_seconds': preprocess_seconds,
                         'official_crf_seconds': crf_seconds, 'area_reorder_seconds': area_seconds,
                         'setop_seconds': setop_seconds, 'gate_seconds': gate_seconds,
                         'save_seconds': save_seconds, 'case_seconds': time.monotonic()-began},
              'seconds': time.monotonic()-began, 'device': 'cpu', 'threads': torch.get_num_threads(),
              'new_encoder_calls': 0, 'query_GT_reads': 0, 'training': False}
    pc.write(record_path, record)
    return record


def timing_summary(records):
    return {k: {'mean': float(np.mean([r['timing'][k] for r in records])),
                'median': float(np.median([r['timing'][k] for r in records])),
                'p95': float(np.quantile([r['timing'][k] for r in records], .95)),
                'sum': float(sum(r['timing'][k] for r in records))} for k in TIMING_NAMES}


def produce(workers):
    rows, _ = prepare()
    assert not (OUT/'sealed.json').exists(), 'Preserve all sealed natural outputs'
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_crf) as pool:
        for i, record in enumerate(pool.map(produce_one, rows, chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 25 == 0:
                progress = {'phase': 'mfu-boundary-natural', 'completed': i, 'n': len(rows),
                            'cohort': record['cohort'], 'seconds': round(time.monotonic()-began, 2),
                            'mean_case_seconds': float(np.mean([r['seconds'] for r in records])),
                            'mean_crf_seconds': float(np.mean([r['timing']['official_crf_seconds'] for r in records]))}
                pc.write(OUT/'progress.json', progress)
                print(progress, flush=True)
    seal = {'state': 'ALL1860_TEN_ARMS_NATURAL_MASKS_SEALED', 'n': len(rows), 'mask_n': len(rows)*len(ARMS),
            'arms': ARMS, 'records': records, 'config_sha256': pc.sha(OUT/'config.json'),
            'manifest_sha256': pc.sha(OUT/'manifest.json'), 'wall_seconds': time.monotonic()-began,
            'timing': timing_summary(records),
            'cohorts': {c: {'n': sum(r['cohort'] == c for r in records),
                             'gate_true_n': sum(r['cohort'] == c and r['gate_true'] for r in records),
                             'timing': timing_summary([r for r in records if r['cohort'] == c]),
                             'logical_gated_crf_mean_seconds_per_case': sum(r['timing']['official_crf_seconds'] for r in records if r['cohort'] == c and not r['gate_true'])/sum(r['cohort'] == c for r in records)} for c in COHORTS},
            'workers': workers, 'threads_per_worker': 2, 'training': False,
            'new_encoder_calls': 0, 'query_GT_reads': 0, 'confirm1000_accessed': False}
    pc.write(OUT/'sealed.json', seal)
    print({'state': seal['state'], 'n': seal['n'], 'mask_n': seal['mask_n'], 'wall_seconds': seal['wall_seconds']}, flush=True)


def metric(bits, g, v, ng, nv):
    pred = bits & v
    tp, n = sc.popcount(pred & g), sc.popcount(pred)
    fp, fn = n-tp, ng-tp
    error = float(abs(np.log2(n/ng))) if n and ng else (0. if not n and not ng else None)
    return {'I': tp, 'U': tp+fp+fn, 'TP': tp, 'FP': fp, 'FN': fn, 'GT': ng, 'predicted': n,
            'IoU': 100*tp/max(tp+fp+fn, 1), 'target_fraction': ng/max(nv, 1),
            'abs_log2_area_error': error, 'infinite_area_error': error is None, 'zero_predicted_area': n == 0}


def edits(pred, baseline, g, v):
    a, b = pred & v, baseline & v
    add, delete = a & ~b, b & ~a
    at, dt = sc.popcount(add & g), sc.popcount(delete & g)
    return dict(zip(EDIT_NAMES, (at, sc.popcount(add)-at, dt, sc.popcount(delete)-dt)))


def score_one(job):
    row, receipt = job
    assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
    assert pc.sha(row['source_prediction_path']) == row['source_prediction_sha256']
    with np.load(receipt['prediction_path'], allow_pickle=False) as z:
        masks = {a: z[a].copy() for a in ARMS}
    with np.load(row['source_prediction_path'], allow_pickle=False) as z:
        masks.update({a: z[k].copy() for a, k in CONTROL_KEYS.items()})
        assert np.array_equal(masks['mfu'], z[MFU])
    for a in BASE_ARMS:
        assert np.array_equal(masks['G_'+a], masks['T'] if row['gate_true'] else masks[a])
    b = row['binding']
    gt, valid = pc.truth(b)
    g, v = pc.pack(gt), pc.pack(valid)
    ng, nv = int(gt.sum()), int(valid.sum())
    metrics = {a: metric(p, g, v, ng, nv) for a, p in masks.items()}
    edit_table = {base: {a: edits(masks[a], masks[base], g, v) for a in ARMS} for base in EDIT_BASELINES}
    for base, table in edit_table.items():
        for a, e in table.items():
            assert metrics[a]['TP'] == metrics[base]['TP']+e['add_TP']-e['delete_TP']
            assert metrics[a]['FP'] == metrics[base]['FP']+e['add_FP']-e['delete_FP']
    return {'index': row['index'], 'cohort': row['cohort'], 'dataset': row['dataset'],
            'fold': b['row']['fold'], 'class_id': b['row']['global_class_id'],
            'query_photo': str((ASSETS/b['row']['query_path']).resolve()),
            'reference_rho': row['reference_rho'], 'gate_true': row['gate_true'],
            'metrics': metrics, 'edits': edit_table}


def paired_cohort(records, names, summary, rng):
    from scipy.sparse import csr_matrix
    draws = {a: [] for a in names}
    domain_draws = {}
    for d in lab.ALL_DOMAINS:
        rs = [r for r in records if r['dataset'] == d]
        photos = sorted({r['query_photo'] for r in rs})
        slots = sorted({(str(r['fold']), str(r['class_id'])) for r in rs})
        pm, sm = {p: i for i, p in enumerate(photos)}, {s: i for i, s in enumerate(slots)}
        pi = [pm[r['query_photo']] for r in rs]
        si = [sm[(str(r['fold']), str(r['class_id']))] for r in rs]
        shape = (len(photos), len(slots))
        presence = csr_matrix((np.ones(len(rs)), (pi, si)), shape=shape)
        weights = rng.multinomial(len(photos), np.full(len(photos), 1/len(photos)), size=1000)
        present = (weights @ presence) > 0
        domain_draws[d] = {}
        for a in names:
            ii = csr_matrix(([r['metrics'][a]['I'] for r in rs], (pi, si)), shape=shape)
            uu = csr_matrix(([r['metrics'][a]['U'] for r in rs], (pi, si)), shape=shape)
            values = ((100*(weights @ ii)/np.maximum(weights @ uu, 1))*present).sum(1)/present.sum(1)
            domain_draws[d][a] = values
            draws[a].append(values)
    macro_draws = {a: np.mean(v, axis=0) for a, v in draws.items()}
    means = {a: float(np.mean([summary[d][a]['class_miou'] for d in lab.ALL_DOMAINS])) for a in names}
    domains = {d: make_contrasts(domain_draws[d], {a: summary[d][a]['class_miou'] for a in names}) for d in lab.ALL_DOMAINS}
    return {'domains': domains, 'macro10': make_contrasts(macro_draws, means)}, macro_draws


def make_contrasts(draws, means):
    pairs = [(a, b) for b in EDIT_BASELINES for a in ARMS]
    pairs += [('G_'+a, a) for a in BASE_ARMS]
    return {a+'__minus__'+b: {'candidate': a, 'baseline': b, 'delta_pp': means[a]-means[b],
                              'paired_query_photo_bootstrap95_pp': np.quantile(draws[a]-draws[b], [.025, .975]).tolist()}
            for a, b in pairs}


def score(workers):
    rows, config = prepare()
    seal = pc.read(OUT/'sealed.json')
    assert seal['state'] == 'ALL1860_TEN_ARMS_NATURAL_MASKS_SEALED' and seal['n'] == 1860 and seal['mask_n'] == 18600
    assert seal['config_sha256'] == pc.sha(OUT/'config.json') and seal['manifest_sha256'] == pc.sha(OUT/'manifest.json')
    receipts = {(r['cohort'], r['index']): r for r in seal['records']}
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_cpu) as pool:
        jobs = ((r, receipts[(r['cohort'], r['index'])]) for r in rows)
        for i, r in enumerate(pool.map(score_one, jobs, chunksize=1), 1):
            records.append(r)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'mfu-boundary-score', 'completed': i, 'n': len(rows), 'seconds': round(time.monotonic()-began, 2)}, flush=True)
    old_reports = {COHORTS[0]: pc.read(DEV/'report.json')['records'], COHORTS[1]: pc.read(FIVE/'all35/report.json')['records500']}
    for cohort in COHORTS:
        old = {r['index']: r for r in old_reports[cohort]}
        for r in (r for r in records if r['cohort'] == cohort):
            previous = old[r['index']]
            assert previous['query_photo'] == r['query_photo']
            for a, old_a in dict(CONTROL_KEYS, mfu=MFU).items():
                assert all(r['metrics'][a][k] == previous['metrics'][old_a][k] for k in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted'))
    names = tuple(CONTROL_KEYS)+ARMS
    cohorts, paired, samples = {}, {}, {}
    rng = np.random.default_rng(20261013)
    for cohort in COHORTS:
        rs = [r for r in records if r['cohort'] == cohort]
        summary = pr.summary(rs, names)
        for table in summary.values():
            for m in table.values():
                m.pop('paired_query_photo_bootstrap95_pp', None)
        means = {a: float(np.mean([summary[d][a]['class_miou'] for d in lab.ALL_DOMAINS])) for a in names}
        paired[cohort], samples[cohort] = paired_cohort(rs, names, summary, rng)
        edit_table = {d: {b: {a: {k: sum(r['edits'][b][a][k] for r in rs if r['dataset'] == d) for k in EDIT_NAMES}
                              for a in ARMS} for b in EDIT_BASELINES} for d in lab.ALL_DOMAINS}
        cohorts[cohort] = {'n': len(rs), 'gate_true_n': sum(r['gate_true'] for r in rs),
                           'gate_true_by_domain': dict(Counter(r['dataset'] for r in rs if r['gate_true'])),
                           'summary': summary, 'macro10': means, 'edits': edit_table,
                           'ranking': sorted(ARMS, key=lambda a: (-means[a], a))}
    joint_means = {a: float(np.mean([cohorts[c]['macro10'][a] for c in COHORTS])) for a in names}
    joint_draws = {a: np.mean([samples[c][a] for c in COHORTS], axis=0) for a in names}
    paired['equal_cohort_macro10'] = make_contrasts(joint_draws, joint_means)
    paired['seed'] = 20261013
    paired['repeats'] = 1000
    pc.write(OUT/'paired_contrasts.json', paired)
    ranking = sorted(ARMS, key=lambda a: (-joint_means[a], a))
    result = {'state': 'COMPLETE_TWO_EXPOSED_DEVELOPMENT_COHORTS_MFU_BOUNDARY_FAMILY',
              'n': len(rows), 'arms': names, 'cohorts': cohorts, 'equal_cohort_macro10': joint_means,
              'ranking': ranking, 'winner': ranking[0], 'records': records, 'edit_names': EDIT_NAMES,
              'production': {k: v for k, v in seal.items() if k != 'records'},
              'all_existing_control_and_mfu_integer_identity': True, 'all_gate_mask_identity': True,
              'all_area16_block_counts_preserved': True, 'query_GT_only_after_all18600_masks_sealed': True,
              'new_encoder_calls': 0, 'training': False, 'parameter_fitting': False,
              'confirm1000_accessed': False, 'independent_confirmation': False,
              'score_seconds': time.monotonic()-began, 'config_sha256': pc.sha(OUT/'config.json'),
              'seal_sha256': pc.sha(OUT/'sealed.json')}
    pc.write(OUT/'report.json', result)
    lines = ['# MFU固定边界操作与参考门控：两个已暴露开发队列', '',
             '基底p=(M+U)/2，使用原连续源，严格p>.5；五臂为原mask、一次原FoRIS1024 CRF、OR、AND、每16×16块面积保持重排。各臂另保留固定reference_rho>.9?T:arm版本。全部18600自然mask封存后评分；不训练、不新增编码、不调整rho；未访问confirm1000。',
             '两队列分别计算十域等权class mIoU，再等权平均两队列。class槽为(fold,global_class)池化I/U。开发描述性比较；非独立确认。配对区间为每域query照片簇bootstrap1000次，seed20261013；详情在paired_contrasts.json。', '',
             '|臂|开发1360|旧暴露500|两队列等权|相对MFU pp|相对门控MFU pp|',
             '|---|---:|---:|---:|---:|---:|']
    for a in tuple(CONTROL_KEYS)+tuple(ranking):
        lines.append(f'|{a}|{cohorts[COHORTS[0]]["macro10"][a]:.6f}|{cohorts[COHORTS[1]]["macro10"][a]:.6f}|{joint_means[a]:.6f}|{joint_means[a]-joint_means["mfu"]:+.6f}|{joint_means[a]-joint_means["G_mfu"]:+.6f}|')
    for cohort, c in cohorts.items():
        lines += ['', '## '+cohort, '', f'样本{c["n"]}；rho>.9路由至T：{c["gate_true_n"]}。', '',
                  '|域|'+'|'.join(names)+'|', '|---|'+'|'.join(['---:']*len(names))+'|']
        for d, table in c['summary'].items():
            lines.append('|'+d+'|'+'|'.join(f'{table[a]["class_miou"]:.3f}' for a in names)+'|')
        lines += ['', '|臂|基线|add TP|add FP|delete TP|delete FP|', '|---|---|---:|---:|---:|---:|']
        for a in ARMS:
            for b in ('FoRIS', 'mfu', 'G_mfu'):
                totals = [sum(c['edits'][d][b][a][k] for d in lab.ALL_DOMAINS) for k in EDIT_NAMES]
                lines.append('|'+a+'|'+b+'|'+'|'.join(map(str, totals))+'|')
    lines += ['', '## 边界算子实测耗时', '', '|项|mean s|median s|p95 s|', '|---|---:|---:|---:|']
    for k, t in seal['timing'].items():
        lines.append(f'|{k}|{t["mean"]:.6f}|{t["median"]:.6f}|{t["p95"]:.6f}|')
    lines += ['', f'2workers×2threads，总墙钟{seal["wall_seconds"]:.3f}秒。这里是已有M/U/T后的边界回放；不含上游编码和源产生，不以并发吞吐代替在线串行延迟。',
              '全部class/episode/面积/FN/FP、逐例整数与五基线四项编辑在report.json；两队列分别及等权总效应的配对区间在paired_contrasts.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    completion = {k: v for k, v in result.items() if k not in ('records', 'cohorts')}
    completion['cohorts'] = {c: {k: v for k, v in t.items() if k not in ('summary', 'edits')} for c, t in cohorts.items()}
    completion.update(report_path=str(OUT/'report.md'), report_sha256=pc.sha(OUT/'report.json'))
    pc.write(OUT/'completion.json', completion)
    print({'state': result['state'], 'winner': ranking[0], 'equal_cohort_macro10': joint_means,
           'score_seconds': result['score_seconds'], 'report_path': str(OUT/'report.md')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    initialize_cpu()
    if args.mode == 'freeze':
        rows, config = prepare()
        print({'state': config['state'], 'n': len(rows), 'arms': ARMS,
               'gate_true_n': dict(Counter(r['cohort'] for r in rows if r['gate_true'])), 'path': str(OUT)}, flush=True)
    elif args.mode == 'produce':
        produce(args.workers)
    else:
        score(args.workers)
