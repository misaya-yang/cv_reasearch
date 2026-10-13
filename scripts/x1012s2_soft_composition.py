"""Frozen 35 quarter-grid convex mixtures of exact B/M/U/T source readouts.

Cache-only source recovery, then all 1360 natural predictions, then scoring.
No query annotation in source recovery or production; no encoder or fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'src')]
import lab
import plan_common as pc
import plan_report as pr
import repo_methods as rm
from ics.methods import mean_control, rcg

OUT = lab.ASSETS/'a/x1012s2_soft_composition'
R1 = lab.ASSETS/'a/x1012_R1'
P0 = lab.ASSETS/'a/x1012_P0/fields'
BC = lab.ASSETS/'a/x1012_B4/seed150/controls'
INSID3 = lab.ASSETS/'a/x1012_R6_insid3/development1360'
SOURCES = ('B', 'M', 'U', 'T')
WEIGHTS = [(b, m, u, 4-b-m-u) for b in range(5) for m in range(5-b) for u in range(5-b-m)]
ARMS = tuple(f'b{b}m{m}u{u}t{t}' for b, m, u, t in WEIGHTS)
VERTICES = {'B': 'b4m0u0t0', 'M': 'b0m4u0t0', 'U': 'b0m0u4t0', 'T': 'b0m0u0t4'}
BOOL = 'Boolean_r00097'
LUT = np.array([int(i).bit_count() for i in range(256)], np.uint8)


def initialize():
    pc.setup('cpu')
    torch.set_num_threads(1)


def up64(x, device='cpu', normalize=False):
    s = lab.up(torch.as_tensor(x, dtype=torch.float32, device=device).reshape(64, 64))
    if normalize:
        s = lab.minmax(s)
    return s.cpu().numpy()


def source_files():
    return [Path(__file__), Path(lab.__file__), Path(pc.__file__), Path(pr.__file__),
            Path(rm.__file__), Path(mean_control.__file__), Path(rcg.__file__)]


def freeze():
    for folder in ('frozen', 'source_records', 'source_masks', 'mean_fields', 'original_T_frames', 'records', 'predictions'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    if not (OUT/'config.json').exists():
        assert len(WEIGHTS) == len(set(WEIGHTS)) == 35 and all(sum(w) == 4 for w in WEIGHTS)
        config = {
            'state': 'FROZEN_BEFORE_SOURCE_RECOVERY_AND_NATURAL_PREDICTIONS', 'n': 1360,
            'source_order': SOURCES, 'formulas': {a: dict(zip(SOURCES, [v/4 for v in w])) for a, w in zip(ARMS, WEIGHTS)},
            'weight_grid': [0, .25, .5, .75, 1], 'weight_sum': 1, 'mixtures': 35,
            'B': 'bitwise original complete FoRIS CRF mask as float32 0/1 at1024; preserve original boundary',
            'M': 'actual stored MEAN64 bilinear1024 align_corners=False; no minmax/clip; absent or mismatching stored field is recovered with exact repo_methods.mean_field; strict>.5 must equal existing B4 MEAN control',
            'U': 'R1 VP_UN float32 field -> original CPU bilinear1024 -> minmax; strict>.5 bitwise equals sealed R1 VP_UN',
            'T': 'P0 c32 -> original bilinear1024 -> minmax; strict>.5 bitwise equals P0 mask/T; original MPS replay only for CPU numerical mismatch, no mask repair',
            'combination': 'float64 exact weighted sum of source float32 values, implemented integer quarter weights and threshold>2; no extra normalization, clipping, CRF or source modification',
            'original_T_frame_storage': 'only exceptional original MPS1024 source frames; all ordinary source frames temporary; no mixed float1024 fields stored',
            'existing_boolean_control': 'U & (B | M | T), bitwise packed; known root r00097, not a new searched arm',
            'selection': 'maximum equal-domain mean class mIoU over all35 fixed global formulas; deterministic lexicographic arm tie-break; development selection only',
            'resources': {'cpu_workers': 3, 'threads_per_worker': 1, 'MPS': 'single coordinator only for exceptional original T rendering'},
            'new_encoder_calls': 0, 'training': False, 'parameter_fitting': False, 'query_GT_in_inference': False,
            'source_sha256': {str(p): pc.sha(p) for p in source_files()},
            'dependency_sha256': {str(p): pc.sha(p) for p in (R1/'manifest.json', R1/'sealed.json', P0/'sealed.json', BC/'sealed.json')},
        }
        for i, p in enumerate(source_files()):
            shutil.copyfile(p, OUT/'frozen'/f'{i}_{p.name}')
        shutil.copyfile(R1/'manifest.json', OUT/'manifest.json')
        pc.write(OUT/'config.json', config)
        (OUT/'SPEC.md').write_text(
            '# Frozen exact-source soft composition\n\n'
            'Enumerate all35 global convex combinations of B/M/U/T at quarter weights summing to1. '
            'B is the complete original FoRIS CRF binary mask as0/1. M/U/T are their actual1024 continuous '
            'readouts; each single-source >.5 must reproduce its existing natural mask bitwise. '
            'No renormalization/clipping after mixing and no extra CRF. Each mixture uses exact float64 '
            'accumulation of float32 source values and strict>.5. B4 controls, R1 and P0 define source identities.\n\n'
            'Recover missing/mismatched MEAN64 through the original repository function. Reproduce rare '
            'P0 CPU/MPS boundary discrepancies through the original MPS renderer, never through GT or mask '
            'patching. Only those exceptional source1024 frames may be retained; all mixed fields are temporary '
            'and masks are packed/compressed. Seal all1360 natural predictions before scoring. '
            'Selection is highest equal-domain mean class mIoU on development, with a global formula. '
            'This finite operation batch is not a future search-depth or method-count limit.\n', encoding='utf-8')
    cfg = pc.read(OUT/'config.json')
    assert all(pc.sha(p) == h for p, h in cfg['source_sha256'].items()), 'Frozen soft-composition source drift'
    assert all(pc.sha(p) == h for p, h in cfg['dependency_sha256'].items()), 'Frozen source dependency drift'
    rows = pc.read(OUT/'manifest.json')
    assert len(rows) == 1360
    return rows


def prepare_one(binding):
    name = pc.key(binding)
    path = OUT/'source_records'/f'{name}.json'
    if path.exists():
        return pc.read(path)
    began = time.monotonic()
    with pc.inference_only(binding):
        bc = pc.read(BC/'records'/f'{name}.json')
        r1 = pc.read(R1/'records'/f'{name}.json')
        p0 = pc.read(P0/'records'/f'{name}.json')
        assert pc.sha(bc['path']) == bc['sha256']
        assert pc.sha(r1['field_path']) == r1['field_sha256']
        assert pc.sha(r1['prediction_path']) == r1['prediction_sha256']
        assert pc.sha(p0['path']) == p0['sha256']
        with np.load(bc['path'], allow_pickle=False) as z:
            natural = {'B': z['FoRIS'].copy(), 'M': z['MEAN'].copy()}
        with np.load(r1['prediction_path'], allow_pickle=False) as z:
            natural['U'] = z['VP_UN'].copy()
        with np.load(r1['field_path'], allow_pickle=False) as z:
            u = up64(z['VP_UN'], normalize=True)
        assert np.array_equal(pc.pack(u > .5), natural['U']), (name, 'U natural identity')
        with np.load(p0['path'], allow_pickle=False) as z:
            t = up64(z['c32'], normalize=True)
            natural['T'] = z['mask/T'].copy()
        t_xor = int(LUT[pc.pack(t > .5)^natural['T']].sum())
        entry = binding.get('mean64')
        old_m_xor = None
        if entry is not None:
            digest = pc.sha(entry['path'])
            if 'sha256' in entry:
                assert digest == entry['sha256']
            with np.load(entry['path'], allow_pickle=False) as z:
                mean = np.asarray(z[entry['key']], dtype=np.float32).reshape(64, 64)
            old_m_xor = int(LUT[pc.pack(up64(mean) > .5)^natural['M']].sum())
        if entry is None or old_m_xor:
            mean = rm.mean_field(binding)
            recomputed_xor = int(LUT[pc.pack(up64(mean) > .5)^natural['M']].sum())
            assert recomputed_xor == 0, (name, 'Exact original MEAN replay still mismatches', old_m_xor, recomputed_xor)
            mp = OUT/'mean_fields'/f'{name}.npz'
            np.savez_compressed(mp, mean=mean)
            m_source = {'mode': 'exact_original_repo_function_replay', 'path': str(mp), 'key': 'mean',
                        'sha256': pc.sha(mp), 'initial_cached_mask_xor': old_m_xor}
        else:
            m_source = {'mode': 'existing_actual_continuous_MEAN', 'path': entry['path'],
                        'key': entry['key'], 'sha256': digest, 'initial_cached_mask_xor': 0}
        masks = OUT/'source_masks'/f'{name}.npz'
        np.savez_compressed(masks, **natural)
    record = {'index': name, 'dataset': binding['dataset'],
              'mask_path': str(masks), 'mask_sha256': pc.sha(masks),
              'M': m_source, 'R1_field_path': r1['field_path'], 'R1_field_sha256': r1['field_sha256'],
              'P0_field_path': p0['path'], 'P0_field_sha256': p0['sha256'],
              'B4_control_sha256': bc['sha256'], 'R1_prediction_sha256': r1['prediction_sha256'],
              'T_CPU_mask_xor': t_xor, 'T_mode': 'original_MPS_required' if t_xor else 'CPU_bitwise_original',
              'M_field_min': float(mean.min()), 'M_field_max': float(mean.max()),
              'seconds': time.monotonic()-began, 'new_encoder_calls': 0, 'query_GT_reads': 0}
    pc.write(path, record)
    return record


def prepare(workers):
    initialize()
    rows = freeze()
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, record in enumerate(pool.map(prepare_one, rows, chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'source-recovery', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-began, 2)}, flush=True)
    by_key = {pc.key(b): b for b in rows}
    exceptions = [r for r in records if r['T_mode'] == 'original_MPS_required']
    if exceptions:
        assert torch.backends.mps.is_available()
        print({'phase': 'original-T-MPS-render-only', 'n': len(exceptions), 'workers': 1, 'encoder_calls': 0}, flush=True)
    for r in exceptions:
        with pc.inference_only(by_key[r['index']]):
            with np.load(r['P0_field_path'], allow_pickle=False) as z:
                t = up64(z['c32'], device='mps', normalize=True)
            with np.load(r['mask_path'], allow_pickle=False) as z:
                expected = z['T']
            assert np.array_equal(pc.pack(t > .5), expected), (r['index'], 'Original MPS T replay mismatch')
            path = OUT/'original_T_frames'/f"{r['index']}.npz"
            np.savez_compressed(path, T=t)
        r.update(T_mode='original_MPS_actual_frame', T_frame_path=str(path), T_frame_sha256=pc.sha(path), T_final_mask_xor=0)
        pc.write(OUT/'source_records'/f"{r['index']}.json", r)
    if exceptions:
        torch.mps.empty_cache()
    pc.write(OUT/'sources_sealed.json', {
        'state': 'ALL_SOURCE_READOUTS_BITWISE_MATCH_NATURAL_MASKS', 'n': len(rows), 'records': records,
        'config_sha256': pc.sha(OUT/'config.json'), 'manifest_sha256': pc.sha(OUT/'manifest.json'),
        'original_T_MPS_exception_n': sum(r['T_mode'] == 'original_MPS_actual_frame' for r in records),
        'MEAN_exact_replay_n': sum(r['M']['mode'] == 'exact_original_repo_function_replay' for r in records),
        'seconds': time.monotonic()-began, 'new_encoder_calls': 0, 'query_GT_reads': 0,
    })
    print({'phase': 'source-recovery', 'state': 'SEALED', 'n': len(rows)}, flush=True)


def frames(record):
    with np.load(record['mask_path'], allow_pickle=False) as z:
        natural = {s: z[s].copy() for s in SOURCES}
    with np.load(record['M']['path'], allow_pickle=False) as z:
        m = up64(z[record['M']['key']])
    with np.load(record['R1_field_path'], allow_pickle=False) as z:
        u = up64(z['VP_UN'], normalize=True)
    if record['T_mode'] == 'original_MPS_actual_frame':
        with np.load(record['T_frame_path'], allow_pickle=False) as z:
            t = z['T'].copy()
    else:
        with np.load(record['P0_field_path'], allow_pickle=False) as z:
            t = up64(z['c32'], normalize=True)
    maps = (pc.unpack(natural['B']).astype(np.float32), m, u, t)
    for source, field in zip(SOURCES, maps):
        assert np.array_equal(pc.pack(field > .5), natural[source]), (record['index'], source, 'Source identity changed')
    return np.stack([s.ravel() for s in maps]), natural


def produce_one(job):
    binding, source = job
    name = pc.key(binding)
    receipt_path = OUT/'records'/f'{name}.json'
    if receipt_path.exists():
        r = pc.read(receipt_path)
        assert pc.sha(r['prediction_path']) == r['prediction_sha256']
        return r
    began = time.monotonic()
    with pc.inference_only(binding):
        source_frames, natural = frames(source)
        packed = np.empty((35, 131072), dtype=np.uint8)
        weights = np.asarray(WEIGHTS, np.float64)
        for start in range(0, 1024*1024, 65536):
            stop = min(start+65536, 1024*1024)
            summed = weights@source_frames[:, start:stop].astype(np.float64)
            packed[:, start//8:stop//8] = np.packbits(summed > 2., axis=1)
        for source_name, arm in VERTICES.items():
            assert np.array_equal(packed[ARMS.index(arm)], natural[source_name]), (name, arm, 'Single-source mixture identity')
        predictions = {a: packed[i] for i, a in enumerate(ARMS)}
        predictions[BOOL] = natural['U'] & (natural['B'] | natural['M'] | natural['T'])
        path = OUT/'predictions'/f'{name}.npz'
        assert not path.exists(), 'Preserve orphaned soft-composition output: '+name
        np.savez_compressed(path, **predictions)
    record = {'index': name, 'dataset': binding['dataset'], 'prediction_path': str(path),
              'prediction_sha256': pc.sha(path), 'source_record_sha256': pc.sha(OUT/'source_records'/f'{name}.json'),
              'seconds': time.monotonic()-began, 'single_source_mask_xor': dict.fromkeys(SOURCES, 0),
              'new_encoder_calls': 0, 'query_GT_reads': 0, 'training': False}
    pc.write(receipt_path, record)
    return record


def produce(workers):
    initialize()
    rows = freeze()
    source_seal = pc.read(OUT/'sources_sealed.json')
    assert source_seal['n'] == len(rows) and source_seal['config_sha256'] == pc.sha(OUT/'config.json')
    sources = {r['index']: r for r in source_seal['records']}
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, record in enumerate(pool.map(produce_one, ((b, sources[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 50 == 0:
                print({'phase': 'soft-produce', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-began, 2)}, flush=True)
    pc.write(OUT/'sealed.json', {
        'state': 'ALL1360_ALL35_NATURAL_MIXTURES_SEALED', 'n': len(rows), 'arms': ARMS, 'records': records,
        'config_sha256': pc.sha(OUT/'config.json'), 'manifest_sha256': pc.sha(OUT/'manifest.json'),
        'sources_seal_sha256': pc.sha(OUT/'sources_sealed.json'), 'seconds': time.monotonic()-began,
        'case_seconds_sum': sum(r['seconds'] for r in records), 'workers': workers, 'threads_per_worker': 1,
        'new_encoder_calls': 0, 'query_GT_reads': 0, 'training': False,
    })
    print({'phase': 'soft-produce', 'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows)}, flush=True)


def popcount(bits):
    return int(LUT[bits].sum(dtype=np.uint64))


def score_one(job):
    binding, receipt = job
    assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
    gt, valid = pc.truth(binding)
    g, v = pc.pack(gt), pc.pack(valid)
    ng, nv = int(gt.sum()), int(valid.sum())
    metrics, edits = {}, {}
    with np.load(receipt['prediction_path'], allow_pickle=False) as z:
        b = z[VERTICES['B']] & v
        for arm in (*ARMS, BOOL):
            pred = z[arm] & v
            tp, n = popcount(pred & g), popcount(pred)
            fp, fn = n-tp, ng-tp
            error = float(abs(np.log2(n/ng))) if n and ng else (0. if not ng and not n else None)
            metrics[arm] = {'I': tp, 'U': tp+fp+fn, 'TP': tp, 'FP': fp, 'FN': fn, 'GT': ng,
                            'predicted': n, 'IoU': 100*tp/max(tp+fp+fn, 1), 'target_fraction': ng/max(nv, 1),
                            'abs_log2_area_error': error, 'infinite_area_error': error is None, 'zero_predicted_area': n == 0}
            added, deleted = pred & ~b, b & ~pred
            atp, dtp = popcount(added & g), popcount(deleted & g)
            edits[arm] = {'add_TP': atp, 'add_FP': popcount(added)-atp,
                          'delete_TP': dtp, 'delete_FP': popcount(deleted)-dtp}
    for label, source in (('FoRIS', 'B'), ('MEAN', 'M'), ('U', 'U'), ('T', 'T')):
        metrics[label] = dict(metrics[VERTICES[source]])
        edits[label] = dict(edits[VERTICES[source]])
    return {'index': pc.key(binding), 'episode_id': binding['row']['episode_id'], 'dataset': binding['dataset'],
            'fold': binding['row']['fold'], 'class_id': binding['row']['global_class_id'],
            'query_photo': str((lab.ASSETS/binding['row']['query_path']).resolve()), 'metrics': metrics, 'edits_vs_FoRIS': edits}


def score(workers):
    initialize()
    rows = freeze()
    seal = pc.read(OUT/'sealed.json')
    assert seal['state'] == 'ALL1360_ALL35_NATURAL_MIXTURES_SEALED' and seal['n'] == len(rows)
    assert pc.sha(OUT/'config.json') == seal['config_sha256']
    assert pc.sha(OUT/'manifest.json') == seal['manifest_sha256']
    receipts = {r['index']: r for r in seal['records']}
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, r in enumerate(pool.map(score_one, ((b, receipts[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(r)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'soft-score', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-began, 2)}, flush=True)
    old = {r['index']: r for r in pc.read(R1/'report.json')['records']}
    insid3 = {r['index']: r for r in pc.read(INSID3/'episodes.json')}
    for r in records:
        previous, ins = old[r['index']], insid3[r['index']]
        assert r['query_photo'] == previous['query_photo'] == ins['query_photo']
        for a, b in (('FoRIS', 'FoRIS'), ('MEAN', 'MEAN'), ('U', 'VP_UN'), ('T', 'T')):
            assert all(r['metrics'][a][k] == previous['metrics'][b][k] for k in ('I', 'U', 'GT', 'predicted')), (r['index'], a, 'Baseline score identity')
        for a in ('insid3.bilinear', 'insid3.crf640'):
            m = dict(ins['metrics'][a]); m['infinite_area_error'] = m['predicted'] == 0 and m['GT'] > 0
            if m['infinite_area_error']:
                m['abs_log2_area_error'] = None
            r['metrics'][a] = m
    all_arms = ('FoRIS', 'MEAN', 'U', 'T', BOOL, 'insid3.bilinear', 'insid3.crf640') + ARMS
    summary = pr.summary(records, all_arms)
    means = {a: float(np.mean([t[a]['class_miou'] for t in summary.values()])) for a in all_arms}
    rank = sorted(ARMS, key=lambda a: (-means[a], a))
    best = rank[0]
    contrasts = {}
    for baseline in ('U', BOOL, 'insid3.crf640'):
        projected = [dict(r, metrics={'FoRIS': r['metrics'][baseline], best: r['metrics'][best]}) for r in records]
        contrasts['versus_'+baseline] = pr.summary(projected, ('FoRIS', best))
    pc.write(OUT/'paired_contrasts.json', contrasts)
    edit_sums = {d: {a: {k: sum(r['edits_vs_FoRIS'][a][k] for r in records if r['dataset'] == d)
                            for k in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')}
                     for a in (*ARMS, BOOL, 'FoRIS', 'MEAN', 'U', 'T')} for d in summary}
    result = {'state': 'COMPLETE', 'n': len(records), 'arms': all_arms, 'summary': summary, 'records': records,
              'equal_domain_means': means, 'ranking': rank, 'best_arm': best,
              'best_weights': dict(zip(SOURCES, [v/4 for v in WEIGHTS[ARMS.index(best)]])),
              'edits_vs_FoRIS': edit_sums, 'training': False, 'new_encoder_calls': 0,
              'source_singleton_score_identity': True, 'query_GT_only_after_all_natural_predictions_sealed': True,
              'production': {k: v for k, v in seal.items() if k != 'records'}, 'score_seconds': time.monotonic()-began}
    pc.write(OUT/'report.json', result)
    lines = ['# 35个全局连续凸组合：完整1360开发结果', '',
             '全部35公式及来源在自然预测前冻结；全部1360预测封存后评分。零训练、零新编码。B是原完整CRF二值输出；M/U/T是各自原连续读出，四个单来源均逐位复现原自然mask。',
             '最终只做严格>.5，没有额外归一化、裁剪或CRF。M若越出[0,1]保持其实际值。8个P0 T的CPU/MPS单像素差用原MPS连续场重放处理，未用GT修mask。',
             '主指标按(fold,global_class)池化I/U再观察槽等权；十域均值等权。95%CI按query照片配对bootstrap1000次，seed20261012；iSAID ignore按原协议排除。全部为开发结果，最优公式尚未独立确认。', '',
             f'最优：`{best}`，权重 `{result["best_weights"]}`，十域均值 **{means[best]:.6f}**。', '',
             '|域|FoRIS B|MEAN M|R1 U|P0 T|布尔r00097|INSID3 crf640|最优连续|对B pp|对B 95%CI|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for d, table in summary.items():
        ci = table[best]['paired_query_photo_bootstrap95_pp']
        cells = '|'.join(f'{table[a]["class_miou"]:.3f}' for a in ('FoRIS', 'MEAN', 'U', 'T', BOOL, 'insid3.crf640', best))
        lines.append(f'|{d}|{cells}|{table[best]["delta_to_foris_pp"]:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|')
    lines += ['', '|公式|B|M|U|T|十域均值|相对B|相对U|相对布尔|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in rank:
        cells = '|'.join(f'{v/4:g}' for v in WEIGHTS[ARMS.index(a)])
        lines.append(f'|{a}|{cells}|{means[a]:.6f}|{means[a]-means["FoRIS"]:+.6f}|{means[a]-means["U"]:+.6f}|{means[a]-means[BOOL]:+.6f}|')
    lines += ['', '## 最优组合对原完整FoRIS的四项增删', '',
              '|域|add TP|add FP|delete TP|delete FP|FN/GT|FP/GT|面积误差中位|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d, table in summary.items():
        v, e = table[best], edit_sums[d][best]
        lines.append(f'|{d}|{e["add_TP"]}|{e["add_FP"]}|{e["delete_TP"]}|{e["delete_FP"]}|{v["FN_GT"]:.4f}|{v["FP_GT"]:.4f}|{v["median_abs_log2_area_error"]}|')
    lines += ['', '全部公式、同例对照的class/episode/三档面积/FN/FP/面积误差/CI及逐例四项增删都在report.json；最优对U、r00097、INSID3的配对CI在paired_contrasts.json。',
              'source recovery、组合枚举均只消费已有输入与预测；耗时单列，不能当作从RGB开始的在线总延迟。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    completion = {'state': 'COMPLETE', 'n': len(rows), 'mixtures': 35, 'best_arm': best,
                  'best_weights': result['best_weights'], 'best_macro': means[best],
                  'baseline_macro': {a: means[a] for a in ('FoRIS', 'MEAN', 'U', 'T', BOOL, 'insid3.crf640')},
                  'development_selection_only': True, 'independent_confirmation_pending': True,
                  'new_encoder_calls': 0, 'training': False, 'query_GT_in_inference': False,
                  'source_singletons_bitwise_match': True, 'score_seconds': time.monotonic()-began,
                  'report_path': str(OUT/'report.md'), 'report_sha256': pc.sha(OUT/'report.json')}
    pc.write(OUT/'completion.json', completion)
    print(completion, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'prepare', 'produce', 'score'))
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    initialize()
    if args.mode == 'freeze':
        rows = freeze()
        print({'state': 'FROZEN', 'n': len(rows), 'formulas': len(ARMS), 'out': str(OUT)}, flush=True)
    elif args.mode == 'prepare':
        prepare(args.workers)
    elif args.mode == 'produce':
        produce(args.workers)
    else:
        score(args.workers)
