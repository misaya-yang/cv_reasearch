"""Fixed official-CRF and area-preserving boundary operators on the soft winner.

a: exact frozen .25 M + .5 U + .25 T mask.
b: official FoRIS1024 CRF(a, original query RGB), unchanged parameters.
c: within each16x16 block, b first / a's continuous p second / row-major ties,
   keeping a's original foreground count. No fitting or additional encoder.
d/e: a OR b (accept CRF additions only) / a AND b (accept deletions only).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '2'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'scripts/x1012'), str(ROOT/'src')]
import x1012s2_soft_composition as sc
import lab
import plan_common as pc
import plan_report as pr

# The imported soft runner configures its own one-thread workers. This task
# has a separately authorized two-thread resource envelope; formulas unchanged.
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '2'

OUT = lab.ASSETS/'a/x1012s2_boundary_composition'
SOFT = lab.ASSETS/'a/x1012s2_soft_composition'
OFFICIAL = lab.ASSETS/'third_party/foris_official'
SOFT_ARM = 'b0m1u2t1'
ARMS = ('soft', 'soft_crf', 'soft_crf_area16', 'soft_or_crf', 'soft_and_crf')


def initialize_cpu():
    pc.setup('cpu')
    torch.set_num_threads(2)


def initialize_crf():
    global HOST, ADAPTER, LOAD_INPUTS, CRF_REFINE, INIT_SECONDS
    started = time.monotonic()
    initialize_cpu()
    HOST, ADAPTER = pc.host_cpu()
    from ics.official_data import load_inputs
    from utils.refinement import crf_refine
    LOAD_INPUTS, CRF_REFINE = load_inputs, crf_refine
    assert Path(inspect.getsourcefile(inspect.unwrap(CRF_REFINE))).resolve() == (OFFICIAL/'utils/refinement.py').resolve()
    assert HOST.image_size == 1024 and HOST._crf_band_px == 10 and HOST._crf_p_core == .95
    assert ADAPTER.encoder is None and ADAPTER.encoder_calls == 0
    torch.set_num_threads(2)
    INIT_SECONDS = time.monotonic()-started


def blocks(array):
    return np.asarray(array).reshape(64, 16, 64, 16).transpose(0, 2, 1, 3).reshape(4096, 256)


def unblocks(array):
    return array.reshape(64, 64, 16, 16).transpose(0, 2, 1, 3).reshape(1024, 1024)


def area_preserving_reorder(a, b, p):
    counts = blocks(a).sum(1)
    primary = blocks(b).astype(np.int8)
    secondary = blocks(p)
    positions = np.broadcast_to(np.arange(256), primary.shape)
    # Last key is primary in np.lexsort. This is an exact priority order even
    # if a source's original continuous range extends outside [0,1].
    order = np.lexsort((positions, -secondary, -primary), axis=1)
    ranked = np.zeros((4096, 256), dtype=bool)
    ranked[np.arange(4096)[:, None], order] = np.arange(256)[None, :] < counts[:, None]
    assert np.array_equal(ranked.sum(1), counts)
    result = unblocks(ranked)
    return result, {'mixed_blocks': int(((counts > 0) & (counts < 256)).sum()),
                    'foreground_before': int(counts.sum()), 'foreground_after': int(result.sum()),
                    'all_block_counts_equal': True,
                    'p_min': float(p.min()), 'p_max': float(p.max()),
                    'score_2b_plus_p_exact_priority_equivalent': bool(np.max(np.ptp(secondary, axis=1)) < 2)}


def source_files():
    return [Path(__file__), Path(sc.__file__), Path(lab.__file__), Path(pc.__file__), Path(pr.__file__),
            ROOT/'src/ics/official_data.py', ROOT/'scripts/cached_dino.py',
            ROOT/'src/ics/native_basis.py', ROOT/'src/ics/m4_crf.py', ROOT/'src/ics/m4_crf_cached.py',
            OFFICIAL/'models/foris.py', OFFICIAL/'utils/refinement.py', OFFICIAL/'utils/data.py']


def prepare():
    for folder in ('frozen', 'records', 'predictions'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    if not (OUT/'config.json').exists():
        dependency_paths = [SOFT/'manifest.json', SOFT/'config.json', SOFT/'sources_sealed.json', SOFT/'sealed.json']
        binaries = [lab.ASSETS/'runtime/macos/crf/ics_permutohedral_m4.so',
                    lab.ASSETS/'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
        config = {
            'state': 'FROZEN_BEFORE_BOUNDARY_NATURAL_OUTPUTS', 'n': 1360, 'arms': ARMS,
            'a': 'bitwise frozen soft arm b0m1u2t1: (.25 M + .5 U + .25 T) > .5',
            'continuous_p': 'exact original source float32 frames, accumulated in float64 as (M+2U+T)/4; no normalization, clipping or source changes',
            'b': 'official utils.refinement.crf_refine with original FoRIS1024 host CRF, original query crop/resize/normalization via load_inputs and host.set_target',
            'crf_parameters': {'image_size': 1024, 'alpha': 24., 'beta': .03, 'gamma': 8.,
                               'spatial_weight': 3., 'bilateral_weight': 20., 'compatibility': 1.,
                               'iterations': 10, 'solver': 'fw', 'regularizer': 'l2', 'stepsize': 1.,
                               'lambda': 1., 'x0_weight': 0., 'band_px': 10, 'p_core': .95},
            'c': 'each16x16 block keeps a foreground count; exact lexicographic descending b, descending p, ascending row-major position; equivalent to stable descending2*b+p when within-block p range<2',
            'd': 'a OR b; accept only CRF additions; reuse the exact same b, no additional CRF',
            'e': 'a AND b; accept only CRF deletions; reuse the exact same b, no additional CRF',
            'area_constraint_scope': 'only operator c; not a constraint on other candidates or future combinations',
            'iSAID': 'area preserved before evaluation ignore filtering; no ignore mask in inference',
            'new_encoder_calls': 0, 'training': False, 'parameter_fitting': False, 'query_GT_in_inference': False,
            'selection': 'compare five complete fixed operations; no requirement each step or dataset improves; development comparison, not confirmation',
            'resources': {'workers': 2, 'threads_per_worker': 2, 'device': 'cpu'},
            'source_sha256': {str(p): pc.sha(p) for p in source_files()},
            'dependency_sha256': {str(p): pc.sha(p) for p in dependency_paths},
            'native_binary_sha256': {str(p): pc.sha(p) for p in binaries},
        }
        for i, p in enumerate(source_files()):
            shutil.copyfile(p, OUT/'frozen'/f'{i}_{p.name}')
        shutil.copyfile(SOFT/'manifest.json', OUT/'manifest.json')
        pc.write(OUT/'config.json', config)
        (OUT/'SPEC.md').write_text(
            '# Frozen boundary composition on the existing soft winner\n\n'
            'a is the exact saved .25M+.5U+.25T >.5 mask; b applies once the original '
            'FoRIS1024 CRF using its unchanged parameters and original query RGB preprocessing. '
            'c ranks each16x16 block by b foreground first, a continuous p second, then row-major '
            'ties, selecting exactly the original a foreground count. Lexicographic ordering is '
            'used directly; 2*b+p equivalence is recorded from the actual within-block p range. '
            'The area constraint belongs only to c. No fitting, additional encoder, changed soft '
            'weights, grid or extra CRF variants. Also retain a OR b and a AND b to accept only '
            'additions or deletions, reusing the same b with no additional CRF. All1360 natural '
            'b/c/OR/AND outputs seal before scoring.\n\n'
            'Two CPU workers, two threads each. Record original source reconstruction, query '
            'preprocessing, original CRF, area reorder and full case wall times separately. '
            'Compare complete outputs and four edits to original FoRIS and a; costs do not include '
            'upstream DINO or the already cached soft producer.\n', encoding='utf-8')
    config = pc.read(OUT/'config.json')
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'Boundary source drift'
    assert all(pc.sha(p) == h for p, h in config['dependency_sha256'].items()), 'Boundary dependency drift'
    assert all(pc.sha(p) == h for p, h in config['native_binary_sha256'].items()), 'CRF binary drift'
    rows = pc.read(OUT/'manifest.json')
    assert len(rows) == 1360
    return rows


def produce_one(binding):
    name = pc.key(binding)
    receipt_path = OUT/'records'/f'{name}.json'
    if receipt_path.exists():
        record = pc.read(receipt_path)
        assert pc.sha(record['prediction_path']) == record['prediction_sha256']
        return record
    began = time.monotonic()
    with pc.inference_only(binding):
        started = time.monotonic()
        source = pc.read(SOFT/'source_records'/f'{name}.json')
        old = pc.read(SOFT/'records'/f'{name}.json')
        assert pc.sha(SOFT/'source_records'/f'{name}.json') == old['source_record_sha256']
        assert pc.sha(old['prediction_path']) == old['prediction_sha256']
        for path, digest in ((source['M']['path'], source['M']['sha256']),
                             (source['R1_field_path'], source['R1_field_sha256']),
                             (source['P0_field_path'], source['P0_field_sha256'])):
            assert pc.sha(path) == digest
        if source['T_mode'] == 'original_MPS_actual_frame':
            assert pc.sha(source['T_frame_path']) == source['T_frame_sha256']
        with np.load(old['prediction_path'], allow_pickle=False) as z:
            a_bits = z[SOFT_ARM].copy()
        a = pc.unpack(a_bits)
        frames, _ = sc.frames(source)
        p = (frames[1].astype(np.float64)+2*frames[2].astype(np.float64)+frames[3].astype(np.float64)).reshape(1024, 1024)/4
        assert np.array_equal(p > .5, a), (name, 'Soft continuous/mask identity')
        source_seconds = time.monotonic()-started
        started = time.monotonic()
        _, _, query = LOAD_INPUTS(binding['row'], lab.ASSETS)
        HOST.set_target(query)
        image = HOST._tgt_image.clone().unsqueeze(0)
        assert image.dtype == torch.float32 and tuple(image.shape) == (1, 3, 1024, 1024)
        preprocess_seconds = time.monotonic()-started
        started = time.monotonic()
        b_tensor = CRF_REFINE(HOST._crf, HOST._crf_band_px, HOST._crf_p_core, image, torch.from_numpy(a))
        b = b_tensor.numpy()
        crf_seconds = time.monotonic()-started
        started = time.monotonic()
        c, area_info = area_preserving_reorder(a, b, p)
        reorder_seconds = time.monotonic()-started
        started = time.monotonic()
        accept_additions, accept_deletions = a | b, a & b
        setop_seconds = time.monotonic()-started
        assert ADAPTER.encoder is None and ADAPTER.encoder_calls == 0
        target = OUT/'predictions'/f'{name}.npz'
        assert not target.exists(), 'Preserve orphaned boundary output '+name
        started = time.monotonic()
        np.savez_compressed(target, soft=a_bits, soft_crf=pc.pack(b), soft_crf_area16=pc.pack(c),
                            soft_or_crf=pc.pack(accept_additions), soft_and_crf=pc.pack(accept_deletions))
        save_seconds = time.monotonic()-started
    HOST._ref_images = HOST._ref_masks = HOST._tgt_image = HOST._orig_tgt_size = None
    record = {'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
              'prediction_path': str(target), 'prediction_sha256': pc.sha(target),
              'source_prediction_sha256': old['prediction_sha256'], 'source_record_sha256': old['source_record_sha256'],
              'soft_identity_xor': 0, 'area_reorder': area_info,
              'timing': {'source_seconds': source_seconds, 'query_preprocess_seconds': preprocess_seconds,
                         'official_crf_seconds': crf_seconds, 'area_reorder_seconds': reorder_seconds,
                         'setop_seconds': setop_seconds,
                         'save_seconds': save_seconds, 'case_seconds': time.monotonic()-began,
                         'worker_init_seconds': INIT_SECONDS},
              'seconds': time.monotonic()-began, 'device': 'cpu', 'threads': torch.get_num_threads(),
              'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'training': False, 'parameter_fitting': False}
    pc.write(receipt_path, record)
    return record


def produce(workers):
    initialize_cpu()
    rows = prepare()
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_crf) as pool:
        for i, record in enumerate(pool.map(produce_one, rows, chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 25 == 0:
                progress = {'phase': 'boundary-produce', 'completed': i, 'n': len(rows),
                            'seconds': round(time.monotonic()-began, 2),
                            'mean_case_seconds_so_far': float(np.mean([r['seconds'] for r in records])),
                            'mean_crf_seconds_so_far': float(np.mean([r['timing']['official_crf_seconds'] for r in records]))}
                pc.write(OUT/'progress.json', progress)
                print(progress, flush=True)
    timing = {k: {'mean': float(np.mean([r['timing'][k] for r in records])),
                  'median': float(np.median([r['timing'][k] for r in records])),
                  'p95': float(np.quantile([r['timing'][k] for r in records], .95)),
                  'sum': float(sum(r['timing'][k] for r in records))}
              for k in ('source_seconds', 'query_preprocess_seconds', 'official_crf_seconds', 'area_reorder_seconds', 'setop_seconds', 'save_seconds', 'case_seconds')}
    pc.write(OUT/'sealed.json', {'state': 'ALL1360_NATURAL_BOUNDARY_OUTPUTS_SEALED', 'n': len(rows), 'arms': ARMS,
                              'records': records, 'manifest_sha256': pc.sha(OUT/'manifest.json'),
                              'config_sha256': pc.sha(OUT/'config.json'), 'wall_seconds': time.monotonic()-began,
                              'timing': timing, 'workers': workers, 'threads_per_worker': 2,
                              'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'training': False})
    print({'phase': 'boundary-produce', 'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows)}, flush=True)


def score_one(job):
    binding, receipt = job
    assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
    gt, valid = pc.truth(binding)
    g, v = pc.pack(gt), pc.pack(valid)
    ng, nv = int(gt.sum()), int(valid.sum())
    source = pc.read(SOFT/'source_records'/f"{pc.key(binding)}.json")
    with np.load(source['mask_path'], allow_pickle=False) as z:
        b_foris = z['B'] & v
    metrics, edits_b, edits_a = {}, {}, {}
    with np.load(receipt['prediction_path'], allow_pickle=False) as z:
        a = z['soft'] & v
        for arm in ARMS:
            pred = z[arm] & v
            tp, n = sc.popcount(pred & g), sc.popcount(pred)
            fp, fn = n-tp, ng-tp
            error = float(abs(np.log2(n/ng))) if n and ng else (0. if not ng and not n else None)
            metrics[arm] = {'I': tp, 'U': tp+fp+fn, 'TP': tp, 'FP': fp, 'FN': fn, 'GT': ng,
                            'predicted': n, 'IoU': 100*tp/max(tp+fp+fn, 1), 'target_fraction': ng/max(nv, 1),
                            'abs_log2_area_error': error, 'infinite_area_error': error is None, 'zero_predicted_area': n == 0}
            for baseline, dest in ((b_foris, edits_b), (a, edits_a)):
                added, deleted = pred & ~baseline, baseline & ~pred
                atp, dtp = sc.popcount(added & g), sc.popcount(deleted & g)
                dest[arm] = {'add_TP': atp, 'add_FP': sc.popcount(added)-atp,
                             'delete_TP': dtp, 'delete_FP': sc.popcount(deleted)-dtp}
    return {'index': pc.key(binding), 'episode_id': binding['row']['episode_id'], 'dataset': binding['dataset'],
            'fold': binding['row']['fold'], 'class_id': binding['row']['global_class_id'],
            'query_photo': str((lab.ASSETS/binding['row']['query_path']).resolve()), 'metrics': metrics,
            'edits_vs_FoRIS': edits_b, 'edits_vs_soft': edits_a}


def score(workers):
    initialize_cpu()
    rows = prepare()
    seal = pc.read(OUT/'sealed.json')
    assert seal['state'] == 'ALL1360_NATURAL_BOUNDARY_OUTPUTS_SEALED' and seal['n'] == len(rows)
    assert pc.sha(OUT/'config.json') == seal['config_sha256'] and pc.sha(OUT/'manifest.json') == seal['manifest_sha256']
    receipts = {r['index']: r for r in seal['records']}
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_cpu) as pool:
        for i, r in enumerate(pool.map(score_one, ((b, receipts[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(r)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'boundary-score', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-began, 2)}, flush=True)
    previous = {r['index']: r for r in pc.read(SOFT/'report.json')['records']}
    controls = ('FoRIS', 'MEAN', 'U', 'T', sc.BOOL, 'insid3.bilinear', 'insid3.crf640')
    for r in records:
        old = previous[r['index']]
        assert r['query_photo'] == old['query_photo']
        assert all(r['metrics']['soft'][k] == old['metrics'][SOFT_ARM][k] for k in ('I', 'U', 'GT', 'predicted'))
        for arm in controls:
            r['metrics'][arm] = old['metrics'][arm]
    summary = pr.summary(records, controls+ARMS)
    means = {a: float(np.mean([t[a]['class_miou'] for t in summary.values()])) for a in controls+ARMS}
    contrasts = {}
    for baseline in ('soft', 'soft_crf'):
        projected = [dict(r, metrics={'FoRIS': r['metrics'][baseline], **{a: r['metrics'][a] for a in ARMS}}) for r in records]
        contrasts['versus_'+baseline] = pr.summary(projected, ('FoRIS',)+ARMS)
    pc.write(OUT/'paired_contrasts.json', contrasts)
    edit_sums = {ref: {d: {a: {k: sum(r[ref][a][k] for r in records if r['dataset'] == d)
                                  for k in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')}
                           for a in ARMS} for d in summary} for ref in ('edits_vs_FoRIS', 'edits_vs_soft')}
    ranking = sorted(ARMS, key=lambda a: (-means[a], a))
    result = {'state': 'COMPLETE', 'n': len(rows), 'arms': controls+ARMS, 'summary': summary, 'records': records,
              'equal_domain_means': means, 'ranking': ranking, 'edits': edit_sums,
              'production': {k: v for k, v in seal.items() if k != 'records'},
              'score_seconds': time.monotonic()-began, 'training': False, 'new_encoder_calls': 0,
              'query_GT_only_after_all_natural_predictions_sealed': True,
              'all_soft_identity_xor_zero': True, 'all_c_block_counts_equal_a': True}
    pc.write(OUT/'report.json', result)
    lines = ['# 当前完整软组合的固定边界算子实验', '',
             'a=原冻结soft(.25M+.5U+.25T)；b=一次原FoRIS1024 CRF；c=以b作块内主排序、p作次排序并保留a的每16×16块前景数。稳定row-major同分；面积限制仅属于c算子。另固定a OR b和a AND b，分别只接受CRF新增/删除，无额外CRF成本。',
             '全部1360自然输出封存后评分。原soft权重、来源、场归一化不变；无新网格、训练或编码。官方query输入和原CRF函数/参数未改。',
             '指标按(fold,global_class)池化I/U再观察槽等权；十域均值等权。iSAID忽略区仅在评分使用。照片配对bootstrap1000次，seed20261012。开发结果，不能充作独立确认。', '',
             f'十域均值：a **{means["soft"]:.6f}**；b **{means["soft_crf"]:.6f}**；c **{means["soft_crf_area16"]:.6f}**；a OR b **{means["soft_or_crf"]:.6f}**；a AND b **{means["soft_and_crf"]:.6f}**。', '',
             '|域|FoRIS B|MEAN M|R1 U|a soft|b CRF|c面积保持|a OR b|a AND b|b−a pp|c−a pp|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, t in summary.items():
        cells = '|'.join(f'{t[a]["class_miou"]:.3f}' for a in ('FoRIS', 'MEAN', 'U', *ARMS))
        lines.append(f'|{d}|{cells}|{t["soft_crf"]["class_miou"]-t["soft"]["class_miou"]:+.3f}|{t["soft_crf_area16"]["class_miou"]-t["soft"]["class_miou"]:+.3f}|')
    for ref in ('edits_vs_FoRIS', 'edits_vs_soft'):
        lines += ['', '## '+ref, '', '|域|臂|add TP|add FP|delete TP|delete FP|', '|---|---|---:|---:|---:|---:|']
        for d, table in edit_sums[ref].items():
            for arm in ARMS:
                e = table[arm]
                lines.append(f'|{d}|{arm}|{e["add_TP"]}|{e["add_FP"]}|{e["delete_TP"]}|{e["delete_FP"]}|')
    lines += ['', '## 实测算子耗时', '', '|项目|平均秒|中位秒|p95秒|', '|---|---:|---:|---:|']
    for name, t in seal['timing'].items():
        lines.append(f'|{name}|{t["mean"]:.4f}|{t["median"]:.4f}|{t["p95"]:.4f}|')
    lines += ['', f'两worker×两CPU线程，批处理墙钟{seal["wall_seconds"]:.3f}秒。case时间包含源重建、RGB读取/预处理、CRF、重排和保存；不包含上游DINO/soft产生。CRF和重排时间已单列，不能将并发平均吞吐当作在线串行延迟。',
              '全部class/episode/三档面积/FN/FP/面积误差/对FoRIS CI、逐例四项增删在report.json；a/b/c之间配对CI在paired_contrasts.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    completion = {'state': 'COMPLETE', 'n': len(rows), 'ranking': ranking, 'equal_domain_means': means,
                  'all_soft_identity_xor_zero': True, 'all_c_block_counts_equal_a': True,
                  'new_encoder_calls': 0, 'training': False, 'query_GT_in_inference': False,
                  'score_seconds': time.monotonic()-began, 'report_path': str(OUT/'report.md'),
                  'report_sha256': pc.sha(OUT/'report.json'), 'development_only': True}
    pc.write(OUT/'completion.json', completion)
    print(completion, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    initialize_cpu()
    if args.mode == 'freeze':
        rows = prepare()
        print({'state': 'FROZEN', 'n': len(rows), 'arms': ARMS, 'path': str(OUT)}, flush=True)
    elif args.mode == 'produce':
        produce(args.workers)
    else:
        score(args.workers)
