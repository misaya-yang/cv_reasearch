"""R4: exact realizable pixel-prefix extent calibration of frozen R1 VP_UN.

Four fixed arms; no encoder, no FoRIS inference input, no query annotation
until every natural prediction has been sealed. CPU only, at most 2 x 2 threads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
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

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1012_R4_extent'
R1 = ASSETS / 'a/x1012_R1'
P0 = ASSETS / 'a/x1012_P0/fields'
RUNTIME = OUT / 'runtime'
ARMS = ('UN_fixed', 'UN_JF', 'UN_JFB', 'UN_JFB_permute')
CONTROLS = ('FoRIS', 'MEAN', 'VP_previous')
SALT = 'R4_extent_back_mass_permutation_20261012'


def load_helpers():
    """Byte-identical R1 helper snapshots; no dependence on concurrent cleanup."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    sources = {'lab': R1/'frozen/3_lab.py',
               'plan_common': R1/'frozen/1_plan_common.py',
               'plan_report': R1/'frozen/2_plan_report.py'}
    modules = {}
    for name, source in sources.items():
        dest = RUNTIME / (name + '.py')
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert source.read_bytes() == dest.read_bytes(), 'Runtime helper drift: ' + name
        spec = importlib.util.spec_from_file_location(name, dest)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'lab':
            # The byte-identical helper derives ROOT from __file__; restore its
            # original project/data roots after relocating the runtime snapshot.
            module.ROOT, module.ASSETS = ROOT, ASSETS
            module.BINDINGS = ASSETS/'a/consensus_reference_calibration6380_v1_20261011/source_bindings.json'
            module.BASIS = ASSETS/'native_assets/positional_basis.pt'
        modules[name] = module
    return modules['lab'], modules['plan_common'], modules['plan_report']


lab, pc, pr = load_helpers()


def initialize():
    pc.setup('cpu')
    torch.set_num_threads(2)
    assert lab.DEV.type == 'cpu' and torch.get_num_threads() == 2


def iou_form(purity, capture):
    denominator = purity + capture - purity*capture
    return np.divide(purity*capture, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def prefix_statistics(order, ends, c, back_f, back_b):
    """Objectives at k=0 and every complete equal-score group, including all."""
    n = len(order)
    counts = np.r_[0, ends+1]
    cs = np.r_[0., np.cumsum(c[order], dtype=np.float64)[ends]]
    fs = np.r_[0., np.cumsum(back_f[order], dtype=np.float64)[ends]]
    bs = np.r_[0., np.cumsum(back_b[order], dtype=np.float64)[ends]]
    a_f = np.divide(cs, counts, out=np.zeros_like(cs), where=counts > 0)
    a_b = np.divide((n-counts) - (cs[-1]-cs), n-counts,
                    out=np.zeros_like(cs), where=counts < n)
    cap_f = np.clip(fs, 0., 1.)
    cap_b = np.clip(bs[-1]-bs, 0., 1.)
    jf = iou_form(np.clip(a_f, 0., 1.), cap_f)
    jb = iou_form(np.clip(a_b, 0., 1.), cap_b)
    return counts, jf, .5*(jf+jb), jb


def choose_prefix(score, order, counts, objective, jf, jb):
    # np.argmax is stable; ascending prefix sizes break exact objective ties
    # toward the smallest foreground. Never split an equal-score group.
    pick = int(np.argmax(objective))
    count = int(counts[pick])
    if count == len(order):
        threshold = float(np.nextafter(np.float64(score.min()), -np.inf))
    else:
        threshold = float(score[order[count]])
    # Cast before comparing so the full-set nextafter threshold remains below
    # the minimum when the stored score dtype is float32.
    pred = score.astype(np.float64) > threshold
    assert int(pred.sum()) == count, ('unrealizable strict threshold', count, int(pred.sum()))
    return pred.reshape(1024, 1024), {
        'strict_threshold': threshold, 'selected_pixel_count': count,
        'objective': float(objective[pick]), 'JF': float(jf[pick]), 'JB': float(jb[pick]),
        'realizable_prefixes': int(len(counts)), 'tie_rule': 'smallest_foreground_pixel_count',
    }


def mass1024(value):
    pixels = lab.up(torch.as_tensor(value, dtype=torch.float32).reshape(64, 64)).numpy().ravel().astype(np.float64)
    assert np.isfinite(pixels).all() and float(pixels.min()) >= 0
    total = float(pixels.sum())
    if total:
        pixels /= total
    return pixels, total


def mathematical_check():
    """Tiny complete enumeration checks ties, empty/full sets and both J terms."""
    s = np.array([3, 3, 2, 1, 1, 0], np.float32)
    c = np.array([.95, .75, .4, .6, .1, 0.], np.float64)
    f = np.array([.4, .25, .1, .2, .05, 0.], np.float64)
    b = np.array([0., .05, .1, .1, .3, .45], np.float64)
    order = np.argsort(-s, kind='stable')
    sorted_s = s[order]
    ends = np.r_[np.flatnonzero(sorted_s[:-1] != sorted_s[1:]), len(s)-1]
    counts, jf, jfb, jb = prefix_statistics(order, ends, c, f, b)
    for k, vf, vfb, vb in zip(counts, jf, jfb, jb):
        sel = np.zeros(len(s), bool)
        sel[order[:k]] = True
        af = c[sel].mean() if sel.any() else 0.
        ab = (1-c[~sel]).mean() if (~sel).any() else 0.
        scalar = lambda p, r: p*r/(p+r-p*r) if p+r-p*r > 0 else 0.
        ef, eb = scalar(af, f[sel].sum()), scalar(ab, b[~sel].sum())
        np.testing.assert_allclose((vf, vfb, vb), (ef, .5*(ef+eb), eb), rtol=1e-12, atol=1e-12)
    return {'state': 'PASS', 'fixed_toy_pixels': len(s), 'complete_group_counts': counts.tolist(),
            'checks': ['whole_tied_groups', 'empty_and_full_prefix', 'foreground_J', 'complement_background_J']}


def prepare():
    for folder in ('records', 'predictions', 'frozen'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    if not (OUT/'config.json').exists():
        dependencies = [R1/'sealed.json', R1/'manifest.json', R1/'config.json', P0/'sealed.json']
        sources = [Path(__file__), *(RUNTIME/(n+'.py') for n in ('lab', 'plan_common', 'plan_report'))]
        spec = {
            'state': 'FROZEN_BEFORE_NATURAL_OUTPUTS', 'n': 1360, 'arms': ARMS,
            'reference_count': 1, 'new_encoder_calls': 0, 'query_GT_in_inference': False,
            'FoRIS_output_in_inference': False, 'domain_branch': False,
            'ranking': 'R1 VP_UN float32 continuous 64x64 field; unchanged bilinear1024 align_corners=False',
            'UN_fixed': 'bitwise copy R1 predictions/VP_UN',
            'fwd': 'bilinear1024(P0 c32); no minmax; foreground=c, background=1-c',
            'backward': 'bilinear1024(P0 FP16 back_f/back_b cast FP32); each normalized to unit pixel sum; zero mass remains zero',
            'JF': 'mean_S(c)*sum_S(back_f)/(mean_S(c)+sum_S(back_f)-product); empty foreground J=0',
            'JFB': '.5*(JF(S)+JB(complement S)); JB uses 1-c and back_b; empty background J=0',
            'candidate_sets': 'all exact 1024 pixel strict level sets; complete equal-score groups; empty/full included; no threshold grid',
            'tie_rule': 'first maximum among ascending complete prefix sizes, hence smallest foreground; strict threshold is largest excluded score; full prefix uses float64 nextafter(min,-inf)',
            'permutation': 'same deterministic permutation of 4096 spatial indices for back_f and back_b before interpolation; seed=int(SHA256(salt+NUL+index)[:16],16)',
            'permutation_salt': SALT,
            'fallback': 'reference rho>.9: every arm bitwise reuses R1 VP_UN; record XOR against P0 MPS mask/T without changing R1 CPU replay output',
            'resources': {'device': 'cpu', 'maximum_workers': 2, 'threads_per_worker': 2},
            'runtime_root_override': {'lab.ROOT': str(ROOT), 'lab.ASSETS': str(ASSETS)},
            'runtime_source_provenance': {str(RUNTIME/'lab.py'): str(R1/'frozen/3_lab.py'),
                                          str(RUNTIME/'plan_common.py'): str(R1/'frozen/1_plan_common.py'),
                                          str(RUNTIME/'plan_report.py'): str(R1/'frozen/2_plan_report.py')},
            'source_sha256': {str(p): pc.sha(p) for p in sources},
            'dependency_sha256': {str(p): pc.sha(p) for p in dependencies},
            'mathematical_check': mathematical_check(),
        }
        shutil.copyfile(Path(__file__), OUT/'frozen/r4_extent.py')
        shutil.copyfile(R1/'manifest.json', OUT/'manifest.json')
        pc.write(OUT/'config.json', spec)
        (OUT/'SPEC.md').write_text(
            '# R4 frozen extent experiment\n\n'
            '1360 fixed R1 development episodes; one reference; cached fields only. '
            'UN_fixed copies R1 VP_UN; UN_JF maximizes foreground bidirectional J; '
            'UN_JFB maximizes equally weighted foreground/complement-background J; '
            'UN_JFB_permute repeats JFB with the shared SHA-seeded backward-mass permutation.\n\n'
            'The ranking remains the bilinear1024 continuous VP_UN field. Evaluate every realizable '
            'strict level set, including complete tied groups and empty/full masks; exact objective ties '
            'select the smallest foreground. P0 c32 is interpolated as purity; each interpolated backward '
            'mass is normalized to sum one. Rho>.9 reuses the original T fallback for all arms. '
            'No threshold/temperature/lambda search. All natural masks are sealed before query GT scoring.\n\n'
            'CPU only: maximum two workers, two threads each. Helpers are byte-identical R1 frozen '
            'sources copied into this task runtime; configuration records paths, SHA256 and relocated '
            'lab root overrides. FoRIS/MEAN/old VP enter scoring only.\n', encoding='utf-8')
    config = pc.read(OUT/'config.json')
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'R4 source drift'
    assert all(pc.sha(p) == h for p, h in config['dependency_sha256'].items()), 'R4 dependency drift'
    rows = pc.read(OUT/'manifest.json')
    assert len(rows) == 1360 and pc.sha(OUT/'manifest.json') == pc.sha(R1/'manifest.json')
    return rows


def produce_one(binding):
    name = pc.key(binding)
    receipt_path = OUT/'records'/f'{name}.json'
    if receipt_path.exists():
        old = pc.read(receipt_path)
        assert pc.sha(old['prediction_path']) == old['prediction_sha256']
        return old
    started = time.monotonic()
    with pc.inference_only(binding):
        r1 = pc.read(R1/'records'/f'{name}.json')
        p0 = pc.read(P0/'records'/f'{name}.json')
        assert pc.sha(r1['field_path']) == r1['field_sha256']
        assert pc.sha(r1['prediction_path']) == r1['prediction_sha256']
        assert pc.sha(p0['path']) == p0['sha256']
        with np.load(r1['field_path'], allow_pickle=False) as z:
            field = z['VP_UN'].copy()
            rho = float(z['reference_rho'][0])
        with np.load(r1['prediction_path'], allow_pickle=False) as z:
            fixed = z['VP_UN'].copy()
        with np.load(p0['path'], allow_pickle=False) as z:
            fields = {k: z[k].copy() for k in ('c32', 'back_f', 'back_b', 'ref_stats', 'mask/T')}
        assert rho == float(fields['ref_stats'][2])
        assert np.array_equal(pc.pack(pc.readout(field)), fixed), 'UN_fixed field/mask inconsistency'
        predictions = {'UN_fixed': fixed}
        info, mass_info = {}, {}
        fallback_p0_xor = int(np.unpackbits(fixed ^ fields['mask/T']).sum()) if rho > .9 else None
        if rho > .9:
            for arm in ARMS[1:]:
                predictions[arm] = fixed.copy()
                info[arm] = {'raw_T_fallback': True, 'strict_threshold': None,
                             'selected_pixel_count': int(pc.unpack(fixed).sum())}
        else:
            score = lab.up(torch.as_tensor(field, dtype=torch.float32)).numpy().ravel()
            assert score.size == 1024*1024 and np.isfinite(score).all()
            c = lab.up(torch.as_tensor(fields['c32'], dtype=torch.float32).reshape(64, 64)).numpy().ravel().astype(np.float64)
            assert np.isfinite(c).all() and float(c.min()) >= 0 and float(c.max()) <= 1
            order = np.argsort(-score, kind='stable')
            sorted_s = score[order]
            ends = np.r_[np.flatnonzero(sorted_s[:-1] != sorted_s[1:]), len(score)-1]
            bf, total_f = mass1024(fields['back_f'])
            bb, total_b = mass1024(fields['back_b'])
            counts, jf, jfb, jb = prefix_statistics(order, ends, c, bf, bb)
            for arm, obj in (('UN_JF', jf), ('UN_JFB', jfb)):
                pred, detail = choose_prefix(score, order, counts, obj, jf, jb)
                predictions[arm], info[arm] = pc.pack(pred), detail
            seed = int(hashlib.sha256((SALT+'\0'+name).encode()).hexdigest()[:16], 16)
            permutation = np.random.default_rng(seed).permutation(4096)
            pbf, ptf = mass1024(fields['back_f'].reshape(-1)[permutation])
            pbb, ptb = mass1024(fields['back_b'].reshape(-1)[permutation])
            pcounts, pjf, pjfb, pjb = prefix_statistics(order, ends, c, pbf, pbb)
            pred, detail = choose_prefix(score, order, pcounts, pjfb, pjf, pjb)
            predictions['UN_JFB_permute'], info['UN_JFB_permute'] = pc.pack(pred), dict(detail, seed=seed)
            mass_info = {'upsampled_back_f_sum_before_normalization': total_f,
                         'upsampled_back_b_sum_before_normalization': total_b,
                         'permuted_f_sum_before_normalization': ptf,
                         'permuted_b_sum_before_normalization': ptb}
        path = OUT/'predictions'/f'{name}.npz'
        assert not path.exists(), 'Preserve orphaned R4 prediction: ' + str(path)
        np.savez_compressed(path, **predictions)
    record = {
        'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
        'prediction_path': str(path), 'prediction_sha256': pc.sha(path),
        'R1_field_path': r1['field_path'], 'R1_field_sha256': r1['field_sha256'],
        'R1_prediction_sha256': r1['prediction_sha256'], 'P0_field_sha256': p0['sha256'],
        'reference_rho': rho, 'fallback': rho > .9, 'UN_fixed_R1_mask_xor': 0,
        'R1_CPU_fallback_vs_P0_MPS_T_xor': fallback_p0_xor,
        'extent': info, 'mass': mass_info, 'seconds': time.monotonic()-started,
        'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'FoRIS_output_consumed': False,
    }
    pc.write(receipt_path, record)
    return record


def produce(workers):
    initialize()
    rows = prepare()
    if (OUT/'sealed.json').exists():
        print({'phase': 'R4-produce', 'state': 'ALREADY_SEALED'}, flush=True)
        return
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, receipt in enumerate(pool.map(produce_one, rows, chunksize=1), 1):
            records.append(receipt)
            if i <= 3 or i % 50 == 0:
                print({'phase': 'R4-produce', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-started, 1)}, flush=True)
    pc.write(OUT/'sealed.json', {
        'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows), 'records': records,
        'manifest_sha256': pc.sha(OUT/'manifest.json'), 'config_sha256': pc.sha(OUT/'config.json'),
        'wall_seconds': time.monotonic()-started, 'case_seconds_sum': sum(r['seconds'] for r in records),
        'workers': workers, 'threads_per_worker': 2, 'device': 'cpu',
        'fallback_n': sum(r['fallback'] for r in records),
        'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
    })
    print({'phase': 'R4-produce', 'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows)}, flush=True)


def score_one(job):
    binding, receipt = job
    assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
    with np.load(receipt['prediction_path'], allow_pickle=False) as z:
        predictions = {a: pc.unpack(z[a]) for a in ARMS}
    return pr.score_record(binding, predictions)


def score(workers):
    initialize()
    rows = prepare()
    seal = pc.read(OUT/'sealed.json')
    assert seal['state'] == 'ALL_NATURAL_OUTPUTS_SEALED' and seal['n'] == len(rows)
    assert pc.sha(OUT/'config.json') == seal['config_sha256']
    assert pc.sha(OUT/'manifest.json') == seal['manifest_sha256']
    receipts = {r['index']: r for r in seal['records']}
    assert len(receipts) == len(rows)
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, record in enumerate(pool.map(score_one, ((b, receipts[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'R4-score', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-started, 1)}, flush=True)
    prior = {r['index']: r for r in pc.read(R1/'report.json')['records']}
    for r in records:
        old = prior[r['index']]
        assert r['query_photo'] == old['query_photo']
        assert r['metrics']['UN_fixed'] == old['metrics']['VP_UN'], 'R1 score identity failure'
        for arm in CONTROLS:
            r['metrics'][arm] = old['metrics'][arm]
    notes = [
        '固定R1的1360例单参考开发任务；只读R1 VP_UN连续场和P0合法字段，新增编码0、无域分支、推理不消费FoRIS输出。',
        '四臂全部自然预测封存后才评分；UN_fixed逐位复用R1 VP_UN，评分整数与R1完全相同。',
        '沿bilinear1024 VP_UN排序，在所有可实现严格水平集上最大化J：同分组不拆分，包含空/全图，目标同分取最小前景；没有阈值网格。',
        'JF使用bilinear1024(c32)的前景纯度与归一化back_f捕获率；JFB对前景JF和补集背景JB等权平均，JB使用1-c和back_b。',
        '置换臂用SHA256固定种子共同置换back_f/back_b的64网格空间位置，再插值；其他规则相同。rho>.9时全部臂逐位复用R1冻结的原T回退。',
        'R1的CPU原T重放与P0的MPS原T在Fundus两例各有1像素差异；R4全部保留R1冻结输出。初始P0逐位相等断言已在评分前修正为记录差异，未改变推理结果或参数，原config/源码与修复回执保留。',
        'CPU最多2 worker，每worker2线程；独立runtime复用R1封存共享模块原字节，实际来源/路径/SHA及ROOT覆盖均记录在config.json。',
    ]
    report = pr.report('R4_extent', OUT, records, CONTROLS+ARMS, notes,
                       extra={'production': {k: v for k, v in seal.items() if k != 'records'},
                              'score_seconds_before_summary': time.monotonic()-started,
                              'R1_controls_report_sha256': pc.sha(R1/'report.json')})
    contrasts = {}
    for baseline in ('UN_fixed', 'UN_JFB_permute'):
        projected = [dict(r, metrics={'FoRIS': r['metrics'][baseline],
                                     'UN_JF': r['metrics']['UN_JF'],
                                     'UN_JFB': r['metrics']['UN_JFB']}) for r in records]
        contrasts['versus_'+baseline] = pr.summary(projected, ('FoRIS', 'UN_JF', 'UN_JFB'))
    pc.write(OUT/'paired_contrasts.json', contrasts)
    lines = ['\n## R4 配对机制对照\n',
             '|域|方法|对照|class差值pp|query照片配对95%区间|',
             '|---|---|---|---:|---|']
    for contrast, table in contrasts.items():
        for domain, arms in table.items():
            for arm in ('UN_JF', 'UN_JFB'):
                v = arms[arm]
                ci = v['paired_query_photo_bootstrap95_pp']
                lines.append(f"|{domain}|{arm}|{contrast.removeprefix('versus_')}|{v['delta_to_foris_pp']:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|")
    report_path = ROOT/'docs/research/x1012_tasks/R4_extent.md'
    with report_path.open('a') as f:
        f.write('\n'.join(lines)+'\n')
    rank = sorted(ARMS, key=lambda a: (min(t[a]['delta_to_foris_pp'] for t in report['summary'].values()),
                                     np.mean([t[a]['delta_to_foris_pp'] for t in report['summary'].values()])), reverse=True)
    pc.write(OUT/'completion.json', {
        'state': 'COMPLETE', 'n': len(rows), 'candidate_rank': rank,
        'ten_domain_nonnegative_arms': [a for a in ARMS if all(t[a]['delta_to_foris_pp'] >= 0 for t in report['summary'].values())],
        'ten_domain_minus05_gate_arms': [a for a in ARMS if all(t[a]['delta_to_foris_pp'] >= -.5 for t in report['summary'].values())],
        'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'UN_fixed_R1_score_identity': True,
        'score_seconds_total': time.monotonic()-started,
        'report_path': str(report_path), 'report_sha256': pc.sha(OUT/'report.json'),
        'contrasts_sha256': pc.sha(OUT/'paired_contrasts.json'),
    })
    print({'phase': 'R4-complete', 'n': len(rows), 'rank': rank,
           'seconds': round(time.monotonic()-started, 1)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    initialize()
    if args.mode == 'freeze':
        rows = prepare()
        print({'state': 'FROZEN', 'n': len(rows), 'config': str(OUT/'config.json')}, flush=True)
    elif args.mode == 'produce':
        produce(args.workers)
    else:
        score(args.workers)
