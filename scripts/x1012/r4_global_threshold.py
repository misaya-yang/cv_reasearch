"""One fixed global-threshold experiment on the new R1 VP_UN field.

Nine thresholds, 1360 existing one-reference episodes, zero new encoding.
Every natural mask is sealed before GT scoring. No per-domain inference rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
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
OUT = ASSETS / 'a/x1012_R4_global'
R1 = ASSETS / 'a/x1012_R1'
RUNTIME = OUT / 'runtime'
TAUS = (.40, .425, .45, .475, .50, .525, .55, .575, .60)
ARMS = tuple(f'UN_tau{round(t * 1000):03d}' for t in TAUS)
FIXED = 'UN_tau500'
CONTROLS = ('FoRIS', 'MEAN')


def load_helpers():
    """Use byte-identical R1 helpers without changing concurrent source files."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    sources = {'lab': R1/'frozen/3_lab.py',
               'plan_common': R1/'frozen/1_plan_common.py',
               'plan_report': R1/'frozen/2_plan_report.py'}
    modules = {}
    for name, source in sources.items():
        dest = RUNTIME/(name + '.py')
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert source.read_bytes() == dest.read_bytes(), 'Runtime helper drift: ' + name
        spec = importlib.util.spec_from_file_location(name, dest)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'lab':
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


def prepare():
    for folder in ('records', 'predictions', 'frozen'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    if not (OUT/'config.json').exists():
        dependencies = [R1/'sealed.json', R1/'manifest.json', R1/'config.json']
        sources = [Path(__file__), *(RUNTIME/(n+'.py') for n in ('lab', 'plan_common', 'plan_report'))]
        config = {
            'state': 'FROZEN_BEFORE_NATURAL_OUTPUTS', 'n': 1360,
            'hypothesis': 'R1 VP_UN is positive on 8/10 domains but misses COCO and ISIC; R4 bidirectional extent objectives failed. Test whether a single scalar on this new field removes both deficits without losing other domains.',
            'positive_decision': 'Freeze the globally selected scalar before first new-500 scoring; independent confirmation is still required.',
            'negative_decision': 'Reject this fixed scalar family as a solution to the ten-domain gap; do not extend the grid or deploy per-domain best thresholds.',
            'taus': TAUS, 'arms': dict(zip(ARMS, TAUS)),
            'field': 'R1 VP_UN float32 64x64',
            'readout': 'torch bilinear1024 align_corners=False, unchanged lab.minmax, strict > tau',
            'fallback': 'reference rho>.9: bitwise R1 VP_UN raw-T fallback for every tau, unchanged by calibration',
            'fixed_identity': 'recompute tau=.5 and require packed masks identical to R1 VP_UN for all 1360 before sealing',
            'reference_count': 1, 'new_encoder_calls': 0,
            'query_GT_in_inference': False, 'FoRIS_output_in_inference': False,
            'domain_branch': False, 'grid_extension_allowed': False,
            'selection': 'maximize minimum domain class-mIoU delta to FoRIS; then unweighted mean domain delta; then closest tau to .5; then smaller tau as deterministic residual tie-break',
            'lodo': 'repeat identical selection on nine domains; report only held-domain result; diagnostic, not a per-domain deployment rule',
            'scoring': 'all 1360 x 9 masks sealed before query GT; frozen R1 FoRIS and MEAN metrics reused only in scoring',
            'resources': {'device': 'cpu', 'maximum_workers': 2, 'threads_per_worker': 2},
            'runtime_root_override': {'lab.ROOT': str(ROOT), 'lab.ASSETS': str(ASSETS)},
            'runtime_source_provenance': {str(RUNTIME/(n+'.py')): str(p) for n, p in {
                'lab': R1/'frozen/3_lab.py', 'plan_common': R1/'frozen/1_plan_common.py',
                'plan_report': R1/'frozen/2_plan_report.py'}.items()},
            'source_sha256': {str(p): pc.sha(p) for p in sources},
            'dependency_sha256': {str(p): pc.sha(p) for p in dependencies},
        }
        shutil.copyfile(Path(__file__), OUT/'frozen/r4_global_threshold.py')
        shutil.copyfile(R1/'manifest.json', OUT/'manifest.json')
        pc.write(OUT/'config.json', config)
    config = pc.read(OUT/'config.json')
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'Source drift'
    assert all(pc.sha(p) == h for p, h in config['dependency_sha256'].items()), 'Dependency drift'
    rows = pc.read(OUT/'manifest.json')
    assert len(rows) == 1360 and pc.sha(OUT/'manifest.json') == pc.sha(R1/'manifest.json')
    return rows


def produce_one(binding):
    name = pc.key(binding)
    receipt_path = OUT/'records'/f'{name}.json'
    if receipt_path.exists():
        receipt = pc.read(receipt_path)
        assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
        return receipt
    started = time.monotonic()
    with pc.inference_only(binding):
        source = pc.read(R1/'records'/f'{name}.json')
        assert pc.sha(source['field_path']) == source['field_sha256']
        assert pc.sha(source['prediction_path']) == source['prediction_sha256']
        with np.load(source['field_path'], allow_pickle=False) as z:
            field, rho = z['VP_UN'].copy(), float(z['reference_rho'][0])
        with np.load(source['prediction_path'], allow_pickle=False) as z:
            fixed = z['VP_UN'].copy()
        score = lab.minmax(lab.up(torch.as_tensor(field, dtype=torch.float32))).numpy()
        assert score.shape == (1024, 1024) and np.isfinite(score).all()
        replay = pc.pack(score > .5)
        assert np.array_equal(replay, fixed), (name, 'tau=.5 differs from R1')
        predictions = {arm: fixed.copy() if rho > .9 else pc.pack(score > tau)
                       for arm, tau in zip(ARMS, TAUS)}
        assert np.array_equal(predictions[FIXED], fixed)
        path = OUT/'predictions'/f'{name}.npz'
        assert not path.exists(), 'Preserve orphaned prediction: ' + str(path)
        np.savez_compressed(path, **predictions)
    receipt = {
        'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
        'prediction_path': str(path), 'prediction_sha256': pc.sha(path),
        'R1_field_path': source['field_path'], 'R1_field_sha256': source['field_sha256'],
        'R1_prediction_sha256': source['prediction_sha256'],
        'reference_rho': rho, 'fallback': rho > .9, 'tau500_R1_mask_xor': 0,
        'seconds': time.monotonic()-started, 'new_encoder_calls': 0,
        'inference_query_GT_reads': 0, 'FoRIS_output_consumed': False,
    }
    pc.write(receipt_path, receipt)
    return receipt


def produce(workers):
    initialize()
    rows = prepare()
    if (OUT/'sealed.json').exists():
        print({'phase': 'R4-global-produce', 'state': 'ALREADY_SEALED'}, flush=True)
        return
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, receipt in enumerate(pool.map(produce_one, rows, chunksize=1), 1):
            records.append(receipt)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'R4-global-produce', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-started, 1)}, flush=True)
    pc.write(OUT/'sealed.json', {
        'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows), 'mask_n': len(rows)*len(ARMS),
        'records': records, 'manifest_sha256': pc.sha(OUT/'manifest.json'),
        'config_sha256': pc.sha(OUT/'config.json'), 'wall_seconds': time.monotonic()-started,
        'case_seconds_sum': sum(r['seconds'] for r in records),
        'workers': workers, 'threads_per_worker': 2, 'device': 'cpu',
        'fallback_n': sum(r['fallback'] for r in records),
        'tau500_R1_mask_xor': sum(r['tau500_R1_mask_xor'] for r in records),
        'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
    })
    print({'phase': 'R4-global-produce', 'state': 'ALL_NATURAL_OUTPUTS_SEALED',
           'n': len(rows), 'mask_n': len(rows)*len(ARMS)}, flush=True)


def score_one(job):
    binding, receipt = job
    assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
    with np.load(receipt['prediction_path'], allow_pickle=False) as z:
        predictions = {a: pc.unpack(z[a]) for a in ARMS}
    return pr.score_record(binding, predictions)


def ranked(summary, domains):
    def objective(arm):
        deltas = [summary[d][arm]['delta_to_foris_pp'] for d in domains]
        tau = TAUS[ARMS.index(arm)]
        return (min(deltas), float(np.mean(deltas)), -abs(tau-.5), -tau)
    return sorted(ARMS, key=objective, reverse=True)


def score(workers):
    initialize()
    rows = prepare()
    seal = pc.read(OUT/'sealed.json')
    assert seal['state'] == 'ALL_NATURAL_OUTPUTS_SEALED' and seal['n'] == len(rows)
    assert seal['mask_n'] == len(rows)*len(ARMS)
    assert pc.sha(OUT/'config.json') == seal['config_sha256']
    assert pc.sha(OUT/'manifest.json') == seal['manifest_sha256']
    receipts = {r['index']: r for r in seal['records']}
    assert len(receipts) == len(rows)
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, record in enumerate(pool.map(score_one, ((b, receipts[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 100 == 0:
                print({'phase': 'R4-global-score', 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic()-started, 1)}, flush=True)
    previous = {r['index']: r for r in pc.read(R1/'report.json')['records']}
    for record in records:
        old = previous[record['index']]
        assert record['query_photo'] == old['query_photo']
        assert record['metrics'][FIXED] == old['metrics']['VP_UN'], 'R1 metric identity failure'
        for arm in CONTROLS:
            record['metrics'][arm] = old['metrics'][arm]
    report = pr.report('R4_global', OUT, records, CONTROLS+ARMS, [
        '动机：R1新VP_UN场八域正、COCO/ISIC负；R4双向J范围目标失败。这里只检验该新场的一个全局标量，不以旧IND0阈值失败替代实测。',
        '固定1360单参考开发任务，阈值仅为.40/.425/.45/.475/.50/.525/.55/.575/.60；不扩网格、不按域分支、新增编码0、推理不消费FoRIS输出。',
        '同R1 bilinear1024后minmax及strict>tau；rho>.9的原T回退逐位保留R1，不受tau影响。tau=.5逐位及评分整数完全复现R1。',
        '12240个自然掩码全部封存后才读GT。FoRIS/MEAN逐例指标仅在评分阶段复用R1；CPU 2 worker x 2线程。',
        '唯一全局阈值按十域最差class差值最大、再域均值最大、再距.5最近选择。留一域仅用九域选择，报告被留域；不是按域部署规则。',
        '全局选择使用本开发集GT；所报配对区间是固定臂的描述性区间，不校正九候选选择偏差。新500首次评分前须冻结选择，不能以本表证明独立确认成功。',
    ], extra={'production': {k: v for k, v in seal.items() if k != 'records'},
              'score_seconds_before_summary': time.monotonic()-started,
              'R1_controls_report_sha256': pc.sha(R1/'report.json')})
    projected = [dict(r, metrics={'FoRIS': r['metrics'][FIXED], **{a: r['metrics'][a] for a in ARMS}})
                 for r in records]
    versus_fixed = pr.summary(projected, ('FoRIS',)+ARMS)
    pc.write(OUT/'paired_to_fixed.json', versus_fixed)
    domains = list(report['summary'])
    rank = ranked(report['summary'], domains)
    selected = rank[0]
    lodo = {}
    for held in domains:
        train = [d for d in domains if d != held]
        arm = ranked(report['summary'], train)[0]
        lodo[held] = {'held_domain': held, 'selection_domains': train,
                      'selected_arm': arm, 'selected_tau': TAUS[ARMS.index(arm)],
                      'held_result': report['summary'][held][arm],
                      'held_versus_fixed': versus_fixed[held][arm]}
    pc.write(OUT/'lodo.json', lodo)
    selected_tau = TAUS[ARMS.index(selected)]
    selected_stats = {d: {'versus_FoRIS': report['summary'][d][selected],
                          'versus_fixed_05': versus_fixed[d][selected]} for d in domains}
    completion = {
        'state': 'COMPLETE', 'n': len(rows), 'mask_n': len(rows)*len(ARMS),
        'selected_arm': selected, 'selected_tau': selected_tau, 'candidate_rank': rank,
        'selected_worst_domain_delta_pp': min(report['summary'][d][selected]['delta_to_foris_pp'] for d in domains),
        'selected_mean_domain_delta_pp': float(np.mean([report['summary'][d][selected]['delta_to_foris_pp'] for d in domains])),
        'ten_domain_nonnegative_arms': [a for a in ARMS if all(report['summary'][d][a]['delta_to_foris_pp'] >= 0 for d in domains)],
        'ten_domain_minus05_gate_arms': [a for a in ARMS if all(report['summary'][d][a]['delta_to_foris_pp'] >= -.5 for d in domains)],
        'selected_domain_results': selected_stats,
        'lodo_nonnegative_domains': [d for d, v in lodo.items() if v['held_result']['delta_to_foris_pp'] >= 0],
        'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
        'tau500_R1_mask_xor': seal['tau500_R1_mask_xor'], 'tau500_R1_score_identity': True,
        'fallback_n': seal['fallback_n'], 'score_seconds_total': time.monotonic()-started,
        'config_sha256': pc.sha(OUT/'config.json'), 'seal_sha256': pc.sha(OUT/'sealed.json'),
        'manifest_sha256': pc.sha(OUT/'manifest.json'), 'report_sha256': pc.sha(OUT/'report.json'),
        'paired_to_fixed_sha256': pc.sha(OUT/'paired_to_fixed.json'), 'lodo_sha256': pc.sha(OUT/'lodo.json'),
        'prediction_pattern': str(OUT/'predictions/{index}.npz'),
    }
    pc.write(OUT/'completion.json', completion)
    lines = ['', '## 全局一次选择与留一域诊断', '',
             f'唯一全局选择 **tau={selected_tau:g}**，最差域对FoRIS **{completion["selected_worst_domain_delta_pp"]:+.3f}pp**，十域等权平均差 **{completion["selected_mean_domain_delta_pp"]:+.3f}pp**。',
             f'全部十域非负候选：{completion["ten_domain_nonnegative_arms"]}；全部十域至少−0.5pp候选：{completion["ten_domain_minus05_gate_arms"]}。', '',
             '|域|全局tau对FoRIS pp [95% CI]|全局tau对固定.5 pp [95% CI]|LODO九域选tau|被留域对FoRIS pp [95% CI]|',
             '|---|---|---|---:|---|']
    def delta_cell(v):
        lo, hi = v['paired_query_photo_bootstrap95_pp']
        return f"{v['delta_to_foris_pp']:+.3f} [{lo:+.3f},{hi:+.3f}]"
    for d in domains:
        lines.append(f"|{d}|{delta_cell(report['summary'][d][selected])}|{delta_cell(versus_fixed[d][selected])}|{lodo[d]['selected_tau']:g}|{delta_cell(lodo[d]['held_result'])}|")
    lines += ['', '|tau|十域最差delta pp|十域均值delta pp|非负域数|', '|---:|---:|---:|---:|']
    for arm, tau in zip(ARMS, TAUS):
        ds = [report['summary'][d][arm]['delta_to_foris_pp'] for d in domains]
        lines.append(f'|{tau:g}|{min(ds):+.3f}|{np.mean(ds):+.3f}|{sum(v >= 0 for v in ds)}/10|')
    lines += ['', f'固定.5全1360逐位XOR总数={seal["tau500_R1_mask_xor"]}；rho>.9固定回退{seal["fallback_n"]}例。',
              f'配置 SHA256：`{completion["config_sha256"]}`。',
              f'12240掩码封存清单 SHA256：`{completion["seal_sha256"]}`。',
              f'逐例掩码：`{completion["prediction_pattern"]}`，选择键 `{selected}`；各文件SHA在 `sealed.json`。',
              f'对固定.5配对区间：`{OUT/"paired_to_fixed.json"}`；留一域：`{OUT/"lodo.json"}`。']
    with (ROOT/'docs/research/x1012_tasks/R4_global.md').open('a') as f:
        f.write('\n'.join(lines)+'\n')
    print({'phase': 'R4-global-complete', 'selected_tau': selected_tau,
           'worst_delta_pp': completion['selected_worst_domain_delta_pp'],
           'nonnegative': completion['ten_domain_nonnegative_arms'],
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
