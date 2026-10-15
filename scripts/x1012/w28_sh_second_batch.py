"""W28: repeat the frozen W27 six-arm ablation on disjoint S query photos.

Preregistered in W28_sh_second_batch_PLAN.md. No method changes or seed search:
exclude W27 tasks and query photos, then cap each remaining domain at 150 with
seed 20261014. This gives 871 tasks in six domains. Four exhausted S domains
are explicitly absent; no old examples, T, or new-photo holdout fill the gap.

Reuse w27.infer without editing it. All six masks are sealed before scoring.
Compare the new six-domain aggregate with W27 restricted to those same six
domains. Both batches are development measurements, not clean holdouts.

    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w28_sh_second_batch.py produce [workers]
    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w28_sh_second_batch.py check [workers]
    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w28_sh_second_batch.py report
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import w27_sh_single_line as w

c, ARMS = w.c, w.ARMS
OUT = c.task_path('W28')
PLAN = ROOT / 'docs/research/x1012_tasks/W28_sh_second_batch_PLAN.md'
REPORT = ROOT / 'docs/research/x1012_tasks/W28_sh_second_batch_REPORT.md'
EVIDENCE = ROOT / 'evidence/research/w28_sh_second_batch_S.jsonl.gz'
SAMPLE_SEED = 20261014
EXPECTED = dict(zip(c.DOMAINS, (150, 150, 150, 150, 150, 0, 121, 0, 0, 0)))
COMMON = tuple(d for d in c.DOMAINS if EXPECTED[d])
KEYS = ('SH-U0', 'SH-Mprime', 'SH+-U0+', 'SH+-Mprime+')


def save(name, value):
    c.write(OUT / name, value)


def load(name):
    return c.read(OUT / name)


def reference_photo(binding):
    return str(binding['row'].get('reference_photo_id') or binding['row']['reference_path'])


def sample():
    old = w.sample()
    old_ids = {str(b['index']) for b in old}
    old_queries = {c.photo(b) for b in old}
    old_references = {reference_photo(b) for b in old}
    s = [b for b in c.rows() if c.half(b) == 'S']
    rng, rows, coverage = np.random.default_rng(SAMPLE_SEED), [], {}
    for d in c.DOMAINS:
        remaining = [b for b in s if b['dataset'] == d and str(b['index']) not in old_ids]
        pool = sorted((b for b in remaining if c.photo(b) not in old_queries), key=lambda b: str(b['index']))
        chosen = [pool[i] for i in sorted(rng.choice(len(pool), min(150, len(pool)), replace=False))]
        rows.extend(chosen)
        coverage[d] = {'remaining_tasks': len(remaining), 'available_after_query_exclusion': len(pool),
                       'selected': len(chosen), 'query_photos': len({c.photo(b) for b in chosen}),
                       'old_query_overlap': len({c.photo(b) for b in chosen} & old_queries),
                       'queries_seen_as_old_reference': len({c.photo(b) for b in chosen} & old_references),
                       'references_overlapping_old_references': len({reference_photo(b) for b in chosen} & old_references)}
    counts = Counter(b['dataset'] for b in rows)
    assert {d: counts[d] for d in c.DOMAINS} == EXPECTED
    assert len(rows) == len({str(b['index']) for b in rows}) == 871
    assert not {str(b['index']) for b in rows} & old_ids
    assert not {c.photo(b) for b in rows} & old_queries
    return rows, coverage


def identity(rows, coverage):
    w.verify_frozen()
    old_checks = w.load('checks.json')
    assert old_checks['passed'] and old_checks['fusion_equal_W23'] == {'SH': 1285, 'SH+': 1285}
    return {'code_sha256': w.sha(__file__), 'plan_sha256': w.sha(PLAN), 'sample_seed': SAMPLE_SEED,
            'w27_code_sha256': w.sha(Path(w.__file__)),
            'w27_manifest_sha256': w.sha(w.OUT / 'manifest.json'),
            'w27_checks_sha256': w.sha(w.OUT / 'checks.json'),
            'w27_report_sha256': w.sha(w.OUT / 'report.json'),
            'indices': [str(b['index']) for b in rows], 'counts': EXPECTED, 'coverage': coverage,
            'bindings': {str(b['index']): hashlib.sha256(json.dumps(
                {'row': b['row'], 'fit': b['fit']}, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest() for b in rows},
            'method': 'exact unmodified w27.infer; exact frozen W23 simple_head',
            'split': 'S', 'T_read': False, 'new_photo_holdout_read': False}


def verify_frozen():
    m = load('manifest.json')
    assert m['code_sha256'] == w.sha(__file__)
    assert m['plan_sha256'] == w.sha(PLAN)
    assert m['w27_code_sha256'] == w.sha(Path(w.__file__))
    for name in ('manifest', 'checks', 'report'):
        assert m[f'w27_{name}_sha256'] == w.sha(w.OUT / f'{name}.json')
    w.verify_frozen()


def ordered(name):
    indexed = w.indexed_jsonl(OUT / name)
    ids = load('manifest.json')['indices']
    assert set(indexed) == set(ids), f'{name}: incomplete or different sample'
    return [indexed[i] for i in ids]


def predict_one(binding):
    import psutil
    masks, audit = w.infer(binding)
    path = OUT / 'predictions' / (c.key(binding) + '.npz')
    c.save_npz(path, **{a: c.pack(masks[a]) for a in ARMS})
    return dict(w.metadata(binding), **audit, prediction=str(path), payload_sha256=w.sha(path),
                worker_rss_mib=psutil.Process().memory_info().rss / 1024 ** 2)


def produce_all(workers):
    assert 1 <= workers <= 4
    OUT.mkdir(parents=True, exist_ok=True)
    rows, coverage = sample()
    m = identity(rows, coverage)
    if (OUT / 'manifest.json').exists():
        assert load('manifest.json') == m, 'Resume identity mismatch'
    else:
        save('manifest.json', m)
    stamp = time.time_ns()
    top = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True, check=True).stdout
    (OUT / f'top_{stamp}.txt').write_text(top)
    print(top, flush=True)
    path = OUT / 'predictions.jsonl'
    done = w.indexed_jsonl(path)
    for e in done.values():
        assert w.sha(e['prediction']) == e['payload_sha256']
    todo = [b for b in rows if str(b['index']) not in done]
    started, snapshots = time.perf_counter(), [{'completed': len(done), **w.resource()}]
    print(f'new S batch {len(rows)}, saved {len(done)}, todo {len(todo)}, workers {workers}, torch threads 2', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=w.initialize) as pool, path.open('a') as stream:
        for n, e in enumerate(pool.map(predict_one, todo, chunksize=1), 1):
            stream.write(json.dumps(e, allow_nan=False) + '\n')
            stream.flush()
            if n % 20 == 0 or n == len(todo):
                os.fsync(stream.fileno())
                snapshots.append({'completed': len(done)+n, **w.resource()})
                save(f'resources_{stamp}.json', {'workers': workers, 'torch_threads': 2, 'snapshots': snapshots})
                print(f'predicted {len(done)+n}/871, elapsed {time.perf_counter()-started:.1f}s, '
                      f'CPU {snapshots[-1]["cpu_percent"]:.1f}%, available {snapshots[-1]["available_gib"]:.2f} GiB', flush=True)
    verify_frozen()
    complete = ordered('predictions.jsonl')
    for e in complete:
        assert w.sha(e['prediction']) == e['payload_sha256']
    seal = {'n': len(complete), 'predictions_jsonl_sha256': w.sha(path),
            'payload_sha256': {e['index']: e['payload_sha256'] for e in complete},
            'sealed_unix_ns': time.time_ns(), 'T_read': False, 'new_photo_holdout_read': False}
    if (OUT / 'sealed.json').exists():
        assert load('sealed.json')['payload_sha256'] == seal['payload_sha256']
    else:
        save('sealed.json', seal)
    save(f'production_{stamp}.json', {'n': len(complete), 'produced_this_run': len(todo), 'workers': workers,
         'torch_threads': 2, 'wall_seconds': time.perf_counter()-started,
         'worker_rss_mib_max': max(e['worker_rss_mib'] for e in complete),
         'episode_seconds_mean': float(np.mean([e['sec']['total'] for e in complete]))})
    print('New batch sealed; no query GT opened. Run check.', flush=True)


def score_one(job):
    binding, prediction = job
    assert c.half(binding) == 'S'
    assert w.sha(prediction['prediction']) == prediction['payload_sha256']
    with np.load(prediction['prediction'], allow_pickle=False) as archive:
        masks = {a: c.unpack(archive[a]) for a in ARMS}
    if prediction['fallback']:
        assert all(np.array_equal(mask, masks['SH']) for mask in masks.values())
    gt, valid = c.truth(binding)
    return dict(prediction, gt=int(gt.sum()), **{a: c.iu(masks[a], gt, valid) for a in ARMS})


def check(workers):
    assert 1 <= workers <= 4
    verify_frozen()
    seal = load('sealed.json')
    assert seal['n'] == 871 and seal['predictions_jsonl_sha256'] == w.sha(OUT / 'predictions.jsonl')
    predicted = ordered('predictions.jsonl')
    for e in predicted:
        assert e['split'] == 'S' and e['encoder_calls'] == 0 and e['raw_requests'] == 2
        assert e['shared_graph'] and e['recomposed_xor_pixels'] == 0
        assert e['point_solve_calls'] == e['prototype_solve_calls'] == (0 if e['fallback'] else 1)
    rows, coverage = sample()
    assert identity(rows, coverage) == load('manifest.json')
    print('Frozen W27 inference, sample/query-photo isolation and per-case reconstruction audits PASS', flush=True)
    indexed = {e['index']: e for e in predicted}
    path = OUT / 'episodes.jsonl'
    done = w.indexed_jsonl(path)
    jobs = [(b, indexed[str(b['index'])]) for b in rows if str(b['index']) not in done]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=w.initialize) as pool, path.open('a') as stream:
        for n, e in enumerate(pool.map(score_one, jobs, chunksize=4), 1):
            stream.write(json.dumps(e, allow_nan=False)+'\n')
            stream.flush()
            if n % 200 == 0 or n == len(jobs):
                print(f'scored {len(done)+n}/871', flush=True)
    scored = ordered('episodes.jsonl')
    save('checks.json', {'passed': True, 'n': len(scored), 'old_task_overlap': 0,
         'old_query_photo_overlap': 0, 'reconstruction_xor_pixels': 0,
         'frozen_W27_reused': True, 'prior_W27_identity': {'SH': 1285, 'SH+': 1285},
         'episodes_sha256': w.sha(path), 'sealed_sha256': w.sha(OUT / 'sealed.json'),
         'scoring_wall_seconds': time.perf_counter()-started,
         'T_read': False, 'new_photo_holdout_read': False})
    keep = ('index', 'dataset', 'fold', 'cls', 'photo', 'split', 'gt', 'rho', 'fallback',
            'raw_requests', 'encoder_calls', 'point_solve_calls', 'prototype_solve_calls',
            'shared_graph', 'recomposed_xor_pixels', 'sec') + ARMS
    with gzip.open(EVIDENCE, 'wt') as stream:
        for e in scored:
            stream.write(json.dumps({k: e[k] for k in keep}, allow_nan=False)+'\n')
    print('Checks PASS; compact S evidence exported. Report is now allowed.', flush=True)


def write_report(r):
    full, active = r['new_full'], r['new_active']
    names = dict(zip(c.DOMAINS, c.DOMAIN_NAMES))
    lines = ['# W28：另一批 S 样本的最终 SH 单线消融', '', '2026-10-14。'
             '抽样与读法见 [预先固定的计划](W28_sh_second_batch_PLAN.md)。', '',
             '## 1 样本与实现检查', '',
             '新批871例，原批任务重合0、查询照片重合0。只在S余池中固定种子抽取，不使用T或新照片留出。'
             '这不是新的未曝光留出：参考图可重合，其他历史开发曝光仍存在。', '',
             '| 域 | 剩余任务 | 排除旧query照片后 | 新取任务 | 新query照片数 | 新query曾作旧reference | reference与旧reference重合数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in c.DOMAINS:
        v = r['coverage'][d]
        lines.append(f'| {names[d]} | {v["remaining_tasks"]} | {v["available_after_query_exclusion"]} | '
                     f'{v["selected"]} | {v["query_photos"]} | {v["queries_seen_as_old_reference"]} | '
                     f'{v["references_overlapping_old_references"]} |')
    lines += ['', 'iSAID、Lung、Fundus、Road已无剩余S任务，本次未复测，不在新批均值中使用旧例。', '',
              '直接复用未修改的W27 `infer`，调用同一个冻结W23 `simple_head`；W27已在原1,285例上100%复现W23。'
              '新批每例的缓存请求/求解次数/共享图/零编码器调用检查通过；捕获字段重组SH的XOR总计0，'
              'SH+直接取冻结函数原始输出，回退例六臂逐像素一致。全部新预测封存后才读新批S查询GT。', '',
              '## 2 新批六臂绝对 mIoU', '',
              '域内(fold,class)池化I/U后槽等权，再对这六个域等权。FoRIS只取同例既有S记录。', '',
              '| 域 | n / 非回退 | U0 | M′ | SH | U0+ | M′+ | SH+ | FoRIS |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, v in full['domains'].items():
        n_active = active['domains'].get(d, {}).get('n', 0)
        lines.append(f'| {names[d]} | {v["n"]} / {n_active} | '+' | '.join(f'{v["scores"][a]:.2f}' for a in ARMS+('foris',))+' |')
    lines.append('| 六域等权 | 871 / '+str(active['n'])+' | '+' | '.join(f'{full["mean"]["scores"][a]:.2f}' for a in ARMS+('foris',))+' |')
    for title, keys in (('3 新批融合减单线', KEYS),
                        ('4 新批 CRF 后减 CRF 前', ('U0+-U0', 'Mprime+-Mprime', 'SH+-SH'))):
        lines += ['', '## '+title, '', '差值单位pp，附95%照片配对bootstrap区间。逐域500次、seed=c.SEED；'
                  '均值500次域内照片分层、域间等权、seed=c.SEED+1，与W27同一计算口径。', '',
                  '| 域 | '+' | '.join(k.replace('Mprime','M′') for k in keys)+' |', '|---|'+'---:|'*len(keys)]
        for d, v in full['domains'].items():
            lines.append('| '+names[d]+' | '+' | '.join(w.cell(v['effects'][k]) for k in keys)+' |')
        lines.append('| 六域等权 | '+' | '.join(w.cell(full['mean']['effects'][k]) for k in keys)+' |')
    lines += ['', '## 5 非回退子集', '',
              '| 域 | n | SH−U0 | SH−M′ | SH+−U0+ | SH+−M′+ |', '|---|---:|---:|---:|---:|---:|']
    for d, v in active['domains'].items():
        lines.append(f'| {names[d]} | {v["n"]} | '+' | '.join(w.cell(v['effects'][k]) for k in KEYS)+' |')
    lines.append(f'| {active["mean_domain_count"]}域等权 | {active["n"]} | '+' | '.join(w.cell(active['mean']['effects'][k]) for k in KEYS)+' |')
    lines += ['', '## 6 新旧批次的同域比较', '',
              '旧批只取相同六域，共900例。下面两批均为相同六域等权；旧十域均值不放进同批比较。'
              '任务与类别/照片构成不同，因此分数高低不表示方法改变。', '',
              '| 方法 | 旧六域 CRF前 | 新六域 CRF前 | 旧六域 CRF后 | 新六域 CRF后 |', '|---|---:|---:|---:|---:|']
    old = r['old_common_full']
    for a in ('U0', 'Mprime', 'SH'):
        lines.append(f'| {w.NAMES[a]} | {old["mean"]["scores"][a]:.2f} | {full["mean"]["scores"][a]:.2f} | '
                     f'{old["mean"]["scores"][a+"+"]:.2f} | {full["mean"]["scores"][a+"+"]:.2f} |')
    lines += ['', '| 批次 | SH−U0 | SH−M′ | SH+−U0+ | SH+−M′+ |', '|---|---:|---:|---:|---:|']
    for label, scope in (('旧六域900例', old), ('新六域871例', full)):
        lines.append('| '+label+' | '+' | '.join(w.cell(scope['mean']['effects'][k]) for k in KEYS)+' |')
    lines += ['', '## 7 结论与论文主张', '', r['conclusion'], '',
              f'新批CRF前两项总体比较均获支持：{r["pre_supported"]}；非回退子集CRF后两项总体比较均获支持：{r["active_supported"]}。', '',
              '逐域显著负值（95%区间上界<0；描述性统计，未做多重比较校正）：']
    for k in ('SH+-U0+', 'SH+-Mprime+'):
        ds = [names[d] for d,v in full['domains'].items() if v['effects'][k]['hi']<0]
        lines.append('- '+k.replace('Mprime','M′')+'：'+('、'.join(ds) if ds else '无')+'。')
    lines += ['', '不据此修改方法、阈值、权重或CRF，不重抽。只支持覆盖的六域开发复测；'
              '不能写十域新批确认、逐域全优或全新照片上的泛化确认，服务器计划保持不变。', '',
              '## 8 运行与可复现性', '', '```json', json.dumps(r['runtime'],ensure_ascii=False,indent=2), '```', '',
              '代码：`scripts/x1012/w28_sh_second_batch.py`；逐例证据：'
              '`evidence/research/w28_sh_second_batch_S.jsonl.gz`；新批样本冻结、掩码、封存、I/U记录和完整JSON：'
              '`../cv_data/a/x1013_W28/`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def report():
    verify_frozen()
    checks = load('checks.json')
    assert checks['passed'] and checks['episodes_sha256']==w.sha(OUT/'episodes.jsonl')
    started = time.perf_counter()
    base = {e['index']: e for e in w.evidence('x1013_E1_iu')}
    new = [dict(e, foris=base[e['index']]['foris']) for e in ordered('episodes.jsonl')]
    old = [dict(e, foris=base[e['index']]['foris']) for e in w.ordered_records('episodes.jsonl') if e['dataset'] in COMMON]
    assert len(old)==900
    new_full = w.analysed_scope(new, c.SEED)
    new_active = w.analysed_scope([e for e in new if not e['fallback']], c.SEED)
    old_full = w.analysed_scope(old, c.SEED)
    old_active = w.analysed_scope([e for e in old if not e['fallback']], c.SEED)
    post = w.supported(new_full, True)
    conclusion = ('新批六域复现了固定融合的总体优势：CRF后相对U0+、M′+的均值增益95%区间下界都>0。'
                  if post else '新批六域未完整复现“CRF后总体稳定优于两条单线”；按两项比较的点估计与区间报告，不重抽或调参。')
    production = [c.read(p) for p in sorted(OUT.glob('production_*.json'))]
    resources = [c.read(p) for p in sorted(OUT.glob('resources_*.json'))]
    snapshots = [s for run in resources for s in run['snapshots']]
    result = {'new_full': new_full, 'new_active': new_active, 'old_common_full': old_full,
              'old_common_active': old_active, 'coverage': load('manifest.json')['coverage'],
              'checks': checks, 'post_supported': post, 'pre_supported': w.supported(new_full, False),
              'active_supported': w.supported(new_active, True), 'conclusion': conclusion,
              'runtime': {'production_runs': production, 'scoring_wall_seconds': checks['scoring_wall_seconds'],
                          'analysis_wall_seconds': time.perf_counter()-started,
                          'sampled_cpu_percent_max': max(s['cpu_percent'] for s in snapshots),
                          'sampled_available_gib_min': min(s['available_gib'] for s in snapshots),
                          'raw_cache_requests': sum(e['raw_requests'] for e in new), 'encoder_calls': 0,
                          'T_read': False, 'new_photo_holdout_read': False,
                          'code_sha256': w.sha(__file__), 'plan_sha256': w.sha(PLAN),
                          'episodes_sha256': w.sha(OUT/'episodes.jsonl'), 'frozen_W27_verified': True}}
    save('report.json', result)
    write_report(result)
    print(conclusion, flush=True)
    for k in KEYS:
        print(k, w.cell(new_full['mean']['effects'][k]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('produce','check','report'))
    p.add_argument('workers', nargs='?', type=int, default=4)
    args = p.parse_args()
    if args.command=='produce':
        produce_all(args.workers)
    elif args.command=='check':
        check(args.workers)
    else:
        report()


if __name__=='__main__':
    main()
