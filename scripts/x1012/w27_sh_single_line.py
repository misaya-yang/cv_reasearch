"""W27: strict final-SH single-line ablation, W23's exact S sample only.

Motivation, fixed six arms, and interpretation are preregistered in
docs/research/x1012_tasks/W27_sh_single_line_PLAN.md. Capture the two solves
inside the frozen W23 simple_head without changing their returned values.
Keep the exact shared graph, float arithmetic, .5 threshold, .5/.5 fusion,
and rho>.9 fallback. Apply the same official CRF independently to U0 and M'.

produce seals all six masks before any query GT is opened; check scores the
sealed S predictions and requires every fusion I/U to equal W23; report runs
only after that check passes. T and new-photo holdout have no code path here.
No encoder, parameter selection, frozen-source edits, or baseline reruns.

    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w27_sh_single_line.py produce [workers]
    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w27_sh_single_line.py check [workers]
    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w27_sh_single_line.py report
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
sys.path[:0] = [str(ROOT / 'scripts/x1013'), str(ROOT / 'scripts/x1012')]
import common as c
import score
import w23_simple_pipeline as sh

OUT = c.task_path('W27')
PLAN = ROOT / 'docs/research/x1012_tasks/W27_sh_single_line_PLAN.md'
REPORT = ROOT / 'docs/research/x1012_tasks/W27_sh_single_line_REPORT.md'
EVIDENCE = ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz'
ARMS = ('U0', 'Mprime', 'SH', 'U0+', 'Mprime+', 'SH+')
NAMES = {'U0': 'U0', 'Mprime': 'M′', 'SH': 'SH', 'U0+': 'U0+', 'Mprime+': 'M′+', 'SH+': 'SH+'}
COMPARISONS = {'SH-U0': ('SH', 'U0'), 'SH-Mprime': ('SH', 'Mprime'),
               'SH+-U0+': ('SH+', 'U0+'), 'SH+-Mprime+': ('SH+', 'Mprime+'),
               'U0+-U0': ('U0+', 'U0'), 'Mprime+-Mprime': ('Mprime+', 'Mprime'),
               'SH+-SH': ('SH+', 'SH')}
EXPECTED = dict(zip(c.DOMAINS, (150, 150, 150, 150, 150, 110, 150, 105, 108, 62)))
EXPECTED_ACTIVE = dict(zip(c.DOMAINS, (150, 116, 150, 150, 146, 107, 150, 105, 0, 16)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    c.write(OUT / name, value)


def load(name):
    return c.read(OUT / name)


def evidence(name):
    if name not in ('w23_simple_pipeline_S', 'x1013_E1_iu'):
        raise ValueError('Only the declared S evidence is allowed')
    with gzip.open(ROOT / f'evidence/research/{name}.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def sample():
    rows = sh.rows_for('S')
    assert len(rows) == 1285
    assert all(c.half(b) == 'S' for b in rows)
    assert dict(Counter(b['dataset'] for b in rows)) == EXPECTED
    old = evidence('w23_simple_pipeline_S')
    assert {str(b['index']) for b in rows} == {e['index'] for e in old}
    return rows


def identity(rows):
    sources = [ROOT / 'scripts/x1012' / n for n in
               ('w23_simple_pipeline.py', 'r5_candidate.py', 'w19_simple_head.py',
                'w09_integration_lambda.py', 'lab.py', 'plan_common.py')]
    sources += sorted((ROOT / 'scripts/x1013').glob('*.py'))
    sources += sorted((c.RUNTIME / 'runtime').rglob('*.py'))
    return {'plan_sha256': sha(PLAN), 'producer_sha256': sha(__file__),
            'frozen_sources': {str(p): sha(p) for p in sources},
            'basis_sha256': sha(c.ASSETS / 'native_assets/positional_basis.pt'),
            'old_S_sha256': sha(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz'),
            'indices': [str(b['index']) for b in rows],
            'bindings': {str(b['index']): hashlib.sha256(json.dumps(
                {'row': b['row'], 'fit': b['fit']}, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest() for b in rows},
            'counts': EXPECTED, 'active_counts': EXPECTED_ACTIVE,
            'arms': list(ARMS), 'split': 'S', 'seed': c.SEED,
            'GT_access': 'only check, after sealed.json exists',
            'T_read': False, 'new_photo_holdout_read': False}


def verify_frozen():
    manifest = load('manifest.json')
    assert manifest['plan_sha256'] == sha(PLAN), 'Plan changed after run began'
    assert manifest['producer_sha256'] == sha(__file__), 'Experiment code changed after run began'
    for path, digest in manifest['frozen_sources'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    assert manifest['basis_sha256'] == sha(c.ASSETS / 'native_assets/positional_basis.pt')
    assert manifest['old_S_sha256'] == sha(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz')


def initialize():
    c.initialize()
    import torch
    torch.set_num_threads(2)


def prediction_path(binding):
    return OUT / 'predictions' / (c.key(binding) + '.npz')


def metadata(binding):
    row = binding['row']
    return {'index': str(binding['index']), 'dataset': binding['dataset'], 'fold': row['fold'],
            'cls': row['global_class_id'], 'photo': c.photo(binding), 'split': 'S'}


def infer(binding):
    """Capture final SH's own two solves; no duplicated head implementation."""
    import torch
    import produce
    import w09_integration_lambda as w09
    import w19_simple_head as w19
    if c.half(binding) != 'S':
        raise RuntimeError('W27 only accepts S bindings')
    captured = {'U0': [], 'Mprime': [], 'graphs': [], 'raw_requests': 0}
    original_u, original_m, original_raw = w09.solve, w19.mean_solve, c.rc.raw

    def point(*args, **kwargs):
        value = original_u(*args, **kwargs)
        captured['U0'].append(np.asarray(value).copy())
        captured['graphs'].append(id(args[2]))
        return value

    def prototype(*args, **kwargs):
        value = original_m(*args, **kwargs)
        captured['Mprime'].append(np.asarray(value).copy())
        captured['graphs'].append(id(args[2]))
        return value

    def raw(*args, **kwargs):
        captured['raw_requests'] += 1
        return original_raw(*args, **kwargs)

    started = time.perf_counter()
    with c.rc.pc.inference_only(binding):
        w09.solve, w19.mean_solve, c.rc.raw = point, prototype, raw
        try:
            pre, final, rho = sh.simple_head(binding, crf=True)
        finally:
            w09.solve, w19.mean_solve, c.rc.raw = original_u, original_m, original_raw
        frozen_seconds = time.perf_counter() - started
        assert captured['raw_requests'] == 2
        if rho > .9:
            assert len(captured['U0']) == len(captured['Mprime']) == 0
            masks = {a: pre for a in ARMS}
            assert np.array_equal(pre, final)
        else:
            assert len(captured['U0']) == len(captured['Mprime']) == 1
            assert len(set(captured['graphs'])) == 1
            point_field = c.rc.lab.minmax(c.rc.lab.up(torch.from_numpy(
                captured['U0'][0].reshape(64, 64).astype(np.float32)))).numpy()
            prototype_field = c.rc.lab.up(torch.from_numpy(captured['Mprime'][0])).numpy()
            recomposed = .5 * prototype_field.astype(np.float64) + .5 * point_field.astype(np.float64) > .5
            assert np.array_equal(recomposed, pre), 'Captured fields do not exactly reconstruct final SH'
            masks = {'U0': point_field > .5, 'Mprime': prototype_field > .5, 'SH': pre, 'SH+': final}
            for a in ('U0', 'Mprime'):
                masks[a + '+'] = produce.crf(binding, masks[a])
        encoder_calls = int(getattr(getattr(produce, 'CRF_ADAPTER', None), 'encoder_calls', 0))
        assert encoder_calls == 0
    audit = {'rho': rho, 'fallback': rho > .9, 'raw_requests': captured['raw_requests'],
             'point_solve_calls': len(captured['U0']), 'prototype_solve_calls': len(captured['Mprime']),
             'shared_graph': len(set(captured['graphs'])) <= 1,
             'recomposed_xor_pixels': 0, 'encoder_calls': encoder_calls,
             'sec': {'frozen_SH_plus': frozen_seconds, 'total': time.perf_counter() - started}}
    return masks, audit


def predict_one(binding):
    import psutil
    masks, audit = infer(binding)
    path = prediction_path(binding)
    c.save_npz(path, **{a: c.pack(masks[a]) for a in ARMS})
    return dict(metadata(binding), **audit, prediction=str(path), payload_sha256=sha(path),
                worker_rss_mib=psutil.Process().memory_info().rss / 1024 ** 2)


def indexed_jsonl(path):
    indexed = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['index'] not in indexed, f'Duplicate record {row["index"]}'
            indexed[row['index']] = row
    return indexed


def ordered_records(name):
    indexed = indexed_jsonl(OUT / name)
    ids = load('manifest.json')['indices']
    assert set(indexed) == set(ids), f'{name}: incomplete or wrong sample'
    return [indexed[i] for i in ids]


def resource():
    import psutil
    return {'cpu_percent': psutil.cpu_percent(interval=.2),
            'available_gib': psutil.virtual_memory().available / 1024 ** 3,
            'swap_out_bytes': psutil.swap_memory().sout}


def produce_all(workers):
    assert 1 <= workers <= 4
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sample()
    ident = identity(rows)
    if (OUT / 'manifest.json').exists():
        assert load('manifest.json') == ident, 'Resume identity mismatch'
    else:
        save('manifest.json', ident)
    stamp = time.time_ns()
    top = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True, check=True).stdout
    (OUT / f'top_{stamp}.txt').write_text(top)
    print(top, flush=True)
    path = OUT / 'predictions.jsonl'
    done = indexed_jsonl(path)
    for e in done.values():
        assert sha(e['prediction']) == e['payload_sha256']
    todo = [b for b in rows if str(b['index']) not in done]
    started, snapshots = time.perf_counter(), [{'completed': len(done), **resource()}]
    print(f'S {len(rows)}, saved {len(done)}, todo {len(todo)}, workers {workers}, torch 2 threads', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, path.open('a') as stream:
        for n, e in enumerate(pool.map(predict_one, todo, chunksize=1), 1):
            stream.write(json.dumps(e, allow_nan=False) + '\n')
            stream.flush()
            if n % 20 == 0 or n == len(todo):
                os.fsync(stream.fileno())
                snapshots.append({'completed': len(done) + n, **resource()})
                save(f'resources_{stamp}.json', {'workers': workers, 'torch_threads': 2, 'snapshots': snapshots})
                print(f'predicted {len(done)+n}/1285, elapsed {time.perf_counter()-started:.1f}s, '
                      f'CPU {snapshots[-1]["cpu_percent"]:.1f}%, available {snapshots[-1]["available_gib"]:.2f} GiB', flush=True)
    verify_frozen()
    complete = ordered_records('predictions.jsonl')
    active = Counter(e['dataset'] for e in complete if not e['fallback'])
    assert {d: active[d] for d in c.DOMAINS} == EXPECTED_ACTIVE
    seal = {'n': len(complete), 'predictions_jsonl_sha256': sha(path),
            'payload_sha256': {e['index']: e['payload_sha256'] for e in complete},
            'sealed_unix_ns': time.time_ns(), 'plan_sha256': ident['plan_sha256'],
            'producer_sha256': ident['producer_sha256'], 'T_read': False, 'new_photo_holdout_read': False}
    for e in complete:
        assert sha(e['prediction']) == e['payload_sha256']
    if (OUT / 'sealed.json').exists():
        previous = load('sealed.json')
        assert previous['payload_sha256'] == seal['payload_sha256']
    else:
        save('sealed.json', seal)
    save(f'production_{stamp}.json', {'n': len(complete), 'produced_this_run': len(todo), 'workers': workers,
         'torch_threads': 2, 'wall_seconds': time.perf_counter() - started,
         'worker_rss_mib_max': max(e['worker_rss_mib'] for e in complete),
         'episode_seconds_mean': float(np.mean([e['sec']['total'] for e in complete]))})
    print('All predictions sealed. No query GT has been opened. Run check.', flush=True)


def score_one(job):
    binding, prediction = job
    assert c.half(binding) == 'S'
    path = prediction_path(binding)
    assert sha(path) == prediction['payload_sha256']
    with np.load(path, allow_pickle=False) as archive:
        masks = {a: c.unpack(archive[a]) for a in ARMS}
    if prediction['fallback']:
        assert all(np.array_equal(mask, masks['SH']) for mask in masks.values())
    gt, valid = c.truth(binding)
    return dict(prediction, gt=int(gt.sum()), **{a: c.iu(masks[a], gt, valid) for a in ARMS})


def check(workers):
    assert 1 <= workers <= 4
    verify_frozen()
    sealed = load('sealed.json')
    assert sealed['predictions_jsonl_sha256'] == sha(OUT / 'predictions.jsonl')
    predictions = ordered_records('predictions.jsonl')
    assert sealed['n'] == len(predictions) == 1285
    audit_ok = all(e['encoder_calls'] == 0 and e['raw_requests'] == 2 and e['shared_graph'] and
                   e['recomposed_xor_pixels'] == 0 and e['point_solve_calls'] == (0 if e['fallback'] else 1)
                   and e['prototype_solve_calls'] == (0 if e['fallback'] else 1) for e in predictions)
    assert audit_ok, 'Inference audit failed; stop before analysis'
    print('Check 1/2: shared graph, exact field reconstruction, fallback, cache-only audit PASS', flush=True)
    indexed = {e['index']: e for e in predictions}
    rows = sample()
    path = OUT / 'episodes.jsonl'
    done = indexed_jsonl(path)
    jobs = [(b, indexed[str(b['index'])]) for b in rows if str(b['index']) not in done]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, path.open('a') as stream:
        for n, row in enumerate(pool.map(score_one, jobs, chunksize=4), 1):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            if n % 200 == 0 or n == len(jobs):
                print(f'scored {len(done)+n}/1285', flush=True)
    scored = ordered_records('episodes.jsonl')
    old = {e['index']: e for e in evidence('w23_simple_pipeline_S')}
    same = {a: sum(e[a][:2] == old[e['index']][a][:2] for e in scored) for a in ('SH', 'SH+')}
    passed = audit_ok and same['SH'] == same['SH+'] == 1285
    result = {'n': 1285, 'inference_audit_passed': audit_ok, 'reconstruction_xor_pixels': 0,
              'fusion_equal_W23': same, 'required': 1285, 'passed': passed,
              'episodes_sha256': sha(path), 'sealed_sha256': sha(OUT / 'sealed.json'),
              'scoring_wall_seconds': time.perf_counter() - started,
              'T_read': False, 'new_photo_holdout_read': False}
    save('checks.json', result)
    print('Check 3:', same, 'PASS' if passed else 'FAIL', flush=True)
    if not passed:
        mismatches = [{'index': e['index'], 'new_SH': e['SH'], 'old_SH': old[e['index']]['SH'],
                       'new_SH+': e['SH+'], 'old_SH+': old[e['index']]['SH+']}
                      for e in scored if any(e[a][:2] != old[e['index']][a][:2] for a in ('SH', 'SH+'))]
        save('identity_mismatches.json', mismatches)
        raise RuntimeError('Fusion identity gate failed; diagnose before interpreting any ablation result')
    with gzip.open(EVIDENCE, 'wt') as stream:
        for e in scored:
            keep = ('index', 'dataset', 'fold', 'cls', 'photo', 'split', 'gt', 'rho', 'fallback',
                    'raw_requests', 'encoder_calls', 'point_solve_calls', 'prototype_solve_calls',
                    'shared_graph', 'recomposed_xor_pixels', 'sec') + ARMS
            stream.write(json.dumps({k: e[k] for k in keep}, allow_nan=False) + '\n')
    print('All checks PASS. Compact S evidence exported; report is now allowed.', flush=True)


def bootstrap_domain(rows, arms, rng):
    """Same slot/presence arithmetic as score.paired_ci, with all six arms paired."""
    from scipy import sparse
    photos = list(dict.fromkeys(e['photo'] for e in rows))
    slots = list(dict.fromkeys((e['fold'], e['cls']) for e in rows))
    pi, si = {v: i for i, v in enumerate(photos)}, {v: i for i, v in enumerate(slots)}
    p = [pi[e['photo']] for e in rows]
    s = [si[(e['fold'], e['cls'])] for e in rows]
    shape = (len(photos), len(slots))
    weights = np.stack([np.bincount(rng.integers(len(photos), size=len(photos)), minlength=len(photos)) for _ in range(500)])
    presence = sparse.csr_matrix((np.ones(len(rows)), (p, s)), shape=shape)
    represented = weights @ presence > 0
    output = {}
    for a in arms:
        ints = sparse.csr_matrix(([e[a][0] for e in rows], (p, s)), shape=shape)
        unions = sparse.csr_matrix(([e[a][1] for e in rows], (p, s)), shape=shape)
        values = 100 * (weights @ ints) / np.maximum(weights @ unions, 1)
        output[a] = (values * represented).sum(1) / represented.sum(1)
    return output


def effect(rows, a, b, rng):
    lo, hi = score.paired_ci(rows, a, b, rng)
    return {'pp': score.miou(rows, a) - score.miou(rows, b), 'lo': lo, 'hi': hi}


def analysed_scope(rows, seed):
    rng = np.random.default_rng(seed)
    mean_rng = np.random.default_rng(seed + 1)
    domains, boot = {}, {}
    arms = ARMS + ('foris',)
    for d in c.DOMAINS:
        dr = [e for e in rows if e['dataset'] == d]
        if not dr:
            continue
        domains[d] = {'n': len(dr), 'photos': len({e['photo'] for e in dr}),
                      'slots': len({(e['fold'], e['cls']) for e in dr}),
                      'scores': {a: score.miou(dr, a) for a in arms},
                      'effects': {key: effect(dr, a, b, rng) for key, (a, b) in COMPARISONS.items()}}
        boot[d] = bootstrap_domain(dr, arms, mean_rng)
    means = {a: float(np.mean([v['scores'][a] for v in domains.values()])) for a in arms}
    mean_boot = {a: np.mean(np.stack([v[a] for v in boot.values()]), axis=0) for a in arms}
    effects = {}
    for key, (a, b) in COMPARISONS.items():
        lo, hi = np.percentile(mean_boot[a] - mean_boot[b], [2.5, 97.5]).tolist()
        effects[key] = {'pp': means[a] - means[b], 'lo': lo, 'hi': hi}
    interactions = {}
    for single in ('U0', 'Mprime'):
        before, after = f'SH-{single}', f'SH+-{single}+'
        draws = (mean_boot['SH+'] - mean_boot[single + '+']) - (mean_boot['SH'] - mean_boot[single])
        lo, hi = np.percentile(draws, [2.5, 97.5]).tolist()
        interactions[single] = {'pp': effects[after]['pp'] - effects[before]['pp'], 'lo': lo, 'hi': hi}
    return {'n': len(rows), 'domains': domains, 'mean': {'scores': means, 'effects': effects},
            'crf_interactions': interactions, 'mean_domain_count': len(domains)}


def supported(scope, post):
    keys = ('SH+-U0+', 'SH+-Mprime+') if post else ('SH-U0', 'SH-Mprime')
    return all(scope['mean']['effects'][key]['lo'] > 0 for key in keys)


def claim_result(full, active):
    post, pre = supported(full, True), supported(full, False)
    if post:
        primary = '本机 S 半的匹配消融支持：最终 SH+ 的固定融合在十域平均上优于 U0+ 和 M′+ 两条单线。'
    else:
        primary = '本机 S 半的匹配消融不足以支持“最终 SH+ 稳定优于任一单线”；须按两项比较的点估计、区间和域间取舍收窄主张。'
    stages = ('CRF 前的两项总体比较也均为正且区间不含0，互补在精修前已经存在。' if pre else
              'CRF 前的两项总体比较未全部获得区间下界>0，不能声称精修前已稳定优于两条单线。')
    active_text = ('非回退子集也支持两项总体比较，共同回退不解释该融合优势。' if supported(active, True) else
                   '非回退子集未同时支持两项总体比较，不能把共同回退部分当作融合互补证据。')
    negatives = {key: [d for d, v in full['domains'].items() if v['effects'][key]['hi'] < 0]
                 for key in ('SH+-U0+', 'SH+-Mprime+')}
    return {'post_supported': post, 'pre_supported': pre, 'active_post_supported': supported(active, True),
            'primary': primary, 'stages': stages, 'active': active_text,
            'post_significantly_negative_domains': negatives,
            'limit': '所有新增结果只来自已曝光的开发 S 半；不替代 SH 的新照片或官方协议确认。'
                     '不作逐域全优、因果机制或速度新主张，不改方法或根据消融调参。'}


def cell(value):
    return f'{value["pp"]:+.2f} [{value["lo"]:+.2f}, {value["hi"]:+.2f}]'


def write_report(result):
    checks = result['checks']
    full, active = result['full'], result['active']
    lines = ['# W27：最终 SH 的严格单线消融结果', '', '2026-10-14。定义和读法见 [预先写定的计划](W27_sh_single_line_PLAN.md)。', '',
             '只使用 W23 同一批 S 半1,285例，非回退1,090例；没有读取T半或新照片留出，没有编码器调用。'
             'U0、M′、融合的输入、去偏、图、求解、阈值、CRF及回退均匹配。'
             'M′ 直接按原场>.5，不额外minmax；固定融合权重为.5/.5。', '',
             '## 1 实现检查', '',
             '- 直接调用冻结 `simple_head`，原样捕获两次求解返回值；非回退每例恰好一次 U0、一次 M′，二者使用同一个 L；每例恰好两次缓存 O24 请求、零编码器调用。',
             '- 捕获字段重组融合与冻结 SH 的 CRF 前掩码逐像素一致，XOR总计0；回退例六臂逐像素一致。',
             f'- 与W23旧S记录对照：SH {checks["fusion_equal_W23"]["SH"]}/1,285、SH+ '
             f'{checks["fusion_equal_W23"]["SH+"]}/1,285，整数I/U完全一致。全部预测封存后才读取查询GT；所有检查PASS后才解释消融。', '',
             '## 2 六臂绝对分数', '',
             '指标为每域(fold,class)槽内池化I/U后槽等权，十域再等权，单位为mIoU百分数。'
             'FoRIS仅取同例既有S记录，作为上下文；未重跑FoRIS。', '',
             '| 域 | n / 非回退 | U0 | M′ | SH融合 | U0+ | M′+ | SH+融合 | FoRIS |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        v = full['domains'][d]
        lines.append(f'| {name} | {v["n"]} / {EXPECTED_ACTIVE[d]} | ' +
                     ' | '.join(f'{v["scores"][a]:.2f}' for a in ARMS + ('foris',)) + ' |')
    lines.append('| 十域等权 | 1285 / 1090 | ' + ' | '.join(f'{full["mean"]["scores"][a]:.2f}' for a in ARMS + ('foris',)) + ' |')
    for title, keys in (('3 融合与单线的配对差值', ('SH-U0', 'SH-Mprime', 'SH+-U0+', 'SH+-Mprime+')),
                        ('4 同一条线的 CRF 后减 CRF 前', ('U0+-U0', 'Mprime+-Mprime', 'SH+-SH'))):
        lines += ['', '## ' + title, '', '差值单位pp；逐域为 `score.paired_ci` 的照片成组95%区间（500次，rng=c.SEED）。'
                  '十域均值区间按域内照片成组、域间等权的配对分层bootstrap计算（500次，rng=c.SEED+1）。', '',
                  '| 域 | ' + ' | '.join(key.replace('Mprime', 'M′') for key in keys) + ' |',
                  '|---|' + '---:|' * len(keys)]
        for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
            lines.append('| ' + name + ' | ' + ' | '.join(cell(full['domains'][d]['effects'][key]) for key in keys) + ' |')
        lines.append('| 十域等权 | ' + ' | '.join(cell(full['mean']['effects'][key]) for key in keys) + ' |')
    lines += ['', '## 5 非回退子集', '',
              '仅保留rho≤.9的1,090例，九个有样本域等权；Fundus没有非回退例，不纳入该条件均值。'
              'Road仅16例，报告区间并保留小样本限制。采样与统计规则相同。', '',
              '| 域 | 非回退n | SH−U0 | SH−M′ | SH+−U0+ | SH+−M′+ |', '|---|---:|---:|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        if d not in active['domains']:
            continue
        v = active['domains'][d]
        lines.append(f'| {name} | {v["n"]} | ' + ' | '.join(cell(v['effects'][key]) for key in
                     ('SH-U0', 'SH-Mprime', 'SH+-U0+', 'SH+-Mprime+')) + ' |')
    lines.append('| 九域等权 | 1090 | ' + ' | '.join(cell(active['mean']['effects'][key]) for key in
                 ('SH-U0', 'SH-Mprime', 'SH+-U0+', 'SH+-Mprime+')) + ' |')
    lines += ['', '## 6 对论文主张的影响', '', result['claims']['primary'], '', result['claims']['stages'], '', result['claims']['active'], '',
              'CRF对融合相对单线优势的改变量（CRF后差值减CRF前差值，配对分层bootstrap）：', '',
              '| 比较基准 | 十域等权变化pp [95%区间] |', '|---|---:|']
    for a in ('U0', 'Mprime'):
        lines.append(f'| {NAMES[a]} | {cell(full["crf_interactions"][a])} |')
    for key, ds in result['claims']['post_significantly_negative_domains'].items():
        lines += ['', f'{key.replace("Mprime", "M′")} 显著为负的域：' + ('、'.join(ds) if ds else '无') + '。']
    lines += ['', result['claims']['limit'], '',
              '现行主方法SH+保持冻结。本报告只更新最终SH的互补/融合贡献证据，不改写旧MFU结果，'
              '不把本次开发消融写成新的独立确认。', '', '## 7 运行与可复现性', '', '```json',
              json.dumps(result['runtime'], ensure_ascii=False, indent=2), '```', '',
              '代码：`scripts/x1012/w27_sh_single_line.py`；紧凑证据：'
              '`evidence/research/w27_sh_single_line_S.jsonl.gz`；外置掩码、预测封存、I/U断点记录、'
              '检查和完整JSON报告：`../cv_data/a/x1013_W27/`。']
    REPORT.write_text('\n'.join(lines) + '\n')


def report():
    verify_frozen()
    checks = load('checks.json')
    assert checks['passed'] and checks['episodes_sha256'] == sha(OUT / 'episodes.jsonl')
    started = time.perf_counter()
    base = {e['index']: e for e in evidence('x1013_E1_iu')}
    rows = [dict(e, foris=base[e['index']]['foris']) for e in ordered_records('episodes.jsonl')]
    full = analysed_scope(rows, c.SEED)
    active_rows = [e for e in rows if not e['fallback']]
    assert len(active_rows) == 1090
    active = analysed_scope(active_rows, c.SEED)
    productions = [c.read(p) for p in sorted(OUT.glob('production_*.json'))]
    resources = [c.read(p) for p in sorted(OUT.glob('resources_*.json'))]
    snapshots = [s for run in resources for s in run['snapshots']]
    result = {'full': full, 'active': active, 'claims': claim_result(full, active), 'checks': checks,
              'runtime': {'production_runs': productions, 'scoring_wall_seconds': checks['scoring_wall_seconds'],
                          'analysis_wall_seconds': time.perf_counter() - started,
                          'sampled_cpu_percent_max': max(s['cpu_percent'] for s in snapshots),
                          'sampled_available_gib_min': min(s['available_gib'] for s in snapshots),
                          'raw_cache_requests': sum(e['raw_requests'] for e in rows),
                          'encoder_calls': 0, 'T_read': False, 'new_photo_holdout_read': False,
                          'code_sha256': sha(__file__), 'plan_sha256': sha(PLAN),
                          'episodes_sha256': sha(OUT / 'episodes.jsonl'), 'frozen_sources_verified': True}}
    save('report.json', result)
    write_report(result)
    print(result['claims']['primary'], flush=True)
    print(result['claims']['stages'], flush=True)
    for key in ('SH-U0', 'SH-Mprime', 'SH+-U0+', 'SH+-Mprime+'):
        print(key, cell(full['mean']['effects'][key]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('produce', 'check', 'report'))
    parser.add_argument('workers', nargs='?', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'produce':
        produce_all(args.workers)
    elif args.command == 'check':
        check(args.workers)
    else:
        report()


if __name__ == '__main__':
    main()
