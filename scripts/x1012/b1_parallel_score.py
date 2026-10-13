"""Parallel CPU scoring of fully sealed B1 outputs; no prediction generation.

The parent validates the entire natural seal and every artifact before spawning
workers. Natural metrics, exact grouped-threshold oracle, common-K cohorts, and
reports use the original B1/plan_report implementations and formulas.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import b1_kshot as base

lab, pc, pr, OUT, KS = base.lab, base.pc, base.pr, base.OUT, base.KS


def worker_init():
    pc.setup('cpu')


def score_one(job):
    """Original B1 per-case score body, calling its original oracle/metrics."""
    binding, receipt = job
    start = time.monotonic()
    with np.load(receipt['prediction_path'], allow_pickle=False) as f:
        predictions = {a: pc.unpack(f[a]) for a in f.files}
    record = pr.score_record(binding, predictions)
    gt, valid = pc.truth(binding)
    thresholds = {}
    with np.load(receipt['field_path'], allow_pickle=False) as f:
        for k in binding['available_k']:
            for method in ('IND0', 'INDN'):
                arm = f'K{k}/{method}'
                oracle, threshold = base.threshold_oracle(f[arm], gt, valid)
                record['metrics'][arm + '/oracle_threshold'] = pr.metric(oracle, gt, valid)
                thresholds[arm] = threshold
    record.update(available_k=binding['available_k'], oracle_thresholds=thresholds,
                  scoring_seconds=time.monotonic() - start)
    return record


def score(workers):
    pc.setup('cpu')
    started = time.monotonic()
    # This complete hash validation finishes before pool creation or any GT read.
    seal, rows, receipts = base.verify_seal(OUT, 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED')
    validation_seconds = time.monotonic() - started
    execution = {'entrypoint': str(Path(__file__).resolve()), 'entrypoint_sha256': pc.sha(__file__),
                 'reused_scoring_source_sha256': {str(p): pc.sha(p) for p in
                     (Path(base.__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__))},
                 'workers': workers, 'threads_per_worker': 4, 'device': 'cpu', 'start_method': 'spawn',
                 'dispatch': 'ProcessPoolExecutor.map, chunksize1, manifest order retained',
                 'natural_seal_sha256': pc.sha(OUT / 'sealed.json'),
                 'complete_artifact_validation_before_worker_GT': True,
                 'validation_seconds': validation_seconds, 'n': len(rows),
                 'predictions_generated': 0, 'new_encoder_calls': 0}
    records = []
    parallel_start = time.monotonic()
    jobs = ((binding, receipts[pc.key(binding)]) for binding in rows)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as pool:
        for i, record in enumerate(pool.map(score_one, jobs, chunksize=1), 1):
            records.append(record)
            if i <= 3 or i % 100 == 0:
                print({'task': 'B1-score', 'workers': workers, 'completed': i, 'n': len(rows),
                       'seconds': round(time.monotonic() - parallel_start, 1)}, flush=True)
    execution.update(parallel_case_wall_seconds=time.monotonic() - parallel_start,
                     case_scoring_seconds_sum=sum(r['scoring_seconds'] for r in records))
    # The following report/cohort logic is the original B1 score logic unchanged.
    arms_for = lambda ks: ('FoRIS',) + tuple(f'K{k}/{method}{suffix}' for k in ks for method in ('IND0', 'INDN')
                                            for suffix in ('', '/oracle_threshold'))
    project = lambda rs, arms: [dict(r, metrics={a: r['metrics'][a] for a in arms}) for r in rs]
    notes = ['完整K-shot清单；K=1/2/3/5嵌套参考，固定温度.05，统一中心和全参考共同softmax归一化。',
             'FoRIS固定为同query、同首参考的K1完整CRF；只读精确缓存或B1独占缓存重放，0encoder。',
             'oracle_threshold读取GT但只作诊断，按完整同分组选择可实现的严格阈值；自然预测全部先seal。',
             '主表K1涵盖全部available query；附表每组K严格使用共同可用query分母，禁止跨表混样本解释K增益。',
             f'本次仅将既有逐例评分分派{workers}个CPU worker，每worker4线程；主进程先验证完整seal及所有产物。未增加预测或改变C1候选。']
    arms = arms_for((1,))
    primary = pr.report('B1', OUT, project(records, arms), arms, notes,
                        extra={'production_cost': {k: v for k, v in seal.items() if k != 'records'},
                               'score_execution': execution})
    common_tables = {}
    for k in (2, 3, 5):
        ks = tuple(v for v in KS if v <= k)
        selected = [r for r in records if all(v in r['available_k'] for v in ks)]
        if not selected:
            continue
        arms = arms_for(ks)
        report = pr.report(f'B1-K{k}-common', OUT / f'common_K{k}', project(selected, arms), arms,
                           notes + [f'本表K={ks}全部共享同一query集合；不具备K{k}的域/任务不进入本表。'],
                           extra={'score_execution': execution})
        common_tables[str(k)] = {'K': ks, 'n': len(selected), 'summary': report['summary']}
    availability = {d: {str(k): sum(r['dataset'] == d and k in r['available_k'] for r in records) for k in KS}
                    for d in lab.ALL_DOMAINS}
    execution['total_score_seconds'] = time.monotonic() - started
    pc.write(OUT / 'kshot_comparisons.json', {'state': 'COMPLETE', 'n': len(records), 'availability': availability,
             'common_query_tables': common_tables, 'K1_summary': primary['summary'], 'records': records,
             'score_seconds': execution['total_score_seconds'], 'score_execution': execution,
             'confirm_scored': False, 'predictions_generated': 0, 'new_encoder_calls': 0})
    pc.write(OUT / 'scoring_execution.json', dict(execution, state='COMPLETE'))
    path = lab.ROOT / 'docs/research/x1012_tasks/B1.md'
    with path.open('a') as f:
        f.write('\n各K共同query附表：' + '、'.join(f'B1-K{k}-common.md' for k in common_tables) + '。\n')
        f.write('\n逐域available_k及同分母对照保存在 `kshot_comparisons.json`；LVIS缺少K5，不补造参考。\n')
        f.write(f'\n评分入口 `{execution["entrypoint"]}`，SHA `{execution["entrypoint_sha256"]}`；'
                f'{workers} workers × 4 CPU线程。逐例评分wall {execution["parallel_case_wall_seconds"]:.3f}s；'
                '本次生成预测数为0。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2, help='CPU scoring workers, each fixed at four threads')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    score(args.workers)
