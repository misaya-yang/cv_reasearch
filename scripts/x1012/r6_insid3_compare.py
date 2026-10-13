"""Join completed R6 and official INSID3 integer metrics; never read GT/masks."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import time

REPO = Path(os.environ.get('CV_RESEARCH_ROOT', Path.cwd())).resolve()
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012_R6_query_space'
INSID3 = ASSETS / 'a/x1012_R6_insid3/development1360'
OFFICIAL = ('insid3.bilinear', 'insid3.crf640')


def main(wait=False):
    source = OUT / 'frozen/r6_query_space.py'
    spec = importlib.util.spec_from_file_location('r6_fixed_natural_task', source)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    while not (INSID3 / 'report.json').exists():
        if not wait:
            raise FileNotFoundError('Official INSID3 development1360 report is not complete')
        time.sleep(5)
    task.check_config()
    task.initialize()
    source_report = OUT / 'report.before_INSID3.json'
    if not source_report.exists():
        shutil.copyfile(OUT / 'report.json', source_report)
    report = task.read(source_report)
    official_report = task.read(INSID3 / 'report.json')
    official_seal = task.read(INSID3 / 'sealed.json')
    assert official_report['n'] == official_seal['n'] == report['n'] == 1360
    assert official_report['prediction_seal_sha256'] == task.sha(INSID3 / 'sealed.json')
    official = {row['index']: row for row in task.read(INSID3 / 'episodes.json')}
    rows = task.read(OUT / 'manifest.json')
    old_rows = {task.pc.key(b): b for b in task.read(INSID3 / 'manifest.json')}
    assert set(official) == {r['index'] for r in report['records']} == set(old_rows)
    for row in rows:
        old = old_rows[task.pc.key(row)]
        assert old['dataset'] == row['dataset']
        for role in ('reference', 'query'):
            for suffix in ('path', 'crop', 'rgb_hash', 'mask_hash'):
                assert old['row'].get(role + '_' + suffix) == row['row'].get(role + '_' + suffix)
        for role in ('reference_raw', 'query_raw_identity'):
            assert old['fit'][role]['payload_path'] == row['fit'][role]['payload_path']
    receipt_path = OUT / 'INSID3_comparison.freeze.json'
    receipt = {'state': 'FROZEN_INTEGER_ONLY_JOIN', 'n': 1360, 'source_sha256': task.sha(__file__),
               'R6_report_before_join_sha256': task.sha(source_report),
               'INSID3_episodes_sha256': task.sha(INSID3 / 'episodes.json'),
               'INSID3_report_sha256': task.sha(INSID3 / 'report.json'),
               'INSID3_seal_sha256': task.sha(INSID3 / 'sealed.json'),
               'INSID3_config_sha256': official_report['config_sha256'],
               'natural_predictions_unchanged': True, 'query_GT_reads': 0, 'new_predictions': 0,
               'bootstrap': 'R6 same query-photo seed20261012,1000; reaggregate all controls with identical estimator',
               'controls': OFFICIAL}
    if receipt_path.exists():
        import json
        assert task.read(receipt_path) == json.loads(json.dumps(receipt))
    else:
        shutil.copyfile(__file__, OUT / 'frozen/r6_insid3_compare.py')
        task.write(receipt_path, receipt)
    records = copy.deepcopy(report['records'])
    for record in records:
        old = official[record['index']]
        assert old['query_photo'] == record['query_photo']
        assert str(old['class_id']) == str(record['class_id']) and str(old['fold']) == str(record['fold'])
        for name in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted'):
            assert old['metrics']['FoRIS'][name] == record['metrics']['FoRIS'][name], (record['index'], name, 'FoRIS integer identity')
        for arm in OFFICIAL:
            metric = dict(old['metrics'][arm])
            if metric['GT'] == 0:
                metric['abs_log2_area_error'] = 0. if metric['predicted'] == 0 else None
            elif metric['predicted'] == 0:
                metric['abs_log2_area_error'] = None
            metric['infinite_area_error'] = metric['abs_log2_area_error'] is None
            record['metrics'][arm] = metric
    started = time.monotonic()
    extra = dict(report['extra'], INSID3_comparison=receipt)
    notes = [line for line in report['notes'] if 'INSID3完成后再加入' not in line]
    notes.append('完整官方INSID3 bilinear与原CRF640同1360例单独列出；仅合并已评分整数，全1360同配对与FoRIS I/U等整数一致，不新增GT读取或预测。')
    arms = tuple(report['arms']) + OFFICIAL
    integrated = task.pr.report('R6_query_space', OUT, records, arms, notes, extra=extra)
    paired = {}
    for arm in task.ARMS:
        for baseline in OFFICIAL:
            rows2 = [dict(r, metrics={'FoRIS': r['metrics'][baseline], arm: r['metrics'][arm]}) for r in records]
            paired[arm + '_minus_' + baseline] = task.pr.summary(rows2, ('FoRIS', arm))
    task.write(OUT / 'INSID3_contrasts.json', paired)
    completion = task.read(OUT / 'completion.json')
    completion.update(state='COMPLETE_WITH_INSID3', report_sha256=task.sha(OUT / 'report.json'),
                      INSID3_comparison_sha256=task.sha(receipt_path),
                      INSID3_contrasts_sha256=task.sha(OUT / 'INSID3_contrasts.json'),
                      integer_only_join_seconds=time.monotonic() - started,
                      raw_UN_delta_to_INSID3={baseline: {d: s['raw_UN']['delta_to_foris_pp']
                                                        for d, s in paired['raw_UN_minus_' + baseline].items()}
                                             for baseline in OFFICIAL},
                      ten_domain_nonnegative_vs_FoRIS_and_INSID3_crf640=[a for a in task.ARMS if all(
                          table[a]['class_miou'] >= max(table['FoRIS']['class_miou'], table['insid3.crf640']['class_miou'])
                          for table in integrated['summary'].values())])
    task.write(OUT / 'completion.json', completion)
    print(completion, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
