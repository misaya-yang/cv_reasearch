#!/usr/bin/env python3
"""Score only fully sealed axis-study masks; independent packed-pixel arithmetic."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import run_foris_axis_contradictions as run

OUT = run.OUT
FRAMES = ('cli', 'original')
NAMES = (*run.ARMS, 'FoRIS.archive', 'MEAN', 'R0', 'N')
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def count(bits):
    return int(POPCOUNT[bits].sum(dtype=np.int64))


def metrics(packed, truth, base, size):
    assert packed.dtype == np.uint8 and packed.size == (size + 7) // 8
    assert not np.any(np.unpackbits(packed)[size:])
    tp = count(packed & truth)
    fp = count(packed & ~truth)
    fn = count(truth & ~packed)
    adds, deletes = packed & ~base, base & ~packed
    result = {'I': tp, 'U': tp + fp + fn, 'TP': tp, 'FP': fp, 'FN': fn,
              'IoU': 100 * tp / max(tp + fp + fn, 1),
              'addTP': count(adds & truth), 'addFP': count(adds & ~truth),
              'deleteTP': count(deletes & truth), 'deleteFP': count(deletes & ~truth)}
    base_tp, base_fp = count(base & truth), count(base & ~truth)
    assert tp == base_tp + result['addTP'] - result['deleteTP']
    assert fp == base_fp + result['addFP'] - result['deleteFP']
    return result


def aggregate(records, expected, name, frame, observed):
    slots = defaultdict(lambda: [0, 0])
    for r in records:
        value = r['metrics'][frame][name]
        slot = slots[(str(r['fold']), str(r['class_id']))]
        slot[0] += value['I']
        slot[1] += value['U']
    folds = []
    for fold, classes in expected.items():
        keys = [(str(fold), str(c)) for c in classes if not observed or (str(fold), str(c)) in slots]
        if keys:
            folds.append(sum(100 * slots[k][0] / max(slots[k][1], 1) for k in keys) / len(keys))
    return sum(folds) / len(folds) if folds else 0.0


def main():
    start = time.monotonic()
    seal = run.read(OUT / 'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED_BEFORE_SCORING'
    assert seal['config_sha256'] == run.sha(OUT / 'config.json')
    assert seal['manifest_sha256'] == run.sha(OUT / 'manifest.json')
    manifest = run.read(OUT / 'manifest.json')
    assert len(manifest) == len(seal['records']) == 1018
    producers = {r['id']: r for r in seal['records']}
    for item in manifest:
        r = producers[item['id']]
        assert run.sha(r['prediction_path']) == r['prediction_sha256']
        assert run.sha(r['field_path']) == r['field_sha256']
        for source in item['controls'].values():
            assert run.sha(source['path']) == source['sha256']
    assert not (OUT / 'score.json').exists(), 'Preserve an existing score report.'
    from ics.official_data import array_hash
    records = []
    for i, item in enumerate(manifest, 1):
        row = item['row']
        with Image.open(row['query_mask_path']) as im:
            gt0 = np.asarray(im.convert('L')) > 0
        assert array_hash(gt0.astype(np.uint8)) == row['query_mask_hash']
        controls = {}
        for name, source in item['controls'].items():
            with np.load(source['path'], allow_pickle=False) as z:
                controls['FoRIS.archive' if name == 'FoRIS' else name] = {
                    frame: z[source[frame + '_key']].copy() for frame in FRAMES}
        record = {'id': item['id'], 'dataset': item['dataset'], 'episode_id': row['episode_id'],
                  'fold': row['fold'], 'class_id': row['loader_class_id'],
                  'primary_cohort': item['primary_cohort'], 'contradictions': item['contradictions'], 'metrics': {}}
        with np.load(producers[item['id']]['prediction_path'], allow_pickle=False) as z:
            for frame in FRAMES:
                h, w = (1024, 1024) if frame == 'cli' else tuple(row['query_size_hw'])
                yy, xx = np.arange(h) * gt0.shape[0] // h, np.arange(w) * gt0.shape[1] // w
                truth = np.packbits(gt0[yy[:, None], xx[None, :]])
                baseline = z[frame + '/source.crf']
                record['metrics'][frame] = {
                    name: metrics(z[frame + '/' + name] if name in run.ARMS else controls[name][frame], truth, baseline, h * w)
                    for name in NAMES}
        records.append(record)
        if i % 200 == 0:
            print(json.dumps({'scored': i, 'total': len(manifest), 'seconds': round(time.monotonic() - start, 2)}), flush=True)
    expected = run.read(OUT / 'expected_classes.json')
    prior = run.read(run.PRIOR / 'report.json')
    summary = {}
    for domain in run.DOMAINS:
        subset = [r for r in records if r['dataset'] == domain and r['primary_cohort']]
        assert len(subset) == 200
        scores = {frame: {policy: {name: aggregate(subset, expected[domain], name, frame, policy == 'observed') for name in NAMES}
                          for policy in ('observed', 'declared_slots')} for frame in FRAMES}
        for frame in FRAMES:
            assert abs(scores[frame]['observed']['FoRIS.archive'] - prior['summary'][domain]['class_miou'][frame]['observed']['FoRIS']) < 1e-9
        delta = scores['original']['observed']['axis.crf'] - scores['original']['observed']['source.crf']
        edits = {k: sum(r['metrics']['original']['axis.crf'][k] for r in subset) for k in ('addTP', 'addFP', 'deleteTP', 'deleteFP')}
        summary[domain] = {'n': len(subset), 'scores': scores, 'primary_delta_pp': delta, 'axis_gross_edits': edits,
                           'episodes_improved': sum(r['metrics']['original']['axis.crf']['IoU'] > r['metrics']['original']['source.crf']['IoU'] for r in subset),
                           'episodes_worsened': sum(r['metrics']['original']['axis.crf']['IoU'] < r['metrics']['original']['source.crf']['IoU'] for r in subset)}
    cases = [r for r in records if r['contradictions']]
    strong = {r['episode_id']: r['metrics'] for r in run.read(run.ROOT / 'evidence/local/strong600_case_conflicts_20261011/report.json')['records']}
    for r in cases:
        if r['episode_id'] in strong:
            old = strong[r['episode_id']]
            assert r['metrics']['original']['FoRIS.archive']['I'] == old['B']['I']
            assert r['metrics']['original']['FoRIS.archive']['U'] == old['B']['U']
            r['known_expensive_S_comparison_only'] = old['S']
    report = {'state': 'COMPLETE_FIXED_AXIS_CAUSAL_CONTROL', 'n': len(records), 'primary_n': 1000,
              'summary': summary, 'contradiction_cases': cases, 'records': records,
              'new_encoder_calls': 0, 'new_views': 0, 'query_gt_reads': len(records),
              'sealed_sha256': run.sha(OUT / 'sealed.json'), 'score_seconds': time.monotonic() - start,
              'uniform_positive_primary_point_differences': all(v['primary_delta_pp'] > 0 for v in summary.values()),
              'strong_method_goal_proven': False, 'independent_confirmation': False,
              'source_archive_mismatch_tasks': seal['production_summary']['archive_mismatch_tasks']}
    run.write(OUT / 'score.json', report)
    lines = ['# 五域匹配轴与已知矛盾完整结果', '',
             '主队列五域各200共1000，附加案例不改分母。原图按class/fold池化I/U，再平均观察类与观察fold。', '',
             '|域|原FoRIS|轴对照|差值pp|MEAN|R0|N|', '|---|---:|---:|---:|---:|---:|---:|']
    for domain, value in summary.items():
        s = value['scores']['original']['observed']
        lines.append(f"|{domain}|{s['source.crf']:.6f}|{s['axis.crf']:.6f}|{value['primary_delta_pp']:+.6f}|{s['MEAN']:.6f}|{s['R0']:.6f}|{s['N']:.6f}|")
    lines += ['', '实现对照不等于原创方法。所有队列已曝光，点差不冒称独立泛化确认。CLI和固定官方槽位见score.json。', '',
              '|已知矛盾任务|原FoRIS IoU|轴对照 IoU|增真|增假|删真|删假|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in cases:
        m = r['metrics']['original']
        a = m['axis.crf']
        lines.append(f"|{r['episode_id']}|{m['source.crf']['IoU']:.3f}|{a['IoU']:.3f}|{a['addTP']}|{a['addFP']}|{a['deleteTP']}|{a['deleteFP']}|")
    lines += ['', f"总任务{len(records)}，新增encoder0，新增视图0，预测先封存再评分。评分{report['score_seconds']:.3f}s。",
              f"原重放与封存控制有差异的任务数：{report['source_archive_mismatch_tasks']}。", '',
              '强S仅在三个已有反例中作为高编码成本的比较结果，不参与推断，不冒称本方法的零成本输入。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'state': report['state'], 'domain_deltas': {k: v['primary_delta_pp'] for k, v in summary.items()},
                      'uniform_positive': report['uniform_positive_primary_point_differences'], 'contradiction_cases': len(cases)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
