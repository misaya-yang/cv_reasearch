"""E2: freeze selection from S once, then score only that arm on T."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import time
import numpy as np
import common as c
import score

CANDIDATES = ('C1', 'UFraw+', 'Ugm+', 'UF+', 'M+', 'MFU')


def choose_from_s():
    records = score.read_records('E1')
    if len(records) != 6752 or any(row['split'] != 'S' for row in records):
        raise RuntimeError('E2 requires the complete 6,752-row S-only E1 score.')
    if any(any(arm not in row for arm in (*CANDIDATES, 'foris')) for row in records):
        raise RuntimeError('E1 is missing a frozen E2 candidate or FoRIS reference.')
    reference = score.domain_scores(records, 'foris')
    summaries = []
    for arm in CANDIDATES:
        deltas = {domain: score.miou([r for r in records if r['dataset'] == domain], arm) -
                  reference[domain] for domain in c.DOMAINS}
        summaries.append({'arm': arm, 'min_domain_delta': min(deltas.values()),
                          'mean_domain_delta': float(np.mean(list(deltas.values()))),
                          'domain_deltas': deltas})
    winner = max(summaries, key=lambda row: (row['min_domain_delta'], row['mean_domain_delta']))
    source = c.task_path('E1') / 'iu.jsonl.gz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {'state': 'SELECTION_FROZEN_BEFORE_T', 'selected_arm': winner['arm'],
            'selection_rule': 'maximize the minimum of ten S-domain deltas vs full FoRIS; tie-break by ten-domain mean',
            'candidate_set': list(CANDIDATES), 'S_n': len(records),
            'T6_Sself_condition_met': False,
            'S_selection_input_sha256': digest, 'candidate_results': summaries,
            'selected_result': winner, 'T_scored': False}


def score_one(job):
    binding, selected_arm = job
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    record = c.metadata(binding, fields)
    record.update(gt=int(gt.sum()), split='T')
    with np.load(c.prediction_path('E1', binding), allow_pickle=False) as archive:
        for name in (selected_arm, 'UFraw'):
            record[name] = c.iu(c.unpack(archive[name]), gt, valid)
    record['foris'] = c.iu(c.rc.lab.arm_cli(binding, 'B').numpy(), gt, valid)
    import repo_methods as repo
    record['foris_noCRF'] = c.iu(c.readout(repo.foris_score(binding)), gt, valid)
    return record


def read_partial(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Ignore a torn final line; its index will be recomputed under the
            # already-frozen arm and rule.
            continue
    unique = {row['index']: row for row in records}
    return list(unique.values())


def delta_rows(records, arm, reference):
    result = {}
    rng = np.random.default_rng(c.SEED)
    for domain in c.DOMAINS:
        subset = [row for row in records if row['dataset'] == domain]
        delta = score.miou(subset, arm) - score.miou(subset, reference)
        low, high = score.paired_ci(subset, arm, reference, rng)
        result[domain] = {'delta_pp': delta, 'ci95': [low, high]}
    return result


def render_report(out, selection, records):
    arm = selection['selected_arm']
    selected = delta_rows(records, arm, 'foris')
    equal_area = delta_rows(records, 'UFraw', 'foris_noCRF')
    lines = [
        '# E2 report-half result', '',
        f"E2 selection was frozen from the 6,752 S rows before reading T. The selected arm is **{arm}**.",
        f"Selection criterion: maximize the minimum of ten domain deltas vs complete FoRIS; tie-break by the ten-domain mean. T6 S-self was ineligible (its condition failed).",
        'T was scored once for the selected arm and the prespecified UFraw equal-area comparison. '
        'Intervals use 500 paired query-photo bootstrap replicates; differences are class mIoU percentage points.', '',
        '## Selected arm vs complete FoRIS', '',
        '|Domain|Selected|FoRIS|Difference [95% CI]|', '|---|---:|---:|---:|'
    ]
    selected_mean = []
    for domain in c.DOMAINS:
        subset = [row for row in records if row['dataset'] == domain]
        delta = selected[domain]['delta_pp']
        low, high = selected[domain]['ci95']
        selected_mean.append(delta)
        lines.append(f"|{domain}|{score.miou(subset, arm):.2f}|{score.miou(subset, 'foris'):.2f}|{delta:+.2f} [{low:+.2f},{high:+.2f}]|")
    lines += ['', f"Ten-domain equal-weight mean: {np.mean(selected_mean):+.2f} pp; nonnegative domains: {sum(x >= 0 for x in selected_mean)}/10.", '',
              '## Area bands vs complete FoRIS', '', '|GT area|Mean episode-IoU difference (pp)|', '|---|---:|']
    for band in ('<2%', '2-20%', '>20%'):
        subset = [row for row in records if score.area_band(row) == band]
        delta = np.mean([score.episode_iou(row, arm) - score.episode_iou(row, 'foris')
                         for row in subset]) if subset else None
        lines.append(f"|{band}|{'—' if delta is None else f'{delta:+.2f}'}|")
    lines += ['', '## Equal-area ranking: UFraw vs FoRIS-noCRF', '',
              '|Domain|UFraw − FoRIS-noCRF [95% CI]|', '|---|---:|']
    for domain in c.DOMAINS:
        low, high = equal_area[domain]['ci95']
        lines.append(f"|{domain}|{equal_area[domain]['delta_pp']:+.2f} [{low:+.2f},{high:+.2f}]|")
    lines += ['', 'T was not used to change the selected arm, formula, or comparison set.', '']
    report_path = c.ROOT / 'docs/research/x1013_tasks/E2_T_REPORT.md'
    report_path.write_text('\n'.join(lines))
    return {'selected_vs_foris': selected, 'UFraw_vs_foris_noCRF': equal_area,
            'equal_weight_mean_delta_pp': float(np.mean(selected_mean)),
            'nonnegative_domains': int(sum(x >= 0 for x in selected_mean)),
            'area_band_episode_iou_delta_pp': {
                band: (float(np.mean([score.episode_iou(row, arm) - score.episode_iou(row, 'foris')
                                     for row in records if score.area_band(row) == band]))
                       if any(score.area_band(row) == band for row in records) else None)
                for band in ('<2%', '2-20%', '>20%')},
            'report_path': str(report_path)}


def run(workers):
    out = c.task_path('E2')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'scored.json').exists():
        raise RuntimeError('E2 is already complete; refusing to reopen T.')
    selection_path = out / 'selection.json'
    if selection_path.exists():
        selection = c.read(selection_path)
        if selection.get('state') != 'SELECTION_FROZEN_BEFORE_T':
            raise RuntimeError('Existing E2 selection is not a valid frozen selection.')
    else:
        selection = choose_from_s()
        c.write(selection_path, selection)
        print(f"E2 selection frozen from S only: {selection['selected_arm']}", flush=True)

    prediction_checks = []
    bindings = [binding for binding in c.rows() if c.half(binding) == 'T']
    for binding in bindings:
        path = c.prediction_path('E1', binding)
        if not path.exists():
            prediction_checks.append(str(binding['index']))
            continue
        with np.load(path, allow_pickle=False) as archive:
            if not all(name in archive.files for name in (selection['selected_arm'], 'UFraw')):
                prediction_checks.append(str(binding['index']))
    if prediction_checks:
        raise RuntimeError(f"E2 is missing sealed T predictions: {prediction_checks[:5]}")

    partial = out / 'T_partial.jsonl'
    records = read_partial(partial)
    known = {row['index'] for row in records}
    todo = [binding for binding in bindings if str(binding['index']) not in known]
    state = {'state': 'T_SCORING_FROZEN_SELECTION', 'selected_arm': selection['selected_arm'],
             'T_n': len(bindings), 'completed_n': len(records), 'T_scored': False}
    c.write(out / 'progress.json', state)
    started = time.monotonic()
    print(f"E2 score T: {len(todo)} remaining / {len(bindings)}; fixed arm={selection['selected_arm']}", flush=True)
    with partial.open('a') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=c.initialize) as pool:
            for number, record in enumerate(pool.map(score_one,
                    ((binding, selection['selected_arm']) for binding in todo), chunksize=8), 1):
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                if number % 100 == 0 or number == len(todo):
                    stream.flush()
                    print(f'E2 T score: {len(records) + number}/{len(bindings)}, {time.monotonic() - started:.1f}s', flush=True)
    records = read_partial(partial)
    if len(records) != len(bindings):
        raise RuntimeError(f'E2 T score count mismatch: {len(records)} != {len(bindings)}')
    score.save_records(out / 'iu.jsonl.gz', records)
    score.save_records(c.ROOT / 'evidence/research/x1013_E2_T_iu.jsonl.gz', records)
    summary = render_report(out, selection, records)
    c.write(out / 'scored.json', {'state': 'COMPLETE', 'n': len(records), 'scope': 'T only',
                                  'selected_arm': selection['selected_arm'], 'T_scored': True,
                                  'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                                  'seconds': time.monotonic() - started, 'summary': summary})
    c.write(out / 'progress.json', {'state': 'COMPLETE', 'selected_arm': selection['selected_arm'],
                                    'T_n': len(records), 'T_scored': True})
    import report_b
    report_b.render()
    print(f"E2 complete; selected={selection['selected_arm']}; report={summary['report_path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('The plan allows at most six CPU processes.')
    run(args.workers)


if __name__ == '__main__':
    main()
