"""Score W67 only after all 486 predictions are sealed; never feed GT to inference."""
import gzip
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import w67_correspondence_validation as run
import common as c
import score as sc
from diagnostic_metrics import exact_best_iou
from w59_window_analysis import draws
from w63_task_switch import metrics as switch_metrics

OUT = run.OUT
ARMS = tuple(a+p for a in run.ARMS for p in ('', '+'))
SEED = 2026101619


def j(iu):
    return iu[0]/max(iu[1], 1)


def net_edits(old, new, gt, valid):
    added, deleted = ~old & new & valid, old & ~new & valid
    return dict(add_tp=int((added & gt).sum()), add_fp=int((added & ~gt).sum()),
                delete_tp=int((deleted & gt).sum()), delete_fp=int((deleted & ~gt).sum()))


def net_summary(records, arm):
    delta = 100*np.array([j(r[arm])-j(r['SH'+'+'*(arm.endswith('+'))]) for r in records])
    positive, negative = delta[delta > 1e-12], delta[delta < -1e-12]
    gain = float(positive.mean()) if len(positive) else 0.
    loss = float(-negative.mean()) if len(negative) else 0.
    return dict(n=len(records), wins=len(positive), losses=len(negative), ties=len(delta)-len(positive)-len(negative),
                changed_win_fraction=len(positive)/max(len(positive)+len(negative), 1),
                gain_if_win_pp=gain, loss_if_loss_pp=loss,
                breakeven_win_fraction=loss/(gain+loss) if gain+loss else None,
                episode_mean_delta_pp=float(delta.mean()),
                edits={key: sum(r['edits'][arm][key] for r in records) for key in records[0]['edits'][arm]},
                common_positive={key: sum(r['common_positive_edits'][arm][key] for r in records) for key in records[0]['edits'][arm]},
                common_negative={key: sum(r['common_negative_edits'][arm][key] for r in records) for key in records[0]['edits'][arm]})


def macro(records, arms=ARMS, include_foris=False):
    rng = np.random.default_rng(SEED)
    if include_foris:
        arms = tuple(arms)+('foris',)
    domains = [d for d in run.DOMAINS if any(r['dataset'] == d for r in records)]
    points, samples, detail = {}, {}, {}
    for d in domains:
        selected = [r for r in records if r['dataset'] == d]
        points[d] = {a: sc.miou(selected, a) for a in arms}
        samples[d] = draws(selected, arms, rng)
        detail[d] = dict(n=len(selected), values=points[d], comparisons={})
    pairs = [(a+p, b+p) for a, b in (('Repair', 'SH'), ('Repair', 'Null'), ('Null', 'SH')) for p in ('', '+')]
    if include_foris:
        pairs += [(a+'+', 'foris') for a in run.ARMS]
    comparisons = {}
    for a, b in pairs:
        key = a+' - '+b
        ds = []
        for d in domains:
            diff = samples[d][a]-samples[d][b]; ds.append(diff)
            detail[d]['comparisons'][key] = dict(delta_pp=points[d][a]-points[d][b], ci_pp=np.percentile(diff, [2.5, 97.5]).tolist())
        comparisons[key] = dict(delta_pp=float(np.mean([points[d][a]-points[d][b] for d in domains])),
                                ci_pp=np.percentile(np.mean(ds, axis=0), [2.5, 97.5]).tolist())
    return dict(n=len(records), domains=detail, values={a: float(np.mean([points[d][a] for d in domains])) for a in arms}, comparisons=comparisons)


def reference_summary(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['unit']].append(r)
    pooled = []; stability = {a: [] for a in ARMS}
    all_gain = {a: 0 for a in ('Repair', 'Repair+', 'Null', 'Null+')}
    all_loss = dict(all_gain)
    for unit, group in grouped.items():
        group.sort(key=lambda r: r['member']); assert len(group) == 3
        pooled.append(dict(group[0], **{a: np.sum([r[a] for r in group], axis=0).tolist() for a in ARMS}))
        for a in ARMS:
            stability[a].append(dict(mean=float(np.mean([j(r[a]) for r in group])),
                                     worst=float(min(j(r[a]) for r in group)),
                                     area_log2_sd=float(np.std([np.log2((r[a][2]+1)/1048576) for r in group]))))
        for a in all_gain:
            ref = 'SH'+'+'*a.endswith('+')
            ds = [j(r[a])-j(r[ref]) for r in group]
            all_gain[a] += all(x > 1e-12 for x in ds)
            all_loss[a] += all(x < -1e-12 for x in ds)
    stability_means = {a: {k: float(np.mean([v[k] for v in vals])) for k in vals[0]} for a, vals in stability.items()}
    rng = np.random.default_rng(SEED)
    bootstrap = rng.integers(len(pooled), size=(500, len(pooled)))
    differences = {}
    for a in all_gain:
        ref = 'SH'+'+'*a.endswith('+')
        differences[a] = {}
        for k in ('mean', 'worst', 'area_log2_sd'):
            values = np.array([x[k]-y[k] for x, y in zip(stability[a], stability[ref])])
            scale = 1 if k == 'area_log2_sd' else 100
            differences[a][k] = dict(delta=float(values.mean()*scale), ci=(np.percentile(values[bootstrap].mean(1), [2.5, 97.5])*scale).tolist())
    return dict(n_queries=len(pooled), pooled=macro(pooled),
                per_reference={member: macro([r for r in records if r['member'] == member]) for member in ('R0', 'R1', 'R2')},
                stability=stability_means, differences=differences, all_three_gain=all_gain, all_three_loss=all_loss)


def switches_summary(records, manifest):
    grouped = defaultdict(list)
    bindings = {x['job_id']: x['binding'] for x in manifest['episodes']}
    for r in records:
        grouped[r['unit']].append(r)
    pairs = []
    for unit, group in grouped.items():
        group.sort(key=lambda r: r['member']); assert len(group) == 2
        (ga, va), (gb, vb) = [c.rc.pc.truth(bindings[r['job_id']]) for r in group]
        masks = []
        for r in group:
            with np.load(r['prediction'], allow_pickle=False) as z:
                masks.append({a: c.unpack(z[a]) for a in ARMS})
        common_fg, common_bg = ga & gb & va & vb, ~(ga | gb) & va & vb
        pair = dict(unit=unit, dataset=group[0]['dataset'])
        for a in ARMS:
            pair[a] = switch_metrics(masks[0][a], masks[1][a], ga, gb, va, vb)
            if not a.startswith('SH'):
                ref = 'SH'+'+'*a.endswith('+')
                pair[a]['relative_to_SH'] = {
                    'common_fg_repair': sum(int((~m[ref] & m[a] & common_fg).sum()) for m in masks),
                    'common_fg_destroy': sum(int((m[ref] & ~m[a] & common_fg).sum()) for m in masks),
                    'common_bg_repair': sum(int((m[ref] & ~m[a] & common_bg).sum()) for m in masks),
                    'common_bg_destroy': sum(int((~m[ref] & m[a] & common_bg).sum()) for m in masks)}
        pairs.append(pair)
    summaries = {}
    for name, selected in [('all', pairs)]+[(d, [p for p in pairs if p['dataset'] == d]) for d in c.DOMAINS if any(p['dataset'] == d for p in pairs)]:
        rng = np.random.default_rng(SEED); indices = rng.integers(len(selected), size=(500, len(selected)))
        values = {a: {k: np.array([p[a][k] for p in selected]) for k in ('matched_mean', 'matched_min', 'D_match')} for a in ARMS}
        summary = dict(n_pairs=len(selected), arms={}, comparisons={})
        for a in ARMS:
            summary['arms'][a] = {k: float(100*v.mean()) for k, v in values[a].items()}
            summary['arms'][a]['both_ge_half'] = sum(p[a]['both_matched_ge_half'] for p in selected)
            if not a.startswith('SH'):
                summary['arms'][a]['relative_to_SH'] = {k: sum(p[a]['relative_to_SH'][k] for p in selected) for k in selected[0][a]['relative_to_SH']}
        for a, b in [('Repair', 'SH'), ('Repair+', 'SH+'), ('Repair', 'Null'), ('Repair+', 'Null+')]:
            summary['comparisons'][a+' - '+b] = {}
            for k in values[a]:
                delta = 100*(values[a][k]-values[b][k])
                summary['comparisons'][a+' - '+b][k] = dict(delta_pp=float(delta.mean()), ci_pp=np.percentile(delta[indices].mean(1), [2.5, 97.5]).tolist())
        summaries[name] = summary
    return summaries, pairs


def main():
    if (OUT/'scored.json').exists():
        print('Already scored; retained', run.sha(OUT/'summary.json')); return
    complete = json.loads((OUT/'complete.json').read_text())
    assert run.sha(OUT/'episodes.jsonl') == complete['episodes_sha256']
    assert run.sha(OUT/'config.json') == complete['config_sha256']
    config = json.loads((OUT/'config.json').read_text())
    assert run.sha(OUT/'manifest.json') == config['manifest_sha256']
    manifest = json.loads((OUT/'manifest.json').read_text())
    records_by_id = run.read_records(OUT/'episodes.jsonl', 'job_id')
    assert len(records_by_id) == manifest['n'] == complete['n']
    # Freeze scorer before the first query label is read.
    run.write(OUT/'scoring_started.json', dict(scorer_sha256=run.sha(Path(__file__)), utc=run.utc(), seed=SEED))
    run.init()
    with gzip.open(ROOT/'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        old = {str(r['index']): r for r in map(json.loads, f)}
    records = []; started = time.perf_counter()
    for item in manifest['episodes']:
        r = dict(records_by_id[item['job_id']])
        assert run.sha(r['prediction']) == r['prediction_sha256']
        gt, valid = c.rc.pc.truth(item['binding']); r['gt'] = int(gt.sum())
        with np.load(r['prediction'], allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files if k.endswith('64'))
            masks = {a: c.unpack(z[a]) for a in ARMS}
            for a in ARMS:
                r[a] = c.iu(masks[a], gt, valid)
            r['edits'], r['common_positive_edits'], r['common_negative_edits'] = {}, {}, {}
            for a in ('Repair', 'Repair+', 'Null', 'Null+'):
                ref = 'SH'+'+'*a.endswith('+')
                r['edits'][a] = net_edits(masks[ref], masks[a], gt, valid)
                r['common_positive_edits'][a] = net_edits(masks[ref], masks[a], gt, valid & c.unpack(z['common_positive']))
                r['common_negative_edits'][a] = net_edits(masks[ref], masks[a], gt, valid & c.unpack(z['common_negative']))
            if r['panel'] == 'main':
                for key in ('dataset', 'photo', 'fold', 'cls', 'gt'):
                    assert old[r['index']][key] == r[key], ('FoRIS pairing mismatch', r['job_id'], key)
                r['foris'] = old[r['index']]['foris']
                r['oracle'] = {}
                for a in run.ARMS:
                    field, _, _ = run.field_from_arrays(z, a, r['fallback'])
                    assert np.array_equal(field > .5, masks[a])
                    r['oracle'][a] = exact_best_iou(field, gt, valid)
                    assert r['oracle'][a]+1e-12 >= j(r[a])
        records.append(r)
        if len(records) % 30 == 0:
            print('scored', len(records), '/', len(records_by_id), 'elapsed', round(time.perf_counter()-started), flush=True)
    main_records = [r for r in records if r['panel'] == 'main']
    summary = dict(n_tasks=len(records), scope='Fixed S development comparison; no T/new-photo confirmation',
                   bootstrap='500 paired query-photo draws within domain; observed fold/class slots; fixed reference bindings',
                   main=macro(main_records, include_foris=True), subsets={})
    for name, selected in [('A', [r for r in main_records if r['half'] == 'A']), ('B', [r for r in main_records if r['half'] == 'B']),
                            ('nonfallback', [r for r in main_records if not r['fallback']]),
                            ('without_suim', [r for r in main_records if r['dataset'] != 'suim'])]:
        summary['subsets'][name] = macro(selected)
    summary['net'] = {panel: {a: net_summary([r for r in records if r['panel'] == panel], a) for a in ('Repair', 'Repair+', 'Null', 'Null+')} for panel in ('main', 'reference', 'switch')}
    summary['oracle'] = {a: dict(episode_mean=100*float(np.mean([r['oracle'][a] for r in main_records])),
                               delta_from_SH_pp=100*float(np.mean([r['oracle'][a]-r['oracle']['SH'] for r in main_records])),
                               reached_half=sum(r['oracle'][a] >= .5 for r in main_records)) for a in run.ARMS}
    summary['reference'] = reference_summary([r for r in records if r['panel'] == 'reference'])
    summary['switch'], pairs = switches_summary([r for r in records if r['panel'] == 'switch'], manifest)
    active = [r for r in records if not r['fallback']]
    resources = [json.loads(line) for line in (OUT/'resources.jsonl').read_text().splitlines()]
    summary['execution'] = dict(fallback=sum(r['fallback'] for r in records), pixel_identical_baselines=sum(r['baseline_pixel_identity'] for r in records),
                                zero_checks=sum(r['zero_strength_checked'] for r in records), mps=0, encoder_calls=0,
                                query_GT_reads_during_inference=complete['query_GT_reads'], max_c_error=max(r['c_error'] for r in records),
                                max_mass_error=max(r['diagnostics'][a]['row_mass_relative_error'] for r in active for a in ('Repair', 'Null')),
                                sampled_cpu_peak=max(r['tree_cpu_machine_percent'] for r in resources),
                                sampled_rss_peak_gib=max(r['rss_gib'] for r in resources),
                                total_task_seconds=float(sum(r['seconds'] for r in records)),
                                timing_scope='Cached features; each task includes all 3 arms and technical verification; not deployment end-to-end latency',
                                per_arm_median={a: {k: float(np.median([r['timings'][a][k] for r in active])) for k in active[0]['timings'][a]} for a in run.ARMS},
                                mean_c_change={a: float(np.mean([r['diagnostics'][a]['c_abs_mean'] for r in active])) for a in ('Repair', 'Null')})
    evidence = ROOT/'evidence/research/w67_correspondence_validation_S.jsonl.gz'
    with gzip.open(evidence, 'wt') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
    with gzip.open(OUT/'switch_pairs.jsonl.gz', 'wt') as f:
        for r in pairs: f.write(json.dumps(r, allow_nan=False)+'\n')
    run.write(OUT/'summary.json', summary)
    run.write(ROOT/'evidence/research/w67_correspondence_validation_summary.json', summary)
    run.write(OUT/'scored.json', dict(n=len(records), scorer_sha256=run.sha(Path(__file__)), evidence_sha256=run.sha(evidence), utc=run.utc()))
    print(json.dumps(dict(main=summary['main']['comparisons'], execution=summary['execution']), indent=2))


if __name__ == '__main__': main()
