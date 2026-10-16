"""W70 score-only diagnostics after all fixed information outputs are sealed.

No threshold selection, classifier fitting, segmentation changes, or expansion.
"""
import gzip
import json
from pathlib import Path
import sys
from collections import defaultdict
from functools import lru_cache
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import w70_midlayer_information as run
import common as c

OUT = run.OUT
SIGNALS = ('S19', 'S20', 'S23', 'P19', 'B0')
MODES = ('plain', 'condition_B0', 'condition_B0_S23')
SEED = 2026101674


def token_sum(x):
    return np.asarray(x).reshape(64, 16, 64, 16).sum((1, 3))


def concordance(x, y, cells=None):
    """Weighted Mann-Whitney, with exact tied groups and cell restriction.

    y may be binary or a soft foreground fraction. Fractional labels weight
    foreground/background mass, including the unavoidable same-score ties.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    cells = np.zeros(len(x), int) if cells is None else np.asarray(cells)
    numerator = denominator = 0.; used = 0
    for cell in np.unique(cells):
        select = cells == cell; xx, yy = x[select], y[select]
        p, n = yy.sum(), (1-yy).sum()
        if not p or not n:
            continue
        idx = np.argsort(xx, kind='stable'); xx, yy = xx[idx], yy[idx]
        starts = np.r_[0, np.flatnonzero(xx[1:] != xx[:-1])+1]
        positive = np.add.reduceat(yy, starts)
        negative = np.add.reduceat(1-yy, starts)
        negative_before = np.cumsum(negative)-negative
        numerator += float(np.sum(positive*(negative_before+.5*negative)))
        denominator += float(p*n); used += 1
    return dict(value=numerator/denominator if denominator else None,
                pair_weight=denominator, informative_cells=used)


def region_metrics(z, cov, ok, gt_cells):
    out = {}
    for side in ('IN', 'OUT'):
        full = z[side].ravel().astype(bool)
        keep = full & ok.ravel()
        scores = {a: z[a].ravel()[keep] for a in SIGNALS}
        y = cov.ravel()[keep]; hard = y >= .5
        # Unlabelled quantiles use every region token, before GT/valid filtering.
        cuts = np.quantile(z['S23'].ravel()[full], [.2, .4, .6, .8]) if full.any() else np.zeros(4)
        sbin = np.searchsorted(cuts, scores['S23'], side='right')
        fbin = np.floor(scores['B0']*10).astype(int)
        cells = dict(plain=None, condition_B0=fbin, condition_B0_S23=fbin*5+sbin)
        values = {mode: {a: concordance(scores[a], hard, cc) for a in SIGNALS} for mode, cc in cells.items()}
        soft = {a: concordance(scores[a], y, cells['condition_B0_S23']) for a in SIGNALS}
        mass = gt_cells.ravel()[keep]
        out[side] = dict(n_region=int(full.sum()), n_valid=int(keep.sum()), foreground_tokens=int(hard.sum()),
                         background_tokens=int((~hard).sum()), gt_pixel_mass=int(mass.sum()),
                         majority_fg_gt_pixel_mass=int(mass[hard].sum()), soft_foreground_mass=float(y.sum()),
                         values=values, soft_joint_condition=soft)
    return out


def counterfactual_accounting(z, gt, valid):
    """Arithmetic for a fixed-sign interpretation; no edited mask is constructed."""
    old = c.unpack(z['SH']) & valid
    tp_cells = token_sum(old & gt); fp_cells = token_sum(old & ~gt & valid)
    gt_cells, valid_cells = token_sum(gt), token_sum(valid)
    ok = valid_cells >= 128
    I, U = int(tp_cells.sum()), int((old | gt).sum())
    base = I/max(U, 1)
    result = {}
    for a in ('S19', 'S20', 'S23', 'P19'):
        add = z['OUT'] & ok & (z[a] > 0)
        remove = z['IN'] & ok & (z[a] <= 0)
        atp = int((gt_cells-tp_cells)[add].sum())
        afp = int((valid_cells-gt_cells-fp_cells)[add].sum())
        dtp = int(tp_cells[remove].sum()); dfp = int(fp_cells[remove].sum())
        result[a] = dict(add_tokens=int(add.sum()), remove_tokens=int(remove.sum()),
                         add_TP=atp, add_FP=afp, delete_TP=dtp, delete_FP=dfp,
                         baseline_iou=base,
                         hypothetical_iou=(I+atp-dtp)/max(U+afp-dfp, 1),
                         hypothetical_delta_pp=100*((I+atp-dtp)/max(U+afp-dfp, 1)-base))
    return result


@lru_cache(maxsize=64)
def bootstrap_weights(keys):
    """Reuse the identical, label-independent paired draws across signals."""
    rng = np.random.default_rng(SEED); groups = defaultdict(list)
    for i, (dataset, group, photo) in enumerate(keys):
        groups[(dataset, group)].append(i)
    # Main queries are unique by construction; grouping remains explicit.
    weights = np.zeros((1000, len(keys)), dtype=np.int32)
    for repeat in range(1000):
        draw = []
        for indices in groups.values():
            photos = list(dict.fromkeys(keys[i][2] for i in indices))
            for photo in rng.choice(photos, len(photos)):
                draw.extend(i for i in indices if keys[i][2] == photo)
        weights[repeat] = np.bincount(draw, minlength=len(keys))
    return weights


def bootstrap_mean(values, records):
    """Photo bootstrap, stratified by the fixed domain/category composition."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if not finite.any():
        return dict(mean=None, n=0, ci95=None, ci975=None)
    keys = tuple((r['dataset'], r.get('group'), r['photo']) for r in records)
    weights = bootstrap_weights(keys)
    counts = weights @ finite.astype(float)
    totals = weights @ np.where(finite, values, 0.)
    samples = totals[counts > 0]/counts[counts > 0]
    return dict(mean=float(np.nanmean(values)), n=int(finite.sum()),
                ci95=np.percentile(samples, [2.5, 97.5]).tolist(),
                ci975=np.percentile(samples, [1.25, 98.75]).tolist())


def get_value(record, side, mode, signal):
    if record['missing_reference_rim']:
        return np.nan
    value = record['regions'][side]['values'][mode][signal]['value']
    return np.nan if value is None else value


def compare(rows, side, mode):
    values = {a: np.array([get_value(r, side, mode, a) for r in rows]) for a in SIGNALS}
    return dict(n_tasks=len(rows), signals={a: bootstrap_mean(v, rows) for a, v in values.items()},
                differences={a+' - '+b: bootstrap_mean(values[a]-values[b], rows)
                             for a, b in (('S19', 'S23'), ('S19', 'P19'), ('S20', 'S23'), ('S19', 'B0'))})


def action_summary(rows, a):
    eligible = [r for r in rows if not r['missing_reference_rim']]
    if not eligible:
        return dict(n=0)
    vals = np.array([r['accounting'][a]['hypothetical_delta_pp'] for r in eligible])
    positive, negative = vals[vals > 1e-12], vals[vals < -1e-12]
    gain = float(positive.mean()) if len(positive) else 0.
    loss = float(-negative.mean()) if len(negative) else 0.
    totals = {k: sum(r['accounting'][a][k] for r in eligible)
              for k in ('add_tokens', 'remove_tokens', 'add_TP', 'add_FP', 'delete_TP', 'delete_FP')}
    return dict(n=len(eligible), hypothetical_delta_pp=bootstrap_mean(vals, eligible),
                wins=len(positive), losses=len(negative), ties=int(np.sum(abs(vals) <= 1e-12)),
                mean_gain_if_win_pp=gain, mean_loss_if_loss_pp=loss,
                changed_win_fraction=len(positive)/max(len(positive)+len(negative), 1),
                descriptive_breakeven_fraction=loss/(gain+loss) if gain+loss else None, totals=totals,
                wrong_edit_pixel_fraction=(totals['add_FP']+totals['delete_TP'])/max(sum(totals[k] for k in ('add_TP','add_FP','delete_TP','delete_FP')), 1))


def reference_panel(groups, scored):
    queries = []
    for group in groups:
        members = group['members']; records = [scored[x['job_id']] for x in members]
        gt, valid = c.rc.pc.truth(members[0]['binding'])
        cov = token_sum(gt)/np.maximum(token_sum(valid), 1); ok = token_sum(valid) >= 128
        with np.load(records[0]['signal_path'], allow_pickle=False) as z:
            zones = {side: z[side].copy() for side in ('IN', 'OUT')}
        result = dict(unit=group['group_id'], dataset=records[0]['dataset'], photo=records[0]['photo'], regions={})
        for side in zones:
            keep = zones[side] & ok; y = cov[keep] >= .5
            result['regions'][side] = {}
            for a in ('S19', 'S20', 'S23', 'P19'):
                values = []
                for r in records:
                    with np.load(r['signal_path'], allow_pickle=False) as z:
                        values.append(None if a not in z else concordance(z[a][keep], y)['value'])
                good = [v for v in values if v is not None]
                result['regions'][side][a] = dict(per_reference=values, mean=float(np.mean(good)) if len(good)==3 else None,
                                                 worst=min(good) if len(good)==3 else None,
                                                 sd=float(np.std(good)) if len(good)==3 else None)
        queries.append(result)
    summary = {}
    for side in ('IN', 'OUT'):
        summary[side] = {}
        for a in ('S19', 'S20', 'S23', 'P19'):
            good = [q['regions'][side][a] for q in queries if q['regions'][side][a]['mean'] is not None]
            summary[side][a] = dict(n_queries=len(good), **{k: float(np.mean([v[k] for v in good])) if good else None for k in ('mean', 'worst', 'sd')})
    return dict(n_queries=len(queries), scope='Small fixed diagnostic panel; no general migration claim', summary=summary, queries=queries)


def switch_panel(groups, scored):
    pairs = []
    for group in groups:
        ma, mb = group['members']; ra, rb = scored[ma['job_id']], scored[mb['job_id']]
        ga, va = c.rc.pc.truth(ma['binding']); gb, vb = c.rc.pc.truth(mb['binding'])
        common = va & vb; vc = token_sum(common); ok = vc >= 128
        ya = token_sum(ga & common)/np.maximum(vc, 1) >= .5
        yb = token_sum(gb & common)/np.maximum(vc, 1) >= .5
        with np.load(ra['signal_path'], allow_pickle=False) as aa, np.load(rb['signal_path'], allow_pickle=False) as bb:
            region = (aa['IN'] | aa['OUT'] | bb['IN'] | bb['OUT']) & ok
            aonly, bonly = region & ya & ~yb, region & yb & ~ya
            fg, bg = region & ya & yb, region & ~(ya | yb)
            row = dict(unit=group['group_id'], dataset=ra['dataset'], photo=ra['photo'],
                       aonly=int(aonly.sum()), bonly=int(bonly.sum()), common_fg=int(fg.sum()), common_bg=int(bg.sum()), signals={})
            for a in ('S19', 'S20', 'S23', 'P19'):
                if a not in aa or a not in bb:
                    row['signals'][a] = None; continue
                delta = aa[a]-bb[a]
                accuracy = lambda v: float(np.mean((v > 0)+.5*(v == 0))) if len(v) else None
                ap, bp = accuracy(delta[aonly]), accuracy(-delta[bonly])
                row['signals'][a] = dict(direction_A=ap, direction_B=bp,
                    balanced_task_direction=.5*(ap+bp) if ap is not None and bp is not None else None,
                    common_fg_errors=int(((aa[a] <= 0) & fg).sum()+((bb[a] <= 0) & fg).sum()),
                    common_bg_errors=int(((aa[a] > 0) & bg).sum()+((bb[a] > 0) & bg).sum()))
        pairs.append(row)
    summary = {}
    for a in ('S19', 'S20', 'S23', 'P19'):
        good = [p['signals'][a] for p in pairs if p['signals'][a] is not None and p['signals'][a]['balanced_task_direction'] is not None]
        available = [p for p in pairs if p['signals'][a] is not None]
        summary[a] = dict(n_pairs=len(good), balanced_task_direction=float(np.mean([v['balanced_task_direction'] for v in good])) if good else None,
                          n_pairs_common=len(available),
                          common_fg_errors=sum(p['signals'][a]['common_fg_errors'] for p in available),
                          common_bg_errors=sum(p['signals'][a]['common_bg_errors'] for p in available),
                          common_fg_decisions=2*sum(p['common_fg'] for p in available),
                          common_bg_decisions=2*sum(p['common_bg'] for p in available))
    return dict(n_pairs=len(pairs), scope='Small fixed legal-mask switch panel; information scores, not final segmentation', summary=summary, pairs=pairs)


def score():
    if (OUT/'scored.json').exists():
        print('Scoring already sealed', run.sha(OUT/'summary.json')); return
    complete = json.loads((OUT/'complete.json').read_text())
    assert run.sha(OUT/'signals.jsonl') == complete['signals_sha256']
    assert run.sha(OUT/'config.json') == complete['config_sha256']
    config = json.loads((OUT/'config.json').read_text())
    assert run.sha(OUT/'manifest.json') == config['manifest_sha256']
    manifest = json.loads((OUT/'manifest.json').read_text())
    raw = run.read_jsonl(OUT/'signals.jsonl', 'job_id')
    assert len(raw) == complete['n_tasks'] == manifest['n_tasks']
    run.write(OUT/'scoring_started.json', dict(source_sha256=run.sha(Path(__file__)), utc=run.utc(), seed=SEED))
    run.initialize(load_encoder=False)
    rows = []
    for group in manifest['groups']:
        for item in group['members']:
            r = dict(raw[item['job_id']]); assert run.sha(r['signal_path']) == r['signal_sha256']
            gt, valid = c.rc.pc.truth(item['binding'])
            gt_cells, valid_cells = token_sum(gt), token_sum(valid)
            cov = gt_cells/np.maximum(valid_cells, 1); ok = valid_cells >= 128
            with np.load(r['signal_path'], allow_pickle=False) as z:
                iu = c.iu(c.unpack(z['SH']), gt, valid)
                if 'expected_iu' in item['baseline']:
                    assert iu == item['baseline']['expected_iu'] and int(gt.sum()) == item['baseline']['expected_gt']
                r['base_iu'], r['gt'] = iu, int(gt.sum())
                if not r['missing_reference_rim']:
                    r['regions'] = region_metrics(z, cov, ok, gt_cells)
                    r['accounting'] = counterfactual_accounting(z, gt, valid)
            rows.append(r)
    main = [r for r in rows if r['panel'] == 'main']
    summary = dict(scope='Fixed S information diagnostic; no segmentation modification, new masks, training or expansion',
                   n_tasks=len(rows), n_main=len(main), primary={}, secondary={}, strata={}, protection={}, soft_condition={}, support={})
    for side in ('IN', 'OUT'):
        summary['primary'][side] = compare(main, side, 'condition_B0_S23')
        summary['secondary'][side] = {mode: compare(main, side, mode) for mode in ('plain', 'condition_B0')}
        summary['soft_condition'][side] = {}
        soft_values = {}
        for a in SIGNALS:
            vals = [r['regions'][side]['soft_joint_condition'][a]['value'] if not r['missing_reference_rim'] else None for r in main]
            soft_values[a] = np.array([np.nan if v is None else v for v in vals])
            summary['soft_condition'][side][a] = bootstrap_mean(soft_values[a], main)
        summary['soft_condition'][side]['differences'] = {a+' - '+b: bootstrap_mean(soft_values[a]-soft_values[b], main)
                                                        for a, b in (('S19','S23'),('S19','P19'))}
        ready = [r for r in main if not r['missing_reference_rim']]
        totals = {k: sum(r['regions'][side][k] for r in ready) for k in
                  ('n_region','n_valid','foreground_tokens','background_tokens','gt_pixel_mass','majority_fg_gt_pixel_mass')}
        pairs = [r['regions'][side]['values']['condition_B0_S23']['S19']['pair_weight'] for r in ready]
        all_gt = sum(r['gt'] for r in ready)
        summary['support'][side] = dict(**totals, majority_fg_fraction_of_region_gt=totals['majority_fg_gt_pixel_mass']/max(totals['gt_pixel_mass'], 1),
                                       region_gt_fraction_of_all_gt=totals['gt_pixel_mass']/max(all_gt,1),
                                       all_gt_pixel_mass=all_gt, joint_condition_pairs_sum=sum(pairs),
                                       joint_condition_pairs_median=float(np.median(pairs)),
                                       joint_condition_evaluable_queries=sum(p>0 for p in pairs))
    partitions = [(g, [r for r in main if r['group'] == g]) for g in ('add', 'remove', 'correct')]
    partitions += [(d, [r for r in main if r['dataset'] == d]) for d in run.DOMAINS]
    partitions += [('repeat_'+p, [r for r in main if r['repeat'] == p]) for p in ('A', 'B')]
    for name, selected in partitions:
        summary['strata'][name] = {side: compare(selected, side, 'condition_B0_S23') for side in ('IN','OUT')}
    for name, selected in [('all',main)]+[(g,[r for r in main if r['group']==g]) for g in ('add','remove','correct')]:
        summary['protection'][name] = {a: action_summary(selected,a) for a in ('S19','S20','S23','P19')}
    indexed = {r['job_id']:r for r in rows}
    summary['reference'] = reference_panel([g for g in manifest['groups'] if g['panel']=='reference'], indexed)
    summary['switch'] = switch_panel([g for g in manifest['groups'] if g['panel']=='switch'], indexed)
    resources = [json.loads(line) for line in (OUT/'resources.jsonl').read_text().splitlines()]
    summary['execution'] = dict(baseline_pixel_identity=sum(r['baseline_pixel_identity'] for r in rows),
        missing_reference_rim=sum(r['missing_reference_rim'] for r in rows), encoder_calls=complete['encoder_calls'],
        max_O24_relative_l2=max(t['last_layer_relative_l2'] for r in rows for t in r['encoder_trace']),
        min_O24_token_cosine=min(t['last_layer_min_cosine'] for r in rows for t in r['encoder_trace']),
        cpu_machine_percent_peak=max(r['cpu_machine_percent'] for r in resources), rss_gib_peak=max(r['rss_gib'] for r in resources),
        query_GT_file_reads_during_signal_production=0, candidate_segmentation_masks=0, model_parameter_updates=0, mps=0)
    evidence = ROOT/'evidence/research/w70_midlayer_information_cases.jsonl.gz'
    with gzip.open(evidence,'wt') as stream:
        for r in rows: stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    run.write(OUT/'summary.json', summary)
    run.write(ROOT/'evidence/research/w70_midlayer_information_summary.json', summary)
    run.write(OUT/'scored.json', dict(n_tasks=len(rows), source_sha256=run.sha(Path(__file__)), evidence_sha256=run.sha(evidence), utc=run.utc()))
    print(json.dumps(dict(primary=summary['primary'], protection=summary['protection']['correct'], execution=summary['execution']),ensure_ascii=False,indent=2))


if __name__ == '__main__': score()
