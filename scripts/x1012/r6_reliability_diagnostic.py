"""Read-only error/reliability diagnosis on already exposed R1 and R5.

No model, encoder, fitted head, candidate, parameter selection or prediction
write. One CPU thread; GT is used only in explicitly offline diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent/'cv_data'
OUT = ASSETS/'a/x1012_R6_diagnostic'
R1, R4, R5 = [ASSETS/'a'/n for n in ('x1012_R1', 'x1012_R4_global', 'x1012_R5_candidate')]
R5FINAL, R5RESULT = R5/'final_tau0475', ASSETS/'a/x1012_R5_results'
COUNTS = ('valid', 'FG', 'BG', 'TP', 'FP', 'FN', 'TN',
          'highFG_TP', 'highFG_FP', 'highBG_FN', 'highBG_TN')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def helpers():
    modules = {}
    for name, source in [('lab', R1/'frozen/3_lab.py'), ('plan_common', R1/'frozen/1_plan_common.py')]:
        dest = OUT/'runtime'/(name+'.py')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert dest.read_bytes() == source.read_bytes()
        spec = importlib.util.spec_from_file_location(name, dest)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'lab':
            module.ROOT, module.ASSETS = ROOT, ASSETS
            module.DEV = torch.device('cpu')
        modules[name] = module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return modules['lab'], modules['plan_common']


lab, pc = helpers()


def query_identity(binding):
    row = binding['row']
    return [binding['dataset'], str(row['fold']), str(row['global_class_id']),
            row['query_rgb_hash'], row['query_mask_hash'], row.get('query_crop')]


def reference_identity(binding):
    row = binding['row']
    return [row['reference_rgb_hash'], row['reference_mask_hash'], row.get('reference_crop')]


def prepare():
    dependencies = [R1/'manifest.json', R1/'sealed.json', R1/'config.json',
                    R4/'report.json', R4/'sealed.json', R4/'config.json',
                    R5/'manifest.json', R5/'sealed.json', R5/'config.json',
                    R5FINAL/'sealed.json', R5FINAL/'config.json',
                    R5RESULT/'report.json', R5RESULT/'selection.json']
    spec = {
        'state': 'FIXED_OFFLINE_DIAGNOSTIC_ON_EXPOSED_DATA', 'n_R1': 1360, 'n_R5': 500,
        'motivation': 'Frozen R5 UN loses four domains; R4 scalar calibration did not resolve conflicts. Determine whether low reference density identifies wrong decisions and whether errors persist across available alternate references for the same query task.',
        'positive_next_decision': 'If errors concentrate in low-e tokens and change with reference, retain uncertainty/coverage as a live failure mechanism.',
        'negative_next_decision': 'If high-e confident wrong pixels remain substantial or errors persist across references, do not treat e or graph confidence as task-correctness evidence in the next training-free construction.',
        'method_unchanged': 'same already frozen final tau=.475, rho>.9 unchanged raw-T fallback; R1 uses saved R4_global UN_tau475, R5 uses saved final candidate',
        'reliability': 'original e=sigmoid(log_e-median(log_e)); use saved e when available; reconstruct identical proxy from saved FP16 log_e only for R5 fallback, where e did not participate in prediction',
        'e_strata': 'native 4096-token e quartiles per episode; q1<=25%, q2<=50%, q3<=75%, q4>75%; tied e values remain together. Every 16x16 pixel block inherits its original unary e stratum.',
        'high_confidence': 'fixed diagnostic score>=.75 foreground or score<=.25 background, on unchanged bilinear1024 then minmax VP_UN. These values are never chosen using GT or used to generate a candidate.',
        'primary_population': 'rho<=.9 graph-active episodes; fallback reported separately because it did not use e',
        'same_query_task': 'dataset, fold, class, query_rgb_hash, query_mask_hash, query_crop all equal; reference RGB/mask/crop identities must differ',
        'pair_analysis': 'all available unordered different-reference pairs; errors and GT on CLI1024; cross-R1/R5 separately; no pair selection by outcomes; descriptive observational comparisons, not randomized intervention',
        'pair_persistence': '2*intersection(error_A,error_B)/(error_A+error_B); separately FP,FN,high-confidence FP/FN. Also prediction XOR/valid, mask IoU and episode-IoU difference.',
        'uncertainty': 'same-query sample counts and domain coverage explicit; no unsupported extrapolation to sparse LVIS/SUIM paired samples',
        'reference_count': 1, 'new_encoder_calls': 0, 'fitted_models': 0,
        'GT_role': 'offline diagnosis only', 'new_candidates': 0, 'parameter_search': False,
        'device': 'cpu', 'workers': 1, 'threads': 1,
        'source_sha256': sha(__file__),
        'helper_sha256': {str(OUT/'runtime'/(n+'.py')): sha(OUT/'runtime'/(n+'.py')) for n in ('lab', 'plan_common')},
        'input_sha256': {str(p): sha(p) for p in dependencies},
    }
    config_path = OUT/'config.json'
    if config_path.exists():
        assert read(config_path) == spec, 'Diagnostic inputs or definitions changed'
    else:
        write(config_path, spec)
        (OUT/'frozen').mkdir(exist_ok=True)
        shutil.copyfile(__file__, OUT/'frozen/r6_reliability_diagnostic.py')
    jobs = []
    for cohort, source, predictions, score_source, mask_key, metric_key in (
        ('R1', R1, R4, R4/'report.json', 'UN_tau475', 'UN_tau475'),
        ('R5', R5, R5FINAL, R5RESULT/'report.json', 'candidate', 'candidate'),
    ):
        rows = read(source/'manifest.json')
        fields = {r['index']: r for r in read(source/'sealed.json')['records']}
        masks = {r['index']: r for r in read(predictions/'sealed.json')['records']}
        scores = {r['index']: r for r in read(score_source)['records']}
        for binding in rows:
            index = pc.key(binding)
            jobs.append({'cohort': cohort, 'binding': binding, 'field': fields[index],
                         'prediction': masks[index], 'mask_key': mask_key,
                         'saved_metrics': scores[index]['metrics'][metric_key],
                         'saved_FoRIS_metrics': scores[index]['metrics']['FoRIS'],
                         'task_id': digest(query_identity(binding)),
                         'reference_id': digest(reference_identity(binding))})
    assert len(jobs) == 1860
    return jobs, spec


def load(job, verify=True):
    f, p = job['field'], job['prediction']
    if verify:
        assert sha(f['field_path']) == f['field_sha256']
        assert sha(p['prediction_path']) == p['prediction_sha256']
    with np.load(f['field_path'], allow_pickle=False) as z:
        field, rho = z['VP_UN'].copy(), float(z['reference_rho'][0])
        e = z['explanation'].copy() if 'explanation' in z else expit(z['log_e'].astype(np.float64)-np.median(z['log_e'].astype(np.float64))).astype(np.float32)
    with np.load(p['prediction_path'], allow_pickle=False) as z:
        pred = pc.unpack(z[job['mask_key']])
    gt, valid = pc.truth(job['binding'])
    score = lab.minmax(lab.up(torch.from_numpy(field))).numpy()
    expected = score > (.5 if rho > .9 else .475)
    assert np.array_equal(expected, pred), 'Frozen final-mask reproduction mismatch'
    return e.reshape(-1), rho, score, pred, gt, valid


def blocksum(mask):
    return mask.reshape(64, 16, 64, 16).sum(axis=(1, 3), dtype=np.int64).reshape(-1)


def episode(job):
    e, rho, score, pred, gt, valid = load(job)
    pred &= valid
    tp, fp, fn, tn = pred & gt, pred & ~gt & valid, ~pred & gt, ~pred & ~gt & valid
    hi, lo = (score >= .75) & valid, (score <= .25) & valid
    arrays = (valid, gt, ~gt & valid, tp, fp, fn, tn, hi & gt, hi & ~gt, lo & gt, lo & ~gt)
    blocked = np.stack([blocksum(a) for a in arrays], axis=1)
    cuts = np.quantile(e, [.25, .5, .75])
    bins = np.searchsorted(cuts, e, side='left')
    strata = np.stack([blocked[bins == q].sum(0) for q in range(4)])
    totals = dict(zip(COUNTS, map(int, strata.sum(0))))
    expected = job['saved_metrics']
    assert all(totals[k] == expected[k] for k in ('TP', 'FP', 'FN'))
    assert totals['FG'] == expected['GT'] and totals['TP']+totals['FP'] == expected['predicted']
    return {'cohort': job['cohort'], 'index': pc.key(job['binding']), 'dataset': job['binding']['dataset'],
            'task_id': job['task_id'], 'reference_id': job['reference_id'], 'graph_active': rho <= .9,
            'reference_rho': rho, 'e_cuts': cuts.tolist(), 'token_n_by_quartile': [int((bins == q).sum()) for q in range(4)],
            'e_quartile_integer_counts': strata.tolist(), 'saved_metrics': expected,
            'episode_delta_to_FoRIS_pp': expected['IoU']-job['saved_FoRIS_metrics']['IoU']}


def ratio(a, b):
    return float(a/b) if b else None


def derived(matrix):
    totals = matrix.sum(0)
    t = dict(zip(COUNTS, map(int, totals)))
    rows = []
    for q, values in enumerate(matrix):
        c = dict(zip(COUNTS, map(int, values)))
        rows.append({'quartile': q+1, 'counts': c,
                     'foreground_fraction': ratio(c['FG'], c['valid']),
                     'fraction_of_all_GT_FG': ratio(c['FG'], t['FG']),
                     'fraction_of_all_FP': ratio(c['FP'], t['FP']),
                     'fraction_of_all_FN': ratio(c['FN'], t['FN']),
                     'predicted_FG_error_rate': ratio(c['FP'], c['TP']+c['FP']),
                     'predicted_BG_error_rate': ratio(c['FN'], c['TN']+c['FN']),
                     'highconf_FG_error_rate': ratio(c['highFG_FP'], c['highFG_TP']+c['highFG_FP']),
                     'highconf_BG_error_rate': ratio(c['highBG_FN'], c['highBG_TN']+c['highBG_FN']),
                     'fraction_of_all_highconf_FP': ratio(c['highFG_FP'], t['highFG_FP']),
                     'fraction_of_all_highconf_FN': ratio(c['highBG_FN'], t['highBG_FN'])})
    return {'totals': t, 'quartiles': rows,
            'highconf_FP_fraction_of_all_FP': ratio(t['highFG_FP'], t['FP']),
            'highconf_FN_fraction_of_all_FN': ratio(t['highBG_FN'], t['FN'])}


def summarize_episodes(records):
    output = {}
    for cohort in ('R1', 'R5', 'combined'):
        for domain in (*lab.ALL_DOMAINS, 'all_domains_pooled'):
            for active in (True, False):
                rows = [r for r in records if (cohort == 'combined' or r['cohort'] == cohort)
                        and (domain == 'all_domains_pooled' or r['dataset'] == domain) and r['graph_active'] == active]
                if not rows:
                    continue
                values = np.asarray([r['e_quartile_integer_counts'] for r in rows], dtype=np.int64).sum(0)
                output['/'.join((cohort, domain, 'graph_active' if active else 'fallback'))] = dict(
                    n=len(rows), **derived(values))
    return output


def pair_record(a, b, state_a, state_b):
    ea, rhoa, sa, pa, gt, valid = state_a
    eb, rhob, sb, pb, gt_b, valid_b = state_b
    assert np.array_equal(gt, gt_b) and np.array_equal(valid, valid_b)
    pa, pb = pa & valid, pb & valid
    err_a, err_b = (pa ^ gt) & valid, (pb ^ gt) & valid
    masks = {'error': (err_a, err_b), 'FP': (pa & ~gt, pb & ~gt),
             'FN': (~pa & gt, ~pb & gt),
             'highconf_FP': ((sa >= .75) & ~gt & valid, (sb >= .75) & ~gt & valid),
             'highconf_FN': ((sa <= .25) & gt, (sb <= .25) & gt)}
    overlap = {}
    for name, (ma, mb) in masks.items():
        overlap[name] = {'A': int(ma.sum()), 'B': int(mb.sum()), 'intersection': int((ma & mb).sum()),
                         'union': int((ma | mb).sum())}
    spearman = float(spearmanr(ea, eb).statistic)
    return {'dataset': a['binding']['dataset'], 'task_id': a['task_id'],
            'A': [a['cohort'], pc.key(a['binding'])], 'B': [b['cohort'], pc.key(b['binding'])],
            'cross_R1_R5': a['cohort'] != b['cohort'], 'both_graph_active': rhoa <= .9 and rhob <= .9,
            'valid': int(valid.sum()), 'prediction_xor': int((pa ^ pb).sum()),
            'prediction_intersection': int((pa & pb).sum()), 'prediction_union': int((pa | pb).sum()),
            'episode_IoU_A': a['saved_metrics']['IoU'], 'episode_IoU_B': b['saved_metrics']['IoU'],
            'e_spearman': spearman if np.isfinite(spearman) else None,
            'overlap': overlap}


def summarize_pairs(records):
    output = {}
    for domain in (*lab.ALL_DOMAINS, 'all_domains_pooled'):
        for scope in ('all_pairs', 'cross_R1_R5'):
            for active in (False, True):
                rows = [r for r in records if (domain == 'all_domains_pooled' or r['dataset'] == domain)
                        and (scope == 'all_pairs' or r['cross_R1_R5']) and (not active or r['both_graph_active'])]
                if not rows:
                    continue
                sums = {k: sum(r[k] for r in rows) for k in ('valid', 'prediction_xor', 'prediction_intersection', 'prediction_union')}
                error = {}
                for label in rows[0]['overlap']:
                    c = {k: sum(r['overlap'][label][k] for r in rows) for k in ('A', 'B', 'intersection', 'union')}
                    error[label] = dict(counts=c, symmetric_persistence=ratio(2*c['intersection'], c['A']+c['B']),
                                        jaccard=ratio(c['intersection'], c['union']))
                abs_delta = [abs(r['episode_IoU_A']-r['episode_IoU_B']) for r in rows]
                corr = [r['e_spearman'] for r in rows if r['e_spearman'] is not None]
                output['/'.join((domain, scope, 'both_graph_active' if active else 'all_modes'))] = {
                    'pair_n': len(rows), 'query_task_n': len({r['task_id'] for r in rows}), 'counts': sums,
                    'prediction_xor_fraction': ratio(sums['prediction_xor'], sums['valid']),
                    'prediction_iou': ratio(sums['prediction_intersection'], sums['prediction_union']),
                    'episode_abs_IoU_change_mean_pp': float(np.mean(abs_delta)),
                    'episode_abs_IoU_change_median_pp': float(np.median(abs_delta)),
                    'e_spearman_median': float(np.median(corr)) if corr else None, 'error_overlap': error}
    return output


def main():
    started = time.monotonic()
    jobs, spec = prepare()
    records = []
    for i, job in enumerate(jobs, 1):
        records.append(episode(job))
        if i <= 2 or i % 100 == 0:
            print({'phase': 'R6-offline', 'n': i, 'total': len(jobs), 'seconds': round(time.monotonic()-started, 1)}, flush=True)
    write(OUT/'episode_counts.json', {'count_columns': COUNTS, 'records': records})
    groups = defaultdict(list)
    for job in jobs:
        groups[job['task_id']].append(job)
    pair_records, coverage = [], {}
    for domain in lab.ALL_DOMAINS:
        gs = [g for g in groups.values() if g[0]['binding']['dataset'] == domain]
        multiref = [g for g in gs if len({j['reference_id'] for j in g}) > 1]
        coverage[domain] = {'episode_n': sum(len(g) for g in gs), 'query_task_n': len(gs),
                            'multiref_query_task_n': len(multiref), 'multiref_episode_n': sum(len(g) for g in multiref),
                            'cross_R1_R5_query_task_n': sum(len({j['cohort'] for j in g}) == 2 for g in multiref)}
        for group in multiref:
            states = [load(j, verify=False) for j in group]
            for (a, ja), (b, jb) in combinations(enumerate(group), 2):
                if ja['reference_id'] != jb['reference_id']:
                    pair_records.append(pair_record(ja, jb, states[a], states[b]))
    write(OUT/'pair_counts.json', pair_records)
    summary = {
        'state': 'COMPLETE_OFFLINE_DIAGNOSTIC', 'definitions': spec,
        'episode_n': len(records), 'all_saved_scoring_integers_reproduced': True,
        'quartile_count_columns': COUNTS, 'reliability_summary': summarize_episodes(records),
        'same_query_coverage': coverage, 'same_query_pair_summary': summarize_pairs(pair_records),
        'seconds': time.monotonic()-started, 'new_encoder_calls': 0, 'fitted_models': 0,
        'new_candidates': 0, 'post_score_parameter_changes': False,
        'episode_counts_path': str(OUT/'episode_counts.json'), 'episode_counts_sha256': sha(OUT/'episode_counts.json'),
        'pair_counts_path': str(OUT/'pair_counts.json'), 'pair_counts_sha256': sha(OUT/'pair_counts.json'),
        'config_sha256': sha(OUT/'config.json'),
        'limits': ['Existing exposed data only; diagnostic GT cannot enter inference.',
                   'Same-query pair availability is uneven and not a random reference experiment.',
                   'Pixel-pooled summary can be dominated by large masks; per-domain populations are provided.',
                   'Quartiles describe reference explanation density, not a calibrated probability of correctness.',
                   'No FoRIS or INSID3 method comparison is newly inferred from this diagnostic.']}
    write(OUT/'summary.json', summary)
    print({'phase': 'R6-complete', 'episodes': len(records), 'reference_pairs': len(pair_records),
           'seconds': round(time.monotonic()-started, 1), 'summary': str(OUT/'summary.json')}, flush=True)


if __name__ == '__main__':
    main()
