"""Analyze existing W27/W28 S masks; no inference or feature reads.

W27/W28 establish average fusion gains, but not independent error sources or
the source of residual error. At a common .5 threshold, convex score fusion
must contain U0 & Mprime and be contained in U0 | Mprime. Quantify the common
errors this family cannot change, and the corrections/tradeoffs it makes on
disagreements. A GT disagreement oracle is diagnostic only, not a method or
an information bound on DINO features. CRF is outside the convexity statement.

If common errors dominate, choosing between these fixed fields cannot solve
most residual pixel errors; otherwise disagreement resolution merits study.
Neither outcome authorizes changing the frozen method or running a new model.
Only sealed W27/W28 S masks and their S labels are read. Per-case counts are
resumable; all output data go outside the checkout, with compact evidence.

    ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/sh_fusion_error_analysis.py --workers 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'scripts/x1013')]
import w27_sh_single_line as w

c, score = w.c, w.score
OUT = c.OUT / 'x1013_SH_fusion_mechanism'
EVIDENCE = ROOT / 'evidence/research/sh_fusion_error_S.jsonl.gz'
ARMS = ('U0', 'Mprime', 'SH', 'SH+', 'disagreement_oracle')


def forbidden(*args, **kwargs):
    raise RuntimeError('This analysis cannot read features or run inference')


def initialize():
    w.initialize()
    c.rc.raw = forbidden
    w.sh.simple_head = forbidden


def sources():
    jobs, identity = [], {}
    bindings = {str(b['index']): b for b in c.rows()}
    for cohort, expected in (('W27', 1285), ('W28', 871)):
        base = c.task_path(cohort)
        manifest, checks, seal = [c.read(base / name) for name in
                                  ('manifest.json', 'checks.json', 'sealed.json')]
        assert checks['passed'] and checks['n'] == seal['n'] == expected
        assert checks['episodes_sha256'] == w.sha(base / 'episodes.jsonl')
        assert checks['sealed_sha256'] == w.sha(base / 'sealed.json')
        assert seal['predictions_jsonl_sha256'] == w.sha(base / 'predictions.jsonl')
        assert not seal['T_read'] and not seal['new_photo_holdout_read']
        producer = ROOT / 'scripts/x1012' / ('w27_sh_single_line.py' if cohort == 'W27'
                                            else 'w28_sh_second_batch.py')
        assert w.sha(producer) == manifest.get('producer_sha256', manifest.get('code_sha256'))
        records = w.indexed_jsonl(base / 'episodes.jsonl')
        assert len(records) == expected and set(records) == set(manifest['indices'])
        for idx in manifest['indices']:
            row, binding = records[idx], bindings[idx]
            assert row['split'] == 'S' and c.half(binding) == 'S'
            assert row['payload_sha256'] == seal['payload_sha256'][idx]
            assert row['photo'] == c.photo(binding) and row['dataset'] == binding['dataset']
            jobs.append((cohort, binding, row))
        identity[cohort] = {name: w.sha(base / name) for name in
                            ('manifest.json', 'checks.json', 'sealed.json', 'episodes.jsonl')}
    return jobs, identity


def count(mask):
    return int(np.count_nonzero(mask))


def one(job):
    cohort, binding, old = job
    assert c.half(binding) == 'S'
    assert w.sha(old['prediction']) == old['payload_sha256']
    with np.load(old['prediction'], allow_pickle=False) as archive:
        masks = {arm: c.unpack(archive[arm]) for arm in ARMS[:-1]}
    gt, valid = c.truth(binding)
    assert int(gt.sum()) == old['gt']
    for arm, mask in masks.items():
        assert c.iu(mask, gt, valid) == old[arm], f'{cohort}/{old["index"]}/{arm}: I/U mismatch'
    u, m, f, post = [masks[a] & valid for a in ARMS[:-1]]
    disagreement = (u ^ m) & valid
    common_fn, common_fp = (~u & ~m & gt), (u & m & ~gt)
    common_error = common_fn | common_fp
    error, post_error = (f ^ gt) & valid, (post ^ gt) & valid
    assert not np.any((u & m) & ~f) and not np.any(f & ~(u | m))
    assert not np.any((f ^ u) & ~disagreement)
    oracle = (u & m) | (gt & disagreement)
    oracle_iu = c.iu(oracle, gt, valid)
    assert oracle_iu[0] >= old['SH'][0] and oracle_iu[1] <= old['SH'][1]
    counts = {'valid': count(valid), 'gt': count(gt), 'disagreement': count(disagreement),
              'common_fn': count(common_fn), 'common_fp': count(common_fp),
              'pre_fn': count(~f & gt), 'pre_fp': count(f & ~gt),
              'pre_error': count(error), 'disagreement_error': count(disagreement & error),
              'post_error': count(post_error),
              'post_common_remaining': count(common_error & post_error),
              'common_fixed_by_crf': count(common_error & ~post_error)}
    assert counts['pre_error'] == counts['common_fn'] + counts['common_fp'] + counts['disagreement_error']
    for name, single in (('U0', u), ('Mprime', m)):
        edits = {'add_tp': count(f & ~single & gt), 'add_fp': count(f & ~single & ~gt),
                 'delete_tp': count(~f & single & gt), 'delete_fp': count(~f & single & ~gt)}
        counts.update({name + '_' + key: value for key, value in edits.items()})
        corrected, introduced = edits['add_tp'] + edits['delete_fp'], edits['add_fp'] + edits['delete_tp']
        assert count((single ^ gt) & valid) - counts['pre_error'] == corrected - introduced
    keep = ('index', 'dataset', 'fold', 'cls', 'photo', 'split', 'fallback')
    return dict({key: old[key] for key in keep}, cohort=cohort, counts=counts,
                **{arm: old[arm] for arm in ARMS[:-1]}, disagreement_oracle=oracle_iu)


def ratio(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def summarize(rows):
    totals = {key: sum(row['counts'][key] for row in rows) for key in rows[0]['counts']}
    ratios = {
        'disagreement_pct_valid': ratio(totals['disagreement'], totals['valid']),
        'common_pct_pre_error': ratio(totals['common_fn'] + totals['common_fp'], totals['pre_error']),
        'common_pct_pre_FN': ratio(totals['common_fn'], totals['pre_fn']),
        'common_pct_pre_FP': ratio(totals['common_fp'], totals['pre_fp']),
        'original_common_pct_post_error': ratio(totals['post_common_remaining'], totals['post_error']),
        'CRF_fixed_pct_original_common': ratio(totals['common_fixed_by_crf'],
                                              totals['common_fn'] + totals['common_fp'])}
    scores = {arm: score.miou(rows, arm) for arm in ARMS}
    return {'n': len(rows), 'scores': scores,
            'oracle_minus_SH_pp': scores['disagreement_oracle'] - scores['SH'],
            'pixel_totals': totals, 'pixel_pooled_percent': ratios}


def scope(rows):
    domains = {domain: summarize([row for row in rows if row['dataset'] == domain])
               for domain in c.DOMAINS if any(row['dataset'] == domain for row in rows)}
    macro_scores = {arm: float(np.mean([v['scores'][arm] for v in domains.values()])) for arm in ARMS}
    return {'n': len(rows), 'domains': domains, 'domain_equal_mean_scores': macro_scores,
            'domain_equal_oracle_minus_SH_pp': macro_scores['disagreement_oracle'] - macro_scores['SH'],
            'domain_equal_mean_percent': {key: float(np.mean([v['pixel_pooled_percent'][key]
                for v in domains.values() if v['pixel_pooled_percent'][key] is not None]))
                for key in next(iter(domains.values()))['pixel_pooled_percent']},
            'all_pixels_pooled': summarize(rows)['pixel_pooled_percent']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, source_identity = sources()
    identity = {'analysis_sha256': w.sha(__file__), 'sources': source_identity,
                'cohorts': {'W27': 1285, 'W28': 871}, 'feature_reads': 0, 'encoder_calls': 0,
                'head_calls': 0, 'CRF_calls': 0, 'T_read': False, 'new_photo_holdout_read': False,
                'ratios': 'Descriptive pixel counts, pooled within domain; not IoU contributions',
                'oracle': 'Query-GT diagnostic on pre-CRF hard-mask disagreements only'}
    identity_path = OUT / 'identity.json'
    if identity_path.exists():
        assert c.read(identity_path) == identity
    else:
        c.write(identity_path, identity)
    stamp = time.time_ns()
    top = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True, check=True).stdout
    (OUT / f'top_{stamp}.txt').write_text(top)
    path = OUT / 'counts.jsonl'
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row['cohort'], row['index']
            assert key not in done
            done[key] = row
    todo = [job for job in jobs if (job[0], str(job[1]['index'])) not in done]
    started = time.perf_counter()
    print(f'Existing sealed masks only: {len(jobs)} cases, {len(todo)} remaining, {args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, path.open('a') as stream:
        for n, row in enumerate(pool.map(one, todo, chunksize=4), 1):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            done[row['cohort'], row['index']] = row
            if n % 200 == 0 or n == len(todo):
                os.fsync(stream.fileno())
                print(f'Analyzed {len(done)}/{len(jobs)}, {time.perf_counter()-started:.1f}s', flush=True)
    assert set(done) == {(cohort, str(b['index'])) for cohort, b, _ in jobs}
    rows = [done[cohort, str(b['index'])] for cohort, b, _ in jobs]
    report = {'identity': identity, 'counts_sha256': w.sha(path), 'checks': {
        'S_only': True, 'sealed_IU_reproduced': len(rows), 'payload_hashes_verified': len(rows),
        'convex_containment_cases': len(rows), 'pixel_error_conservation_cases': len(rows)},
        'scopes': {}, 'analysis_wall_seconds_this_run': time.perf_counter() - started}
    for cohort in ('W27', 'W28'):
        subset = [row for row in rows if row['cohort'] == cohort]
        report['scopes'][cohort + '_full'] = scope(subset)
        report['scopes'][cohort + '_active'] = scope([row for row in subset if not row['fallback']])
    common_domains = {row['dataset'] for row in rows if row['cohort'] == 'W28'}
    report['scopes']['W27_common6_active'] = scope([row for row in rows if row['cohort'] == 'W27'
                                                  and row['dataset'] in common_domains and not row['fallback']])
    c.write(OUT / 'report.json', report)
    with gzip.open(EVIDENCE, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    for name, result in report['scopes'].items():
        print(name, json.dumps({'n': result['n'], 'percent': result['domain_equal_mean_percent'],
              'oracle_minus_SH_pp': result['domain_equal_oracle_minus_SH_pp']}), flush=True)
    print('PASS: hashes, saved I/U, convex containment, and error-count identities.', flush=True)


if __name__ == '__main__':
    main()
