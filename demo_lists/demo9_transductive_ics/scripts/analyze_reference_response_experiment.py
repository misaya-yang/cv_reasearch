#!/usr/bin/env python3
"""Analyze frozen finite response pilot I/U receipts; JSON stdout only.

No masks, labels, tensor/model imports, fitting, downloads or file writes.
Photo-connected blocks join ALL support/query roles, including across folds.
Percentile cluster bootstrap is paired across arms; it is exploratory on reused
DEPENDENT development photos, never an independent method-score certificate.
When a replicate omits classes its class mean uses only represented classes;
this finite-cluster resampling is not a fixed-80-class population interval.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

BASE = Path(__file__).resolve().parents[1]
# The existing tics package initializer imports torch. Only this exact pure
# protocol module is loaded; analysis itself never initializes the package.
_spec = importlib.util.spec_from_file_location(
    '_reference_response_analysis_protocol', BASE/'tics/reference_response_protocol.py')
_protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_protocol)
coco_photo_id = _protocol.coco_photo_id

ARMS = ('foris_public', 'static', 'response')
LEDGER = ('recovered_fn', 'lost_tp', 'added_fp', 'removed_fp')
SEED = 31027
REPLICATES = 2000


def _iu(pair):
    if (not isinstance(pair, (list, tuple)) or len(pair) != 2
            or any(type(value) is not int or value < 0 for value in pair)
            or pair[0] > pair[1]):
        raise ValueError('Original-resolution I/U must be two nonnegative integers with I <= U')
    return pair


def validate_records(report):
    if report.get('schema') != 'reference_response_experiment_v1' or report.get('state') != 'COMPLETED_PILOT':
        raise ValueError('Expected completed finite-pilot report schema/state')
    if report.get('test_exposure_scope') != 'reused_development':
        raise ValueError('This analyzer is bounded to the explicitly reused development pilot')
    rows = report.get('records')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty frozen pilot records required')
    ids = set()
    required = {'id', 'c', 'fold', 'support', 'query', 'original_iu', 'ledger', 'response_audit'}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError('Missing frozen record metadata/IU/ledger/audit')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('Unique nonempty episode IDs required')
        ids.add(row['id'])
        if (type(row['fold']) is not int or row['fold'] not in range(4) or type(row['c']) is not int
                or row['c'] not in range(80) or row['c'] % 4 != row['fold']):
            raise ValueError('Official held-class/fold mismatch')
        for role in ('support', 'query'):
            coco_photo_id(row[role])
            if not row[role].startswith('val2014/'):
                raise ValueError('Pilot must retain original COCO2014 validation logical paths')
        if coco_photo_id(row['support']) == coco_photo_id(row['query']):
            raise ValueError('Query must differ from its labelled reference')
        if not isinstance(row['original_iu'], dict) or set(row['original_iu']) != set(ARMS):
            raise ValueError('All three matched arms are required in every record')
        for pair in row['original_iu'].values():
            _iu(pair)
        if not isinstance(row['ledger'], dict) or set(row['ledger']) != {'static', 'response'}:
            raise ValueError('Both learned-arm pixel ledgers are required')
        for ledger in row['ledger'].values():
            if (not isinstance(ledger, dict) or set(ledger) != set(LEDGER)
                    or any(type(value) is not int or value < 0 for value in ledger.values())):
                raise ValueError('Pixel ledger must contain four nonnegative integer counts')
        if not isinstance(row['response_audit'], dict):
            raise ValueError('Caller response/freeze audit must be a mapping')
        if 'original_oracle_iu' in row:
            oracle = row['original_oracle_iu']
            if not isinstance(oracle, dict) or set(oracle) != {'pixel_false_positive_removed'}:
                raise ValueError('Only the declared pixel false-positive-removal diagnostic is allowed')
            _iu(oracle['pixel_false_positive_removed'])
    # Reject nonfinite data anywhere, including an unchecked source/audit field.
    json.dumps(report, allow_nan=False)
    return rows


def class_scores(rows, arms=ARMS):
    totals = {arm: {} for arm in arms}
    episodes = Counter()
    for row in rows:
        episodes[row['c']] += 1
        for arm in arms:
            pair = row['original_iu'][arm]
            current = totals[arm].setdefault(row['c'], [0, 0])
            current[0] += pair[0]
            current[1] += pair[1]
    scores, table = {}, {}
    for arm in arms:
        if not totals[arm] or any(union == 0 for _, union in totals[arm].values()):
            raise ValueError('Every represented class must have positive aggregate union in every arm')
        scores[arm] = 100*sum(i/u for i, u in totals[arm].values())/len(totals[arm])
    for category in sorted(episodes):
        table[str(category)] = dict(episodes=episodes[category], arms={
            arm: dict(I=totals[arm][category][0], U=totals[arm][category][1],
                      iou_percent=100*totals[arm][category][0]/totals[arm][category][1]) for arm in arms})
    return dict(class_miou_percent=scores, classes=len(episodes), episodes=len(rows), class_table=table)


def photo_connected_groups(rows):
    """Transitive UID components across support/query positions AND folds."""
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    owner = {}
    for index, row in enumerate(rows):
        for role in ('support', 'query'):
            uid = coco_photo_id(row[role])
            if uid in owner:
                parent[find(index)] = find(owner[uid])
            else:
                owner[uid] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda indices: indices[0])


def _percentile(values, probability):
    ordered = sorted(values)
    offset = probability*(len(ordered)-1)
    low = int(offset)
    high = min(low+1, len(ordered)-1)
    return ordered[low]+(offset-low)*(ordered[high]-ordered[low])


def paired_cluster_bootstrap(rows, groups, *, deadline):
    # Fold-only components are sampled within their fold. Cross-fold components
    # remain whole and are sampled within the identical fold-set stratum. This
    # never breaks shared photos merely to fabricate fold-independent units.
    strata = {}
    for group in groups:
        folds = tuple(sorted({rows[index]['fold'] for index in group}))
        strata.setdefault(folds, []).append(group)
    rng = random.Random(SEED)
    comparisons = {'response_minus_static': [], 'response_minus_foris_public': []}
    class_counts = []
    attempts = 0
    while len(class_counts) < REPLICATES:
        if time.monotonic() >= deadline:
            raise TimeoutError('Finite CPU analysis exceeded its wall-time budget')
        attempts += 1
        if attempts > REPLICATES*10:
            raise ValueError('Too many zero-union bootstrap draws; no interval fabricated')
        sampled = []
        for blocks in strata.values():
            for _ in blocks:
                sampled.extend(rows[index] for index in rng.choice(blocks))
        try:
            summary = class_scores(sampled)
        except ValueError:
            continue
        score = summary['class_miou_percent']
        comparisons['response_minus_static'].append(score['response']-score['static'])
        comparisons['response_minus_foris_public'].append(score['response']-score['foris_public'])
        class_counts.append(summary['classes'])
    original = class_scores(rows)['class_miou_percent']
    result = {}
    for name, values in comparisons.items():
        control = 'static' if name.endswith('static') else 'foris_public'
        low, high = _percentile(values, .025), _percentile(values, .975)
        result[name] = dict(difference_pp=original['response']-original[control],
                            exploratory_95_percentile_interval_pp=[low, high],
                            finite_pilot_signal='positive' if low > 0 else 'negative' if high < 0 else 'unresolved',
                            fraction_resamples_positive=sum(value > 0 for value in values)/len(values))
    return dict(comparisons=result, seed=SEED, replicates=REPLICATES, attempted_replicates=attempts,
                resampling_unit='whole all-role photo-UID connected components',
                fold_policy='stratify by component fold-set; cross-fold shared-photo components never split',
                strata=[dict(folds=list(folds), connected_groups=len(blocks)) for folds, blocks in sorted(strata.items())],
                replicate_represented_classes=dict(min=min(class_counts), max=max(class_counts)),
                class_mean_policy='sum I/U per represented class; equal mean over classes present in each replicate',
                episode_IID_bootstrap=False, population_or_independent_confirmation_interval=False,
                singleton_strata_no_bootstrap_variation=[list(folds) for folds, blocks in sorted(strata.items()) if len(blocks) == 1])


def _training_summary(report, scored_folds):
    receipts = report.get('training_receipts')
    if not isinstance(receipts, list):
        raise ValueError('Explicit training_receipts list required, even if empty')
    result = []
    seen = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or type(receipt.get('fold')) is not int or receipt['fold'] not in range(4):
            raise ValueError('Each training receipt requires its fold')
        if receipt['fold'] in seen:
            raise ValueError('Duplicate training receipt fold')
        seen.add(receipt['fold'])
        body = receipt.get('receipt', receipt)
        if not isinstance(body, dict):
            raise ValueError('Invalid training receipt body')
        state = body.get('state')
        epochs = body.get('completed_epochs')
        if state not in ('CONVERGED_PATIENCE', 'PARTIAL_BUDGET', 'ERROR') or type(epochs) is not int or epochs < 0:
            raise ValueError('Declared finite trainer state/completed epochs required')
        last_loss = body.get('fitting_loss', body.get('train_loss'))
        if last_loss is None and body.get('history'):
            last_loss = body['history'][-1].get('train_loss')
        result.append(dict(fold=receipt['fold'], state=state, completed_epochs=epochs,
                           best=body.get('best'), last_completed_epoch_fitting_loss=last_loss,
                           reason=body.get('reason'), sufficient_real_task_training_established=False))
    return sorted(result, key=lambda value: value['fold']), sorted(set(scored_folds)-seen)


def analyze_report(report, *, timeout_seconds=60.):
    """Importable root-driver interface; report is small plain JSON metadata.

    Per-fold training_receipts may be flat {fold, state, completed_epochs, best,
    history/fitting_loss} or {fold, receipt:<native trainer receipt>}.
    Only the producer can verify prediction freezing and original-resolution
    masks. This analyzer preserves caller audits and never treats them as an
    independently verified sequencing proof.
    """
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError('Analysis timeout must be finite and at most60 CPU seconds')
    started = time.monotonic()
    deadline = started+timeout_seconds
    rows = validate_records(report)
    overall = class_scores(rows)
    folds = sorted({row['fold'] for row in rows})
    fold_scores = {str(fold): class_scores([row for row in rows if row['fold'] == fold]) for fold in folds}
    groups = photo_connected_groups(rows)
    group_audit = [dict(episode_ids=[rows[index]['id'] for index in group],
                        folds=sorted({rows[index]['fold'] for index in group}),
                        photo_uids=sorted({coco_photo_id(rows[index][role]) for index in group for role in ('support', 'query')}))
                   for group in groups]
    bootstrap = paired_cluster_bootstrap(rows, groups, deadline=deadline)
    training, missing = _training_summary(report, folds)
    partial = [receipt['fold'] for receipt in training if receipt['state'] == 'PARTIAL_BUDGET']
    errors = [receipt['fold'] for receipt in training if receipt['state'] == 'ERROR']
    ledger_totals = {arm: {key: sum(row['ledger'][arm][key] for row in rows) for key in LEDGER}
                     for arm in ('static', 'response')}
    diagnostics = {}
    oracle_rows = [row for row in rows if 'original_oracle_iu' in row]
    if oracle_rows:
        transformed = [dict(row, original_iu={'pixel_false_positive_removed': row['original_oracle_iu']['pixel_false_positive_removed']})
                       for row in oracle_rows]
        diagnostics['pixel_false_positive_removed'] = dict(
            **class_scores(transformed, arms=('pixel_false_positive_removed',)),
            scored_rows=len(oracle_rows), total_rows=len(rows), query_GT_diagnostic=True,
            deployable_method=False, method_comparison_or_selection=False,
            diagnostic_scope='Only removal of predicted false-positive pixels; not a full oracle solution.')
    warnings = ['Reused DEPENDENT development photos; no independent method-score or generalization claim.',
                'Finite small pilot subset; all-four-fold coverage does not imply a standard full protocol.',
                'Budget-limited or insufficient fitting cannot establish information unavailability or method invalidity.',
                'Caller freezes all three masks before first query-GT pixels; count-only analysis cannot independently verify that boundary.',
                'Bootstrap means use represented classes per draw; intervals are exploratory finite-photo-group summaries.']
    if partial:
        warnings.append('PARTIAL_BUDGET training remains explicit: folds '+str(partial)+'.')
    if errors or missing:
        warnings.append('Incomplete training provenance: ERROR folds '+str(errors)+'; missing receipts '+str(missing)+'.')
    result = dict(schema='reference_response_analysis_v1', state='ANALYZED_COMPLETED_PILOT',
                  source_experiment_state=report['state'], metric='original_resolution_equal_class_mean_of_sum_I_over_sum_U',
                  overall=overall, by_fold=fold_scores,
                  fold_coverage=dict(scored_folds=folds, all_four_folds=folds == [0, 1, 2, 3],
                                     missing_folds=sorted(set(range(4))-set(folds)),
                                     episodes_per_fold={str(fold): sum(row['fold'] == fold for row in rows) for fold in folds}),
                  paired_photo_cluster_bootstrap=bootstrap, connected_photo_groups=group_audit,
                  training=training, partial_budget_training_folds=partial, error_training_folds=errors,
                  missing_training_receipt_folds=missing, pixel_ledger_totals=ledger_totals,
                  ledger_reference=report.get('ledger_reference', 'caller_reference_not_declared'),
                  caller_response_audits={row['id']: row['response_audit'] for row in rows},
                  query_GT_freeze_boundary_independently_verified=False,
                  controls=dict(naive='static: same information/supervision/capacity, distinct optimizer after matched initialization',
                                strong='foris_public: complete public FoRIS pipeline', control_fairness_independently_verified=False),
                  diagnostics=diagnostics,
                  qualification=dict(test_exposure_scope='reused_development', dependent_development_sample=True,
                                     small_finite_pilot=True, sufficient_training_established=False,
                                     independent_method_score_claim=False, information_unavailable_claim=False,
                                     method_invalidity_claim=False, full_protocol_claim=False),
                  warnings=warnings, elapsed_seconds=time.monotonic()-started, torch_imported='torch' in sys.modules)
    if result['torch_imported']:
        raise AssertionError('CPU analysis unexpectedly imported torch')
    json.dumps(result, allow_nan=False)
    return result


def cpu_check():
    checks = []
    def record(name, **evidence):
        checks.append(dict(name=name, passed=True, **evidence))
    def row(identity, category, fold, support, query, *, control=(1, 10), response=(3, 10)):
        return dict(id=identity, c=category, fold=fold,
                    support=f'val2014/COCO_val2014_{support:012d}.jpg',
                    query=f'val2014/COCO_val2014_{query:012d}.jpg',
                    original_iu=dict(foris_public=list(control), static=list(control), response=list(response)),
                    ledger={arm: dict(recovered_fn=0, lost_tp=0, added_fp=0, removed_fp=0) for arm in ('static', 'response')},
                    response_audit=dict(caller_masks_frozen_before_query_GT=True))
    def report(rows):
        return dict(schema='reference_response_experiment_v1', state='COMPLETED_PILOT', records=rows,
                    test_exposure_scope='reused_development', training_receipts=[
                        dict(fold=fold, state='PARTIAL_BUDGET', completed_epochs=1,
                             best=dict(static=dict(epoch=1), response=dict(epoch=1)), fitting_loss=dict(static=1., response=1.))
                        for fold in sorted({value['fold'] for value in rows})])
    unequal = [row('a', 0, 0, 1, 2, control=(9, 10)), row('b', 0, 0, 3, 4, control=(0, 100))]
    scores = class_scores(unequal)['class_miou_percent']
    assert abs(scores['static']-100*9/110) < 1e-12 and scores['static'] != 45.
    record('sum_I_over_sum_U_per_class_differs_from_episode_mean', expected_percent=100*9/110)
    chain = [row('a', 0, 0, 10, 11), row('b', 1, 1, 11, 12), row('c', 2, 2, 12, 13), row('d', 3, 3, 20, 21)]
    assert photo_connected_groups(chain) == [[0, 1, 2], [3]]
    record('transitive_support_query_shared_UID_groups_cross_fold_without_splitting')
    rows = [row(f'f{fold}_e{episode}', fold, fold, 100+100*fold+2*episode, 101+100*fold+2*episode)
            for fold in range(4) for episode in range(3)]
    identical = deepcopy(rows)
    for value in identical:
        value['original_iu']['response'] = list(value['original_iu']['static'])
    zero = analyze_report(report(identical))
    for paired in zero['paired_photo_cluster_bootstrap']['comparisons'].values():
        assert paired['difference_pp'] == 0 and paired['exploratory_95_percentile_interval_pp'] == [0., 0.]
    record('matched_identical_arms_have_exact_zero_difference_and_interval')
    positive = analyze_report(report(rows))
    assert all(value['exploratory_95_percentile_interval_pp'][0] > 0
               for value in positive['paired_photo_cluster_bootstrap']['comparisons'].values())
    negative_rows = deepcopy(rows)
    for value in negative_rows:
        value['original_iu']['response'] = [0, 10]
    negative = analyze_report(report(negative_rows))
    assert all(value['exploratory_95_percentile_interval_pp'][1] < 0
               for value in negative['paired_photo_cluster_bootstrap']['comparisons'].values())
    record('paired_interval_sign_matches_constant_positive_and_negative_fixture')
    assert positive['fold_coverage']['all_four_folds'] and positive['partial_budget_training_folds'] == [0, 1, 2, 3]
    assert not positive['qualification']['independent_method_score_claim']
    record('four_fold_coverage_preserved_with_explicit_partial_training_and_dependent_pilot_qualification')
    rejected = 0
    for mutation in ('missing_arm', 'negative', 'intersection_exceeds_union', 'bool', 'float', 'zero_class_union'):
        invalid = report(deepcopy(rows))
        value = invalid['records'][0]
        if mutation == 'missing_arm':
            del value['original_iu']['response']
        elif mutation == 'zero_class_union':
            for value in invalid['records']:
                value['original_iu']['response'] = [0, 0]
        else:
            value['original_iu']['response'] = {'negative': [-1, 1], 'intersection_exceeds_union': [2, 1],
                                               'bool': [True, 2], 'float': [1., 2]}[mutation]
        try:
            analyze_report(invalid)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid/missing matched I/U accepted')
    assert rejected == 6
    record('missing_arm_and_invalid_IU_rejected', rejected_cases=6)
    oracle = report(deepcopy(rows))
    for value in oracle['records']:
        value['original_oracle_iu'] = dict(pixel_false_positive_removed=[1, 2])
    diagnostic = analyze_report(oracle)
    assert diagnostic['overall'] == positive['overall']
    assert diagnostic['paired_photo_cluster_bootstrap'] == positive['paired_photo_cluster_bootstrap']
    assert diagnostic['diagnostics']['pixel_false_positive_removed']['query_GT_diagnostic']
    assert not diagnostic['diagnostics']['pixel_false_positive_removed']['deployable_method']
    record('query_GT_oracle_diagnostic_never_enters_method_comparison_or_selection')
    one_fold = analyze_report(report(rows[:3]))
    assert not one_fold['fold_coverage']['all_four_folds'] and one_fold['fold_coverage']['missing_folds'] == [1, 2, 3]
    record('missing_fold_reported_without_full_protocol_or_method_failure_claim')
    singleton = analyze_report(report(chain))
    assert singleton['paired_photo_cluster_bootstrap']['singleton_strata_no_bootstrap_variation'] == [[0, 1, 2], [3]]
    assert singleton['connected_photo_groups'][0]['folds'] == [0, 1, 2]
    record('cross_fold_components_remain_whole_and_singleton_stratum_uncertainty_limit_explicit')
    assert 'torch' not in sys.modules
    return dict(schema='reference_response_analysis_cpu_v1', state='CPU_ANALYSIS_CHECK_PASSED', passed=len(checks),
                checks=checks, seed=SEED, bootstrap_replicates=REPLICATES, torch_imported=False,
                real_task_gain_measured=False, files_written=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=60.)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.cpu_check == (args.report is not None):
        parser.error('Choose exactly one of --report or --cpu-check')
    if args.out is not None and args.out.exists():
        raise ValueError('Preserve the previous analysis/CPU receipt')
    if args.cpu_check:
        output = cpu_check()
    else:
        raw = args.report.read_bytes()
        output = analyze_report(json.loads(raw), timeout_seconds=args.timeout_seconds)
        output['report_sha256'] = hashlib.sha256(raw).hexdigest()
    output['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(output, stream, allow_nan=False)
    print(json.dumps(output, allow_nan=False))


if __name__ == '__main__':
    main()
