#!/usr/bin/env python3
"""CPU-only paired analysis of the frozen full-rank angular mechanism, without fitting."""
import argparse
import base64
import json
import zlib
from pathlib import Path
import numpy as np

ARMS = ('foris_crf', 'native_identity', 'paired_geometry', 'pooled_geometry',
        'fg_aug_geometry', 'shuffled_geometry', 'frost95_geometry')
LEDGER = ('recovered_fn', 'lost_tp', 'added_fp', 'removed_fp')


from analyze_rice_core import digest, task, counts, class_table, aggregate, paired


def unpack_mask(value):
    shape = value['shape']
    if not shape or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError('Invalid packed mask shape')
    if value.get('bitorder', 'big') != 'big':
        raise ValueError('Only declared big-bitorder masks accepted')
    if value.get('encoding', 'numpy.packbits-big/base64') != 'numpy.packbits-big/base64':
        raise ValueError('Unknown packed mask encoding')
    raw = base64.b64decode(value.get('data', value.get('base64')), validate=True)
    codec = value.get('codec')
    if codec == 'zlib_np_packbits_big':
        raw = zlib.decompress(raw)
    elif codec is not None:
        raise ValueError('Unknown packed mask codec')
    size = int(np.prod(shape))
    if len(raw) != (size+7)//8:
        raise ValueError('Packed mask length does not match shape')
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='big')[:size].reshape(shape)


def analyze(report, manifest, report_sha=None, manifest_sha=None):
    if report.get('state') != 'COMPLETED':
        raise ValueError('Only the completed frozen angular experiment is analyzed')
    if manifest.get('schema') != 'native_angular_assets_v1' or manifest.get('state') != 'PREPARED_ASSETS':
        raise ValueError('Frozen native angular assets manifest required')
    arms = tuple(manifest.get('arms', []))
    if len(arms) != 7 or len(set(arms)) != 7 or set(arms) != set(ARMS):
        raise ValueError('Frozen seven-arm full-rank angular comparison required')
    expected = {task(r): r for r in manifest['frozen_episodes']}
    rows = report['records']
    if len(expected) != 40 or len(rows) != 40 or len({task(r) for r in rows}) != 40:
        raise ValueError('Exactly forty unique prepared/observed tasks required')
    if set(expected) != {task(r) for r in rows}:
        raise ValueError('Actual task identities differ from frozen manifest')
    if any(sum(k[0] == f for k in expected) != 10 for f in range(4)):
        raise ValueError('Exactly ten tasks per fold0..3 required')
    photo_tasks = {}
    mask_checked = 0
    for row in rows:
        if row.get('replay_identity_exact') is not True:
            raise ValueError('Actual native identity replay must be asserted exact in every task')
        frozen = expected[task(row)]
        for role in ('support', 'query'):
            if row.get(role) != frozen[role]:
                raise ValueError('Actual support/query photo differs from frozen task')
            photo_tasks.setdefault(row[role], set()).add(task(row))
        for field in ('iu', 'original_iu'):
            if set(row.get(field, {})) != set(arms):
                raise ValueError('Missing/extra scored arm: ' + field)
            for arm in arms:
                counts(row[field][arm])
            if row[field]['native_identity'] != row[field]['foris_crf']:
                raise ValueError('Native identity replay changes '+field+' counts')
        bits = row.get('prediction_bits', {})
        if bits:
            if 'foris_crf' not in bits or 'native_identity' not in bits:
                raise ValueError('Saved masks must include both native and replay')
            native, replay = unpack_mask(bits['foris_crf']), unpack_mask(bits['native_identity'])
            if native.shape != replay.shape or not np.array_equal(native, replay):
                raise ValueError('Packed native identity mask differs from full FoRIS')
            mask_checked += 1
    repeated = [dict(photo=p, tasks=[list(k) for k in sorted(tasks)])
                for p, tasks in sorted(photo_tasks.items()) if len(tasks) > 1]
    scores, comparisons, oracle = {}, {}, {}
    for field in ('iu', 'original_iu'):
        scores[field] = {}
        for arm in arms:
            table, totals = class_table(rows, arm, field)
            scores[field][arm] = {**aggregate(table), 'per_class': [dict(fold=k[0], c=k[1],
                I=int(totals[k][0]), U=int(totals[k][1]), iou=table[k]) for k in sorted(table)]}
        comparisons[field] = {arm+' vs foris_crf': paired(rows, arm, 'foris_crf', field)
                             for arm in arms if arm != 'foris_crf'}
        for baseline in ('native_identity', 'pooled_geometry', 'fg_aug_geometry', 'shuffled_geometry', 'frost95_geometry'):
            comparisons[field]['paired_geometry vs '+baseline] = paired(rows, 'paired_geometry', baseline, field)
        selected = []
        choices = []
        for row in rows:
            # Stable arm order resolves ties; GT selection is diagnostic only.
            arm = max(arms, key=lambda name: row[field][name][0] / max(row[field][name][1], 1))
            selected.append({**row, field: {'episode_oracle': row[field][arm]}})
            choices.append(dict(fold=row['fold'], e=row['e'], c=row['c'], arm=arm))
        table, _ = class_table(selected, 'episode_oracle', field)
        oracle[field] = {**aggregate(table), 'choices': choices,
            'scope': 'GT chooses best frozen arm per episode; finite bank diagnostic, NOT a class-mIoU ceiling or deployable method'}
    harm = {}
    for arm in arms:
        entries = []
        for row in rows:
            ledger = row.get('ledger', {}).get(arm)
            if ledger is None:
                raise ValueError('Missing native FoRIS harm ledger for ' + arm)
            if set(ledger) != set(LEDGER) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ledger.values()):
                raise ValueError('Invalid harm ledger')
            current, anchor = row['iu'][arm], row['iu']['foris_crf']
            if current[0]-anchor[0] != ledger['recovered_fn']-ledger['lost_tp']:
                raise ValueError('Harm ledger disagrees with intersection change')
            if current[1]-anchor[1] != ledger['added_fp']-ledger['removed_fp']:
                raise ValueError('Harm ledger disagrees with union change')
            if arm in ('foris_crf', 'native_identity') and any(ledger.values()):
                raise ValueError('Exact native identity must have zero harm ledger')
            entries.append(ledger)
        harm[arm] = {k: sum(r[k] for r in entries) for k in LEDGER}
        harm[arm]['per_fold'] = {str(f): {k: sum(row['ledger'][arm][k] for row in rows if row['fold'] == f)
                                              for k in LEDGER} for f in range(4)}
        harm[arm]['scope'] = 'Model1024 pixel counts relative to native foris_crf; not original-resolution counts'
    # These diagnostic strata explain the fixed intervention; they never pick
    # an arm, fit a gate or change a deployment threshold using query labels.
    evidence = {}
    if any('evidence_chain' in row for row in rows):
        if not all('evidence_chain' in row for row in rows):
            raise ValueError('Incomplete shared-pass evidence chain')
        for arm in arms:
            transitions = []
            for row in rows:
                chain = row['evidence_chain']
                if chain.get('query_GT_used_only_after_all_predictions') is not True or chain.get('diagnostic_scores_never_drive_selection') is not True:
                    raise ValueError('Query diagnostics must not drive prediction')
                ref, q = chain['reference_orbit'][arm], chain['query_score'][arm]
                ref0, q0 = chain['reference_orbit']['foris_crf'], chain['query_score']['foris_crf']
                if ref['state'] != 'SCORED' or q['state'] != 'SCORED':
                    continue
                ref_delta = ref['mean_squared_chord_to_view_mean']-ref0['mean_squared_chord_to_view_mean']
                auc_delta = q['auc']-q0['auc']
                fpr_delta = q['fpr_at_at_least_95tpr']-q0['fpr_at_at_least_95tpr']
                a,b = row['original_iu'][arm],row['original_iu']['foris_crf']
                iou_delta = 100*(a[0]/max(a[1],1)-b[0]/max(b[1],1))
                transitions.append(dict(fold=row['fold'],e=row['e'],c=row['c'],
                    reference_chord_scatter_delta=ref_delta,query_auc_delta=auc_delta,
                    query_fpr95_delta=fpr_delta,original_episode_iou_delta_pp=iou_delta))
            evidence[arm] = dict(valid_tasks=len(transitions), per_task=transitions,
                reference_more_stable_query_auc_worse=sum(r['reference_chord_scatter_delta']<0 and r['query_auc_delta']<0 for r in transitions),
                query_auc_better_final_mask_worse=sum(r['query_auc_delta']>0 and r['original_episode_iou_delta_pp']<0 for r in transitions),
                query_auc_and_final_mask_better=sum(r['query_auc_delta']>0 and r['original_episode_iou_delta_pp']>0 for r in transitions),
                scope='Descriptive trajectories on reused development tasks; not causal proof, calibration or a new selection method')
    return dict(state='CPU_FROZEN_OUTPUT_ANALYSIS', report_sha256=report_sha, manifest_sha256=manifest_sha,
        records=40, arms=list(arms), aggregation='Within class sumI/sumU, within fold class mean, equal mean across four folds',
        model_and_original_resolution_separate=True, scores=scores, comparisons=comparisons,
        harm_ledger=harm, evidence_chain=evidence, episode_oracle=oracle, repeated_all_role_photos=repeated,
        native_identity=dict(runtime_exact_assertions=40, both_resolution_IU_equal=True,
            packed_mask_independent_checks=mask_checked,
            qualification='Missing packed masks rely on runtime exact replay assertion; I/U equality alone does not prove mask identity'),
        independent_class_bootstrap_photo_warning=bool(repeated),
        methods_audit_records=[dict(fold=row['fold'], e=row['e'], c=row['c'], methods=row.get('methods', {}))
                               for row in rows],
        scope='Same forty development tasks as completed RICE core; not an independent new test; no full benchmark claim')


def self_check():
    manifest = dict(schema='native_angular_assets_v1', state='PREPARED_ASSETS', frozen_episodes=[], arms=list(ARMS))
    rows = []
    for fold in range(4):
        for e in range(10):
            row = dict(fold=fold, e=e, c=4*e+fold, support='s'+str(fold)+'_'+str(e), query='q'+str(fold)+'_'+str(e))
            manifest['frozen_episodes'].append(dict(row))
            row.update(replay_identity_exact=True, iu={a: [2, 4] for a in ARMS}, original_iu={a: [1, 4] for a in ARMS},
                       ledger={a: {k: 0 for k in LEDGER} for a in ARMS})
            rows.append(row)
    report = dict(state='COMPLETED', records=rows)
    result = analyze(report, manifest)
    assert result['scores']['iu']['foris_crf']['fold_mean_class_miou'] == 50
    assert result['scores']['original_iu']['foris_crf']['fold_mean_class_miou'] == 25
    assert result['comparisons']['iu']['paired_geometry vs foris_crf']['exploratory_delta95_pp'] == [0., 0.]
    rows[0]['iu']['paired_geometry'] = [3, 4]
    rows[0]['ledger']['paired_geometry']['recovered_fn'] = 1
    changed = analyze(report, manifest)
    assert changed['episode_oracle']['iu']['choices'][0]['arm'] == 'paired_geometry'
    assert changed['comparisons']['iu']['paired_geometry vs foris_crf']['delta_pp'] == .625
    assert changed['comparisons']['original_iu']['paired_geometry vs foris_crf']['delta_pp'] == 0
    mask = dict(shape=[2, 2], bitorder='big', base64=base64.b64encode(bytes([0b10100000])).decode())
    rows[0]['prediction_bits'] = dict(foris_crf=mask, native_identity=dict(mask))
    assert analyze(report, manifest)['native_identity']['packed_mask_independent_checks'] == 1
    rows[0]['prediction_bits']['native_identity'] = {**mask, 'base64': base64.b64encode(bytes([0b01010000])).decode()}
    try: analyze(report, manifest)
    except ValueError: pass
    else: raise AssertionError('Equal-count but different replay masks accepted')
    rows[0]['prediction_bits']['native_identity'] = dict(mask)
    # Aggregation must use summed counts, not the mean of episode IoUs.
    extra = [dict(fold=0, c=0, iu={'a': [1, 1]}), dict(fold=0, c=0, iu={'a': [0, 9]})]
    assert class_table(extra, 'a', 'iu')[0][(0, 0)] == 10
    rows[1]['support'] = rows[0]['support']; manifest['frozen_episodes'][1]['support'] = rows[0]['support']
    assert analyze(report, manifest)['independent_class_bootstrap_photo_warning']
    rows[1]['query'] = 'wrong'
    try: analyze(report, manifest)
    except ValueError: pass
    else: raise AssertionError('Photo identity mismatch accepted')
    print(json.dumps(dict(state='CPU_NATIVE_ANGULAR_ANALYSIS_PASSED', class_sumIU=True, resolutions_separate=True,
                         paired_bootstrap=True, shared_photo_warning=True, no_GPU_or_real_task=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path); p.add_argument('--manifest', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check: self_check(); return
    if any(v is None for v in (a.report, a.manifest, a.out)):
        p.error('--report, --manifest and --out are required')
    if a.out.exists(): raise ValueError('Preserve existing analysis; use a fresh output')
    result = analyze(json.loads(a.report.read_text()), json.loads(a.manifest.read_text()), digest(a.report), digest(a.manifest))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'], records=40, shared_photo_warning=result['independent_class_bootstrap_photo_warning'])))


if __name__ == '__main__': main()
