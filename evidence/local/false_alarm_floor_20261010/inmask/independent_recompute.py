"""Independent, read-only audit of sealed inmask evidence; no helper import.

Reconstructs fixed-rule masks from existing fields and the complete FoRIS
original-frame mask. COCO200/joint3: every saved field. PACO600: every existing
episode count/aggregate, with eight predeclared physical-mask spot checks.
LVIS200: current 1400-score-reuse batch bindings and existing table aggregates.
Already-frozen mixed200: all table aggregates and four PASCAL pixel checks.
No encoder, feature generation, parameter search, or retained predictions.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ASSETS = REPO.parent / 'cv_data'
PIN = 'ba73a7b1d79ba5e03ad21e3a6fbd4fea192f92e9b521d2f168417d81e939da2e'
SPOTS = [0, 149, 150, 299, 300, 449, 450, 599]
PASCAL_SPOTS = [80, 89, 100, 119]
SELECTION_PIN = 'f305b05ad86a3ec36e5c6fca92022b9e2828698d800d0e37d62598c40530dbef'
COUNTS, ERRORS = Counter(), []


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(kind, actual, expected, where, tol=0):
    COUNTS[kind] += 1
    equal = abs(actual - expected) <= tol if tol and actual is not None and expected is not None else actual == expected
    if not equal:
        ERRORS.append(dict(kind=kind, where=where, actual=actual, expected=expected))


def pooled(rows, key=None):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        f = row['fields'].get(key) if key else None
        tp = row['hit'] if f is None else f['rule_hit']
        fp = row['false_alarm'] if f is None else f['rule_false_alarm']
        groups[row['cls']][0] += tp
        groups[row['cls']][1] += row['g'] + fp
    return {c: 100 * i / max(u, 1) for c, (i, u) in sorted(groups.items())}


def average(values):
    return float(np.mean(list(values)))


def paired_interval(rows, key):
    """Same specified 1000-draw seed0 estimand, independently rebuilt counts."""
    rng = np.random.RandomState(0)
    cls = sorted({r['cls'] for r in rows})
    index = np.array([cls.index(r['cls']) for r in rows])
    i0 = np.array([r['hit'] for r in rows])
    u0 = np.array([r['g'] + r['false_alarm'] for r in rows])
    i1 = np.array([r['fields'].get(key, {}).get('rule_hit', r['hit']) for r in rows])
    u1 = np.array([r['g'] + r['fields'].get(key, {}).get('rule_false_alarm', r['false_alarm']) for r in rows])
    values = []
    for _ in range(1000):
        draws = rng.randint(len(rows), size=len(rows))
        c = index[draws]
        seen = np.unique(c)
        def metric(i, u):
            total_i = np.bincount(c, weights=i[draws], minlength=len(cls))[seen]
            total_u = np.bincount(c, weights=u[draws], minlength=len(cls))[seen]
            return np.mean(total_i / np.maximum(total_u, 1))
        values.append(100 * (metric(i1, u1) - metric(i0, u0)))
    return np.quantile(values, [.025, .975]).tolist()


def ranking(score, positive, negative, outside=False):
    active = positive + negative > 0
    _, groups = np.unique(score[active], return_inverse=True)
    pg = np.bincount(groups, weights=positive[active])
    ng = np.bincount(groups, weights=negative[active])
    P, N = float(pg.sum()), float(ng.sum())
    if not P or not N:
        return {'miss_auc' if outside else 'auc': None}
    # Ascending score: a positive beats lower-scored negatives; ties half-credit.
    auc = float(np.sum(pg * (np.cumsum(ng) - .5 * ng)) / (P * N))
    tp, fp = np.cumsum(pg[::-1]), np.cumsum(ng[::-1])
    if outside:
        return dict(miss_auc=auc, missed=P, miss_at_1pct=float(max(np.r_[0., tp[fp <= .01 * N]])))
    def at(frac):
        return float(fp[np.flatnonzero(tp >= frac * P - 1e-9)[0]])
    return dict(auc=auc, kept95_mass=at(.95), kept90_mass=at(.90), kept95=at(.95)/N, kept90=at(.90)/N, false_alarm=N)


def decode_mask(row, role):
    path = Path(row[role + '_mask_path'])
    if not path.exists():
        path = ASSETS / path
    with Image.open(path) as im:
        value = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    h = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    check('mask_tensor_hash', h.hexdigest(), row[role + '_mask_hash'], row['episode_id'] + '/' + role)
    return value.astype(bool)


def align(value, hw):
    if value.shape == tuple(hw):
        return value
    # Exact nearest source indices; these inspected sources use native-size GT.
    iy = np.arange(hw[0], dtype=np.int64) * value.shape[0] // hw[0]
    ix = np.arange(hw[1], dtype=np.int64) * value.shape[1] // hw[1]
    return value[iy[:, None], ix[None, :]]


def preflight(name):
    root = HERE / name
    config, seal, report = (read(root / f) for f in ('config.json', 'sealed.json', 'report.json'))
    where, foris = Path(config['where']), Path(config['foris'])
    manifest, index_rows = read(where / 'manifest.json'), lines(foris / 'inference.jsonl')
    index = {r['episode_id']: r for r in index_rows}
    episodes = lines(root / 'episodes.jsonl')
    n = len(config['ids'])
    check('manifest_hash', sha(where / 'manifest.json'), config['manifest_sha256'], name)
    check('baseline_index_hash', sha(foris / 'inference.jsonl'), config['foris_index_sha256'], name)
    check('helper_pin', config['instrument_sha256'], PIN, name)
    check('generator_hash', sha(config['generator_path']), config['generator_sha256'], name)
    check('profile_hash', sha(config['profile']), config['profile_sha256'], name)
    check('seal_config_hash', sha(root / 'config.json'), seal['config_sha256'], name)
    check('seal_report_hash', sha(root / 'sealed.json'), report['seal_sha256'], name)
    check('seal_gt_gate', seal['query_GT_read'], False, name)
    check('seal_count', seal['n'], n, name)
    check('episode_order', [e['id'] for e in episodes], config['ids'], name)
    check('manifest_order', [r['episode_id'] for r in manifest[:n]], config['ids'], name)
    check('frame', report['frame'], 'original', name)
    check('grid', report['grid'], config['grid'], name)
    if name == 'existing_zoom_crossdataset40':
        addendum = read(HERE.parent / 'pascal_existing_cohort_addendum.json')
        audited = {r['episode_id']: r for r in addendum['episodes']}
        check('mixed_original_manifest', sha(where / 'manifest.json'), addendum['manifest_sha256'], name)
        check('mixed_original_index', sha(foris / 'inference.jsonl'), addendum['inference_sha256'], name)
        check('mixed_original_seal', sha(foris / 'sealed.json'), addendum['sealed_sha256'], name)
        for row, e in zip(manifest[:n], episodes):
            a = audited[e['id']]
            check('mixed_archived_prediction_hash', sha(a['prediction_path']), a['prediction_sha256'], e['id'])
            check('mixed_source_prediction_hash', sha(a['source_FoRIS_path']), a['source_FoRIS_sha256'], e['id'])
            with np.load(a['prediction_path'], allow_pickle=False) as z:
                hw = tuple(int(x) for x in z['original_hw'])
                mask = np.unpackbits(z['original/foris.crf'], count=hw[0]*hw[1])
            with np.load(a['source_FoRIS_path'], allow_pickle=False) as z:
                source_mask = np.unpackbits(z['original/foris.crf'], count=hw[0]*hw[1])
            check('mixed_source_mask_bitexact', bool(np.array_equal(mask, source_mask)), True, e['id'])
            check('mixed_archived_predicted_area', int(mask.sum()), e['hit'] + e['false_alarm'], e['id'])
            index[e['id']]['prediction_sha256'] = a['prediction_sha256']
    for i, (row, e) in enumerate(zip(manifest[:n], episodes)):
        record = read(root / 'records' / f'{i:06d}.json')
        file = root / 'fields' / f'{i:06d}.npz'
        loc = name + '/' + row['episode_id']
        check('field_record_hash', sha(file), record['sha256'], loc)
        check('field_seal_hash', record['sha256'], seal['fields'][e['id']], loc)
        check('record_source_pin', record['script_sha256'], config['generator_sha256'], loc)
        check('record_id', record['id'], e['id'], loc)
        check('record_frame', record['prediction_frame'], 'original', loc)
        check('class_binding', e['cls'], f"{row['fold']}:{row.get('loader_class_id', row.get('class_id'))}", loc)
        check('dataset_binding', e['dataset'], row['dataset'], loc)
    ledger = foris / 'score' / 'scored_episodes.jsonl'
    if ledger.exists():
        old = {r['episode_id']: r for r in lines(ledger)}
        for e in episodes:
            old_iu = old[e['id']]['iu']
            old_iu = old_iu['original']['foris.crf'] if 'original' in old_iu else old_iu['foris.crf']
            check('baseline_old_ledger_I', e['hit'], old_iu[0], name + '/' + e['id'])
            check('baseline_old_ledger_U', e['g'] + e['false_alarm'], old_iu[1], name + '/' + e['id'])
    else:
        assert name == 'lvis', 'Unknown baseline archive format'
        paired_manifest = {r['episode_id']: r for r in read(foris / 'manifest.json')}
        for row, e in zip(manifest[:n], episodes):
            b = index[e['id']]
            file = foris / 'predictions' / b['prediction_file']
            check('lvis_archived_prediction_hash', sha(file), b['prediction_sha256'], e['id'])
            for key in ('reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash', 'query_mask_hash'):
                check('lvis_archived_input_binding', row[key], paired_manifest[e['id']][key], e['id'] + '/' + key)
            with np.load(file, allow_pickle=False) as z:
                hw = tuple(int(x) for x in (z['original_hw'] if 'original_hw' in z.files else row['query_size_hw']))
                area = int(np.unpackbits(z['original/foris.crf'], count=hw[0]*hw[1]).sum())
            check('lvis_archived_predicted_area', area, e['hit'] + e['false_alarm'], e['id'])
    return dict(root=root, config=config, seal=seal, report=report, manifest=manifest[:n], index=index, episodes=episodes)


def physical(data, indices):
    name, grid = data['root'].name, data['config']['grid']
    details, decisions = [], defaultdict(Counter)
    for i in indices:
        row, e = data['manifest'][i], data['episodes'][i]
        ident = row['episode_id']
        baseline_record = data['index'][ident]
        prediction_file = Path(data['config']['foris']) / 'predictions' / baseline_record.get('filename', baseline_record.get('prediction_file'))
        check('physical_prediction_hash', sha(prediction_file), baseline_record['prediction_sha256'], name + '/' + ident)
        with np.load(prediction_file, allow_pickle=False) as z:
            hw = tuple(int(x) for x in z['original_hw'])
            pred = np.unpackbits(z['original/foris.crf'], count=hw[0]*hw[1]).reshape(hw).astype(bool)
        check('physical_frame_shape', list(hw), row['query_size_hw'], name + '/' + ident)
        truth0 = decode_mask(row, 'query')
        truth = align(truth0, hw)
        ignore = align(decode_mask(row, 'query_ignore'), hw) if 'query_ignore_mask_path' in row else np.zeros(hw, bool)
        valid = ~ignore
        base = pred & valid
        gt = truth & valid
        g, tp, fp = int(gt.sum()), int((base & gt).sum()), int((base & ~gt).sum())
        for k, value in [('g', g), ('hit', tp), ('false_alarm', fp)]:
            check('physical_baseline_count', value, e[k], name + '/' + ident + '/' + k)
        yy, xx = np.arange(hw[0])*grid//hw[0], np.arange(hw[1])*grid//hw[1]
        node = yy[:, None] * grid + xx[None, :]
        masses = lambda mask: np.bincount(node[mask], minlength=grid*grid).astype(np.float64)
        hit, false, missed, rest = (masses(v) for v in (base & gt, base & ~gt, gt & ~base, ~gt & ~base & valid))
        full_area = int(pred.sum())
        results = {}
        with np.load(data['root'] / 'fields' / f'{i:06d}.npz', allow_pickle=False) as z:
            for key in z.files:
                v = z[key]
                check('field_shape', list(v.shape), [grid*grid], name + '/' + ident + '/' + key)
                check('field_finite', bool(np.isfinite(v).all()), True, name + '/' + ident + '/' + key)
                if key.startswith('aux.'):
                    continue
                zero = .5 if 'back_auc' in key else 0.
                raw_keep = v > zero
                selected_area = int((pred & raw_keep[node]).sum())
                abstain = selected_area < .1 * full_area
                keep = np.ones(grid*grid, bool) if abstain else raw_keep
                after = base & keep[node]
                new_tp, new_fp = int((after & gt).sum()), int((after & ~gt).sum())
                changed = bool(np.any(base & ~after))
                old = e['fields'][key]
                for metric, value in [('rule_hit', new_tp), ('rule_false_alarm', new_fp), ('changed', changed)]:
                    check('physical_rule_count', value, old[metric], name + '/' + ident + '/' + key + '/' + metric)
                for metric, value in {**ranking(v, hit, false), **ranking(v, missed, rest, True)}.items():
                    check('physical_ranking', value, old[metric], name + '/' + ident + '/' + key + '/' + metric, 1e-12)
                selected_hit = float(hit[raw_keep].sum())
                d = decisions[key]
                d['present'] += 1
                d['abstained'] += int(abstain)
                d['changed'] += int(changed)
                d['full_mask_vs_TP_abstention_disagrees'] += int(abstain != (selected_hit < .1 * tp))
                results[key] = dict(TP=new_tp, FP=new_fp, I=new_tp, U=g+new_fp, removed_TP=tp-new_tp, removed_FP=fp-new_fp,
                                    full_predicted_area=full_area, raw_selected_predicted_area=selected_area, abstained=abstain)
        if name != 'coco' or i in (0, 100, 199):
            details.append(dict(index=i, id=ident, cls=e['cls'], hw=list(hw), ignore_pixels=int(ignore.sum()),
                                baseline=dict(TP=tp, FP=fp, I=tp, U=g+fp, g=g), fields=results))
    return dict(episodes=len(indices), decisions={k: dict(v) for k, v in sorted(decisions.items())}, examples=details)


def summarize(data):
    rows, report = data['episodes'], data['report']['datasets']
    result = {}
    for dataset in sorted({e['dataset'] for e in rows}):
        part = [e for e in rows if e['dataset'] == dataset]
        f = report[dataset]['floor']
        scores = pooled(part)
        base = average(scores.values())
        check('summary_baseline', base, f['foris'], data['root'].name, 1e-10)
        check('summary_slots', len(scores), f['observed_fold_class_slots'], data['root'].name)
        byfold = {fold: [e for e in part if e['cls'].split(':')[0] == fold] for fold in sorted({e['cls'].split(':')[0] for e in part})}
        folds = {fold: dict(episodes=len(rr), observed_slots=len(pooled(rr)), baseline=average(pooled(rr).values())) for fold, rr in byfold.items()}
        field_result = {}
        for key, tab in report[dataset]['fields'].items():
            present = [e['fields'][key] for e in part if key in e['fields']]
            scored = [x for x in present if x['auc'] is not None]
            outside = [x for x in present if x['miss_auc'] is not None]
            tp = sum(e['fields'][key]['rule_hit'] if key in e['fields'] else e['hit'] for e in part)
            fp = sum(e['fields'][key]['rule_false_alarm'] if key in e['fields'] else e['false_alarm'] for e in part)
            point = average(pooled(part, key).values())
            computed = dict(rule=point, delta=point-base, present=len(present)/len(part), separable=len(scored),
                changed=sum(bool(x['changed']) for x in present)/len(part), hits_kept=tp/max(sum(e['hit'] for e in part), 1),
                false_alarms_kept=fp/max(sum(e['false_alarm'] for e in part), 1),
                auc=average(x['auc'] for x in scored) if scored else None,
                kept95=sum(x['kept95_mass'] for x in scored)/sum(x['false_alarm'] for x in scored) if scored else None,
                kept90=sum(x['kept90_mass'] for x in scored)/sum(x['false_alarm'] for x in scored) if scored else None,
                miss_auc=average(x['miss_auc'] for x in outside) if outside else None,
                miss_at_1pct=sum(x['miss_at_1pct'] for x in outside)/sum(x['missed'] for x in outside) if outside else None)
            for metric, value in computed.items():
                check('summary_field', value, tab[metric], data['root'].name + '/' + key + '/' + metric, 1e-10)
            interval = paired_interval(part, key) if key == 'back_auc' else None
            if interval is not None:
                for j, v in enumerate(interval):
                    check('selected_rule_bootstrap', v, tab['interval'][j], data['root'].name + '/' + dataset, 1e-10)
            missing = [e for e in part if key not in e['fields']]
            field_result[key] = dict(**computed, missing_baseline_fallback_count=len(missing),
                                    missing_baseline_TP=sum(e['hit'] for e in missing), missing_baseline_FP=sum(e['false_alarm'] for e in missing),
                                    TP=tp, FP=fp, interval_reported=tab['interval'],
                                    interval_independently_recomputed=interval,
                                    per_fold={fold: average(pooled(rr, key).values()) for fold, rr in byfold.items()})
        old_path = Path(data['config']['foris']) / 'score' / 'paired_results.json'
        if old_path.exists():
            old_report = read(old_path)
            if 'frames' in old_report:
                old = old_report['frames']['original']
                archive = dict(original_report_fixed_slot_counts=old['fold_class_counts'], original_report_fixed_slot_baseline=old['miou']['foris.crf'])
            else:
                old = old_report['datasets'][dataset]
                equal_fold = average(v['baseline'] for v in folds.values())
                check('mixed_archived_equal_fold_summary', equal_fold, old['miou']['foris.crf'], dataset, 1e-10)
                archive = dict(original_report_fixed_slot_counts=None, original_report_fixed_slot_baseline=None,
                               original_report_observed_equal_fold_baseline=old['miou']['foris.crf'])
        else:
            summary = read(Path(data['config']['foris']).parent / 'summary.json')
            archived_value = summary['batches']['lvis_mean200_20261008']['frames']['original']['foris']
            check('lvis_archived_batch_summary', base, archived_value, 'lvis', 1e-10)
            archive = dict(original_report_fixed_slot_counts=None, original_report_fixed_slot_baseline=None,
                           archive='lvis_foris1400_score_reuse_20261009/summary.json batches.lvis_mean200_20261008 original',
                           archived_batch_baseline=archived_value)
        result[dataset] = dict(n=len(part), original_frame=True, observed_slots=len(scores), per_fold=folds,
                              observed_slot_baseline=base, observed_equal_fold_baseline=average(v['baseline'] for v in folds.values()),
                              **archive,
                              class_scores=scores, fields=field_result)
    return result


def joint_records(data):
    root = data['root']
    plan, replay, success, init, bindings = (read(root / f) for f in ('joint_plan.json', 'raw_replay.json', 'SUCCESS.json', 'encoder_initialization.json', 'raw_bindings.json'))
    check('joint_plan_binding', sha(root / 'joint_plan.json'), data['seal']['joint_plan_sha256'], 'joint')
    check('joint_replay_binding', sha(root / 'raw_replay.json'), data['seal']['raw_replay_sha256'], 'joint')
    check('joint_success_seal', sha(root / 'sealed.json'), success['sealed_sha256'], 'joint')
    check('joint_raw_index', sha(root / 'raw_bindings.json'), replay['raw_bindings_sha256'], 'joint')
    check('joint_profile', sha(plan['profile_path']), replay['profile_sha256'], 'joint')
    check('joint_prefix', plan['prefix_rows'], data['manifest'], 'joint')
    check('joint_calls', sum(b['telemetry']['encoder_calls'] for b in bindings), 9, 'joint')
    check('joint_unique_inputs', len({b['key'] for b in bindings}), 9, 'joint')
    for path, digest in plan['source_sha256'].items():
        check('joint_source_hash', sha(path), digest, path)
    for b in bindings:
        for part in ('input_file', 'entry', 'payload'):
            path_key = 'input_path' if part == 'input_file' else part + '_path'
            check('joint_raw_file_hash', sha(b[path_key]), b[part + '_sha256'], b['operation'] + '/' + b['episode_id'])
        check('joint_new_encode', b['new_cache_entry'], True, b['key'])
        check('joint_raw_parity_record', b['native_raw_serialized_parity'], True, b['key'])
        check('joint_unit_parity_record', b['serialized_unit_feature_parity'], True, b['key'])
    times = []
    for i in range(3):
        r = read(root / 'records' / f'{i:06d}.json')
        times.append(dict(id=r['id'], separate_two_forward_seconds=r['separate_seconds'], joint_forward_seconds=r['joint_seconds'],
                          separate_encode_cpu_copy_seconds=sum(b['telemetry']['encode_and_cpu_copy_seconds'] for b in r['raw_requests'] if b['operation'].startswith('separate.')),
                          joint_encode_cpu_copy_seconds=next(b['telemetry']['encode_and_cpu_copy_seconds'] for b in r['raw_requests'] if b['operation']=='joint.canvas'),
                          episode_field_record_seconds=r['seconds']))
    peaks = {key: max(b['telemetry']['observed_peak'][key] for b in bindings) for key in ('allocated_bytes', 'driver_bytes', 'rss_peak_bytes')}
    check('joint_observed_peak', peaks, success['observed_peak'], 'joint')
    return dict(encoder_calls=9, unique_raw_inputs=9, raw_files_hashed_without_feature_recompute=27,
                raw_tensor_parity='Producer receipts verified by binding/file hash; auditor did not re-normalize raw features or encode.',
                initialization_seconds=init['seconds'], per_episode=times,
                mean_separate_forward_seconds=average(r['separate_two_forward_seconds'] for r in times),
                mean_joint_forward_seconds=average(r['joint_forward_seconds'] for r in times),
                observed_peak=peaks, exact_allocator_peak=False, sampling_interval_seconds=.001,
                requested_cap_bytes=success['requested_cap_bytes'], recommended_max_memory_bytes=success['recommended_max_memory_bytes'],
                caveat='MPS allocator/driver maxima were sampled; driver cache persists between arms; RSS is cumulative OS ru_maxrss. This is not a cold segmentation latency or an exact per-arm peak.')


def main():
    started = time.monotonic()
    check('live_helper_pin', sha(HERE.parent / 'inmask_evidence.py'), PIN, 'live')
    data = {name: preflight(name) for name in ('coco', 'joint_coco_try', 'paco', 'lvis', 'existing_zoom_crossdataset40')}
    selection = read(HERE / 'selection.json')
    check('selection_hash', sha(HERE/'selection.json'), SELECTION_PIN, 'selection')
    for name, digest in selection['design_report_sha256'].items():
        check('selection_design_report', sha(HERE/name/'report.json'), digest, name)
    check('selection_rule_pin', selection['chosen'], 'back_auc', 'selection')
    protocol = dict(state='SEALED_INPUTS_CHECKED_BEFORE_THIS_AUDIT_GT_READ', audit_sha256=sha(__file__), helper_sha256=PIN,
        mask_rule='score > 0 (back_auc names > .5); abstain to complete FoRIS if raw kept pixels < .1 * complete FoRIS area BEFORE GT/ignore',
        missing_field='unchanged complete FoRIS TP/FP per episode',
        projection='original pixel (y,x) belongs to floor(y*grid/H), floor(x*grid/W); no resized output mask',
        physical_episodes=dict(coco='all200', joint_coco_try='all3', paco=SPOTS, lvis=[], existing_zoom_crossdataset40=PASCAL_SPOTS),
        selection_sha256=SELECTION_PIN,
        aggregation='all observed fold:class pooled I/U; equal-fold and declared fixed slots separately identified',
        sources={name: dict(config_sha256=sha(d['root']/'config.json'), seal_sha256=sha(d['root']/'sealed.json'), report_sha256=sha(d['root']/'report.json')) for name,d in data.items()},
        pre_GT_errors=ERRORS.copy(), pre_GT_checks=dict(COUNTS))
    (HERE / 'independent_recompute_frozen.json').write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    assert not ERRORS, 'Input binding failure; no query GT may be read.'
    physical_results = {name: physical(d, range(len(d['episodes'])) if name in ('coco','joint_coco_try') else SPOTS if name=='paco' else PASCAL_SPOTS if name=='existing_zoom_crossdataset40' else []) for name,d in data.items()}
    summaries = {name: summarize(d) for name,d in data.items()}
    joint = joint_records(data['joint_coco_try'])
    design_fields = set(summaries['coco']['coco']['fields']) & set(summaries['paco']['paco_part']['fields'])
    eligible = sorted(k for k in design_fields if not any(tag in k for tag in ('diag.', 'foris_score', 'fg_peak')))
    worst = {k: min(summaries['coco']['coco']['fields'][k]['delta'], summaries['paco']['paco_part']['fields'][k]['delta']) for k in eligible}
    chosen = max(worst, key=worst.get)
    check('mechanical_selection_matches_frozen', chosen, selection['chosen'], 'selection')
    transfer = [summaries['lvis']['lvis']['fields'][chosen], summaries['existing_zoom_crossdataset40']['pascal_part']['fields'][chosen]]
    receipt = dict(state='PASS' if not ERRORS else 'FAIL', audit_sha256=sha(__file__), protocol_sha256=sha(HERE/'independent_recompute_frozen.json'),
        helper_sha256=PIN, checks=dict(COUNTS), mismatches=ERRORS, physical=physical_results, summaries=summaries, joint_cost=joint,
        mechanical_design_selection=dict(design=['paco/paco_part','coco/coco'], eligible_fields=eligible, worst_delta=worst,
            strictly_positive_on_both=[k for k in eligible if worst[k]>0], chosen=chosen if worst[chosen]>0 else None),
        predefined_transfer_readout=dict(units=['lvis/lvis','existing_zoom_crossdataset40/pascal_part'], denominator=2,
            lower_bound_gt_minus_point3=sum(v['interval_independently_recomputed'][0]>-.3 for v in transfer),
            point_delta_at_least_1=sum(v['delta']>=1 for v in transfer), passed=False,
            auxiliary_mixed_domains=['coco','lvis','paco_part','suim'], auxiliary_counted_in_denominator=False),
        timing_seconds=time.monotonic()-started,
        limitations=['Already exposed development episodes; no independent confirmation or official all-fold/all-dataset benchmark claim.',
            'AUC and kept95/90 are post-seal GT-aware ranking diagnostics; actual fixed-rule mask IU is separate.',
            'Recorded source guard, field hashes and seal-before-truth code order bind this evaluation; they cannot prove absence of prior human/model exposure.',
            'PACO physical GT checked only at eight predeclared indices; all600 baseline IU checked against the existing original-frame ledger and every aggregate recomputed.',
            'PASCAL physical GT checked only at four predeclared positions; allmixed200 baseline masks source-bitexact and baseline IU equal the original ledger.',
            'LVIS200 bound to the 1400 score-reuse archive selected batch, not another historical 200_cached summary.'])
    (HERE/'independent_recompute_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=receipt['state'], checks=sum(COUNTS.values()), mismatches=len(ERRORS), physical_episodes={k:v['episodes'] for k,v in physical_results.items()},
                         observed_baselines={k:next(iter(v.values()))['observed_slot_baseline'] for k,v in summaries.items()},
                         design=receipt['mechanical_design_selection'],seconds=receipt['timing_seconds']),allow_nan=False))


if __name__ == '__main__':
    main()
