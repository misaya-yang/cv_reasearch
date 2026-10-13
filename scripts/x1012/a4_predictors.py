"""A4 predictor diagnostics: seal GT-free descriptors, then fit on a photo half.

Only score reads the existing A4 metric report. It never reopens query GT and
does not select a predictor on the reporting half. The original A4 upper-bound
report/completion remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import lab
import plan_common as pc

OUT = lab.ASSETS / 'a/x1012_A4_predictors'
FIELDS = lab.ASSETS / 'a/x1012_P0/fields'
UPPER = lab.ASSETS / 'a/x1012_A4'
FEATURES = ('ref_loo_auroc', 'ref_background_fg_mass', 'rho', 'ref_fg_area',
            'query_e_below_median_fraction', 'indn_ind0_area_ratio', 'j_difference')
REFERENCE_FEATURES = FEATURES[:4]
REQUIRED = ('c32', 'back_f', 'log_e', 'mask/IND0', 'mask/INDN', 'ref_stats')
SPLIT_SALT = 'A4_predictors_query_photo_20261012'
QUANTILES = tuple(np.linspace(0, 1, 21).tolist())
FORMULAS = {
    'ref_loo_auroc': 'R-only IND debias; unit(r-mean(r)); tau=.05 soft-coverage kernel posterior with Chebyshev<=1 grid neighbors excluded; equal-token AUROC on y>=.9 versus y<=.1, ties=.5',
    'ref_background_fg_mass': 'sum((1-y)*R_LOO_posterior)/sum(1-y), same excluded-neighbor posterior',
    'rho': '1-sum(y*(y>=.9))/sum(y)',
    'ref_fg_area': 'mean(reference coverage y on 64x64)',
    'query_e_below_median_fraction': 'mean(saved log_e < median(saved log_e)), strict inequality; FP16 ties retained',
    'indn_ind0_area_ratio': 'count(INDN pixels)/count(IND0 pixels); missing if IND0 is empty',
    'j_difference': 'J(INDN)-J(IND0); native S=mask coverage64>.5; pure=mean(c32[S]); capture=sum(back_f[S]); J=pure*capture/(pure+capture-pure*capture); empty S gives J=0',
}


def photo(binding):
    return str((lab.ASSETS / binding['row']['query_path']).resolve())


def split(photo_path):
    digest = hashlib.sha256((SPLIT_SALT + '\0' + photo_path).encode()).hexdigest()
    return 'train' if int(digest[:16], 16) < 2**63 else 'report'


def weighted_auc(scores, positive, weights=None):
    """Weighted Mann-Whitney AUC; equal scores receive half the pair mass."""
    x, pos = np.asarray(scores, dtype=np.float64), np.asarray(positive, dtype=bool)
    w = np.ones(x.shape, np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    assert x.shape == pos.shape == w.shape
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, pos, w = x[valid], pos[valid], w[valid]
    total_pos, total_neg = float(w[pos].sum()), float(w[~pos].sum())
    info = {'n': int(valid.sum()), 'positive_weight': total_pos, 'negative_weight': total_neg}
    if not total_pos or not total_neg:
        return None, info
    order = np.argsort(x, kind='stable')
    x, pos, w = x[order], pos[order], w[order]
    numerator, lower_negative = 0., 0.
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    for lo, hi in zip(starts, ends):
        group_pos = float(w[lo:hi][pos[lo:hi]].sum())
        group_neg = float(w[lo:hi][~pos[lo:hi]].sum())
        numerator += group_pos * (lower_negative + .5 * group_neg)
        lower_negative += group_neg
    return float(numerator / (total_pos * total_neg)), info


def reference_statistics(r, y):
    """Reference-only leave-neighborhood-out posterior, in bounded CPU blocks."""
    r, y = lab.unit(r.float()), torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    n = r.shape[0]
    side = int(n**.5)
    assert side * side == n == y.numel() and bool((y >= 0).all()) and bool((y <= 1).all())
    assert float(y.sum()) > 0, 'Reference mask has no foreground mass'
    centered = lab.unit(r - r.mean(0))
    ids = torch.arange(n)
    rows, cols = ids // side, ids % side
    posterior = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        similarity = centered[start:stop] @ centered.T
        excluded = ((rows[start:stop, None] - rows[None, :]).abs() <= 1)
        excluded &= ((cols[start:stop, None] - cols[None, :]).abs() <= 1)
        similarity.masked_fill_(excluded, -torch.inf)
        maximum = similarity.max(1, keepdim=True).values
        kernel = torch.exp((similarity - maximum) / .05)
        posterior[start:stop] = (kernel @ y) / kernel.sum(1).clamp_min(1e-30)
    assert torch.isfinite(posterior).all()
    coverage, values = y.numpy(), posterior.numpy()
    hard_fg, hard_bg = coverage >= .9, coverage <= .1
    missing = {}
    if not hard_fg.any() or not hard_bg.any():
        auc = None
        missing['ref_loo_auroc'] = 'no_hard_FG' if not hard_fg.any() else 'no_hard_BG'
    else:
        selected = hard_fg | hard_bg
        auc, _ = weighted_auc(values[selected], hard_fg[selected])
    background = 1 - coverage
    if float(background.sum()) == 0:
        mass = None
        missing['ref_background_fg_mass'] = 'no_background_mass'
    else:
        mass = float(np.sum(background * values, dtype=np.float64) / np.sum(background, dtype=np.float64))
    stats = {'ref_loo_auroc': auc, 'ref_background_fg_mass': mass,
             'rho': float(1 - np.sum(coverage * hard_fg, dtype=np.float64) / np.sum(coverage, dtype=np.float64)),
             'ref_fg_area': float(np.mean(coverage, dtype=np.float64))}
    return {'features': stats, 'missing': missing, 'hard_fg_tokens': int(hard_fg.sum()),
            'hard_bg_tokens': int(hard_bg.sum()), 'foreground_mass': float(coverage.sum()),
            'new_encoder_calls': 0, 'query_GT_reads': 0}


def reference_descriptor(binding):
    from p0_manifest import raw_metadata
    from ics.official_data import array_hash
    metadata = raw_metadata(binding['fit']['reference_raw']['payload_path'])
    if 'input_sha256' in binding['fit']['reference_raw']:
        assert binding['fit']['reference_raw']['input_sha256'] == metadata['input_sha256']
    pair = [metadata['input_sha256'], binding['row']['reference_mask_hash']]
    key = hashlib.sha256(json.dumps(pair, separators=(',', ':')).encode()).hexdigest()
    path = OUT / 'reference_cache' / (key + '.json')
    if path.exists():
        record = pc.read(path)
        assert record['cache_pair'] == pair and record['tensor_sha256'] == metadata['tensor_sha256']
        return record, path
    assert pc.sha(metadata['payload_path']) == metadata['payload_sha256']
    assert pc.sha(metadata['entry_path']) == metadata['entry_sha256']
    with Image.open(binding['row']['reference_mask_path']) as image:
        native_mask = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
    assert array_hash(native_mask) == pair[1], 'Frozen reference mask changed'
    resized = F.interpolate(torch.from_numpy(native_mask)[None, None].float(), (1024, 1024), mode='nearest')[0, 0]
    y = lab.coverage64(resized).flatten()
    r = lab.debias(lab.feats(binding, 'reference'))
    start = time.monotonic()
    record = reference_statistics(r, y)
    record.update(cache_pair=pair, cache_key=key, tensor_sha256=metadata['tensor_sha256'],
                  payload_sha256=metadata['payload_sha256'], reference_mask_path=binding['row']['reference_mask_path'],
                  seconds=time.monotonic() - start)
    pc.write(path, record)
    return record, path


def bidirectional_j(packed_mask, c, back_f):
    mask = torch.from_numpy(pc.unpack(packed_mask)).float()
    selected = (lab.coverage64(mask).numpy().ravel() > .5)
    if not selected.any():
        return 0., {'native_tokens': 0, 'pure': 0., 'capture': 0.}
    purity = float(np.mean(c[selected], dtype=np.float64))
    capture = float(np.sum(back_f[selected], dtype=np.float64))
    denominator = purity + capture - purity * capture
    j = purity * capture / max(denominator, 1e-30)
    return float(j), {'native_tokens': int(selected.sum()), 'pure': purity, 'capture': capture}


def source_field(binding):
    name = pc.key(binding)
    path = FIELDS / 'records' / (name + '.json')
    if not path.is_file():
        raise FileNotFoundError('P0 field record missing: ' + str(path))
    record = pc.read(path)
    assert record['index'] == name and record['episode_id'] == binding['row']['episode_id']
    assert record['new_encoder_calls'] == record['query_GT_reads'] == 0
    return {'record_path': str(path), 'record_sha256': pc.sha(path),
            'path': record['path'], 'sha256': record['sha256']}


def produce():
    pc.setup('cpu')
    rows = pc.seed_rows(150)
    assert len({pc.key(b) for b in rows}) == len(rows)
    sources = {pc.key(b): source_field(b) for b in rows}
    for name, source in sources.items():
        with np.load(source['path'], allow_pickle=False) as z:
            assert set(REQUIRED) <= set(z.files), 'P0 fields incomplete: ' + name
    for name in ('descriptors', 'reference_cache', 'frozen'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if not config_path.exists():
        files = (Path(__file__), Path(pc.__file__), Path(lab.__file__),
                 Path(pc.__file__).with_name('plan_report.py'), Path(pc.__file__).with_name('p0_manifest.py'))
        for i, path in enumerate(files):
            shutil.copyfile(path, OUT / 'frozen' / f'{i}_{path.name}')
        pc.write(OUT / 'manifest.json', rows)
        pc.write(config_path, {'state': 'FROZEN', 'n': len(rows), 'cohort': 'pc.seed_rows(150), fixed ten domains',
                              'features': list(FEATURES), 'formulas': FORMULAS, 'field_sources': sources,
                              'reference_cache_pair': ['reference input tensor hash', 'native reference mask hash'],
                              'photo_split_salt': SPLIT_SALT, 'train_half': 'first64bits(sha256(salt+NUL+photo)) < 2**63',
                              'threshold_quantiles': list(QUANTILES), 'threshold_tie': 'choose IND0 when x==threshold',
                              'fit_objective': 'training-half equal-domain mean class-pooled mIoU; one global direction/threshold per feature',
                              'missing_fallback': 'better training-half global constant IND0/INDN, ties IND0',
                              'threads': 4, 'workers': 1, 'source_sha256': {str(p): pc.sha(p) for p in files},
                              'basis_sha256': pc.sha(lab.BASIS), 'new_encoder_calls': 0,
                              'query_GT_in_inference': False, 'A4_metric_report_opened_in_inference': False})
    config = pc.read(config_path)
    assert config['field_sources'] == sources and pc.read(OUT / 'manifest.json') == rows
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items())
    assert pc.sha(lab.BASIS) == config['basis_sha256']
    if (OUT / 'sealed.json').exists():
        print({'task': 'A4_predictors', 'state': 'ALREADY_SEALED', 'n': len(rows)}, flush=True)
        return
    began, records = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        name, path = pc.key(b), OUT / 'descriptors' / (pc.key(b) + '.json')
        source = sources[name]
        if path.exists():
            descriptor = pc.read(path)
            assert descriptor['episode_id'] == b['row']['episode_id'] and descriptor['source_field_sha256'] == source['sha256']
            assert pc.sha(descriptor['reference_cache_path']) == descriptor['reference_cache_sha256']
        else:
            with pc.inference_only(b):
                assert pc.sha(source['record_path']) == source['record_sha256']
                assert pc.sha(source['path']) == source['sha256'], 'P0 field changed'
                with np.load(source['path'], allow_pickle=False) as z:
                    values = {k: z[k].copy() for k in REQUIRED}
                reference, reference_path = reference_descriptor(b)
                features, missing = dict(reference['features']), dict(reference['missing'])
                log_e = np.asarray(values['log_e'], dtype=np.float64).ravel()
                c = np.asarray(values['c32'], dtype=np.float64).ravel()
                back_f = np.asarray(values['back_f'], dtype=np.float64).ravel()
                assert c.size == log_e.size == back_f.size == 4096
                assert np.isfinite(c).all() and np.isfinite(log_e).all() and np.isfinite(back_f).all()
                assert (back_f >= 0).all()
                features['query_e_below_median_fraction'] = float(np.mean(log_e < np.median(log_e)))
                area0 = int(np.count_nonzero(pc.unpack(values['mask/IND0'])))
                area_n = int(np.count_nonzero(pc.unpack(values['mask/INDN'])))
                features['indn_ind0_area_ratio'] = area_n / area0 if area0 else None
                if not area0:
                    missing['indn_ind0_area_ratio'] = 'IND0_empty; ratio undefined'
                j0, aux0 = bidirectional_j(values['mask/IND0'], c, back_f)
                jn, auxn = bidirectional_j(values['mask/INDN'], c, back_f)
                features['j_difference'] = jn - j0
                assert set(features) == set(FEATURES)
                assert np.isclose(features['rho'], float(values['ref_stats'][2]), atol=2e-6)
                assert np.isclose(features['ref_fg_area'], float(values['ref_stats'][0]), atol=2e-6)
                descriptor = {'index': name, 'episode_id': b['row']['episode_id'], 'dataset': b['dataset'],
                              'query_photo': photo(b), 'half': split(photo(b)), 'features': features, 'missing': missing,
                              'source_field_path': source['path'], 'source_field_sha256': source['sha256'],
                              'reference_cache_path': str(reference_path), 'reference_cache_sha256': pc.sha(reference_path),
                              'aux': {'IND0_area': area0, 'INDN_area': area_n, 'J_IND0': j0, 'J_INDN': jn,
                                      'J_IND0_parts': aux0, 'J_INDN_parts': auxn, 'saved_back_f_total': float(back_f.sum())},
                              'new_encoder_calls': 0, 'query_GT_reads': 0, 'A4_metric_report_reads': 0}
            pc.write(path, descriptor)
        records.append({'index': name, 'path': str(path), 'sha256': pc.sha(path)})
        if i <= 3 or i % 50 == 0:
            print({'task': 'A4_predictors', 'completed': i, 'n': len(rows),
                   'reference_cache_n': len(list((OUT / 'reference_cache').glob('*.json'))),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    pc.write(OUT / 'sealed.json', {'state': 'ALL_DESCRIPTORS_SEALED', 'n': len(rows), 'records': records,
             'manifest_sha256': pc.sha(OUT / 'manifest.json'), 'config_sha256': pc.sha(config_path),
             'seconds': time.monotonic() - began, 'new_encoder_calls': 0, 'query_GT_reads': 0,
             'A4_metric_report_reads': 0})
    print({'state': 'ALL_DESCRIPTORS_SEALED', 'n': len(rows)}, flush=True)


def delta_iou(record):
    return record['metrics']['INDN']['IoU'] - record['metrics']['IND0']['IoU']


def objective(records, choose_n):
    slots = {}
    for record, selected in zip(records, choose_n):
        key = (record['dataset'], str(record['fold']), str(record['class_id']))
        totals = slots.setdefault(key, [0, 0])
        metric = record['metrics']['INDN' if selected else 'IND0']
        totals[0] += metric['I']; totals[1] += metric['U']
    domains = {}
    for (domain, _, _), (inter, union) in slots.items():
        domains.setdefault(domain, []).append(100 * inter / max(union, 1))
    return float(np.mean([np.mean(v) for v in domains.values()]))


def apply_selector(values, selector):
    values = np.asarray(values, dtype=np.float64)
    if selector['kind'] == 'constant':
        return np.full(values.shape, selector['choose_N'], dtype=bool)
    result = np.full(values.shape, selector['missing_choose_N'], dtype=bool)
    finite = np.isfinite(values)
    result[finite] = selector['direction'] * (values[finite] - selector['threshold']) > 0
    return result


def fit_selector(values, train_records):
    x = np.asarray(values, dtype=np.float64)
    all0, alln = np.zeros(len(x), bool), np.ones(len(x), bool)
    constant0, constantn = objective(train_records, all0), objective(train_records, alln)
    missing_n = constantn > constant0
    best = {'kind': 'constant', 'choose_N': False, 'train_objective': constant0,
            'missing_choose_N': missing_n, 'direction': None, 'threshold': None}
    if constantn > constant0 + 1e-12:
        best.update(choose_N=True, train_objective=constantn)
    finite = x[np.isfinite(x)]
    thresholds = np.unique(np.quantile(finite, QUANTILES)) if len(finite) else np.array([])
    for threshold in thresholds:
        for direction in (1, -1):
            rule = {'kind': 'threshold', 'direction': direction, 'threshold': float(threshold),
                    'missing_choose_N': missing_n}
            value = objective(train_records, apply_selector(x, rule))
            if value > best['train_objective'] + 1e-12:
                best = dict(rule, train_objective=value)
    delta = np.asarray([delta_iou(r) for r in train_records])
    auc, _ = weighted_auc(x, delta > 0, np.abs(delta))
    orientation = best['direction'] if best['kind'] == 'threshold' else (None if auc is None else (1 if auc >= .5 else -1))
    best.update(auc_orientation=orientation, train_raw_weighted_auc=auc, train_n=len(x),
                train_finite_n=int(np.isfinite(x).sum()), quantile_candidate_n=len(thresholds),
                constant_IND0_objective=constant0, constant_INDN_objective=constantn)
    return best


def predictor_auc(feature, descriptors, metrics, selector):
    x = np.asarray([np.nan if d['features'][feature] is None else d['features'][feature] for d in descriptors], dtype=np.float64)
    delta = np.asarray([delta_iou(r) for r in metrics])
    raw, counts = weighted_auc(x, delta > 0, np.abs(delta))
    orientation = selector['auc_orientation']
    oriented = None if raw is None or orientation is None else (raw if orientation == 1 else 1 - raw)
    finite = x[np.isfinite(x)]
    return {'raw_weighted_auc': raw, 'train_oriented_weighted_auc': oriented,
            'auc_orientation_from_training': orientation, 'n': len(x), 'missing_n': int((~np.isfinite(x)).sum()),
            'feature_unique_n': int(len(np.unique(finite))), 'feature_std': float(finite.std()) if len(finite) else None,
            'delta_tie_n': int((delta == 0).sum()), 'nonzero_delta_auc_counts': counts}


def score():
    import plan_report as pr
    pc.setup('cpu')
    seal, config = pc.read(OUT / 'sealed.json'), pc.read(OUT / 'config.json')
    assert seal['state'] == 'ALL_DESCRIPTORS_SEALED'
    assert pc.sha(OUT / 'manifest.json') == seal['manifest_sha256'] and pc.sha(OUT / 'config.json') == seal['config_sha256']
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items())
    rows = pc.read(OUT / 'manifest.json')
    assert rows == pc.seed_rows(150) and len(rows) == seal['n']
    descriptors = []
    for row, sealed in zip(rows, seal['records']):
        assert pc.key(row) == sealed['index'] and pc.sha(sealed['path']) == sealed['sha256']
        descriptor = pc.read(sealed['path'])
        assert descriptor['query_photo'] == photo(row) and descriptor['half'] == split(photo(row))
        assert pc.sha(descriptor['source_field_path']) == descriptor['source_field_sha256']
        assert pc.sha(descriptor['reference_cache_path']) == descriptor['reference_cache_sha256']
        descriptors.append(descriptor)
    # This is the first place this script is allowed to read A4 metric values.
    upper = pc.read(UPPER / 'report.json')
    source_records = {r['index']: r for r in upper['records']}
    assert upper['n'] == len(rows) == len(source_records)
    before_gt = pc.read(UPPER / 'sources_sealed_before_GT.json')
    upper_sources = {r['index']: r for r in before_gt['sources']}
    metrics = []
    for b, descriptor in zip(rows, descriptors):
        record = source_records[pc.key(b)]
        assert record['episode_id'] == b['row']['episode_id'] and record['dataset'] == b['dataset']
        assert record['query_photo'] == descriptor['query_photo']
        assert upper_sources[pc.key(b)]['sha256'] == descriptor['source_field_sha256']
        metrics.append(record)
    train_ids = [i for i, d in enumerate(descriptors) if d['half'] == 'train']
    report_ids = [i for i, d in enumerate(descriptors) if d['half'] == 'report']
    assert train_ids and report_ids
    assert set(descriptors[i]['query_photo'] for i in train_ids).isdisjoint(descriptors[i]['query_photo'] for i in report_ids)
    assert {metrics[i]['dataset'] for i in train_ids} == {metrics[i]['dataset'] for i in report_ids} == set(lab.ALL_DOMAINS)
    train_records = [metrics[i] for i in train_ids]
    selectors = {}
    for feature in FEATURES:
        values = [np.nan if descriptors[i]['features'][feature] is None else descriptors[i]['features'][feature] for i in train_ids]
        selectors[feature] = fit_selector(values, train_records)
    source_hash = pc.sha(UPPER / 'report.json')
    pc.write(OUT / 'selectors.json', {'state': 'TRAIN_HALF_SELECTORS_FROZEN', 'selectors': selectors,
             'train_indices': [descriptors[i]['index'] for i in train_ids],
             'report_indices': [descriptors[i]['index'] for i in report_ids],
             'descriptor_seal_sha256': pc.sha(OUT / 'sealed.json'), 'A4_metric_report_sha256': source_hash})
    choices = {}
    report_descriptors = [descriptors[i] for i in report_ids]
    report_metrics = [metrics[i] for i in report_ids]
    for feature in FEATURES:
        values = [np.nan if d['features'][feature] is None else d['features'][feature] for d in report_descriptors]
        choices[feature] = apply_selector(values, selectors[feature]).tolist()
    pc.write(OUT / 'report_choices.json', {'state': 'REPORT_HALF_CHOICES_FROZEN',
             'indices': [d['index'] for d in report_descriptors], 'choose_N': choices,
             'selectors_sha256': pc.sha(OUT / 'selectors.json'), 'test_metrics_used_to_choose': False})
    arms = ('FoRIS', 'IND0', 'INDN') + tuple('selector_' + f for f in FEATURES)
    records = []
    for j, record in enumerate(report_metrics):
        selected_metrics = {a: dict(record['metrics'][a]) for a in ('FoRIS', 'IND0', 'INDN')}
        for feature in FEATURES:
            selected_metrics['selector_' + feature] = dict(record['metrics']['INDN' if choices[feature][j] else 'IND0'])
        records.append({k: v for k, v in record.items() if k in ('index', 'episode_id', 'dataset', 'fold', 'class_id', 'query_photo')})
        records[-1]['metrics'] = selected_metrics
    auc = {feature: {'all_domains': predictor_auc(feature, report_descriptors, report_metrics, selectors[feature]),
                    'per_domain': {d: predictor_auc(feature,
                                     [a for a, b in zip(report_descriptors, report_metrics) if b['dataset'] == d],
                                     [b for b in report_metrics if b['dataset'] == d], selectors[feature])
                                   for d in lab.ALL_DOMAINS}}
           for feature in FEATURES}
    pc.write(OUT / 'predictor_auroc.json', auc)
    signal = {f: auc[f]['all_domains']['train_oriented_weighted_auc'] for f in FEATURES}
    strong = [f for f, value in signal.items() if value is not None and value > .75]
    all_low = all(value is not None and value < .65 for value in signal.values())
    ref_low = all(signal[f] is not None and signal[f] < .65 for f in REFERENCE_FEATURES)
    state = 'PREDICTORS_BELOW_065_REFERENCE_SWITCH_CLOSED' if all_low else ('SIGNAL_ABOVE_075_CONTINUOUS_MODULATION_ELIGIBLE' if strong else 'PREDICTOR_EVIDENCE_INCONCLUSIVE')
    completion = {'state': state, 'n': len(rows), 'train_n': len(train_ids), 'report_n': len(report_ids),
                  'per_half_domain_n': {half: dict(Counter(metrics[i]['dataset'] for i in ids)) for half, ids in (('train', train_ids), ('report', report_ids))},
                  'heldout_train_oriented_weighted_auroc': signal, 'all_predictors_below_065': all_low,
                  'reference_only_predictors_below_065': ref_low, 'above_075_predictors': strong,
                  'continuous_modulation_run': False, 'selector_chosen_on_report_half': False,
                  'confirm_scored': False, 'new_encoder_calls': 0, 'additional_query_GT_reads': 0,
                  'A4_metric_report_sha256': source_hash, 'upper_report_preserved': str(UPPER / 'report.json')}
    result = pr.report('A4_predictors', OUT, records, arms, [
        '只评估A4原1360个固定已曝光开发任务；参考input hash与R mask hash为缓存键。七个描述量全部先封存，生产阶段不读取query GT或A4 metric report。',
        '按查询照片的固定SHA256切半；所有同照片任务属于同一半。每个单变量只在训练半拟合一个跨十域的方向与21分位候选阈值，不按域分支；缺失走训练半较优常数。',
        '该表仅为留出报告半：选择结果的整数指标直接复用原A4已评分两臂，未重新读GT。阈值与report choices在留出评测前分别封存。',
        'AUROC按|INDN IoU-IND0 IoU|加权；IoU ties为零权，descriptor score ties获0.5配对质量；方向只从训练半取得。缺少hardFG/hardBG的LOO、空IND0的面积比均记缺失。',
        '所有选择器均报告；不按留出分数挑最佳，也不把该开发切半视为confirm。只输出.65/.75判据，不启动连续调制或生成新配方。',
        '固定描述量定义及参数见本任务config.json；FP16 log_e/back_f按现存字段使用，E<median可能近似常数仍如实保留。',
    ], extra={'selectors': selectors, 'predictor_auroc': auc, 'completion': completion})
    assert result['n'] == len(report_ids) and pc.sha(UPPER / 'report.json') == source_hash
    pc.write(OUT / 'completion.json', completion)
    report_path = lab.ROOT / 'docs/research/x1012_tasks/A4_predictors.md'
    lines = ['\n## 描述量与AUROC\n', '|描述量|train拟合规则|留出AUROC|缺失n|不同值n|', '|---|---|---:|---:|---:|']
    for feature in FEATURES:
        rule = selectors[feature]
        text = ('constant ' + ('INDN' if rule['choose_N'] else 'IND0')) if rule['kind'] == 'constant' else f"INDN if {rule['direction']}*(x-{rule['threshold']:.6g})>0"
        summary = auc[feature]['all_domains']
        value = summary['train_oriented_weighted_auc']
        lines.append(f"|{feature}|{text}|{pr.fmt(value)}|{summary['missing_n']}|{summary['feature_unique_n']}|")
    lines += ['\n## 判据\n', '全部七量留出AUROC均低于.65才关闭参考统计切换；有量高于.75才具备后续连续调制资格。不可估计的AUROC不伪装成低于阈值。',
              '\n' + json.dumps(completion, ensure_ascii=False, indent=2),
              '\n原A4上界报告保持独立：' + str(UPPER / 'report.json') + '。']
    with report_path.open('a') as f:
        f.write('\n'.join(lines) + '\n')
    print(completion, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    args = parser.parse_args()
    produce() if args.mode == 'produce' else score()
