#!/usr/bin/env python3
"""CPU-only extra-supervised selector measurement on immutable saved tables.

Native class-mIoU is available ONLY for direct0 plus the argmax nominees of eight
predeclared label-free scores. All-candidate labels are PATCH IoU: full-candidate
selection is reported separately in that metric, never tested against 62 native
mIoU. No GT-dependent candidate inclusion, reconstruction of absent I/U, GPU,
model search, external download, or scientific-method claim.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

SCORES = ('rtg', 'cyc', 'link', 'fms', 'rtg*cyc', 'rtg*link', 'cyc*link', 'rtg*cyc*link')
MODEL_NAMES = ('ridge', 'hist_gradient_boosting')
SEED = 2044


def canonical_photo_id(value):
    value = str(value)
    basename = Path(value).name
    match = re.search(r'(\d+)(?:\.[A-Za-z0-9]+)?$', basename)
    return str(int(match.group(1))) if match else value


def load_manifest(path, rows):
    document = json.loads(Path(path).read_text())
    entries = document if isinstance(document, list) else document['records']
    by_episode = {}
    for r in entries:
        e = int(r['e'])
        if e in by_episode:
            raise ValueError(f'duplicate manifest episode {e}')
        if 'image_ids' in r:
            ids = r['image_ids']
        elif 'photo_ids' in r:
            ids = r['photo_ids']
        else:
            ids = [r['support_id'], r['query_id']] + r['pool_ids']
        photo_ids = frozenset(canonical_photo_id(x) for x in ids)
        if len(photo_ids) < 2:
            raise ValueError(f'manifest episode {e} has fewer than two photo IDs')
        by_episode[e] = (int(r['c']), photo_ids)
    required = {int(r['e']) for r in rows}
    if not required <= set(by_episode):
        raise ValueError(f'manifest missing episodes {sorted(required - set(by_episode))}')
    for r in rows:
        c, ids = by_episode[int(r['e'])]
        if c != int(r['c']):
            raise ValueError(f'manifest class mismatch at e={r["e"]}')
        if len(ids) < len(r['pseudo']['t']) + 1:
            raise ValueError(f'manifest lacks full support/query/pool photos at e={r["e"]}')
    return {e: ids for e, (_, ids) in by_episode.items()}


def ranks(values):
    """Average ranks handle ties without invented ordering information."""
    order = np.argsort(values, kind='stable')
    out = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        out[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return out / (len(values) - 1) if len(values) > 1 else np.full(1, .5)


def candidate_features(pseudo):
    raw = np.stack([np.asarray(pseudo[k], dtype=float) for k in SCORES], axis=1)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('non-finite or empty label-free score array')
    rank = np.stack([ranks(raw[:, k]) for k in range(raw.shape[1])], axis=1)
    std = raw.std(axis=0)
    z = (raw - raw.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
    direct = np.zeros((len(raw), 1))
    direct[0] = 1
    return np.concatenate([raw, rank, z, direct], axis=1)


def native_nominees(row):
    """Candidate set/recovered counts depend ONLY on direct and raw argmax choices.

    best1-pseudo is NEVER read here, even when a legal nominee happens to coincide
    with that oracle's index. Excluding a coincident nominee would itself use GT.
    """
    native = {0: np.asarray(row['iu']['1shot'], dtype=float)}
    choices = {}
    for key in SCORES:
        index = int(np.argmax(row['pseudo'][key]))
        iu = np.asarray(row['iu']['sel-' + key], dtype=float)
        if iu.shape != (2,) or not np.isfinite(iu).all() or min(iu) < 0 or iu[0] > iu[1]:
            raise ValueError(f'invalid saved I/U for e={row["e"]}, key={key}')
        if index in native and not np.array_equal(native[index], iu):
            raise ValueError(f'inconsistent saved I/U for same candidate at e={row["e"]}')
        native[index] = iu
        choices[key] = index
    first = native[0]
    if first.shape != (2,) or not np.isfinite(first).all() or min(first) < 0 or first[0] > first[1]:
        raise ValueError('invalid saved direct I/U')
    return native, choices


def purged_train_indices(rows, photos, held_out):
    tests = [i for i, r in enumerate(rows) if int(r['c']) == held_out]
    held_photos = frozenset().union(*(photos[int(rows[i]['e'])] for i in tests))
    unpurged = [i for i, r in enumerate(rows) if int(r['c']) != held_out]
    train = [i for i in unpurged if not (photos[int(rows[i]['e'])] & held_photos)]
    assert all(not (photos[int(rows[i]['e'])] & held_photos) for i in train)
    return train, tests, len(unpurged), len(held_photos)


def fixed_models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return dict(ridge=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        hist_gradient_boosting=HistGradientBoostingRegressor(max_iter=200, max_depth=3,
            min_samples_leaf=20, learning_rate=.05, l2_regularization=1.0,
            random_state=SEED, early_stopping=False))


def native_miou(classes, iu):
    return 100 * float(np.mean([iu[classes == c, 0].sum() / max(iu[classes == c, 1].sum(), 1)
                               for c in np.unique(classes)]))


def bootstrap_native(classes, arrays, repetitions, seed=SEED):
    """Frozen OOF paired sensitivity; does not rerun fitting or imply independence."""
    classes = np.asarray(classes)
    unique = np.unique(classes)
    rng = np.random.default_rng(seed)
    episode_draws = {k: np.zeros(repetitions) for k in arrays}
    class_ratios = {k: [] for k in arrays}
    for c in unique:
        take = np.flatnonzero(classes == c)
        sampled = rng.integers(0, len(take), size=(repetitions, len(take)))
        for name, iu in arrays.items():
            subset = iu[take]
            episode_draws[name] += (subset[sampled, 0].sum(1) /
                                    np.maximum(subset[sampled, 1].sum(1), 1)) / len(unique) * 100
            class_ratios[name].append(subset[:, 0].sum() / max(subset[:, 1].sum(), 1) * 100)
    class_draw = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    class_block_draws = {k: np.asarray(v)[class_draw].mean(1) for k, v in class_ratios.items()}
    result = {}
    for name, iu in arrays.items():
        point = native_miou(classes, iu)
        result[name] = dict(native_class_miou=point, gain_vs_1shot_pp=point - native_miou(classes, arrays['1shot']))
        for label, draws in [('paired_episode_stratified_sensitivity', episode_draws),
                             ('paired_class_block_sensitivity', class_block_draws)]:
            result[name][label] = dict(score_95=list(np.quantile(draws[name], [.025, .975])),
                gain_vs_1shot_95=list(np.quantile(draws[name] - draws['1shot'], [.025, .975])),
                gain_vs_raw_rtg_95=list(np.quantile(draws[name] - draws['raw_rtg'], [.025, .975])))
    return result


def bootstrap_patch(classes, arrays, repetitions, seed=SEED):
    classes = np.asarray(classes)
    unique = np.unique(classes)
    rng = np.random.default_rng(seed)
    episode = {k: np.zeros(repetitions) for k in arrays}
    class_sums, class_counts = {k: [] for k in arrays}, []
    for c in unique:
        take = np.flatnonzero(classes == c)
        sample = rng.integers(0, len(take), size=(repetitions, len(take)))
        class_counts.append(len(take))
        for k, values in arrays.items():
            episode[k] += values[take][sample].sum(1) / len(classes)
            class_sums[k].append(values[take].sum())
    draw = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    block = {k: np.asarray(v)[draw].sum(1) / np.asarray(class_counts)[draw].sum(1)
             for k, v in class_sums.items()}
    result = {}
    for k, values in arrays.items():
        result[k] = dict(mean_episode_patch_iou=float(values.mean()))
        for label, samples in [('paired_episode_stratified_sensitivity', episode),
                               ('paired_class_block_sensitivity', block)]:
            result[k][label] = dict(score_95=list(np.quantile(samples[k], [.025, .975])),
                gain_vs_direct_95=list(np.quantile(samples[k] - samples['direct'], [.025, .975])),
                gain_vs_raw_rtg_95=list(np.quantile(samples[k] - samples['raw_rtg'], [.025, .975])))
    return result


def self_check():
    pseudo = {k: [.5, .9, .1] for k in SCORES}
    pseudo['link'] = [1, .2, .1]
    pseudo['t'] = [.1, .3, .99]
    row = dict(e=0, c=1, pseudo=pseudo, iu={'1shot': [1, 10], 'best1-pseudo': [99, 100]})
    row['iu'].update({'sel-' + k: ([1, 10] if k == 'link' else [3, 10]) for k in SCORES})
    legal, _ = native_nominees(row)
    assert set(legal) == {0, 1} and 2 not in legal
    before = candidate_features(pseudo)
    row['iu']['best1-pseudo'] = [0, 100]
    pseudo['t'] = [.99, .8, 0]
    assert set(native_nominees(row)[0]) == {0, 1}
    assert np.array_equal(before, candidate_features(pseudo))
    assert before.shape == (3, 25)
    assert np.array_equal(ranks(np.array([2, 2, 1])), np.array([.75, .75, 0]))
    rows = [dict(e=0, c=1), dict(e=1, c=2), dict(e=2, c=2)]
    photos = {0: {'1', '2'}, 1: {'2', '3'}, 2: {'4', '5'}}
    train, tests, _, _ = purged_train_indices(rows, photos, 1)
    assert train == [2] and tests == [0]
    identical = np.array([[2., 3.], [1., 4.], [4., 5.]])
    b = bootstrap_native(np.array([1, 1, 2]), {'1shot': identical, 'raw_rtg': identical}, 50)
    assert b['raw_rtg']['paired_episode_stratified_sensitivity']['gain_vs_1shot_95'] == [0., 0.]
    assert canonical_photo_id('COCO_val2014_000000000001.jpg') == '1'
    print(json.dumps(dict(self_check='PASSED', gt_dependent_nominee_excluded=True,
        quality_labels_do_not_enter_features=True, photo_overlap_purge=True,
        same_candidate_IU_verified=True, tied_ranks=True, paired_identical_zero_interval=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--select', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()
    if a.self_check:
        self_check()
        return
    if not all((a.select, a.manifest, a.out)) or a.bootstrap < 100:
        ap.error('--select, --manifest, --out required; bootstrap >=100')
    t0 = time.monotonic()
    source = json.loads(a.select.read_text())
    rows = source['records']
    if not rows or len({int(r['e']) for r in rows}) != len(rows):
        raise ValueError('empty table or duplicate episode IDs')
    photos = load_manifest(a.manifest, rows)
    features = [candidate_features(r['pseudo']) for r in rows]
    targets = [np.asarray(r['pseudo']['t'], dtype=float) for r in rows]
    natives, raw_choices = [], []
    for r, x, y in zip(rows, features, targets):
        if len(x) != len(y) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError(f'invalid saved patch targets at e={r["e"]}')
        native, raw = native_nominees(r)
        natives.append(native)
        raw_choices.append(raw)
    classes = np.array([int(r['c']) for r in rows])
    predictions = {name: {} for name in MODEL_NAMES}
    report = dict(state='RUNNING', args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.select, a.manifest, Path(__file__))},
        protocol='Development-only single-fold LOCO; purge ALL support/query/pool photo IDs. Additional supervised measuring devices.',
        fixed_features=dict(raw=list(SCORES), within_episode_ranks=True, within_episode_zscores=True, direct_flag=True,
            forbidden=['class ID', 'episode ID', 'GT-dependent candidate inclusion', 'GT quality as input']),
        fixed_models=dict(ridge='StandardScaler + Ridge(alpha=1)',
            hist_gradient_boosting='max_iter=200, max_depth=3, min_samples_leaf=20, lr=.05, l2=1, random_state=2044, early_stopping=False'),
        native_policy='direct0 plus eight score argmax nominees only; best1-pseudo never supplies candidates or I/U',
        limitations=['Training targets are patch-grid IoU, not native IoU.',
            'Native subset oracle and learned scores are lower-bound evidence of achievable policies, not information upper bounds.',
            'Intervals are frozen-OOF paired sensitivity: photo-sharing test units are not independent, models are not refit per bootstrap.',
            'The 62 native-class-mIoU cutoff does not apply to full-candidate patch metrics.',
            'A fixed-model negative cannot reject all selectors, missing native I/U cannot be fabricated.'],
        folds=[], records=[])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def write():
        report['elapsed_s'] = time.monotonic() - t0
        tmp = a.out.with_suffix(a.out.suffix + '.tmp')
        tmp.write_text(json.dumps(report, allow_nan=False))
        tmp.replace(a.out)
    write()
    try:
        for c in np.unique(classes):
            tr, te, unpurged, held_photos = purged_train_indices(rows, photos, int(c))
            fold = dict(held_out_class=int(c), test_episodes=len(te), train_episodes_before_photo_purge=unpurged,
                train_episodes_after_photo_purge=len(tr), purged_train_episodes=unpurged-len(tr),
                held_out_photo_ids=held_photos, photo_intersection_verified_zero=True, model_status={})
            report['folds'].append(fold)
            print(json.dumps(dict(event='fold_start', **fold)), flush=True)
            if not tr:
                fold['status'] = 'NOT_EVALUABLE_NO_TRAIN_AFTER_PURGE'
                write()
                continue
            X = np.concatenate([features[i] for i in tr])
            y = np.concatenate([targets[i] for i in tr])
            fold['train_candidate_rows'] = len(y)
            for name, model in fixed_models().items():
                try:
                    model.fit(X, y)
                    for i in te:
                        score = model.predict(features[i])
                        if not np.isfinite(score).all():
                            raise ValueError('nonfinite model prediction')
                        predictions[name][i] = score
                    fold['model_status'][name] = 'EVALUATED'
                except Exception as ex:
                    for i in te:
                        predictions[name].pop(i, None)
                    fold['model_status'][name] = 'NOT_EVALUABLE_FIT_ERROR: ' + repr(ex)
            fold['status'] = 'EVALUATED' if all(v == 'EVALUATED' for v in fold['model_status'].values()) else 'PARTIAL_OR_ERROR'
            print(json.dumps(dict(event='fold_done', **fold)), flush=True)
            write()
        eligible = sorted(set.intersection(*(set(predictions[name]) for name in MODEL_NAMES)))
        report['coverage'] = dict(input_episodes=len(rows), common_evaluable_episodes=len(eligible),
            input_classes=len(np.unique(classes)), common_evaluable_classes=len(np.unique(classes[eligible])),
            candidate_counts=dict(Counter(len(x) for x in features)),
            native_nominee_count_mean=float(np.mean([len(x) for x in natives])),
            native_nominee_count_min=min(map(len, natives)), native_nominee_count_max=max(map(len, natives)),
            native_candidate_fraction=float(sum(map(len, natives)) / sum(map(len, features))))
        if not eligible:
            report['state'] = 'NOT_EVALUABLE'
            report['cutoff_62'] = 'NOT_EVALUABLE: no common held-out photo-purged predictions'
            write()
            return
        native_arrays, patch_arrays = {}, {}
        for i in eligible:
            row, n, raw, t = rows[i], natives[i], raw_choices[i], targets[i]
            legal = sorted(n)
            picks = {'1shot': 0, **{'raw_' + k: raw[k] for k in SCORES}}
            # GT is used only in this declared evaluation intervention, after legal nominees freeze.
            picks['diagnostic_native_nominee_episode_oracle'] = max(legal, key=lambda j: (n[j][0] / max(n[j][1], 1), -j))
            patch_picks = {'direct': 0, **{'raw_' + k: raw[k] for k in SCORES},
                           'diagnostic_full_candidate_patch_oracle': int(np.argmax(t)),
                           'diagnostic_nominee_patch_oracle': max(legal, key=lambda j: (t[j], -j))}
            choices = {}
            for name in MODEL_NAMES:
                scores = predictions[name][i]
                chosen = max(legal, key=lambda j: (scores[j], -j))
                full_chosen = int(np.argmax(scores))
                picks[name] = chosen
                patch_picks[name] = full_chosen
                choices[name] = dict(native_nominee_index=chosen, full_candidate_index=full_chosen,
                                     full_candidate_outside_native_nominees=full_chosen not in n)
            for key, chosen in picks.items():
                native_arrays.setdefault(key, []).append(n[chosen])
            for key, chosen in patch_picks.items():
                patch_arrays.setdefault(key, []).append(t[chosen])
            report['records'].append(dict(e=int(row['e']), c=int(row['c']), native_nominees=legal,
                full_candidates=len(t), choices=choices, native_iu={k: n[j].tolist() for k, j in picks.items()},
                full_candidate_patch_quality={k: float(t[j]) for k, j in patch_picks.items()},
                patch_oracle_in_native_nominees=int(np.argmax(t)) in n))
        selected_classes = classes[eligible]
        native_arrays = {k: np.asarray(v) for k, v in native_arrays.items()}
        patch_arrays = {k: np.asarray(v) for k, v in patch_arrays.items()}
        report['native_class_miou'] = bootstrap_native(selected_classes, native_arrays, a.bootstrap)
        report['full_candidate_patch_metric_ONLY'] = bootstrap_patch(selected_classes, patch_arrays, a.bootstrap)
        report['anchors_on_same_evaluable_episodes'] = {
            k: native_miou(selected_classes, np.asarray([rows[i]['iu'][k] for i in eligible]))
            for k in ('1shot', 'pool', 'true') if all(k in rows[i]['iu'] for i in eligible)}
        report['nominee_coverage_diagnostic'] = dict(
            patch_oracle_inside_legal_fraction=float(np.mean([r['patch_oracle_in_native_nominees'] for r in report['records']])),
            learned_full_candidate_outside_legal_fraction={name: float(np.mean([
                r['choices'][name]['full_candidate_outside_native_nominees'] for r in report['records']])) for name in MODEL_NAMES})
        full_coverage = len(eligible) == len(rows)
        report['cutoff_62'] = {name: dict(native_score=report['native_class_miou'][name]['native_class_miou'],
            reaches_62=report['native_class_miou'][name]['native_class_miou'] >= 62,
            coverage_complete=full_coverage,
            interpretation='Development threshold for this frozen nominee policy only; no family-wide rejection or external proof')
            for name in MODEL_NAMES}
        report['state'] = 'COMPLETED' if full_coverage else 'PARTIALLY_EVALUABLE'
        write()
        print(json.dumps(dict(state=report['state'], coverage=report['coverage'],
            native_class_miou={k: v['native_class_miou'] for k, v in report['native_class_miou'].items()},
            cutoff_62=report['cutoff_62'])), flush=True)
    except BaseException as ex:
        report.update(state='ERROR', error=repr(ex))
        write()
        raise


if __name__ == '__main__':
    main()
