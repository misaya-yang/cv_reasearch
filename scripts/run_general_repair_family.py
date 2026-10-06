#!/usr/bin/env python3
"""Exact canonical UNION all-general repair branch, using sealed DEV atoms only.

General policy: (origin OR positive producers) AND every negative producer.
All existing allowed origins are retained. This is NOT the larger family that
mixes canonical and general deletion policies separately for each action.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for env_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[env_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.experiment import sha, metric, photo_groups
from run_exact_family_selection import checked, mask, recipe_lut, zeta, edit_attribution


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def exact_score(i, u):
    return sum((Fraction(int(a), int(b)) for a, b in zip(i, u)), Fraction()) * Fraction(100, len(i))


def named_general(u, v, arms, score=None):
    if not u or u & v:
        raise ValueError('Reduced general branch needs nonempty disjoint U,V')
    origin = (u & -u).bit_length() - 1
    rec = dict(policy='general_delete', origin=origin, positive=u, negative=v,
               cost=bin(u | v).count('1'), origin_method=arms[origin]['id'],
               add_methods=[arm['id'] for i, arm in enumerate(arms) if u & (1 << i) and i != origin],
               delete_methods=[arm['id'] for i, arm in enumerate(arms) if v & (1 << i)],
               formula='(origin OR positives) AND intersection negative masks')
    if score is not None:
        rec['score'] = float(score)
        rec['exact_score_fraction'] = str(score)
    return rec


def lut(rec, length):
    if rec['policy'] == 'canonical':
        return recipe_lut(rec, length)
    p = np.arange(1 << length)
    return ((p & rec['negative']) == rec['negative']) & ((p & rec['positive']) != 0)


def recipe_counts(h, rec):
    on = lut(rec, int(round(np.log2(h.shape[1]))))
    i = h[:, on, 1].sum(axis=1, dtype=np.int64)
    u = h[:, :, 1].sum(axis=1, dtype=np.int64) + h[:, on, 0].sum(axis=1, dtype=np.int64)
    return i, u


def toy_check():
    """Exhaustively compare original Boolean family with reduced representation."""
    length = 3
    patterns = np.arange(1 << length)
    original = set()
    reduced = set()
    for origin in range(length):
        others = [i for i in range(length) if i != origin]
        for a in range(1 << len(others)):
            for b in range(1 << len(others)):
                positive = 1 << origin
                negative = 0
                for j, i in enumerate(others):
                    if a & (1 << j):
                        positive |= 1 << i
                    if b & (1 << j):
                        negative |= 1 << i
                truth_table = ((patterns & positive) != 0) & ((patterns & negative) == negative)
                original.add(truth_table.tobytes())
    for u in range(1, 1 << length):
        for v in range(1 << length):
            if u & v:
                continue
            rec = named_general(u, v, [{'id': str(i)} for i in range(length)])
            if not u & (1 << rec['origin']) or v & (1 << rec['origin']):
                raise ValueError('Illegal origin mapping')
            reduced.add(lut(rec, length).tobytes())
    if original != reduced:
        raise ValueError('Reduced general branch changes original allowed-origin family')
    # An overlapping positive/negative selection collapses to AND(negative).
    for u in range(1, 1 << length):
        for v in range(1, 1 << length):
            if u & v:
                on = ((patterns & u) != 0) & ((patterns & v) == v)
                if not np.array_equal(on, (patterns & v) == v):
                    raise ValueError('Overlap normalization failed')
    return dict(n=length, all_original_syntactic_recipes=length * 4 ** (length - 1),
                reduced_recipes=3 ** length - 2 ** length,
                distinct_truth_tables=len(original), full_truth_table_sets_exact=True)


def identity(row):
    return tuple(row.get(k) for k in ('key', 'c', 'fold', 'e', 'batch', 'public_batch', 'support', 'query'))


def prepare(a):
    started = time.monotonic()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    toy = toy_check()
    source = a.source
    manifest = read(source / 'manifest.json')
    arms = read(source / 'arms.json')
    rows = manifest['rows']
    complete = read(source / 'histogram_complete.json')
    if complete['state'] != 'MEMBERSHIP_HISTOGRAM_COMPLETE' or len(rows) != 4000:
        raise ValueError('Expected terminal full4000 histogram')
    if complete['arms'] != [arm['id'] for arm in arms] or len(arms) != 13:
        raise ValueError('Expected exact existing extended13 order')
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Sample draw keys must be distinct; episode repeats are retained')
    h = np.load(source / 'class_patterns.npy', mmap_mode='r')
    patterns = np.load(source / 'patterns.npy', mmap_mode='r')
    if h.shape != (80, 8192, 2) or h.dtype != np.int64:
        raise ValueError('Unexpected class membership histogram')
    if patterns.shape != (4000, 8192, 2) or patterns.dtype != np.uint32:
        raise ValueError('Unexpected per-draw membership histogram')
    cls = np.array([r['c'] for r in rows])
    class_ids = np.unique(cls)
    recomputed = np.zeros(h.shape, dtype=np.int64)
    for start in range(0, len(rows), 64):
        block = np.asarray(patterns[start:start + 64])
        if np.any(block.sum(axis=(1, 2), dtype=np.int64) != 1024 ** 2):
            raise ValueError('Nonexhaustive per-draw histogram')
        for c in np.unique(cls[start:start + 64]):
            recomputed[np.searchsorted(class_ids, c)] += block[cls[start:start + 64] == c].sum(axis=0, dtype=np.int64)
    if not np.array_equal(h, recomputed):
        raise ValueError('Class atoms differ from complete per-draw atoms')
    with np.load(source / 'baseline_counts.npz', allow_pickle=False) as z:
        baseline = {arm['id']: z[arm['id']].copy() for arm in arms}
    p = np.arange(1 << len(arms))
    truth = patterns[:, :, 1].sum(axis=1, dtype=np.int64)
    for i, arm in enumerate(arms):
        on = (p & (1 << i)) != 0
        iu = np.stack((patterns[:, on, 1].sum(axis=1, dtype=np.int64),
                       truth + patterns[:, on, 0].sum(axis=1, dtype=np.int64)), axis=1)
        if not np.array_equal(iu, baseline[arm['id']]):
            raise ValueError('Original source baseline I/U does not close')
    write(a.out / 'manifest.json', manifest)
    files = {}
    for name in ('manifest.json', 'arms.json', 'class_patterns.npy', 'patterns.npy',
                 'baseline_counts.npz', 'histogram_complete.json', 'input_receipt.json',
                 'finalists_sealed.json', 'optimization.json'):
        files[str(source / name)] = sha(source / name)
    policies = {}
    for policy, canonical, held in [('strict', a.strict_canonical, a.strict_held),
                                     ('extended', source, a.extended_held)]:
        destination = a.out / policy
        destination.mkdir()
        active = [i for i, arm in enumerate(arms) if policy == 'extended' or arm['strict_eligible']]
        if active != list(range(12 if policy == 'strict' else 13)):
            raise ValueError('Policy producer map changed; no guessed projection')
        selected_arms = [arms[i] for i in active]
        write(destination / 'arms.json', selected_arms)
        length = len(selected_arms)
        canonical_arms = read(canonical / 'arms.json')
        held_arms = read(held / 'arms.json')
        if canonical_arms != selected_arms or held_arms != selected_arms:
            raise ValueError('Canonical source arm order/provenance differs')
        for provider in (canonical, held):
            provider_manifest = read(provider / 'manifest.json')
            if [identity(r) for r in provider_manifest['rows']] != [identity(r) for r in rows]:
                raise ValueError('Canonical comparison draw identities differ')
            for old, new in zip(provider_manifest['rows'], rows):
                for arm in selected_arms:
                    if old['masks'][arm['id']] != new['masks'][arm['id']]:
                        raise ValueError('Canonical source mask receipt differs')
                if old['packet'] != new['packet']:
                    raise ValueError('Canonical GT packet differs')
        canonical_opt = read(canonical / 'optimization.json')
        if 'allowed_origin_ids' in canonical_opt:
            allowed_origin_ids = canonical_opt['allowed_origin_ids']
            origin_contract = 'explicit allowed_origin_ids'
        else:
            inventory = sorted(entry['origin'] for entry in canonical_opt['per_origin'])
            if inventory != list(range(length)) or canonical_opt['recipes_evaluated'] != length * 4 ** (length - 1):
                raise ValueError('Legacy origin inventory is not the complete existing bank')
            allowed_origin_ids = [selected_arms[i]['id'] for i in inventory]
            origin_contract = 'legacy explicit per_origin inventory plus full recipe count'
        if allowed_origin_ids != [arm['id'] for arm in selected_arms]:
            raise ValueError('Reduction requires exactly existing ALL allowed origins')
        global_rec = dict(canonical_opt['selected_recipes'][0], policy='canonical')
        final = read(canonical / 'final_recipes.json')['recipes']
        global_key = next('recipe%d' % i for i, r in enumerate(final)
                          if (r['origin'], r['a'], r['b']) == (global_rec['origin'], global_rec['a'], global_rec['b']))
        global_seal = read(canonical / 'finalists_sealed.json')
        held_selection = read(held / 'selection.json')
        held_seal = read(held / 'sealed.json')
        if global_seal['state'] != 'ALL_FINALIST_MASKS_SEALED' or held_seal['state'] != 'ALL_HELDFOLD_MASKS_SEALED':
            raise ValueError('Canonical complete mask comparison is unsealed')
        if global_seal['n'] != len(rows) or held_seal['n'] != len(rows):
            raise ValueError('Incomplete canonical controls')
        checked(canonical / 'manifest.json', global_seal['manifest_sha256'])
        checked(canonical / 'final_recipes.json', global_seal['recipes_sha256'])
        checked(held / 'manifest.json', held_seal['manifest_sha256'])
        checked(held / 'selection.json', held_seal['selection_sha256'])
        checks = {'global': global_rec}
        training = {}
        for fold in range(4):
            entry = held_selection['folds'][str(fold)]
            checks[str(fold)] = dict(entry['selected'], policy='canonical')
            training[str(fold)] = dict(training_n=entry['training_n'], held_n=entry['held_n'],
                                       training_classes=entry['training_classes'], held_classes=entry['held_classes'])
        for label, rec in checks.items():
            if label == 'global':
                values = np.asarray(h)
            else:
                values = np.load(a.extended_held / ('train_fold%s_patterns.npy' % label))
            if policy == 'strict':
                values = values[:, :4096, :] + values[:, 4096:, :]
            reference = np.load((canonical / 'class_patterns.npy') if label == 'global' else held / ('train_fold%s_patterns.npy' % label))
            if not np.array_equal(values, reference):
                raise ValueError('Canonical training atoms differ from projected existing source')
            if np.any(values.sum(axis=1)[:, 1] <= 0) or values.sum(axis=(1, 2)).max() >= 2 ** 53:
                raise ValueError('Score arithmetic preconditions fail')
            i, u = recipe_counts(values, rec)
            score = exact_score(i, u)
            if abs(float(score) - rec['score']) > 1e-10:
                raise ValueError('Old canonical selected score parity failed')
            rec['exact_score_fraction'] = str(score)
            path = destination / (label + '_patterns.npy')
            np.save(path, values)
            files[str(path)] = sha(path)
        for provider, names in [(canonical, ('manifest.json', 'arms.json', 'optimization.json', 'final_recipes.json', 'finalists_sealed.json')),
                                (held, ('manifest.json', 'arms.json', 'selection.json', 'sealed.json'))]:
            for name in names:
                files[str(provider / name)] = sha(provider / name)
        for fold in range(4):
            path = held / ('train_fold%d_patterns.npy' % fold)
            files[str(path)] = sha(path)
        policies[policy] = dict(arms=selected_arms, canonical=checks, training=training,
                               source_global=str(canonical), source_held=str(held),
                               canonical_global_npz_key=global_key,
                               canonical_global_seal=global_seal, canonical_held_seal=held_seal,
                               allowed_origin_ids=allowed_origin_ids, allowed_origin_contract_source=origin_contract,
                               general_recipes_per_search=3 ** length - 2 ** length)
    write(a.out / 'prepared.json', dict(state='IMMUTABLE_ATOMS_AND_CANONICAL_PARITY_PREPARED', n=len(rows),
          source=str(source), policies=policies, files=files, toy_audit=toy,
          source_baseline_IU_all13_exact=True, per_draw_class_histogram_exact=True,
          source_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('run_exact_family_selection.py')),
          GT_use='existing labelled DEV atoms for global/train3 selection; no per-query routing',
          scope='canonical UNION all-general, NOT action-wise mixed canonical/general deletion',
          cost='outer complete producer count proxy only, not atomic upstream DAG/time',
          seconds=time.monotonic() - started))
    write(a.out / 'state.json', dict(state='PREPARED', seconds=time.monotonic() - started))
    print('PREPARED', flush=True)


def optimize_general(job):
    path, policy, label, arms = job
    started = time.monotonic()
    h = np.load(path)
    length = len(arms)
    classes = h.shape[0]
    truth = h[:, :, 1].sum(axis=1)
    if np.any(truth <= 0):
        raise ValueError('Empty-class convention must be explicit')
    pop = np.array([bin(x).count('1') for x in range(1 << length)])
    guard = 100 * 4 * (classes + 8) * np.finfo(np.float64).eps
    best = {cost: None for cost in range(1, length + 1)}
    count = 0
    rational_checks = 0
    for negative in range(1 << length):
        free = [i for i in range(length) if not negative & (1 << i)]
        size = 1 << len(free)
        if size == 1:
            continue
        small = np.arange(size, dtype=np.int64)
        indices = np.full(size, negative, dtype=np.int64)
        positive_bits = np.zeros(size, dtype=np.int64)
        for j, i in enumerate(free):
            positive_bits |= ((small >> j) & 1) << i
        indices |= positive_bits
        table = zeta(h[:, indices, :], True)
        local = small[1:]
        foreground = (table[:, -1:, :] - table[:, (size - 1) ^ local, :]).transpose(1, 0, 2)
        numerator = foreground[:, :, 1]
        denominator = truth[None, :] + foreground[:, :, 0]
        scores = 100 * np.mean(numerator / denominator, axis=1)
        costs = pop[negative] + pop[positive_bits[1:]]
        count += size - 1
        for cost in np.unique(costs):
            candidates = np.flatnonzero(costs == cost)
            maximum = float(scores[candidates].max())
            old = best[int(cost)]
            if old is not None and maximum + guard < float(old['fraction']) - guard:
                continue
            candidates = candidates[scores[candidates] >= maximum - 2 * guard]
            for pos in candidates:
                score = exact_score(numerator[pos], denominator[pos])
                rational_checks += 1
                positive = int(positive_bits[pos + 1])
                origin = (positive & -positive).bit_length() - 1
                tie = (origin, positive, negative)
                if old is None or score > old['fraction'] or (score == old['fraction'] and tie < old['tie']):
                    old = dict(fraction=score, tie=tie, recipe=named_general(positive, negative, arms, score))
            best[int(cost)] = old
    if count != 3 ** length - 2 ** length:
        raise ValueError('General finite family not exhausted')
    selected = min(best.values(), key=lambda value: (-value['fraction'], value['recipe']['cost'], value['tie']))
    return dict(policy=policy, label=label, selected=selected['recipe'],
                best_by_cost={str(k): value['recipe'] for k, value in best.items()},
                recipes_evaluated=count, rational_finalist_comparisons=rational_checks,
                float64_roundoff_guard_pp=guard,
                arithmetic='integer atoms; conservative IEEE float64 guard; exact Fraction comparison of every potentially winning near-max candidate',
                origin_mapping='O=lowest set bit of positive U, positives=U without O, negative=V; U nonempty and U/V disjoint',
                seconds=time.monotonic() - started)


def optimize(a):
    started = time.monotonic()
    prep = read(a.out / 'prepared.json')
    if prep['state'] != 'IMMUTABLE_ATOMS_AND_CANONICAL_PARITY_PREPARED':
        raise ValueError('Preparation is incomplete')
    checked(__file__, prep['source_sha256'])
    for path, digest in prep['files'].items():
        checked(path, digest)
    jobs = [(str(a.out / policy / (label + '_patterns.npy')), policy, label, data['arms'])
            for policy, data in prep['policies'].items() for label in ('global', '0', '1', '2', '3')]
    results = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(optimize_general, jobs):
            policy, label = result['policy'], result['label']
            results.setdefault(policy, {})[label] = result
            write(a.out / 'state.json', dict(state='GENERAL_BRANCH_EXACT_SEARCH', completed=sum(map(len, results.values())),
                                           total=len(jobs), seconds=time.monotonic() - started))
            print(json.dumps(dict(policy=policy, label=label, selected=result['selected'], seconds=result['seconds'])), flush=True)
    selection = {}
    for policy, data in prep['policies'].items():
        selection[policy] = {}
        for label, result in results[policy].items():
            canonical = data['canonical'][label]
            general = result['selected']
            cscore, gscore = Fraction(canonical['exact_score_fraction']), Fraction(general['exact_score_fraction'])
            chosen = min([canonical, general], key=lambda r: (-Fraction(r['exact_score_fraction']), r['cost'], r['policy']))
            if Fraction(chosen['exact_score_fraction']) < cscore:
                raise ValueError('Union family lost canonical option')
            selection[policy][label] = dict(canonical=canonical, general=general, selected=chosen,
                                             general_minus_canonical_train_pp=float(gscore - cscore))
    write(a.out / 'selection.json', dict(state='EXACT_BRANCH_UNION_GLOBAL_AND_TRAINFOLD_SELECTED', policies=selection,
          allowed_origin_ids={p: d['allowed_origin_ids'] for p, d in prep['policies'].items()},
          general_search=results, canonical_search_reused_not_rerun=True,
          scope=prep['scope'], selection='one global DEV recipe and one fixed train3 recipe per held fold; query GT is absent from inference',
          tie_rule='exact rational score within new general branch; then outer producer proxy and lexical origin/U/V; branch union score/cost then policy',
          inherited_canonical_arithmetic='existing exhaustive float64 optimizer; selected cached recipes rationally recounted, old canonical search not repeated',
          seconds=time.monotonic() - started))
    write(a.out / 'state.json', dict(state='RECIPES_FROZEN_BEFORE_MASK_CONSTRUCTION', seconds=time.monotonic() - started))


def render_recipe(rec, values, length):
    if rec['policy'] == 'general_delete':
        result = np.zeros(131072, dtype=np.uint8)
        for i in range(length):
            if rec['positive'] & (1 << i):
                result |= values[i]
        for i in range(length):
            if rec['negative'] & (1 << i):
                result &= values[i]
        return result
    origin = values[rec['origin']]
    result = origin.copy()
    others = [i for i in range(length) if i != rec['origin']]
    for j, i in enumerate(others):
        if rec['a'] & (1 << j):
            result |= values[i] & ~origin
    for j, i in enumerate(others):
        if rec['b'] & (1 << j):
            result &= ~(origin & ~values[i])
    return result


def selected_indices(rec, length):
    if rec['policy'] == 'general_delete':
        return {i for i in range(length) if (rec['positive'] | rec['negative']) & (1 << i)}
    others = [i for i in range(length) if i != rec['origin']]
    return {rec['origin']} | {i for j, i in enumerate(others) if (rec['a'] | rec['b']) & (1 << j)}


def row_recipes(selection, row):
    output = {}
    for branch in ('canonical', 'general', 'selected'):
        output[branch + '_global'] = selection['global'][branch]
        output[branch + '_held'] = selection[str(row['fold'])][branch]
    return output


def mask_one(job):
    row, policy, data, selection, destination = job
    recs = row_recipes(selection, row)
    arms = data['arms']
    needed = set().union(*(selected_indices(r, len(arms)) for r in recs.values()))
    values = {}
    for i in needed:
        record = row['masks'][arms[i]['id']]
        checked(record['path'], record['sha256'])
        values[i] = mask(record)
    outputs = {name: render_recipe(r, values, len(arms)) for name, r in recs.items()}
    for role, directory, seal, npz_key in [
            ('canonical_global', data['source_global'], data['canonical_global_digest'], data['canonical_global_npz_key']),
            ('canonical_held', data['source_held'], data['canonical_held_digest'], 'recipe0')]:
        path = Path(directory) / 'finalists' / (row['key'] + '.npz')
        checked(path, seal)
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(outputs[role], z[npz_key]):
                raise ValueError('Reconstructed canonical mask differs from original sealed comparison')
    path = Path(destination) / 'predictions' / (row['key'] + '.npz')
    np.savez_compressed(path, **outputs)
    return row['key'], sha(path)


POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def count_one(job):
    row, arms, destination, digest = job
    path = Path(destination) / 'predictions' / (row['key'] + '.npz')
    checked(path, digest)
    checked(row['packet']['path'], row['packet']['sha256'])
    with np.load(row['packet']['path'], allow_pickle=False) as z:
        truth = z['truth'].copy()
    if truth.dtype != np.uint8 or truth.shape != (131072,):
        raise ValueError('Expected packed query GT')
    native_ids = [i for i, arm in enumerate(arms) if arm['npz_key'] == 'native']
    if len(native_ids) != 1:
        raise ValueError('Expected one complete native control')
    record = row['masks'][arms[native_ids[0]]['id']]
    checked(record['path'], record['sha256'])
    native = mask(record)
    result = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            pred = z[name]
            add, delete = pred & ~native, native & ~pred
            result[name] = dict(iu=[int(POPCOUNT[pred & truth].sum()), int(POPCOUNT[pred | truth].sum())],
                                add_TP=int(POPCOUNT[add & truth].sum()), add_FP=int(POPCOUNT[add & ~truth].sum()),
                                delete_TP=int(POPCOUNT[delete & truth].sum()), delete_FP=int(POPCOUNT[delete & ~truth].sum()))
    return result


def summary(rows, arrays, corrections):
    cls = np.array([r['c'] for r in rows])
    groups = photo_groups(rows)
    n_groups = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(n_groups, size=(2000, n_groups))
    weights = np.stack([np.bincount(draw, minlength=n_groups) for draw in draws])[:, groups]
    scores = {name: metric(values, cls) for name, values in arrays.items()}
    samples = {name: np.array([metric(values, cls, weight) for weight in weights]) for name, values in arrays.items()}
    report = dict(n=len(rows), classes=len(np.unique(cls)), photo_groups=n_groups,
                  largest_photo_group=int(np.bincount(groups).max()), scores=scores, contrasts={},
                  bootstrap=dict(draws=2000, rng='RandomState(0)', unit='connected support/query photographs'))
    for name in corrections:
        report['contrasts'][name] = {}
        for base, values in arrays.items():
            if base == name:
                continue
            delta = arrays[name][:, 0] / np.maximum(arrays[name][:, 1], 1) - values[:, 0] / np.maximum(values[:, 1], 1)
            report['contrasts'][name][base] = dict(gain=scores[name] - scores[base],
                    ci95=np.percentile(samples[name] - samples[base], [2.5, 97.5]).tolist(),
                    up=int((delta > 1e-12).sum()), down=int((delta < -1e-12).sum()), tie=int((np.abs(delta) <= 1e-12).sum()))
    for field in ('fold', 'batch'):
        report[field + 's'] = {}
        for label in sorted({str(r[field]) for r in rows}):
            ix = np.array([str(r[field]) == label for r in rows])
            values = {name: metric(iu[ix], cls[ix]) for name, iu in arrays.items()}
            report[field + 's'][label] = dict(n=int(ix.sum()), scores=values,
                                            gain_vs_native={name: score - values['native'] for name, score in values.items()})
    report['corrections_vs_native'] = {name: {key: int(sum(r[key] for r in records))
              for key in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')} for name, records in corrections.items()}
    report['class_macro_edit_attribution'] = edit_attribution(rows, arrays, corrections)
    return report


def finalize(a):
    started = time.monotonic()
    prep = read(a.out / 'prepared.json')
    selection = read(a.out / 'selection.json')
    if selection['state'] != 'EXACT_BRANCH_UNION_GLOBAL_AND_TRAINFOLD_SELECTED':
        raise ValueError('Selection not frozen')
    rows = read(a.out / 'manifest.json')['rows']
    patterns = np.load(Path(prep['source']) / 'patterns.npy', mmap_mode='r')
    source_arms = read(Path(prep['source']) / 'arms.json')
    all_reports = {}
    for policy, data in prep['policies'].items():
        destination = a.out / policy
        (destination / 'predictions').mkdir()
        hashes = {}
        compact_arms = [{key: arm[key] for key in ('id', 'npz_key')} for arm in data['arms']]
        compact = dict(arms=compact_arms, source_global=data['source_global'], source_held=data['source_held'],
                       canonical_global_npz_key=data['canonical_global_npz_key'])
        jobs = [(row, policy, dict(compact,
                    canonical_global_digest=data['canonical_global_seal']['predictions'][row['key']],
                    canonical_held_digest=data['canonical_held_seal']['predictions'][row['key']]),
                 selection['policies'][policy], str(destination)) for row in rows]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            for key, digest in pool.map(mask_one, jobs, chunksize=4):
                hashes[key] = digest
        write(destination / 'sealed.json', dict(state='ALL_GLOBAL_AND_HELDFOLD_MASKS_SEALED', n=len(rows), predictions=hashes,
              manifest_sha256=sha(a.out / 'manifest.json'), selection_sha256=sha(a.out / 'selection.json'),
              query_GT_read_in_mask_construction=False, earlier_GT_used_for_global_and_trainfold_selection=True,
              policy_scope=prep['scope']))
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            counts = list(pool.map(count_one, [(row, compact_arms, str(destination), hashes[row['key']]) for row in rows], chunksize=4))
        names = list(counts[0])
        arrays = {name: np.array([record[name]['iu'] for record in counts], dtype=np.int64) for name in names}
        corrections = {name: [dict(key=row['key'], c=row['c'], fold=row['fold'], batch=row['batch'],
                             **{key: int(record[name][key]) for key in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')})
                             for row, record in zip(rows, counts)] for name in names}
        truth = patterns[:, :, 1].sum(axis=1, dtype=np.int64)
        for name in names:
            expected = np.zeros((len(rows), 2), dtype=np.int64)
            labels = ['global'] if name.endswith('_global') else [str(f) for f in range(4)]
            branch = name.split('_')[0]
            for label in labels:
                rec = selection['policies'][policy][label][branch]
                on = lut(rec, len(data['arms']))
                # Strict atoms project away extended bit12; equivalent LUT repeats twice.
                if policy == 'strict':
                    on = np.tile(on, 2)
                ix = np.ones(len(rows), dtype=bool) if label == 'global' else np.array([r['fold'] == int(label) for r in rows])
                expected[ix, 0] = patterns[ix][:, on, 1].sum(axis=1, dtype=np.int64)
                expected[ix, 1] = truth[ix] + patterns[ix][:, on, 0].sum(axis=1, dtype=np.int64)
            if not np.array_equal(expected, arrays[name]):
                raise ValueError('Complete actual mask recount differs from membership recipe')
        with np.load(Path(prep['source']) / 'baseline_counts.npz', allow_pickle=False) as z:
            for arm in data['arms']:
                arrays[arm['id']] = z[arm['id']].copy()
        native = next(arm['id'] for arm in data['arms'] if arm['npz_key'] == 'native')
        arrays['native'] = arrays[native]
        for name in names:
            records = corrections[name]
            addtp = np.array([r['add_TP'] for r in records])
            deltp = np.array([r['delete_TP'] for r in records])
            addfp = np.array([r['add_FP'] for r in records])
            delfp = np.array([r['delete_FP'] for r in records])
            if not np.array_equal(arrays[name][:, 0] - arrays['native'][:, 0], addtp - deltp) or not np.array_equal(arrays[name][:, 1] - arrays['native'][:, 1], addfp - delfp):
                raise ValueError('Four-edit exact accounting failed')
        report = summary(rows, arrays, corrections)
        for branch in ('canonical', 'general', 'selected'):
            rec = selection['policies'][policy]['global'][branch]
            if abs(report['scores'][branch + '_global'] - rec['score']) > 1e-10:
                raise ValueError('Global selection score differs from actual masks')
        report.update(policy=policy, recipes=selection['policies'][policy], all_actual_IU_matches_histogram=True,
                      all_original_policy_baseline_IU_exact=True, original_canonical_masks_bitexact=True,
                      exposure='reused DEV; global CIs conditional on selected recipe and not selection-adjusted; held-fold selection uses train3 class folds',
                      parameter_label_boundary=('strict cached bank' if policy == 'strict' else 'extended: sizecut retains allfresh600-fitted thresholds; recipe train3 split does not remove upstream parameter label exposure'),
                      scope=prep['scope'], seal_sha256=sha(destination / 'sealed.json'))
        np.savez_compressed(destination / 'counts.npz', **arrays)
        write(destination / 'report.json', report)
        all_reports[policy] = report
    write(a.out / 'report.json', dict(state='COMPLETE_GENERAL_REPAIR_FAMILY_RECOUNT', n=len(rows), policies=all_reports,
          source_receipt_sha256=sha(a.out / 'prepared.json'), selection_sha256=sha(a.out / 'selection.json'),
          no_new_encoder_or_GPU=True, CPU_finalize_seconds=time.monotonic() - started))
    lines = ['# Canonical union all-general repair family', '',
             'All bank origins are retained. General output = (origin OR positives) AND all negative masks. '
             'This evaluates the union of the complete old canonical family and the complete all-general branch; '
             'it does not claim the larger action-wise mixed-deletion family.', '',
             'Global recipes are GT-selected DEV optima; held-fold outputs use a fixed recipe selected on the other '
             'three folds. Paired intervals use 2,000 RandomState(0) connected-photo draws. Global intervals are '
             'conditional, not selection-adjusted. Extended sizecut upstream label exposure remains.', '']
    for policy, report in all_reports.items():
        lines += ['## ' + policy, '', '| Output | mIoU | vs native [95% CI] | vs canonical same protocol [95% CI] |', '|---|---:|---:|---:|']
        for name in ('canonical_global', 'general_global', 'selected_global', 'canonical_held', 'general_held', 'selected_held'):
            base = 'canonical_global' if name.endswith('_global') else 'canonical_held'
            native_delta = report['contrasts'][name]['native']
            change = report['contrasts'][name].get(base, dict(gain=0., ci95=[0., 0.]))
            lines.append('| %s | %.6f | %+.6f [%+.6f, %+.6f] | %+.6f [%+.6f, %+.6f] |' %
                         (name, report['scores'][name], native_delta['gain'], *native_delta['ci95'], change['gain'], *change['ci95']))
        lines += ['', 'Selected global recipe: `' + json.dumps(report['recipes']['global']['selected']) + '`', '']
    (a.out / 'report.md').write_text('\n'.join(lines) + '\n')
    write(a.out / 'state.json', dict(state='COMPLETE_GENERAL_REPAIR_FAMILY_RECOUNT', n=len(rows),
                                    CPU_finalize_seconds=time.monotonic() - started))
    print('COMPLETE_GENERAL_REPAIR_FAMILY_RECOUNT', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selfcheck', 'prepare', 'optimize', 'finalize', 'all'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--strict-canonical', type=Path)
    parser.add_argument('--strict-held', type=Path)
    parser.add_argument('--extended-held', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers != 4:
        raise ValueError('This owned run is fixed at CPU4, one thread per worker')
    if args.stage == 'selfcheck':
        print(json.dumps(toy_check()), flush=True)
        return
    if any(getattr(args, name) is None for name in ('source', 'strict_canonical', 'strict_held', 'extended_held', 'out')):
        parser.error('All explicit immutable source paths are required')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('optimize', 'all'):
        optimize(args)
    if args.stage in ('finalize', 'all'):
        finalize(args)


if __name__ == '__main__':
    main()
