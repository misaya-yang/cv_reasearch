#!/usr/bin/env python3
"""CPU candidate inference and separately invoked ground-truth scoring."""
import argparse
import concurrent.futures
import importlib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODULES = ('reference_evidence', 'local_structure', 'query_partition',
           'cross_image_matching', 'rgb_complement', 'decision_risk', 'invariance_support',
           'cross_image_matching_batch2', 'rgb_complement_extra')


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def frozen_imports(frozen):
    """Import exactly the saved package, including inside spawned workers."""
    sys.path.insert(0, str(Path(frozen) / 'src'))
    for name in list(sys.modules):
        if name == 'ics' or name.startswith('ics.'):
            del sys.modules[name]
    importlib.invalidate_caches()


def verify_snapshots(run, hashes):
    for relative, expected in hashes.items():
        if file_sha(Path(run) / 'source' / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)


def registry(modules):
    methods, controls = {}, {}
    for name in modules:
        mod = importlib.import_module('ics.cpu100.' + name)
        for target, values in ((methods, mod.METHODS), (controls, mod.CONTROLS)):
            overlap = set(target) & set(values)
            if overlap:
                raise ValueError('Duplicate IDs: ' + repr(overlap))
            target.update(values)
    if set(methods) & set(controls):
        raise ValueError('A control cannot count as a method')
    return methods, controls


def initialize(threads, frozen=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    if frozen is not None:
        frozen_imports(frozen)


def one_episode(row, out, modules, selected, selected_controls):
    import numpy as np
    from ics.cpu100.common import Result, load_episode, prototype_margin, render, sha
    start = time.monotonic()
    ep = load_episode(row)
    methods, controls = registry(modules)
    arms = {k: methods[k] for k in selected}
    arms.update({k: controls[k] for k in selected_controls})
    arms['dino_prototype.control'] = lambda e: Result(prototype_margin(e), {'control': True})
    case = Path(out) / row['id']
    case.mkdir()
    receipts = {}
    for name, function in arms.items():
        tick = time.monotonic()
        try:
            result = function(ep)
            # Every arm, including the shared baseline, uses exactly the same
            # known-padding rule before physical interpolation.
            result.margin = np.asarray(result.margin, float).reshape(ep.q_hw).copy()
            result.margin.ravel()[ep.q_valid <= 0] = -1.0
            rendered = render(ep, result)
            target = case / (name + '.npz')
            np.savez_compressed(target, margin=rendered['margin'],
                                mask_work=np.packbits(rendered['work']),
                                mask_original=np.packbits(rendered['original']),
                                original_shape=np.array(ep.original_shape))
            receipts[name] = dict(state='complete', seconds=time.monotonic() - tick,
                                  predicted_pixels=int(rendered['original'].sum()), info=result.info,
                                  output_sha256=sha(target), query_GT_read=False)
        except Exception as error:
            receipts[name] = dict(state='unavailable' if type(error).__name__ == 'RGBUnavailable' else 'failed',
                                  seconds=time.monotonic() - tick, error=repr(error),
                                  traceback=traceback.format_exc(), query_GT_read=False)
        write(case / 'receipt.json', dict(id=row['id'], arms=receipts, source_sha256=row['sha256'],
              producer=ep.producer, query_GT_read=False, elapsed=time.monotonic() - start))
    return dict(id=row['id'], arms=receipts, seconds=time.monotonic() - start,
                producer=ep.producer, input_sha256=row['sha256'],
                receipt_sha256=sha(case / 'receipt.json'))


def infer(args):
    initialize(args.threads)
    from ics.cpu100.common import sha
    bound = json.loads(args.manifest.read_text())
    if bound['schema'] != 'DINO_ONLY_FEATURE_INPUT_V1':
        raise ValueError('Native DINO input manifest required')
    methods, controls = registry(args.modules)
    selected = list(methods) if args.methods == ['all'] else args.methods
    selected_controls = list(controls) if args.controls == ['all'] else args.controls
    if not set(selected) <= set(methods):
        raise ValueError('Unknown method selection')
    if not set(selected_controls) <= set(controls):
        raise ValueError('Unknown control selection')
    if not bound['rows'] or len({r['id'] for r in bound['rows']}) != len(bound['rows']):
        raise ValueError('Nonempty unique input IDs required')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate selected methods')
    if len(set(selected_controls)) != len(selected_controls):
        raise ValueError('Duplicate selected controls')
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = args.out / 'source'
    snapshots = {}
    sources = list((ROOT / 'src/ics/cpu100').glob('*.py')) + [Path(__file__),
        ROOT / 'src/ics/methods/direct_dino_features.py', ROOT / 'src/ics/experiment.py',
        ROOT / 'src/ics/__init__.py', ROOT / 'src/ics/methods/__init__.py']
    for source in sources:
        relative = source.relative_to(ROOT)
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        snapshots[str(relative)] = sha(target)
    config = dict(modules=args.modules, methods=selected, controls=selected_controls, workers=args.workers,
                  threads=args.threads, manifest_sha256=sha(args.manifest), input_rows=bound['rows'],
                  source_hashes=snapshots, quality_scored=False, query_GT_read=False,
                  protocol='raw native DINO, signed margin renderer, invalid query padding margin=-1 for every arm, no FoRIS scores/masks')
    write(args.out / 'config.json', config)
    write(args.out / 'manifest.json', bound)
    expected_arms = selected + selected_controls + ['dino_prototype.control']
    receipts = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(args.threads, str(frozen))) as pool:
        futures = {pool.submit(one_episode, r, str(args.out), args.modules, selected,
                               selected_controls): r['id'] for r in bound['rows']}
        for future in concurrent.futures.as_completed(futures):
            try:
                receipt = future.result()
            except Exception as error:
                receipt = dict(id=futures[future], state='failed', error=repr(error))
            receipts.append(receipt)
            write(args.out / 'status.json', dict(state='running', total=len(bound['rows']),
                  finished=len(receipts), elapsed=time.time() - started))
            print(json.dumps({'id': receipt['id'], 'finished': len(receipts), 'total': len(bound['rows'])}), flush=True)
    verify_snapshots(args.out, snapshots)
    if sha(args.manifest) != config['manifest_sha256']:
        raise ValueError('Input manifest changed during inference')
    outcomes = {arm: {'complete': 0, 'failed': 0, 'unavailable': 0, 'missing': 0} for arm in expected_arms}
    for receipt in receipts:
        for arm in expected_arms:
            state = receipt.get('arms', {}).get(arm, {}).get('state', 'missing')
            outcomes[arm][state if state in outcomes[arm] else 'failed'] += 1
    all_complete = all(v['complete'] == len(bound['rows']) for v in outcomes.values())
    write(args.out / 'sealed.json', dict(state='sealed', receipts=receipts, elapsed=time.time() - started,
          methods=selected, manifest_sha256=sha(args.manifest), source_hashes=snapshots,
          config_sha256=sha(args.out / 'config.json'), bound_manifest_sha256=sha(args.out / 'manifest.json'),
          expected_arms=expected_arms, arm_outcomes=outcomes, all_arms_complete=all_complete,
          execution_source='frozen saved package in spawned workers',
          query_GT_read=False, quality_scored=False))
    write(args.out / 'status.json', dict(state='sealed', finished=len(receipts), total=len(bound['rows']),
          all_arms_complete=all_complete, arm_outcomes=outcomes, elapsed=time.time() - started))


def score(args):
    run = args.out
    sealed = json.loads((run / 'sealed.json').read_text())
    if file_sha(run / 'config.json') != sealed['config_sha256'] or file_sha(run / 'manifest.json') != sealed['bound_manifest_sha256']:
        raise ValueError('Sealed configuration/input manifest changed')
    verify_snapshots(run, sealed['source_hashes'])
    frozen_imports(run / 'source')
    import numpy as np
    from PIL import Image
    from ics.cpu100.common import sha
    from ics.experiment import metric, photo_groups
    config = json.loads((run / 'config.json').read_text())
    evaluation = json.loads(args.evaluation_rows.read_text())
    rows_by_id = {r['key']: r for r in evaluation}
    if len(rows_by_id) != len(evaluation):
        raise ValueError('Duplicate scoring episode IDs')
    ids = [r['id'] for r in config['input_rows']]
    rows = [rows_by_id[i] for i in ids]
    sealed_receipts = {r['id']: r for r in sealed['receipts']}
    for input_row in config['input_rows']:
        identity = input_row['id']
        receipt_path = run / identity / 'receipt.json'
        if not receipt_path.exists() or file_sha(receipt_path) != sealed_receipts[identity].get('receipt_sha256'):
            raise ValueError('Missing or changed sealed episode receipt: ' + identity)
        rec = json.loads(receipt_path.read_text())
        if rec['source_sha256'] != input_row['sha256'] or rec['producer'] != sealed_receipts[identity]['producer']:
            raise ValueError('Input/producer binding mismatch: ' + identity)
    common_arms = set.intersection(*[
        {a for a, rec in json.loads((run / i / 'receipt.json').read_text())['arms'].items()
         if rec['state'] == 'complete'} for i in ids])
    base = 'dino_prototype.control'
    if base not in common_arms:
        raise ValueError('Complete common control missing')
    excluded = {arm: sealed['arm_outcomes'][arm] for arm in sealed['expected_arms'] if arm not in common_arms}
    if excluded and not args.allow_partial:
        raise ValueError('Incomplete requested methods/controls; explicit --allow-partial required: ' + repr(excluded))
    arrays = {a: np.zeros((len(rows), 2)) for a in sorted(common_arms)}
    corrections = {a: [] for a in common_arms}
    for index, row in enumerate(rows):
        receipt = json.loads((run / row['key'] / 'receipt.json').read_text())['arms']
        with Image.open(args.data / 'annotations' / (row['query'][:-4] + '.png')) as image:
            gt = np.asarray(image) == row['c'] + 1
        masks = {}
        for arm in common_arms:
            path = run / row['key'] / (arm + '.npz')
            if sha(path) != receipt[arm]['output_sha256']:
                raise ValueError('Sealed prediction changed')
            with np.load(path, allow_pickle=False) as z:
                shape = tuple(z['original_shape'])
                masks[arm] = np.unpackbits(z['mask_original'], count=int(np.prod(shape))).reshape(shape).astype(bool)
            if masks[arm].shape != gt.shape:
                raise ValueError('Original-size complete prediction required')
            arrays[arm][index] = [(masks[arm] & gt).sum(), (masks[arm] | gt).sum()]
        for arm, mask in masks.items():
            added, deleted = mask & ~masks[base], ~mask & masks[base]
            corrections[arm].append(dict(c=row['c'], add_TP=int((added & gt).sum()),
                add_FP=int((added & ~gt).sum()), delete_TP=int((deleted & gt).sum()), delete_FP=int((deleted & ~gt).sum())))
            edit = corrections[arm][-1]
            if (arrays[arm][index, 0] - arrays[base][index, 0] != edit['add_TP'] - edit['delete_TP']
                    or arrays[arm][index, 1] - arrays[base][index, 1] != edit['add_FP'] - edit['delete_FP']):
                raise ValueError('Per-episode edit/count closure failed')
    classes = np.array([r['c'] for r in rows])
    groups = photo_groups(rows)
    draws = np.random.RandomState(0).randint(groups.max() + 1, size=(2000, groups.max() + 1))
    weights = np.stack([np.bincount(d, minlength=groups.max() + 1) for d in draws])[:, groups]
    scores = {a: metric(v, classes) for a, v in arrays.items()}
    samples = {a: np.array([metric(v, classes, w) for w in weights]) for a, v in arrays.items()}
    class_actions = {}
    for arm in arrays:
        class_actions[arm] = {}
        for cls in sorted(set(classes)):
            mask = classes == cls
            initial_i, initial_u = np.sum(arrays[base][mask], axis=0)
            final_i, final_u = np.sum(arrays[arm][mask], axis=0)
            edits = {key: int(sum(r[key] for r in corrections[arm] if r['c'] == cls))
                     for key in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')}
            delta = 100 * (final_i / max(final_u, 1) - initial_i / max(initial_u, 1))
            initial_j = initial_i / max(initial_u, 1)
            formula = 100 * ((edits['add_TP'] - edits['delete_TP']) - initial_j * (edits['add_FP'] - edits['delete_FP'])) / final_u if initial_u > 0 and final_u > 0 else None
            if formula is not None and abs(formula - delta) > 1e-9:
                raise ValueError('Per-class exact IoU edit formula failed')
            class_actions[arm][str(int(cls))] = dict(**edits, baseline_I=int(initial_i), baseline_U=int(initial_u),
                final_I=int(final_i), final_U=int(final_u), gain=float(delta), exact_formula_gain=formula)
    report = dict(n=len(rows), classes=len(set(classes)), scores=scores,
        gain_vs_dino_prototype={a: dict(gain=scores[a] - scores[base],
             ci95=np.percentile(samples[a] - samples[base], [2.5, 97.5]).tolist()) for a in arrays},
        actions_vs_dino_prototype=corrections, sealed_sha256=sha(run / 'sealed.json'),
        class_actions_vs_dino_prototype=class_actions,
        gain_vs_controls={a: {c: dict(gain=scores[a] - scores[c],
            ci95=np.percentile(samples[a] - samples[c], [2.5, 97.5]).tolist())
            for c in config['controls'] if c in arrays} for a in config['methods'] if a in arrays},
        requested_method_count=len(config['methods']), scored_method_count=len(set(config['methods']) & common_arms),
        all_requested_arms_complete=not excluded, explicitly_excluded_incomplete_arms=excluded,
        config_sha256=sha(run / 'config.json'), input_manifest_sha256=sealed['bound_manifest_sha256'],
        method_source_hashes=sealed['source_hashes'], score_runner_sha256=file_sha(__file__),
        evaluation_rows_sha256=sha(args.evaluation_rows),
        exposure='reused development; not independent confirmation', original_resolution=True,
        quality_scope='descriptive activity probe' if len(rows) <= 4 else 'development measurement',
        producer_resolution=sorted({r['producer']['model_input_side'] for r in sealed['receipts'] if 'producer' in r}),
        checkpoint_hashes=sorted({r['producer'].get('model_assets',r['producer'])['checkpoint_sha256']
            for r in sealed['receipts'] if 'producer' in r}),
        bootstrap=dict(draws=2000, unit='connected support/query photographs', rng='RandomState(0)'))
    score_dir = run / 'score'
    score_dir.mkdir(exist_ok=False)
    write(score_dir / 'report.json', report)
    np.savez_compressed(score_dir / 'counts.npz', **arrays, classes=classes)
    print(json.dumps({'n': len(rows), 'scores': scores}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for stage in ('list', 'infer'):
        a = sub.add_parser(stage)
        a.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
        if stage == 'infer':
            a.add_argument('--manifest', type=Path, required=True)
            a.add_argument('--out', type=Path, required=True)
            a.add_argument('--methods', nargs='+', default=['all'])
            a.add_argument('--controls', nargs='+', default=['all'])
            a.add_argument('--workers', type=int, default=3)
            a.add_argument('--threads', type=int, default=2)
    a = sub.add_parser('score')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--evaluation-rows', type=Path, required=True)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--allow-partial', action='store_true', help='Explicitly score only arms complete on every original input episode')
    args = p.parse_args()
    if args.stage == 'list':
        methods, controls = registry(args.modules)
        print(json.dumps({'methods': list(methods), 'controls': list(controls)}, indent=2))
    elif args.stage == 'infer':
        if min(args.workers, args.threads) < 1 or args.workers * args.threads > 28:
            raise ValueError('CPU resource bound exceeded')
        infer(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
