#!/usr/bin/env python3
"""Fixed100 saved-field-only tail rank and centered equal-L1 controls.

prepare/probe do not produce candidate masks. infer has no image/model/encoder
path. Only score reads query labels, after all100 fields and masks are sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from raw_feature_cache import file_hash
from lvis_atomic_study import write, point

ASSETS = REPO.parent / 'cv_data'
SOURCE = ASSETS / 'a/lvis_reference_erasure100_20261009'
PILOT = ASSETS / 'a/lvis_atomic1400_20261009/pilot100'
DEFAULT_ROOT = ASSETS / 'a/lvis_tail_rank100_20261009'
BASELINES = ['foris.crf', 'mean', 'mean.graph_only', 'foris.fg_anchor.crf']
NEW_ARMS = {'mean.tail_rank': 'candidate_unary',
            'mean.no_tail_reference': 'no_reference_unary',
            'mean.tail_global_amplitude': 'amplitude_unary'}
ARMS = BASELINES + list(NEW_ARMS)


def index(path):
    return {r['episode_id']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def verify_seal(root):
    seal = json.loads((root / 'sealed.json').read_text())
    if seal['n'] != 100 or seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('Require completed sealed100 producer')
    for name, key in [('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('inference.jsonl', 'inference_index_sha256')]:
        if file_hash(root / name) != seal[key]:
            raise ValueError('Source producer seal changed')
    return seal


def source_paths():
    return [Path(__file__).resolve(), REPO / 'src/ics/methods/tail_rank_control.py',
            REPO / 'src/ics/methods/rcg.py', REPO / 'src/ics/metrics.py',
            REPO / 'scripts/run_m4_baselines.py', REPO / 'scripts/lvis_atomic_study.py',
            REPO / 'scripts/raw_feature_cache.py']


def prepare(root):
    verify_seal(SOURCE); verify_seal(PILOT)
    rows = json.loads((SOURCE / 'manifest.json').read_text())
    if rows != json.loads((PILOT / 'manifest.json').read_text()) or len(rows) != 100:
        raise ValueError('Use the same already frozen100')
    fields, predictions = index(SOURCE / 'inference.jsonl'), index(PILOT / 'inference.jsonl')
    tasks = []
    for row in rows:
        eid = row['episode_id']; field = fields[eid]; pred = predictions[eid]
        fp, pp = SOURCE / 'fields' / field['filename'], PILOT / 'predictions' / pred['filename']
        if file_hash(fp) != field['fields_sha256'] or file_hash(pp) != pred['prediction_sha256']:
            raise ValueError('Input field/prediction identity differs')
        tasks.append(dict(episode_id=eid, filename=field['filename'], source_field=str(fp),
                          source_field_sha256=field['fields_sha256'], baseline_prediction=str(pp),
                          baseline_prediction_sha256=pred['prediction_sha256']))
    sources = {str(p): file_hash(p) for p in source_paths()}
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists():
        old = json.loads((root / 'config.json').read_text())
        if old['source_sha256'] == sources:
            validate(root); return old
        if (root / 'inference.jsonl').exists() or (root / 'sealed.json').exists():
            raise ValueError('Do not modify a launched experiment source')
    write(root / 'manifest.json', rows); write(root / 'tasks.json', tasks)
    config = dict(state='PREPARED_ONLY', n=100, arms=ARMS, source_sha256=sources,
                  producer_seal_sha256={str(p): file_hash(p / 'sealed.json') for p in (SOURCE, PILOT)},
                  manifest_sha256=file_hash(root / 'manifest.json'), tasks_sha256=file_hash(root / 'tasks.json'),
                  group='original token s>.5; distinct from label-diagnostic postCRF ROI',
                  candidate='G: s+.25*(locked_rank(guide[G])-locked_rank(s[G])); outside: exact parent y',
                  no_tail_reference='G: s; outside: exact parent y',
                  amplitude_control='G: d=parent_y-s; dc=d-mean(d); kappa=sum(abs(actual local_y-s))/sum(abs(dc)); y=s+kappa*dc FP64; outside exact parent y',
                  group_fallback='nG<2: all three arms reuse parent y; centered ratio is not applied',
                  centered_zero_denominator='for nG>=2 assert local budget zero and reuse s on G',
                  no_clipping=True, no_parameter_search=True,
                  parent_fields='saved source_s,parent_guide,parent_unary,parent_a,parent_H_CSR,parent_mean',
                  readout='same actual H/A; CG x0=y rtol1e-7 atol1e-9 max300;lambda16',
                  render='original bilinear1024 align_corners=False >.5; original from1024 bool',
                  extra_CRF=False, encoder_calls=0, query_GT_in_inference=False,
                  exposure='same already exposed100; development only',
                  claim='classic localized CDF regrouping, not new reference information or established gain',
                  scope='fixed100 only; no next batch, other dataset or encoder path')
    write(root / 'config.json', config)
    snapshots = root / 'source'; snapshots.mkdir(exist_ok=True)
    for p in source_paths():
        target = snapshots / p.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    for name, producer in [('erasure', SOURCE), ('old_pilot', PILOT)]:
        folder = root / 'producer_snapshots' / name; folder.mkdir(parents=True, exist_ok=True)
        for filename in ('config.json', 'manifest.json', 'sealed.json', 'inference.jsonl'):
            shutil.copy2(producer / filename, folder / filename)
        if (producer / 'source').exists():
            shutil.copytree(producer / 'source', folder / 'source', dirs_exist_ok=True)
    write(root / 'activity.json', dict(state='PREPARED_ONLY', n=100, completed=0, encoder_calls=0))
    return config


def validate(root):
    config = json.loads((root / 'config.json').read_text())
    for path, digest in config['source_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Frozen source changed: ' + path)
    for path, digest in config['producer_seal_sha256'].items():
        if file_hash(Path(path) / 'sealed.json') != digest: raise ValueError('Producer seal changed')
        verify_seal(Path(path))
    for name in ('manifest', 'tasks'):
        if file_hash(root / (name + '.json')) != config[name + '_sha256']: raise ValueError('Cohort/input identity changed')
    return config


def read_parent(task):
    import numpy as np
    from scipy import sparse
    if file_hash(task['source_field']) != task['source_field_sha256'] or file_hash(task['baseline_prediction']) != task['baseline_prediction_sha256']:
        raise ValueError('Input arrays changed')
    with np.load(task['source_field']) as z:
        s, guide, unary, a, field = (z[k].copy().reshape(-1) for k in
                                    ('source_s', 'parent_guide', 'parent_unary', 'parent_a', 'parent_mean'))
        h = sparse.csr_matrix((z['parent_H_data'].copy(), z['parent_H_indices'].copy(), z['parent_H_indptr'].copy()),
                              shape=tuple(z['parent_H_shape']))
    if s.size != 4096 or h.shape != (4096, 4096) or field.dtype != np.float32:
        raise ValueError('Expected the actual parent64-square system')
    return s, guide, unary, a, h, field


def parent_replay(task):
    import numpy as np
    from ics.methods.tail_rank_control import solve_parent
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    s, guide, unary, a, h, field = read_parent(task)
    got, solver = solve_parent(unary, a, h)
    if not np.array_equal(got, field): raise ValueError('Saved H/unary does not replay original MEAN field bit exactly')
    mask = mask_from_field(got.reshape(64, 64))
    with np.load(task['baseline_prediction']) as saved:
        old = np.unpackbits(saved['cli/mean'], count=1024**2).reshape(1024, 1024).astype(bool)
        shape = tuple(saved['original_hw'])
        original = np.unpackbits(saved['original/mean'], count=int(np.prod(shape))).reshape(shape).astype(bool)
    if not np.array_equal(mask, old) or not np.array_equal(render(mask, shape), original):
        raise ValueError('Original MEAN mask replay differs')
    return s, guide, unary, a, h, field, solver


def probe(root):
    import numpy as np
    import torch
    from scipy import sparse
    from ics.methods.tail_rank_control import build_unaries, solve_parent
    from ics.methods.rcg import rank
    torch.set_num_threads(2); validate(root)
    task = json.loads((root / 'tasks.json').read_text())[0]
    s, guide, unary, a, h, field, solver = parent_replay(task)
    actual = build_unaries(s, guide, unary)
    def parent(s, g): return (s + .25 * (rank(g) - rank(s))).astype(np.float64)
    # Full G: candidate and field reproduce the original parent operation.
    ss = np.array([.6, .7, .8, .9], np.float32); gg = np.array([.1, .9, .4, .2], np.float32)
    yy = parent(ss, gg); full = build_unaries(ss, gg, yy)
    assert np.array_equal(full['candidate_unary'], yy)
    aa = np.ones(4, np.float64); hh = sparse.eye(4, dtype=np.float64).tocsr()
    assert np.array_equal(solve_parent(full['candidate_unary'], aa, hh)[0], solve_parent(yy, aa, hh)[0])
    # Fixed constant guide has rank .5 within G, but is not graph-only.
    ss = np.array([.1, .6, .8, .2], np.float32); gg = np.ones(4, np.float32)
    constant = build_unaries(ss, gg, parent(ss, gg))
    assert np.array_equal(constant['rank_guide_group'], np.full(2, .5, np.float32))
    assert not np.array_equal(constant['candidate_unary'], constant['no_reference_unary'])
    assert np.array_equal(constant['candidate_unary'][~constant['group']], parent(ss, gg)[~constant['group']])
    # Both nG=0 and nG=1 explicitly bypass the centered ratio as well.
    for ss in (np.array([.1, .2, .3, .4], np.float32), np.array([.1, .8, .3, .4], np.float32)):
        yy = parent(ss, np.arange(4, dtype=np.float32)); small = build_unaries(ss, np.arange(4, dtype=np.float32), yy)
        assert all(np.array_equal(small[k], yy) for k in ('candidate_unary', 'no_reference_unary', 'amplitude_unary'))
    ss = np.array([.1, .6, .8, .2], np.float32)
    zero = build_unaries(ss, ss.copy(), parent(ss, ss))
    assert zero['diagnostics']['centered_global_group_l1'] == 0
    assert zero['diagnostics']['candidate_group_l1'] == 0
    assert np.array_equal(zero['amplitude_unary'][zero['group']], ss[zero['group']].astype(np.float64))
    folder = root.with_name(root.name + '_preparation'); folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(state='PREPARATION_CHECKS_PASS_NO_INFERENCE', episode_id=task['episode_id'], n_fixed=100,
        config_sha256=file_hash(root / 'config.json'), checks=dict(parent_field_bit_exact=True,
            parent_CLI_and_original_masks_bit_exact=True, full_G_candidate_and_field_bit_exact=True,
            outside_G_parent_unary_bit_exact=True, small_G_all_arms_parent_fallback_bit_exact=True,
            constant_G_guide_rank_half=True, constant_G_guide_not_graph_only=True,
            zero_centered_budget_explicit_fallback=True, actual_equal_L1_budget_closed=True),
        actual_first_case=actual['diagnostics'], original_solver=solver,
        encoder_calls=0, new_candidate_masks=0, new_miou=0, query_GT_opened=False)
    write(folder / 'receipt.json', receipt); print(json.dumps(dict(state=receipt['state'], receipt=str(folder / 'receipt.json'))), flush=True)


def infer(root):
    import numpy as np
    import torch
    from ics.methods.tail_rank_control import build_unaries, solve_parent
    from ics.methods.rcg import mask_from_field
    from run_m4_baselines import render
    torch.set_num_threads(2); validate(root)
    if (root / 'sealed.json').exists(): print('Fixed100 already sealed; no restart'); return
    tasks = json.loads((root / 'tasks.json').read_text()); done = {}
    for folder in ('fields', 'predictions'): (root / folder).mkdir(exist_ok=True)
    if (root / 'inference.jsonl').exists():
        done = index(root / 'inference.jsonl')
        for x in done.values():
            if file_hash(root / 'fields' / x['filename']) != x['fields_sha256'] or file_hash(root / 'predictions' / x['filename']) != x['prediction_sha256']:
                raise ValueError('Resume artifacts changed')
    started = time.monotonic()
    with (root / 'inference.jsonl').open('a', buffering=1) as ledger:
        for task in tasks:
            if task['episode_id'] in done: continue
            began = time.monotonic(); s, guide, unary, a, h, old, original_solver = parent_replay(task)
            result = build_unaries(s, guide, unary)
            fields = dict(group=result['group'].reshape(64, 64), rank_s_group=result['rank_s_group'], rank_guide_group=result['rank_guide_group'])
            solvers = {}; packed = {}
            with np.load(task['baseline_prediction']) as saved:
                shape = tuple(saved['original_hw']); packed['original_hw'] = saved['original_hw'].copy()
                for arm in BASELINES:
                    for frame in ('cli', 'original'): packed[frame + '/' + arm] = saved[frame + '/' + arm].copy()
            for arm, key in NEW_ARMS.items():
                z, info = solve_parent(result[key], a, h); fields[key] = result[key].reshape(64, 64)
                fields[arm] = z.reshape(64, 64); solvers[arm] = info
                mask = mask_from_field(z.reshape(64, 64)); packed['cli/' + arm] = np.packbits(mask)
                packed['original/' + arm] = np.packbits(render(mask, shape))
            name = task['filename']; np.savez_compressed(root / 'fields' / name, **fields)
            np.savez_compressed(root / 'predictions' / name, **packed)
            record = dict(episode_id=task['episode_id'], filename=name, fields_sha256=file_hash(root / 'fields' / name),
                prediction_sha256=file_hash(root / 'predictions' / name), source_field_sha256=task['source_field_sha256'],
                parent_field_and_masks_bit_exact=True, diagnostics=result['diagnostics'], solvers=solvers,
                parent_solver=original_solver, seconds=time.monotonic() - began, encoder_calls=0, query_GT_in_inference=False)
            ledger.write(json.dumps(record) + '\n'); done[task['episode_id']] = record
            write(root / 'activity.json', dict(state='INFERENCE', n=100, completed=len(done), encoder_calls=0, elapsed_seconds=time.monotonic()-started))
    assert len(done) == 100; validate(root)
    write(root / 'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=100, config_sha256=file_hash(root / 'config.json'),
        manifest_sha256=file_hash(root / 'manifest.json'), inference_index_sha256=file_hash(root / 'inference.jsonl'),
        tasks_sha256=file_hash(root / 'tasks.json'), encoder_calls=0, query_GT_in_inference=False,
        all_parent_field_and_mask_replays_bit_exact=True))
    write(root / 'activity.json', dict(state='INFERENCE_COMPLETE', n=100, completed=100, encoder_calls=0))


def score(root):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts, gross_edits
    torch.set_num_threads(2); config = validate(root); verify_seal(root)
    rows = json.loads((root / 'manifest.json').read_text()); ii = index(root / 'inference.jsonl')
    prior = index(PILOT / 'episode_metrics.jsonl'); records = []
    for row in rows:
        rec = ii[row['episode_id']]; path = root / 'predictions' / rec['filename']
        if file_hash(path) != rec['prediction_sha256'] or file_hash(root / 'fields' / rec['filename']) != rec['fields_sha256']: raise ValueError('Scoring artifacts changed')
        with Image.open(row['query_mask_path']) as im: raw = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        if array_hash(raw) != row['query_mask_hash']: raise ValueError('Frozen query labels changed')
        item = dict(episode_id=row['episode_id'], fold=row['fold'], class_id=row['loader_class_id'], query_photo_id=row['query_photo_id'], frames={})
        with np.load(path) as saved:
            for frame, shape in [('cli', (1024, 1024)), ('original', tuple(row['query_size_hw']))]:
                truth = f.interpolate(torch.from_numpy(raw)[None, None].float(), shape, mode='nearest')[0, 0].numpy() > .5
                masks = {arm: np.unpackbits(saved[frame+'/'+arm], count=int(np.prod(shape))).reshape(shape).astype(bool) for arm in ARMS}
                iu = {arm: counts(mask, truth) for arm, mask in masks.items()}
                for arm in BASELINES:
                    if iu[arm] != prior[row['episode_id']]['frames'][frame]['iu'][arm]: raise ValueError('Reused baseline I/U differs')
                edits = {arm: {base: gross_edits(mask, masks[base], truth) for base in ARMS} for arm, mask in masks.items()}
                item['frames'][frame] = dict(iu=iu, edits=edits, truth_pixels=int(truth.sum()))
        records.append(item)
    (root/'episode_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report = dict(state='COMPLETE', n=100, frames={}, encoder_calls=0, exposure=config['exposure'],
                  group_is_GT_diagnostic_ROI=False, inference_sealed_before_candidate_labels=True,
                  generalization_or_new_information_claim=False)
    for frame in ('cli', 'original'):
        flat = [dict(r, **r['frames'][frame]) for r in records]; points = point(flat, ARMS); pairs = {}
        for arm in NEW_ARMS:
            for base in ARMS:
                deltas = [100*(r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][base][0]/max(r['iu'][base][1],1)) for r in flat]
                pairs[arm+' vs '+base] = dict(delta_pp=points[arm]-points[base], cases_up=sum(d>1e-10 for d in deltas),
                    cases_down=sum(d< -1e-10 for d in deltas), cases_equal=sum(abs(d)<=1e-10 for d in deltas),
                    edits=np.sum([r['edits'][arm][base] for r in flat],axis=0).tolist(), edit_order=['add_TP','add_FP','delete_TP','delete_FP'])
        report['frames'][frame] = dict(miou=points,pairs=pairs)
    write(root/'report.json',report); write(root/'activity.json',dict(state='COMPLETE',n=100,completed=100,encoder_calls=0))
    print(json.dumps({frame: report['frames'][frame]['miou'] for frame in ('cli','original')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','probe','infer','score','run'])
    parser.add_argument('--out',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if args.probe and args.mode!='prepare': parser.error('--probe is only for prepare')
    if args.mode=='prepare':
        prepare(args.out)
        if args.probe: probe(args.out)
    elif args.mode=='probe': probe(args.out)
    elif args.mode=='infer': infer(args.out)
    elif args.mode=='score': score(args.out)
    else: infer(args.out); score(args.out)


if __name__=='__main__': main()
