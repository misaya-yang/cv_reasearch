"""Paired cached scene-reconstruction experiment, with prediction-first scoring.

The input interface is one reference image, its mask, and one query image.
Query annotations and archived predictions are used only after every new
prediction has been sealed. DeepGlobe's historical custom panel is exploratory.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'cv_data'
DEFAULT = DATA / 'a/autonomous_scene_reconstruction600_20261010'
PROFILE = DATA / 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'
ARMS = ('scene', 'source_kernel', 'support_ridge')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def imports(code_root=REPO):
    sys.path[:0] = [str(code_root / 'src'), str(code_root / 'scripts'), str(DATA / 'third_party/foris_official')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import load_inputs, array_hash
    from raw_feature_cache import RawFeatureCache
    from utils.data import build_transform
    return np, torch, F, load_inputs, array_hash, RawFeatureCache, build_transform


def baseline(root, record, arm):
    filename = record.get('prediction_file', record.get('filename'))
    path = root / 'predictions' / filename
    digest = sha(path)
    assert digest == record['prediction_sha256']
    np, *_ = imports()
    with np.load(path, allow_pickle=False) as z:
        cli = ('cli1024/' if 'cli1024/' + arm in z else 'cli/') + arm
        assert cli in z and 'original/' + arm in z
    return dict(path=str(path), sha256=digest,
                keys={'cli1024': cli, 'original': 'original/' + arm})


def prepare(out, workers):
    assert not (out / 'config.json').exists(), 'Refuse to replace a prepared experiment'
    np, torch, F, load_inputs, _, RawFeatureCache, build_transform = imports()
    cache = RawFeatureCache(PROFILE.parent.parent, read(PROFILE))
    transform = build_transform(1024)
    panels = []
    pilot = DATA / 'a/joint_role_pilot200_20261010'
    precs = {r['episode_id']: r for r in rows(pilot / 'inference.jsonl')}
    for row in read(pilot / 'manifest.json'):
        panels.append((row, {a: baseline(pilot, precs[row['episode_id']], a)
                             for a in ('foris.crf', 'mean')}))
    coco = DATA / 'a/coco_role_competition200_20261010'
    crecs = {r['episode_id']: r for r in rows(coco / 'inference.jsonl')}
    for row in read(coco / 'manifest.json'):
        panels.append((row, {a: baseline(coco, crecs[row['episode_id']], a)
                             for a in ('foris.crf', 'mean')}))
    lvis = DATA / 'a/lvis_mean200_20261008/run'
    foris = DATA / 'a/lvis_foris1400_score_reuse_20261009/run'
    mrecs = {r['episode_id']: r for r in rows(lvis / 'inference.jsonl')}
    frecs = {r['episode_id']: r for r in rows(foris / 'inference.jsonl')}
    for row in read(lvis / 'manifest.json'):
        panels.append((row, {'foris.crf': baseline(foris, frecs[row['episode_id']], 'foris.crf'),
                             'mean': baseline(lvis, mrecs[row['episode_id']], 'mean')}))
    assert len(panels) == 600 and len({r['episode_id'] for r, _ in panels}) == 600
    sources, tasks = {}, []
    for row, archived in panels:
        reference, reference_mask, query = load_inputs(row, DATA)
        requests = []
        for role, image in (('reference', reference), ('query', query)):
            inputs = transform(image).numpy()
            key = cache.key(inputs)
            entry = cache.folder / key / 'entry.json'
            info = read(entry)
            assert info['profile_id'] == cache.profile_id
            assert info['features']['O/24']['shape'] == [4096, 1024]
            requests.append(dict(role=role, key=key, entry_sha256=sha(entry),
                                 input_sha256=info['input_tensor_hash'],
                                 payload_sha256=info['file_sha256'],
                                 tensor_sha256=info['features']['O/24']['tensor_sha256']))
        tasks.append(dict(episode_id=row['episode_id'], raw=requests, baselines=archived))
    source_files = ['scripts/autonomous_scene_reconstruction.py', 'src/ics/__init__.py',
                    'src/ics/methods/__init__.py',
                    'src/ics/methods/autonomous_context_transport.py',
                    'src/ics/official_data.py', 'src/ics/metrics.py',
                    'scripts/raw_feature_cache.py', 'scripts/cached_dino.py',
                    'src/ics/native_basis.py', 'src/ics/m4_crf.py', 'src/ics/m4_crf_cached.py']
    out.mkdir(parents=True, exist_ok=True)
    for relative in source_files:
        path = REPO / relative
        dest = out / 'frozen' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        sources[relative] = sha(path)
    manifest = [r for r, _ in panels]
    write(out / 'manifest.json', manifest)
    write(out / 'tasks.json', tasks)
    write(out / 'config.json', dict(
        n=600, workers=workers, threads_per_worker=2,
        manifest_sha256=sha(out / 'manifest.json'), tasks_sha256=sha(out / 'tasks.json'),
        profile_path=str(PROFILE), profile_sha256=sha(PROFILE), source_sha256=sources,
        basis_path=str(DATA / 'native_assets/positional_basis.pt'),
        basis_sha256=sha(DATA / 'native_assets/positional_basis.pt'),
        external_source_sha256={str(DATA / 'third_party/foris_official' / relative):
                                sha(DATA / 'third_party/foris_official' / relative)
                                for relative in ('models/foris.py', 'utils/data.py', 'utils/refinement.py')},
        contrast=list(ARMS), new_encoder_calls=0,
        input_representation='whole FP32 O24 L2; source input-only semantic APD branch shared by all arms',
        reference_coverage='nearest resize legal mask to1024; exact16x16 patch area fractions',
        readout='signed64 field bilinear1024 strict>0; direct and exact official FoRIS boundary CRF, followed by binary original-size renderer',
        primary_metric='released FoRIS CLI1024 fold/class pooled I/U; also separate original frame',
        observed_class_metric=True,
        exposure='Historical exposed development panels: Deep custom100, PACO100, COCO twofold200, LVIS first200. No independent confirmation claim.',
        query_GT_in_inference=False,
        timing_scope='verified existing raw-cache IO plus new head/readout/CRF; excludes original encoder cost',
        coco_expected_classes=read(coco / 'config.json')['expected_classes'],
        paco_expected_classes=read(pilot / 'config.json')['paco_expected_classes']))
    for folder in ('predictions', 'fields', 'records'):
        (out / folder).mkdir(exist_ok=True)
    print(json.dumps(dict(state='PREPARED', n=600, root=str(out))), flush=True)


def initialize(root):
    global OUT, CFG, CACHE, TRANSFORM, PROJECTION, HOST
    global np, torch, F, load_inputs, array_hash, fit_predict
    OUT = Path(root)
    CFG = read(OUT / 'config.json')
    sys.path[:0] = [str(OUT / 'frozen/src'), str(OUT / 'frozen/scripts')]
    np, torch, F, load_inputs, array_hash, RawFeatureCache, build_transform = imports(OUT / 'frozen')
    # Explicit frozen module loading prevents a later live edit changing workers.
    import importlib.util
    module_path = OUT / 'frozen/src/ics/methods/autonomous_context_transport.py'
    spec = importlib.util.spec_from_file_location('frozen_autonomous_scene', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fit_predict = module.fit_predict
    torch.set_num_threads(CFG['threads_per_worker'])
    torch.manual_seed(0)
    CACHE = RawFeatureCache(PROFILE.parent.parent, read(PROFILE))
    TRANSFORM = build_transform(1024)
    basis = torch.load(CFG['basis_path'], map_location='cpu', weights_only=True)['basis'].float()
    PROJECTION = torch.eye(1024) - basis @ basis.T
    from cached_dino import CachedDINO, cache_host
    HOST = cache_host(DATA, CachedDINO(CACHE), mask_refiner='crf')


def infer(task):
    i, row, source = task
    assert sha(__file__) == CFG['source_sha256']['scripts/autonomous_scene_reconstruction.py']
    recfile = OUT / 'records' / f'{i:06d}.json'
    if recfile.exists():
        record = read(recfile)
        assert record['episode_id'] == row['episode_id']
        assert sha(OUT / 'predictions' / record['filename']) == record['prediction_sha256']
        assert sha(OUT / 'fields' / record['filename']) == record['fields_sha256']
        return record
    begun = time.monotonic()
    reference, reference_mask, query = load_inputs(row, DATA)
    arrays = []
    for image, request in zip((reference, query), source['raw']):
        inputs = TRANSFORM(image).numpy()
        assert CACHE.key(inputs) == request['key']
        assert sha(CACHE.folder / request['key'] / 'entry.json') == request['entry_sha256']
        raw = CACHE.read(inputs, ('O/24',))['O/24']
        arrays.append(torch.from_numpy(raw))
    loading_seconds = time.monotonic() - begun
    tick = time.monotonic()
    r, q = (F.normalize(x, dim=1) for x in arrays)
    binary = F.interpolate(reference_mask.float()[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
    mu = q.mean(0)
    semantic = float(F.normalize(r[binary].mean(0), dim=0) @ mu / (mu.norm() + 1e-6)) if binary.any() else None
    apd = semantic is None or semantic < .8
    if apd:
        r, q = (F.normalize(x @ PROJECTION.T, dim=1) for x in (r, q))
    mask1024 = F.interpolate(reference_mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
    coverage = mask1024.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1).numpy()
    result = fit_predict(r, coverage, q, query_grid_hw=(64, 64))
    assert set(result['fields']) == set(ARMS)
    head_seconds = time.monotonic() - tick
    tick = time.monotonic()
    image = TRANSFORM(query)[None]
    shape = tuple(row['query_size_hw'])
    predictions, fields, crf_seconds = {}, {}, {}
    for arm in ARMS:
        field = torch.as_tensor(result['fields'][arm], dtype=torch.float64).reshape(1, 1, 64, 64)
        assert bool(torch.isfinite(field).all())
        fields[arm] = field.numpy().reshape(64, 64)
        direct = F.interpolate(field, (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > 0
        crf_start = time.monotonic()
        final = HOST._finalize_mask(direct, image)
        crf_seconds[arm] = time.monotonic() - crf_start
        for suffix, predicted in (('direct', direct), ('crf', final)):
            for frame, hw in (('cli1024', (1024, 1024)), ('original', shape)):
                rendered = predicted if hw == (1024, 1024) else F.interpolate(
                    predicted.float()[None, None], hw, mode='bilinear', align_corners=False)[0, 0] > .5
                predictions[frame + '/' + arm + '.' + suffix] = np.packbits(rendered.numpy().reshape(-1))
    filename = f'{i:06d}.npz'
    np.savez_compressed(OUT / 'predictions' / filename, original_hw=np.asarray(shape), **predictions)
    np.savez_compressed(OUT / 'fields' / filename, **fields)
    record = dict(episode_id=row['episode_id'], dataset=row['dataset'], filename=filename,
                  prediction_sha256=sha(OUT / 'predictions' / filename),
                  fields_sha256=sha(OUT / 'fields' / filename), apd_applied=apd,
                  semantic=semantic, diagnostics=result['diagnostics'],
                  loading_seconds=loading_seconds, head_seconds=head_seconds,
                  crf_seconds=crf_seconds, readout_seconds=time.monotonic() - tick,
                  total_seconds=time.monotonic() - begun, raw_requests=source['raw'],
                  encoder_calls=0, raw_writes=0, query_GT_reads=0, baseline_mask_reads=0)
    write(recfile, record)
    return record


def verify_configuration(out):
    cfg = read(out / 'config.json')
    assert sha(out / 'manifest.json') == cfg['manifest_sha256']
    assert sha(out / 'tasks.json') == cfg['tasks_sha256']
    assert sha(cfg['profile_path']) == cfg['profile_sha256']
    assert sha(cfg['basis_path']) == cfg['basis_sha256']
    for relative, digest in cfg['source_sha256'].items():
        assert sha(out / 'frozen' / relative) == digest
    assert sha(__file__) == cfg['source_sha256']['scripts/autonomous_scene_reconstruction.py']
    for path, digest in cfg['external_source_sha256'].items():
        assert sha(path) == digest
    return cfg


def run(out):
    assert not (out / 'sealed.json').exists(), 'Refuse to overwrite a sealed experiment'
    cfg = verify_configuration(out)
    manifest, sources = read(out / 'manifest.json'), read(out / 'tasks.json')
    tasks = [(i, row, source) for i, (row, source) in enumerate(zip(manifest, sources))]
    begun, receipts = time.monotonic(), []
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(cfg['workers'], mp_context=context,
                                               initializer=initialize, initargs=(str(out),)) as executor:
        futures = {executor.submit(infer, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            receipts.append((futures[future], record))
            count = len(receipts)
            if count % 20 == 0 or count == cfg['n']:
                state = dict(state='INFERENCE', completed=count, n=cfg['n'], seconds=time.monotonic()-begun)
                write(out / 'activity.json', state)
                print(json.dumps(state), flush=True)
    receipts = [r for _, r in sorted(receipts)]
    assert len(receipts) == cfg['n'] and all(r['query_GT_reads'] == 0 for r in receipts)
    write(out / 'sealed.json', dict(n=len(receipts), receipts=receipts, config_sha256=sha(out / 'config.json'),
                                   inference_seconds=time.monotonic()-begun, query_GT_read=False))
    write(out / 'activity.json', dict(state='SEALED', n=len(receipts)))


def score(out):
    cfg = verify_configuration(out)
    seal = read(out / 'sealed.json')
    assert seal['n'] == cfg['n'] and sha(out / 'config.json') == seal['config_sha256']
    np, torch, F, _, array_hash, *_ = imports(out / 'frozen')
    from PIL import Image
    from ics.metrics import summarize, counts, gross_edits
    import ics.metrics as metrics_module
    assert Path(metrics_module.__file__).resolve() == (out / 'frozen/src/ics/metrics.py').resolve()
    manifests, sources = read(out / 'manifest.json'), read(out / 'tasks.json')
    records, grouped = [], {}
    for row, source, rec in zip(manifests, sources, seal['receipts']):
        assert row['episode_id'] == source['episode_id'] == rec['episode_id']
        assert sha(out / 'predictions' / rec['filename']) == rec['prediction_sha256']
        with Image.open(row['query_mask_path']) as im:
            gt = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        assert array_hash(gt) == row['query_mask_hash']
        with np.load(out / 'predictions' / rec['filename'], allow_pickle=False) as z:
            for frame, hw in (('cli1024', (1024, 1024)), ('original', tuple(row['query_size_hw']))):
                truth = F.interpolate(torch.from_numpy(gt).float()[None, None], hw, mode='nearest')[0, 0].numpy() > .5
                old = {}
                for arm, b in source['baselines'].items():
                    assert sha(b['path']) == b['sha256']
                    with np.load(b['path'], allow_pickle=False) as prior:
                        old[arm] = np.unpackbits(prior[b['keys'][frame]], count=truth.size).reshape(hw).astype(bool)
                entry = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
                             class_id=row['loader_class_id'], query_photo_id=row['query_photo_id'],
                             frame=frame, iu={a: counts(p, truth) for a, p in old.items()}, edits={})
                for arm in ARMS:
                    for suffix in ('direct', 'crf'):
                        label = arm + '.' + suffix
                        predicted = np.unpackbits(z[frame + '/' + label], count=truth.size).reshape(hw).astype(bool)
                        entry['iu'][label] = counts(predicted, truth)
                        entry['edits'][label] = gross_edits(predicted, old['foris.crf'], truth)
                records.append(entry)
                grouped.setdefault((row['dataset'], frame), []).append(entry)
    (out / 'scored_episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    report = {}
    for (dataset, frame), group in grouped.items():
        expected = cfg['coco_expected_classes'] if dataset == 'coco' else None
        report[dataset + '/' + frame] = summarize(group, baselines=('foris.crf', 'source_kernel.crf'),
                                                  expected_classes=expected, repetitions=2000,
                                                  unit='query_photo')
        if dataset == 'paco_part':
            report[dataset + '/' + frame + '/fixed303'] = summarize(
                group, baselines=('foris.crf', 'source_kernel.crf'),
                expected_classes=cfg['paco_expected_classes'], repetitions=2000, unit='query_photo')
    write(out / 'report.json', dict(results=report, seal_sha256=sha(out / 'sealed.json'),
                                   scored_sha256=sha(out / 'scored_episodes.jsonl'),
                                   exposure=cfg['exposure'], primary_frame='cli1024',
                                   latency_scope=cfg['timing_scope']))
    write(out / 'activity.json', dict(state='COMPLETE', n=cfg['n']))
    print(json.dumps(dict(state='COMPLETE', root=str(out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'score'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'experiment.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare(args.out, args.workers)
        elif args.command == 'run':
            run(args.out)
        else:
            score(args.out)


if __name__ == '__main__':
    main()
