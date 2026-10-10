"""Frozen extended-benchmark O24 producer, complete FoRIS, then legal scoring.

One MPS FP32 B2 producer writes only missing whole R/Q cache entries. A bounded
CPU worker runs the unchanged source FoRIS and exact M4 CRF. Query annotations
are denied during inference and opened only after every output is sealed.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import builtins
from collections import Counter, defaultdict, deque
import concurrent.futures
import fcntl
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'cv_data'
DEFAULT_PREPARED = DATA / 'a/extended_benchmarks_seed0_20261010_v3'
DEFAULT_OUT = DATA / 'a/extended_foris_seed0_20261010'
DEFAULT_PROFILE = DATA / 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class QueryLabelGuard:
    def __init__(self, manifest, assets):
        self.forbidden = set()
        for row in manifest:
            for name in ('query_mask_path', 'query_ignore_mask_path'):
                if name in row:
                    self.forbidden.add(str(Path(row[name]).resolve()))
            if row.get('query_annotation_path'):
                self.forbidden.add(str((assets / row['query_annotation_path']).resolve()))
        self.attempts = 0

    def check(self, path):
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = str(Path(os.fsdecode(path)).resolve())
            if resolved in self.forbidden:
                self.attempts += 1
                raise RuntimeError('Query annotation denied during inference: ' + resolved)

    def __enter__(self):
        from PIL import Image
        self.saved = (builtins.open, io.open, Image.open)
        def guarded(original):
            def call(path, *args, **kwargs):
                self.check(path)
                return original(path, *args, **kwargs)
            return call
        builtins.open, io.open, Image.open = map(guarded, self.saved)
        return self

    def __exit__(self, *args):
        from PIL import Image
        builtins.open, io.open, Image.open = self.saved


def imports(out):
    sys.path[:0] = [str(out / 'frozen/src'), str(out / 'frozen/scripts'),
                    str(out / 'frozen/assets/third_party/foris_official')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.extended_benchmark_data import load_inputs, load_query_ground_truth
    from raw_feature_cache import RawFeatureCache
    from utils.data import build_transform
    return np, torch, F, load_inputs, load_query_ground_truth, RawFeatureCache, build_transform


def prepare(args):
    prepared = read(args.prepared / 'prepared.json')
    validation = read(args.prepared / 'validation.json')
    if prepared['state'] not in ('EXTENDED_PILOT_FROZEN', 'EXTENDED_PILOT_FROZEN_WITH_BLOCKERS'):
        raise ValueError('Prepared benchmark manifest is not ready')
    if validation['state'] != 'PASSED' or sha(args.prepared / 'validation.json') != prepared['validation_sha256']:
        raise ValueError('Prepared validation is incomplete or changed')
    if sha(args.prepared / 'manifest.json') != prepared['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    manifest = read(args.prepared / 'manifest.json')
    if len(manifest) != prepared['n'] or len({r['episode_id'] for r in manifest}) != len(manifest):
        raise ValueError('Prepared episode count/identity differs')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh output; infer resumes only its frozen run')
    args.out.mkdir(parents=True)
    local_files = ['scripts/run_extended_foris.py', 'scripts/cached_dino.py', 'scripts/raw_feature_cache.py',
        'src/ics/__init__.py', 'src/ics/data.py', 'src/ics/foris.py', 'src/ics/representations.py', 'src/ics/native_basis.py',
        'src/ics/official_data.py', 'src/ics/extended_benchmark_data.py', 'src/ics/metrics.py',
        'src/ics/m4_crf.py', 'src/ics/m4_crf_cached.py']
    sources = {}
    for relative in local_files:
        source = REPO / relative
        if relative in ('src/ics/official_data.py', 'src/ics/extended_benchmark_data.py'):
            source = args.prepared / 'source/local' / Path(relative).name
        destination = args.out / 'frozen' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources[relative] = sha(destination)
    frozen_assets = args.out / 'frozen/assets'
    external = {}
    for leaf in ('foris_official', 'crf_source'):
        source_root = args.assets / 'third_party' / leaf
        for source in source_root.rglob('*'):
            if source.is_file() and source.suffix in ('.py', '.cpp', '.h', '.hpp', '.cu', '.cuh'):
                relative = source.relative_to(args.assets)
                destination = frozen_assets / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                external[str(relative)] = sha(destination)
    basis = frozen_assets / 'native_assets/positional_basis.pt'
    basis.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.assets / 'native_assets/positional_basis.pt', basis)
    model = frozen_assets / 'demo4_cache/models/dinov3-vitl16-timm'
    model.parent.mkdir(parents=True, exist_ok=True)
    model.symlink_to(args.assets / 'demo4_cache/models/dinov3-vitl16-timm', target_is_directory=True)
    profile = read(args.profile)
    expected = {'weights_sha256': args.assets / 'demo4_cache/models/dinov3-vitl16-timm/model.safetensors',
        'model_config_sha256': args.assets / 'demo4_cache/models/dinov3-vitl16-timm/config.json',
        'encoder_source_sha256': args.out / 'frozen/src/ics/data.py',
        'observer_source_sha256': args.out / 'frozen/src/ics/representations.py'}
    for field, path in expected.items():
        if sha(path) != profile[field]:
            raise ValueError('Raw-cache profile differs: ' + field)
    if sha(frozen_assets / 'third_party/foris_official/utils/data.py') != profile['preprocessing']['source_sha256']:
        raise ValueError('Raw-cache preprocessing differs')
    shutil.copyfile(args.prepared / 'manifest.json', args.out / 'manifest.json')
    shutil.copyfile(args.prepared / 'prepared.json', args.out / 'prepared.json')
    shutil.copyfile(args.prepared / 'validation.json', args.out / 'input_validation.json')
    write(args.out / 'profile.json', profile)
    cfg = dict(n=len(manifest), datasets=dict(Counter(r['dataset'] for r in manifest)),
        original_assets=str(args.assets), frozen_assets=str(frozen_assets), profile_path=str(args.profile),
        profile_sha256=sha(args.profile), source_sha256=sources, external_source_sha256=external,
        manifest_sha256=sha(args.out / 'manifest.json'), prepared_sha256=sha(args.out / 'prepared.json'),
        basis_sha256=sha(basis), weights_sha256=profile['weights_sha256'],
        producer_device='mps', encoder_dtype='float32', encoder_batch=2, branches=['O/24'],
        workers=1, threads_each=2, max_pending=2, no_existing_cache_entry_rewrites=True,
        full_pipeline='unchanged source FoRIS; exact original CRF solver/settings; prepared geometry backend',
        image_size=1024, svd_components=500, tau=.6, resize_to_orig_size=False,
        query_annotation_access='Denied via builtins.open, io.open and PIL.Image.open during producer and CPU inference',
        metric='released CLI1024: fold mean of expected-class pooled I/U; original frame separately; direct nearest GT/ignore per frame',
        scope=prepared['pilot_scope'], pending_benchmarks=prepared['pending_datasets'],
        unique_rgb_inputs=len({r[role + '_rgb_hash'] for r in manifest for role in ('reference', 'query')}))
    write(args.out / 'config.json', cfg)
    for folder in ('predictions', 'fields', 'records', 'producer_records'):
        (args.out / folder).mkdir()
    print(json.dumps(dict(state='PREPARED', root=str(args.out), n=cfg['n'],
        unique_rgb_inputs=cfg['unique_rgb_inputs'], datasets=cfg['datasets'])), flush=True)


def verify(out):
    cfg = read(out / 'config.json')
    if sha(out / 'manifest.json') != cfg['manifest_sha256'] or sha(out / 'prepared.json') != cfg['prepared_sha256']:
        raise ValueError('Frozen protocol changed')
    if sha(Path(cfg['profile_path'])) != cfg['profile_sha256']:
        raise ValueError('Shared raw profile changed')
    for relative, digest in cfg['source_sha256'].items():
        if sha(out / 'frozen' / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    for relative, digest in cfg['external_source_sha256'].items():
        if sha(Path(cfg['frozen_assets']) / relative) != digest:
            raise ValueError('Frozen external source changed: ' + relative)
    return cfg


def initialize_worker(root):
    global OUT, CFG, np, torch, F, LOAD, CACHE, ADAPTER, HOST, WORKER_GUARD
    OUT = Path(root)
    CFG = verify(OUT)
    np, torch, F, LOAD, _, cache_class, _ = imports(OUT)
    torch.set_num_threads(CFG['threads_each'])
    torch.manual_seed(0)
    profile_path = Path(CFG['profile_path'])
    CACHE = cache_class(profile_path.parent.parent, read(profile_path))
    from cached_dino import CachedDINO, cache_host
    ADAPTER = CachedDINO(CACHE)
    # Frozen assets use their own new CRF runtime, preserving shared runtime.
    HOST = cache_host(CFG['frozen_assets'], ADAPTER, mask_refiner='crf')
    WORKER_GUARD = QueryLabelGuard(read(OUT / 'manifest.json'), Path(CFG['original_assets']))


def baseline(task):
    index, row, producer = task
    record_path = OUT / 'records' / f'{index:06d}.json'
    if record_path.exists():
        record = read(record_path)
        if sha(OUT / 'predictions' / record['filename']) != record['prediction_sha256']:
            raise ValueError('Existing baseline prediction changed')
        return record
    start = time.monotonic()
    ADAPTER.used_entries = {}
    with torch.inference_mode(), WORKER_GUARD:
        reference, reference_mask, query = LOAD(row, Path(CFG['original_assets']))
        from ics.foris import run_foris
        mask, stages, _, _ = run_foris(HOST, reference, reference_mask, query)
        if ADAPTER.encoder is not None or ADAPTER.encoder_calls != 0:
            raise RuntimeError('CPU baseline constructed or invoked an encoder')
        if set(ADAPTER.used_entries) != {request['key'] for request in producer['raw']}:
            raise RuntimeError('FoRIS read inputs outside the recorded whole R/Q pair')
        shape = tuple(row['query_size_hw'])
        original = F.interpolate(mask.float()[None, None], shape, mode='bilinear', align_corners=False)[0, 0] > .5
        filename = f'{index:06d}.npz'
        np.savez_compressed(OUT / 'predictions' / filename, original_hw=np.asarray(shape),
            **{'cli/foris.crf': np.packbits(mask.numpy().reshape(-1)),
               'original/foris.crf': np.packbits(original.numpy().reshape(-1))})
        np.savez_compressed(OUT / 'fields' / filename, score=stages['score'].float().numpy())
        record = dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
            filename=filename, prediction_sha256=sha(OUT / 'predictions' / filename),
            field_sha256=sha(OUT / 'fields' / filename), seconds=time.monotonic() - start,
            raw=producer['raw'], encoder_calls=0, raw_cache_writes=0,
            query_label_attempts=WORKER_GUARD.attempts, query_GT_reads=0)
        write(record_path, record)
    return record


def infer(out):
    cfg = verify(out)
    np, torch, F, load_inputs, _, cache_class, build_transform = imports(out)
    torch.set_num_threads(cfg['threads_each'])
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for this frozen FP32 producer')
    profile_path = Path(cfg['profile_path'])
    cache = cache_class(profile_path.parent.parent, read(profile_path))
    transform = build_transform(1024)
    manifest = read(out / 'manifest.json')
    encoder, completed, pending = None, {}, deque()
    start = time.monotonic()
    guard = QueryLabelGuard(manifest, Path(cfg['original_assets']))
    context = multiprocessing.get_context('spawn')
    def receive():
        future = pending.popleft()
        record = future.result()
        completed[record['index']] = record
        write(out / 'activity.json', dict(state='INFERENCE', complete=len(completed), total=len(manifest),
            last=record['episode_id'], seconds=time.monotonic() - start, worker_seconds=record['seconds']))
        print(json.dumps(dict(n=len(completed), total=len(manifest), episode_id=record['episode_id'],
            baseline_seconds=record['seconds'], wall_seconds=time.monotonic() - start)), flush=True)
    with torch.inference_mode(), guard, concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg['workers'], mp_context=context, initializer=initialize_worker,
            initargs=(str(out),)) as pool:
        for index, row in enumerate(manifest):
            producer_start = time.monotonic()
            reference, _, query = load_inputs(row, Path(cfg['original_assets']))
            inputs = torch.stack([transform(reference), transform(query)])
            arrays = inputs.numpy()
            keys = [cache.key(x) for x in arrays]
            missing = [i for i, key in enumerate(keys) if not (cache.folder / key / 'entry.json').is_file()]
            encoding_seconds, newly_written = 0.0, []
            if missing:
                if encoder is None:
                    from ics.data import TimmDINOv3
                    initialized = time.monotonic()
                    encoder = TimmDINOv3(Path(cfg['frozen_assets']) / 'demo4_cache/models/dinov3-vitl16-timm')
                    encoder = encoder.to('mps').eval().requires_grad_(False)
                    write(out / 'encoder_initialization.json', dict(seconds=time.monotonic() - initialized,
                        producer_device='mps', dtype='float32', batch=2))
                tick = time.monotonic()
                maps = encoder.get_intermediate_layers(inputs.to('mps'), n=1, reshape=True)[0]
                torch.mps.synchronize()
                features = maps.cpu().flatten(2).transpose(1, 2).contiguous().numpy()
                encoding_seconds = time.monotonic() - tick
                for i in missing:
                    # Check under the writer lock: never append provenance or
                    # replace a payload that another producer already created.
                    with cache._locked(arrays[i], True):
                        entry = cache.folder / keys[i] / 'entry.json'
                        if entry.exists():
                            old = cache._read(arrays[i], ('O/24',))['O/24']
                            if not np.array_equal(old, features[i]):
                                raise ValueError('Concurrent input/profile produced different O24')
                        else:
                            cache._write(arrays[i], {'O/24': features[i]}, dict(
                                experiment=str(out), episode_id=row['episode_id'], view_role=('reference', 'query')[i],
                                batch_index=i, batch_size=2, producer='MPS FP32 whole R/Q', oracle_input=False))
                            newly_written.append(keys[i])
                del maps, features
            requests = []
            for role, key in zip(('reference', 'query'), keys):
                entry = cache.folder / key / 'entry.json'
                info = read(entry)
                requests.append(dict(role=role, key=key, entry_path=str(entry), entry_sha256=sha(entry),
                    input_tensor_hash=info['input_tensor_hash'], payload_sha256=info['file_sha256'],
                    tensor_sha256=info['features']['O/24']['tensor_sha256']))
            producer = dict(index=index, episode_id=row['episode_id'], raw=requests,
                paired_encoder_calls=int(bool(missing)), newly_written_keys=newly_written,
                encoding_seconds=encoding_seconds, producer_seconds=time.monotonic() - producer_start,
                query_label_attempts=guard.attempts, query_GT_reads=0)
            producer_path = out / 'producer_records' / f'{index:06d}.json'
            if producer_path.exists():
                saved = read(producer_path)
                if saved['episode_id'] != row['episode_id'] or saved['raw'] != requests:
                    raise ValueError('Recorded producer inputs changed on resume')
                producer = saved
            else:
                write(producer_path, producer)
            pending.append(pool.submit(baseline, (index, row, producer)))
            if len(pending) >= cfg['max_pending']:
                receive()
        while pending:
            receive()
    ordered = [completed[i] for i in range(len(manifest))]
    for record in ordered:
        if sha(out / 'predictions' / record['filename']) != record['prediction_sha256']:
            raise ValueError('Unsealed prediction changed')
    write(out / 'inference.json', ordered)
    producers = [read(out / 'producer_records' / f'{i:06d}.json') for i in range(len(manifest))]
    write(out / 'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=len(manifest),
        manifest_sha256=sha(out / 'manifest.json'), config_sha256=sha(out / 'config.json'),
        inference_sha256=sha(out / 'inference.json'), predictions=[r['prediction_sha256'] for r in ordered],
        paired_encoder_calls=sum(r['paired_encoder_calls'] for r in producers),
        new_raw_inputs=len({k for r in producers for k in r['newly_written_keys']}),
        encoding_seconds=sum(r['encoding_seconds'] for r in producers),
        producer_seconds=sum(r['producer_seconds'] for r in producers),
        CPU_baseline_seconds=sum(r['seconds'] for r in ordered), wall_seconds=time.monotonic() - start,
        query_label_attempts=guard.attempts + sum(r['query_label_attempts'] for r in ordered), query_GT_reads=0))
    print(json.dumps(dict(state='ALL_PREDICTIONS_SEALED', root=str(out), n=len(manifest))), flush=True)


def score(out):
    cfg = verify(out)
    seal = read(out / 'sealed.json')
    if seal['state'] != 'ALL_PREDICTIONS_SEALED' or seal['n'] != cfg['n']:
        raise ValueError('Cannot score incomplete predictions')
    for name, field in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                        ('inference.json', 'inference_sha256')]:
        if sha(out / name) != seal[field]:
            raise ValueError('Sealed scoring input changed')
    np, torch, _, _, load_gt, _, _ = imports(out)
    from ics.metrics import counts
    prepared = read(out / 'prepared.json')
    records = read(out / 'inference.json')
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    details = []
    for row, record in zip(read(out / 'manifest.json'), records):
        if row['episode_id'] != record['episode_id']:
            raise ValueError('Prediction order differs')
        path = out / 'predictions' / record['filename']
        if sha(path) != record['prediction_sha256']:
            raise ValueError('Prediction changed after sealing')
        result = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
            class_id=row['loader_class_id'], query_photo_id=row['query_photo_id'], iu={})
        if row['dataset'] == 'isaid':
            result['query_scene_id'] = row['query_photo_id'].split('_')[0]
        with np.load(path, allow_pickle=False) as predictions:
            for frame, hw in [('cli', (1024, 1024)), ('original', tuple(row['query_size_hw']))]:
                truth, ignore = load_gt(row, hw)
                prediction = np.unpackbits(predictions[frame + '/foris.crf'], count=int(np.prod(hw))).reshape(hw).astype(bool)
                iu = counts(prediction, truth, ignore=ignore)
                result['iu'][frame] = iu
                result.setdefault('ignored_pixels', {})[frame] = 0 if ignore is None else int(ignore.sum())
                target = totals[row['dataset']][frame][(row['fold'], row['loader_class_id'])]
                target[0] += iu[0]
                target[1] += iu[1]
        details.append(result)
    reports = {}
    for dataset in cfg['datasets']:
        expected = {int(k.split('/')[1]): v['expected_class_ids'] for k, v in prepared['folds'].items()
                    if k.startswith(dataset + '/')}
        frames = {}
        for frame in ('cli', 'original'):
            folds, missing = {}, {}
            for fold, classes in expected.items():
                values = [totals[dataset][frame].get((fold, c), [0, 0]) for c in classes]
                folds[str(fold)] = float(100 * np.mean([i / max(u, 1) for i, u in values]))
                missing[str(fold)] = [c for c in classes if (fold, c) not in totals[dataset][frame]]
            frames[frame] = dict(miou=float(np.mean(list(folds.values()))), per_fold=folds,
                missing_expected_classes=missing,
                per_class_iu={f'{f}/{c}': iu for (f, c), iu in sorted(totals[dataset][frame].items())})
        reports[dataset] = dict(n=cfg['datasets'][dataset], frames=frames,
            scope='first20 sequential draws/fold; incomplete class coverage; pilot only' if dataset == 'isaid'
                else 'released default seed0 full600 draws; original duplicate draws retained')
    write(out / 'episode_metrics.json', details)
    report = dict(state='SCORED_AFTER_FULL_SEAL', n=cfg['n'], datasets=reports,
        metric=cfg['metric'], query_label_role='scoring only after all660 outputs sealed',
        ignore_preserved=True, sealed_sha256=sha(out / 'sealed.json'), timing=seal,
        unavailable_benchmarks=cfg['pending_benchmarks'])
    write(out / 'report.json', report)
    write(out / 'COMPLETE.json', dict(state='COMPLETE', n=cfg['n'], report_sha256=sha(out / 'report.json'),
        sealed_sha256=sha(out / 'sealed.json'), all_expected_predictions_scored=True))
    print(json.dumps(dict(state='COMPLETE', n=cfg['n'], miou={d: {f: v['miou'] for f, v in r['frames'].items()}
        for d, r in reports.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'infer', 'score', 'run'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--assets', type=Path, default=DATA)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    for name in ('out', 'prepared', 'assets', 'profile'):
        setattr(args, name, getattr(args, name).resolve())
    if args.mode in ('prepare', 'run'):
        prepare(args)
        if args.mode == 'run':
            frozen = args.out / 'frozen/scripts/run_extended_foris.py'
            os.execv(sys.executable, [sys.executable, str(frozen), 'infer', '--out', str(args.out)])
        return
    with open(args.out / 'controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'infer':
            infer(args.out)
        score(args.out)


if __name__ == '__main__':
    main()
