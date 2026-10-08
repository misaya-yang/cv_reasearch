#!/usr/bin/env python3
"""Freeze official episodes and evaluate complete M4 baselines without feature archives.

prepare uses the unedited official loaders. evaluate reads R/R-mask/Q only,
seals all predictions, then score opens query masks in a separate phase.
The initial B0 is the historical RCG + fine readout at a fixed .5 cut.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_rows(path, repair_partial_last=False):
    if path.suffix == '.jsonl':
        lines = path.read_text().splitlines(keepends=True)
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not repair_partial_last or i != len(lines)-1 or line.endswith('\n'):
                    raise
                # A crashed append has no committed result. Keep every complete
                # record and recompute only the uncommitted episode.
                path.write_text(''.join(lines[:-1]))
        return rows
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data['episodes']


@contextmanager
def locked_run(path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path/'run.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Inference or scoring for this run is already live')
        try:
            lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(a):
    from ics.official_data import FOLDS, build_dataset, file_hash, record_episode, reset_sampling_seed
    receipt = json.loads((a.assets/'setup/download_receipt.json').read_text())
    verified = {x['id'] for x in receipt['assets'] if x['status'] == 'VERIFIED'}
    required = {'coco2014' if name == 'coco' else 'ics_datasets' for name in a.datasets}
    fresh = a.split in ('dev', 'val')
    seed = {'dev': 1, 'val': 2}.get(a.split, 0)
    # Fresh manifests use the unchanged metadata-driven sampling pools. A
    # missing selected input fails that dataset; it is never replaced. SUIM
    # separately requires its complete mask-derived pool in build_dataset().
    if not a.smoke and not fresh and required-verified:
        raise FileNotFoundError('Official full-pool SHA receipt pending: '+str(sorted(required-verified)))
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Frozen preparation requires a fresh directory')
    a.out.mkdir(parents=True, exist_ok=True)
    hashes, all_rows, metadata, pending = {}, [], {}, {}
    for name in a.datasets:
        folds = [a.fold] if a.fold is not None else ([FOLDS[name][0]] if a.smoke else FOLDS[name])
        dataset_rows, dataset_metadata = [], {}
        try:
            for fold in folds:
                reset_sampling_seed(seed)
                ds = build_dataset(name, fold, a.assets)
                if a.smoke:
                    n = min(len(ds), 1)
                elif fresh:
                    n = 200 if name == 'lvis' else 2000 if name == 'suim' else 500
                else:
                    n = len(ds)
                key = f'{name}/{fold}'
                dataset_metadata[key] = dict(official_length=len(ds), expected_class_ids=list(ds.class_ids),
                                            selected_length=n, smoke=a.smoke)
                for index in range(n):
                    row = record_episode(ds, index, fold, a.assets, a.out/'masks', hashes)
                    if fresh:
                        row['episode_id'] = f'{a.split}_s{seed}/'+row['episode_id']
                    dataset_rows.append(row)
                    if (index+1) % 100 == 0:
                        print(json.dumps(dict(task='prepare', dataset=name, fold=fold, n=index+1, total=n)), flush=True)
        except FileNotFoundError as error:
            if not fresh:
                raise
            pending[name] = str(error)
            print(json.dumps(dict(task='prepare', dataset=name, state='WAITING_INPUT', error=str(error))), flush=True)
            continue
        # Only completed datasets enter the replay manifest. Other datasets
        # proceed independently; partially sampled datasets are not scored.
        for fold in folds:
            rows = [r for r in dataset_rows if r['fold'] == fold]
            dest = a.out/'manifests'/name/f'fold_{fold}.jsonl'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        all_rows.extend(dataset_rows)
        metadata.update(dataset_metadata)
    write(a.out/'manifest.json', all_rows)
    source = a.assets/'third_party/foris_official'
    write(a.out/'prepared.json', dict(state='SMOKE_ONLY' if a.smoke else 'FRESH_MANIFEST_FROZEN' if fresh else 'OFFICIAL_MANIFEST_FROZEN',
        seed=seed, split_role=a.split if fresh else 'official', pending_datasets=pending,
        shots=1, shuffle=False, num_workers=0, folds=metadata, n=len(all_rows),
        manifest_sha256=file_hash(a.out/'manifest.json'),
        loader_sha256={n: file_hash(source/'datasets'/(n+'.py')) for n in a.datasets},
        query_label_role='protocol preparation, not candidate scoring',
        duplicates='official legal draws retained',
        rng='official loaders use NumPy; inference must leave its RNG unchanged'))
    print(json.dumps(dict(task='prepare', n=len(all_rows), state='SMOKE_ONLY' if a.smoke else 'FROZEN', pending=pending)), flush=True)


def prepare_dev(a):
    from ics.development_data import prepare_development
    from ics.official_data import file_hash
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Development preparation requires a fresh output directory')
    rows, audit = prepare_development(a.assets, a.datasets, a.out/'masks')
    write(a.out/'manifest.json', rows)
    write(a.out/'prepared.json', dict(state='DEVELOPMENT_MANIFEST_FROZEN', split_role='dev',
        n=len(rows), manifest_sha256=file_hash(a.out/'manifest.json'), datasets=audit,
        labels='hashes/identities prepared here; query labels excluded from inference',
        historical_exposure='reused development; legacy original photo/object IDs partly missing'))
    print(json.dumps(dict(task='prepare-dev', n=len(rows), datasets=audit)), flush=True)


def overlap(a):
    from ics.development_data import exclude_development
    from ics.official_data import file_hash
    if a.manifest is None or a.dev_manifest is None:
        raise ValueError('overlap requires --manifest (official) and --dev-manifest')
    original_prepared = json.loads((a.manifest.parent/'prepared.json').read_text())
    if original_prepared['state'] != 'OFFICIAL_MANIFEST_FROZEN':
        raise ValueError('A smoke cannot define the official confirmation set')
    if file_hash(a.manifest) != original_prepared['manifest_sha256']:
        raise ValueError('Official frozen manifest changed')
    development_prepared = json.loads((a.dev_manifest.parent/'prepared.json').read_text())
    if development_prepared['state'] != 'DEVELOPMENT_MANIFEST_FROZEN':
        raise ValueError('Require frozen development identities')
    if file_hash(a.dev_manifest) != development_prepared['manifest_sha256']:
        raise ValueError('Development frozen manifest changed')
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Overlap audit requires a fresh output directory')
    expected = defaultdict(dict)
    for key, value in original_prepared['folds'].items():
        name, fold = key.split('/')
        expected[name][fold] = value['expected_class_ids']
    confirm, audit = exclude_development(read_rows(a.manifest), read_rows(a.dev_manifest), expected)
    write(a.out/'manifest.json', confirm)
    audit.update(official_manifest_sha256=file_hash(a.manifest),
                 development_manifest_sha256=file_hash(a.dev_manifest),
                 confirm_manifest_sha256=file_hash(a.out/'manifest.json'),
                 development_preparation=development_prepared['datasets'])
    write(a.out/'overlap_report.json', audit)
    write(a.out/'prepared.json', dict(state='CONFIRM_MANIFEST_AUDITED' if audit['eligible_for_confirmation']
        else 'PROTOCOL_IDENTITY_UNRESOLVED', n=len(confirm), folds=original_prepared['folds'],
        manifest_sha256=audit['confirm_manifest_sha256'],
        overlap_audit_sha256=file_hash(a.out/'overlap_report.json'),
        eligible_for_confirmation=audit['eligible_for_confirmation']))
    print(json.dumps({k: audit[k] for k in ('official_n', 'development_n', 'confirm_n',
                     'excluded_official_draws', 'empty_confirm_classes', 'eligible_for_confirmation')}), flush=True)


def prediction_file(row):
    from hashlib import sha256
    return sha256(row['episode_id'].encode()).hexdigest()+'.npz'


def run_config(a):
    from ics.official_data import file_hash
    from ics.methods import rcg
    from ics import fine_readout
    import importlib.metadata
    code = [Path(__file__), REPO/'scripts/run_m4_baselines.py',
            *sorted((REPO/'src/ics').rglob('*.py')),
            *sorted((a.assets/'third_party/foris_official').rglob('*.py')),
            *sorted((a.assets/'third_party/crf_source/src/CRF').glob('*.py'))]
    code += sorted((a.assets/'third_party/crf_source/src/PermutohedralFiltering/source/cpu').rglob('*.cpp'))
    code += sorted((a.assets/'third_party/crf_source/src/PermutohedralFiltering/source/cpu').rglob('*.h'))
    prepared_path = a.manifest.parent/'prepared.json'
    prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else None
    if a.split == 'official' and (prepared is None or prepared['state'] != 'OFFICIAL_MANIFEST_FROZEN'):
        raise ValueError('Official evaluation requires the complete prepared official manifest')
    if a.split == 'confirm':
        raise ValueError('Confirmation disabled until development overlap/exposure audit is frozen')
    fresh = prepared is not None and prepared.get('state') == 'FRESH_MANIFEST_FROZEN'
    if a.split == 'val' and not fresh:
        raise ValueError('Validation requires fresh seed2 official-loader manifests')
    if fresh and prepared.get('split_role') != a.split:
        raise ValueError('Prepared development/validation split differs')
    return dict(candidate_id='B0', parent_baseline_id='complete FoRIS',
        split_role=a.split, datasets=a.datasets, branch=subprocess.check_output(
            ['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip(),
        commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        assets=str(a.assets), manifest_path=str(a.manifest), manifest_sha256=file_hash(a.manifest),
        source_sha256={str(p): file_hash(p) for p in code},
        prepared_protocol=prepared,
        dependencies={n: importlib.metadata.version(n) for n in
                      ('torch', 'torchvision', 'timm', 'numpy', 'scipy', 'scikit-learn', 'pycocotools')},
        weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
        basis_sha256=file_hash(a.assets/'native_assets/positional_basis.pt'),
        encoder_device=a.encoder_device, encoder_dtype='float32', threads=a.threads,
        mask_refiner='original CRF CPU lattice/solver, 10 iterations',
        rcg=rcg.CONFIG, fine_readout=fine_readout.CONFIG,
        arms=['foris.crf', 'rcg', 'rcg.fine', 'mean'],
        bootstrap_unit='query_photo' if fresh else 'episode',
        bootstrap_repetitions=10000 if fresh else 100000,
        official_score_arms=['foris.crf'],
        official_B0_predictions='computed and sealed, withheld from official reproduction scores until the audited single B0 confirmation look',
        metric='original loader query size; 1024 CLI metric retained separately',
        query_mask_in_inference=False, feature_archive=False)


def _infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import file_hash, load_inputs
    from ics.foris import build_host, run_foris
    from ics.m4_crf import install
    from ics.methods import mean_control, rcg
    from ics import fine_readout

    if a.encoder_device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    selected = [r for r in read_rows(a.manifest) if r['dataset'] in a.datasets]
    if not selected or len({r['episode_id'] for r in selected}) != len(selected):
        raise ValueError('Require nonempty unique draw IDs, not unique image pairs')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'sealed.json').exists():
        old = json.loads((a.out/'config.json').read_text())
        requested = dict(assets=str(a.assets), manifest_sha256=file_hash(a.manifest),
                         datasets=a.datasets, split_role=a.split,
                         encoder_device=a.encoder_device, threads=a.threads)
        if not a.resume or any(old[k] != v for k, v in requested.items()):
            raise ValueError('A sealed run is immutable; require --resume with the original inputs')
        print('Predictions already sealed; using their archived implementation and score phase', flush=True)
        return
    config = run_config(a)
    if (a.out/'config.json').exists():
        old = json.loads((a.out/'config.json').read_text())
        if not a.resume or {k:v for k,v in old.items() if k != 'commit'} != {k:v for k,v in config.items() if k != 'commit'}:
            raise ValueError('Existing run/config changed; cannot mix implementations')
    else:
        write(a.out/'config.json', config)
        write(a.out/'manifest.json', selected)
        snapshots = {}
        for origin, digest in config['source_sha256'].items():
            p = Path(origin)
            if p.is_relative_to(REPO):
                relative = Path('repo')/p.relative_to(REPO)
            else:
                relative = Path('assets')/p.relative_to(a.assets)
            target = a.out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            if file_hash(target) != digest:
                raise ValueError('Code changed during source snapshot')
            snapshots[origin] = dict(path=str(relative), sha256=digest)
        write(a.out/'source/source_manifest.json', snapshots)
    progress_path = a.out/'inference.jsonl'
    done = read_rows(progress_path, repair_partial_last=True) if progress_path.exists() else []
    completed = {r['episode_id']: r for r in done}
    for r in done:
        p = a.out/'predictions'/r['prediction_file']
        if file_hash(p) != r['prediction_sha256']:
            raise ValueError('Completed prediction changed')
    (a.out/'predictions').mkdir(exist_ok=True)
    (a.out/'fields').mkdir(exist_ok=True)
    torch.set_num_threads(a.threads)
    init_begin = time.monotonic()
    _, backend = install(a.assets/'third_party/crf_source', a.assets/'runtime/macos/crf')
    host = build_host({'projection_basis': str(a.assets/'native_assets/positional_basis.pt')},
                      'cpu', str(a.assets/'third_party/foris_official'),
                      weights=str(a.assets/'demo4_cache/models/dinov3-vitl16-timm'))
    host.encoder.to(a.encoder_device)
    encoder_times = []

    def sync():
        if a.encoder_device == 'mps':
            torch.mps.synchronize()

    def extract(imgs):
        sync(); started = time.monotonic()
        b, t = imgs.shape[:2]
        maps = host.encoder.get_intermediate_layers(
            imgs.reshape(b*t, *imgs.shape[2:]).to(a.encoder_device), n=1, reshape=True)[0]
        sync()
        result = maps.cpu().reshape(b, t, *maps.shape[1:])
        encoder_times.append(time.monotonic()-started)
        return result
    host._extract_features = extract
    refinement_times = []
    original_finalize = host._finalize_mask

    def finalize(*args, **kwargs):
        start = time.monotonic()
        result = original_finalize(*args, **kwargs)
        refinement_times.append(time.monotonic()-start)
        return result
    host._finalize_mask = finalize
    write(a.out/'initialization.json', dict(seconds=time.monotonic()-init_begin, crf=backend))
    from run_m4_baselines import render
    # One fixed first episode of each dataset provides five timing probes. The
    # immutable official_index remains the sampling identity; no resampling.
    first = {}
    for row in selected:
        first.setdefault(row['dataset'], row['episode_id'])
    timing_ids = set(first.values())
    order = [r for r in selected if r['episode_id'] in timing_ids]
    order += [r for r in selected if r['episode_id'] not in timing_ids]
    with torch.inference_mode(), open(progress_path, 'a', buffering=1) as log:
        for row in order:
            if row['episode_id'] in completed:
                continue
            sync(); start = time.monotonic(); encoder_times.clear(); refinement_times.clear()
            rgb, gold, query = load_inputs(row, a.assets)
            rng_before = np.random.get_state()
            native, got, transformed, target = run_foris(host, rgb, gold, query)
            native_seconds = time.monotonic()-start
            paired_encoding_seconds = sum(encoder_times)
            refined_seconds = sum(refinement_times)
            processed = F.normalize(got['deb'][0].float(), dim=1)
            q, r = (processed[i].flatten(1).T.half().contiguous() for i in (1, 0))
            cov = F.interpolate(transformed[None, None].float(), (64, 64), mode='area')[0, 0].numpy()
            score = got['score'].float().numpy()
            at = time.monotonic(); z, solver = rcg.predict(q, r, cov, score)
            rcg_seconds = time.monotonic()-at
            at = time.monotonic(); mf, _ = mean_control.predict(q, r, cov, score)
            mean_seconds = time.monotonic()-at
            at = time.monotonic()
            debiased = bool((F.normalize(got['raw'][0].float(), dim=1)-got['deb'][0]).abs().max() > 1e-4)
            fine_features = fine_readout.shifted_features(host, target, debiased)
            fine = fine_readout.field(fine_features, q, z)
            sync(); fine_seconds = time.monotonic()-at
            del fine_features
            if not all(np.array_equal(x, y) for x, y in zip(rng_before, np.random.get_state())):
                raise RuntimeError('Inference consumed official NumPy sampling RNG; record a joint official sample stream')
            masks = {'foris.crf': native.numpy(), 'rcg': rcg.mask_from_field(z),
                     'rcg.fine': fine_readout.mask(fine), 'mean': rcg.mask_from_field(mf)}
            shape = (query.height, query.width)
            payload = {kind+'/'+arm: np.packbits(m if kind == 'cli' else render(m, shape))
                       for arm, m in masks.items() for kind in ('cli', 'original')}
            filename = prediction_file(row)
            destination = a.out/'predictions'/filename
            np.savez_compressed(destination, original_hw=np.array(shape), **payload)
            np.savez_compressed(a.out/'fields'/filename, score=score, rcg=z, rcg_fine=fine, mean=mf)
            record = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
                prediction_file=filename, prediction_sha256=file_hash(destination),
                fields_sha256=file_hash(a.out/'fields'/filename), timing_probe=row['episode_id'] in timing_ids,
                foris_seconds=native_seconds, paired_encoding_seconds=paired_encoding_seconds,
                original_finalizer_seconds=refined_seconds, rcg_head_seconds=rcg_seconds,
                mean_head_seconds=mean_seconds, fine_head_seconds=fine_seconds,
                B0_inference_upper_bound_seconds=native_seconds-refined_seconds+rcg_seconds+fine_seconds,
                combined_pipeline_seconds=time.monotonic()-start, solver=solver,
                official_sampling_rng_unchanged=True)
            log.write(json.dumps(record)+'\n'); completed[row['episode_id']] = record
            print(json.dumps(dict(n=len(completed), total=len(selected), last=row['episode_id'],
                                  B0_seconds=record['B0_inference_upper_bound_seconds'])), flush=True)
            del got, masks, q, r
    write(a.out/'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=len(selected),
        manifest_sha256=file_hash(a.out/'manifest.json'), config_sha256=file_hash(a.out/'config.json'),
        initialization_sha256=file_hash(a.out/'initialization.json'),
        source_manifest_sha256=file_hash(a.out/'source/source_manifest.json'),
        inference_index_sha256=file_hash(progress_path), query_labels_opened=False))


def infer(a):
    with locked_run(a.out):
        _infer(a)


def _score(a):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from ics.official_data import array_hash, file_hash
    from ics.metrics import counts, gross_edits, summarize
    seal = json.loads((a.out/'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('Inference must be complete before query scoring')
    for file, key in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                      ('inference.jsonl', 'inference_index_sha256')]:
        if file_hash(a.out/file) != seal[key]:
            raise ValueError('Sealed input changed: '+file)
    for file, key in [('initialization.json', 'initialization_sha256'),
                      ('source/source_manifest.json', 'source_manifest_sha256')]:
        if key in seal and file_hash(a.out/file) != seal[key]:
            raise ValueError('Sealed provenance changed: '+file)
    if (a.out/'report.json').exists():
        print('Existing scored report retained; no new query-label or statistical look', flush=True)
        return
    config = json.loads((a.out/'config.json').read_text())
    score_arms = config['official_score_arms'] if config['split_role'] == 'official' else config['arms']
    bases = tuple(b for b in ('foris.crf', 'rcg', 'rcg.fine') if b in score_arms)
    rows = read_rows(a.out/'manifest.json')
    index = {r['episode_id']: r for r in read_rows(a.out/'inference.jsonl')}
    ledger_path = (a.out.parent/'ledger.jsonl' if config['split_role'] == 'explore'
                   else REPO/'evidence/local/frozen_dino_plan_20261008/ledger.jsonl')
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = read_rows(ledger_path) if ledger_path.exists() else []
    identity = dict(candidate_id=config['candidate_id'], config_sha256=seal['config_sha256'],
                    manifest_sha256=seal['manifest_sha256'], split_role=config['split_role'],
                    methods_scored=score_arms)
    past = [r for r in ledger if r.get('identity') == identity]
    if past and past[0]['run'] != str(a.out):
        raise ValueError('This candidate/split already has a scoring look in another run')
    if not past:
        with open(ledger_path, 'a') as f:
            f.write(json.dumps(dict(event='SCORE_STARTED', identity=identity, run=str(a.out),
                utc=datetime.now(timezone.utc).isoformat(), statistical_look_count=1))+'\n')
    destination = a.out/'episode_metrics.jsonl'
    existing = read_rows(destination, repair_partial_last=True) if destination.exists() else []
    completed = {r['episode_id'] for r in existing}
    with open(destination, 'a', buffering=1) as log:
        for row in rows:
            if row['episode_id'] in completed:
                continue
            path = a.out/'predictions'/index[row['episode_id']]['prediction_file']
            if file_hash(path) != index[row['episode_id']]['prediction_sha256']:
                raise ValueError('Prediction changed')
            with Image.open(row['query_mask_path']) as im:
                truth_source = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
            if array_hash(truth_source) != row['query_mask_hash']:
                raise ValueError('Frozen query annotation changed')
            shape = tuple(row['query_size_hw'])
            if 'query_mask_size_hw' in row and truth_source.shape != tuple(row['query_mask_size_hw']):
                raise ValueError('Frozen source annotation geometry changed')
            # Match the public evaluator even when the original BMP/image
            # sizes disagree. Map raw GT directly to each frame, never via a
            # second nearest-resize through the other metric's frame.
            truth_tensor = torch.from_numpy(truth_source)[None, None].float()
            truth = F.interpolate(truth_tensor, shape, mode='nearest')[0, 0].numpy() > .5
            truth_cli = F.interpolate(truth_tensor,
                                      (1024, 1024), mode='nearest')[0, 0].numpy() > .5
            with np.load(path, allow_pickle=False) as z:
                masks = {kind: {arm: np.unpackbits(z[kind+'/'+arm], count=int(np.prod(hw))).reshape(hw).astype(bool)
                               for arm in score_arms}
                         for kind, hw in [('original', shape), ('cli', (1024, 1024))]}
            r = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
                     class_id=row['loader_class_id'], global_class_id=row['global_class_id'],
                     query_photo_id=row.get('query_photo_id'),
                     truth_pixels=int(truth.sum()), source_truth_pixels=int(truth_source.sum()),
                     source_mask_size_hw=list(truth_source.shape), evaluation_size_hw=list(shape),
                     iu={}, cli_iu={}, gross_edits={})
            for arm in score_arms:
                r['iu'][arm] = counts(masks['original'][arm], truth)
                r['cli_iu'][arm] = counts(masks['cli'][arm], truth_cli)
                r['gross_edits'][arm] = {b: gross_edits(masks['original'][arm], masks['original'][b], truth)
                                         for b in bases}
            log.write(json.dumps(r)+'\n'); existing.append(r)
    grouped = defaultdict(list)
    for row in existing:
        grouped[row['dataset']].append(row)
    report = dict(split_role=config['split_role'], n=len(existing), datasets={}, methods_scored=score_arms,
                  primary_metric='original loader query size', statistical_look_count=1,
                  edit_order=['add_TP', 'add_FP', 'delete_TP', 'delete_FP'],
                  prediction_seal_sha256=file_hash(a.out/'sealed.json'))
    for name, items in grouped.items():
        expected = None
        if config['split_role'] == 'official' or config['prepared_protocol'] and config['prepared_protocol'].get('state') == 'FRESH_MANIFEST_FROZEN':
            expected = {key.split('/')[1]: meta['expected_class_ids']
                        for key, meta in config['prepared_protocol']['folds'].items()
                        if key.startswith(name+'/')}
        s = summarize(items, baselines=bases,
                      repetitions=1 if config['split_role'] == 'official' else config.get('bootstrap_repetitions', 100000),
                      expected_classes=expected, unit=config.get('bootstrap_unit', 'episode'))
        if config.get('bootstrap_unit') == 'query_photo':
            s['episode_bootstrap_appendix'] = summarize(items, baselines=bases, repetitions=10000,
                                                       expected_classes=expected)['paired']
        cli = [dict(r, iu=r['cli_iu']) for r in items]
        s['cli_miou'] = summarize(cli, baselines=('foris.crf',), repetitions=1,
                                  expected_classes=expected)['miou']
        s['gross_edits'] = {arm: {base: np.sum([r['gross_edits'][arm][base] for r in items], axis=0).tolist()
                                 for base in bases} for arm in score_arms}
        timings = [index[r['episode_id']] for r in items]
        s['timing'] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p95=float(np.quantile(v, .95)))
                       for k in ('foris_seconds', 'B0_inference_upper_bound_seconds', 'combined_pipeline_seconds')
                       for v in [[t[k] for t in timings]]}
        report['datasets'][name] = s
    write(a.out/'report.json', report)
    print(json.dumps({n: d['miou'] for n, d in report['datasets'].items()}), flush=True)


def score(a):
    with locked_run(a.out):
        _score(a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'prepare-dev', 'overlap', 'evaluate', 'score'])
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    p.add_argument('--datasets', default='coco,lvis,pascal_part,paco_part,suim')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--dev-manifest', type=Path)
    p.add_argument('--fold', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--split', choices=['smoke', 'dev', 'val', 'explore', 'official', 'confirm'], default='smoke')
    p.add_argument('--encoder-device', choices=['mps', 'cpu'], default='mps')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-score', action='store_true')
    a = p.parse_args()
    a.assets, a.out = a.assets.resolve(), a.out.resolve()
    a.datasets = a.datasets.split(',')
    from ics.official_data import FOLDS
    if set(a.datasets)-set(FOLDS):
        p.error('Unknown dataset')
    if a.mode == 'evaluate' and a.manifest is None:
        p.error('evaluate requires a frozen --manifest')
    if a.manifest is not None:
        a.manifest = a.manifest.resolve()
    if a.dev_manifest is not None:
        a.dev_manifest = a.dev_manifest.resolve()
    if a.mode == 'prepare':
        prepare(a)
    elif a.mode == 'prepare-dev':
        prepare_dev(a)
    elif a.mode == 'overlap':
        overlap(a)
    elif a.mode == 'score':
        score(a)
    else:
        infer(a)
        if not a.no_score:
            score(a)


if __name__ == '__main__':
    main()
