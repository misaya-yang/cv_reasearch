#!/usr/bin/env python3
"""Fixed RGB-texture inference/score, 4CPU/8GB ceiling, no new encoder.

Manifest binds original reference/query RGB, full reference binary mask and
existing q/r/score caches for locked MEAN. Query annotations enter score only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PRIMARY = 'reference_texture'
ARMS = dict(field=PRIMARY, color_histogram_control='texture_color.control',
            variance_control='texture_variance.control', peak_control='texture_peak.control',
            zero_control='mean.control', standalone_control='texture_standalone.control')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def initialize(threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def episode(row, out, memory_limit):
    import numpy as np
    from PIL import Image
    from ics.experiment import load_inputs, sha
    from ics.methods.huber_graph import make_mean_inputs
    from ics.methods.prepared_cpu_bundle import mean_base
    from ics.methods.reference_texture import predict
    started, cpu = time.perf_counter(), time.process_time()
    (q, r, cov, score), source = load_inputs('/', row)
    base_started = time.perf_counter()
    if not np.asarray(cov).any():
        base, base_info, graph_info = np.zeros((64, 64), np.float32), dict(empty_reference=True), {}
    else:
        graph, graph_info = make_mean_inputs(q, r, cov, score)
        base, base_info = mean_base(graph, graph_info.get('graph_storage_dtype', 'float64'))
    base_seconds = time.perf_counter()-base_started
    rgb_started = time.perf_counter()
    input_hashes = {key: sha(row[key]) for key in ('reference_rgb', 'reference_mask', 'query_rgb')}
    with Image.open(row['reference_rgb']) as image:
        reference = np.asarray(image.convert('RGB')).copy()
    with Image.open(row['query_rgb']) as image:
        query = np.asarray(image.convert('RGB')).copy()
    with Image.open(row['reference_mask']) as image:
        mask = np.asarray(image).copy()
    if reference.shape[:2] != tuple(row['support_image_hw']) or query.shape[:2] != tuple(row['query_image_hw']):
        raise ValueError('Original RGB H/W binding mismatch')
    rgb_read_seconds = time.perf_counter()-rgb_started
    result = predict(reference, mask, query, base, original_shape=tuple(row['query_image_hw']))
    if any(sha(row[key]) != value for key, value in input_hashes.items()):
        raise ValueError('Bound original RGB/reference mask changed during inference')
    out = Path(out); occurrence = row['occurrence_id']
    pp, fp, op = (out/name/(occurrence+'.npz') for name in ('predictions', 'fields', 'original_predictions'))
    np.savez_compressed(pp, **{ARMS[key]: np.packbits(value) for key, value in result['masks_work'].items()})
    np.savez_compressed(op, **{ARMS[key]: np.packbits(value) for key, value in result['masks_original'].items()})
    np.savez_compressed(fp, **{arm: result[key] for key, arm in ARMS.items()})
    peak = rss()
    if peak > memory_limit:
        raise MemoryError('Owned worker lifetime peak exceeds declared worker memory share')
    receipt = dict(occurrence_id=occurrence, source_key=row['key'], inputs=source,
                   original_rgb_and_full_reference_mask_hashes=input_hashes,
                   input_keys_read=['q', 'r', 'cov', 'score', 'original_reference_RGB',
                                    'complete_reference_binary_mask', 'original_query_RGB'],
                   base=base_info, pregraph=graph_info, mean_recompute_seconds=base_seconds,
                   rgb_read_seconds=rgb_read_seconds, method=result['info'],
                   prediction_sha256=sha(pp), field_sha256=sha(fp), original_prediction_sha256=sha(op),
                   wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
                   process_peak_rss_bytes=peak, query_gt_opened=False, new_encoder_forwards=0,
                   independent_candidate_methods=1, pid=os.getpid())
    dump(out/'receipts'/(occurrence+'.json'), receipt)
    return receipt


def infer(args):
    from ics.experiment import sha
    from ics.methods.reference_texture import Config
    if not 1 <= args.workers <= 2 or not 1 <= args.threads <= 2 or args.workers*args.threads > 4:
        raise ValueError('At most 2workers x2threads, 4CPU')
    if not 0 < args.memory_gb <= 8:
        raise ValueError('At most 8 decimal GB memory')
    source = json.loads(args.manifest.read_text())
    source = source if isinstance(source, list) else source['episodes']
    if len(source) != args.expected:
        raise ValueError('Manifest must match predeclared count; no hidden ROI/GT subset selection')
    inference, evaluation = [], []
    required = ('c', 'fold', 'support', 'query', 'feature_export', 'packet_export', 'reference_rgb',
                'reference_mask', 'query_rgb', 'support_image_hw', 'query_image_hw', 'query_annotation')
    for i, original in enumerate(source):
        row = dict(original)
        if any(key not in row for key in required):
            raise ValueError('Missing explicitly bound manifest input')
        row['occurrence_id'] = f'{i:06d}'
        row.setdefault('key', f"{row['fold']}_{row.get('e', i)}_{row['c']}")
        for key in ('feature_export', 'packet_export', 'reference_rgb', 'reference_mask', 'query_rgb', 'query_annotation'):
            path = Path(row[key])
            row[key] = str((path if path.is_absolute() else args.root/path).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        for key in ('support_image_hw', 'query_image_hw'):
            if len(row[key]) != 2 or any(int(v) != v or v < 1 for v in row[key]):
                raise ValueError('Positive original RGB H/W metadata required')
        for spec in row.get('evaluation_controls', {}).values():
            path = Path(spec['path'])
            spec['path'] = str((path if path.is_absolute() else args.root/path).resolve())
            spec['sha256'] = sha(spec['path'])
        # Annotation path is evaluation-only; no query bytes are opened here.
        row['query_foreground_value'] = int(row['c'])+1
        evaluation.append(row)
        inference.append({key: row[key] for key in ('occurrence_id', 'key', 'feature_export', 'packet_export',
                            'reference_rgb', 'reference_mask', 'query_rgb', 'support_image_hw', 'query_image_hw')})
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'fields', 'original_predictions', 'receipts'):
        (args.out/name).mkdir()
    paths = (Path(__file__), ROOT/'src/ics/methods/reference_texture.py', ROOT/'src/ics/experiment.py',
             ROOT/'src/ics/methods/huber_graph.py', ROOT/'src/ics/methods/prepared_cpu_bundle.py',
             ROOT/'src/ics/methods/rcg.py', ROOT/'scripts/run_cpu_feature_candidates.py')
    config = dict(root=str(args.root.resolve()), backend='reference_rgb_texture', primary=PRIMARY,
                  method=asdict(Config()), workers=args.workers, threads=args.threads,
                  cpu_budget=args.workers*args.threads, memory_budget_bytes=int(args.memory_gb*1e9),
                  exposure='fixed first4 reused public600; activity/cost screen, not independent confirmation',
                  source_manifest_sha256=sha(args.manifest), new_encoder_forwards=0, independent_methods=1,
                  novelty_claim=False, code_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
                  reference_mask_contract='original complete supplied binary mask, not upsampled cov64',
                  texture_contract='radial SUM power fractions; strongest single-band peak index+fraction control')
    dump(args.out/'config.json', config); dump(args.out/'inference_manifest.json', inference)
    dump(args.out/'evaluation_manifest.json', evaluation)
    began = time.perf_counter(); receipts, peaks = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(args.threads,)) as pool:
        futures = {pool.submit(episode, row, str(args.out), int(args.memory_gb*1e9/args.workers)): row for row in inference}
        for future in as_completed(futures):
            record = future.result(); occurrence = record['occurrence_id']
            receipts[occurrence] = sha(args.out/'receipts'/(occurrence+'.json'))
            peaks[record['pid']] = max(peaks.get(record['pid'], 0), record['process_peak_rss_bytes'])
            print(json.dumps(dict(done=len(receipts), total=len(inference), occurrence_id=occurrence,
                                  wall_seconds=record['wall_seconds'], mean_seconds=record['mean_recompute_seconds'],
                                  texture_and_render_seconds=record['method']['predict_with_all_renderers_seconds'])), flush=True)
    peak_owned = sum(peaks.values())+rss()
    if peak_owned > int(args.memory_gb*1e9):
        raise MemoryError('Conservative sum of owned process peaks exceeded declared total memory')
    seal = dict(state='ALL_PREDICTIONS_SEALED', n=len(receipts), receipts=receipts,
                elapsed_seconds=time.perf_counter()-began, peak_owned_rss_bytes=peak_owned,
                config_sha256=sha(args.out/'config.json'),
                inference_manifest_sha256=sha(args.out/'inference_manifest.json'),
                evaluation_manifest_sha256=sha(args.out/'evaluation_manifest.json'),
                query_gt_opened=False, new_encoder_forwards=0, independent_methods=1)
    dump(args.out/'sealed.json', seal)


def score(args):
    # Legacy 1024 scorer already enforces metadata/receipt/prediction seals.
    from run_cpu_feature_candidates import score as score_work
    score_work(args)
    import numpy as np
    from PIL import Image
    from ics.experiment import sha, unpack, summarize
    from ics.methods.reference_texture import render
    run = args.out
    work_report_path = run/'score/report.json'
    work_report = json.loads(work_report_path.read_text())
    work_report.update(independent_methods=1, scientific_originality_claim=False,
                       activity_cost_screen=True, independent_confirmation=False,
                       interpretation='Fixed first4 complete activity/cost screen, not a population effect or originality result')
    dump(work_report_path, work_report)
    rows = json.loads((run/'evaluation_manifest.json').read_text())
    arrays, corrections, details = {}, {}, []
    for row in rows:
        occurrence = row['occurrence_id']
        receipt = json.loads((run/'receipts'/(occurrence+'.json')).read_text())
        original = run/'original_predictions'/(occurrence+'.npz')
        if sha(original) != receipt['original_prediction_sha256']:
            raise ValueError('Original prediction changed after seal')
        with Image.open(row['query_annotation']) as image:
            truth = np.asarray(image) == row['query_foreground_value']
        if truth.shape != tuple(row['query_image_hw']):
            raise ValueError('Original GT/HW mismatch')
        with np.load(row['packet_export'], allow_pickle=False) as packet:
            _, native = render(unpack(packet['native']).astype(np.float32), tuple(row['query_image_hw']))
        with np.load(original, allow_pickle=False) as packet:
            masks = {key: np.unpackbits(packet[key])[:truth.size].reshape(truth.shape).astype(bool) for key in packet.files}
        masks['native'] = native
        for key, mask in masks.items():
            intersection, union = int((mask & truth).sum()), int((mask | truth).sum())
            arrays.setdefault(key, []).append([intersection, union])
            add, delete = mask & ~native, native & ~mask
            record = dict(key=occurrence, c=row['c'], fold=row['fold'], batch='fixed_smoke4',
                          add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                          delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
            corrections.setdefault(key, []).append(record)
            details.append(dict(record, arm=key, intersection=intersection, union=union))
    report = summarize(rows, {key: np.asarray(value, dtype=np.int64) for key, value in arrays.items()}, corrections)
    report.update(primary=PRIMARY, original_resolution=True, independent_confirmation=False,
                  exposure='four fixed reused episodes; small activity/cost screen', independent_methods=1,
                  predictions_sealed_before_scoring=True, new_encoder_forwards=0,
                  query_annotation_sha256={row['occurrence_id']: sha(row['query_annotation']) for row in rows},
                  native_variant='cached native1024 binary -> common bilinear original >.5',
                  interpretation='No population-effect conclusion or scientific originality claim from four cases')
    dump(run/'score'/'original_report.json', report)
    dump(run/'score'/'original_episode_metrics.json', details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--expected', type=int, default=4)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--memory-gb', type=float, default=8)
    p = sub.add_parser('score'); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'infer':
        infer(args)
    else:
        initialize(1); score(args)


if __name__ == '__main__':
    main()
