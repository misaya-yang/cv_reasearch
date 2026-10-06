#!/usr/bin/env python3
"""CPU-only cached-feature inference, followed by separately invoked scoring.

Example (paths must be bound to the new instance):
  python scripts/run_cpu_feature_candidates.py infer --manifest rows.json \
    --root /workspace --out run_v1 --workers 6 --threads 4 --memory-gb 60
  python scripts/run_cpu_feature_candidates.py score --out run_v1

Manifest rows preserve every occurrence. Required: c, fold, support, query,
feature_export, packet_export. Optional base_field_export/base_field_key reuse
an existing continuous MEAN/RCG field. Additional complete baselines can be
supplied through evaluation_controls={name:{path:...,key:...}}; they are opened
only by score. A run cannot resume into or overwrite an existing output folder.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def initialize_threads(threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss * 1024)


def linux_rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def one_episode(row, run, config):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.experiment import load_inputs, render, sha
    from ics.methods import reference_occupancy as method

    start, cpu_start = time.perf_counter(), time.process_time()
    inputs, source = load_inputs(config['root'], row)
    q, r, cov, score = inputs
    if row.get('base_field_export'):
        path = Path(row['base_field_export'])
        with np.load(path, allow_pickle=False) as data:
            base = data[row['base_field_key']].astype(np.float32)
        base_info = dict(source='existing_field', path=str(path), sha256=sha(path),
                         key=row['base_field_key'])
    elif float(np.ptp(score)) < 1e-9:
        base = np.zeros_like(score, dtype=np.float32)
        base_info = dict(source='source_host_constant_field_empty_rule')
    elif config['base'] == 'mean':
        from ics.methods.mean_graph import predict
        base, base_info = predict(q, r, cov, score)
        base_info['source'] = 'recomputed_locked_MEAN_a0.25_l16'
    else:
        from ics.methods.rcg import predict
        base, base_info = predict(q, r, cov, score)
        base_info['source'] = 'recomputed_locked_RCG_a0.5_l16'
    if base.shape != score.shape or not np.isfinite(base).all():
        raise ValueError('Invalid cached/generated baseline field')
    base_name = 'base.' + config['base'] + '.control'
    masks = {base_name: render(base)}
    fields = {base_name: base}
    audits = {}
    prepared = method.prepare(np.asarray(q), np.asarray(r), cov, base.shape,
                              method.Config(**config['method']))
    # Rankings are computed once per episode. Same-budget controls use no GT.
    upsampled = F.interpolate(torch.from_numpy(base.copy())[None, None], (1024, 1024),
                              mode='bilinear', align_corners=False)[0, 0].numpy().ravel()
    base_mask = masks[base_name].ravel()
    fg = np.flatnonzero(base_mask)
    bg = np.flatnonzero(~base_mask)
    delete_order = fg[np.argsort(upsampled[fg], kind='stable')]
    add_order = bg[np.argsort(-upsampled[bg], kind='stable')]
    for strength in config['strengths']:
        tag = format(strength, '.6g')
        for mode in ('regional', 'pointwise'):
            name = f'{mode}.l{tag}' + ('.control' if mode == 'pointwise' else '')
            field, info = method.apply(base, prepared, strength, mode)
            fields[name], audits[name] = field, info
            masks[name] = render(field)
        primary = masks[f'regional.l{tag}'].ravel()
        n_add = int((primary & ~base_mask).sum())
        n_delete = int((~primary & base_mask).sum())
        control = base_mask.copy()
        control[delete_order[:n_delete]] = False
        control[add_order[:n_add]] = True
        masks[f'count_matched.l{tag}.control'] = control.reshape(1024, 1024)
        audits[f'count_matched.l{tag}.control'] = dict(add_pixels=n_add, delete_pixels=n_delete,
                                                      budget_source=f'regional.l{tag}')
    occurrence = row['occurrence_id']
    pred_path, field_path = run / 'predictions' / (occurrence + '.npz'), run / 'fields' / (occurrence + '.npz')
    np.savez_compressed(pred_path, **{name: np.packbits(mask) for name, mask in masks.items()})
    np.savez_compressed(field_path, **fields)
    receipt = dict(occurrence_id=occurrence, source_key=row['key'], inputs=source,
                   base=base_info, method=prepared['info'], arms=audits,
                   prediction_sha256=sha(pred_path), field_sha256=sha(field_path),
                   wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start,
                   process_peak_rss_bytes=peak_rss_bytes(), query_gt_opened=False,
                   new_encoder_forwards=0, pid=os.getpid())
    write(run / 'receipts' / (occurrence + '.json'), receipt)
    return receipt


def worker(jobs, results, run, config):
    initialize_threads(config['threads'])
    import torch
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    if config.get('backend','occupancy') == 'prepared':
        from ics.methods.prepared_cpu_bundle import one_episode as run_episode
    else:
        run_episode = one_episode
    while True:
        row = jobs.get()
        if row is None:
            return
        try:
            receipt = run_episode(row, Path(run), config)
            results.put(dict(ok=True, occurrence_id=row['occurrence_id'],
                             wall_seconds=receipt['wall_seconds'],
                             peak_rss_bytes=receipt['process_peak_rss_bytes']))
        except Exception as error:
            results.put(dict(ok=False, occurrence_id=row['occurrence_id'],
                             error_type=type(error).__name__, detail=str(error),
                             traceback=traceback.format_exc()))


def infer(args):
    from ics.experiment import sha
    from ics.methods.reference_occupancy import Config
    from dataclasses import asdict
    backend = getattr(args,'backend','occupancy')
    prepared_methods = list(getattr(args,'prepared_methods',[]))
    if backend == 'prepared':
        from ics.methods.prepared_cpu_bundle import METHODS
        if (not prepared_methods or len(set(prepared_methods)) != len(prepared_methods)
                or any(name not in METHODS for name in prepared_methods)):
            raise ValueError('Distinct recognized prepared methods required')
        if args.base != 'mean' or args.strengths != [1.0] or args.primary_strength != 1.0:
            raise ValueError('Prepared contracts use locked MEAN and their fixed parameters, not occupancy strengths')
        if args.primary_method not in prepared_methods:
            raise ValueError('Predeclare --primary-method among the requested prepared methods')
    if min(args.workers, args.threads, args.cpu_budget) < 1:
        raise ValueError('Positive CPU and worker counts required')
    if args.workers * args.threads > min(args.cpu_budget, 30):
        raise ValueError('workers * threads exceeds the declared total CPU budget')
    if args.memory_gb <= 0 or args.memory_gb > 60:
        raise ValueError('Memory budget must be positive and at most 60 decimal GB')
    if hasattr(os, 'sched_getaffinity') and args.workers * args.threads > len(os.sched_getaffinity(0)):
        raise ValueError('Requested CPU parallelism exceeds the current process affinity')
    # Quota, not host CPU count, limits a container's available CPU time.
    quota_file = Path('/sys/fs/cgroup/cpu.max')
    if quota_file.exists():
        quota, period = quota_file.read_text().split()
        if quota != 'max' and args.workers * args.threads > int(quota) / int(period) + 1e-9:
            raise ValueError('Requested CPU parallelism exceeds the actual cgroup quota')
    effective_memory = int(args.memory_gb * 1e9)
    memory_file = Path('/sys/fs/cgroup/memory.max')
    if memory_file.exists():
        limit = memory_file.read_text().strip()
        if limit != 'max':
            effective_memory = min(effective_memory, int(int(limit) * .9))
    source = json.loads(args.manifest.read_text())
    rows = source if isinstance(source, list) else source['episodes']
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty manifest')
    evaluation, inference = [], []
    for index, original in enumerate(rows):
        row = dict(original)
        for key in ('c', 'fold', 'support', 'query', 'feature_export', 'packet_export'):
            if key not in row:
                raise ValueError('Missing manifest field: ' + key)
        row['c'], row['fold'] = int(row['c']), int(row['fold'])
        row['occurrence_id'] = f'{index:06d}'
        row.setdefault('key', f"{row['fold']}_{row.get('e', index)}_{row['c']}")
        path_keys = ('feature_export','packet_export','base_field_export')
        if backend == 'prepared' and 'color_bottleneck' in prepared_methods:
            path_keys += ('query_rgb_export','query_image_export')
        for key in path_keys:
            if row.get(key):
                row[key] = str(resolve(args.root, row[key]))
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
        row.setdefault('base_field_key', args.base_key)
        if backend == 'prepared' and 'color_bottleneck' in prepared_methods:
            if not row.get('query_rgb_export') and not row.get('query_image_export'):
                raise ValueError('Color method requires bound query_rgb_export or query_image_export for every row')
        if backend == 'prepared' and 'reference_shape' in prepared_methods:
            for key in ('support_image_hw','query_image_hw'):
                value=row.get(key)
                if (not isinstance(value,(list,tuple)) or len(value)!=2
                        or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 or int(v)!=v for v in value)):
                    raise ValueError('Shape method requires original positive integer H/W metadata: '+key)
                row[key]=[int(v) for v in value]
        for spec in row.get('evaluation_controls', {}).values():
            spec['path'] = str(resolve(args.root, spec['path']))
            spec['sha256'] = sha(spec['path'])
        evaluation.append(row)
        inference.append({key: row[key] for key in ('occurrence_id', 'key', 'feature_export',
                          'packet_export', 'base_field_export', 'base_field_key',
                          'query_rgb_export', 'query_image_export','support_image_hw','query_image_hw') if key in row})
    if any(not math.isfinite(s) or s < 0 for s in args.strengths) or len(set(args.strengths)) != len(args.strengths):
        raise ValueError('Distinct nonnegative strengths required')
    if args.primary_strength not in args.strengths:
        raise ValueError('Primary strength must be among the predeclared strengths')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('predictions', 'fields', 'receipts'):
        (args.out / name).mkdir()
    config = dict(root=str(args.root.resolve()), base=args.base, method=asdict(Config()),
                  strengths=args.strengths, workers=args.workers, threads=args.threads,
                  cpu_budget=args.cpu_budget, memory_budget_bytes=effective_memory,
                  exposure=args.exposure, source_manifest_sha256=sha(args.manifest),
                  primary=f'regional.l{format(args.primary_strength, ".6g")}',
                  complete_mask_size=[1024, 1024], encoder_forwards=0,
                  code_sha256={str(path.relative_to(REPO)): sha(path) for path in (
                      Path(__file__), REPO / 'src/ics/methods/reference_occupancy.py',
                      REPO / 'src/ics/methods/mean_graph.py', REPO / 'src/ics/methods/rcg.py',
                      REPO / 'src/ics/experiment.py')})
    if backend == 'prepared':
        from ics.methods.reference_adjacency import Config as AdjacencyConfig
        from ics.methods.huber_graph import Config as HuberConfig
        from ics.methods.color_bottleneck import Config as ColorConfig
        from ics.methods.reference_constellation import Config as ConstellationConfig
        from ics.methods.reference_shape import Config as ShapeConfig
        from ics.methods.reference_covariance import Config as CovarianceConfig
        from ics.methods.query_recurrence import Config as RecurrenceConfig
        config.update(backend=backend,prepared_methods=prepared_methods,primary=args.primary_method,
                      primary_methods=prepared_methods,independent_methods=len(prepared_methods),
                      method=None,strengths=None,
                      method_configs={name:asdict(cls()) for name,cls in (
                          ('adjacency',AdjacencyConfig),('huber',HuberConfig),
                          ('color_bottleneck',ColorConfig),('constellation',ConstellationConfig),
                          ('reference_shape',ShapeConfig),('reference_covariance',CovarianceConfig),
                          ('query_recurrence',RecurrenceConfig))
                          if name in prepared_methods})
        for name in ('prepared_cpu_bundle','reference_adjacency','huber_graph','color_bottleneck','reference_constellation',
                     'reference_shape','reference_covariance','query_recurrence'):
            path=REPO/'src/ics/methods'/f'{name}.py'
            config['code_sha256'][str(path.relative_to(REPO))]=sha(path)
    write(args.out / 'config.json', config)
    write(args.out / 'inference_manifest.json', inference)
    write(args.out / 'evaluation_manifest.json', evaluation)
    context = mp.get_context('spawn')
    jobs, results = context.Queue(maxsize=args.workers), context.Queue()
    processes = [context.Process(target=worker, args=(jobs, results, str(args.out), config))
                 for _ in range(min(args.workers, len(inference)))]
    start = time.perf_counter()
    for process in processes:
        process.start()
    write(args.out / 'running.json', dict(state='INFERENCE_RUNNING', coordinator_pid=os.getpid(),
                                        worker_pids=[p.pid for p in processes]))
    sent, done, peak_rss = 0, 0, 0
    failure = None
    try:
        for _ in processes:
            jobs.put(inference[sent]); sent += 1
        while done < len(inference):
            rss = linux_rss(os.getpid()) + sum(linux_rss(p.pid) for p in processes if p.is_alive())
            peak_rss = max(peak_rss, rss)
            if rss > config['memory_budget_bytes']:
                # Only process objects created by this invocation are signalled.
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                raise MemoryError('Owned worker RSS exceeded the declared memory budget')
            try:
                result = results.get(timeout=.5)
            except queue.Empty:
                dead = [p.pid for p in processes if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'Worker exited before all results arrived: {dead}')
                continue
            if not result['ok']:
                failure = result
                raise RuntimeError('Episode failed: ' + result['occurrence_id'] + ': ' + result['detail'])
            done += 1
            print(json.dumps(dict(done=done, total=len(inference), elapsed_seconds=time.perf_counter() - start,
                                  last_episode_seconds=result['wall_seconds'], owned_rss_bytes=rss)), flush=True)
            if sent < len(inference):
                jobs.put(inference[sent]); sent += 1
        receipt_paths = [args.out / 'receipts' / (r['occurrence_id'] + '.json') for r in inference]
        seal = dict(state='ALL_PREDICTIONS_SEALED', n=len(inference),
                    config_sha256=sha(args.out / 'config.json'),
                    inference_manifest_sha256=sha(args.out / 'inference_manifest.json'),
                    evaluation_manifest_sha256=sha(args.out / 'evaluation_manifest.json'),
                    receipts={p.stem: sha(p) for p in receipt_paths},
                    elapsed_seconds=time.perf_counter() - start, peak_owned_rss_bytes=peak_rss,
                    query_gt_opened=False, encoder_forwards=0)
        write(args.out / 'sealed.json', seal)
        write(args.out / 'running.json', dict(state='INFERENCE_COMPLETED', n=done))
    except BaseException as error:
        write(args.out / 'failure.json', dict(state='INFERENCE_INCOMPLETE', n_completed=done,
                                            error_type=type(error).__name__, detail=str(error), episode=failure))
        raise
    finally:
        for process in processes:
            if process.is_alive():
                try:
                    jobs.put(None, timeout=.5)
                except queue.Full:
                    break
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join()


def score(args):
    import numpy as np
    from ics.experiment import sha, summarize, unpack
    run = args.out
    seal = json.loads((run / 'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED':
        raise ValueError('A complete prediction seal is required before scoring')
    for name in ('config', 'inference_manifest', 'evaluation_manifest'):
        if sha(run / (name + '.json')) != seal[name + '_sha256']:
            raise ValueError('Sealed metadata changed: ' + name)
    config = json.loads((run / 'config.json').read_text())
    rows = json.loads((run / 'evaluation_manifest.json').read_text())
    arrays, corrections, details, expected_arms = {}, {}, [], None
    metric_rows = []
    for row in rows:
        occurrence = row['occurrence_id']
        rp, pp = run / 'receipts' / (occurrence + '.json'), run / 'predictions' / (occurrence + '.npz')
        if sha(rp) != seal['receipts'][occurrence]:
            raise ValueError('Receipt changed')
        receipt = json.loads(rp.read_text())
        if (sha(pp) != receipt['prediction_sha256']
                or sha(run / 'fields' / (occurrence + '.npz')) != receipt['field_sha256']
                or sha(row['packet_export']) != receipt['inputs']['packet_sha256']):
            raise ValueError('Prediction or evaluation packet changed')
        with np.load(row['packet_export'], allow_pickle=False) as packet:
            truth, native = unpack(packet['truth']), unpack(packet['native'])
        with np.load(pp, allow_pickle=False) as predictions:
            masks = {'native': native, **{k: unpack(predictions[k]) for k in predictions.files}}
        for name, spec in row.get('evaluation_controls', {}).items():
            if name in masks:
                raise ValueError('An evaluation control would overwrite an existing arm: ' + name)
            if sha(spec['path']) != spec['sha256']:
                raise ValueError('Sealed evaluation control changed: ' + name)
            with np.load(spec['path'], allow_pickle=False) as data:
                masks[name] = unpack(data[spec['key']])
        if expected_arms is None:
            expected_arms = set(masks)
        if set(masks) != expected_arms:
            raise ValueError('Unpaired evaluation controls')
        for arm, mask in masks.items():
            intersection, union = int((mask & truth).sum()), int((mask | truth).sum())
            arrays.setdefault(arm, []).append([intersection, union])
            add, delete = mask & ~native, native & ~mask
            corr = dict(key=occurrence, c=row['c'], fold=row['fold'], batch=str(row.get('batch', 'unspecified')),
                        add_TP=int((add & truth).sum()), delete_FP=int((delete & ~truth).sum()),
                        delete_TP=int((delete & truth).sum()), add_FP=int((add & ~truth).sum()))
            corrections.setdefault(arm, []).append(corr)
            details.append(dict(corr, arm=arm, intersection=intersection, union=union))
        # The shared scorer historically compares basenames. Give it unique IDs
        # for complete canonical photo identities so shard basenames cannot alias.
        metric_rows.append(dict(row))
    identity_to_id = {}
    for row in metric_rows:
        for role in ('support', 'query'):
            identity = str(row.get(role + '_photo_id', row[role]))
            identity_to_id.setdefault(identity, len(identity_to_id))
            row[role] = f'canonical_photo_{identity_to_id[identity]}'
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in arrays.items()}
    report, draws = summarize(metric_rows, arrays, corrections)
    report.update(exposure=config['exposure'], primary=config['primary'],
                  predictions_sealed_before_scoring=True, new_encoder_forwards=0,
                  inference_wall_seconds=seal['elapsed_seconds'],
                  inference_peak_owned_rss_bytes=seal['peak_owned_rss_bytes'],
                  independent_confirmation=False,
                  interpretation='Candidate efficacy and originality are unestablished until these comparisons are assessed.')
    if config.get('backend') == 'prepared':
        report.update(primary_methods=config['primary_methods'],independent_methods=config['independent_methods'],
                      execution_backend='shared prepared-candidate inference; not another method')
    output = run / 'score'
    output.mkdir(exist_ok=False)
    write(output / 'report.json', report)
    write(output / 'episode_metrics.json', details)
    np.savez_compressed(output / 'counts.npz', **arrays)
    np.save(output / 'bootstrap_photo_draws.npy', draws)
    lines = ['# Cached-feature candidate results', '', '| Arm | Class-summed mIoU |', '|---|---:|']
    lines += [f'| {arm} | {value:.6f} |' for arm, value in report['scores'].items()]
    lines += ['', 'Primary: ' + config['primary'], 'Exposure: ' + config['exposure'], '',
              '| Primary versus | Gain, pp | Paired 95% CI |', '|---|---:|---|']
    for base, item in report['contrasts'][config['primary']].items():
        lines.append(f'| {base} | {item["gain"]:+.6f} | {item["ci95"]} |')
    if config.get('backend') == 'prepared':
        lines += ['', '| Predeclared candidate | Comparison | Gain, pp | Paired 95% CI |', '|---|---|---:|---|']
        for arm in config['primary_methods']:
            for base,item in report['contrasts'][arm].items():
                lines.append(f'| {arm} | {base} | {item["gain"]:+.6f} | {item["ci95"]} |')
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(scores=report['scores'], primary=report['contrasts'][config['primary']])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    inf = sub.add_parser('infer')
    inf.add_argument('--manifest', type=Path, required=True)
    inf.add_argument('--root', type=Path, required=True)
    inf.add_argument('--out', type=Path, required=True)
    inf.add_argument('--workers', type=int, default=6)
    inf.add_argument('--threads', type=int, default=4)
    inf.add_argument('--cpu-budget', type=int, default=30)
    inf.add_argument('--memory-gb', type=float, default=60)
    inf.add_argument('--base', choices=['mean', 'rcg'], default='mean')
    inf.add_argument('--backend', choices=['occupancy','prepared'], default='occupancy')
    inf.add_argument('--prepared-methods', nargs='+', choices=['adjacency','huber','color_bottleneck','constellation',
                                                            'reference_shape','reference_covariance','query_recurrence'],
                     default=['adjacency','huber','color_bottleneck','constellation'])
    inf.add_argument('--primary-method', choices=['adjacency','huber','color_bottleneck','constellation',
                                                 'reference_shape','reference_covariance','query_recurrence'])
    inf.add_argument('--base-key', default='mean.control')
    inf.add_argument('--strengths', nargs='+', type=float, default=[1.0])
    inf.add_argument('--primary-strength', type=float, default=1.0)
    inf.add_argument('--limit', type=int)
    inf.add_argument('--exposure', default='reused development cohort; not independent confirmation')
    scoring = sub.add_parser('score')
    scoring.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    initialize_threads(args.threads if args.command == 'infer' else 1)
    if args.command == 'infer':
        infer(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
