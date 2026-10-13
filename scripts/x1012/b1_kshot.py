"""B1: nested cache-only K-shot, sealed natural masks, then threshold diagnostics.

The FoRIS control always uses this query's exact first reference. Missing controls
are replayed from O/24 into B1's own directory; no encoder is constructed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import builtins
import io
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
from raw_feature_cache import tensor_hash

OUT = lab.ASSETS / 'a/x1012_B1'
P0 = lab.ASSETS / 'a/x1012_P0'
KS = (1, 2, 3, 5)
TEMPERATURE = .05


@contextmanager
def legal_inference(binding, reference_masks=()):
    """Query annotations are denied; every explicit K-shot R annotation is legal.

    Content-addressed constant masks can be the same path for different images.
    That file remains legal only when explicitly supplied as an R annotation.
    """
    from PIL import Image
    row = binding['row']
    allowed = {Path(row['reference_mask_path']).resolve()}
    allowed.update(Path(p).resolve() for p in reference_masks)
    forbidden = {Path(row[k]).resolve() for k in ('query_mask_path', 'query_ignore_mask_path') if row.get(k)} - allowed
    saved = builtins.open, io.open, os.open, Image.open

    def wrap(fn):
        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() in forbidden:
                raise RuntimeError('Query GT forbidden during B-task prediction production')
            return fn(path, *args, **kwargs)
        return guarded

    builtins.open, io.open, os.open, Image.open = map(wrap, saved)
    try:
        with torch.inference_mode():
            yield
    finally:
        builtins.open, io.open, os.open, Image.open = saved


def read_raw(metadata, counters):
    """Consume a bound FP32 O/24, checking metadata, payload, and tensor SHA."""
    start = time.monotonic()
    entry = Path(metadata['entry_path'])
    assert pc.sha(entry) == metadata['entry_sha256'], 'Raw entry changed'
    info = pc.read(entry)
    branch = info['features']['O/24']
    payload = Path(metadata['payload_path'])
    assert payload == entry.parent / info['file']
    assert pc.sha(payload) == info['file_sha256'] == metadata['payload_sha256'], 'Raw payload changed'
    assert branch['shape'] == [4096, 1024] and branch['dtype'] == 'float32'
    with np.load(payload, allow_pickle=False) as z:
        raw = z['O/24'].copy()
    assert raw.shape == (4096, 1024) and raw.dtype == np.float32 and np.isfinite(raw).all()
    assert tensor_hash(raw) == branch['tensor_sha256'] == metadata['tensor_sha256']
    counters['raw_requests'] = counters.get('raw_requests', 0) + 1
    counters['logical_payload_bytes'] = counters.get('logical_payload_bytes', 0) + payload.stat().st_size
    counters['raw_io_seconds'] = counters.get('raw_io_seconds', 0.) + time.monotonic() - start
    return torch.from_numpy(raw)


def region_labels(q):
    centroids = lab.kmeans(lab.unit(q), 64, iters=6)
    return (lab.unit(q) @ centroids.T).argmax(1)


def region_mean(values, labels):
    mass = torch.bincount(labels, minlength=64).to(values.dtype).clamp_min(1)
    sums = torch.zeros(64, dtype=values.dtype, device=values.device).index_add_(0, labels, values)
    return (sums / mass)[labels]


def independent_fields(references, query, coverages, block=256, labels=None):
    """Equal-image kernel mixture with one shared centre and stable row softmax.

    Reference vectors are already positionally projected/unit-normalised. Query
    blocks bound peak temporary similarities to block x (K*4096), FP32.
    E is mean kernel mass per reference token, hence comparable across K.
    """
    assert len(references) == len(coverages) and references
    r = torch.cat(references)
    y = torch.cat(coverages)
    mu = .5 * (torch.stack([v.mean(0) for v in references]).mean(0) + query.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(query - mu)
    posterior, maxima, log_explanation = [], [], []
    for offset in range(0, len(query), block):
        sim = qc[offset:offset + block] @ rc.T
        maximum = sim.max(1).values
        kernel = torch.exp((sim - maximum[:, None]) / TEMPERATURE)
        evidence = kernel.sum(1)
        posterior.append((kernel @ y) / evidence.clamp_min(1e-30))
        maxima.append(maximum)
        log_explanation.append(evidence.clamp_min(1e-30).log() + maximum / TEMPERATURE - np.log(len(r)))
    c = torch.cat(posterior)
    smax = torch.cat(maxima)
    if labels is None:
        labels = region_labels(query)
    return {'c': c, 'IND0': region_mean(c, labels),
            'INDN': region_mean(c * lab.minmax(smax), labels),
            'sim_max': smax, 'log_mean_kernel_mass': torch.cat(log_explanation)}


def source_hashes(*extra):
    paths = (Path(__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__),
             lab.BASIS, lab.ROOT / 'scripts/raw_feature_cache.py', *extra)
    return {str(p): pc.sha(p) for p in paths}


def check_sources(config):
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'Frozen task source changed'


def verify_seal(root, state):
    """Validate every prediction/field before the first scoring-only GT read."""
    seal = pc.read(root / 'sealed.json')
    assert seal['state'] == state
    assert pc.sha(root / 'manifest.json') == seal['manifest_sha256']
    assert pc.sha(root / 'config.json') == seal['config_sha256']
    records = {r['index']: r for r in seal['records']}
    rows = pc.read(root / 'manifest.json')
    assert len(rows) == len(records) == seal['n']
    for b in rows:
        r = records[pc.key(b)]
        for name in ('prediction', 'field'):
            if name + '_path' in r:
                assert pc.sha(r[name + '_path']) == r[name + '_sha256'], 'Sealed output changed'
    return seal, rows, records


def foris_control(binding, same_pair_sources, replay_state):
    """Reuse only the identical R/Q/mask pair; otherwise replay K1 in B1."""
    signature = binding.get('pair_signature')
    source = same_pair_sources.get(signature)
    if source is not None:
        for name in ('reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash', 'query_mask_hash',
                     'reference_crop', 'query_crop'):
            assert source['row'].get(name) == binding['row'].get(name), 'K1 FoRIS requires the exact same pair'
        assert all(source['fit'][role]['payload_path'] == binding['fit'][role]['payload_path']
                   for role in ('reference_raw', 'query_raw_identity'))
    arm = binding.get('old_arms', {}).get('B')
    origin = 'binding_exact_pair'
    if arm is None and source is not None:
        arm = source.get('old_arms', {}).get('B')
        origin = 'P0_exact_pair'
        receipt = P0 / 'foris/new/records' / (pc.key(source) + '.json')
        if arm is None and receipt.exists():
            record = pc.read(receipt)
            assert record['episode_id'] == source['row']['episode_id']
            arm = {'path': record['prediction_path'], 'sha256': record['prediction_sha256'], 'cli_key': 'cli/foris.crf'}
    if arm is not None:
        if arm.get('sha256'):
            assert pc.sha(arm['path']) == arm['sha256']
        with np.load(arm['path'], allow_pickle=False) as f:
            packed = f[arm['cli_key']].copy()
        return packed, {'origin': origin, 'path': arm['path'], 'sha256': pc.sha(arm['path']),
                        'seconds': 0., 'cache_only_FoRIS_calls': 0, 'raw_cache_requests': 0, 'logical_payload_bytes': 0}
    start = time.monotonic()
    if not replay_state:
        host, adapter = pc.host_cpu()
        from ics.foris import run_foris
        from ics.official_data import load_inputs
        replay_state.update(host=host, adapter=adapter, run=run_foris, load=load_inputs)
    adapter = replay_state['adapter']
    adapter.bind(binding)
    reference, mask, query = replay_state['load'](binding['row'], lab.ASSETS)
    final, _, _, _ = replay_state['run'](replay_state['host'], reference, mask, query)
    packed = pc.pack(final.cpu().numpy())
    raw_requests = len(adapter.memo)
    payload_bytes = sum(Path(adapter.bound[key]['payload_path']).stat().st_size for key in adapter.memo)
    adapter.memo.clear()
    assert adapter.encoder is None and adapter.encoder_calls == 0
    return packed, {'origin': 'B1_own_exact_K1_cache_replay', 'seconds': time.monotonic() - start,
                    'cache_only_FoRIS_calls': 1, 'new_encoder_calls': 0,
                    'raw_cache_requests': raw_requests, 'logical_payload_bytes': payload_bytes}


def produce(args):
    pc.setup('cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if not config_path.exists():
        rows = pc.read(P0 / 'kshot.json')
        assert rows and len({pc.key(b) for b in rows}) == len(rows)
        for b in rows:
            refs = b['references']
            available = [k for k in KS if k <= len(refs)]
            assert b['available_k'] == available and available
            photos = [r['photo'] for r in refs]
            assert len(photos) == len(set(photos))
            query_photo = str((lab.ASSETS / b['row']['query_path']).resolve())
            assert query_photo not in photos
            assert all(r['dataset'] == b['dataset'] and str(r['fold']) == str(b['row']['fold']) and
                       str(r['class_id']) == str(b['row']['global_class_id']) for r in refs)
            assert refs[0]['raw']['payload_path'] == b['fit']['reference_raw']['payload_path']
        pc.write(OUT / 'manifest.json', rows)
        pc.write(config_path, {'state': 'FROZEN', 'task': 'B1', 'n': len(rows), 'cpu_threads': 4,
                              'query_block': args.query_block, 'K': KS, 'temperature': TEMPERATURE,
                              'reference_weights': 'uniform 1/K; common token-count normalisation; no separate FG/BG-area normalisation',
                              'centre': '.5*(mean_k(mean_tokens(R_k))+mean_tokens(Q))',
                              'natural_rule': 'K64 query regions; FP32 bilinear1024, minmax, strict >.5',
                              'control': 'same pair K1 FoRIS complete; cached exact pair or own CPU/CRF replay',
                              'kshot_manifest_sha256': pc.sha(P0 / 'kshot.json'),
                              'source_sha256': source_hashes(lab.ROOT / 'scripts/cached_dino.py', lab.ROOT / 'src/ics/foris.py',
                                                             lab.ROOT / 'scripts/x1012/p0_manifest.py'),
                              'new_encoder_calls': 0, 'query_GT_reads': 0})
    config = pc.read(config_path)
    check_sources(config)
    if (OUT / 'sealed.json').exists():
        verify_seal(OUT, 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED')
        print({'state': 'ALREADY_SEALED', 'task': 'B1', 'n': config['n']}, flush=True)
        return
    rows = pc.read(OUT / 'manifest.json')
    for name in ('predictions', 'fields', 'records'):
        (OUT / name).mkdir(exist_ok=True)
    explore = pc.read(P0 / 'explore.json')
    same_pair = {b['pair_signature']: b for b in explore if b.get('pair_signature')}
    start, cpu = time.monotonic(), time.process_time()
    receipts, replay_state = [], {}
    for i, b in enumerate(rows, 1):
        name = pc.key(b)
        receipt_path = OUT / 'records' / (name + '.json')
        if receipt_path.exists():
            receipt = pc.read(receipt_path)
            assert all(pc.sha(receipt[k + '_path']) == receipt[k + '_sha256'] for k in ('prediction', 'field'))
            receipts.append(receipt)
            continue
        prediction_path, field_path = OUT / 'predictions' / (name + '.npz'), OUT / 'fields' / (name + '.npz')
        assert not prediction_path.exists() and not field_path.exists(), 'Preserve incomplete task output'
        tick = time.monotonic()
        counters = {}
        refs = b['references'][:max(b['available_k'])]
        with legal_inference(b, [r['view']['mask_path'] for r in refs]):
            query = lab.debias(read_raw(b['fit']['query_raw_identity'], counters))
            references = [lab.debias(read_raw(r['raw'], counters)) for r in refs]
            coverages = [lab.coverage64(lab.mask1024(r['view']['mask_path'])).flatten() for r in refs]
            labels = region_labels(query)
            predictions, fields = {}, {}
            for k in b['available_k']:
                result = independent_fields(references[:k], query, coverages[:k], config['query_block'], labels)
                for method in ('IND0', 'INDN'):
                    arm = f'K{k}/{method}'
                    predictions[arm] = pc.pack(pc.readout(result[method]))
                    fields[arm] = result[method].numpy()
                fields[f'K{k}/log_mean_kernel_mass'] = result['log_mean_kernel_mass'].numpy()
            predictions['FoRIS'], baseline_receipt = foris_control(b, same_pair, replay_state)
        np.savez_compressed(prediction_path, **predictions)
        np.savez_compressed(field_path, **fields)
        receipt = {'index': name, 'episode_id': b['row']['episode_id'], 'dataset': b['dataset'],
                   'available_k': b['available_k'], 'reference_ids': [r['id'] for r in refs],
                   'prediction_path': str(prediction_path), 'prediction_sha256': pc.sha(prediction_path),
                   'field_path': str(field_path), 'field_sha256': pc.sha(field_path),
                   'FoRIS_control': baseline_receipt, 'seconds': time.monotonic() - tick,
                   'new_encoder_calls': 0, 'query_GT_reads': 0, **counters}
        pc.write(receipt_path, receipt)
        receipts.append(receipt)
        del references, query, coverages, labels, result, fields, predictions
        lab.free()
        if i <= 3 or i % 50 == 0:
            print({'task': 'B1', 'completed': i, 'n': len(rows), 'seconds': round(time.monotonic() - start, 1)}, flush=True)
    pc.write(OUT / 'sealed.json', {'state': 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED', 'n': len(rows), 'records': receipts,
             'manifest_sha256': pc.sha(OUT / 'manifest.json'), 'config_sha256': pc.sha(config_path),
             'invocation_wall_seconds': time.monotonic() - start, 'invocation_cpu_seconds': time.process_time() - cpu,
             'episode_seconds_sum': sum(r['seconds'] for r in receipts),
             'K1_FoRIS_seconds': sum(r['FoRIS_control']['seconds'] for r in receipts),
             'cache_only_FoRIS_calls': sum(r['FoRIS_control']['cache_only_FoRIS_calls'] for r in receipts),
             'kernel_raw_requests': sum(r['raw_requests'] for r in receipts),
             'FoRIS_raw_requests': sum(r['FoRIS_control']['raw_cache_requests'] for r in receipts),
             'raw_requests': sum(r['raw_requests'] + r['FoRIS_control']['raw_cache_requests'] for r in receipts),
             'logical_payload_bytes': sum(r['logical_payload_bytes'] + r['FoRIS_control']['logical_payload_bytes'] for r in receipts),
             'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'cpu_threads': 4, 'new_encoder_calls': 0, 'query_GT_reads': 0})


def threshold_oracle(field, gt, valid):
    """Exact *realizable* strict threshold, with complete equal-score groups."""
    score = lab.minmax(lab.up(torch.as_tensor(field).reshape(64, 64))).numpy()
    values, target = score[valid], gt[valid]
    if not target.any():
        threshold = float(values.max()) if len(values) else float(score.max())
        return score > threshold, threshold
    order = np.argsort(-values, kind='stable')
    ordered = values[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(ordered) - 1]
    intersection = np.cumsum(target[order], dtype=np.int64)[ends]
    union = ends + 1 + int(target.sum()) - intersection
    end = int(ends[np.argmax(intersection / np.maximum(union, 1))])
    threshold = float(np.nextafter(ordered[end], np.float32(-np.inf), dtype=np.float32))
    return score > threshold, threshold


def score():
    pc.setup('cpu')
    seal, rows, receipts = verify_seal(OUT, 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED')
    start = time.monotonic()
    records = []
    for i, b in enumerate(rows, 1):
        receipt = receipts[pc.key(b)]
        with np.load(receipt['prediction_path'], allow_pickle=False) as f:
            predictions = {a: pc.unpack(f[a]) for a in f.files}
        record = pr.score_record(b, predictions)
        gt, valid = pc.truth(b)
        thresholds = {}
        with np.load(receipt['field_path'], allow_pickle=False) as f:
            for k in b['available_k']:
                for method in ('IND0', 'INDN'):
                    arm = f'K{k}/{method}'
                    oracle, threshold = threshold_oracle(f[arm], gt, valid)
                    record['metrics'][arm + '/oracle_threshold'] = pr.metric(oracle, gt, valid)
                    thresholds[arm] = threshold
        record.update(available_k=b['available_k'], oracle_thresholds=thresholds)
        records.append(record)
        if i <= 3 or i % 100 == 0:
            print({'task': 'B1-score', 'completed': i, 'n': len(rows)}, flush=True)
    arms_for = lambda ks: ('FoRIS',) + tuple(f'K{k}/{method}{suffix}' for k in ks for method in ('IND0', 'INDN')
                                            for suffix in ('', '/oracle_threshold'))
    project = lambda rs, arms: [dict(r, metrics={a: r['metrics'][a] for a in arms}) for r in rs]
    notes = ['完整K-shot清单；K=1/2/3/5嵌套参考，固定温度.05，统一中心和全参考共同softmax归一化。',
             'FoRIS固定为同query、同首参考的K1完整CRF；只读精确缓存或B1独占缓存重放，0encoder。',
             'oracle_threshold读取GT但只作诊断，按完整同分组选择可实现的严格阈值；自然预测全部先seal。',
             '主表K1涵盖全部available query；附表每组K严格使用共同可用query分母，禁止跨表混样本解释K增益。']
    arms = arms_for((1,))
    primary = pr.report('B1', OUT, project(records, arms), arms, notes, extra={'production_cost': {k: v for k, v in seal.items() if k != 'records'}})
    common_tables = {}
    for k in (2, 3, 5):
        ks = tuple(v for v in KS if v <= k)
        selected = [r for r in records if all(v in r['available_k'] for v in ks)]
        if not selected:
            continue
        arms = arms_for(ks)
        report = pr.report(f'B1-K{k}-common', OUT / f'common_K{k}', project(selected, arms), arms,
                           notes + [f'本表K={ks}全部共享同一query集合；不具备K{k}的域/任务不进入本表。'])
        common_tables[str(k)] = {'K': ks, 'n': len(selected), 'summary': report['summary']}
    availability = {d: {str(k): sum(r['dataset'] == d and k in r['available_k'] for r in records) for k in KS}
                    for d in lab.ALL_DOMAINS}
    pc.write(OUT / 'kshot_comparisons.json', {'state': 'COMPLETE', 'n': len(records), 'availability': availability,
             'common_query_tables': common_tables, 'K1_summary': primary['summary'], 'records': records,
             'score_seconds': time.monotonic() - start, 'confirm_scored': False, 'new_encoder_calls': 0})
    path = lab.ROOT / 'docs/research/x1012_tasks/B1.md'
    with path.open('a') as f:
        f.write('\n各K共同query附表：' + '、'.join(f'B1-K{k}-common.md' for k in common_tables) + '。\n')
        f.write('\n逐域available_k及同分母对照保存在 `kshot_comparisons.json`；LVIS缺少K5，不补造参考。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--query-block', type=int, default=256, help='Only bounds temporary memory; frozen on first produce')
    args = parser.parse_args()
    if args.query_block <= 0:
        parser.error('--query-block must be positive')
    produce(args) if args.mode == 'produce' else score()
