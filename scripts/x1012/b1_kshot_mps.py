"""Device-selectable B1 entrypoint; original B1/B2/B3 frozen sources stay intact.

Use --device mps only after the other MPS job has ended. Independent O/24
kernel/region calculations use the selected device; exact-pair FoRIS controls
remain cache-only CPU/CRF through the original host_cpu implementation.
"""
from __future__ import annotations

import argparse
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
import b1_kshot as base

lab, pc, OUT, P0 = base.lab, base.pc, base.OUT, base.P0


def weighted_region_mean(values, labels):
    """Fixed N x 64 membership weights, without variable-size token indexing."""
    membership = torch.nn.functional.one_hot(labels, num_classes=64).to(values.dtype)
    means = (membership.T @ values) / membership.sum(0).clamp_min(1)
    return membership @ means


def independent_fields(references, query, coverages, labels):
    if query.device.type == 'cpu':
        return base.independent_fields(references, query, coverages, block=256, labels=labels)
    assert query.device.type == 'mps' and len(query) % 256 == 0
    assert len(references) in base.KS and len(references) == len(coverages)
    r, y = torch.cat(references), torch.cat(coverages)
    mu = .5 * (torch.stack([v.mean(0) for v in references]).mean(0) + query.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(query - mu)
    posterior, maxima, log_explanation = [], [], []
    for offset in range(0, len(query), 256):
        sim = qc[offset:offset + 256] @ rc.T
        maximum = sim.max(1).values
        kernel = torch.exp((sim - maximum[:, None]) / base.TEMPERATURE)
        evidence = kernel.sum(1)
        posterior.append((kernel @ y) / evidence.clamp_min(1e-30))
        maxima.append(maximum)
        log_explanation.append(evidence.clamp_min(1e-30).log() + maximum / base.TEMPERATURE - float(np.log(len(r))))
    c, smax = torch.cat(posterior), torch.cat(maxima)
    return {'c': c, 'IND0': weighted_region_mean(c, labels),
            'INDN': weighted_region_mean(c * lab.minmax(smax), labels),
            'sim_max': smax, 'log_mean_kernel_mass': torch.cat(log_explanation)}


def produce(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS is unavailable; no CPU fallback or mixed-device outputs')
    pc.setup(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if not config_path.exists():
        rows = pc.read(P0 / 'kshot.json')
        assert rows and len({pc.key(b) for b in rows}) == len(rows)
        for b in rows:
            refs = b['references']
            assert b['available_k'] == [k for k in base.KS if k <= len(refs)] and b['available_k']
            photos = [r['photo'] for r in refs]
            assert len(photos) == len(set(photos))
            assert str((lab.ASSETS / b['row']['query_path']).resolve()) not in photos
            assert all(r['dataset'] == b['dataset'] and str(r['fold']) == str(b['row']['fold']) and
                       str(r['class_id']) == str(b['row']['global_class_id']) for r in refs)
            assert refs[0]['raw']['payload_path'] == b['fit']['reference_raw']['payload_path']
        # Do not silently adopt predictions that predate this device freeze.
        if any((OUT / name).exists() and any((OUT / name).iterdir()) for name in ('records', 'predictions', 'fields')):
            raise RuntimeError('Existing B1 artifacts without a frozen config must be preserved, not mixed')
        pc.write(OUT / 'manifest.json', rows)
        pc.write(config_path, {'state': 'FROZEN', 'task': 'B1', 'n': len(rows), 'device': args.device,
                 'entrypoint': str(Path(__file__).resolve()), 'cpu_threads': 4, 'query_block': 256,
                 'K': base.KS, 'temperature': base.TEMPERATURE,
                 'reference_weights': 'uniform 1/K; common token-count normalisation; no separate FG/BG-area normalisation',
                 'centre': '.5*(mean_k(mean_tokens(R_k))+mean_tokens(Q))',
                 'natural_rule': 'K64 query regions; FP32 bilinear1024, minmax, strict >.5',
                 'MPS_shape_policy': 'Q blocks exactly256; four finite reference sizes4096*K; fixed N*64 region membership weights',
                 'control': 'same pair K1 FoRIS complete; cached exact pair or own host_cpu/CRF replay',
                 'FoRIS_device': 'cpu', 'score_device': 'cpu', 'mps_jobs': int(args.device == 'mps'),
                 'kshot_manifest_sha256': pc.sha(P0 / 'kshot.json'),
                 'source_sha256': base.source_hashes(Path(__file__), lab.ROOT / 'scripts/cached_dino.py',
                                     lab.ROOT / 'src/ics/foris.py', lab.ROOT / 'scripts/x1012/p0_manifest.py'),
                 'new_encoder_calls': 0, 'query_GT_reads': 0})
    config = pc.read(config_path)
    frozen_device = config.get('device', 'cpu')
    if frozen_device != args.device:
        raise RuntimeError(f'B1 is frozen on {frozen_device}; requested {args.device}. Existing outputs cannot be mixed.')
    if config.get('entrypoint') != str(Path(__file__).resolve()):
        raise RuntimeError('Existing B1 config belongs to another frozen entrypoint; preserve and resume it there')
    assert config['query_block'] == 256
    base.check_sources(config)
    if (OUT / 'sealed.json').exists():
        base.verify_seal(OUT, 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED')
        print({'state': 'ALREADY_SEALED', 'task': 'B1', 'device': frozen_device, 'n': config['n']}, flush=True)
        return
    rows = pc.read(OUT / 'manifest.json')
    for name in ('predictions', 'fields', 'records'):
        (OUT / name).mkdir(exist_ok=True)
    same_pair = {b['pair_signature']: b for b in pc.read(P0 / 'explore.json') if b.get('pair_signature')}
    start, cpu = time.monotonic(), time.process_time()
    receipts, replay_state = [], {}
    for i, b in enumerate(rows, 1):
        name = pc.key(b)
        receipt_path = OUT / 'records' / (name + '.json')
        if receipt_path.exists():
            receipt = pc.read(receipt_path)
            assert receipt.get('device', 'cpu') == frozen_device, 'Mixed CPU/MPS record is forbidden'
            assert all(pc.sha(receipt[k + '_path']) == receipt[k + '_sha256'] for k in ('prediction', 'field'))
            receipts.append(receipt)
            continue
        prediction_path, field_path = OUT / 'predictions' / (name + '.npz'), OUT / 'fields' / (name + '.npz')
        assert not prediction_path.exists() and not field_path.exists(), 'Preserve incomplete task output'
        tick = time.monotonic()
        counters = {}
        refs = b['references'][:max(b['available_k'])]
        with base.legal_inference(b, [r['view']['mask_path'] for r in refs]):
            query = lab.debias(base.read_raw(b['fit']['query_raw_identity'], counters).to(lab.DEV))
            references = [lab.debias(base.read_raw(r['raw'], counters).to(lab.DEV)) for r in refs]
            coverages = [lab.coverage64(lab.mask1024(r['view']['mask_path'])).flatten().to(lab.DEV) for r in refs]
            labels = base.region_labels(query)
            predictions, fields = {}, {}
            for k in b['available_k']:
                result = independent_fields(references[:k], query, coverages[:k], labels)
                for method in ('IND0', 'INDN'):
                    arm = f'K{k}/{method}'
                    predictions[arm] = pc.pack(pc.readout(result[method]))
                    fields[arm] = result[method].cpu().numpy()
                fields[f'K{k}/log_mean_kernel_mass'] = result['log_mean_kernel_mass'].cpu().numpy()
            # Original helper explicitly constructs FoRIS(device='cpu'), with Replay.encoder=None.
            predictions['FoRIS'], baseline_receipt = base.foris_control(b, same_pair, replay_state)
        np.savez_compressed(prediction_path, **predictions)
        np.savez_compressed(field_path, **fields)
        receipt = {'index': name, 'episode_id': b['row']['episode_id'], 'dataset': b['dataset'],
                   'device': frozen_device, 'available_k': b['available_k'], 'reference_ids': [r['id'] for r in refs],
                   'prediction_path': str(prediction_path), 'prediction_sha256': pc.sha(prediction_path),
                   'field_path': str(field_path), 'field_sha256': pc.sha(field_path),
                   'FoRIS_control': baseline_receipt, 'seconds': time.monotonic() - tick,
                   'new_encoder_calls': 0, 'query_GT_reads': 0, **counters}
        pc.write(receipt_path, receipt)
        receipts.append(receipt)
        del references, query, coverages, labels, result, fields, predictions
        lab.free()
        if i <= 3 or i % 50 == 0:
            print({'task': 'B1', 'device': frozen_device, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - start, 1)}, flush=True)
    pc.write(OUT / 'sealed.json', {'state': 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED', 'n': len(rows), 'records': receipts,
             'device': frozen_device, 'entrypoint': str(Path(__file__).resolve()),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--query-block', type=int, choices=(256,), default=256)
    args = parser.parse_args()
    produce(args) if args.mode == 'produce' else base.score()
