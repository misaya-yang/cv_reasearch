"""One complete cached queue, with all new masks sealed before query truth."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '4'
import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'evidence/local/task_conflict_execution6200_20261011'
sys.path.insert(0, str(OLD))
legacy_spec = importlib.util.spec_from_file_location('legacy_task_conflict_producer', OLD/'producer.py')
legacy = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy)
sys.path.insert(0, str(HERE))
from density_core import compute
QueryTruthGuard = legacy.QueryTruthGuard

ARMS = ('background_vetted_complete', 'unvetted_anchor_complete',
        'reference_only_protected_complete', 'same_edit_budget_foris_margin')
INPUT_SHAS = {'inputs.jsonl': '6704d3ddc2bfd5cc16c47287dacebb9f5b2aeb46705dd0e079aafcb5dddbde0c',
              'score_bindings.jsonl': '92e79d934b2192e8e0f1da62fbe1e57bd4f6237e4d3f74ce30ea3a5cb0d1f037',
              'raw_metadata.json': '8e3433e8f745e90bc6615955717f7afae5d2fad8481375be3e0704346e275858'}
read, lines, write, sha = legacy.read, legacy.lines, legacy.write, legacy.sha


def load_masks(source, scoring):
    masks, score = legacy.masks_and_score(source)
    binding = scoring['old_final_masks']['N']
    assert sha(binding['path']) == binding['sha256']
    with np.load(binding['path'], allow_pickle=False) as z:
        for frame in ('cli', 'original'):
            masks[frame]['N'] = legacy.unpack(z, binding[frame+'_key'], masks[frame]['B'].shape)
    return masks, score


def frame_outputs(masks, score, dense):
    b, g, r, n = (masks[k] for k in ('B', 'G', 'R0', 'N'))
    assert all(m.shape == b.shape for m in (g, r, n))
    assert not np.any(n & ~(b | g | r))
    span = float(np.ptp(score))
    d = 2*legacy.resize((score-score.min())/max(span, 1e-6), b.shape)-1
    conf = np.abs(d) if span else np.zeros(b.shape)
    signed = np.where(b, conf, -conf)
    da, dd = (~b) & (g | r | n), b & (~n)
    agreed = (b == g) & (g == r)
    rendered = {name: None if h is None else np.clip(legacy.resize(h, b.shape), -1, 1)
                for name, h in ((k, dense[k]) for k in ('reference_h', 'vetted_h', 'unvetted_h'))}
    outputs, cert = {}, {}
    for arm, qkey in zip(ARMS[:3], ('vetted_h', 'unvetted_h', None)):
        hr = rendered['reference_h']
        hq = rendered[qkey] if qkey else hr
        if hr is None or hq is None:
            add, delete = np.zeros(b.shape, bool), np.zeros(b.shape, bool)
            pred = b.copy()
        else:
            ha, hd = (hr, hr) if qkey is None else (np.minimum(hr, hq), np.maximum(hr, hq))
            add, delete = da & (signed+0.5*ha > 0), dd & (signed+0.5*hd < 0)
            pred = (b & ~delete) | add
            alternate = b.copy()
            alternate[(~b) & da & (ha > 2*conf)] = True
            alternate[b & dd & (hd < -2*conf)] = False
            assert np.array_equal(pred, alternate)
        assert np.array_equal(pred[agreed], b[agreed])
        assert np.array_equal(pred[b & n], b[b & n])
        assert not np.any((pred != b) & (conf >= 0.5))
        outputs[arm] = pred
        cert[arm] = dict(add=int(add.sum()), delete=int(delete.sum()))
    main = outputs[ARMS[0]]
    ka, kd = int((main & ~b).sum()), int((b & ~main).sum())
    ea, ed = da & (conf < 0.5), dd & (conf < 0.5)
    add, ca = legacy.select(ea, d, ka, True)
    delete, cd = legacy.select(ed, d, kd, False)
    budget = (b | add) & ~delete
    alternate = b.copy()
    for domain, count, descending, value in ((ea, ka, True, True), (ed, kd, False, False)):
        idx = np.flatnonzero(domain.ravel())
        rank = np.argsort(-d.ravel()[idx] if descending else d.ravel()[idx], kind='stable')
        alternate.ravel()[idx[rank[:count]]] = value
    assert np.array_equal(alternate, budget)
    assert int((budget & ~b).sum()) == ka and int((b & ~budget).sum()) == kd
    assert np.array_equal(budget[agreed], b[agreed])
    assert np.array_equal(budget[b & n], b[b & n])
    outputs[ARMS[-1]] = budget
    cert[ARMS[-1]] = dict(add=ka, delete=kd, add_selection=ca, delete_selection=cd)
    return outputs, dict(edit_counts=cert, D_add=int(da.sum()), D_delete=int(dd.sum()),
                         B_N_common=int((b & n).sum()), agreed_pixels=int(agreed.sum()),
                         formula_reconstructions=4)


def numeric_reference(actual, reference):
    out = {}
    for key in ('reference_h', 'vetted_h', 'unvetted_h'):
        a, b = actual[key], reference[key]
        out[key] = dict(actual_available=a is not None, reference_available=b is not None)
        if a is not None and b is not None:
            out[key].update(max_abs=float(np.max(np.abs(a-b))),
                            rms=float(np.sqrt(np.mean((a-b)**2))),
                            native_sign_difference=int(np.count_nonzero((a > 0) != (b > 0))))
    for key in ('source_scale_squared', 'MA_vetted'):
        a, b = actual['stats'].get(key), reference['stats'].get(key)
        if a is not None and b is not None:
            out[key+'_absolute_difference'] = abs(a-b)
    out['query_FG_bank_identity_agreement'] = {key: actual['stats'].get(key) == reference['stats'].get(key)
                                             for key in ('vetted_query_FG_index_sha256', 'unvetted_query_FG_index_sha256')}
    out['fallback_agreement'] = {key: actual['stats'].get(key) == reference['stats'].get(key)
                                 for key in ('reference_reason', 'vetted_reason', 'unvetted_reason')}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    assert sha(args.protocol) == args.protocol_sha
    protocol = read(args.protocol)
    assert tuple(protocol['new_arms']) == ARMS
    assert protocol['state'] == 'FROZEN_FOUR_COMPLETE_ARMS_FOR_SOLE_ROOT_PRODUCTION_BEFORE_NEW_FINAL_MASKS'
    for path_key, digest_key in (
            ('execution_preflight_receipt_path', 'execution_preflight_receipt_sha256'),
            ('execution_inputs_path', 'execution_inputs_sha256'),
            ('execution_score_bindings_path', 'execution_score_bindings_sha256'),
            ('execution_raw_metadata_path', 'execution_raw_metadata_sha256')):
        binding = protocol['source_bindings']
        assert sha(ROOT/binding[path_key]) == binding[digest_key]
    for name, digest in INPUT_SHAS.items():
        assert sha(OLD/name) == digest
    sources, scores = lines(OLD/'inputs.jsonl'), lines(OLD/'score_bindings.jsonl')
    assert len(sources) == len(scores) == 6200
    guard, reader = QueryTruthGuard([r['row'] for r in scores]), legacy.RawReader()
    torch.set_num_threads(4)
    with (HERE/'.producer.lock').open('a+') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE/'sealed.json').exists(), 'Completed producer cannot run twice'
        assert not (HERE/'config.json').exists(), 'Do not silently resume or alter a bound run'
        for name in ('predictions', 'records'):
            (HERE/name).mkdir(exist_ok=True)
            assert not any((HERE/name).iterdir()), 'Inspect any previous attempt before restart'
        config = dict(state='FROZEN_REFERENCE_BG_COMPLETE6200', protocol_sha256=args.protocol_sha,
                      protocol_path=str(args.protocol.resolve()), n=6200, arms=list(ARMS),
                      input_root=str(OLD), input_sha256=INPUT_SHAS,
                      source_sha256={str(p): sha(p) for p in (Path(__file__), HERE/'density_core.py',
                          OLD/'producer.py', ROOT/'evidence/local/nine_window_execution_20261011/truth_guard.py')},
                      workers=1, CPU_threads=4)
        write(HERE/'config.json', config)
        (HERE/'protocol.frozen.json').write_bytes(args.protocol.read_bytes())
        started, cpu = time.monotonic(), time.process_time()
        totals, runtime_counts, records = Counter(), Counter(), []
        write(HERE/'activity.json', dict(state='RUNNING_FULL6200', pid=os.getpid(), completed=0,
                                        total=6200, protocol_sha256=args.protocol_sha))
        with guard:
            for i, (source, scoring) in enumerate(zip(sources, scores)):
                begin = time.monotonic()
                assert source['index'] == scoring['index'] == i
                assert source['row']['episode_id'] == scoring['row']['episode_id']
                costs, tick = {}, time.monotonic()
                masks, score = load_masks(source, scoring)
                p = legacy.reference_coverage(source)
                a = (masks['cli']['B'] & masks['cli']['G'] & masks['cli']['R0']).reshape(64, 16, 64, 16).mean((1, 3)).ravel()
                costs['saved_masks_margin_reference_seconds'] = time.monotonic()-tick
                tick = time.monotonic()
                rr = reader.load(source['raw']['reference_full'])
                rq = rr if source['raw']['reference_full']['key'] == source['raw']['query_full']['key'] else reader.load(source['raw']['query_full'])
                costs['raw_IO_seconds'] = time.monotonic()-tick
                tick, failure, runtime = time.monotonic(), None, 'mps_float32'
                costs['failed_MPS_seconds'] = 0.0
                try:
                    dense = compute(rr, rq, p, a, device='mps', dtype=torch.float32)
                    torch.mps.synchronize()
                except RuntimeError as exc:
                    failure = str(exc)
                    costs['failed_MPS_seconds'] = time.monotonic()-tick
                    try:
                        torch.mps.synchronize()
                    except RuntimeError:
                        pass
                    torch.mps.empty_cache()
                    tick, runtime = time.monotonic(), 'cpu_float32_actual_backend_fallback'
                    dense = compute(rr, rq, p, a, device='cpu', dtype=torch.float32)
                costs['shared_dense_cues_seconds'] = time.monotonic()-tick
                comparison = None
                if i < 3:
                    tick = time.monotonic()
                    reference = compute(rr, rq, p, a, device='cpu', dtype=torch.float64)
                    comparison = numeric_reference(dense, reference)
                    costs['first3_CPU64_numeric_reference_seconds'] = time.monotonic()-tick
                    del reference
                coverage_sha = {'reference': legacy.tensor_sha(p), 'anchor': legacy.tensor_sha(a)}
                del rr, rq, p, a
                tick = time.monotonic()
                payload = {'original_hw': np.asarray(source['row']['query_size_hw'])}
                frames = {}
                for frame in ('cli', 'original'):
                    outputs, frames[frame] = frame_outputs(masks[frame], score, dense)
                    for arm, mask in outputs.items():
                        payload[frame+'/'+arm] = np.packbits(mask)
                    del outputs
                costs['two_frame_readout_budget_verify_seconds'] = time.monotonic()-tick
                tick = time.monotonic()
                path = HERE/'predictions'/f'{i:06}.npz'
                temporary = path.with_suffix('.pending.npz')
                np.savez_compressed(temporary, **payload)
                temporary.replace(path)
                rec = dict(index=i, binding_index=source['binding_index'], episode_id=source['row']['episode_id'],
                           dataset=source['row']['dataset'], filename=path.name, prediction_sha256=sha(path),
                           dense_stats=dense['stats'], dense_runtime=runtime, MPS_failure=failure, frames=frames,
                           first3_CPU64_numeric_reference=comparison, coverage_sha256=coverage_sha,
                           source_N=scoring['old_final_masks']['N'], costs=costs,
                           raw_keys=[source['raw'][k]['key'] for k in ('reference_full', 'query_full')],
                           query_GT_reads=0, new_encoder_calls=0, new_CRF_calls=0)
                write(HERE/'records'/f'{i:06}.json', rec)
                costs['packed_masks_record_write_seconds'] = time.monotonic()-tick
                rec['total_seconds'] = time.monotonic()-begin
                write(HERE/'records'/f'{i:06}.json', rec)
                totals.update(costs); runtime_counts[runtime] += 1
                records.append({k: rec[k] for k in ('index', 'binding_index', 'episode_id', 'filename', 'prediction_sha256')})
                del masks, score, dense, payload, frames
                write(HERE/'activity.json', dict(state='RUNNING_FULL6200', pid=os.getpid(), completed=i+1,
                      total=6200, last_episode_id=rec['episode_id'], protocol_sha256=args.protocol_sha))
                if i < 3 or (i+1) % 50 == 0:
                    print(dict(stage='produce', completed=i+1, total=6200, seconds=time.monotonic()-started), flush=True)
            assert len(records) == 6200 and not guard.attempts
            seal = dict(state='ALL6200_FOUR_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT', n=6200, mask_count=49600,
                        protocol_sha256=args.protocol_sha, config_sha256=sha(HERE/'config.json'), records=records,
                        query_GT_guard=guard.receipt(), new_encoder_calls=0, new_CRF_calls=0, new_dense_archives=0)
            write(HERE/'sealed.json', seal)
        receipt = dict(state=seal['state'], n=6200, mask_count=49600, wall_seconds=time.monotonic()-started,
                       CPU_seconds=time.process_time()-cpu, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       phase_seconds=dict(totals), raw_reader=dict(reader.cost, unique_keys=len(reader.used)),
                       runtime_counts=dict(runtime_counts), guard=guard.receipt(), seal_sha256=sha(HERE/'sealed.json'),
                       new_encoder_calls=0, new_CRF_calls=0, new_dense_archives=0, resumed=False)
        write(HERE/'production_receipt.json', receipt)
        write(HERE/'activity.json', dict(state=seal['state'], completed=6200, total=6200, pid=os.getpid(),
                                        seal_sha256=receipt['seal_sha256']))
        print(receipt, flush=True)


if __name__ == '__main__':
    main()
