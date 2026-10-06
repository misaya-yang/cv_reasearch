#!/usr/bin/env python3
"""Bounded CPU profile/parity audit of two unchanged-contract M2 modules.

Reads inference-only cached q/r/cov/score; never opens query truth. Limits this
process to two available CPUs and two BLAS/PyTorch threads. Writes only a fresh
output namespace; model encoding and cohort scoring are outside this audit.
"""
from __future__ import annotations
import argparse
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time

for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--candidate-file', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 4:
        parser.error('Only the first one to four bound smoke rows are allowed')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_repo / 'src'))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from ics.experiment import load_inputs, sha
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    baseline = module(args.baseline_file, 'pro_relations_baseline_audit')
    candidate = module(args.candidate_file, 'pro_relations_candidate_audit') if args.candidate_file else None
    rows = json.loads(args.manifest.read_text())[:args.limit]
    receipt = dict(threads=2, cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                   baseline_sha256=sha(args.baseline_file),
                   candidate_sha256=sha(args.candidate_file) if args.candidate_file else None,
                   manifest_sha256=sha(args.manifest), query_gt_opened=False,
                   new_encoder_forwards=0, episodes=[])
    with threadpool_limits(limits=2):
        receipt['threadpools'] = threadpool_info()
        for row in rows:
            inputs, source = load_inputs('/root', row)
            q, r, cov, score = inputs
            inputs = tuple(np.asarray(value) for value in inputs)
            record = dict(key=row['key'], sources=source, methods={})
            outputs = {}
            for tag, implementation in [('baseline', baseline), ('candidate', candidate)]:
                if implementation is None:
                    continue
                profile = cProfile.Profile()
                start, cpu = time.perf_counter(), time.process_time()
                result = profile.runcall(implementation.predict, *inputs)
                wall, cpu = time.perf_counter() - start, time.process_time() - cpu
                profile.dump_stats(str(args.out / f'{row["key"]}_{tag}.prof'))
                with (args.out / f'{row["key"]}_{tag}_profile.txt').open('w') as stream:
                    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
                fields = {arm: result['field' if arm == 'signed' else f'{arm}_control']
                          for arm in implementation.ARMS}
                outputs[tag] = fields
                np.savez_compressed(args.out / f'{row["key"]}_{tag}_fields.npz', **fields)
                record['methods'][tag] = dict(wall_seconds=wall, cpu_seconds=cpu, info=result['info'])
            if candidate:
                record['parity'] = {}
                for arm in baseline.ARMS:
                    before, after = outputs['baseline'][arm], outputs['candidate'][arm]
                    bm, bw = baseline.render_original(before, row['query_image_hw'])
                    am, aw = candidate.render_original(after, row['query_image_hw'])
                    record['parity'][arm] = dict(max_abs_field_error=float(np.max(np.abs(before - after))),
                                                field_array_equal=bool(np.array_equal(before, after)),
                                                work_mask_xor=int(np.count_nonzero(bw != aw)),
                                                original_mask_xor=int(np.count_nonzero(bm != am)))
                    if not np.allclose(before, after, rtol=0, atol=1e-10) or np.any(bm != am) or np.any(bw != aw):
                        raise RuntimeError(f'Parity failed for {row["key"]}/{arm}')
            receipt['episodes'].append(record)
            receipt['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if receipt['peak_rss_bytes'] > 4 * 1024**3:
                raise RuntimeError('Peak RSS exceeded 4GiB')
            (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            print(json.dumps(dict(key=row['key'], timings={k:v['wall_seconds'] for k,v in record['methods'].items()},
                                  parity=record.get('parity'), peak_rss_bytes=receipt['peak_rss_bytes'])), flush=True)


if __name__ == '__main__':
    main()
