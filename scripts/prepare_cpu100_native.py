#!/usr/bin/env python3
"""Fresh frozen CPU DINO1024 features; read only the known reference mask."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def one(row, args):
    from PIL import Image
    import numpy as np
    key = row['key']
    case = args.out / key
    case.mkdir(exist_ok=False)
    ref = args.data / row['support']
    query = args.data / row['query']
    ref_annotation = args.data / 'annotations' / (row['support'][:-4] + '.png')
    with Image.open(ref_annotation) as im:
        binary = np.asarray(im) == int(row['c']) + 1
    mask_path = case / 'reference_mask.png'
    Image.fromarray(binary.astype(np.uint8) * 255).save(mask_path)
    start = time.monotonic()
    command = [sys.executable, str(Path(__file__).with_name('run_direct_dino_features.py')), 'extract',
               '--model-dir', str(args.model_dir), '--reference-rgb', str(ref),
               '--reference-mask', str(mask_path), '--query-rgb', str(query),
               '--out', str(case / 'native'), '--threads', str(args.threads)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = str(args.threads)
    with (case / 'extract.log').open('w') as log:
        job = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(case / 'running.json', dict(pid=job.pid, started=time.time(), command=command, query_GT_read=False))
        code = job.wait()
    receipt = dict(id=key, exit_code=code, seconds=time.monotonic() - start, query_GT_read=False)
    write(case / 'receipt.json', receipt)
    if code:
        return None, receipt
    native_manifest = json.loads((case / 'native' / 'manifest.json').read_text())['rows'][0]
    with Image.open(ref) as im:
        ref_shape = [im.height, im.width]
    with Image.open(query) as im:
        query_shape = [im.height, im.width]
    native_manifest.update(id=key, q_rgb=str(query), r_rgb=str(ref), reference_mask=str(mask_path),
                          q_rgb_sha256=sha(query), r_rgb_sha256=sha(ref), reference_mask_sha256=sha(mask_path),
                          reference_geometry=dict(view_side=1024, resized_hw=[1024, 1024],
                              padding_top_left=[0, 0], original_hw=ref_shape),
                          original_shape=query_shape)
    # Labels/class identity are retained only in a separate future scoring manifest.
    write(case / 'inference_row.json', native_manifest)
    return native_manifest, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=6)
    args = p.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > 28:
        raise ValueError('At most28 extraction threads, leaving monitor and inference capacity')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.rows.read_text())
    if isinstance(rows, dict):
        rows = rows['rows']
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Feature preparation rows must be unique')
    started = time.time()
    receipts, inference_rows = [], []
    write(args.out / 'status.json', dict(state='running', total=len(rows), finished=0, started=started,
          workers=args.workers, threads=args.threads, query_GT_read=False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, row, args): row['key'] for row in rows}
        for future in concurrent.futures.as_completed(futures):
            try:
                row, receipt = future.result()
            except Exception as error:
                row = None
                receipt = dict(id=futures[future], exit_code=1, error=repr(error), query_GT_read=False)
                write(args.out / futures[future] / 'receipt.json', receipt)
            receipts.append(receipt)
            if row:
                inference_rows.append(row)
            write(args.out / 'status.json', dict(state='running', total=len(rows), finished=len(receipts),
                 failed=sum(r['exit_code'] != 0 for r in receipts), started=started,
                 elapsed=time.time() - started, receipts=receipts, query_GT_read=False))
            print(json.dumps(receipt), flush=True)
    order = {r['key']: i for i, r in enumerate(rows)}
    inference_rows.sort(key=lambda r: order[r['id']])
    write(args.out / 'manifest.json', dict(schema='DINO_ONLY_FEATURE_INPUT_V1', rows=inference_rows,
          input_resolution=1024, processed_FoRIS_cache_used=False, query_GT_read=False,
          input_rows_sha256=sha(args.rows)))
    write(args.out / 'evaluation_rows.json', rows)
    failed = sum(r['exit_code'] != 0 for r in receipts)
    write(args.out / 'status.json', dict(state='failed' if failed else 'complete', total=len(rows),
          finished=len(receipts), failed=failed, elapsed=time.time() - started, receipts=receipts,
          manifest_sha256=sha(args.out / 'manifest.json'), query_GT_read=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
