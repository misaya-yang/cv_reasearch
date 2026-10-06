#!/usr/bin/env python3
"""Seal Pro M4 masks from actual RGB and a pre-bound local frozen encoder.

No weights/downloads are fetched. Cached complete-native fallback masks must
be bound before inference. This stage never opens query annotation files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-dir', type=Path)
    group.add_argument('--encoder-factory', help='Existing module:function returning an RGB encoder')
    parser.add_argument('--encoder-binding', type=Path, required=True)
    parser.add_argument('--native-binding', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--without-ref-canvas', action='store_true', help='Explicit three-arm smoke only')
    args = parser.parse_args()
    if not 1 <= args.threads <= 30:
        parser.error('threads must be between 1 and 30')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from PIL import Image
    import torch
    from ics.methods import pro_paired_environment as method
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    encoder_binding = json.loads(args.encoder_binding.read_text())
    native_binding = json.loads(args.native_binding.read_text())
    method._native_binding(native_binding)
    if not encoder_binding.get('producer'):
        raise ValueError('Bind frozen encoder producer identity')
    source = json.loads(args.manifest.read_text())
    source = source if isinstance(source, list) else source['episodes']
    if len(source) != args.expected or not source:
        raise ValueError('Manifest count differs from predeclared expected count')
    required = ('id', 'reference_rgb', 'reference_mask', 'query_rgb', 'native_npz', 'native_sha256')
    rows = []
    for row in source:
        item = {key: row[key] for key in required}  # No query GT or class/fold enters inference.
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', item['id']) or item['id'] in ('.', '..'):
            raise ValueError('Safe occurrence ID required')
        for key in ('reference_rgb', 'reference_mask', 'query_rgb', 'native_npz'):
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
        if sha(item['native_npz']) != item['native_sha256']:
            raise ValueError('Native fallback prediction hash mismatch')
        rows.append(item)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Occurrence IDs must be unique; repeated photos need distinct IDs')
    if args.out.exists():
        raise FileExistsError('Require a new output directory')
    if args.model_dir:
        for filename, binding_key in (('config.json', 'config_sha256'), ('model.safetensors', 'checkpoint_sha256')):
            if not (args.model_dir/filename).is_file():
                raise FileNotFoundError(args.model_dir/filename)
            if sha(args.model_dir/filename) != encoder_binding.get(binding_key):
                raise ValueError('Model asset hash differs from predeclared encoder binding: '+filename)
        from ics.data import TimmDINOv3
        encoder = method.FrozenTimmRGBEncoder(TimmDINOv3(str(args.model_dir)), args.device)
    else:
        module, function = args.encoder_factory.rsplit(':', 1)
        encoder = getattr(importlib.import_module(module), function)(binding=encoder_binding, device=args.device)
    args.out.mkdir(parents=True)
    (args.out/'predictions').mkdir()
    (args.out/'receipts').mkdir()
    (args.out/'statistics').mkdir()
    dump(args.out/'inference_manifest.json', rows)
    config = dict(method=method.Config().__dict__, encoder_binding=encoder_binding,
                  native_binding=native_binding, device=args.device, threads=args.threads,
                  ref_canvas=not args.without_ref_canvas, encoder_factory=args.encoder_factory,
                  encoder_source_sha256=sha(REPO/'src/ics/data.py') if args.model_dir else None,
                  source_sha256={str(path): sha(path) for path in (Path(__file__), Path(method.__file__))},
                  source_manifest_sha256=sha(args.manifest), query_gt_used=False,
                  cost_scope='cached native fallback; its original encoding/readout cost is not remeasured')
    dump(args.out/'config.json', config)
    seal = dict(state='INFERRING', prediction_sha256={}, receipt_sha256={}, statistics_sha256={}, query_gt_used=False,
                config_sha256=sha(args.out/'config.json'), manifest_sha256=sha(args.out/'inference_manifest.json'))
    run_started = time.perf_counter()
    for row in rows:
        input_hashes = {key: sha(row[key]) for key in ('reference_rgb', 'reference_mask', 'query_rgb')}
        with Image.open(row['reference_rgb']) as image:
            reference = np.asarray(image.convert('RGB')).copy()
        with Image.open(row['reference_mask']) as image:
            reference_mask = np.asarray(image).copy()
        if reference_mask.ndim != 2 or not np.isin(reference_mask, (0, 1, 255)).all():
            raise ValueError('Supply an already binary reference mask, not a semantic-class annotation')
        reference_mask = reference_mask != 0
        with Image.open(row['query_rgb']) as image:
            query = np.asarray(image.convert('RGB')).copy()

        def native_fallback(_reference, _mask, _query, _query_features):
            if sha(row['native_npz']) != row['native_sha256']:
                raise ValueError('Bound native fallback file changed')
            with np.load(row['native_npz'], allow_pickle=False) as archive:
                work, original = archive['mask_work'].copy(), archive['mask_original'].copy()
            return dict(mask_work=work, mask_original=original,
                        info=dict(encoder_forwards=0, cost_scope='precomputed sealed full native prediction',
                                  native_prediction_sha256=row['native_sha256']))

        result = method.predict(reference, reference_mask, query, encoder, native_fallback,
                                native_binding=native_binding, encoder_binding=encoder_binding,
                                progress=lambda event: print(json.dumps(dict(occurrence=row['id'], **event)), flush=True),
                                include_ref_canvas=not args.without_ref_canvas)
        if any(sha(row[key]) != digest for key, digest in input_hashes.items()):
            raise ValueError('RGB or reference mask changed during inference')
        predictions, fields = {}, {}
        for arm, output in result['arms'].items():
            predictions[arm+'_work'] = output['mask_work'].astype(np.uint8)
            predictions[arm+'_original'] = output['mask_original'].astype(np.uint8)
            if 'field' in output:
                fields[arm+'_coarse'] = output['field']
        prediction_path = args.out/'predictions'/(row['id']+'.npz')
        np.savez_compressed(prediction_path, **predictions, **fields)
        receipt = dict(info=result['info'], arms={name: output['info'] for name, output in result['arms'].items()},
                       input_sha256=input_hashes,
                       process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                                  (1 if sys.platform == 'darwin' else 1024)),
                       native_prediction_sha256=row['native_sha256'])
        receipt_path = args.out/'receipts'/(row['id']+'.json')
        dump(receipt_path, receipt)
        if 'sufficient_statistics' in result:
            statistic_path = args.out/'statistics'/(row['id']+'.npz')
            np.savez_compressed(statistic_path, **result['sufficient_statistics'])
            seal['statistics_sha256'][row['id']] = sha(statistic_path)
        seal['prediction_sha256'][row['id']] = sha(prediction_path)
        seal['receipt_sha256'][row['id']] = sha(receipt_path)
        print(json.dumps(dict(occurrence=row['id'], sealed=len(seal['prediction_sha256']),
                              encoder_forwards=result['info']['encoder_forwards'])), flush=True)
    seal.update(state='ALL_PREDICTIONS_SEALED', n=len(rows),
                run_seconds_excluding_model_loading=time.perf_counter()-run_started)
    dump(args.out/'sealed.json', seal)


if __name__ == '__main__':
    main()
