#!/usr/bin/env python3
"""Run Pro M5 on local native H20 assets, with a real block-adapter factory.

Factory signature: factory(assets: dict, device: torch.device) -> dict with
reference_adapters, query_adapters, final_norm, project, asset_receipt. Each
adapter implements the BlockAdapter protocol; Q/K include actual LN/QKnorm/
RoPE. The factory must load exact local weights, no pretrained/download calls.

Input .pt (weights_only=True): q_h20, r_h20 [all_tokens,D], q_patch_ids,
r_patch_ids, q/r [4096,D] bound normalized Ward features, cov/base [64,64],
query_image_hw [2], reference_has_foreground scalar bool. No query GT input.
Asset JSON must contain existing weights_path and source_path plus SHA256s,
and native_state_producer metadata. Use --check-assets for a read-only preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--adapter-factory', required=True, help='local Python module:callable')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-assets', action='store_true')
    parser.add_argument('--attention-query-chunk', type=int, default=64)
    args = parser.parse_args()
    assets = json.loads(args.assets.read_text())
    for label in ('weights', 'source'):
        path = Path(assets[label + '_path']).resolve(strict=True)
        actual = sha(path)
        if actual != assets[label + '_sha256']:
            raise ValueError(label + ' SHA does not match')
        assets[label + '_path'] = str(path)
    if not isinstance(assets.get('native_state_producer'), dict) or not assets['native_state_producer']:
        raise ValueError('Bound native H20 producer required')
    module, factory_name = args.adapter_factory.rsplit(':', 1)
    factory = getattr(importlib.import_module(module), factory_name)
    if args.check_assets:
        print(json.dumps(dict(state='LOCAL_ASSET_PATHS_HASHES_AND_FACTORY_PRESENT',
                              native_model_equivalence='not_run',factory=args.adapter_factory)))
        return
    if args.input is None or args.out is None:
        parser.error('--input and --out required for inference')
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite existing run: ' + str(args.out))
    import numpy as np
    import torch
    from ics.methods.pro_message_extrapolation import Config, build_native_cache, predict
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cpu')
    # CPU-only runner. Factory must not move the bound model to a different device.
    pipeline = factory(assets, device)
    source = torch.load(args.input, map_location=device, weights_only=True)
    keys = ('q_h20','r_h20','q_patch_ids','r_patch_ids','q','r','cov','base',
            'query_image_hw','reference_has_foreground')
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError('Native state packet missing: ' + ','.join(missing))
    if tuple(source['base'].shape) != (64,64) or tuple(source['q'].shape) != (4096,1024):
        raise ValueError('Actual DINOv3-L/16 1024 work contract required')
    for role in ('q','r'):
        if source[role+'_h20'].shape[1] != 1024 or source[role+'_h20'].dtype != torch.float32:
            raise ValueError('Bound native FP32 H20 width1024 required')
    final_norm, project = pipeline['final_norm'], pipeline['project']
    caches = {}
    with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
        for role, label in (('q','query'),('r','reference')):
            caches[role] = build_native_cache(source[role+'_h20'],source[role+'_patch_ids'],
                                              pipeline[label+'_adapters'],final_norm,project,
                                              producer=assets['native_state_producer'])
        result = predict(caches['q'],caches['r'],source['q'].cpu().numpy(),source['r'].cpu().numpy(),
                         source['cov'].cpu().numpy(),source['base'].cpu().numpy(),
                         source['query_image_hw'].tolist(),
                         reference_has_foreground=bool(source['reference_has_foreground']),
                         cfg=Config(attention_query_chunk=args.attention_query_chunk))
    args.out.mkdir(parents=True, exist_ok=False)
    fields_path, masks_path = args.out/'fields.npz', args.out/'masks.npz'
    np.savez_compressed(fields_path,**result['fields'])
    np.savez_compressed(masks_path,**{name+'.work':mask for name,mask in result['work_masks'].items()},
                        **{name+'.original':mask for name,mask in result['original_masks'].items()})
    receipt = dict(method='Pro_M5_fixed_native_QK_message_extrapolation_v0',
                   input_sha256=sha(args.input),assets_sha256=sha(args.assets),
                   code_sha256=sha(ROOT/'src/ics/methods/pro_message_extrapolation.py'),
                   runner_sha256=sha(__file__),fields_sha256=sha(fields_path),masks_sha256=sha(masks_path),
                   adapter_factory=args.adapter_factory,asset_receipt=pipeline['asset_receipt'],
                   inference=result['info'],query_GT_read=False,device='cpu',
                   state='PREDICTIONS_SEALED',real_episodes_claimed=0,
                   note='No scoring here; actual episode identity/exposure must be bound by evaluator')
    (args.out/'sealed.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out.resolve()),state=receipt['state'])))


if __name__ == '__main__':
    main()
