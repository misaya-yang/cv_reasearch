#!/usr/bin/env python3
"""Run Pro M5 on local native H20 assets, with a real block-adapter factory.

Factory signature: factory(assets: dict, device: torch.device) -> dict with
reference_adapters, query_adapters, final_norm, project, asset_receipt. Each
adapter implements the BlockAdapter protocol; Q/K include actual LN/QKnorm/
RoPE. The factory must load exact local weights, no pretrained/download calls.
Alternatively factory returns model, project, asset_receipt and the input packet
contains FP32 images [reference,query,3,1024,1024]; actual native SDPA capture
then supplies all QKV without manually reconstructing RoPE. The actual model
must be frozen eval timm Eva, never a synthetic replacement.

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
    parser.add_argument('--adapter-factory',default='ics.methods.pro_message_extrapolation:load_local_eva_pipeline',
                        help='local Python module:callable; default audited offline Eva factory')
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
    producer = assets['native_state_producer']
    if args.adapter_factory.endswith(':load_local_eva_pipeline') and type(assets.get('projection_enabled')) is not bool:
        raise ValueError('Bind original episode projection_enabled before using the default Eva factory')
    required_producer = dict(weights_sha256=assets['weights_sha256'],source_sha256=assets['source_sha256'],
                             suffix_start_state='after_block20',work_resolution=1024,
                             patch_grid=[64,64],precision='float32',projection_frozen=True)
    for key, value in required_producer.items():
        if producer.get(key) != value:
            raise ValueError('Native-state producer contract missing/mismatched: '+key)
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
    from ics.methods.pro_message_extrapolation import Config, build_native_cache, capture_native_pair, predict
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cpu')
    # CPU-only runner. Factory must not move the bound model to a different device.
    pipeline = factory(assets, device)
    source = torch.load(args.input, map_location=device, weights_only=True)
    if source.get('producer') != producer:
        raise ValueError('Native H20 packet producer does not match bound assets')
    image_mode = 'images' in source
    keys = ('cov','base','query_image_hw','reference_has_foreground')
    if not image_mode:
        keys += ('q_h20','r_h20','q_patch_ids','r_patch_ids','q','r')
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError('Native state packet missing: ' + ','.join(missing))
    if tuple(source['base'].shape) != (64,64) or tuple(source['cov'].shape) != (64,64):
        raise ValueError('Actual DINOv3-L/16 1024 work contract required')
    if image_mode:
        if source['images'].shape != (2,3,1024,1024) or source['images'].dtype != torch.float32:
            raise ValueError('FP32 normalized paired reference/query RGB images required')
    else:
        for role in ('q','r'):
            if tuple(source[role+'_h20'].shape) != (4101,1024) or source[role+'_h20'].dtype != torch.float32:
                raise ValueError('Bound native FP32 H20 4096patch+5prefix by1024 required')
            if source[role].shape != (4096,1024) or source[role].dtype != torch.float32:
                raise ValueError('Fresh FP32 Ward features required; old FP16 caches are not the Pro producer')
            if not torch.equal(source[role+'_patch_ids'],torch.arange(5,4101)):
                raise ValueError('Bound actual five-prefix patch sequence required')
    caches = {}
    capture_receipt = dict(actual_encoded_images=0,mode='bound_native_H20_packet')
    with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
        if image_mode:
            caches['r'],caches['q'],capture_receipt = capture_native_pair(pipeline['model'],source['images'],
                                          pipeline['project'],producer=producer)
            source['q'],source['r'] = caches['q'].projected_native,caches['r'].projected_native
        else:
            for role, label in (('q','query'),('r','reference')):
                caches[role] = build_native_cache(source[role+'_h20'],source[role+'_patch_ids'],
                                                  pipeline[label+'_adapters'],pipeline['final_norm'],pipeline['project'],
                                                  producer=assets['native_state_producer'])
        for role in ('q','r'):
            if caches[role].qkv[0][0].shape != (16,4101,64):
                raise ValueError('Actual native DINOv3-L 16heads/64head-width required')
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
                   inference=result['info'],capture=capture_receipt,query_GT_read=False,device='cpu',
                   state='PREDICTIONS_SEALED',real_episodes_claimed=0,
                   note='No scoring here; actual episode identity/exposure must be bound by evaluator')
    (args.out/'sealed.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out.resolve()),state=receipt['state'])))


if __name__ == '__main__':
    main()
