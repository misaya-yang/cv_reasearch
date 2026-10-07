#!/usr/bin/env python3
"""Independent DINO-only feature input and full-mask inference; no GT reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def export_local(source, out):
    """Strip existing real RGB128 outputs to exactly the DINO-only inputs."""
    import numpy as np
    seals = json.loads((source/'sealed4.json').read_text())
    rows = []
    out.mkdir(parents=True, exist_ok=False)
    for entry in seals['cases']:
        key = entry['id']
        case = source/key
        for name in ('tokens.npz', 'receipt.json'):
            if sha(case/name) != entry['hashes'][name]:
                raise ValueError('Existing native-DINO output seal mismatch')
        receipt = json.loads((case/'receipt.json').read_text())
        geometry = dict(receipt['query_geometry'], view_side=128)
        producer = dict(kind='frozen_DINOv3_FP32_native_final_LN_patches_then_unit',
                        FoRIS_Part1_applied=False, model_input_side=128,
                        source_tokens_sha256=sha(case/'tokens.npz'),
                        source_receipt_sha256=sha(case/'receipt.json'),
                        model_assets=receipt['producer_binding']['assets'],
                        scope='Existing real whole-R/Q RGB128 forwards; not RGB1024 features')
        target = out/(key+'.npz')
        with np.load(case/'tokens.npz', allow_pickle=False) as tokens:
            np.savez_compressed(target, q=tokens['query_rgb'], r=tokens['reference_rgb'],
                foreground_weight=receipt['reference_foreground_patch_coverage'],
                valid_weight=receipt['reference_valid_patch_coverage'],
                query_geometry_json=json.dumps(geometry), producer_json=json.dumps(producer))
        rows.append(dict(id=key, feature_pack=str(target.resolve()), sha256=sha(target),
                         original_shape=receipt['query_geometry']['original_hw']))
    dump(out/'manifest.json', dict(schema='DINO_ONLY_FEATURE_INPUT_V1', rows=rows,
         input_member_keys=['q', 'r', 'foreground_weight', 'valid_weight', 'query_geometry_json', 'producer_json'],
         no_old_base_score_mask_or_query_GT_exported=True))


def extract(args):
    """Default target path: two fresh frozen-DINO RGB1024 forwards, no Part1."""
    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    config = json.loads((args.model_dir/'config.json').read_text())
    if config.get('architecture') != 'vit_large_patch16_dinov3':
        raise ValueError('Pinned DINOv3-L/16 architecture required, not a shape-compatible other model')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_hash = sha(args.model_dir/'model.safetensors')
    config_hash = sha(args.model_dir/'config.json')
    model = TimmDINOv3(str(args.model_dir)).m.to('cpu').float().eval().requires_grad_(False)
    if model.num_prefix_tokens != 5:
        raise ValueError('Pinned five-prefix DINOv3 Eva interface required')
    size = 1024
    images = [Image.open(p).convert('RGB') for p in (args.reference_rgb, args.query_rgb)]
    mask = np.asarray(Image.open(args.reference_mask))
    if mask.shape != (images[0].height, images[0].width) or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError('Complete original binary R mask required, not a semantic class map')
    features = []
    with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
        for image in images:
            a = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).copy()
            x = torch.from_numpy(a).permute(2, 0, 1).float()/255
            mean, std = x.new_tensor((.485, .456, .406))[:, None, None], x.new_tensor((.229, .224, .225))[:, None, None]
            tokens = model.forward_features(((x-mean)/std)[None])
            if tuple(tokens.shape) != (1, 4101, 1024) or tokens.dtype != torch.float32:
                raise ValueError('Actual RGB1024 FP32 native DINO patch interface mismatch')
            features.append(tokens[0, 5:].numpy().copy())
    resized = np.asarray(Image.fromarray((mask != 0).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST), float)
    foreground = resized.reshape(64, 16, 64, 16).mean((1, 3)).ravel()
    if checkpoint_hash != sha(args.model_dir/'model.safetensors') or config_hash != sha(args.model_dir/'config.json'):
        raise ValueError('Frozen model assets changed during extraction')
    producer = dict(kind='frozen_DINOv3_FP32_native_final_LN_patches', FoRIS_Part1_applied=False,
                    model_input_side=1024, frozen=True, new_encoder_forwards=2,
                    architecture=config['architecture'],
                    checkpoint_sha256=checkpoint_hash,
                    model_config_sha256=config_hash,
                    source_image_hashes=[sha(p) for p in (args.reference_rgb, args.query_rgb)],
                    reference_mask_sha256=sha(args.reference_mask))
    args.out.mkdir(parents=True, exist_ok=False)
    target = args.out/'features.npz'
    geometry = dict(view_side=1024, resized_hw=[1024, 1024], padding_top_left=[0, 0])
    np.savez_compressed(target, q=features[1], r=features[0], foreground_weight=foreground,
        valid_weight=np.ones(4096), query_geometry_json=json.dumps(geometry), producer_json=json.dumps(producer))
    dump(args.out/'manifest.json', dict(schema='DINO_ONLY_FEATURE_INPUT_V1', rows=[
         dict(id='pair', feature_pack=str(target.resolve()), sha256=sha(target),
              original_shape=[images[1].height, images[1].width])]))


def infer(manifest, out):
    import numpy as np
    from PIL import Image
    from ics.methods.direct_dino_features import predict
    bound = json.loads(manifest.read_text())
    if bound['schema'] != 'DINO_ONLY_FEATURE_INPUT_V1':
        raise ValueError('Explicit direct-DINO input provenance required')
    out.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT/'src/ics/methods/direct_dino_features.py', ROOT/'src/ics/data.py']
    snapshots = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        frozen = out/'source'/relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(path.read_bytes())
        snapshots[str(relative)] = sha(frozen)
    results = []
    for row in bound['rows']:
        tick = time.perf_counter()
        path = Path(row['feature_pack'])
        if sha(path) != row['sha256']:
            raise ValueError('Bound direct feature pack changed')
        with np.load(path, allow_pickle=False) as packet:
            producer = json.loads(str(packet['producer_json']))
            if producer.get('FoRIS_Part1_applied') is not False:
                raise ValueError('This entry requires direct DINO outputs, not processed FoRIS cache')
            if producer.get('kind') not in ('frozen_DINOv3_FP32_native_final_LN_patches',
                                           'frozen_DINOv3_FP32_native_final_LN_patches_then_unit'):
                raise ValueError('Explicit native DINOv3 feature producer required')
            side = producer['model_input_side']
            q, r = packet['q'], packet['r']
            if side not in (128, 1024) or q.shape != ((side//16)**2, 1024) or r.shape != q.shape:
                raise ValueError('Actual DINOv3-L patch count, resolution and channels must match the producer')
            if q.dtype not in (np.dtype('float32'), np.dtype('float64')) or r.dtype != q.dtype:
                raise ValueError('Native FP32 or its explicit FP64 unit export required; old FP16 cache is excluded')
            assets = producer.get('model_assets', producer)
            if len(assets.get('checkpoint_sha256', '')) != 64:
                raise ValueError('Frozen DINOv3 checkpoint identity required')
            result = predict(q, r, packet['foreground_weight'], packet['valid_weight'],
                json.loads(str(packet['query_geometry_json'])), row['original_shape'])
        case = out/row['id']
        case.mkdir()
        np.savez_compressed(case/'prediction.npz', field=result['field'], margin=result['margin'],
                            mask_work=result['work'], mask_original=result['original'])
        Image.fromarray(result['original'].astype(np.uint8)*255).save(case/'mask_original.png')
        results.append(dict(id=row['id'], producer=producer, info=result['info'], seconds=time.perf_counter()-tick,
            predicted_original_pixels=int(result['original'].sum()), original_shape=row['original_shape'],
            hashes={name: sha(case/name) for name in ('prediction.npz', 'mask_original.png')}))
    dump(out/'sealed.json', dict(state='DINO_ONLY_COMPLETE_PREDICTIONS_SEALED', results=results,
        manifest_sha256=sha(manifest), query_GT_read=False, quality_scored=False,
        inference_source_snapshot_hashes=snapshots,
        source_hashes={str(p): sha(p) for p in (Path(__file__), ROOT/'src/ics/methods/direct_dino_features.py')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    a = sub.add_parser('export-local'); a.add_argument('--source', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('extract')
    a.add_argument('--threads', type=int, default=2)
    for name in ('model-dir', 'reference-rgb', 'reference-mask', 'query-rgb', 'out'):
        a.add_argument('--'+name, type=Path, required=True)
    a = sub.add_parser('infer'); a.add_argument('--manifest', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.stage == 'export-local': export_local(args.source, args.out)
    elif args.stage == 'extract': extract(args)
    else: infer(args.manifest, args.out)


if __name__ == '__main__':
    main()
