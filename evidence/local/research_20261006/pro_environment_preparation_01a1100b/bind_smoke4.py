"""Bind authorized smoke4 RGB, full reference masks and explicit cached-native renderer."""
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
torch.set_num_threads(1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def main():
    work = Path(sys.argv[1])
    work.mkdir(exist_ok=True)
    bound = work/'bound'
    bound.mkdir(exist_ok=False)
    source = Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/bound600_v2/smoke4.json')
    rows = json.loads(source.read_text())
    rgb_root = Path('/root/demo4_cache/data/COCO2014')
    annotation_root = Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
    manifest, records = [], []
    for row in rows:
        reference, query = rgb_root/row['support'], rgb_root/row['query']
        annotation = annotation_root/row['support'].replace('.jpg', '.png')
        with Image.open(annotation) as image:
            full_reference_mask = np.asarray(image) == row['c']+1
        with Image.open(reference) as image:
            reference_hw = image.size[::-1]
        with Image.open(query) as image:
            query_hw = image.size[::-1]
        if (tuple(full_reference_mask.shape) != reference_hw or
                list(reference_hw) != row['support_image_hw'] or list(query_hw) != row['query_image_hw']):
            raise ValueError('Original RGB/annotation/HW binding mismatch')
        mask_path = bound/(row['key']+'_reference_mask.png')
        Image.fromarray(full_reference_mask.astype(np.uint8)).save(mask_path)
        with np.load(row['packet_export'], allow_pickle=False) as packet:
            packed_native = packet['native'].copy()  # Never read packet.truth.
        if packed_native.shape != (131072,) or packed_native.dtype != np.uint8:
            raise ValueError('Unexpected native cache schema')
        native_work = np.unpackbits(packed_native).reshape(1024, 1024).astype(bool)
        native_original = (functional.interpolate(torch.from_numpy(native_work.astype(np.float32))[None, None],
                            query_hw, mode='bilinear', align_corners=False)[0, 0].numpy() > .5)
        native_path = bound/(row['key']+'_native.npz')
        np.savez_compressed(native_path, mask_work=native_work.astype(np.uint8),
                            mask_original=native_original.astype(np.uint8))
        manifest.append(dict(id=row['key'], reference_rgb=str(reference), reference_mask=str(mask_path),
                             query_rgb=str(query), native_npz=str(native_path), native_sha256=sha(native_path)))
        records.append(dict(id=row['key'], reference_annotation=str(annotation), reference_class_id=row['c']+1,
                            reference_annotation_sha256=sha(annotation), full_reference_mask_sha256=sha(mask_path),
                            source_packet=str(row['packet_export']), source_packet_sha256=sha(row['packet_export']),
                            source_packet_keys_read=['native'], original_reference_hw=reference_hw,
                            original_query_hw=query_hw, reference_rgb_sha256=sha(reference), query_rgb_sha256=sha(query),
                            reference_foreground_pixels=int(full_reference_mask.sum()),
                            native_original_variant='Pro explicit bilinear reconstruction from cached1024 binary native; original native output not loaded',
                            native_npz_sha256=sha(native_path), query_gt_read=False,
                            query_annotation=str(annotation_root/row['query'].replace('.jpg', '.png'))))
    recipe = dict(method='full_foris_native', variant='cached_native_work_plus_Pro_original_renderer',
                  cache_schema='packet.native packed1024 binary full-native output',
                  work_renderer='already sealed native1024 binary; not reencoded or re-CRFed',
                  original_renderer='FP32 binary1024 bilinear to actual RGB H/W,align_corners=False,strict>0.5',
                  source_manifest_sha256=sha(source), native_sources=[record['source_packet_sha256'] for record in records],
                  actual_native_deployment_cost='not measured; this is a cache renderer variant')
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    native_binding = dict(method='full_foris_native', recipe_sha256=recipe_hash, recipe=recipe)
    model_root = Path('/root/demo4_cache/models/dinov3-vitl16-timm')
    encoder_binding = dict(producer='local_timm1.0.30_DINOv3_vit_large_patch16_last_LN_FP32_RGB1024_ImageNetnorm',
                           config_sha256=sha(model_root/'config.json'), checkpoint_sha256=sha(model_root/'model.safetensors'),
                           timm_eva_sha256=sha('/root/demo4_cache/env/timm/models/eva.py'),
                           source='existing local TimmDINOv3, no downloads or GPU', dtype='float32')
    dump(work/'manifest4.json', manifest)
    dump(work/'manifest_first.json', manifest[:1])
    dump(work/'manifest_remaining.json', manifest[1:])
    dump(work/'evaluation_binding.json', records)
    dump(work/'native_binding.json', native_binding)
    dump(work/'encoder_binding.json', encoder_binding)
    dump(work/'binding_receipt.json', dict(source_manifest=str(source), source_manifest_sha256=sha(source),
         count=len(records), query_gt_read=False, reference_mask_source='original semantic annotation == declared reference class+1',
         native_original_variant=recipe['variant'], native_recipe_sha256=recipe_hash))
    print(json.dumps(dict(bound=len(manifest), native_variant=recipe['variant'], query_gt_read=False)))


if __name__ == '__main__':
    main()
