#!/usr/bin/env python3
"""Read-only finite TRAIN/DEV preparation using existing same-UID JPEG assets.

Run --cpu-fixture for ten temporary image/mask pairs and the exact production
preparation path. No torch/model import, training, extraction, download or GPU.
An existing output JSON is never overwritten, on success or failure.
"""
import argparse
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
import time

from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
# Importing tics initializes torch through its existing __init__.py. Load
# only the bounded pure asset source and its exact metadata validator.
_asset_spec = importlib.util.spec_from_file_location(
    '_reference_response_assets_cpu', BASE/'tics/reference_response_assets.py')
asset_module = importlib.util.module_from_spec(_asset_spec)
_asset_spec.loader.exec_module(asset_module)
AssetPreparationError = asset_module.AssetPreparationError
ExistingCocoAssets = asset_module.ExistingCocoAssets
file_sha256 = asset_module.file_sha256
frozen_test_rows = asset_module.frozen_test_rows
png_header_size = asset_module.png_header_size
prepare_data = asset_module.prepare_data
read_official_index = asset_module.read_official_index
validate_episode_manifest = asset_module.validate_episode_manifest


def cpu_fixture():
    checks = []
    def record(name, **evidence):
        checks.append(dict(name=name, passed=True, **evidence))
    def reject(function, reason):
        try:
            function()
        except (ValueError, OSError) as exc:
            assert reason in str(exc), (reason, str(exc))
            return
        raise AssertionError('Invalid data contract accepted')
    with tempfile.TemporaryDirectory(prefix='response-assets-fixture-') as temporary:
        root = Path(temporary)
        official = root/'official/COCO2014'
        val = root/'existing/COCO2014'
        train = root/'LVIS/coco/train2017'
        train.mkdir(parents=True)
        def logical(split, uid):
            return f'{split}/COCO_{split}_{uid:012d}.jpg'
        def pair(split, uid, *, dimensions=(10, 8), annotation_size=None, foreground=True):
            name = logical(split, uid)
            jpeg = train/f'{uid:012d}.jpg' if split == 'train2014' else val/name
            mask = official/'annotations'/Path(name).with_suffix('.png')
            jpeg.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', dimensions, (30, 40, 50)).save(jpeg)
            labels = Image.new('L', annotation_size or dimensions)
            if foreground:
                # Covers every c+1 without inventing a VOID policy.
                labels.putdata(list(range(1, 81))[:labels.width*labels.height])
            labels.save(mask)
            return name
        trn = [pair('train2014', uid) for uid in (1, 2)]
        dev = [pair('val2014', uid) for uid in (101, 102)]
        support, query = [pair('val2014', uid) for uid in (201, 202)]
        wrong_shape = pair('train2014', 3, annotation_size=(11, 8))
        wrong_label = pair('train2014', 4, foreground=False)
        cross_split = pair('val2014', 1)
        header_only = pair('val2014', 204)
        assert len(list(root.rglob('*.jpg'))) == 10
        frozen = dict(schema='native_membership_assets_v1', fold_count=4, per_fold=10, seed=0,
                      frozen_episodes=[dict(e=i, c=fold+4*i, fold=fold, support=support, query=query)
                                       for fold in range(4) for i in range(10)])
        manifest_path = root/'manifest40.json'
        manifest_path.write_text(json.dumps(frozen))
        for fold in range(4):
            for kind in ('trn', 'val'):
                path = official/f'splits/{kind}/fold{fold}.pkl'
                path.parent.mkdir(parents=True, exist_ok=True)
                if kind == 'trn':
                    mapping = {c: trn if c % 4 != fold else [] for c in range(80)}
                else:
                    mapping = {c: dev for c in range(fold, 80, 4)}
                with path.open('wb') as stream:
                    pickle.dump(mapping, stream)
        assets = ExistingCocoAssets(official, val, train)
        inspected = assets.inspect(trn[0], 1, split='train', role='support')
        assert inspected['uid'] == 1 and Path(inspected['physical_image']).name == '000000000001.jpg'
        assert inspected['logical_image'].startswith('train2014/COCO_train2014_')
        assert '/annotations/train2014/' in inspected['official_mask']
        record('same_UID_2014_logical_name_to_existing_2017_JPEG_and_2014_official_mask', ten_temporary_pairs=True)
        reject(lambda: assets.inspect(wrong_shape, 1, split='train', role='query'), 'geometry mismatch')
        record('JPEG_PNG_shape_mismatch_rejected')
        reject(lambda: assets.inspect(wrong_label, 1, split='dev', role='query'), 'no c+1 foreground')
        record('missing_c_plus_one_label_rejected')
        bad_index = official/'bad.pkl'
        with bad_index.open('wb') as stream:
            pickle.dump({0: trn, 1: trn}, stream)
        reject(lambda: read_official_index(bad_index, split='train2014', expected_classes={1}), 'Held/wrong class')
        good_index = official/'splits/trn/fold0.pkl'
        index = read_official_index(good_index, split='train2014', expected_classes=set(range(80))-set(range(0, 80, 4)))
        assert len(index) == 60 and all(c % 4 != 0 for c in index)
        record('held_nonempty_pool_rejected_but_official_80_keys_with_20_empty_held_pools_accepted')
        small = dict(schema='reference_response_episodes_v1', annotation_protocol='official_COCO20i',
                     held_fold=0, coco_year=2014, test_scope='reused_development',
                     train=[dict(id='train', c=1, support=trn[0], query=trn[1])],
                     dev=[dict(id='dev', c=2, support=dev[0], query=dev[1])],
                     test=[dict(id='test', c=0, support=support, query=query)])
        leaked = deepcopy(small)
        leaked['dev'][0]['query'] = cross_split
        reject(lambda: validate_episode_manifest(leaked, prior_development_photo_ids={201, 202}), 'Photo leakage')
        leaked = deepcopy(small)
        leaked['dev'][0]['support'] = query
        reject(lambda: validate_episode_manifest(leaked, prior_development_photo_ids={201, 202}), 'Photo leakage')
        record('all_role_TRAIN_DEV_and_DEV_TEST_UID_leakage_rejected')
        changed = deepcopy(frozen)
        changed['frozen_episodes'][0]['c'] = 1
        reject(lambda: frozen_test_rows(changed), 'held class')
        changed = deepcopy(frozen)
        changed['frozen_episodes'].pop()
        reject(lambda: frozen_test_rows(changed), 'Exactly')
        record('frozen_TEST_held_class_or_episode_count_change_rejected')
        # The query file has a valid dimension header and deliberately invalid
        # pixel payload. Preparation must pass, proving no PNG pixel decoder
        # can have been run on it. A guard also rejects the explicit reader.
        query_mask = assets.paths(query)[2]
        query_mask.write_bytes(query_mask.read_bytes()[:24])
        another_mask = assets.paths(header_only)[2]
        another_mask.write_bytes(another_mask.read_bytes()[:24])
        assert png_header_size(another_mask) == (10, 8)
        original_reader = asset_module.foreground_pixels
        def guarded_pixels(path, category):
            if Path(path) == query_mask:
                raise AssertionError('TEST query pixels accessed')
            return original_reader(path, category)
        asset_module.foreground_pixels = guarded_pixels
        try:
            prepared = prepare_data(official, val, train, manifest_path)
        finally:
            asset_module.foreground_pixels = original_reader
        assert prepared['frozen_test_episodes'] == frozen['frozen_episodes']
        assert len(prepared['test_query_checks']) == 40 and all(row['PNG_bytes_read'] == 24 for row in prepared['test_query_checks'])
        assert not prepared['exposure_registry_complete'] and prepared['prior_development_photo_ids'] == [201, 202]
        assert all(row['protocol_validation']['test_exposure_scope'] == 'reused_development' for row in prepared['folds'])
        record('TEST_query_pixels_forbidden_guard_and_truncated_payload_header_only_proof',
               TEST_query_checks=40, PNG_bytes_per_check=24, fresh_confirmation=False)
        assert all(len(row['manifest']['train']) == 60 and len(row['manifest']['dev']) == 60
                   and len(row['manifest']['test']) == 10 for row in prepared['folds'])
        assert all(len(prepared['availability'][str(fold)]) == 60 for fold in range(4))
        assert all(all(pool['train']['same_UID_existing_JPEG_and_official_PNG'] == 2
                       for pool in prepared['availability'][str(fold)].values()) for fold in range(4))
        record('full_preparation_60_base_classes_each_fold_exact_1_TRAIN_1_DEV_and_frozen_10_TEST',
               train_episodes=240, dev_episodes=240, frozen_test_episodes=40,
               full_training_or_gain_claim=False)
        # The source treats every value other than c+1 as background.
        special = assets.paths(wrong_label)[2]
        labels = Image.new('L', (10, 8), 255)
        labels.putpixel((0, 0), 2)
        labels.save(special)
        assert original_reader(special, 1) == 1 and original_reader(special, 0) == 0
        record('INSID3_binary_semantic_equal_c_plus_one_including_255_as_background')
        missing = train/'000000000002.jpg'
        missing.unlink()
        try:
            prepare_data(official, val, train, manifest_path)
        except AssetPreparationError as exc:
            assert 'Insufficient' in str(exc) and exc.evidence['insufficient']
            assert len(exc.evidence['availability']['0']) == 60
        else:
            raise AssertionError('Insufficient class silently dropped/replicated')
        record('insufficient_assets_explicit_with_all_60_class_counts_no_drop_or_replicate')
    assert not root.exists()
    assert 'torch' not in sys.modules
    return dict(schema='reference_response_assets_cpu_v1', state='CPU_DATA_PREPARATION_FIXTURE_PASSED',
                checks=checks, passed=len(checks), temporary_image_mask_pairs=10, fixtures_removed=True,
                torch_imported=False, training_executed=False, real_task_gain_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, default=Path('/root/autodl-tmp/datasets/ics/COCO2014'))
    parser.add_argument('--val-image-root', type=Path, default=Path('/root/demo4_cache/data/COCO2014'))
    parser.add_argument('--existing-train-images', type=Path, default=Path('/root/autodl-tmp/datasets/ics/LVIS/coco/train2017'))
    parser.add_argument('--test-manifest', type=Path, default=BASE/'results/native_membership_v1/causal_v3/manifest40.json')
    parser.add_argument('--seed', type=int, default=31027)
    parser.add_argument('--cpu-fixture', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing JSON proof: '+str(args.out))
    started = time.monotonic()
    exit_code = 0
    try:
        report = cpu_fixture() if args.cpu_fixture else prepare_data(
            args.official_root, args.val_image_root, args.existing_train_images, args.test_manifest, seed=args.seed)
    except Exception as exc:
        report = dict(schema='reference_response_data_preparation_v1', state='ERROR',
                      reason=type(exc).__name__+': '+str(exc), evidence=getattr(exc, 'evidence', {}),
                      training_executed=False, torch_imported='torch' in sys.modules, real_task_gain_measured=False)
        exit_code = 2
    report['elapsed_seconds'] = time.monotonic()-started
    report.setdefault('source_sha256', {}).update({str(Path(__file__).resolve()): file_sha256(__file__),
                                                str(Path(asset_module.__file__).resolve()): file_sha256(asset_module.__file__)})
    if 'torch' in sys.modules:
        raise AssertionError('Preparation unexpectedly imported torch')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state=report['state'], passed=report.get('passed'),
                          receipt=str(args.out), elapsed_seconds=report['elapsed_seconds'],
                          torch_imported=False, training_executed=False)))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
