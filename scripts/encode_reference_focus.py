"""Encode fixed legal reference focus RGB crops, without any query or head.

The selected box is frozen from reference-mask-only geometry. Its actual RGB
input is official canonical1024 resize ->512 crop ->official1024 transform.
Only distinct focus inputs are paired on MPS; existing raw entries are kept.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import builtins
from collections import Counter
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'cv_data'
DEFAULT_MANIFEST = DATA / 'a/joint_role_pilot200_20261010/manifest.json'
DEFAULT_AUDIT = REPO / 'evidence/local/autonomous_20261010/mechanism_research/reference_focus_mask_audit_torch_nearest.json'
DEFAULT_OUT = DATA / 'a/reference_focus200_20261010'
DEFAULT_PROFILE = DATA / 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class ReferenceOnlyGuard:
    """Deny query annotations and RGB sources not also legal reference files."""
    def __init__(self, manifest, assets):
        self.reference_images = {str((assets / row['reference_path']).resolve()) for row in manifest}
        self.forbidden = set()
        self.query_path_reference_aliases = set()
        for row in manifest:
            query_image = str((assets / row['query_path']).resolve())
            if query_image in self.reference_images:
                self.query_path_reference_aliases.add(query_image)
            else:
                self.forbidden.add(query_image)
            for field in ('query_mask_path', 'query_ignore_mask_path'):
                if field in row:
                    self.forbidden.add(str(Path(row[field]).resolve()))
            if row.get('query_annotation_path'):
                self.forbidden.add(str((assets / row['query_annotation_path']).resolve()))
        self.attempts = 0

    def __enter__(self):
        from PIL import Image
        self.original = builtins.open, io.open, Image.open
        def guarded(original):
            def call(path, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    resolved = str(Path(os.fsdecode(path)).resolve())
                    if resolved in self.forbidden:
                        self.attempts += 1
                        raise RuntimeError('Query source denied for reference focus: ' + resolved)
                return original(path, *args, **kwargs)
            return call
        builtins.open, io.open, Image.open = map(guarded, self.original)
        return self

    def __exit__(self, *args):
        from PIL import Image
        builtins.open, io.open, Image.open = self.original


def imports(root):
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts'), str(root / 'external')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import decoded_rgb, array_hash
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from utils.data import build_transform
    return np, torch, F, decoded_rgb, array_hash, RawFeatureCache, tensor_hash, build_transform


def prepare(args):
    audit = read(args.audit)
    manifest = read(args.manifest)
    if audit['status'] != 'REFERENCE_MASK_ONLY_COMPLETE_TORCH_NEAREST' or audit['n'] != 200 or len(manifest) != 200:
        raise ValueError('Require the completed fixed reference-only200 audit')
    if (audit['manifest_sha256'] != sha(args.manifest) or audit['query_images_read'] != 0
            or audit['query_masks_read'] != 0 or audit['encoder_calls'] != 0):
        raise ValueError('Audit input/exposure identity changed')
    if len(audit['episodes']) != 200 or len({row['episode_id'] for row in manifest}) != 200:
        raise ValueError('Focus episode count/identity differs')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Prepare a fresh focus run; existing frozen inputs must remain intact')
    args.out.mkdir(parents=True)
    sources = {}
    for relative in ['scripts/encode_reference_focus.py', 'scripts/raw_feature_cache.py',
                     'src/ics/__init__.py', 'src/ics/data.py', 'src/ics/official_data.py', 'src/ics/representations.py']:
        destination = args.out / 'frozen' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        sources[relative] = sha(destination)
    for relative in ['utils/__init__.py', 'utils/data.py']:
        destination = args.out / 'frozen/external' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.assets / 'third_party/foris_official' / relative, destination)
        sources['external/' + relative] = sha(destination)
    shutil.copyfile(args.manifest, args.out / 'source_manifest.json')
    shutil.copyfile(args.audit, args.out / 'selection_audit.json')
    shutil.copyfile(args.profile, args.out / 'profile.json')
    profile = read(args.profile)
    np, torch, F, decoded_rgb, array_hash, cache_class, tensor_hash, build_transform = imports(args.out / 'frozen')
    torch.set_num_threads(2)
    cache = cache_class(args.profile.parent.parent, profile)
    transform = build_transform(1024)
    weights = args.assets / 'demo4_cache/models/dinov3-vitl16-timm'
    if (sha(weights / 'model.safetensors') != profile['weights_sha256']
            or sha(weights / 'config.json') != profile['model_config_sha256']
            or sources['src/ics/data.py'] != profile['encoder_source_sha256']
            or sources['src/ics/representations.py'] != profile['observer_source_sha256']
            or sources['external/utils/data.py'] != profile['preprocessing']['source_sha256']):
        raise ValueError('Exact existing MPS raw profile differs')
    tasks, unique = [], {}
    guard = ReferenceOnlyGuard(manifest, args.assets)
    with guard:
        for index, (row, selected) in enumerate(zip(manifest, audit['episodes'])):
            if row['episode_id'] != selected['episode_id'] or row['reference_mask_hash'] != selected['reference_mask_hash']:
                raise ValueError('Audit order or reference labels changed')
            box = tuple(selected['box_xyxy'])
            if (len(box) != 4 or any(type(x) is not int or x % 16 for x in box)
                    or box[2] - box[0] != 512 or box[3] - box[1] != 512
                    or min(box[:2]) < 0 or max(box[2:]) > 1024):
                raise ValueError('Require the frozen16-aligned512 box inside canonical1024')
            rgb_path = args.assets / row['reference_path']
            reference = decoded_rgb(rgb_path, row.get('reference_crop'))
            if array_hash(np.asarray(reference)) != row['reference_rgb_hash']:
                raise ValueError('Legal reference RGB differs')
            from PIL import Image
            with Image.open(row['reference_mask_path']) as image:
                mask = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
            if array_hash(mask) != row['reference_mask_hash'] or list(mask.shape) != row['reference_mask_size_hw']:
                raise ValueError('Frozen reference mask differs')
            canonical = transform.transforms[0](reference)
            focus = canonical.crop(box)
            canonical_mask = F.interpolate(torch.from_numpy(mask)[None, None].float(), (1024, 1024), mode='nearest')[0, 0]
            cropped_mask = canonical_mask[box[1]:box[3], box[0]:box[2]].numpy().astype(np.uint8)
            model_mask = F.interpolate(torch.from_numpy(cropped_mask)[None, None].float(), (1024, 1024), mode='nearest')
            coverage = F.avg_pool2d(model_mask, 16, stride=16)[0, 0].numpy()
            canonical_mask_hash = array_hash(canonical_mask.numpy().astype(np.uint8))
            if (canonical_mask_hash != selected['canonical_mask_uint8_array_sha256']
                    or tensor_hash(coverage) != selected['focus_coverage_fp32_array_sha256']):
                raise ValueError('Exact torch-nearest mask/coverage receipt differs')
            if abs(float(coverage.sum()) - selected['focus']['fg_mass_tokens']) > 1e-6:
                raise ValueError('Selected crop coverage differs from the legal mask audit')
            input_tensor = transform(focus).numpy()
            key = cache.key(input_tensor)
            rgb_destination = args.out / 'inputs/rgb' / f'{index:06d}.png'
            mask_destination = args.out / 'inputs/reference_mask512' / f'{index:06d}.png'
            coverage_destination = args.out / 'inputs/coverage64' / f'{index:06d}.npy'
            for destination in (rgb_destination, mask_destination, coverage_destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
            focus.save(rgb_destination)
            Image.fromarray(cropped_mask * 255).save(mask_destination)
            np.save(coverage_destination, coverage, allow_pickle=False)
            task = dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'], box_xyxy=list(box),
                reference_path=row['reference_path'], reference_crop=row.get('reference_crop'),
                reference_photo_id=row['reference_photo_id'], reference_rgb_hash=row['reference_rgb_hash'],
                reference_source_file_sha256=sha(rgb_path), reference_mask_path=row['reference_mask_path'],
                reference_mask_hash=row['reference_mask_hash'], reference_mask_file_sha256=sha(Path(row['reference_mask_path'])),
                canonical_rgb_hash=array_hash(np.asarray(canonical)), focus_rgb_hash=array_hash(np.asarray(focus)),
                canonical_mask_uint8_array_sha256=canonical_mask_hash,
                focus_rgb_path=str(rgb_destination), focus_rgb_file_sha256=sha(rgb_destination),
                focus_mask512_path=str(mask_destination), focus_mask512_hash=array_hash(cropped_mask),
                focus_mask512_file_sha256=sha(mask_destination), coverage64_path=str(coverage_destination),
                coverage64_file_sha256=sha(coverage_destination), coverage64_tensor_sha256=tensor_hash(coverage),
                foreground_model_token_mass=float(coverage.sum()), key=key,
                input_tensor_hash=tensor_hash(input_tensor), initially_cached=(cache.folder / key / 'entry.json').exists())
            tasks.append(task)
            unique.setdefault(key, index)
    write(args.out / 'tasks.json', tasks)
    write(args.out / 'config.json', dict(n=200, datasets=dict(Counter(row['dataset'] for row in manifest)),
        source_manifest_path=str(args.manifest), source_manifest_sha256=sha(args.manifest),
        selection_audit_sha256=sha(args.audit), tasks_sha256=sha(args.out / 'tasks.json'),
        profile_path=str(args.profile), profile_sha256=sha(args.profile), profile_id=cache.profile_id,
        weights_path=str(weights), weights_sha256=profile['weights_sha256'], source_sha256=sources,
        original_assets=str(args.assets), producer_device='mps', dtype='float32', batch_size=2,
        unique_focus_inputs=len(unique), initially_cached_unique=sum(tasks[i]['initially_cached'] for i in unique.values()),
        exact_input='official.Resize1024(legal reference RGB), fixed512 RGBcrop, same official1024 transform',
        mask_rule='canonical1024 nearest reference mask; same box; nearest1024; exact16x16 patch area fraction',
        new_information_scope='actual frozen DINO O24 from true RGB crop; no erased/masked RGB, derived whole feature, head or score',
        selection_rule=audit['rule'], no_existing_cache_entry_rewrites=True, query_RGB_actors_called=0,
        query_GT_reads=0, denied_query_source_attempts=guard.attempts,
        shared_file_reference_query_aliases=len(guard.query_path_reference_aliases)))
    (args.out / 'records').mkdir()
    print(json.dumps(dict(state='PREPARED_REFERENCE_FOCUS', n=200, unique_inputs=len(unique),
        cached_unique=sum(tasks[i]['initially_cached'] for i in unique.values()))), flush=True)


def verify(out):
    cfg = read(out / 'config.json')
    for path, key in [('source_manifest.json', 'source_manifest_sha256'),
                      ('selection_audit.json', 'selection_audit_sha256'), ('tasks.json', 'tasks_sha256')]:
        if sha(out / path) != cfg[key]:
            raise ValueError('Frozen selection/input identity changed: ' + path)
    if sha(Path(cfg['profile_path'])) != cfg['profile_sha256']:
        raise ValueError('Raw-cache profile changed')
    for relative, digest in cfg['source_sha256'].items():
        if sha(out / 'frozen' / relative) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    return cfg


def encode(out):
    cfg = verify(out)
    np, torch, _, _, array_hash, cache_class, tensor_hash, build_transform = imports(out / 'frozen')
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError('This exact producer requires MPS')
    profile_path = Path(cfg['profile_path'])
    cache = cache_class(profile_path.parent.parent, read(profile_path))
    transform = build_transform(1024)
    tasks = read(out / 'tasks.json')
    representatives = {}
    for task in tasks:
        representatives.setdefault(task['key'], task)
    encoded, batches, encoder = set(), [], None
    start = time.monotonic()
    from PIL import Image
    def actual_input(task):
        if sha(Path(task['focus_rgb_path'])) != task['focus_rgb_file_sha256']:
            raise ValueError('Frozen focus RGB file changed')
        with Image.open(task['focus_rgb_path']) as im:
            image = im.convert('RGB')
        if array_hash(np.asarray(image)) != task['focus_rgb_hash']:
            raise ValueError('Frozen focus RGB pixels changed')
        x = transform(image)
        if cache.key(x.numpy()) != task['key'] or tensor_hash(x.numpy()) != task['input_tensor_hash']:
            raise ValueError('Actual model input differs from its pre-encoding freeze')
        return x
    missing = [task for task in representatives.values() if not (cache.folder / task['key'] / 'entry.json').is_file()]
    guard = ReferenceOnlyGuard(read(out / 'source_manifest.json'), Path(cfg['original_assets']))
    with torch.inference_mode(), guard:
        for offset in range(0, len(missing), 2):
            pair = missing[offset:offset + 2]
            if len(pair) == 1:
                pair.append(next(task for task in representatives.values() if task['key'] != pair[0]['key']))
            if pair[0]['key'] == pair[1]['key']:
                raise ValueError('Require two distinct focus model inputs')
            if encoder is None:
                from ics.data import TimmDINOv3
                init_start = time.monotonic()
                encoder = TimmDINOv3(cfg['weights_path']).to('mps').eval().requires_grad_(False)
                write(out / 'encoder_initialization.json', dict(seconds=time.monotonic() - init_start,
                    weights_sha256=cfg['weights_sha256'], device='mps', dtype='float32'))
            pair_inputs = torch.stack([actual_input(task) for task in pair])
            tick = time.monotonic()
            maps = encoder.get_intermediate_layers(pair_inputs.to('mps'), n=1, reshape=True)[0]
            torch.mps.synchronize()
            arrays = maps.cpu().flatten(2).transpose(1, 2).contiguous().numpy()
            encoding_seconds = time.monotonic() - tick
            created = []
            for batch_index, task in enumerate(pair):
                x = pair_inputs[batch_index].numpy()
                with cache._locked(x, True):
                    entry = cache.folder / task['key'] / 'entry.json'
                    if entry.exists():
                        if not np.array_equal(cache._read(x, ('O/24',))['O/24'], arrays[batch_index]):
                            raise ValueError('Repeated focus input changed its raw O24 values')
                    else:
                        cache._write(x, {'O/24': arrays[batch_index]}, dict(experiment=str(out),
                            episode_id=task['episode_id'], view_role='reference_focus512',
                            box_xyxy=task['box_xyxy'], input_protocol=cfg['exact_input'],
                            pair_keys=[t['key'] for t in pair], batch_size=2, batch_index=batch_index,
                            reference_labels_used_for_box_only=True, query_RGB_reads=0, query_GT_reads=0))
                        encoded.add(task['key'])
                        created.append(task['key'])
            del maps, arrays
            batch = dict(batch_index=len(batches), pair_keys=[task['key'] for task in pair],
                pair_episode_ids=[task['episode_id'] for task in pair], newly_written_keys=created,
                encoding_seconds=encoding_seconds, cumulative_seconds=time.monotonic() - start)
            batches.append(batch)
            write(out / 'batches.json', batches)
            write(out / 'activity.json', dict(state='ENCODING', new_unique_inputs=len(encoded),
                total_missing=len(missing), completed_batches=len(batches), seconds=time.monotonic() - start))
            print(json.dumps(dict(batch=len(batches), n_new=len(encoded), total_missing=len(missing),
                seconds=time.monotonic() - start)), flush=True)
        for task in tasks:
            x = actual_input(task).numpy()
            raw = cache.read(x, ('O/24',))['O/24']
            entry = cache.folder / task['key'] / 'entry.json'
            info = read(entry)
            payload = entry.parent / info['file']
            if (sha(payload) != info['file_sha256'] or tensor_hash(raw) != info['features']['O/24']['tensor_sha256']
                    or info['input_tensor_hash'] != task['input_tensor_hash']):
                raise ValueError('Actual payload/tensor readback differs')
            if sha(Path(task['coverage64_path'])) != task['coverage64_file_sha256']:
                raise ValueError('Frozen legal reference coverage changed')
            record = dict(**task, entry_path=str(entry), entry_sha256=sha(entry), payload_path=str(payload),
                payload_sha256=info['file_sha256'], O24=info['features']['O/24'],
                actual_readback_verified=True, new_raw_entry=task['key'] in encoded)
            write(out / 'records' / f"{task['index']:06d}.json", record)
    records = [read(out / 'records' / f'{index:06d}.json') for index in range(len(tasks))]
    write(out / 'input_bindings.json', records)
    bytes_added = sum(Path(next(r['payload_path'] for r in records if r['key'] == key)).stat().st_size for key in encoded)
    seal = dict(state='REFERENCE_FOCUS_O24_COMPLETE', n=200, unique_focus_inputs=len(representatives),
        new_raw_inputs=len(encoded), cache_hit_unique=len(representatives) - len(encoded),
        duplicate_episode_views=200 - len(representatives), paired_encoder_calls=len(batches),
        encoded_input_roles=2 * len(batches), new_payload_bytes=bytes_added,
        encoding_seconds=sum(batch['encoding_seconds'] for batch in batches), wall_seconds=time.monotonic() - start,
        source_manifest_sha256=cfg['source_manifest_sha256'], selection_audit_sha256=cfg['selection_audit_sha256'],
        config_sha256=sha(out / 'config.json'), tasks_sha256=sha(out / 'tasks.json'),
        input_bindings_sha256=sha(out / 'input_bindings.json'), profile_sha256=cfg['profile_sha256'],
        actual_payload_and_O24_tensor_hashes_all_verified=True, query_RGB_actors_called=0, query_GT_reads=0,
        denied_query_source_attempts=guard.attempts, FoRIS_calls=0, candidate_calls=0, scoring_calls=0,
        scientific_claim='Actual new focus semantic vectors are available; geometric purity or encoding alone is not a performance gain.')
    write(out / 'sealed.json', seal)
    write(out / 'COMPLETE.json', seal)
    write(out / 'activity.json', dict(state='COMPLETE', n=200, unique_inputs=len(representatives),
        input_bindings_sha256=seal['input_bindings_sha256']))
    print(json.dumps(seal), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'encode', 'run'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--assets', type=Path, default=DATA)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    for name in ('out', 'assets', 'manifest', 'audit', 'profile'):
        setattr(args, name, getattr(args, name).resolve())
    if args.mode in ('prepare', 'run'):
        prepare(args)
        if args.mode == 'run':
            os.execv(sys.executable, [sys.executable, str(args.out / 'frozen/scripts/encode_reference_focus.py'),
                                     'encode', '--out', str(args.out)])
        return
    with open(args.out / 'controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        encode(args.out)


if __name__ == '__main__':
    main()
