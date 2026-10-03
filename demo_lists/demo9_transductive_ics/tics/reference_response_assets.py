"""Read-only COCO2014 logical assets, with same-UID existing train2017 JPEGs.

No tensor/model library, annotation-derived query features, downloads or copies.
TEST query PNG dimensions are read directly from the 24-byte PNG header; its
pixels are never decoded. Binary labels retain INSID3's exact semantic == c+1
policy: every other value, including 255, is background, with no new void mask.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import pickle
import random
import struct

from PIL import Image

# tics/__init__.py imports model code; load this exact pure metadata source
# directly so CPU asset preparation never initializes that package.
_protocol_spec = importlib.util.spec_from_file_location(
    '_reference_response_protocol_assets', Path(__file__).with_name('reference_response_protocol.py'))
_protocol = importlib.util.module_from_spec(_protocol_spec)
_protocol_spec.loader.exec_module(_protocol)
coco_photo_id = _protocol.coco_photo_id
validate_episode_manifest = _protocol.validate_episode_manifest


class AssetPreparationError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def png_header_size(path):
    """Strict header-only operation: never Pillow/load/verify/getdata on query GT."""
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
            or header[8:12] != b'\x00\x00\x00\x0d' or header[12:16] != b'IHDR'):
        raise AssetPreparationError('Invalid PNG dimension header: '+str(path))
    width, height = struct.unpack('>II', header[16:24])
    if not width or not height:
        raise AssetPreparationError('Empty PNG dimensions: '+str(path))
    return width, height


def foreground_pixels(path, category):
    """Legal TRAIN/DEV labels or labelled support only; semantic == category+1."""
    if type(category) is not int or category not in range(80):
        raise AssetPreparationError('Invalid dense COCO category')
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode not in ('L', 'P'):
            raise AssetPreparationError('Official semantic annotation must be 8-bit indexed PNG')
        return image.histogram()[category+1]


class ExistingCocoAssets:
    def __init__(self, official_root, val_image_root, existing_train_images):
        self.official_root = Path(official_root).resolve()
        self.val_image_root = Path(val_image_root).resolve()
        self.existing_train_images = Path(existing_train_images).resolve()
        self.records = {}
        self.header_only_test_queries = []
        self.foreground_checks = []

    def paths(self, logical_name):
        uid = coco_photo_id(logical_name)
        split = PurePosixPath(logical_name).parts[0]
        if split == 'train2014':
            jpeg = self.existing_train_images/f'{uid:012d}.jpg'
        elif split == 'val2014':
            jpeg = self.val_image_root/logical_name
        else:
            raise AssetPreparationError('Only original COCO2014 logical splits are permitted')
        mask = self.official_root/'annotations'/Path(logical_name).with_suffix('.png')
        return uid, jpeg, mask

    def paired_exists(self, logical_name):
        _, jpeg, mask = self.paths(logical_name)
        return jpeg.is_file() and mask.is_file()

    def inspect(self, logical_name, category, *, split, role):
        if split not in ('train', 'dev', 'test') or role not in ('support', 'query'):
            raise AssetPreparationError('Explicit episode split/role is required')
        uid, jpeg, mask = self.paths(logical_name)
        for path in (jpeg, mask):
            if not path.is_file() or path.stat().st_size == 0:
                raise AssetPreparationError('Missing/empty existing asset: '+str(path))
        with Image.open(jpeg) as image:
            if image.format != 'JPEG':
                raise AssetPreparationError('Expected an existing JPEG: '+str(jpeg))
            jpeg_size = image.size
        mask_size = png_header_size(mask)
        if jpeg_size != mask_size:
            raise AssetPreparationError('JPEG/official PNG geometry mismatch: '+logical_name,
                                        dict(jpeg_size=list(jpeg_size), mask_size=list(mask_size)))
        record = dict(uid=uid, logical_image=logical_name, physical_image=str(jpeg),
                      official_mask=str(mask), size_wh=list(jpeg_size),
                      image_stat=self.stat(jpeg), mask_stat=self.stat(mask),
                      image_reuse='same_UID_train2017_JPEG' if logical_name.startswith('train2014/') else 'existing_val2014_JPEG')
        self.records[logical_name] = record
        if split == 'test' and role == 'query':
            self.header_only_test_queries.append(dict(logical_image=logical_name, c=category,
                                                       PNG_bytes_read=24, pixels_decoded=False))
            return record
        count = foreground_pixels(mask, category)
        if not count:
            raise AssetPreparationError('Official semantic PNG has no c+1 foreground: '+logical_name,
                                        dict(c=category, required_foreground_value=category+1, split=split, role=role))
        self.foreground_checks.append(dict(logical_image=logical_name, c=category,
                                          foreground_pixels=count, split=split, role=role))
        return record

    @staticmethod
    def stat(path):
        value = path.stat()
        return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def read_official_index(path, *, split, expected_classes):
    """Read only caller-supplied existing official class->logical-image pickle."""
    with Path(path).open('rb') as stream:
        index = pickle.load(stream)
    if not isinstance(index, dict):
        raise AssetPreparationError('Official split must be a class->image-list mapping')
    result = {}
    for key, names in index.items():
        if isinstance(key, bool):
            raise AssetPreparationError('Invalid boolean category key')
        category = int(key)
        if category not in range(80) or category in result:
            raise AssetPreparationError('Wrong/duplicate class in official '+split+' index')
        if not isinstance(names, (list, tuple)):
            raise AssetPreparationError('Official class pool must be a list/tuple')
        if category not in expected_classes:
            if names:
                raise AssetPreparationError('Held/wrong class has nonempty official '+split+' pool')
            continue
        by_uid = {}
        for name in names:
            if not isinstance(name, str):
                raise AssetPreparationError('Official logical image names must be strings')
            uid = coco_photo_id(name)
            if PurePosixPath(name).parts[0] != split:
                raise AssetPreparationError('Official logical image split mismatch')
            if uid in by_uid and by_uid[uid] != name:
                raise AssetPreparationError('Ambiguous logical filenames for one UID')
            by_uid[uid] = name
        result[category] = sorted(by_uid.values())
    if set(result) != set(expected_classes):
        raise AssetPreparationError('Missing official classes; no silent class dropping',
                                    dict(missing_classes=sorted(set(expected_classes)-set(result))))
    return result


def frozen_test_rows(document):
    rows = document.get('frozen_episodes')
    if (document.get('schema') != 'native_membership_assets_v1' or not isinstance(rows, list)
            or len(rows) != 40 or document.get('fold_count') != 4 or document.get('per_fold') != 10):
        raise AssetPreparationError('Exactly the existing frozen 40 episodes/four folds are required')
    result = {fold: [] for fold in range(4)}
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'e', 'c', 'fold', 'support', 'query'}:
            raise AssetPreparationError('Frozen TEST schema changed')
        fold, category, episode = row['fold'], row['c'], row['e']
        if (type(fold) is not int or fold not in range(4) or type(category) is not int
                or category not in range(80) or category % 4 != fold or type(episode) is not int or episode < 0):
            raise AssetPreparationError('Frozen TEST held class/fold/episode mismatch')
        identity = f'test_f{fold}_e{episode}'
        if identity in ids:
            raise AssetPreparationError('Duplicate frozen TEST episode identity')
        ids.add(identity)
        for role in ('support', 'query'):
            coco_photo_id(row[role])
            if not row[role].startswith('val2014/'):
                raise AssetPreparationError('Frozen TEST must retain original val2014 paths')
        if coco_photo_id(row['support']) == coco_photo_id(row['query']):
            raise AssetPreparationError('Frozen TEST query equals its labelled support')
        result[fold].append(dict(id=identity, c=category, support=row['support'], query=row['query']))
    if any(len(pool) != 10 for pool in result.values()):
        raise AssetPreparationError('Frozen TEST must retain ten episodes per fold')
    return result


def prepare_data(official_root, val_image_root, existing_train_images, test_manifest, *, seed=31027):
    if type(seed) is not int or seed != 31027:
        raise AssetPreparationError('This finite preparation freezes seed31027')
    official_root, test_manifest = Path(official_root), Path(test_manifest)
    document = json.loads(test_manifest.read_text())
    test_by_fold = frozen_test_rows(document)
    registry = {coco_photo_id(row[role]) for rows in test_by_fold.values()
                for row in rows for role in ('support', 'query')}
    assets = ExistingCocoAssets(official_root, val_image_root, existing_train_images)
    metadata_paths = [test_manifest]
    validation_pool = {}
    for fold in range(4):
        path = official_root/f'splits/val/fold{fold}.pkl'
        metadata_paths.append(path)
        pool = read_official_index(path, split='val2014', expected_classes=set(range(fold, 80, 4)))
        validation_pool.update(pool)
    manifests, availability = [], {}
    for fold in range(4):
        base_classes = set(range(80))-set(range(fold, 80, 4))
        path = official_root/f'splits/trn/fold{fold}.pkl'
        metadata_paths.append(path)
        training_pool = read_official_index(path, split='train2014', expected_classes=base_classes)
        candidates = {}
        availability[str(fold)] = {}
        for category in sorted(base_classes):
            candidates[category] = {}
            counts = {}
            for split, source in (('train', training_pool), ('dev', validation_pool)):
                names = source[category]
                eligible = [name for name in names if coco_photo_id(name) not in registry and assets.paired_exists(name)]
                candidates[category][split] = eligible
                counts[split] = dict(official_index_unique_photos=len(names),
                                     same_UID_existing_JPEG_and_official_PNG=len(eligible),
                                     excluded_frozen_TEST_photos=sum(coco_photo_id(name) in registry for name in names),
                                     selected_geometry_and_foreground_checks_only=True)
            availability[str(fold)][str(category)] = counts
        insufficient = [{"fold": fold, "c": category, "split": split, "available_photos": len(pools[split]),
                         "required_distinct_photos": 2, "required_episodes": 1}
                        for category, pools in candidates.items() for split in ('train', 'dev') if len(pools[split]) < 2]
        if insufficient:
            raise AssetPreparationError('Insufficient paired existing assets; no class/count changed',
                                        dict(insufficient=insufficient, availability=availability))
        rng = random.Random(seed+fold)
        selected = {'train': [], 'dev': []}
        dev_photos = set()
        # DEV excludes ALL forty TEST support/query UIDs, not just this fold.
        for category in sorted(base_classes):
            names = candidates[category]['dev']
            support, query = rng.sample(names, 2)
            dev_photos.update((coco_photo_id(support), coco_photo_id(query)))
            selected['dev'].append(dict(id=f'dev_f{fold}_c{category}', c=category, support=support, query=query))
        for category in sorted(base_classes):
            names = [name for name in candidates[category]['train'] if coco_photo_id(name) not in dev_photos]
            availability[str(fold)][str(category)]['train']['after_selected_DEV_exclusion'] = len(names)
            if len(names) < 2:
                raise AssetPreparationError('Insufficient TRAIN assets after all-role DEV exclusion',
                                            dict(fold=fold, c=category, availability=availability))
            support, query = rng.sample(names, 2)
            selected['train'].append(dict(id=f'train_f{fold}_c{category}', c=category, support=support, query=query))
        manifest = dict(schema='reference_response_episodes_v1', annotation_protocol='official_COCO20i',
                        held_fold=fold, coco_year=2014, test_scope='reused_development',
                        train=selected['train'], dev=selected['dev'], test=test_by_fold[fold])
        # Validate metadata before opening any selected annotation pixels.
        validation = validate_episode_manifest(manifest, prior_development_photo_ids=registry)
        for split in ('train', 'dev', 'test'):
            for row in manifest[split]:
                for role in ('support', 'query'):
                    assets.inspect(row[role], row['c'], split=split, role=role)
        manifests.append(dict(manifest=manifest, protocol_validation=validation,
                              training_projection={key: manifest[key] for key in
                                  ('schema', 'annotation_protocol', 'held_fold', 'coco_year', 'train', 'dev')}))
    # Byte-level metadata/source identities; images and masks carry only stats.
    metadata = {str(path): dict(sha256=file_sha256(path), **ExistingCocoAssets.stat(path))
                for path in metadata_paths}
    sources = (Path(__file__), Path(__file__).with_name('reference_response_protocol.py'))
    return dict(schema='reference_response_data_preparation_v1', state='FINITE_DATA_PREPARATION_PASSED',
                seed=seed, shuffle_seed_rule='31027 + held_fold', folds=manifests, per_fold=dict(train_episodes=60, dev_episodes=60, frozen_test_episodes=10),
                roots=dict(official_root=str(assets.official_root), val_image_root=str(assets.val_image_root),
                           existing_train_images=str(assets.existing_train_images)),
                frozen_test_episodes=document['frozen_episodes'], frozen_TEST_unchanged=True,
                prior_development_photo_ids=sorted(registry), exposure_registry_complete=False,
                exposure_scope='Existing forty reused development tasks; not fresh confirmation.',
                availability=availability, assets=sorted(assets.records.values(), key=lambda row: row['logical_image']),
                selected_foreground_checks=assets.foreground_checks,
                test_query_checks=assets.header_only_test_queries,
                binary_label_policy='semantic == c+1; ALL other values are background; no separate VOID mask',
                metadata=metadata, source_sha256={str(path): file_sha256(path) for path in sources},
                images_copied=False, archives_extracted=False, downloads=False, torch_imported=False,
                training_executed=False, real_task_gain_measured=False,
                limitations=['One TRAIN and one DEV episode per base class is a finite pilot subset, not sufficient training or standard full training.',
                             'Available subset counts verify filename UID and existing paired files; geometry/foreground are checked on selected rows only.',
                             'Stat/UID/geometry checks do not prove JPEG pixel equivalence across dataset releases.',
                             'Prior exposure registry contains only the frozen40 roles and is incomplete.',
                             'TEST query PNG pixels were not decoded; its semantic foreground existence is deliberately unexamined.'])
