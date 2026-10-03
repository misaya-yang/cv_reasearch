"""Metadata-only COCO episode isolation checks; no image/annotation reading.

Class IDs are sampler/scorer metadata, never decoder inputs. Shared photos
within a test split remain dependent even when their episode IDs differ.
"""
from collections import Counter
from pathlib import PurePosixPath
import re


def coco_photo_id(relative_path):
    if not isinstance(relative_path, str) or '\\' in relative_path:
        raise ValueError('Expected a portable relative COCO image path')
    path = PurePosixPath(relative_path)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 2:
        raise ValueError('COCO image paths must be split/file, without traversal')
    if path.parts[0] not in ('train2014', 'val2014', 'train2017', 'val2017'):
        raise ValueError('Unknown original COCO image split')
    match = re.fullmatch(r'(?:COCO_(?:train|val)201[47]_)?(\d{12})\.(?:jpg|jpeg)', path.name)
    if match is None:
        raise ValueError('COCO image identity must contain its original twelve-digit ID')
    if path.name.startswith('COCO_') and not path.name.startswith('COCO_'+path.parts[0]+'_'):
        raise ValueError('COCO filename and directory split disagree')
    return int(match.group(1))


def _components(rows):
    """Episode connectivity induced by any shared support/query photo."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, row in enumerate(rows):
        for role in ('support', 'query'):
            photo = coco_photo_id(row[role])
            if photo in owner:
                parent[find(i)] = find(owner[photo])
            else:
                owner[photo] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row['id'])
    return sorted(groups.values(), key=lambda items: str(items[0]))


def validate_episode_manifest(manifest, *, prior_development_photo_ids):
    """Reject held-class/photo leakage before ANY annotation can be opened.

    This verifies only the supplied metadata/exposure registry, not its
    completeness, files, mask version or benchmark reproduction. The
    returned status is never an independent task-benefit certificate.
    """
    if manifest.get('schema') != 'reference_response_episodes_v1':
        raise ValueError('Wrong response episode schema')
    if manifest.get('annotation_protocol') != 'official_COCO20i':
        raise ValueError('Official-mask protocol must be explicit')
    fold = manifest.get('held_fold')
    if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(4):
        raise ValueError('COCO20i held fold must be0..3')
    year = manifest.get('coco_year')
    if year not in (2014, 2017):
        raise ValueError('Original dataset year must be explicit')
    scope = manifest.get('test_scope')
    if scope not in ('reused_development', 'frozen_confirmation'):
        raise ValueError('Evaluation exposure scope must be explicit')
    if prior_development_photo_ids is None:
        raise ValueError('Prior development exposure registry is required, even when empty')
    registry = set(prior_development_photo_ids)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in registry):
        raise ValueError('Exposure registry contains invalid COCO identities')
    photos, classes, counts = {}, {}, {}
    all_episode_ids = set()
    for split in ('train', 'dev', 'test'):
        rows = manifest.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Nonempty explicitly frozen train/dev/test episodes required')
        photos[split], classes[split] = set(), set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'id', 'c', 'support', 'query'}:
                raise ValueError('Episode schema is id/class/support/query only')
            episode_id = row['id']
            if not isinstance(episode_id, str) or not episode_id or episode_id in all_episode_ids:
                raise ValueError('Episode IDs must be globally unique nonempty strings')
            all_episode_ids.add(episode_id)
            category = row['c']
            if not isinstance(category, int) or isinstance(category, bool) or category not in range(80):
                raise ValueError('Official dense COCO category index must be0..79')
            if (category % 4 == fold) != (split == 'test'):
                raise ValueError('Held category entered base training/development or wrong test fold')
            classes[split].add(category)
            expected_directory = ('train' if split == 'train' else 'val')+str(year)
            identities = []
            for role in ('support', 'query'):
                photo = coco_photo_id(row[role])
                if PurePosixPath(row[role]).parts[0] != expected_directory:
                    raise ValueError('Use declared original train split for training, val split for dev/test')
                photos[split].add(photo)
                identities.append(photo)
            if identities[0] == identities[1]:
                raise ValueError('Query must be a new photo, not the labelled reference itself')
        counts[split] = dict(sorted(Counter(row['c'] for row in rows).items()))
    for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if photos[left] & photos[right]:
            raise ValueError('Photo leakage across ALL support/query roles: '+left+'/'+right)
    old_test_photos = photos['test'] & registry
    if scope == 'frozen_confirmation' and old_test_photos:
        raise ValueError('Previously exposed photo cannot be called fresh confirmation')
    groups = _components(manifest['test'])
    return dict(state='EPISODE_METADATA_CONTRACT_PASSED', held_fold=fold,
                held_class_rule='official dense category index modulo4',
                train_classes=sorted(classes['train']), dev_classes=sorted(classes['dev']),
                test_classes=sorted(classes['test']), episodes_per_class=counts,
                unique_photos={k: len(v) for k, v in photos.items()},
                all_role_photo_splits_disjoint=True, test_exposure_scope=scope,
                test_prior_development_overlap=len(old_test_photos),
                test_photo_connected_groups=groups, independent_episode_bootstrap_allowed=False,
                image_files_verified=False, official_mask_files_verified=False,
                exposure_registry_completeness_verified=False, task_gain_measured=False)
