"""Replay the remaining released FoRIS loaders without exposing query labels.

Preparation observes the unchanged author's loader. Inference receives exactly
reference RGB, reference binary mask and query RGB. DeepGlobe deliberately has
no adapter: the released source does not contain its road sampling protocol.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import official_data as official

file_hash = official.file_hash
array_hash = official.array_hash
reset_sampling_seed = official.reset_sampling_seed

EXTENDED_FOLDS = {'isaid': [0, 1, 2], 'isic': [-1], 'lung': [-1],
                  'fundus': [-1], 'deepglobe': [-1]}
FOLDS = {**official.FOLDS, **EXTENDED_FOLDS}
ALIASES = {'chest': 'lung', 'chest_xray': 'lung'}
METRIC_CONVENTIONS = {
    'released_cli_prediction_hw': [1024, 1024],
    'released_cli_primary': 'models/__init__.py sets resize_to_orig_size=False',
    'additional_frame': 'original loader query RGB size; report separately',
    'label_alignment': 'torch nearest-resize raw GT and raw ignore directly to each prediction frame; >0.5',
    'statistic': 'sum foreground intersection/union within each class; class mean; fold mean',
    'zero_union': 'official denominator max(union, 1); absent expected classes remain zero',
    'ignore': 'iSAID returns floor(raw_mask/255); excluded from intersection and union',
}


class BenchmarkUnavailableError(FileNotFoundError):
    """A missing asset or unreleased protocol; never substitute another pool."""


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in FOLDS:
        raise ValueError('Unknown benchmark: '+name)
    return name


def source_receipt(assets, names):
    """Check the source files against the pinned official snapshot manifest."""
    root = Path(assets).resolve()/'third_party/foris_official'
    manifest_path = root/'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    expected = {r['path']: r['sha256'] for r in manifest['included_files']}
    paths = {'inference.py', 'utils/metrics.py', 'models/__init__.py', 'opts.py',
             'datasets/__init__.py', 'docs/data.md'}
    paths.update('datasets/'+canonical_name(n)+'.py' for n in names
                 if canonical_name(n) != 'deepglobe')
    actual = {}
    for rel in sorted(paths):
        digest = file_hash(root/rel)
        if expected.get(rel) != digest:
            raise ValueError('Pinned official source changed or unlisted: '+rel)
        actual[rel] = digest
    return dict(repository=manifest['repository'], commit=manifest['commit'],
                snapshot_manifest_sha256=file_hash(manifest_path), files=actual)


def build_dataset(name, fold, assets):
    name = canonical_name(name)
    if fold not in FOLDS[name]:
        raise ValueError('Invalid official fold')
    if name in official.FOLDS:
        return official.build_dataset(name, fold, assets)
    if name == 'deepglobe':
        raise BenchmarkUnavailableError(
            'DeepGlobe road loader, released episode pairing and split are absent; '
            'custom deepglobe200 preparation is exploratory and cannot replace them')
    assets = Path(assets).resolve()
    root = assets/'third_party/foris_official'
    if name == 'lung':
        for leaf in ('CXR_png', 'masks'):
            path = assets/'datasets/ics/LungSegmentation'/leaf
            if not path.is_dir():
                raise BenchmarkUnavailableError('Chest X-ray directory not found: '+str(path))
    key = '_ics_extended_official_'+name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, root/'datasets'/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[key] = module
    ds = sys.modules[key].build(SimpleNamespace(
        data_root=str(assets/'datasets/ics'), fold=fold, shots=1))
    if any(len(set(ds.img_metadata_classwise[c])) < 2 for c in
           (ds.class_ids if name == 'isaid' else ds.categories)):
        raise BenchmarkUnavailableError(name+' needs at least two images per evaluated class')
    return ds


def verify_pool(name, assets):
    """Verify complete sampling pools, not only the images eventually drawn.

    Returns the verified SHA map for reuse while recording source identities.
    No RGB transformation, annotation remapping or RNG draw occurs here.
    """
    name, assets = canonical_name(name), Path(assets).resolve()
    if name == 'deepglobe':
        build_dataset(name, -1, assets)
    if name not in EXTENDED_FOLDS:
        raise ValueError('Pool verification here covers the extended benchmarks only')
    expected, receipts, ancillary = {}, {}, {}
    if name == 'isaid':
        rel = 'setup/asset_downloads/iSAID/isaid_extract_receipt.json'
        data = json.loads((assets/rel).read_text())
        if data['status'] != 'READY_FOR_OFFICIAL_LOADER':
            raise BenchmarkUnavailableError('iSAID input receipt is not ready')
        receipts[rel] = file_hash(assets/rel)
        for rows in data['files'].values():
            expected.update({r['path']: r for r in rows})
        split_rel = data['source_splits_receipt']
        split = json.loads((assets/split_rel).read_text())
        if split['status'] != 'VERIFIED':
            raise BenchmarkUnavailableError('iSAID split receipt is not verified')
        receipts[split_rel] = file_hash(assets/split_rel)
        expected.update({'datasets/ics/'+r['path']: r for r in split['files']})
        roots = [assets/'datasets/ics/iSAID/val/images',
                 assets/'datasets/ics/iSAID/val/semantic_png',
                 assets/'datasets/ics/iSAID/splits/val']
        actual = {str(p.relative_to(assets)) for root in roots for p in root.glob('*') if p.is_file()}
    elif name == 'isic':
        rel = 'setup/asset_downloads/isic_input/isic_prepare_receipt.json'
        data = json.loads((assets/rel).read_text())
        if data['status'] != 'READY_FOR_OFFICIAL_LOADER':
            raise BenchmarkUnavailableError('ISIC input receipt is not ready')
        receipts[rel] = file_hash(assets/rel)
        for r in data['files']:
            for field in ('split_image', 'mask'):
                expected[r[field]] = dict(bytes=r[field+'_bytes'], sha256=r[field+'_sha256'])
        image_root = assets/'datasets/ics/ISIC/ISIC2018_Task1-2_Training_Input'
        mask_root = assets/'datasets/ics/ISIC/ISIC2018_Task1_Training_GroundTruth'
        alias = assets/'datasets/ics/ISIC/ISIC2018_Task1_Training_GT'
        if alias.resolve() != mask_root.resolve() or not alias.is_dir():
            raise BenchmarkUnavailableError('ISIC official GT alias changed or missing')
        actual = {str(p.relative_to(assets)) for cat in ('1', '2', '3')
                  for p in (image_root/cat).glob('*') if p.is_file()}
        # The archive includes LICENSE.txt/ATTRIBUTION.txt. The official
        # loader constructs *_segmentation.png paths and never samples them.
        actual.update(str(p.relative_to(assets)) for p in mask_root.glob('*.png') if p.is_file())
        ancillary = {str(p.relative_to(assets)): file_hash(p) for p in mask_root.glob('*')
                     if p.is_file() and p.suffix != '.png'}
    else:
        # Download preparation owns source-cohort identification. Consume only
        # its ready file/SHA receipt, then verify the actual released-loader
        # roots rather than treating an arbitrary local folder as complete.
        ds = build_dataset(name, -1, assets)
        label = 'Chest' if name == 'lung' else 'Fundus'
        rel = f'setup/asset_downloads/{label}_autonomous_20261010/preparation_receipt.json'
        if not (assets/rel).is_file():
            raise BenchmarkUnavailableError(name+' has no verified complete input-pool receipt: '+rel)
        data = json.loads((assets/rel).read_text())
        if data['status'] != 'READY_FOR_OFFICIAL_LOADER':
            raise BenchmarkUnavailableError(name+' input receipt is not ready')
        receipts[rel] = file_hash(assets/rel)
        roots = [Path(ds.img_path), Path(ds.ann_path)]
        prefixes = tuple(str(p.relative_to(assets))+'/' for p in roots)
        expected = {r['path']: r for r in data['files']
                    if r['path'].startswith(prefixes) and Path(r['path']).suffix.lower() == '.png'}
        if not expected:
            raise BenchmarkUnavailableError(name+' ready receipt contains no loader-pool PNGs')
        actual = {str(p.relative_to(assets)) for root in roots for p in root.glob('*')
                  if p.is_file() and p.suffix.lower() == '.png'}
    if actual != set(expected):
        raise BenchmarkUnavailableError(name+' pool differs from its receipt: '
            f'{len(set(expected)-actual)} missing, {len(actual-set(expected))} extra files')
    hashes, total_bytes = {}, 0
    for rel, row in sorted(expected.items()):
        path = assets/rel
        digest = file_hash(path)
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise ValueError('Verified benchmark input changed: '+rel)
        hashes[rel] = digest
        total_bytes += row['bytes']
    return dict(state='COMPLETE_POOL_SHA_VERIFIED', files=len(hashes), bytes=total_bytes,
                receipt_sha256=receipts, asset_sha256=hashes, non_sampling_files=ancillary)


def _source_paths(ds, selection):
    s = selection['sample_episode']
    qname, rname = str(s['tgt_name']), str(s['ref_names'][0])
    name = ds.benchmark
    if name == 'isaid':
        q, r = (Path(ds.img_path)/(n+'.png') for n in (qname, rname))
        qmask, rmask = (Path(ds.ann_path)/(n+'_instance_color_RGB.png') for n in (qname, rname))
        metadata = [Path(ds.datapath)/'splits/val'/f'fold{ds.fold}.txt']
    elif name == 'isic':
        q, r = Path(qname), Path(rname)
        qmask, rmask = (Path(ds.ann_path)/(p.stem+'_segmentation.png') for p in (q, r))
        metadata = []
    elif name == 'lung':
        qmask, rmask = Path(qname), Path(rname)
        def image_path(mask):
            n = str(mask)
            return Path(ds.img_path)/Path(n if 'MCUCXR' in n else n[:-9]+'.png').name
        q, r = image_path(qmask), image_path(rmask)
        metadata = []
    elif name == 'fundus':
        q, r = Path(qname), Path(rname)
        qmask, rmask = (Path(ds.ann_path)/p.name for p in (q, r))
        metadata = []
    else:
        raise ValueError('Unknown extended source mapping')
    return dict(reference_path=r, query_path=q, reference_annotation_path=rmask,
                query_annotation_path=qmask, metadata_paths=metadata)


def _freeze_mask(tensor, role, mask_root, record):
    mask = (tensor.cpu().numpy() > .5).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError('Official binary mask must be two-dimensional')
    digest = array_hash(mask)
    destination = Path(mask_root).resolve()/role/(digest+'.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        Image.fromarray(mask*255).save(destination)
    record[role+'_mask_path'] = str(destination)
    record[role+'_mask_hash'] = digest
    record[role+'_mask_size_hw'] = list(mask.shape)


def record_episode(ds, index, fold, assets, mask_root, hash_cache=None):
    if ds.benchmark in official.FOLDS:
        return official.record_episode(ds, index, fold, assets, mask_root, hash_cache)
    batch, selection = official.sample_observed(ds, index)
    source = _source_paths(ds, selection)
    assets = Path(assets).resolve()
    cached = {} if hash_cache is None else hash_cache
    record = dict(dataset=ds.benchmark, fold=fold, official_index=index,
                  episode_id=f'{ds.benchmark}/{fold}/{index}', shots=1,
                  loader_class_id=int(batch['class_id']), global_class_id=int(batch['class_id']),
                  query_gt_role='protocol preparation only; scoring after prediction seal',
                  official_selection=official._jsonable(selection), source_files={})
    for p in [source['reference_path'], source['query_path'],
              source['reference_annotation_path'], source['query_annotation_path'], *source['metadata_paths']]:
        rel = str(p.relative_to(assets))
        if rel not in cached:
            cached[rel] = file_hash(p)
        record['source_files'][rel] = cached[rel]
    for role, actual in [('reference', batch['ref_imgs'][0]), ('query', batch['tgt_img'])]:
        path = source[role+'_path']
        replay = official.decoded_rgb(path)
        digest = array_hash(np.asarray(actual))
        if array_hash(np.asarray(replay)) != digest:
            raise RuntimeError('Extended source mapping does not reproduce official RGB')
        record[role+'_path'] = str(path.relative_to(assets))
        record[role+'_crop'] = None
        record[role+'_photo_id'] = path.stem
        record[role+'_object_id'] = None
        record[role+'_rgb_hash'] = digest
        record[role+'_size_hw'] = [actual.height, actual.width]
        record[role+'_annotation_path'] = str(source[role+'_annotation_path'].relative_to(assets))
    _freeze_mask(batch['ref_masks'][0], 'reference', mask_root, record)
    _freeze_mask(batch['tgt_mask'], 'query', mask_root, record)
    if 'tgt_ignore_idx' in batch:
        ignore = batch['tgt_ignore_idx']
        if ((ignore > .5) & (batch['tgt_mask'] > .5)).any():
            raise ValueError('Official foreground and ignore overlap')
        _freeze_mask(ignore, 'query_ignore', mask_root, record)
        record['ignore_policy'] = 'official iSAID floor(raw_mask/255), frozen before binary class selection'
    else:
        record['ignore_policy'] = 'released loader has no query ignore index'
    record['gt_alignment'] = METRIC_CONVENTIONS['label_alignment']
    return record


def load_inputs(record, assets):
    """Legal inference adapter; never opens query annotations or ignore masks."""
    return official.load_inputs(record, assets)


def _read_frozen_mask(record, role):
    with Image.open(record[role+'_mask_path']) as im:
        mask = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    if array_hash(mask) != record[role+'_mask_hash']:
        raise ValueError('Frozen '+role+' annotation changed')
    if mask.shape != tuple(record[role+'_mask_size_hw']):
        raise ValueError('Frozen '+role+' annotation geometry changed')
    return mask


def load_query_ground_truth(record, prediction_hw):
    """Scoring-only adapter: direct nearest-resize, preserving iSAID ignore.

    Call only after all candidate predictions are sealed. The explicit frame
    prevents accidental conversion through a second metric's geometry.
    """
    import torch
    import torch.nn.functional as F
    if len(prediction_hw) != 2 or any(int(x) != x or x <= 0 for x in prediction_hw):
        raise ValueError('Require positive integer prediction height and width')
    def resize(role):
        mask = torch.from_numpy(_read_frozen_mask(record, role))[None, None].float()
        return F.interpolate(mask, size=tuple(prediction_hw), mode='nearest')[0, 0].numpy() > .5
    truth = resize('query')
    ignore = resize('query_ignore') if 'query_ignore_mask_path' in record else None
    if ignore is not None and (truth & ignore).any():
        raise ValueError('Frozen query foreground and ignore overlap')
    return truth, ignore
