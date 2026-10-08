"""Bind legacy development packs to official class IDs and exact input identities.

Legacy exporters did not retain photo/object IDs. Record that absence, match
decoded inputs instead, and flag conflicting labels rather than silently
declaring image independence. Query-mask hashes are protocol audit data only.
"""
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .official_data import array_hash, build_dataset, file_hash

PACKS = {'lvis': [f'lvis_f{f}' for f in range(4)],
         'paco_part': [f'paco_part_f{f}' for f in range(4)],
         'pascal_part': ['pascal_part'], 'suim': ['suim']}


def suim_categories(assets):
    source = Path(assets)/'third_party/foris_official/datasets/suim.py'
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name) and t.value.id == 'self'
                and t.attr == 'categories' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Cannot bind official SUIM categories')


def class_maps(assets, dataset, folds):
    if dataset == 'suim':
        return {-1: {i: c for i, c in enumerate(suim_categories(assets))}}
    if dataset == 'coco':
        return {f: {i: i for i in range(80) if i % 4 == f} for f in folds}
    result = {}
    for fold in sorted(folds):
        ds = build_dataset(dataset, fold, assets)
        if dataset == 'pascal_part':
            result[fold] = dict(zip(ds.class_ids, ds.cat_part_name))
        else:
            result[fold] = {ds.class_ids_c[c]: int(c) for c in ds.class_ids_ori}
    return result


def decoded_mask(path, class_id=None):
    with Image.open(path) as im:
        x = np.asarray(im) if class_id is not None else np.asarray(im.convert('L'))
        return (x == class_id+1 if class_id is not None else x > 0).astype(np.uint8)


def bind_inputs(record, assets, reference, query, reference_mask, query_mask, class_id=None, mask_root=None):
    assets = Path(assets).resolve()
    record['source_files'] = {}
    for p in (reference, query, reference_mask, query_mask):
        record['source_files'][str(p.relative_to(assets))] = file_hash(p)
    for role, rgb_path, mask_path in [('reference', reference, reference_mask), ('query', query, query_mask)]:
        with Image.open(rgb_path) as im:
            rgb = np.asarray(im.convert('RGB'))
        mask = decoded_mask(mask_path, class_id)
        if class_id is None and mask.shape != rgb.shape[:2]:
            raise ValueError('Development RGB/annotation geometry differs')
        record[role+'_rgb_hash'] = array_hash(rgb)
        record[role+'_mask_hash'] = array_hash(mask)
        record[role+'_size_hw'] = list(rgb.shape[:2])
        record[role+'_mask_size_hw'] = list(mask.shape)
        record[role+'_path'] = str(rgb_path.relative_to(assets))
        record[role+'_crop'] = None  # Stored PNGs already contain the legacy crop.
        if class_id is not None:
            destination = Path(mask_root)/role/(record[role+'_mask_hash']+'.png')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                Image.fromarray(mask*255).save(destination)
            mask_path = destination
        record[role+'_mask_path'] = str(mask_path.resolve())
    record['query_gt_role'] = 'development protocol preparation only; never inference'
    record['ignore_policy'] = 'frozen binary pack; COCO indexed source uses official foreground c+1'
    return record


def prepare_development(assets, datasets, mask_root):
    assets = Path(assets).resolve()
    rows, audit = [], {}
    for name in datasets:
        if name == 'coco':
            source = assets/'cache/astra_granularity_20261008/coco_fresh600/rows.json'
            original = json.loads(source.read_text())
            maps = class_maps(assets, name, {r['fold'] for r in original})
            base = assets/'demo4_cache/data/COCO2014'
            files = {base/r[k] for r in original for k in ('support', 'query')}
            missing = sorted(str(p.relative_to(assets)) for p in files if not p.is_file())
            if missing:
                raise FileNotFoundError(f'Frozen COCO fresh600 inputs: {len(missing)} RGBs missing; first {missing[:3]}')
            for r in original:
                fold, cid = int(r['fold']), int(r['c'])
                record = dict(dataset=name, fold=fold, loader_class_id=cid, global_class_id=maps[fold][cid],
                    episode_id='dev/coco/'+r['key'], source_key=r['key'], input_origin='latest frozen fresh600',
                    reference_photo_id=r['support'], query_photo_id=r['query'],
                    reference_object_id=None, query_object_id=None)
                rows.append(bind_inputs(record, assets, base/r['support'], base/r['query'],
                    base/'annotations'/Path(r['support']).with_suffix('.png'),
                    base/'annotations'/Path(r['query']).with_suffix('.png'), class_id=cid, mask_root=mask_root))
            audit[name] = dict(n=len(original), source_manifest_sha256=file_hash(source), resized_images=0)
            continue
        sources = []
        for pack in PACKS[name]:
            root = assets/'episodes/claude_packs'/pack
            manifest = root/'episodes.json'
            data = json.loads(manifest.read_text())
            receipt = json.loads((root/'pack_receipt.json').read_text())
            sources.append((pack, root, data['episodes'], receipt, file_hash(manifest)))
        folds = {-1 if name == 'suim' else int(r['source_fold'])
                 for _, _, original, _, _ in sources for r in original}
        maps = class_maps(assets, name, folds)
        for pack, root, original, receipt, digest in sources:
            for r in original:
                source_fold, cid = map(int, r['source_class'].split(':'))
                if source_fold != r['source_fold']:
                    raise ValueError('Legacy source fold/class mapping conflicts')
                fold = -1 if name == 'suim' else source_fold
                record = dict(dataset=name, fold=fold, loader_class_id=cid,
                    global_class_id=maps[fold][cid], episode_id=f"dev/{pack}/{r['e']}/{r['c']}",
                    source_key=f"{r['fold']}_{r['e']}_{r['c']}", source_pack=pack,
                    legacy_export_class_id=r['c'], input_origin='frozen legacy cropped PNG pack',
                    legacy_pack_resizing_reported=receipt['resized_images'] > 0,
                    reference_photo_id=None, query_photo_id=None,
                    reference_object_id=None, query_object_id=None,
                    original_identity='not preserved by exporter; exact decoded-input matching required')
                rows.append(bind_inputs(record, assets, root/'data'/r['support'], root/'data'/r['query'],
                                        root/'ann'/r['support'], root/'ann'/r['query']))
            audit.setdefault(name, dict(n=0, resized_images=0, packs=[]))
            audit[name]['n'] += len(original)
            audit[name]['resized_images'] += receipt['resized_images']
            audit[name]['packs'].append(dict(pack=pack, n=len(original), manifest_sha256=digest))
    if len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('Development draw IDs collide')
    for name, data in audit.items():
        data['classes'] = len({(r['fold'], r['loader_class_id']) for r in rows if r['dataset'] == name})
        data['photo_ids_unknown'] = sum(r['query_photo_id'] is None for r in rows if r['dataset'] == name)
    return rows, audit


def identity(row):
    return (row['dataset'], str(row['global_class_id']), row['reference_rgb_hash'],
            row['reference_mask_hash'], row['query_rgb_hash'])


def exclude_development(official, development, expected_classes):
    """Preserve legal official repeats; remove every matching development draw.

    expected_classes: {dataset: {fold: [loader class ID, ...]}}. A matching input
    with a different target mask is an annotation/object-identity conflict.
    Empty confirmation classes require protocol resolution, never resampling.
    """
    dev = defaultdict(list)
    for row in development:
        dev[identity(row)].append(row)
    confirm, matches, conflicts = [], [], []
    for row in official:
        variants = [('raw', row)]
        variants += [(name, dict(view, dataset=row['dataset'], global_class_id=row['global_class_id']))
                     for name, view in row.get('legacy_export_views', {}).items()]
        candidates = [(kind, representation, dev.get(identity(representation), []))
                      for kind, representation in variants]
        candidates = [(kind, representation, ds) for kind, representation, ds in candidates if ds]
        if not candidates:
            confirm.append(row)
            continue
        exact = {d['episode_id']: d for _, representation, ds in candidates for d in ds
                 if d['query_mask_hash'] == representation['query_mask_hash']}
        bad = {d['episode_id'] for _, representation, ds in candidates for d in ds
               if d['query_mask_hash'] != representation['query_mask_hash']}
        if not exact or bad:
            conflicts.append(dict(official_id=row['episode_id'], development_ids=sorted(bad or exact)))
            continue
        matches.append(dict(official_id=row['episode_id'], development_ids=sorted(exact),
                            matched_views=[kind for kind, _, _ in candidates],
                            reference_photo_id=row.get('reference_photo_id'), query_photo_id=row.get('query_photo_id'),
                            reference_object_id=row.get('reference_object_id'), query_object_id=row.get('query_object_id')))
    count = Counter((r['dataset'], str(r['fold']), str(r['loader_class_id'])) for r in confirm)
    empty = [dict(dataset=name, fold=str(f), class_id=str(c))
             for name, folds in expected_classes.items() for f, cs in folds.items() for c in cs
             if count[(name, str(f), str(c))] == 0]
    shared = {}
    recovered = {key for r in matches for key in r['development_ids']}
    resize_unresolved = [r['episode_id'] for r in development
                         if r.get('legacy_pack_resizing_reported')
                         and r['episode_id'] not in recovered
                         and r.get('original_episode_identity') is None]
    for role in ('reference', 'query'):
        known = {r[role+'_rgb_hash'] for r in development}
        shared[role+'_decoded_input_hashes'] = len({r[role+'_rgb_hash'] for r in official
                                                    if r[role+'_rgb_hash'] in known})
    audit = dict(official_n=len(official), development_n=len(development), confirm_n=len(confirm),
                 excluded_official_draws=len(matches), legal_repeats='preserved; not refilled or deduplicated',
                 matching='dataset/global class + decoded R/R-mask/Q hashes, including the verified old export cap1600 view; query-mask equality verifies the target',
                 matches=matches, annotation_conflicts=conflicts, empty_confirm_classes=empty,
                 eligible_for_confirmation=not conflicts and not empty and not resize_unresolved,
                 unresolved_resized_development_ids=resize_unresolved,
                 shared_inputs=shared, independence='episode disjoint; photo independence not established',
                 original_photo_ids_missing_in_development=sum(r.get('query_photo_id') is None for r in development))
    return confirm, audit
