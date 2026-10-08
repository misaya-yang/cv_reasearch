"""Observe the unedited official loaders and replay their frozen RGB inputs.

Tracing records the chosen image/object/crop without adding random draws or
changing official sampling. Query masks are stored during protocol preparation;
load_inputs never opens them and exposes only R, reference mask, Q.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import glob
from types import SimpleNamespace

import numpy as np
from PIL import Image

FOLDS = {'coco': list(range(4)), 'lvis': list(range(10)),
         'pascal_part': list(range(4)), 'paco_part': list(range(4)), 'suim': [-1]}
TRACE_FIELDS = ('tgt_name', 'ref_names', 'class_sample', 'sel_tgt_id', 'sel_ref_id',
                'tgt_obj_bbox', 'ref_boxes', 'query_img_id', 'support_img_ids',
                'query_obj_box', 'support_boxes', 'tgt_mask_path', 'ref_mask_paths')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def reset_sampling_seed(seed=0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(name, fold, assets):
    """Only path relocation differs from the author's build(args)."""
    assets = Path(assets).resolve()
    root = assets/'third_party/foris_official'
    if name not in FOLDS or fold not in FOLDS[name]:
        raise ValueError('Invalid official dataset/fold')
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    key = '_ics_official_'+name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, root/'datasets'/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    # COCO validation RGB/masks/splits live together in the relocated pool.
    data_root = assets/('demo4_cache/data' if name == 'coco' else 'datasets/ics')
    if name == 'suim':
        # This loader derives its sampling pool by scanning actual masks, unlike
        # the other four metadata-driven loaders. A partial folder would change
        # the official seed-0 draw even in a smoke, so validate the entire pool.
        inventory = json.loads((assets/'setup/manifests/ics_datasets.json').read_text())
        cats = ('FV', 'HD', 'PF', 'RI', 'RO', 'SR', 'WR')
        expected = {r['path']: r for r in inventory['files']
                    if any(r['path'].startswith('SUIM/masks/'+c+'/') for c in cats)}
        actual = {str(Path(p).relative_to(data_root)) for c in cats
                  for p in glob.glob(str(data_root/'SUIM/masks'/c/'*'))}
        if actual != set(expected):
            raise FileNotFoundError('SUIM official mask pool incomplete or changed: '
                                    f'{len(set(expected)-actual)} missing, {len(actual-set(expected))} extra')
        for rel, row in expected.items():
            path = data_root/rel
            if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
                raise ValueError('SUIM official mask changed: '+rel)
    ds = sys.modules[key].build(SimpleNamespace(data_root=str(data_root), fold=fold, shots=1))
    if name == 'suim' and any(len(v) < 2 for v in ds.img_metadata_classwise.values()):
        raise FileNotFoundError('SUIM official nonempty image/mask pools have not arrived completely')
    return ds


def sample_observed(dataset, index):
    selected = {}
    previous = sys.getprofile()

    def trace(frame, event, result):
        if (event == 'return' and frame.f_locals.get('self') is dataset
                and frame.f_code.co_name in ('load_frame', 'sample_episode')):
            selected[frame.f_code.co_name] = {k: frame.f_locals[k] for k in TRACE_FIELDS
                                               if k in frame.f_locals}
        if previous is not None:
            previous(frame, event, result)
    try:
        sys.setprofile(trace)
        batch = dataset[index]  # Includes the author's retry and duplicate rules.
    finally:
        sys.setprofile(previous)
    return batch, selected


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _source_paths(ds, selection):
    name = ds.benchmark
    s = selection.get('sample_episode', selection.get('load_frame', {}))
    if name == 'pascal_part':
        q = Path(ds.img_file.format(s['query_img_id']))
        r = Path(ds.img_file.format(s['support_img_ids'][0]))
        boxes = [s['support_boxes'][0], s['query_obj_box']]
        ids = [str(s['support_img_ids'][0]), str(s['query_img_id'])]
        objects = [dict(box=boxes[0]), dict(box=boxes[1])]
        annotations = [Path(ds.anno_file.format(i)) for i in ids]
        global_class = s['class_sample']
    elif name == 'suim':
        qmask, rmask = Path(s['tgt_name']), Path(s['ref_names'][0])
        q, r = (Path(ds.img_path)/(p.stem+'.jpg') for p in (qmask, rmask))
        boxes, ids, objects = [None, None], [r.stem, q.stem], [None, None]
        annotations = [rmask, qmask]
        global_class = str(s['class_sample'])
    else:
        qname, rname = str(s['tgt_name']), str(s['ref_names'][0])
        base = Path(ds.base_path if name != 'paco_part' else ds.img_path)
        q, r = base/qname, base/rname
        ids, boxes, objects = [rname, qname], [None, None], [None, None]
        global_class = int(s['class_sample'])
        if name == 'paco_part':
            def xywh(box):
                return [int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])]
            boxes = [xywh(s['ref_boxes'][0]), xywh(s['tgt_obj_bbox'])]
            objects = [int(s['sel_ref_id']), int(s['sel_tgt_id'])]
            annotations = [Path(ds.anno_path)/'paco_part_train.pkl',
                           Path(ds.anno_path)/'paco_part_val.pkl']
        elif name == 'coco':
            annotations = [base/'annotations'/Path(n).with_suffix('.png') for n in (rname, qname)]
        else:
            annotations = [Path(ds.anno_path)/'lvis_val.pkl']
    return dict(reference_path=r, query_path=q, reference_crop=boxes[0], query_crop=boxes[1],
                reference_photo_id=ids[0], query_photo_id=ids[1],
                reference_object_id=objects[0], query_object_id=objects[1],
                global_class_id=global_class, annotation_paths=annotations)


def decoded_rgb(path, crop=None):
    with Image.open(path) as im:
        x = np.asarray(im.convert('RGB')).copy()
    if crop is not None:
        x0, y0, x1, y1 = crop
        x = x[y0:y1, x0:x1]  # Preserve official NumPy slicing, including bounds.
    return Image.fromarray(x)


def record_episode(ds, index, fold, assets, mask_root, hash_cache=None):
    """Manifest construction, not candidate inference or confirmation scoring."""
    batch, selection = sample_observed(ds, index)
    source = _source_paths(ds, selection)
    assets, mask_root = Path(assets).resolve(), Path(mask_root).resolve()
    cached = {} if hash_cache is None else hash_cache
    record = dict(dataset=ds.benchmark, fold=fold, official_index=index,
                  episode_id=f'{ds.benchmark}/{fold}/{index}',
                  loader_class_id=int(batch['class_id']), query_gt_role='protocol preparation only')
    for k, value in source.items():
        if k not in ('reference_path', 'query_path', 'annotation_paths'):
            record[k] = _jsonable(value)
    record['source_files'] = {}
    for p in [source['reference_path'], source['query_path'], *source['annotation_paths']]:
        rel = str(p.relative_to(assets))
        if rel not in cached:
            cached[rel] = file_hash(p)
        record['source_files'][rel] = cached[rel]
    for role, actual in [('reference', batch['ref_imgs'][0]), ('query', batch['tgt_img'])]:
        replay = decoded_rgb(source[role+'_path'], source[role+'_crop'])
        digest = array_hash(np.asarray(actual))
        if array_hash(np.asarray(replay)) != digest:
            raise RuntimeError('Recorded source/crop does not reproduce official RGB')
        record[role+'_rgb_hash'] = digest
        record[role+'_path'] = str(source[role+'_path'].relative_to(assets))
        record[role+'_size_hw'] = [actual.height, actual.width]
    for role, tensor in [('reference', batch['ref_masks'][0]), ('query', batch['tgt_mask'])]:
        mask = (tensor.numpy() > 0).astype(np.uint8)
        if list(mask.shape) != record[role+'_size_hw']:
            raise ValueError('Official image/mask geometry differs')
        digest = array_hash(mask)
        destination = mask_root/role/(digest+'.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            Image.fromarray(mask*255).save(destination)
        record[role+'_mask_hash'] = digest
        record[role+'_mask_path'] = str(destination)
    record['ignore_policy'] = 'official five loaders return no tgt_ignore_idx; COCO non-target including 255 becomes background'
    return record


def load_inputs(record, assets):
    """Frozen inference interface: exactly reference RGB + reference mask + query RGB."""
    import torch
    assets = Path(assets)
    images = {}
    for role in ('reference', 'query'):
        images[role] = decoded_rgb(assets/record[role+'_path'], record.get(role+'_crop'))
        if array_hash(np.asarray(images[role])) != record[role+'_rgb_hash']:
            raise ValueError('Frozen RGB input changed')
    with Image.open(record['reference_mask_path']) as im:
        mask = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    if array_hash(mask) != record['reference_mask_hash']:
        raise ValueError('Frozen reference mask changed')
    return images['reference'], torch.from_numpy(mask.astype(bool)), images['query']
