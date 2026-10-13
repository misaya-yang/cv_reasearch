"""Choose unseen original photos by metadata, then reuse official mask loaders.

Only the class and photo draws are fixed here. Official annotation decoding,
object selection, crops and input recording are reused without modification.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / 'cv_data'
OUT = DATA / 'a/x1013_photo_holdout'
SEED = 2026101303
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PIL import Image
from ics import official_data as official
from ics import extended_benchmark_data as extended


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
def order(x): return hashlib.sha256((str(SEED) + ':' + str(x)).encode()).hexdigest()
def reference_pool(photo): return int(order(photo)[:8], 16) % 5 == 0


def select():
    catalog, inventory = read(OUT/'fresh_catalog.json'), read(OUT/'inventory.json')
    assert all(v['freshness_status'] == 'metadata_exposure_registry_disjoint'
               for v in inventory['per_domain'].values()), 'Resolve named old-photo aliases before freezing all-domain holdout'
    groups = defaultdict(list)
    for x in catalog:
        groups[(x['dataset'], int(x['fold']), str(x['class_id']))].append(x)
    candidates, unavailable = {}, []
    for key, entries in groups.items():
        refs = sorted([x for x in entries if reference_pool(x['canonical_id'])], key=lambda x: order(x['path']))
        queries = sorted([x for x in entries if not reference_pool(x['canonical_id'])], key=lambda x: order(x['path']))
        if refs and queries:
            candidates[key] = [refs, queries]
        else:
            unavailable.append({'group': key, 'reference_n': len(refs), 'query_n': len(queries)})
    pairs = []
    for domain in inventory['per_domain']:
        keys = sorted([k for k in candidates if k[0] == domain], key=order)
        used = set()
        while len(used) < 500:
            before = len(used)
            for key in keys:
                refs, queries = candidates[key]
                while queries and queries[-1]['canonical_id'] in used:
                    queries.pop()
                if not queries: continue
                q = queries.pop()
                r = refs[int(order(q['canonical_id'] + ':' + str(key))[:16], 16) % len(refs)]
                used.add(q['canonical_id'])
                index = domain + '_' + order([key, r['path'], q['path']])[:16]
                pairs.append(dict(index=index, dataset=domain, fold=key[1], class_id=q['class_id'],
                                  reference=r, query=q))
                if len(used) == 500: break
            if len(used) == before: break
    write(OUT/'selected_pairs.json', pairs)
    write(OUT/'selection.json', dict(seed=SEED, per_domain_max_unique_query=500,
          n=len(pairs), by_domain=Counter(x['dataset'] for x in pairs), unavailable_groups=unavailable,
          role_split='sha256(seed:canonical_original_photo) modulo5 ==0 reference, others query',
          selection='Round robin eligible fold/class groups, one query parent photo per domain; no mask-area or prediction use',
          inventory_sha256=official.file_hash(OUT/'inventory.json')))
    print(json.dumps({'selected':len(pairs),'domains':Counter(x['dataset'] for x in pairs)}), flush=True)
    return pairs


def identifier(value):
    if isinstance(value, dict):
        if len(value) != 1: return None
        value = next(iter(value.values()))
    if not isinstance(value, (str, np.str_)): return None
    return Path(str(value)).stem.replace('_mask', '')


@contextmanager
def fixed_photo_draws(ds, pair):
    original = np.random.choice
    qname = identifier(pair['query']['loader_name'])
    rname = identifier(pair['reference']['loader_name'])
    cls = pair['class_id']
    def choice(values, *args, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_locals.get('self') is ds:
            if ds.benchmark == 'coco' and values is ds.class_ids:
                return np.asarray([cls])
            if ds.benchmark == 'paco_part' and values is ds.class_ids_ori:
                return np.asarray([cls])
            role = 'reference' if ('ref_names' in frame.f_locals or 'support_img_ids' in frame.f_locals) else 'query'
            wanted = rname if role == 'reference' else qname
            if isinstance(values, (list, tuple, np.ndarray)):
                matches = [v for v in values if identifier(v) == wanted]
                if matches:
                    return np.asarray([matches[0]], dtype=object)
        return original(values, *args, **kwargs)
    np.random.choice = choice
    try: yield
    finally: np.random.choice = original


def loader_index(ds, pair):
    c, q = pair['class_id'], pair['query']['loader_name']
    d = pair['dataset']
    if d == 'lvis': return list(ds.class_ids_ori).index(c)
    if d == 'pascal_part': return ds.cat_part_name.index(c)
    if d == 'suim': return list(ds.categories).index(c)
    if d == 'isaid': return ds.img_metadata.index([q, c])
    if d == 'isic': return int(c)
    return 0


def road_record(pair):
    row = dict(dataset='deepglobe_road', fold=-1, loader_class_id=0, global_class_id=0,
               episode_id=pair['index'], source_files={}, protocol='Existing local road image pool; not official FoRIS road protocol')
    for role in ('reference', 'query'):
        path = DATA / pair[role]['path']
        annotation = path.parent.parent/'mask'/path.name.replace('_sat.jpg', '_mask.png')
        with Image.open(path) as im: rgb = np.asarray(im.convert('RGB'))
        with Image.open(annotation) as im: mask = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        destination = OUT/'masks'/role/(official.array_hash(mask) + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists(): Image.fromarray(mask*255).save(destination)
        row.update({role+'_path':str(path.relative_to(DATA)), role+'_crop':None,
                    role+'_photo_id':pair[role]['canonical_id'], role+'_rgb_hash':official.array_hash(rgb),
                    role+'_size_hw':list(rgb.shape[:2]), role+'_mask_path':str(destination),
                    role+'_mask_hash':official.array_hash(mask), role+'_mask_size_hw':list(mask.shape)})
        row['source_files'].update({str(p.relative_to(DATA)):official.file_hash(p) for p in (path, annotation)})
    return row


def main():
    assert not (OUT/'evaluation.json').exists(), 'Do not change the selected cohort after inference freeze'
    pairs = select()
    datasets, hashes, bindings = {}, {}, []
    for i, pair in enumerate(pairs):
        record_path = OUT/'protocol_records'/(pair['index']+'.json')
        if record_path.exists():
            binding = read(record_path)
        else:
            d, fold = pair['dataset'], pair['fold']
            np.random.seed(int(order(pair['index'])[:8], 16))
            if d == 'deepglobe_road':
                row = road_record(pair)
            else:
                if (d, fold) not in datasets:
                    datasets[d, fold] = extended.build_dataset(d, fold, DATA)
                ds = datasets[d, fold]
                with fixed_photo_draws(ds, pair):
                    row = extended.record_episode(ds, loader_index(ds, pair), fold, DATA, OUT/'masks', hashes)
            row['episode_id'] = 'photo_holdout/' + pair['index']
            for role in ('reference', 'query'):
                assert (DATA/row[role+'_path']).resolve() == (DATA/pair[role]['path']).resolve(), (pair['index'], role, row[role+'_path'], pair[role]['path'])
            binding = dict(index=pair['index'], dataset=d, row=row,
                    canonical_reference_photo=pair['reference']['canonical_id'],
                    canonical_query_photo=pair['query']['canonical_id'])
            write(record_path, binding)
        bindings.append(binding)
        if i % 50 == 0: print(json.dumps({'prepared':i+1,'total':len(pairs)}), flush=True)
    write(OUT/'manifest.json', bindings)
    print(json.dumps({'state':'NEW_PHOTO_MANIFEST_READY','n':len(bindings),'sha256':official.file_hash(OUT/'manifest.json')}), flush=True)


if __name__ == '__main__': main()
