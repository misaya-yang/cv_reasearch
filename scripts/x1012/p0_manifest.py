"""P0-1: cache-only, photo-aware exploration/confirmation and nested K-shot lists.

Only protocol metadata (and iSAID ignore labels) are opened, never predictions or
query scores. Old-photo confirmation is explicitly distinguished from photo holdout.
"""
from __future__ import annotations
import copy
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import lab

OUT = lab.ASSETS / 'a/x1012_P0'
PLAN = lab.ROOT / 'docs/research/INDEPENDENT_BASE_PLAN_20261012.md'
SEED = 20261012
WEAK = {'coco', 'paco_part', *lab.EXT_DOMAINS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(None)
def raw_metadata(payload):
    p = Path(payload); entry = p.parent / 'entry.json'
    info = json.loads(entry.read_text()); f = info['features']['O/24']
    assert f['shape'] == [4096, 1024] and f['dtype'] == 'float32'
    assert p.name == info['file'] and p.is_file()
    return {'payload_path': str(p), 'payload_sha256': info['file_sha256'],
            'entry_path': str(entry), 'entry_sha256': sha(entry),
            'input_sha256': info['input_tensor_hash'], 'tensor_sha256': f['tensor_sha256'],
            'feature_key': 'O/24'}


def normalized(binding):
    b = copy.deepcopy(binding)
    for role in ('reference_raw', 'query_raw_identity'):
        metadata = raw_metadata(b['fit'][role]['payload_path'])
        for key in ('payload_sha256', 'entry_sha256', 'input_sha256', 'tensor_sha256'):
            if key in b['fit'][role]:
                assert b['fit'][role][key] == metadata[key]
        b['fit'][role].update(metadata)
    return b


def observation(b, role):
    r = b['row']; key = 'reference_raw' if role == 'reference' else 'query_raw_identity'
    o = {'dataset': b['dataset'], 'fold': r['fold'], 'class_id': r['global_class_id'],
         'loader_class_id': r['loader_class_id'], 'source_episode': r['episode_id'],
         'source_role': role, 'source_files': r.get('source_files', {}), 'raw': b['fit'][key],
         'view': {k[len(role)+1:]: v for k, v in r.items() if k.startswith(role + '_')}}
    # Path identity excludes all crops of a seen photograph.
    o['photo'] = str((lab.ASSETS / o['view']['path']).resolve())
    o['id'] = digest([o['dataset'], o['fold'], o['class_id'], o['raw']['input_sha256'], o['view']['mask_hash']])
    return o


def pair_signature(r, q):
    return digest([r['id'], q['id']])


def pair_candidates(pool, old_pairs, rng):
    if len(pool) < 2:
        return []
    result = []; seen = set(); tries = 0
    # Rejection sampling avoids materializing O(images^2) for one-class domains.
    limit = min(4000, max(100, len(pool) * len(pool) * 2))
    while len(result) < 20 and tries < limit:
        tries += 1; r, q = rng.sample(pool, 2)
        signature = pair_signature(r, q)
        if r['photo'] == q['photo'] or signature in old_pairs or signature in seen:
            continue
        seen.add(signature); result.append((r, q))
    return result


def ignore_for_reference_as_query(o):
    if o['dataset'] != 'isaid' or o['source_role'] != 'reference':
        return
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(lab.ROOT / 'src'))
    from ics.official_data import array_hash
    source = lab.ASSETS / o['view']['annotation_path']
    # Exactly the released iSAID floor(raw/255) ignore convention, protocol only.
    with Image.open(source) as im:
        raw = np.asarray(im)
    assert raw.ndim == 2
    mask = (raw // 255).astype(bool).astype(np.uint8)
    h = array_hash(mask); path = OUT / 'masks/query_ignore' / (h + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.fromarray(mask * 255).save(path)
    o['view'].update(ignore_mask_path=str(path), ignore_mask_hash=h, ignore_mask_size_hw=list(mask.shape))


def make_pair(r, q, split, independence):
    assert r['dataset'] == q['dataset'] and r['fold'] == q['fold'] and r['class_id'] == q['class_id']
    assert r['photo'] != q['photo']
    ignore_for_reference_as_query(q)
    signature = pair_signature(r, q); idx = 'new_' + signature[:20]
    row = {'dataset': r['dataset'], 'episode_id': 'x1012_P0/' + idx,
           'fold': r['fold'], 'global_class_id': r['class_id'], 'loader_class_id': r['loader_class_id'],
           'official_index': -1, 'query_gt_role': 'protocol only; no scoring before C1',
           'source_files': {**r['source_files'], **q['source_files']},
           'gt_alignment': 'source mask nearest to CLI1024; original official input crop retained',
           'ignore_policy': 'released iSAID ignore convention' if r['dataset'] == 'isaid' else 'no ignore'}
    for role, o in (('reference', r), ('query', q)):
        row.update({role + '_' + k: v for k, v in o['view'].items()})
    return {'dataset': r['dataset'], 'index': idx, 'episode_id': row['episode_id'],
            'cohort': 'x1012_P0_new_pair', 'row': row,
            'fit': {'reference_raw': r['raw'], 'query_raw_identity': q['raw']},
            'old_arms': {}, 'split': split, 'pair_signature': signature,
            'exposure': {'new_pair': True, 'confirmation_level': independence,
                         'source_images_preexisting': True, 'baseline_predictions_previously_scored': True,
                         'photo_holdout_global_claim': False},
            'source_observations': {'reference': r['id'], 'query': q['id']}}


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve existing P0 manifests.'
    OUT.mkdir()
    originals = [normalized(b) for b in lab.bindings() + lab.ext_bindings()]
    marked = {str(b['index']) for n in (3, 40, 60, 100, 150, 200, 300)
              for offset in (0, 1) for b in lab.subset(n, offset=offset)}
    marked.update(str(b['index']) for b in lab.bindings()[::4])
    marked.update(str(b['index']) for b in lab.ext_bindings())
    pools = defaultdict(dict); observations = {}; old_pairs = set(); exposed_photos = set()
    for b in originals:
        b['exposure'] = {'ind_lab_schedule_exposed': str(b['index']) in marked,
                         'baseline_and_repo_arms_previously_scored': True,
                         'globally_untouched': False}
        views = [observation(b, role) for role in ('reference', 'query')]
        for o in views:
            observations[o['id']] = o
            pools[(o['dataset'], str(o['fold']), str(o['class_id']))][o['id']] = o
            if b['exposure']['ind_lab_schedule_exposed']:
                exposed_photos.add(o['photo'])
        old_pairs.add(pair_signature(*views))
    # Keep all unseen views of a photograph together, across classes and crops.
    reserve_photos = {o['photo'] for o in observations.values()
                      if o['dataset'] not in WEAK and o['photo'] not in exposed_photos
                      and int(digest([SEED, o['photo']])[:8], 16) % 2 == 1}
    explore = []
    for b in originals:
        # Exactly the plan: exposed originals plus newly paired development rows.
        if b['exposure']['ind_lab_schedule_exposed'] and all(str((lab.ASSETS / b['row'][role + '_path']).resolve()) not in reserve_photos
               for role in ('reference', 'query')):
            b['split'] = 'explore'; explore.append(b)
    confirm = []; weak_fallback = []
    for group, unique in sorted(pools.items()):
        pool = sorted(unique.values(), key=lambda o: o['id'])
        rng = random.Random(int(digest([SEED, group])[:16], 16))
        if group[0] in WEAK:
            new = pair_candidates(pool, old_pairs, rng)
            midpoint = (len(new) + 1) // 2
            for i, (r, q) in enumerate(new):
                split = 'explore' if i < midpoint else 'confirm'
                b = make_pair(r, q, split, 'old_photo_new_pair')
                (explore if split == 'explore' else confirm).append(b)
                if group[0] == 'paco_part' and split == 'confirm':
                    weak_fallback.append(b)
        else:
            cpool = [o for o in pool if o['photo'] in reserve_photos]
            cnew = pair_candidates(cpool, old_pairs, rng)[:10]
            epool = [o for o in pool if o['photo'] not in reserve_photos]
            enew = pair_candidates(epool, old_pairs, rng)[:20-len(cnew)]
            confirm.extend(make_pair(r, q, 'confirm', 'ind_lab_photo_holdout') for r, q in cnew)
            explore.extend(make_pair(r, q, 'explore', 'development') for r, q in enew)
    assert len({str(b['index']) for b in explore + confirm}) == len(explore) + len(confirm)
    for b in confirm:
        if b['exposure']['confirmation_level'] == 'ind_lab_photo_holdout':
            photos = {str((lab.ASSETS / b['row'][role+'_path']).resolve()) for role in ('reference', 'query')}
            assert not photos & exposed_photos and photos <= reserve_photos
    assert all(all(str((lab.ASSETS / b['row'][role+'_path']).resolve()) not in reserve_photos
                   for role in ('reference', 'query')) for b in explore)
    # K references are nested: K=1,2,3,5 share the same first references.
    query_views = {observation(b, 'query')['id']: observation(b, 'query') for b in explore}
    confirm_queries = {str((lab.ASSETS / b['row']['query_path']).resolve()) for b in confirm}
    kshot = []; skipped = Counter()
    for q in sorted(query_views.values(), key=lambda o: o['id']):
        if q['photo'] in confirm_queries:
            continue
        group = (q['dataset'], str(q['fold']), str(q['class_id']))
        candidates = [o for o in pools[group].values() if o['photo'] != q['photo'] and o['photo'] not in reserve_photos]
        candidates.sort(key=lambda o: digest([SEED, q['id'], o['id']]))
        refs = []; used = set()
        for o in candidates:
            if o['photo'] not in used:
                refs.append(o); used.add(o['photo'])
                if len(refs) == 5:
                    break
        if not refs:
            skipped[q['dataset']] += 1; continue
        base = make_pair(refs[0], q, 'kshot', 'development')
        base['index'] = 'k_' + q['id'][:20]
        base['requested_k'] = [1, 2, 3, 5]
        base['available_k'] = [k for k in (1, 2, 3, 5) if len(refs) >= k]
        base['references'] = refs
        kshot.append(base)
    write(OUT/'explore.json', explore); write(OUT/'confirm.json', confirm)
    write(OUT/'kshot.json', kshot); write(OUT/'observations.json', list(observations.values()))
    stats = {}
    for d in lab.ALL_DOMAINS:
        c = [b for b in confirm if b['dataset'] == d]
        e = [b for b in explore if b['dataset'] == d]
        stats[d] = {'source_episodes': sum(b['dataset'] == d for b in originals),
                    'schedule_exposed_episodes': sum(b['dataset'] == d and str(b['index']) in marked for b in originals),
                    'explore': len(e), 'explore_new_pairs': sum(b['cohort'] == 'x1012_P0_new_pair' for b in e),
                    'confirm': len(c), 'confirm_photo_holdout': sum(b['exposure']['confirmation_level'] == 'ind_lab_photo_holdout' for b in c),
                    'confirm_old_photo_new_pair': sum(b['exposure']['confirmation_level'] == 'old_photo_new_pair' for b in c),
                    'kshot_queries': sum(b['dataset'] == d for b in kshot),
                    'kshot_available': {str(k): sum(b['dataset'] == d and k in b['available_k'] for b in kshot) for k in (1,2,3,5)}}
    config = {'state': 'P0_1_COMPLETE_CONFIRMATION_UNSCORED', 'seed': SEED, 'max_new_pairs_per_fold_class': 20,
              'exposure_schedule': {'N': [3,40,60,100,150,200,300], 'offsets': [0,1], 'stride4': True, 'all_extension': True},
              'photo_holdout_scope': 'IND lab exposure schedule; all original repository baseline outputs already scored',
              'paco_confirmation_exception': 'All 600 exposed; use separately labelled old-photo new-pair evaluation, never claim unseen-photo confirmation',
              'old_photo_confirmation_domains': sorted(WEAK), 'query_gt_scores_read': 0, 'new_encoder_calls': 0,
              'ignore_preparation': 'iSAID original reference labels only, no scoring',
              'source_sha256': {str(p): sha(p) for p in (Path(__file__), Path(lab.__file__), PLAN, lab.BINDINGS)},
              'manifest_sha256': {p.name: sha(p) for p in (OUT/'explore.json',OUT/'confirm.json',OUT/'kshot.json',OUT/'observations.json')},
              'stats': stats, 'seconds': time.monotonic()-started}
    write(OUT/'config.json', config)
    print(json.dumps(config['stats'], ensure_ascii=False), flush=True)
    print(json.dumps({'state': config['state'], 'seconds': config['seconds'], 'new_encoder_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
