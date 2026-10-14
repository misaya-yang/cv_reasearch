"""W12: the frozen MFU under the released FoRIS benchmark protocols (paper main table).

Motivation: MFU was selected on the x1013 S half, scored once on T (ten domains >= complete FoRIS)
and is being confirmed on fresh photos.  All of those pools are our own samplings.  FoRIS Table 1
reports the released seed-0 loaders with per-fold class-mean IoU, then the fold mean.  The paper's
main table needs MFU and complete FoRIS on exactly those episodes, paired, in the released CLI-1024
frame.  The method is frozen (program 342b57ab...): this script reuses the five-arm wrapper of
scripts/x1012s/photo_holdout.py unchanged (main=MFU, U, FoRIS, MEAN, G_mfu).  Official episodes
overlap our development pools; this is the benchmark measurement, not a generalisation test.

    python scripts/x1012/w12_official.py prepare
    python scripts/x1012/w12_official.py predict DATASET[,DATASET...] [workers]
    python scripts/x1012/w12_official.py score   DATASET[,DATASET...]
"""
from __future__ import annotations

import gzip, hashlib, json, multiprocessing, sys, time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1012s'))
import photo_holdout as ph

A = ROOT.parent / 'cv_data/a'
OUT = A / 'x1013_official'
SOURCES = {  # dataset -> (prepared directory, datasets taken from it)
    'official_s0_manifest_20261008': ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim'),
    'extended_benchmarks_seed0_20261010_v3': ('isic',),
    'isaid_seed0_full_20261013': ('isaid',),
    'chest_seed0_manifest_20261010': ('lung',),
    'fundus_seed0_manifest_20261010': ('fundus',),
}
ARMS = ph.ARMS
CACHE = 96  # in-memory O24 features (16 MB each); 400 held ~22 GB with the MPS pool and pushed the host into swap
RAW = A / 'paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158'
PUBLISHED = {'coco': 60.9, 'lvis': 42.8, 'pascal_part': 55.8, 'paco_part': 42.3, 'suim': 59.1,
             'isaid': 53.6, 'isic': 62.9, 'lung': 87.6, 'fundus': 19.7}


def key(row):
    return row['episode_id'].replace('/', '_')


def road_rows():
    """DeepGlobe road has no released protocol: the local custom200 (first 200 Q, next 200 R, seed 20261009).
    Rows follow the convention of the existing Deep100 rows (verified hash-identical on those 100)."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(A / 'x1012s2_runtime/mfu_boundary/runtime'))
    from ics.official_data import array_hash, decoded_rgb
    prep = A / 'deepglobe200_preparation_20261009'
    old = {b['row']['episode_id']: b['row'] for b in ph.read(A / 'x1012_P0/explore.json') + ph.read(A / 'x1012_P0/confirm.json')
           if b['dataset'] == 'deepglobe_road'}
    rows, checked = [], 0
    for i, e in enumerate(map(json.loads, (prep / 'episodes.jsonl').read_text().splitlines())):
        r = dict(dataset='deepglobe_road', episode_id=e['episode_id'], fold=0, global_class_id=0, loader_class_id=0,
                 query_crop=None, reference_crop=None, query_gt_role='scoring_only')
        for role, ident in (('query', e['query_id']), ('reference', e['reference_id'])):
            image = prep / 'inputs' / role / 'image' / f'{i:04d}_{ident}_sat.jpg'
            r[role + '_path'] = str(image.relative_to(ph.ASSETS))
            r[role + '_rgb_hash'] = array_hash(np.asarray(decoded_rgb(image)))
            r[role + '_size_hw'] = [1024, 1024]
            r[role + '_photo_id'] = ident
            with Image.open(prep / 'inputs' / role / 'mask' / f'{i:04d}_{ident}_mask.png') as im:
                m = (np.asarray(im.convert('L')) > 127).astype(np.uint8)
            r[role + '_mask_hash'] = array_hash(m)
            dest = OUT / 'masks/road' / role / (e['episode_id'] + '.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(m * 255).save(dest)
            r[role + '_mask_path'] = str(dest)
            r[role + '_mask_size_hw'] = [1024, 1024]
        if e['episode_id'] in old:
            o = old[e['episode_id']]
            assert all(o[k] == r[k] for k in ('query_rgb_hash', 'reference_rgb_hash', 'query_mask_hash', 'reference_mask_hash')), e
            checked += 1
        rows.append({'index': 'road_' + e['episode_id'], 'dataset': 'deepglobe_road', 'row': r})
    assert checked == 100, checked
    return rows


def prepare():
    rows, folds = road_rows(), {'deepglobe_road/0': [0]}
    for source, datasets in SOURCES.items():
        prepared = ph.read(A / source / 'prepared.json')
        assert not set(datasets) & set(prepared.get('pending_datasets', {})), source
        for name, fold in prepared['folds'].items():
            if name.split('/')[0] in datasets:
                assert fold['selected_length'] == fold['official_length'], name
                folds[name] = fold['expected_class_ids']
        rows += [{'index': key(r), 'dataset': r['dataset'], 'row': r}
                 for r in ph.read(A / source / 'manifest.json') if r['dataset'] in datasets]
    assert len({b['index'] for b in rows}) == len(rows)
    ph.write(OUT / 'manifest.json', rows)
    ph.write(OUT / 'protocol.json', dict(folds=folds, sources={s: ph.sha(A / s / 'manifest.json') for s in SOURCES},
                                         manifest_sha256=ph.sha(OUT / 'manifest.json'), program_sha256=ph.PROGRAM_SHA,
                                         wrapper_sha256=ph.sha(ph.__file__), arms=ARMS, n=len(rows)))
    print(json.dumps(dict(n=len(rows), folds=len(folds))), flush=True)


def bfs_order(rows):
    """Episode order that keeps shared images close (breadth-first over the image-sharing graph).
    Only R/Q input identities are used; no labels or outputs."""
    image = lambda b, role: (b['row'][role + '_rgb_hash'], json.dumps(b['row'].get(role + '_crop')))
    rows = sorted(rows, key=lambda b: b['index'])
    by = defaultdict(list)
    for i, b in enumerate(rows):
        by[image(b, 'reference')].append(i); by[image(b, 'query')].append(i)
    done, out = [False] * len(rows), []
    for seed in sorted(by, key=lambda h: (-len(by[h]), h)):
        queue = deque([seed])
        while queue:
            h = queue.popleft()
            for i in by[h]:
                if not done[i]:
                    done[i] = True; out.append(rows[i])
                    queue.extend(g for g in (image(rows[i], 'reference'), image(rows[i], 'query')) if g != h)
    return out


def raw_cache_index():
    """input-tensor hash -> entry directory of the shared frozen O24 cache (same encoder profile)."""
    index = {}
    for entry in RAW.iterdir():
        if (entry / 'entry.json').is_file():
            index[ph.read(entry / 'entry.json')['input_tensor_hash']] = entry
    return index


def stored_features(entry):
    import numpy as np
    import torch
    info = ph.read(entry / 'entry.json')
    with np.load(entry / info['file'], allow_pickle=False) as z:
        value = z['O/24'].copy()
    assert value.shape == (4096, 1024) and value.dtype == np.float32
    sys.path.insert(0, str(A / 'x1012s2_runtime/mfu_boundary/runtime'))
    from ics.official_data import array_hash
    assert array_hash(value) == info['features']['O/24']['tensor_sha256']
    return torch.from_numpy(value)


def predict(datasets, workers):
    import torch
    assert ph.sha(ph.PROGRAM) == ph.PROGRAM_SHA
    manifest = OUT / 'manifest.json'
    msha = ph.sha(manifest)
    allrows = ph.read(manifest)
    task, rt = ph.runtime()
    source = (ROOT / 'src/ics/data.py').read_text()
    import ast
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'TimmDINOv3')
    path = OUT / 'frozen/encoder.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    text = 'import json\nimport torch\n' + ast.get_source_segment(source, node) + '\n'
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    encoder = ph.module('w12_encoder', path).TimmDINOv3(str(ph.WEIGHTS)).to('mps').eval().requires_grad_(False)
    stored = raw_cache_index()
    cache, started = OrderedDict(), time.monotonic()

    def load(b):  # runs in a decode thread; identical calls to the frozen loader and transform
        reference, _, query = rt.load_inputs(b['row'], ph.ASSETS)
        inputs = {'reference': rt.host._transform(reference), 'query': rt.host._transform(query)}
        return inputs, {role: task.array_hash(x.numpy()) for role, x in inputs.items()}

    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=ph.worker_init) as pool, ThreadPoolExecutor(3) as decode, torch.inference_mode():
        for d in datasets:
            rows = bfs_order([b for b in allrows if b['dataset'] == d])
            todo = [b for b in rows if not (OUT / 'records' / (b['index'] + '.json')).exists()]
            completed, pending = len(rows) - len(todo), set()
            ahead = deque(decode.submit(load, b) for b in todo[:24])
            for n, original in enumerate(todo):
                began = time.monotonic()
                inputs, hashes = ahead.popleft().result()
                if n + 24 < len(todo):
                    ahead.append(decode.submit(load, todo[n + 24]))
                b = json.loads(json.dumps(original))
                features, b['fit'], hits = {}, {}, 0
                for role, h in hashes.items():
                    if h in cache:
                        cache.move_to_end(h)
                    elif h in stored:
                        cache[h] = stored_features(stored[h]); hits += 1
                missing = [role for role, h in hashes.items() if h not in cache]
                if missing:
                    maps = encoder.get_intermediate_layers(torch.stack([inputs[r] for r in missing]).to('mps'), n=1, reshape=True)[0]
                    torch.mps.synchronize()
                    for role, raw in zip(missing, maps.cpu().permute(0, 2, 3, 1).reshape(len(missing), 4096, 1024).contiguous().unbind(0)):
                        assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
                        cache[hashes[role]] = raw
                    del maps
                    if n % 32 == 0:
                        torch.mps.empty_cache()
                for role, h in hashes.items():
                    features[role] = cache[h]
                    entry = OUT / 'inputs' / (h + '.json')
                    if not entry.exists():
                        ph.write(entry, dict(file='in_memory.npz', input_tensor_hash=h,
                                             storage='FP32 tensor supplied directly; no feature payload written'))
                    b['fit']['reference_raw' if role == 'reference' else 'query_raw_identity'] = {
                        'payload_path': str(entry.parent / 'in_memory.npz'), 'entry_path': str(entry)}
                while len(cache) > CACHE:
                    cache.popitem(last=False)
                pending.add(pool.submit(ph.save_prediction, OUT, b, features,
                                        dict(manifest_sha256=msha, new_encoded_images=len(missing), raw_cache_hits=hits,
                                             input_and_encoder_seconds=time.monotonic() - began)))
                if len(pending) >= 2 * workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result(); completed += 1
                        if completed % 50 == 0:
                            print(json.dumps(dict(dataset=d, done=completed, total=len(rows),
                                                  seconds=round(time.monotonic() - started, 1))), flush=True)
            for f in pending:
                f.result(); completed += 1
            records = [ph.read(OUT / 'records' / (b['index'] + '.json')) for b in rows]
            assert all(ph.sha(r['prediction_path']) == r['prediction_sha256'] for r in records)
            ph.write(OUT / 'sealed' / (d + '.json'), dict(state='ALL_FIXED_NATURAL_OUTPUTS_COMPLETE', dataset=d, n=len(rows),
                                                         manifest_sha256=msha, records=records))
            print(json.dumps(dict(dataset=d, state='SEALED', n=len(rows), seconds=round(time.monotonic() - started, 1))), flush=True)


def _iu(b):
    import numpy as np
    import torch
    from PIL import Image
    row = b['row']

    def mask(path):
        with Image.open(path) as image:
            x = torch.from_numpy((np.asarray(image.convert('L')) > 0).astype(np.float32))
        return torch.nn.functional.interpolate(x[None, None], (1024, 1024), mode='nearest')[0, 0].numpy() > .5

    gt = mask(row['query_mask_path'])
    valid = ~mask(row['query_ignore_mask_path']) if row.get('query_ignore_mask_path') else np.ones_like(gt)
    rec = ph.read(OUT / 'records' / (b['index'] + '.json'))
    out = dict(index=b['index'], dataset=b['dataset'], fold=row['fold'], cls=row['loader_class_id'],
               photo=row['query_photo_id'], rho=rec['reference_rho'], gt=int((gt & valid).sum()))
    with np.load(rec['prediction_path'], allow_pickle=False) as z:
        for a in ARMS:
            m = np.unpackbits(z[a])[:1024 * 1024].reshape(1024, 1024).astype(bool) & valid
            i = int((m & gt & valid).sum())
            out[a] = [i, int(m.sum() + (gt & valid).sum()) - i, int(m.sum())]
    return out


def slots(records, folds):
    """Slot index of every episode and the slot lists of each fold (released AverageMeter classes)."""
    import numpy as np
    names = [(int(n.split('/')[1]), c) for n, cs in folds.items() for c in cs]
    index = {s: i for i, s in enumerate(names)}
    by_fold = [np.array([index[(int(n.split('/')[1]), c)] for c in cs]) for n, cs in folds.items()]
    return np.array([index[(r['fold'], r['cls'])] for r in records]), by_fold, len(names)


def official(records, arm, layout, weights=None):
    """Per fold, class IoU = sum I / max(sum U, 1) over the expected classes, class mean; then fold mean."""
    import numpy as np
    s, by_fold, n = layout
    x = np.array([r[arm][:2] for r in records], np.float64)
    w = np.ones(len(records)) if weights is None else weights
    iou = np.bincount(s, w * x[:, 0], n) / np.maximum(np.bincount(s, w * x[:, 1], n), 1)
    return float(np.mean([100 * iou[f].mean() for f in by_fold]))


def score(datasets):
    import numpy as np
    protocol = ph.read(OUT / 'protocol.json')
    allrows = ph.read(OUT / 'manifest.json')
    summary = ph.read(OUT / 'scores/summary.json') if (OUT / 'scores/summary.json').exists() else {}
    for d in datasets:
        seal = ph.read(OUT / 'sealed' / (d + '.json'))
        rows = [b for b in allrows if b['dataset'] == d]
        assert seal['n'] == len(rows)
        with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')) as pool:
            R = list(pool.map(_iu, rows, chunksize=16))
        (OUT / 'scores').mkdir(parents=True, exist_ok=True)
        with gzip.open(OUT / 'scores' / (d + '.jsonl.gz'), 'wt') as f:
            f.writelines(json.dumps(r) + '\n' for r in R)
        folds = {k: v for k, v in protocol['folds'].items() if k.split('/')[0] == d}
        layout = slots(R, folds)
        point = {a: official(R, a, layout) for a in ARMS}
        photos = {ph_: i for i, ph_ in enumerate(sorted({r['photo'] for r in R}))}
        p = np.array([photos[r['photo']] for r in R])
        rng = np.random.default_rng(2026101303)
        diffs = {a: [] for a in ARMS if a != 'main'}
        for _ in range(1000):
            w = np.bincount(rng.integers(0, len(photos), len(photos)), minlength=len(photos))[p]
            m = official(R, 'main', layout, w)
            for a in diffs:
                diffs[a].append(m - official(R, a, layout, w))
        summary[d] = dict(n=len(R), photos=len(photos), official_miou=point, published_foris=PUBLISHED.get(d),
                          main_minus={a: dict(point=point['main'] - point[a], ci95=np.percentile(v, [2.5, 97.5]).tolist())
                                      for a, v in diffs.items()})
        ph.write(OUT / 'scores/summary.json', summary)
        print(json.dumps({d: {a: round(x, 2) for a, x in point.items()}}), json.dumps(
            {a: [round(summary[d]['main_minus'][a]['point'], 2)] + [round(x, 2) for x in summary[d]['main_minus'][a]['ci95']]
             for a in diffs}), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'prepare':
        prepare()
    elif mode == 'predict':
        predict(sys.argv[2].split(','), int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    else:
        score(sys.argv[2].split(','))
