"""W54: does SH+'s Fundus lead over FoRIS hold on fundus images never used before?
Pool A = Fundus-AVSeg (100, other hospital/cameras), pool B = 200 seeded FIVES-train images (same source, other split).
FoRIS/MFU come from the frozen photo_holdout runner, SH/SH+ from the frozen simple_head (W30 wrapper).
Plan and readings: docs/research/x1012_tasks/W54_fundus_fresh_PLAN.md (fixed before this file was written).

    python scripts/x1012/w54_fundus_fresh.py prepare
    python scripts/x1012/w54_fundus_fresh.py foris      # MPS, frozen photo_holdout predict per pool
    python scripts/x1012/w54_fundus_fresh.py sh         # MPS encoder + 2 CPU head workers
    python scripts/x1012/w54_fundus_fresh.py score      # once, after both arms are sealed
"""
import gzip, hashlib, json, multiprocessing, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
sys.path.insert(0, str(ROOT / 'scripts/x1012s'))
import common as c

ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1013_W54'
PY = Path.home() / 'miniconda3/envs/aidemo/bin/python'
A_HOLDOUT_ENCODER = ASSETS / 'a/x1013_photo_holdout/frozen/encoder.py'   # same frozen TimmDINOv3 text as photo_holdout.predict writes
AVSEG = ASSETS / 'datasets/ics/Fundus-AVSeg'
RAR = ASSETS / 'setup/asset_downloads/Fundus_autonomous_20261010/FIVES_v1.rar'
FIVES = 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/train/'
SEED = {'A': 2026101601, 'B': 2026101602}
POOLS = ('A', 'B')
ARMS = ('SH', 'SH+', 'FoRIS', 'MFU', 'U')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(v, indent=1)); t.replace(p)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


# ---------------------------------------------------------------- prepare
def prepare():
    from PIL import Image
    import photo_holdout as ph
    task, rt = ph.runtime()
    od = sys.modules['ics.official_data']          # the frozen runtime's loader (same hashing as load_inputs)
    pools = {}
    names = sorted(p.name for p in (AVSEG / 'images').glob('*.png'))
    assert len(names) == 100
    pools['A'] = [(f'avseg:{n}', AVSEG / 'images' / n, AVSEG / 'annotation' / n) for n in names]
    listing = subprocess.run(['bsdtar', '-tf', str(RAR)], capture_output=True, text=True, check=True).stdout.splitlines()
    have = set(listing)
    train = sorted(Path(m).name for m in listing if m.startswith(FIVES + 'Original/') and m.endswith('.png')
                   and FIVES + 'Ground truth/' + Path(m).name in have)
    assert len(train) == 600, len(train)
    pick = [train[i] for i in sorted(np.random.default_rng(SEED['B']).choice(600, 200, replace=False))]
    dest = OUT / 'fives_train'
    members = [FIVES + d + '/' + n for n in pick for d in ('Original', 'Ground truth')]
    if not all((dest / m).exists() for m in members):
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['bsdtar', '-xf', str(RAR), '-C', str(dest)] + members, check=True)
    pools['B'] = [(f'fives_train:{n}', dest / FIVES / 'Original' / n, dest / FIVES / 'Ground truth' / n) for n in pick]
    for pool, items in pools.items():
        info = {}
        for photo, img, ann in items:
            with Image.open(ann) as im:
                mask = (np.asarray(im.convert('RGB')).max(2) > 0).astype(np.uint8)   # any vessel label
            h = od.array_hash(mask)
            path = OUT / 'masks' / (h + '.png')
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask * 255).save(path)
            rgb = np.asarray(od.decoded_rgb(img, None))
            assert rgb.shape[:2] == mask.shape, (photo, rgb.shape, mask.shape)
            info[photo] = dict(path=str(img.relative_to(ASSETS)), ann=str(ann.relative_to(ASSETS)), rgb_hash=od.array_hash(rgb),
                               size=list(mask.shape), mask_path=str(path), mask_hash=h, fraction=float(mask.mean()))
        perm = derangement(len(items), np.random.default_rng(SEED[pool]))
        rows = []
        for i, (q, _, _) in enumerate(items):
            r = items[perm[i]][0]
            Q, R = info[q], info[r]
            idx = ('avseg_' if pool == 'A' else 'fivestrain_') + Path(Q['path']).stem
            row = dict(dataset='fundus', fold=-1, official_index=i, episode_id=f'W54/{pool}/{idx}', shots=1,
                       loader_class_id=0, global_class_id=0, query_gt_role='scoring after prediction seal', source_files={},
                       official_selection=None, ignore_policy='no ignore region',
                       gt_alignment='torch nearest-resize raw GT to each prediction frame; >0.5')
            for role, x, photo in (('reference', R, r), ('query', Q, q)):
                row.update({role + '_path': x['path'], role + '_crop': None, role + '_photo_id': photo, role + '_object_id': None,
                            role + '_rgb_hash': x['rgb_hash'], role + '_size_hw': x['size'], role + '_annotation_path': x['ann'],
                            role + '_mask_path': x['mask_path'], role + '_mask_hash': x['mask_hash'], role + '_mask_size_hw': x['size']})
            rows.append(dict(index=idx, dataset='fundus', row=row, canonical_reference_photo=r, canonical_query_photo=q))
        for b in rows:                                   # the frozen loader accepts every row
            rt.load_inputs(b['row'], ASSETS)
        write(OUT / f'manifest_{pool}.json', rows)
        frac = [v['fraction'] for v in info.values()]
        print(pool, len(rows), 'vessel fraction median', round(float(np.median(frac)), 4), flush=True)


# ---------------------------------------------------------------- FoRIS / MFU (frozen holdout runner)
def foris():
    """photo_holdout.predict with one change in the encoding loop: its 4-entry reference cache can evict the query's
    features between the lookup and the use when an image is both a reference and a query (every image here), which
    raised KeyError at A 84/100.  Heads (photo_holdout.save_prediction -> five_masks) and record format are unchanged."""
    import torch
    import photo_holdout as ph
    task = rt = encoder = None                      # the runtime may be built only once per process
    for pool in POOLS:
        out, manifest = OUT / f'foris_{pool}', OUT / f'manifest_{pool}.json'
        if (out / 'sealed.json').exists():
            continue
        rows = ph.freeze(out, manifest)
        if rt is None:
            task, rt = ph.runtime()
            encoder = ph.module('w54_foris_encoder', A_HOLDOUT_ENCODER).TimmDINOv3(str(ph.WEIGHTS)).to('mps').eval().requires_grad_(False)
        cache, started = OrderedDict(), time.monotonic()
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'), initializer=ph.worker_init) as pool_, \
                torch.inference_mode():
            pending = set()
            for n, original in enumerate(sorted(rows, key=lambda b: (b['row']['reference_rgb_hash'], str(b['index'])))):
                key = str(original['index'])
                if (out / 'records' / (key + '.json')).exists():
                    old = read(out / 'records' / (key + '.json'))
                    assert old['manifest_sha256'] == sha(manifest) and sha(old['prediction_path']) == old['prediction_sha256']
                    continue
                began = time.monotonic()
                b = json.loads(json.dumps(original))
                reference, _, query = rt.load_inputs(b['row'], ph.ASSETS)
                inputs = {'reference': rt.host._transform(reference), 'query': rt.host._transform(query)}
                hashes = {role: task.array_hash(x.numpy()) for role, x in inputs.items()}
                missing = [role for role in inputs if hashes[role] not in cache]
                if missing:
                    maps = encoder.get_intermediate_layers(torch.stack([inputs[r] for r in missing]).to('mps'), n=1, reshape=True)[0]
                    torch.mps.synchronize()
                    for r, raw in zip(missing, maps.cpu().permute(0, 2, 3, 1).reshape(len(missing), 4096, 1024).contiguous().unbind(0)):
                        assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
                        cache[hashes[r]] = raw
                features, b['fit'] = {r: cache[hashes[r]] for r in inputs}, {}     # taken before any eviction
                for h in hashes.values():
                    cache.move_to_end(h)
                while len(cache) > 8:
                    cache.popitem(last=False)
                for role, h in hashes.items():
                    entry = out / 'inputs' / (h + '.json')
                    if not entry.exists():
                        write(entry, dict(file='in_memory.npz', input_tensor_hash=h,
                                          storage='FP32 tensor supplied directly; no feature payload written',
                                          raw_tensor_sha256=task.array_hash(features[role].numpy())))
                    b['fit']['reference_raw' if role == 'reference' else 'query_raw_identity'] = {
                        'payload_path': str(entry.parent / 'in_memory.npz'), 'entry_path': str(entry)}
                pending.add(pool_.submit(ph.save_prediction, out, b, features,
                                         dict(manifest_sha256=sha(manifest), new_encoded_images=len(missing),
                                              input_and_encoder_seconds=time.monotonic() - began)))
                while len(pending) >= 4:
                    fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in fin:
                        f.result()
                if n % 20 == 0:
                    print(json.dumps(dict(arm='foris', pool=pool, n=n, of=len(rows), seconds=round(time.monotonic() - started))), flush=True)
            for f in pending:
                f.result()
        records = [read(out / 'records' / (str(b['index']) + '.json')) for b in rows]
        assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in records)
        write(out / 'sealed.json', dict(state='ALL_FIXED_NATURAL_OUTPUTS_COMPLETE', n=len(rows), manifest_sha256=sha(manifest),
                                        evaluation_sha256=sha(out / 'evaluation.json'), arms=ph.ARMS, records=records,
                                        completed_at_utc=datetime.now().isoformat()))
        print(pool, 'foris sealed', len(records), flush=True)


# ---------------------------------------------------------------- SH / SH+ (frozen simple_head, W30 wrapper)
def head(job):
    import w23_simple_pipeline as sh
    pool, b, feats, meta = job
    b = dict(b, fit={'reference_raw': {'role': 'reference'}, 'query_raw_identity': {'role': 'query'}})
    original = c.rc.raw
    c.rc.raw = lambda metadata, counters: feats[metadata['role']]
    began = time.monotonic()
    try:
        pre, final, rho = sh.simple_head(b, crf=True)
    finally:
        c.rc.raw = original
    path = OUT / f'sh_{pool}' / 'predictions' / (b['index'] + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, SH=c.pack(pre), **{'SH+': c.pack(final)}); tmp.replace(path)
    rec = dict(index=b['index'], rho=rho, prediction_path=str(path), prediction_sha256=sha(path),
               head_seconds=time.monotonic() - began, query_GT_reads=0, **meta)
    write(OUT / f'sh_{pool}' / 'records' / (b['index'] + '.json'), rec)
    return rec


def worker_init():
    c.initialize()


def run_sh():
    import w30_confirm as w30
    enc = w30.Encoder()
    for pool in POOLS:
        out = OUT / f'sh_{pool}'
        if (out / 'sealed.json').exists():
            continue
        rows = read(OUT / f'manifest_{pool}.json')
        started = time.monotonic()
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool_:
            pending = set()
            for n, b in enumerate(rows):
                if (out / 'records' / (b['index'] + '.json')).exists():
                    continue
                t0 = time.monotonic()
                x, hashes = enc.inputs(b)
                feats, new = enc.features(x, hashes)
                pending.add(pool_.submit(head, (pool, b, feats, dict(new_encoded_images=new, encode_seconds=time.monotonic() - t0))))
                while len(pending) >= 4:
                    fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in fin:
                        f.result()
                if n % 20 == 0:
                    print(json.dumps(dict(pool=pool, n=n, of=len(rows), seconds=round(time.monotonic() - started))), flush=True)
            for f in pending:
                f.result()
        recs = [read(out / 'records' / (b['index'] + '.json')) for b in rows]
        assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in recs)
        write(out / 'sealed.json', dict(n=len(recs), sealed_at=datetime.now().isoformat(), manifest_sha256=sha(OUT / f'manifest_{pool}.json'),
                                        predictions={r['index']: r['prediction_sha256'] for r in recs}))
        print(pool, 'sealed', len(recs), flush=True)


# ---------------------------------------------------------------- score (once)
def truth(row):
    import torch
    from PIL import Image
    with Image.open(row['query_mask_path']) as im:
        x = torch.from_numpy((np.asarray(im.convert('L')) > 0).astype(np.float32))
    return torch.nn.functional.interpolate(x[None, None], (1024, 1024), mode='nearest')[0, 0].numpy() > .5


def score():
    for pool in POOLS:
        assert (OUT / f'foris_{pool}/sealed.json').exists() and (OUT / f'sh_{pool}/sealed.json').exists()
    write(OUT / 'score_started.json', dict(time=datetime.now().isoformat(),
                                           seals={p: [sha(OUT / f'foris_{p}/sealed.json'), sha(OUT / f'sh_{p}/sealed.json')] for p in POOLS}))
    R, L = [], []
    for pool in POOLS:
        rows = read(OUT / f'manifest_{pool}.json')
        fs = {r['index']: r for r in read(OUT / f'foris_{pool}/sealed.json')['records']}
        for b in rows:
            gt = truth(b['row']); valid = np.ones_like(gt)
            srec = read(OUT / f'sh_{pool}/records' / (b['index'] + '.json'))
            frec = fs[b['index']]
            assert sha(srec['prediction_path']) == srec['prediction_sha256'] and sha(frec['prediction_path']) == frec['prediction_sha256']
            out = dict(pool=pool, index=b['index'], query=b['canonical_query_photo'], reference=b['canonical_reference_photo'],
                       rho=srec['rho'], foris_rho=frec['reference_rho'], gt=int(gt.sum()))
            with np.load(srec['prediction_path']) as z:
                for a in ('SH', 'SH+'):
                    out[a] = c.iu(c.unpack(z[a]), gt, valid)
            with np.load(frec['prediction_path']) as z:
                for a, k in (('FoRIS', 'FoRIS'), ('MFU', 'main'), ('U', 'U')):
                    out[a] = c.iu(c.unpack(z[k]), gt, valid)
            R.append(out)
    with gzip.open(OUT / 'scores.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in R)
    with gzip.open(ROOT / 'evidence/research/w54_fundus_fresh.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in R)
    pooled = lambda S, a, w: 100 * float((w * S[a][:, 0]).sum() / max((w * S[a][:, 1]).sum(), 1))
    ci = lambda v: [round(float(x), 2) for x in np.percentile(v, [2.5, 97.5])]
    for pool in POOLS:
        P = [r for r in R if r['pool'] == pool]
        S = {a: np.array([r[a][:2] for r in P], np.float64) for a in ARMS}
        ones = np.ones(len(P))
        point = {a: pooled(S, a, ones) for a in ARMS}
        mean_ep = {a: 100 * float(np.mean(S[a][:, 0] / np.maximum(S[a][:, 1], 1))) for a in ARMS}
        rng, rng2 = np.random.default_rng(2026101303), np.random.default_rng(2026101304)
        imgs = {h: i for i, h in enumerate(sorted({r['query'] for r in P} | {r['reference'] for r in P}))}
        qi, ri = np.array([imgs[r['query']] for r in P]), np.array([imgs[r['reference']] for r in P])
        b1 = {a: [] for a in ARMS}; b2 = {a: [] for a in ARMS}
        for _ in range(2000):
            w = np.bincount(rng.integers(0, len(P), len(P)), minlength=len(P)).astype(np.float64)   # one query photo per episode
            g = np.bincount(rng2.integers(0, len(imgs), len(imgs)), minlength=len(imgs))
            w2 = (g[qi] * g[ri]).astype(np.float64)
            for a in ARMS:
                b1[a].append(pooled(S, a, w)); b2[a].append(pooled(S, a, w2))
        b1 = {a: np.array(v) for a, v in b1.items()}; b2 = {a: np.array(v) for a, v in b2.items()}
        fb = sum(r['rho'] > .9 for r in P)
        same = sum(r['SH+'][:2] == r['MFU'][:2] for r in P)
        L += [f'## pool {pool}: n {len(P)}; SH fallback (rho > .9) {fb}/{len(P)}; SH+ identical to MFU {same}/{len(P)}',
              '  pooled IoU ' + '  '.join(f'{a} {point[a]:.2f} {ci(b1[a])}' for a in ARMS),
              '  episode-mean IoU ' + '  '.join(f'{a} {mean_ep[a]:.2f}' for a in ARMS)]
        for a in ('FoRIS', 'MFU', 'SH'):
            d1, d2 = b1['SH+'] - b1[a], b2['SH+'] - b2[a]
            L.append(f'  SH+ - {a}: {point["SH+"] - point[a]:+.2f}  photo {ci(d1)}  image-2way {ci(d2)}')
        wins = sum(r['SH+'][0] / max(r['SH+'][1], 1) > r['FoRIS'][0] / max(r['FoRIS'][1], 1) for r in P)
        L.append(f'  episodes where SH+ IoU > FoRIS IoU: {wins}/{len(P)}')
        L.append('')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    {'prepare': prepare, 'foris': foris, 'sh': run_sh, 'score': score}[sys.argv[1]]()
