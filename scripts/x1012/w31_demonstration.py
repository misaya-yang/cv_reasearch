"""W31 K0: does a holistic comparison of a hypothesised extent with the reference demonstration carry extent
information that the score field (and the token-level cycle quality J) does not?  S half only.

Motivation (docs/research/x1012_tasks/W31_demonstration_PLAN.md): our ranking is not worse than FoRIS at equal
area in all ten domains (W04); the residual ~10 points are the cut (W14), multiplicative area errors in 20% of the
episodes, and no statistic of the field predicts it (TOKEN_READOUT_CEILING, W15, W16, W26).  The only parameter-free
extent rule that beat fitted constants compares the candidate extent with the reference mask (J), through token
transfer.  K0 renders each candidate extent (outside pixels set to the encoder's null input, square bbox crop,
448^2), encodes it with the same frozen DINOv3 and compares its CLS with the rendered reference demonstration.
Candidates: the top-k of SH's fused field F with k = max(a0, 256) * 2^s, s in -2..2 step .5 (s = 0 is SH itself).
Arms: natural (SH), Dsel (CLS vs own reference), Psel (CLS vs the reference of another class, multi-class domains),
Jsel (rc.quality), D2sel (masked patch-token mean, secondary), orc (GT, diagnostic).
Reading fixed before running: Dsel - natural >= +1.0 and Dsel - Jsel >= +0.5 (ten-domain mean, before the CRF), and
on the six multi-class domains mean(Psel - natural) <= half of mean(Dsel - natural).  Pass -> K1; fail -> closed.
All selections are sealed before any query GT is read.

    python scripts/x1012/w31_demonstration.py fields [workers]
    python scripts/x1012/w31_demonstration.py encode
    python scripts/x1012/w31_demonstration.py score
"""
import gzip, hashlib, json, subprocess, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W31')
LADDER = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2)
ZERO = LADDER.index(0)
N = 1024 * 1024
SIZE = 448
MULTI = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim', 'isaid')
WEIGHTS = ROOT.parent / 'cv_data/demo4_cache/models/dinov3-vitl16-timm'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows():
    import w23_simple_pipeline as sh
    out = sh.rows_for('S')
    assert len(out) == 1285 and all(c.half(b) == 'S' for b in out)
    return out


def meta(b):
    r = b['row']
    return {'index': str(b['index']), 'dataset': b['dataset'], 'fold': r['fold'], 'cls': r['global_class_id'],
            'photo': c.photo(b)}


def jsonl(path):
    return {e['index']: e for e in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


# ---------------------------------------------------------------- fields (CPU, label-free)
def fields_one(b):
    import torch
    import w09_integration_lambda as w09, w19_simple_head as w19, w23_simple_pipeline as sh
    cap = {'U0': [], 'M': [], 'f': []}
    ou, om, oi = w09.solve, w19.mean_solve, c.rc.independent

    def point(*a, **k):
        v = ou(*a, **k); cap['U0'].append(np.asarray(v).copy()); return v

    def proto(*a, **k):
        v = om(*a, **k); cap['M'].append(np.asarray(v).copy()); return v

    def indep(*a, **k):
        v = oi(*a, **k); cap['f'].append(v[4]); return v
    w09.solve, w19.mean_solve, c.rc.independent = point, proto, indep
    try:
        pre, _, rho = sh.simple_head(b, crf=False)
    finally:
        w09.solve, w19.mean_solve, c.rc.independent = ou, om, oi
    out = dict(meta(b), rho=rho, fallback=rho > .9)
    path = OUT / 'candidates' / (c.key(b) + '.npz')
    if rho > .9:
        c.save_npz(path, m=np.stack([c.pack(pre)]))
        return dict(out, payload=str(path), payload_sha256=sha(path))
    lab = c.rc.lab
    u0 = lab.minmax(lab.up(torch.from_numpy(cap['U0'][0].reshape(64, 64).astype(np.float32)))).numpy()
    m = lab.up(torch.from_numpy(cap['M'][0])).numpy()
    F = .5 * m.astype(np.float64) + .5 * u0.astype(np.float64)
    assert np.array_equal(F > .5, pre), 'captured fields do not reproduce SH'
    f = cap['f'][0]
    a0 = int(pre.sum()); base = max(a0, 256)
    order = np.argsort(-F.ravel(), kind='stable')
    masks, J = [], []
    for s in LADDER:
        if s == 0:
            mk = pre
        else:
            k = int(min(max(round(base * 2 ** s), 64), .95 * N))
            mk = np.zeros(N, bool); mk[order[:k]] = True; mk = mk.reshape(1024, 1024)
        masks.append(mk)
        J.append(float(c.rc.quality(mk, f['c32'], f['back_f'].astype(np.float64))))
    c.save_npz(path, m=np.stack([c.pack(x) for x in masks]))
    return dict(out, a0=a0, areas=[int(x.sum()) for x in masks], J=J, payload=str(path), payload_sha256=sha(path))


def fields(workers):
    assert 1 <= workers <= 4
    (OUT / 'candidates').mkdir(parents=True, exist_ok=True)
    print(subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True).stdout, flush=True)
    path, todo = OUT / 'fields.jsonl', rows()
    done = jsonl(path); todo = [b for b in todo if str(b['index']) not in done]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool, \
            path.open('a') as stream:
        for n, e in enumerate(pool.map(fields_one, todo, chunksize=1), 1):
            stream.write(json.dumps(e) + '\n'); stream.flush()
            if n % 100 == 0 or n == len(todo):
                print(f'fields {len(done) + n}/1285, {time.perf_counter() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- encode (one MPS process, label-free)
def render(img, mask):
    """img: normalised (3,1024,1024) tensor; mask: bool (1024,1024) tensor.  Returns image and mask at SIZE^2."""
    import torch, torch.nn.functional as F
    ys, xs = torch.nonzero(mask, as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    side = max(32, int(np.ceil(1.1 * max(y1 - y0 + 1, x1 - x0 + 1))))
    top, left = int(round((y0 + y1) / 2 - side / 2)), int(round((x0 + x1) / 2 - side / 2))
    canvas = torch.zeros(4, side, side)
    src = torch.cat([img * mask, mask[None].float()])
    sy0, sx0 = max(top, 0), max(left, 0); sy1, sx1 = min(top + side, 1024), min(left + side, 1024)
    canvas[:, sy0 - top:sy1 - top, sx0 - left:sx1 - left] = src[:, sy0:sy1, sx0:sx1]
    out = F.interpolate(canvas[None], size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)[0]
    return out[:3], out[3]


def encode():
    import torch, timm
    from safetensors.torch import load_file
    from ics.official_data import decoded_rgb
    c.initialize()
    host, _ = c.rc.pc.host_cpu()
    dev = torch.device('mps')
    model = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
    print(model.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True))
    model = model.eval().to(dev)
    npre = model.num_prefix_tokens
    lab = c.rc.lab
    recs = jsonl(OUT / 'fields.jsonl')
    assert len(recs) == 1285
    binds = {str(b['index']): b for b in rows()}
    path = OUT / 'embeddings.jsonl'
    done = jsonl(path)
    (OUT / 'embeddings').mkdir(exist_ok=True)
    todo = [i for i in sorted(recs) if not recs[i]['fallback'] and i not in done]
    t0, calls = time.perf_counter(), 0
    with path.open('a') as stream, torch.inference_mode():
        for n, i in enumerate(todo, 1):
            b, e = binds[i], recs[i]
            assert sha(e['payload']) == e['payload_sha256']
            row = b['row']
            q = host._transform(decoded_rgb(c.ASSETS / row['query_path'], row.get('query_crop')))
            r = host._transform(decoded_rgb(c.ASSETS / row['reference_path'], row.get('reference_crop')))
            rm = lab.mask1024(row['reference_mask_path'])
            with np.load(e['payload']) as z:
                cands = [torch.from_numpy(c.unpack(x)) for x in z['m']]
            views = [render(r, rm)] + [render(q, m) if m.any() else None for m in cands]
            ok = [v is not None for v in views]
            x = torch.stack([v[0] for v in views if v is not None]).to(dev)
            tok = model.forward_features(x).float().cpu()
            calls += len(x)
            cls = torch.nn.functional.normalize(tok[:, 0], dim=1)
            g = SIZE // 16
            cover = torch.stack([v[1] for v in views if v is not None]).reshape(-1, g, 16, g, 16).mean((2, 4)).reshape(-1, g * g)
            patches = tok[:, npre:]
            w = (cover > .5).float(); w[w.sum(1) == 0] = cover[w.sum(1) == 0]
            pm = torch.nn.functional.normalize((patches * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6), dim=1)
            full_cls = torch.zeros(10, cls.shape[1]); full_pm = torch.zeros(10, pm.shape[1])
            idx = [k for k, o in enumerate(ok) if o]
            full_cls[idx] = cls; full_pm[idx] = pm
            p = OUT / 'embeddings' / (c.key(b) + '.npz')
            np.savez(p, cls=full_cls.numpy(), pm=full_pm.numpy(), ok=np.array(ok))
            stream.write(json.dumps({'index': i, 'payload': str(p), 'payload_sha256': sha(p), 'views': int(len(x))}) + '\n')
            stream.flush()
            if n % 50 == 0 or n == len(todo):
                print(f'encoded {len(done) + n}/{len(recs) - sum(r["fallback"] for r in recs.values())}, '
                      f'{time.perf_counter() - t0:.0f}s, {calls} views', flush=True)


# ---------------------------------------------------------------- score
def pick(values, ok):
    best = max((v, -abs(LADDER[k])) for k, v in enumerate(values) if ok[k])
    return next(k for k, v in enumerate(values) if ok[k] and (v, -abs(LADDER[k])) == best)


def select():
    """Label-free selections, sealed before any GT is opened."""
    recs, emb = jsonl(OUT / 'fields.jsonl'), jsonl(OUT / 'embeddings.jsonl')
    active = sorted(i for i, e in recs.items() if not e['fallback'])
    assert set(emb) == set(active)
    E = {}
    for i in active:
        assert sha(emb[i]['payload']) == emb[i]['payload_sha256']
        with np.load(emb[i]['payload']) as z:
            E[i] = {k: z[k] for k in ('cls', 'pm', 'ok')}
    placebo = {}
    for d in MULTI:
        L = sorted((i for i in active if recs[i]['dataset'] == d), key=str)
        for n, i in enumerate(L):
            placebo[i] = next((L[(n + k) % len(L)] for k in range(1, len(L))
                               if recs[L[(n + k) % len(L)]]['cls'] != recs[i]['cls']), None)
    sel = {}
    for i in active:
        e, z = recs[i], E[i]
        ok = list(z['ok'][1:])
        cos = lambda key, ref: [float(z[key][1 + k] @ ref) for k in range(9)]
        s = {'natural': ZERO, 'Dsel': pick(cos('cls', z['cls'][0]), ok), 'D2sel': pick(cos('pm', z['pm'][0]), ok),
             'Jsel': pick(e['J'], ok)}
        if placebo.get(i):
            s['Psel'] = pick(cos('cls', E[placebo[i]]['cls'][0]), ok)
        sel[i] = dict(s, simD=cos('cls', z['cls'][0]), simD2=cos('pm', z['pm'][0]), placebo=placebo.get(i))
    c.write(OUT / 'selections.json', sel)
    seal = {'selections_sha256': sha(OUT / 'selections.json'), 'sealed_unix_ns': time.time_ns(),
            'plan_sha256': sha(ROOT / 'docs/research/x1012_tasks/W31_demonstration_PLAN.md'),
            'code_sha256': sha(__file__), 'T_read': False, 'new_photo_holdout_read': False}
    if (OUT / 'sealed.json').exists():
        assert c.read(OUT / 'sealed.json')['selections_sha256'] == seal['selections_sha256']
    else:
        c.write(OUT / 'sealed.json', seal)
    return sel


def score_one(job):
    b, e = job
    gt, valid = c.truth(b)
    with np.load(e['payload']) as z:
        ius = [c.iu(c.unpack(x), gt, valid) for x in z['m']]
    return {'index': e['index'], 'gt': int(gt.sum()), 'ius': ius}


def score(workers=4):
    import score as sc
    from scipy.stats import spearmanr
    sel = select()
    recs = jsonl(OUT / 'fields.jsonl')
    binds = {str(b['index']): b for b in rows()}
    path = OUT / 'iu.jsonl'
    got = jsonl(path)
    todo = [(binds[i], recs[i]) for i in recs if i not in got]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool, \
            path.open('a') as stream:
        for g in pool.map(score_one, todo, chunksize=4):
            stream.write(json.dumps(g) + '\n'); got[g['index']] = g
    iou = lambda x: x[0] / max(x[1], 1)
    ARMS = ('natural', 'Dsel', 'Jsel', 'D2sel', 'Psel', 'orc')
    R = []
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    for i, e in recs.items():
        g = got[i]; r = dict(meta(binds[i]), gt=g['gt'], rho=e['rho'], fallback=e['fallback'])
        if e['fallback']:
            for a in ARMS:
                r[a] = g['ius'][0]
        else:
            s = sel[i]
            for a in ARMS[:-2]:
                r[a] = g['ius'][s[a]]
            r['Psel'] = g['ius'][s['Psel']] if 'Psel' in s else g['ius'][ZERO]
            r['orc'] = max(g['ius'], key=iou)
            r['pick'] = {a: LADDER[s[a]] for a in ARMS[:-1] if a in s}
            r['ladder_iou'] = [round(iou(x), 5) for x in g['ius']]
            r['simD'], r['J'] = s['simD'], e['J']
        R.append(r)
    same = sum(r['natural'][:2] == w23[r['index']]['SH'][:2] for r in R)
    lines = [f'implementation check: natural == W23 SH (pre-CRF) in {same}/1285 (need >= 1280)']
    rng = lambda: np.random.default_rng(c.SEED)
    doms = list(c.DOMAINS)
    lines.append('\n## class mIoU difference vs natural (pre-CRF), photo-bootstrap 95% CI\n' + f'{"":12s}' +
                 ''.join(f'{a:>24s}' for a in ARMS[1:]))
    mean = {a: [] for a in ARMS[1:]}
    for d in doms:
        L = [r for r in R if r['dataset'] == d]
        cells = []
        for a in ARMS[1:]:
            v = sc.miou(L, a) - sc.miou(L, 'natural'); mean[a].append(v)
            lo, hi = sc.paired_ci(L, a, 'natural', rng()) if a != 'orc' else (np.nan, np.nan)
            cells.append(f'{v:+7.2f} [{lo:+6.2f},{hi:+6.2f}]')
        lines.append(f'{d[:11]:12s}' + ''.join(f'{x:>24s}' for x in cells))
    lines.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):>+24.3f}' for a in ARMS[1:]))
    dj = np.mean([sc.miou([r for r in R if r['dataset'] == d], 'Dsel') - sc.miou([r for r in R if r['dataset'] == d], 'Jsel') for d in doms])
    multi = [doms.index(d) for d in MULTI]
    pm_, dm_ = np.mean([mean['Psel'][k] for k in multi]), np.mean([mean['Dsel'][k] for k in multi])
    P1, P2, P3 = np.mean(mean['Dsel']) >= 1.0, dj >= .5, dm_ > 0 and pm_ <= .5 * dm_
    lines += ['', f'P1 Dsel - natural = {np.mean(mean["Dsel"]):+.3f} (>= +1.0): {P1}',
              f'P2 Dsel - Jsel = {dj:+.3f} (>= +0.5): {P2}',
              f'P3 multi-class mean Psel - natural = {pm_:+.3f} vs half of Dsel - natural {dm_:+.3f}: {P3}',
              f'K0 premise holds: {bool(P1 and P2 and P3)}']
    A = [r for r in R if not r['fallback']]
    rho = lambda key: [spearmanr(r[key], r['ladder_iou'])[0] for r in A if np.ptp(r['ladder_iou']) > 0]
    sd, sj = rho('simD'), rho('J')
    lines += ['', f'within-episode Spearman with candidate IoU (non-fallback, non-constant): CLS median '
              f'{np.nanmedian(sd):+.3f} mean {np.nanmean(sd):+.3f}; J median {np.nanmedian(sj):+.3f} mean {np.nanmean(sj):+.3f}']
    head = sorted(A, key=lambda r: iou(r['natural']) - iou(r['orc']))[:len(A) // 5]
    for a in ('natural', 'Dsel', 'Jsel', 'D2sel', 'orc'):
        lines.append(f'top-20% headroom episodes ({len(head)}): mean IoU {a} {100 * np.mean([iou(r[a]) for r in head]):.2f}; '
                     f'all non-fallback {100 * np.mean([iou(r[a]) for r in A]):.2f}')
    for a in ('Dsel', 'Jsel', 'D2sel', 'Psel', 'orc'):
        key = a if a != 'orc' else None
        cnt = Counter(r['pick'][a] for r in A if a in r['pick']) if key else Counter(LADDER[int(np.argmax(r['ladder_iou']))] for r in A)
        lines.append(f'pick distribution {a}: ' + ', '.join(f'{s:+g}:{cnt.get(s, 0)}' for s in LADDER))
    text = '\n'.join(lines)
    print(text)
    (OUT / 'report.txt').write_text(text + '\n')
    with gzip.open(ROOT / 'evidence/research/w31_k0_S.jsonl.gz', 'wt') as z:
        for r in R:
            z.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'fields':
        fields(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif cmd == 'encode':
        encode()
    elif cmd == 'score':
        score()
