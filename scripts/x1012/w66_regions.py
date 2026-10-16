"""W66: does the attached mid-level region belong to the target -- is the information in the reference or the query?

Plan: docs/research/x1012_tasks/W66_region_attribution_PLAN.md (sections 1-3 fixed before running).
Sample: the 1,970 W65 G-field episodes (S half, rho <= .9) with W65's photo split (V viewing / X test).
Field G = .5 up(T1 mean) + .5 minmax(up(VP_UN)) (MFU pre-CRF field, W14 `one()`), pooled to the 64 grid.

    python scripts/x1012/w66_regions.py regions     # GT-free: region masks + token arrays + region quantities
    python scripts/x1012/w66_regions.py seal        # hash the GT-free outputs before any label is read
    python scripts/x1012/w66_regions.py labels      # GT token coverage and region labels (scoring only)
    python scripts/x1012/w66_regions.py score       # AUC tables
"""
import gzip, hashlib, json, os, sys, time
from pathlib import Path
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.OUT / 'x1013_W66'
EPISODES = c.OUT / 'x1013_W65/boards/episodes.jsonl'
HI, LO = .8, .15
QUANT = ('B0', 'B1', 'R1', 'R2', 'R3', 'R4', 'Q1', 'Q2', 'Q3')
SIGN = {'B0': 1, 'B1': -1, 'R1': 1, 'R2': -1, 'R3': 1, 'R4': 1, 'Q1': 1, 'Q2': 1, 'Q3': 1}
COMBO = ('R1', 'R4', 'Q1', 'Q2')


def episodes():
    E = [json.loads(l) for l in EPISODES.read_text().splitlines()]
    want = {e['index'] for e in E}
    B = {str(b['index']): b for b in c.rows() if c.half(b) == 'S' and str(b['index']) in want}
    assert len(B) == len(E) == 1970, (len(B), len(E))
    return E, B


def field_G(b, fields):
    import torch
    with np.load(c.task_path('T1') / 'fields' / (c.key(b) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
    return .5 * m + .5 * u


def pool(x):
    return x.reshape(64, 16, 64, 16).mean((1, 3))


def lab_tokens(rgb):
    """sRGB [0,1] -> CIE Lab (D65), averaged per 16x16 token."""
    a = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    xyz = a @ M.T / np.array([.95047, 1., 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([pool(L), pool(A), pool(B)], -1).reshape(4096, 3)


def masks(G64):
    from scipy import ndimage
    N = G64 > .5
    K = G64 >= HI
    if K.sum() < 4 and N.any():
        v = np.sort(G64[N])[::-1]
        K = N & (G64 >= v[max(int(np.ceil(.25 * len(v))), 1) - 1])
    IN = N & ~K
    band = (G64 >= LO) & ~N
    lab_, _ = ndimage.label(band | N, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    keep = np.unique(lab_[N])
    OUTm = band & np.isin(lab_, keep[keep > 0])
    F = G64 < LO
    if F.sum() < 4:
        F = G64 <= np.quantile(G64, .25)
    return K, N, IN, OUTm, F


def region_quantities(Z, K, F, tok):
    z, k, f = Z.ravel(), K.ravel(), F.ravel()
    mean = lambda a, m: float(a[m].mean())
    q = {'n': int(z.sum()), 'nK': int(k.sum())}
    q['B0'] = mean(tok['G64'], z)
    q['B1'] = float(np.log(z.sum() / max(k.sum(), 1)))
    q['R1'] = mean(tok['logdf'], z) - mean(tok['logdf'], k)
    q['R2'] = mean(tok['logdb'], z) - mean(tok['logdb'], k)
    q['R3'] = -(mean(tok['loge'], z) - mean(tok['loge'], k))
    q['R4'] = float(np.log(tok['backf'][z].mean() + 1e-12) - np.log(tok['backf'][k].mean() + 1e-12))
    q['Q1'] = mean(tok['knnK'], z) - mean(tok['knnF'], z)
    q['Q2'] = mean(tok['clusK'], z)
    lab_ = tok['lab']
    mz, mk, mf = lab_[z].mean(0), lab_[k].mean(0), lab_[f].mean(0)
    q['Q3'] = float(np.linalg.norm(mz - mf) - np.linalg.norm(mz - mk))
    return q


def one_regions(e, b):
    import w42_ordered_cut as w42
    fields = c.load_fields(b)
    G64 = pool(field_G(b, fields))
    K, N, IN, OUTm, F = masks(G64)
    cc = np.clip(fields['c32'].astype(np.float64), 1e-6, 1 - 1e-6)
    loge = fields['log_e'].astype(np.float64)
    nb = fields['query_neighbors20'].astype(np.int64)
    k, f = K.ravel(), F.ravel()
    clus = fields['regions64'].astype(np.int64)
    tok = {'G64': G64.ravel(), 'logdf': np.log(cc) + loge, 'logdb': np.log(1 - cc) + loge, 'loge': loge,
           'backf': fields['back_f'].astype(np.float64), 'knnK': k[nb].mean(1), 'knnF': f[nb].mean(1),
           'clusK': np.isin(clus, np.unique(clus[k])).astype(np.float64),
           'lab': lab_tokens(w42.rgb(b['row']['query_path'], b['row'].get('query_crop')))}
    rec = {'index': e['index'], 'dataset': e['dataset'], 'photo': e['photo'], 'split': e['split'],
           'nN': int(N.sum()), 'nK': int(K.sum()), 'nF': int(F.sum())}
    for name, Z in (('IN', IN), ('OUT', OUTm)):
        rec[name] = region_quantities(Z, K, F, tok) if Z.sum() >= 4 and K.any() else None
    np.savez_compressed(OUT / 'regions' / (e['index'] + '.npz'), G64=G64.astype(np.float32), K=K, N=N, IN=IN,
                        OUT=OUTm, F=F, **{t: tok[t].astype(np.float32) for t in tok if t != 'G64'})
    return rec


def regions():
    c.initialize()
    import torch
    torch.set_num_threads(2)
    (OUT / 'regions').mkdir(parents=True, exist_ok=True)
    E, B = episodes()
    path = OUT / 'quantities.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    t0 = time.time()
    with path.open('a') as fh:
        for n, e in enumerate(E, 1):
            if e['index'] in done:
                continue
            fh.write(json.dumps(one_regions(e, B[e['index']])) + '\n'); fh.flush()
            if n % 100 == 0:
                print(n, len(E), f'{time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)


def seal():
    h = hashlib.sha256((OUT / 'quantities.jsonl').read_bytes()).hexdigest()
    plan = ROOT / 'docs/research/x1012_tasks/W66_region_attribution_PLAN.md'
    s = {'quantities_sha256': h, 'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
         'sealed': time.ctime(), 'n': len((OUT / 'quantities.jsonl').read_text().splitlines())}
    (OUT / 'seal.json').write_text(json.dumps(s, indent=1))
    print(s)


def labels():
    c.initialize()
    assert (OUT / 'seal.json').exists(), 'seal the GT-free outputs first'
    E, B = episodes()
    (OUT / 'labels').mkdir(exist_ok=True)
    path = OUT / 'labels.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open('a') as fh:
        for n, e in enumerate(E, 1):
            if e['index'] in done:
                continue
            gt, valid = c.truth(B[e['index']])
            vs = pool(valid.astype(np.float64))
            cov = np.where(vs > 0, pool((gt & valid).astype(np.float64)) / np.maximum(vs, 1e-12), 0)
            ok = vs >= .5
            with np.load(OUT / 'regions' / (e['index'] + '.npz')) as z:
                rec = {'index': e['index']}
                for name in ('IN', 'OUT'):
                    Z = z[name] & ok
                    rec[name] = {'n_lab': int(Z.sum()), 'cov': float(cov[Z].mean()) if Z.any() else None}
            np.savez_compressed(OUT / 'labels' / (e['index'] + '.npz'), cov=cov.astype(np.float32), ok=ok)
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, flush=True)
    print('done', flush=True)


def auc(x, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(x, y, photos, reps=1000, seed=2026101603):
    rng = np.random.default_rng(seed)
    P = np.unique(photos)
    idx = {p: np.flatnonzero(photos == p) for p in P}
    out = []
    for _ in range(reps):
        s = np.concatenate([idx[p] for p in rng.choice(P, len(P))])
        out.append(auc(x[s], y[s]))
    return np.nanpercentile(out, [2.5, 97.5])


def token_auc(e, name, qname):
    """Within-region token AUC for one region (secondary); None if a class has < 3 tokens."""
    with np.load(OUT / 'regions' / (e['index'] + '.npz')) as z, np.load(OUT / 'labels' / (e['index'] + '.npz')) as L:
        Z = (z[name] & L['ok']).ravel()
        y = L['cov'].ravel()[Z] >= .5
        if y.sum() < 3 or (~y).sum() < 3:
            return None
        tok = {'B0': z['G64'].ravel(), 'R1': z['logdf'], 'R2': z['logdb'], 'R3': -z['loge'], 'R4': z['backf'],
               'Q1': z['knnK'] - z['knnF'], 'Q2': z['clusK']}
        if qname not in tok:
            return None
        return auc(SIGN[qname] * tok[qname][Z], y)


def score():
    from scipy.stats import rankdata
    Q = {r['index']: r for r in map(json.loads, (OUT / 'quantities.jsonl').read_text().splitlines())}
    L = {r['index']: r for r in map(json.loads, (OUT / 'labels.jsonl').read_text().splitlines())}
    lines = []
    p = lambda s='': lines.append(s)
    skipped = sum(Q[k]['IN'] is None and Q[k]['OUT'] is None for k in Q)
    p(f'episodes {len(Q)}; with neither region {skipped}; core fallback (nK<4 at .8) recorded per episode')
    result = {}
    for name in ('OUT', 'IN'):
        R = [(Q[k], L[k]) for k in Q if Q[k][name] is not None and L[k][name]['n_lab'] >= 4]
        q = {m: np.array([a[name][m] for a, _ in R]) for m in QUANT}
        q['C'] = np.mean([rankdata(SIGN[m] * q[m]) for m in COMBO], 0)
        y = np.array([b[name]['cov'] >= .5 for _, b in R])
        dom = np.array([a['dataset'] for a, _ in R])
        ph = np.array([a['photo'] for a, _ in R])
        sp = np.array([a['split'] for a, _ in R])
        suim = dom == 'suim'
        p(f'\n## {name}: regions {len(R)} (member {y.sum()}), X half {np.sum(sp == "X")} (member {y[sp == "X"].sum()})')
        doms = sorted(set(dom))
        valid_d = [d for d in doms if min(y[(sp == 'X') & (dom == d)].sum(), (~y[(sp == 'X') & (dom == d)]).sum()) >= 5]
        p('valid domains (X, >=5 per class): ' + ', '.join(valid_d))
        p(f'{"q":4s} {"AUC X [95% CI]":>24s} {"AUC all":>8s} {"X noSUIM":>9s} {"dom+ (X)":>9s} {"tokenAUC":>9s}  per-domain X')
        result[name] = {}
        for m in QUANT + ('C',):
            x = (SIGN.get(m, 1) * q[m]).astype(np.float64)
            X = sp == 'X'
            a = auc(x[X], y[X]); ci = boot(x[X], y[X], ph[X])
            aa = auc(x, y); an = auc(x[X & ~suim], y[X & ~suim])
            per = {d: auc(x[X & (dom == d)], y[X & (dom == d)]) for d in valid_d}
            plus = sum(v > .5 for v in per.values())
            tk = [token_auc(a_, name, m) for a_, _ in R] if m in ('B0', 'R1', 'R2', 'R3', 'R4', 'Q1', 'Q2') else []
            tk = [v for v in tk if v is not None]
            tmean = float(np.mean(tk)) if tk else float('nan')
            p(f'{m:4s} {a:8.3f} [{ci[0]:.3f},{ci[1]:.3f}] {aa:8.3f} {an:9.3f} {plus:4d}/{len(per):<4d} {tmean:9.3f}  '
              + ' '.join(f'{d[:5]}={v:.2f}' for d, v in per.items()))
            result[name][m] = {'auc_X': a, 'ci_X': list(map(float, ci)), 'auc_all': aa, 'auc_X_noSUIM': an,
                               'per_domain_X': per, 'token_auc_mean': tmean, 'n_token_regions': len(tk)}
        b0 = result[name]['B0']['auc_X']
        passed = [m for m in QUANT[2:] + ('C',) if result[name][m]['auc_X'] >= .75 and result[name][m]['auc_X'] >= b0 + .05
                  and sum(v > .5 for v in result[name][m]['per_domain_X'].values()) > len(valid_d) / 2]
        p(f'gate (AUC_X >= .75, >= B0 + .05, majority of valid domains): {passed or "none"}')
        result[name]['passed'] = passed
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'results.json').write_text(json.dumps(result, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    {'regions': regions, 'seal': seal, 'labels': labels, 'score': score}[sys.argv[1]]()
