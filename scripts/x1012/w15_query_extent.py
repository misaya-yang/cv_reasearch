"""W15: can the query's own structure hold the right extent?  Feasibility check (selection half S, GT diagnostic).

Motivation (W14_headroom.md): the remaining ~11-point threshold gap of the fused field G = .5 M + .5 U sits in
~20 % of episodes whose natural mask is off by a factor >= 2 in area (half covered, or a 2.7-5x leak), with no
global direction; wrong objects and missed instances are minor.  Both FoRIS and our lines decide extent by
reference similarity.  Hypothesis: locate with the reference, delineate with a query-only grouping.

Query-only groupings (no reference, no GT), all token-aligned on the 64x64 grid:
  km64   the existing k-means-64 partition of the unit query features (`regions64`, already in the pipeline)
  spK    normalized spectral clustering (Ng-Jordan-Weiss) of the cached query graph L used by the U solve,
         K in {4, 8, 16, 32}
Measures, per episode (rho <= .9):
  nat, natQ   G > .5 at pixel level, and its token-majority quantization (isolates grid quantization)
  orc         per-episode GT-optimal level of G (from W14)
  tok         GT-optimal union of tokens (ceiling of any token-aligned mask)
  SO_g        GT-optimal union of segments of grouping g (exact: prefix of segments sorted by GT precision)
  rule_g      fixed rule: keep a segment iff >= 50 % of its valid pixels are in nat
Readings fixed before running:
  R1 containment, on the top-20 % gap episodes (gap threshold from the whole W14 sample):
     SO at K <= 16 below orc            -> query grouping does not hold extent beyond G's ranking; close.
     SO_8 or SO_16 above orc by >= 5     -> grouping holds the extent; worth a selection rule.
  R2 rule: rule_g - natQ (class mIoU) >= 0 in >= 7 of 9 domains and mean > +1 -> build on it (then T half);
     otherwise majority voting is not enough and only R1 decides whether to continue.
Sample: seeded <= 50 per domain from W14's 300-per-domain sample, rho <= .9 only.  No encoder.

    python scripts/x1012/w15_query_extent.py produce [workers]
    python scripts/x1012/w15_query_extent.py report
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W15')
W14 = c.task_path('W14') / 'episodes.jsonl'
KS = (4, 8, 16, 32)
GROUPS = ('km64',) + tuple(f'sp{k}' for k in KS)


def sample():
    import w14_headroom as w
    g = {json.loads(l)['index'] for l in W14.read_text().splitlines() if '"G"' in l}
    base = [b for b in w.subset([b for b in c.rows() if c.half(b) == 'S'], 300) if str(b['index']) in g]
    rng, out = np.random.default_rng(c.SEED + 15), []
    for d in c.DOMAINS:
        L = sorted((b for b in base if b['dataset'] == d), key=lambda b: str(b['index']))
        out += [L[i] for i in sorted(rng.choice(len(L), min(50, len(L)), replace=False))] if L else []
    return out


def spectral(L, seed=0):
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    from scipy.cluster.vq import kmeans2
    d = np.asarray(L.diagonal()).astype(np.float64)
    W = sparse.diags(d) - L
    s = 1 / np.sqrt(np.maximum(d, 1e-12))
    M = sparse.diags(s) @ W @ sparse.diags(s)
    vals, vecs = eigsh(M, k=max(KS), which='LA', tol=1e-6, v0=np.ones(M.shape[0]))
    vecs = vecs[:, np.argsort(-vals)]
    out = {}
    for k in KS:
        e = vecs[:, :k]
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        _, lab = kmeans2(e, k, minit='++', seed=seed + k, iter=30)
        out[f'sp{k}'] = lab
    return out


def tokens(x):
    return x.reshape(64, 16, 64, 16).sum((1, 3)).ravel().astype(np.int64)


def seg_oracle(lab, v, g):
    n, gs = np.bincount(lab, v), np.bincount(lab, g)
    keep = n > 0
    n, gs = n[keep], gs[keep]
    order = np.argsort(-(gs / n), kind='stable')
    I = np.cumsum(gs[order]); U = g.sum() + np.cumsum(n[order] - gs[order])
    return float(100 * (I / np.maximum(U, 1)).max())


def one(binding):
    import torch
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    g = gt & valid
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
    nat = (.5 * m + .5 * u > .5) & valid
    v_t, g_t, n_t = tokens(valid), tokens(g), tokens(nat)
    iu = lambda mask: [int((mask & g).sum()), int((mask | g).sum())]
    blow = lambda sel: np.repeat(np.repeat(sel.reshape(64, 64), 16, 0), 16, 1) & valid
    r = {'index': str(binding['index']), 'nat': iu(nat), 'natQ': iu(blow((n_t * 2 >= np.maximum(v_t, 1)) & (v_t > 0)))}
    r['tok'] = seg_oracle(np.arange(4096), v_t, g_t)
    labs = {'km64': fields['regions64'].astype(np.int64)}
    labs.update(spectral(c.laplacian(fields)))
    for name, lab in labs.items():
        r['SO_' + name] = seg_oracle(lab, v_t, g_t)
        n, k = np.bincount(lab, v_t, 4096), np.bincount(lab, n_t, 4096)
        r['rule_' + name] = iu(blow(((k * 2 >= n) & (n > 0))[lab]))
        r['n_' + name] = int(len(np.unique(lab)))
    return r


def produce(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sample()
    done = set()
    if (OUT / 'episodes.jsonl').exists():
        done = {json.loads(l)['index'] for l in (OUT / 'episodes.jsonl').read_text().splitlines()}
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, (OUT / 'episodes.jsonl').open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 100 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    W = {r['index']: r for r in map(json.loads, W14.read_text().splitlines()) if 'G' in r}
    gaps = np.array([w['G']['oracle'] - w['G']['natural'] for w in W.values()])
    cut = np.quantile(gaps, .8)
    R = [dict(E1[r['index']], **r, W=W[r['index']]) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    for e in R:
        e['orc'], e['top'] = e['W']['G']['oracle'], e['W']['G']['oracle'] - e['W']['G']['natural'] >= cut
    print('n', len(R), 'top-gap', sum(e['top'] for e in R), f'(gap >= {cut:.1f})')
    names = [d[:6] for d in c.DOMAINS if any(e['dataset'] == d for e in R)]
    doms = [d for d in c.DOMAINS if any(e['dataset'] == d for e in R)]
    print('\n## mean episode IoU: oracles vs natural')
    cols = ['nat', 'natQ', 'orc', 'tok'] + ['SO_' + g for g in GROUPS]
    print(f'{"subset":12s}{"n":>5s}' + ''.join(f'{k:>9s}' for k in cols))
    val = lambda e, k: iou(e[k]) if isinstance(e[k], list) else e[k]
    for nm, L in [('all', R), ('top-gap', [e for e in R if e['top']]), ('rest', [e for e in R if not e['top']])] + \
                 [(d[:12], [e for e in R if e['dataset'] == d]) for d in doms]:
        print(f'{nm:12s}{len(L):5d}' + ''.join(f'{np.mean([val(e, k) for e in L]):9.1f}' for k in cols))
    print('\n## rule (segment majority of nat) minus natQ, class mIoU per domain')
    print(f'{"":10s}' + ''.join(f'{d:>8s}' for d in names) + '    mean  >=0')
    for g in GROUPS:
        v = [score.miou([e for e in R if e['dataset'] == d], 'rule_' + g) - score.miou([e for e in R if e['dataset'] == d], 'natQ') for d in doms]
        print(f'{g:10s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}{sum(x >= 0 for x in v):5d}')
    v = [score.miou([e for e in R if e['dataset'] == d], 'natQ') - score.miou([e for e in R if e['dataset'] == d], 'nat') for d in doms]
    print(f'{"natQ-nat":10s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}')
    print('\n## mean episode-IoU change of rule vs natQ on top-gap / rest')
    for g in GROUPS:
        print(g, *(f'{nm} {np.mean([iou(e["rule_" + g]) - iou(e["natQ"]) for e in L]):+.2f}' for nm, L in
                   (('top', [e for e in R if e['top']]), ('rest', [e for e in R if not e['top']]))))
    with gzip.open(ROOT / 'evidence/research/w15_query_extent_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'nat', 'natQ', 'tok')
                                + tuple('SO_' + g for g in GROUPS) + tuple('rule_' + g for g in GROUPS)}) + '\n')


if __name__ == '__main__':
    produce(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
