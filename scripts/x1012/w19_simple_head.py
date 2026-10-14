"""W19: can FoRIS be replaced by a simpler, faster head on the same frozen DINOv3 features and inputs?

Motivation (W13): MFU = rho>.9 ? T : CRF([.5 M + .5 U > .5]).  M0U+ (U without FoRIS vote/prior) is within
0.21 of MFU, so the only remaining FoRIS dependence is the FoRIS score inside M = mean_control(q, r, cov, score):
guide-ranked, lambda-16 solve on the query kNN graph.  FoRIS's score costs four agglomerative clusterings
(two over all 4096 query tokens), a 4096^2 vote and a semantic correction.  Replace only that score by a
cheap one, keep everything else, and compare accuracy here and runtime in `time` (same process, same threads).
Cheap scores (fixed before running; no tuning):
  g    guide = q . unit(mean of pure reference FG tokens)            (already computed inside mean_control)
  p    FoRIS stage-2 contrast with ONE prototype and no clustering: q.mu_fg - .55 q.mu_bg_orth,
       mu_bg = mean of the 20 % reference BG tokens most similar to mu_fg, orthogonalised to mu_fg
  pr   p + its mean over the existing k-means-64 query regions (cheap stand-in for the seed-cluster prior)
Arms (rho > .9 keeps T, as MFU):  X_k = [.5 M_k + .5 U0 > .5],  X_k+ = official CRF(X_k);  U0 = FoRIS-free U.
Readings: an X_k+ within 0.3 of MFU in mean and not below FoRIS significantly in any domain, at a lower
measured head time than FoRIS, replaces FoRIS.  Otherwise report which FoRIS part carries the difference.
Sample: the W13 seeded subset (S half, <= 150 per domain).

    python scripts/x1012/w19_simple_head.py check          # exact reproduction of cached M with the FoRIS score
    python scripts/x1012/w19_simple_head.py produce [workers]
    python scripts/x1012/w19_simple_head.py report
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

OUT = c.task_path('W19')
SCORES = ('g', 'p', 'pr')
ARMS = tuple(x for k in SCORES for x in ('X' + k, 'X' + k + '+'))


def mean_graph(q):
    """The kNN graph of mean_control, built once (identical code)."""
    import torch
    from scipy import sparse
    sim = q @ q.T; sim.fill_diagonal_(-2); values, idx = sim.topk(20, dim=1); del sim
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), idx.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T); w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel(); w = w / max(float(degree.mean()), 1e-8); degree = np.asarray(w.sum(1)).ravel()
    return sparse.diags(degree) - w


def mean_solve(score, guide, L):
    """mean_control's solve with a given graph (identical arithmetic)."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods import rcg
    s = rcg.minmax(score).ravel(); y = (s + .25 * (rcg.rank(guide) - rcg.rank(s))).astype(np.float64)
    a = .1 + np.abs(2 * s - 1); a = (a / a.mean()).astype(np.float64)
    H = sparse.diags(a) + 16 * L; rhs = a * y
    z, status = cg(H, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
    if status:
        raise RuntimeError('CG failure')
    return z.reshape(64, 64).astype(np.float32)


def heads(binding, fields):
    """Cheap scores and the shared pieces, from FoRIS Part-1 features (as mean_field uses)."""
    import torch
    import torch.nn.functional as F
    import repo_methods as repo
    q, r, m, _ = repo.foris_features(binding)
    q = F.normalize(q.half().cpu().float(), dim=1); r = F.normalize(r.half().cpu().float(), dim=1)
    cov = c.rc.lab.coverage64(m).numpy().ravel()
    fi = np.flatnonzero(cov >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov == cov.max())
    guide = (q @ F.normalize(r[fi].mean(0), dim=0)).numpy()
    fg = cov >= .5 if (cov >= .5).any() else cov == cov.max()
    mu_fg = F.normalize(r[torch.from_numpy(fg)].mean(0), dim=0)
    bg = r[torch.from_numpy(~fg)]
    if len(bg):
        sim_bg = bg @ mu_fg
        hard = bg[sim_bg.topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0)
        mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg
        mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p = (q @ mu_fg - .55 * (q @ mu_bg)).numpy()
    lab = fields['regions64'].astype(np.int64)
    region = (np.bincount(lab, p, 64) / np.maximum(np.bincount(lab, None, 64), 1))[lab]
    return q, r, cov, guide, {'g': guide, 'p': p, 'pr': p + region}


def check(n=6):
    import repo_methods as repo
    rows = [b for b in c.rows() if c.half(b) == 'S'][:n]
    c.initialize()
    for b in rows:
        fields = c.load_fields(b)
        q, r, cov, guide, _ = heads(b, fields)
        z = mean_solve(repo.foris_score(b), guide, mean_graph(q))
        with np.load(c.task_path('T1') / 'fields' / (c.key(b) + '.npz'), allow_pickle=False) as f:
            ref = f['mean']
        print(b['dataset'], 'max |M - cached M| =', float(np.abs(z - ref).max()))


def one(binding, scores=SCORES):
    import torch
    import produce
    from w09_integration_lambda import solve
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index'])}
    if produce.fallback(fields):
        with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
            t_mask = c.unpack(z['T'])
        for a in ARMS if scores == SCORES else ('Xp', 'Xp+'):
            r[a] = c.iu(t_mask, gt, valid)
        return r
    mm = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32)))).numpy().astype(np.float64)
    u0 = mm(solve(fields['c_mod'], fields['log_e'], c.laplacian(fields), 16).reshape(64, 64))
    q, _, _, guide, cheap = heads(binding, fields)
    L = mean_graph(q)
    masks = {}
    for k in scores:
        mk = c.up(mean_solve(cheap[k].reshape(64, 64), guide, L)).astype(np.float64)
        masks['X' + k] = .5 * mk + .5 * u0 > .5
    with c.rc.pc.inference_only(binding):
        for k in scores:
            masks['X' + k + '+'] = produce.crf(binding, masks['X' + k])
    for a in masks:
        r[a] = c.iu(masks[a], gt, valid)
    return r


def one_t(binding):
    return one(binding, ('p',))


def produce_all(workers):
    import w13_plugin as w13
    OUT.mkdir(parents=True, exist_ok=True)
    rows = w13.subset([b for b in c.rows() if c.half(b) == 'S'], 150)
    done = set()
    if (OUT / 'episodes.jsonl').exists():
        done = {json.loads(l)['index'] for l in (OUT / 'episodes.jsonl').read_text().splitlines()}
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, (OUT / 'episodes.jsonl').open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def confirm_t(workers):
    """One-shot T-half confirmation of the pre-fixed choice Xp+ (simplest of the S-qualified heads).
    Reading fixed before running: Xp+ - MFU within 0.3 in mean and no domain significantly below FoRIS."""
    import w13_plugin as w13
    import score
    out = c.task_path('W19T')
    out.mkdir(parents=True, exist_ok=True)
    rows = w13.subset([b for b in c.rows() if c.half(b) == 'T'], 150)
    path = out / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('T sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for n, r in enumerate(pool.map(one_t, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, len(todo), flush=True)
    with gzip.open(ROOT / 'evidence/research/w11_t_ablation.jsonl.gz', 'rt') as f:
        W11 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(W11[r['index']], **r) for r in map(json.loads, path.read_text().splitlines())]
    rng = np.random.default_rng(c.SEED)
    print('n', len(R))
    for ref in ('foris', 'MFU'):
        for a in ('Xp+', 'Xp') if ref == 'foris' else ('Xp+',):
            v, cells, sp, sn = [], [], 0, 0
            for d in c.DOMAINS:
                L = [r for r in R if r['dataset'] == d]
                x = score.miou(L, a) - score.miou(L, ref); lo, hi = score.paired_ci(L, a, ref, rng)
                sp += lo > 0; sn += hi < 0; v.append(x); cells.append(f'{d[:6]} {x:+.2f}[{lo:+.1f},{hi:+.1f}]')
            print(f'{a} - {ref}: mean {np.mean(v):+.2f}  >=0 {sum(x >= 0 for x in v)}  sig+ {sp}  sig- {sn}\n   ' + '  '.join(cells))
        if ref == 'foris':
            v = [score.miou([r for r in R if r['dataset'] == d], 'MFU') - score.miou([r for r in R if r['dataset'] == d], 'foris') for d in c.DOMAINS]
            print(f'MFU - foris on the same T subset: mean {np.mean(v):+.2f}')
    with gzip.open(ROOT / 'evidence/research/w19_simple_head_T.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'Xp', 'Xp+')}) + '\n')


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    with gzip.open(ROOT / 'evidence/research/w13_plugin_S.jsonl.gz', 'rt') as f:
        W13 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **{k: v for k, v in W13[r['index']].items() if k in ('M0U+', 'FU+', 'U0', 'U0+')}, **r)
         for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    print('n', len(R))
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]

    def table(title, ref, rows):
        print(f'\n## {title}\n{"arm":8s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0  sig+  sig-')
        for a in rows:
            v, cells, sp, sn = [], [], 0, 0
            for d in c.DOMAINS:
                L = [r for r in R if r['dataset'] == d]
                x = score.miou(L, a) - score.miou(L, ref)
                lo, hi = score.paired_ci(L, a, ref, rng)
                sp += lo > 0; sn += hi < 0; v.append(x)
                cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]')
            print(f'{a:8s}' + ''.join(f'{s:>17s}' for s in cells) + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}{sp:6d}{sn:6d}')

    table('vs complete FoRIS', 'foris', ('MFU', 'M0U+') + ARMS + ('U0+',))
    table('vs MFU', 'MFU', tuple(a for a in ARMS if a.endswith('+')))
    with gzip.open(ROOT / 'evidence/research/w19_simple_head_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt') + ARMS}) + '\n')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'check':
        check()
    elif cmd == 'confirm_t':
        confirm_t(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == 'produce':
        produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        report()
