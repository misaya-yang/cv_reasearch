"""W24: sensitivity of the FoRIS-free head SH to its inherited constants.  (S half, before the CRF)

Motivation: SH (w23_simple_pipeline.simple_head) inherits three constants that were never examined in this
configuration: the fusion weight on M' (.5), the graph strength lambda of both solves (16) and the background
weight of the prototype contrast p (.55, FoRIS's dino_bg_weight).  A paper needs their sensitivity, and a
badly placed inherited value would be a free gain.
One factor at a time around SH (w = .5, lambda = 16, w_bg = .55):
  w      .3 .4 .6 .7            final = [w M' + (1 - w) U0 > .5]
  lam    4 64                   both solves
  wbg    0 1                    p = q.mu_fg - wbg q.mu_bg_orth
Reading fixed before running: this is a sensitivity table, not a search.  The SH values stay unless an
alternative is >= +0.5 in the ten-domain mean with no domain significantly below SH; such a value would go to a
one-shot T-half check before any change.
Sample: the W13 S subset (<= 150 per domain); rho > .9 episodes keep T for every arm.

    python scripts/x1012/w24_sensitivity.py produce [workers]
    python scripts/x1012/w24_sensitivity.py report
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

OUT = c.task_path('W24')
VARIANTS = (('w', (.3, .4, .6, .7)), ('lam', (4, 64)), ('wbg', (0., 1.)))
ARMS = ('SH',) + tuple(f'{k}={v}' for k, vals in VARIANTS for v in vals)


def mean_solve_lam(score, guide, L, lam):
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from ics.methods import rcg
    s = rcg.minmax(score).ravel(); y = (s + .25 * (rcg.rank(guide) - rcg.rank(s))).astype(np.float64)
    a = .1 + np.abs(2 * s - 1); a = (a / a.mean()).astype(np.float64)
    z, status = cg(sparse.diags(a) + lam * L, a * y, x0=y, rtol=1e-7, atol=1e-9, maxiter=600)
    if status:
        raise RuntimeError('CG failure')
    return z.reshape(64, 64).astype(np.float32)


def one(binding):
    import torch
    import torch.nn.functional as F
    from w09_integration_lambda import solve
    rc, lab = c.rc, c.rc.lab
    gt, valid = c.truth(binding)
    r_, q, _, y, f = rc.independent(binding, {})
    out = {'index': str(binding['index'])}
    if float(f['reference_rho'][0]) > .9:
        mask = (lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))) > .5).numpy()
        for a in ARMS:
            out[a] = c.iu(mask, gt, valid)
        return out
    c_mod, _ = rc.modulate(torch.from_numpy(f['c32']), torch.from_numpy(f['regions64'].astype(np.int64)),
                           f['ind0_32'], f['indn_32'], f['sim_max'], f['back_f'])
    L = rc.graph(q, f['query_neighbors20']).tocsr()
    cov = y.numpy()
    qn, rn = F.normalize(q.float(), dim=1), F.normalize(r_.float(), dim=1)
    fi = np.flatnonzero(cov >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov == cov.max())
    guide = (qn @ F.normalize(rn[fi].mean(0), dim=0)).numpy()
    fg = torch.from_numpy(cov >= .5 if (cov >= .5).any() else cov == cov.max())
    mu_fg = F.normalize(rn[fg].mean(0), dim=0)
    bg = rn[~fg]
    if len(bg):
        hard = bg[(bg @ mu_fg).topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0); mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg
        mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p_of = lambda wbg: (qn @ mu_fg - wbg * (qn @ mu_bg)).numpy()
    up = lambda z: lab.up(torch.from_numpy(np.asarray(z, np.float32).reshape(64, 64))).numpy().astype(np.float64)
    U = lambda lam: lab.minmax(torch.from_numpy(up(solve(c_mod.numpy(), f['log_e'], L, lam)))).numpy()
    M = lambda lam, wbg: up(mean_solve_lam(p_of(wbg).reshape(64, 64), guide, L, lam))
    u16, m16 = U(16), M(16, .55)
    masks = {'SH': .5 * m16 + .5 * u16 > .5}
    for w in VARIANTS[0][1]:
        masks[f'w={w}'] = w * m16 + (1 - w) * u16 > .5
    for lam in VARIANTS[1][1]:
        masks[f'lam={lam}'] = .5 * M(lam, .55) + .5 * U(lam) > .5
    for wbg in VARIANTS[2][1]:
        masks[f'wbg={wbg}'] = .5 * M(16, wbg) + .5 * u16 > .5
    for a in ARMS:
        out[a] = c.iu(masks[a], gt, valid)
    return out


def produce_all(workers):
    import w13_plugin as w13
    OUT.mkdir(parents=True, exist_ok=True)
    rows = w13.subset([b for b in c.rows() if c.half(b) == 'S'], 150)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for r in pool.map(one, todo, chunksize=4):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]
    print('n', len(R), f'\n## class mIoU before CRF minus SH (w=.5, lambda=16, wbg=.55)\n{"arm":9s}' + ''.join(f'{d:>8s}' for d in names) + '    mean  >=0  sig+  sig-')
    for a in ARMS[1:]:
        v, sp, sn = [], 0, 0
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            v.append(score.miou(L, a) - score.miou(L, 'SH')); lo, hi = score.paired_ci(L, a, 'SH', rng)
            sp += lo > 0; sn += hi < 0
        print(f'{a:9s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}{sum(x >= 0 for x in v):5d}{sp:6d}{sn:6d}')
    with gzip.open(ROOT / 'evidence/research/w24_sensitivity_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt') + ARMS}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
