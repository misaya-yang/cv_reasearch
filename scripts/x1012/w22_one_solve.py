"""W22: can the FoRIS-free head Xp be simplified to fewer parts / a single solve?  (S half, before the CRF)

Motivation (W19-W21): Xp = [.5 M' + .5 U0 > .5] matches MFU and is faster than FoRIS, but it still has two lines,
two query kNN graphs (M' builds its own from FoRIS-Part-1 features; U0 uses the cached graph L), two solves with
different weightings and readouts, and U0's novelty modulation (a max-J choice over strengths).  W13/W19 show
the gain comes from combining point evidence (kernel transfer c) with prototype evidence (p), not from any
specific part, so the parts may be redundant.
Arms (all keep rho > .9 -> T):
  Xp     baseline
  Ash    M' solved on U0's graph L instead of its own graph                        (one graph)
  Anm    U0 without novelty modulation: solve(c) instead of solve(c_mod)          (no modulation)
  A1     ONE solve: primary .5 mm(c_mod) + .5 mm(p), U0-style solve, minmax readout (one line)
  A1g    ONE solve with M's rank guide on the same primary, absolute .5 readout   (one line, guided)
  A1nm   ONE solve, primary .5 mm(c) + .5 mm(p), U0-style                       (one line, no modulation)
Reading fixed before running: an arm is a valid simplification if its ten-domain mean is >= Xp - 0.1 and no
domain is significantly below Xp (paired photo bootstrap).  Among valid arms the one with the fewest parts is
taken (A1nm < A1 < A1g < Anm, Ash); it then gets the CRF and the T-half check.
Sample: the W13 S subset (<= 150 per domain).

    python scripts/x1012/w22_one_solve.py produce [workers]
    python scripts/x1012/w22_one_solve.py report
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

OUT = c.task_path('W22')
ARMS = ('Xp', 'Ash', 'Anm', 'A1', 'A1g', 'A1nm')


def one(binding):
    import torch
    import produce
    from w09_integration_lambda import solve
    from w19_simple_head import heads, mean_graph, mean_solve
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index'])}
    if produce.fallback(fields):
        with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
            t_mask = c.unpack(z['T'])
        for a in ARMS:
            r[a] = c.iu(t_mask, gt, valid)
        return r
    mm64 = lambda f: (lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-12))(np.asarray(f, np.float64).ravel())
    mmup = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32).reshape(64, 64)))).numpy().astype(np.float64)
    L, log_e = c.laplacian(fields), fields['log_e']
    q, _, _, guide, cheap = heads(binding, fields)
    p = cheap['p']
    u0 = mmup(solve(fields['c_mod'], log_e, L, 16))
    m_own = c.up(mean_solve(p.reshape(64, 64), guide, mean_graph(q))).astype(np.float64)
    m_sh = c.up(mean_solve(p.reshape(64, 64), guide, L)).astype(np.float64)
    u_nm = mmup(solve(fields['c32'], log_e, L, 16))
    prim = .5 * mm64(fields['c_mod']) + .5 * mm64(p)
    prim_nm = .5 * mm64(fields['c32']) + .5 * mm64(p)
    fields_ = {
        'Xp': .5 * m_own + .5 * u0 > .5,
        'Ash': .5 * m_sh + .5 * u0 > .5,
        'Anm': .5 * m_own + .5 * u_nm > .5,
        'A1': mmup(solve(prim, log_e, L, 16)) > .5,
        'A1g': c.up(c.solve(prim.reshape(64, 64), fields['explanation'], L, guide)) > .5,
        'A1nm': mmup(solve(prim_nm, log_e, L, 16)) > .5,
    }
    for a in ARMS:
        r[a] = c.iu(fields_[a], gt, valid)
    return r


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
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    with gzip.open(ROOT / 'evidence/research/w19_simple_head_S.jsonl.gz', 'rt') as f:
        W19 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    print('n', len(R), ' Xp identical to W19:', np.mean([e['Xp'][:2] == W19[e['index']]['Xp'][:2] for e in R]))
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]
    print(f'\n## class mIoU before CRF minus Xp\n{"arm":6s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0  sig-')
    for a in ARMS[1:]:
        v, cells, sn = [], [], 0
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            x = score.miou(L, a) - score.miou(L, 'Xp'); lo, hi = score.paired_ci(L, a, 'Xp', rng)
            v.append(x); sn += hi < 0; cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]')
        print(f'{a:6s}' + ''.join(f'{s:>17s}' for s in cells) + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}{sn:6d}')
    with gzip.open(ROOT / 'evidence/research/w22_one_solve_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt') + ARMS}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
