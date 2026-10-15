"""W47: is there one linear direction that separates both the reference mask and the query GT (concept consistency)?
S half only, diagnostic (reads query GT).  Plan and readings: docs/research/x1012_tasks/W47_concept_consistency_PLAN.md.

    nice -n 19 python scripts/x1012/w47_concept_consistency.py produce [workers]
    python scripts/x1012/w47_concept_consistency.py report
"""
import gzip, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
from w45_solvability import FOLD, best_cut, ridge, apply

OUT = c.task_path('W47')


def tokens(b):
    """Unit debiased O24 tokens of reference and query, reference coverage, the reference mask, SH capture."""
    import w42_ordered_cut as w42
    got, oi = [], c.rc.independent

    def keep(*a, **k):
        v = oi(*a, **k); got.append(v); return v
    c.rc.independent = keep
    try:
        F, A0, U, M, rho = w42.capture(b)
    finally:
        c.rc.independent = oi
    r_, q_, rmask, y, _ = got[0]
    lab = c.rc.lab
    return (lab.unit(r_).numpy().astype(np.float64), lab.unit(q_).numpy().astype(np.float64),
            y.numpy().astype(np.float64), rmask.numpy().astype(bool), A0, U, rho)


def probes(R, Q, y, yq, rgt, gt, up):
    pq, prr, pjq, pjr = (np.empty(4096) for _ in range(4))
    for a in (False, True):
        tr = FOLD == a
        pq[~tr] = apply(ridge(Q[tr], yq[tr]), Q[~tr])
        prr[~tr] = apply(ridge(R[tr], y[tr]), R[~tr])
        w = ridge(np.concatenate([R[tr], Q[tr]]), np.concatenate([y[tr], yq[tr]]))
        pjq[~tr] = apply(w, Q[~tr]); pjr[~tr] = apply(w, R[~tr])
    pqr = apply(ridge(Q, yq), R)
    return dict(PQ=best_cut(up(pq), gt), PJQ=best_cut(up(pjq), gt), PRR=best_cut(up(prr), rgt),
                PJR=best_cut(up(pjr), rgt), PQR=best_cut(up(pqr), rgt))


def pca(R, Q, k=32):
    X = np.concatenate([R, Q]); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T
    return (R - mu) @ P, (Q - mu) @ P


def one(b, other):
    import torch
    import w31_demonstration as w31
    began = time.monotonic()
    R, Q, y, rgt, A0, U, rho = tokens(b)
    R2, _, y2, rgt2, _, _, _ = tokens(other)
    lab = c.rc.lab
    gt, valid = c.truth(b)
    yq = gt.reshape(64, 16, 64, 16).mean((1, 3)).ravel().astype(np.float64)
    up = lambda t: lab.up(torch.from_numpy(t.reshape(64, 64).astype(np.float32))).numpy().astype(np.float64)
    r = dict(w31.meta(b), rho=rho, fallback=U is None, gt=int(gt.sum()), ref=int(rgt.sum()), other=str(other['index']))
    r['SH'] = c.iu(A0, gt, valid)
    r['full'] = probes(R, Q, y, yq, rgt, gt, up)
    r['full_ctrl'] = probes(R2, Q, y2, yq, rgt2, gt, up)
    Rp, Qp = pca(R, Q); r['k32'] = probes(Rp, Qp, y, yq, rgt, gt, up)
    Rp2, Qp2 = pca(R2, Q); r['k32_ctrl'] = probes(Rp2, Qp2, y2, yq, rgt2, gt, up)
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def pairs(rows):
    """Mismatched reference: within the domain, ordered by index, the next episode (cyclic) of a different class."""
    import w31_demonstration as w31
    cls = lambda b: w31.meta(b)['cls']
    out = []
    for d in dict.fromkeys(b['dataset'] for b in rows):
        D = sorted([b for b in rows if b['dataset'] == d], key=lambda b: str(b['index']))
        for i, b in enumerate(D):
            o = next((D[(i + j) % len(D)] for j in range(1, len(D)) if cls(D[(i + j) % len(D)]) != cls(b)), None)
            out.append((b, o))
    return out


def run(p):
    return one(*p)


def init():
    c.initialize()
    import torch
    torch.set_num_threads(1)


def produce_all(workers):
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [p for p in pairs(w31.rows()) if str(p[0]['index']) not in done]
    assert all(o is not None for _, o in todo)
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(run, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 50 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    w45 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w45_solvability_S.jsonl.gz', 'rt'))}
    gap = max(abs(r['full']['PQ'] - w45[r['index']]['cut_PQ']) for r in R)

    def w45class(e):
        if e['cut_PQ'] < .5:
            return 'representation'
        if e['cut_F'] >= .5:
            return 'readout'
        if e['cut_PR'] < .3:
            return 'transfer'
        return 'other'

    def kind(p):
        if (p['PJQ'] < .5 <= p['PQ']) or (p['PJR'] < .5 <= p['PRR']):
            return 'conflict'
        if p['PJQ'] >= .5 and p['PJR'] >= .5:
            return 'consistent'
        return 'other'
    for r in R:
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
        r['w45'] = w45class(w45[r['index']])
    L = [f'n {len(R)}  check: max |PQ - W45 cut_PQ| {gap:.2e} (need < 1e-6)', '']
    power = {}
    for cap in ('full', 'k32'):
        t = np.mean([kind(r[cap]) == 'consistent' for r in R]); u = np.mean([kind(r[cap + '_ctrl']) == 'consistent' for r in R])
        power[cap] = t - u >= .20
        L.append(f'power at {cap}: consistent with true reference {t:.1%}, with mismatched reference {u:.1%} -> difference {100 * (t - u):+.1f} pts, valid: {power[cap]}')
    L.append('')
    groups = (('transfer-failure catastrophic (reading 2)', [r for r in R if r['cat'] and r['w45'] == 'transfer']),
              ('all catastrophic', [r for r in R if r['cat']]), ('non-catastrophic', [r for r in R if not r['cat']]))
    for cap in ('full', 'k32'):
        for tag, sel in groups:
            for arm in (cap, cap + '_ctrl'):
                P = [r[arm] for r in sel]; k = [kind(p) for p in P]
                L.append(f'[{arm:9s}] {tag}: n {len(sel)}  PQ %.3f PJ(Q) %.3f | PRR %.3f PJ(R) %.3f | PQR %.3f  ' % tuple(
                    np.mean([p[a] for p in P]) for a in ('PQ', 'PJQ', 'PRR', 'PJR', 'PQR')) +
                    '  '.join(f'{x} {k.count(x) / max(len(k), 1):.1%}' for x in ('conflict', 'consistent', 'other')))
            if tag.startswith('transfer'):
                for d in dict.fromkeys(r['dataset'] for r in sel):
                    kk = [kind(r[cap]) for r in sel if r['dataset'] == d]
                    L.append(f'      {d[:11]:12s} n {len(kk):3d}  ' + '  '.join(f'{x} {kk.count(x) / len(kk):5.1%}' for x in ('conflict', 'consistent', 'other')))
        L.append('')
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'ref', 'rho', 'fallback', 'other', 'SH', 'full', 'full_ctrl', 'k32', 'k32_ctrl')
    with gzip.open(ROOT / 'evidence/research/w47_concept_consistency_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
