"""W46: why the reference-defined direction drifts on the query.  Ridge / prototype / aligned / transductive probes on
SH's debiased O24 tokens.  S half only.  Plan and readings: docs/research/x1012_tasks/W46_transfer_PLAN.md.

    nice -n 19 python scripts/x1012/w46_transfer.py produce [workers]
    python scripts/x1012/w46_transfer.py report
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
from w45_solvability import FOLD, best_cut

OUT = c.task_path('W46')
ARMS = ('R1', 'R2', 'R3', 'A1', 'A2', 'T1', 'PQ')


def ridge(X, y, lam):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def apply(w, X):
    return X @ w[:-1] + w[-1]


def shrunk_cov(X):
    C = np.cov(X, rowvar=False)
    return .9 * C + .1 * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])


def sqrtm(C, inv=False):
    e, V = np.linalg.eigh(C)
    e = np.maximum(e, 1e-12)
    return (V * (e ** (-.5 if inv else .5))) @ V.T


def one(b):
    import torch
    import w31_demonstration as w31
    import w42_ordered_cut as w42
    began = time.monotonic()
    got, oi = [], c.rc.independent

    def keep(*a, **k):
        v = oi(*a, **k); got.append(v); return v
    c.rc.independent = keep
    try:
        F, A0, U, M, rho = w42.capture(b)
    finally:
        c.rc.independent = oi
    r_, q_, _, y, _ = got[0]
    lab = c.rc.lab
    R, Q = lab.unit(r_).numpy().astype(np.float64), lab.unit(q_).numpy().astype(np.float64)
    y = y.numpy().astype(np.float64)
    fields, nat_t, W = {}, {}, {}
    for name, lam in (('R1', 1.), ('R2', 100.)):
        W[name] = ridge(R, y, lam); fields[name] = apply(W[name], Q); nat_t[name] = .5
    fg, bg = (R * y[:, None]).sum(0) / max(y.sum(), 1e-9), (R * (1 - y)[:, None]).sum(0) / max((1 - y).sum(), 1e-9)
    d = fg - bg
    W['R3'] = np.concatenate([d, [-.5 * (fg + bg) @ d]]); fields['R3'] = apply(W['R3'], Q); nat_t['R3'] = 0.
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-6)
    fields['A1'] = apply(ridge(z(R), y, 1.), z(Q)); nat_t['A1'] = .5
    Rc = (R - R.mean(0)) @ sqrtm(shrunk_cov(R), inv=True) @ sqrtm(shrunk_cov(Q)) + Q.mean(0)
    fields['A2'] = apply(ridge(Rc, y, 1.), Q); nat_t['A2'] = .5
    tF = F.reshape(64, 16, 64, 16).mean((1, 3)).ravel()
    t1 = np.empty(4096); wt = []
    for a in (False, True):
        tr = FOLD == a
        w = ridge(Q[tr], tF[tr], 1.); wt.append(w); t1[~tr] = apply(w, Q[~tr])
    fields['T1'] = t1; nat_t['T1'] = .5; W['T1'] = .5 * (wt[0] + wt[1])
    up = lambda t: lab.up(torch.from_numpy(t.reshape(64, 64).astype(np.float32))).numpy().astype(np.float64)
    gt, valid = c.truth(b)                                         # scoring / query-label probe only from here
    yq = gt.reshape(64, 16, 64, 16).mean((1, 3)).ravel().astype(np.float64)
    pq = np.empty(4096)
    for a in (False, True):
        tr = FOLD == a
        pq[~tr] = apply(ridge(Q[tr], yq[tr], 1.), Q[~tr])
    fields['PQ'] = pq; nat_t['PQ'] = .5
    wq = ridge(Q, yq, 1.)[:-1]
    r = dict(w31.meta(b), rho=rho, fallback=U is None, gt=int(gt.sum()))
    r['SH'] = c.iu(A0, gt, valid); r['cut_F'] = best_cut(F, gt)
    for a in ARMS:
        f = up(fields[a])
        r['cut_' + a] = best_cut(f, gt); r[a] = c.iu(f > nat_t[a], gt, valid)
    cos = lambda u: float(u @ wq / max(np.linalg.norm(u) * np.linalg.norm(wq), 1e-12))
    r['cos'] = {a: cos(W[a][:-1]) for a in ('R1', 'R3', 'T1')}
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def init():
    c.initialize()
    import torch
    torch.set_num_threads(1)


def produce_all(workers):
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 50 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    w45 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w45_solvability_S.jsonl.gz', 'rt'))}
    same = sum(r['SH'][:2] == w23[r['index']]['SH'][:2] for r in R)
    gap45 = max(abs(r['cut_R1'] - w45[r['index']]['cut_PR']) for r in R)
    for r in R:
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    L = [f'n {len(R)}  checks: SH == W23 {same}/{len(R)} (need >= 1280); max |cut R1 - W45 cut P_R| {gap45:.2e} (need < 1e-6)', '']
    rec = {}
    for tag, sel in (('catastrophic', [r for r in R if r['cat']]), ('non-catastrophic', [r for r in R if not r['cat']])):
        m = {a: np.mean([r['cut_' + a] for r in sel]) for a in ARMS + ('F',)}
        span = m['PQ'] - m['R1']
        rec[tag] = {a: (m[a] - m['R1']) / span for a in ARMS}
        L.append(f'## {tag} (n {len(sel)}): mean best-cut IoU and recovery (arm - R1)/(PQ - R1)')
        L.append('  ' + '  '.join(f'{a} {m[a]:.3f}' + (f' ({rec[tag][a]:+.0%})' if a in ARMS and a not in ('R1', 'PQ') else '') for a in ARMS + ('F',)))
        cs = {a: np.mean([r['cos'][a] for r in sel]) for a in ('R1', 'R3', 'T1')}
        L.append('  cosine with the query-GT direction: ' + '  '.join(f'{a} {v:+.3f}' for a, v in cs.items()))
        L.append('')
    L.append('## class mIoU at the natural threshold (ten-domain mean)')
    L.append('  ' + '  '.join(f'{a} {np.mean([sc.miou(by[d], a) for d in doms]):.2f}' for a in ('SH',) + ARMS))
    L.append(f'{"":12s}' + ''.join(f'{a:>8s}' for a in ARMS) + '   (minus SH)')
    for d in doms:
        L.append(f'{d[:11]:12s}' + ''.join(f'{sc.miou(by[d], a) - sc.miou(by[d], "SH"):+8.2f}' for a in ARMS))
    allrec = {a: np.mean([rec[t][a] for t in rec]) for a in ARMS}

    def kind(rc):
        h2 = max(rc['R2'], rc['R3']) >= .5 and max(rc['A1'], rc['A2']) < .2
        h1 = max(rc['A1'], rc['A2']) >= .5 and max(rc['R2'], rc['R3']) < .2
        h3 = max(rc['R2'], rc['R3'], rc['A1'], rc['A2']) < .2
        return 'H2 (reference-specific spurious directions)' if h2 else 'H1 (covariance shift)' if h1 else 'H3 (concept not fixed by the reference)' if h3 else 'other'
    g = np.mean([sc.miou(by[d], 'T1') - sc.miou(by[d], 'SH') for d in doms])
    lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], 'T1', 'SH') for d in doms], axis=0), [2.5, 97.5])
    pos = sum(sc.miou(by[d], 'T1') - sc.miou(by[d], 'SH') >= 0 for d in doms)
    L += ['', f'reading 2: catastrophic -> {kind(rec["catastrophic"])}; non-catastrophic -> {kind(rec["non-catastrophic"])}',
          f'reading 3: T1 - SH class mIoU {g:+.2f} [{lo:+.2f},{hi:+.2f}] domains >= 0 {pos}/10 -> method validation plan: {bool(g >= 1 and lo > 0 and pos >= 7)}',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback', 'SH', 'cut_F', 'cos') + ARMS + tuple('cut_' + a for a in ARMS)
    with gzip.open(ROOT / 'evidence/research/w46_transfer_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
