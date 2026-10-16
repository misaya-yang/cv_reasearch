"""W45: solvability decomposition.  For every S episode: is the target separable in the query's own O24 features
(cross-fitted ridge probe on query GT), does a reference-trained probe transfer, and does SH's field rank it?
Plan and readings: docs/research/x1012_tasks/W45_solvability_PLAN.md.

    nice -n 19 python scripts/x1012/w45_solvability.py produce [workers]
    python scripts/x1012/w45_solvability.py report
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
from diagnostic_metrics import exact_best_iou, classify_diagnostic, DIAGNOSTIC_CLASSES, SCORING_VERSION

OUT = c.task_path('W45_exact')  # W58 repair: retain the original W45 ledger verbatim.
LAM = 1.0
_bi, _bj = np.meshgrid(np.arange(64) // 8, np.arange(64) // 8, indexing='ij')
FOLD = ((_bi + _bj) % 2).ravel().astype(bool)                     # 8x8-token-block checkerboard


def ridge(X, y):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + LAM * np.eye(Xb.shape[1]); A[-1, -1] -= LAM   # no penalty on the bias
    return np.linalg.solve(A, Xb.T @ y)


def apply(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def best_cut(f, gt, valid=None):
    return exact_best_iou(f, gt, valid)


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
    pr = apply(ridge(R, y), Q)                                     # reference-trained probe on the query (label-free)
    up = lambda t: lab.up(torch.from_numpy(t.reshape(64, 64).astype(np.float32))).numpy().astype(np.float64)
    gt, valid = c.truth(b)                                         # scoring / query-label probe only from here
    yq = gt.reshape(64, 16, 64, 16).mean((1, 3)).ravel().astype(np.float64)
    pq = np.empty(4096)
    for a in (False, True):
        tr = FOLD == a
        pq[~tr] = apply(ridge(Q[tr], yq[tr]), Q[~tr])
    r = dict(w31.meta(b), rho=rho, fallback=U is None, gt=int(gt.sum()), qfrac=float(gt.mean()))
    r['SH'] = c.iu(A0, gt, valid)
    PQ, PR = up(pq), up(pr)
    r['scoring'] = SCORING_VERSION
    for name, field in (('F', F), ('PQ', PQ), ('PR', PR)):
        r['cut_' + name] = best_cut(field, gt, valid)
        r['cut_' + name + '_all'] = r['cut_' + name] if valid.all() else best_cut(field, gt)
    r['PQ'] = c.iu(PQ > .5, gt, valid); r['PR'] = c.iu(PR > .5, gt, valid)
    for field, natural in (('F', 'SH'), ('PQ', 'PQ'), ('PR', 'PR')):
        assert r['cut_' + field] + 1e-12 >= r[natural][0] / max(r[natural][1], 1)
    r['diagnostic'] = classify_diagnostic(r)
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
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    assert len(R) == 1285 and len({r['index'] for r in R}) == 1285
    assert all(r['scoring'] == SCORING_VERSION for r in R)
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    same = sum(r['SH'][:2] == w23[r['index']]['SH'][:2] for r in R)
    for r in R:
        r['SH+'] = w27[r['index']]['SH+']
        r['jsh'] = r['SH+'][0] / max(r['SH+'][1], 1)
    doms = list(c.DOMAINS)

    classify = classify_diagnostic
    assert same == len(R)
    L = [f'n {len(R)}  check: SH == W23 SH {same}/{len(R)} (exact)', '']
    for tag, sel in (('catastrophic (SH+ IoU < .3)', [r for r in R if r['jsh'] < .3]),
                     ('non-catastrophic', [r for r in R if r['jsh'] >= .3])):
        L.append(f'## {tag}: n {len(sel)}')
        L.append('  mean episode IoU: SH+ %.3f | best cut F %.3f, P_R %.3f, P_Q %.3f | natural P_R %.3f, P_Q %.3f' % (
            np.mean([r['jsh'] for r in sel]), np.mean([r['cut_F'] for r in sel]), np.mean([r['cut_PR'] for r in sel]),
            np.mean([r['cut_PQ'] for r in sel]), np.mean([r['PR'][0] / max(r['PR'][1], 1) for r in sel]),
            np.mean([r['PQ'][0] / max(r['PQ'][1], 1) for r in sel])))
        cls = [classify(r) for r in sel]
        L.append('  classes: ' + '  '.join(f'{k} {cls.count(k) / len(cls):.1%}' for k in DIAGNOSTIC_CLASSES))
        for d in doms:
            D = [r for r, k in zip(sel, cls) if r['dataset'] == d]
            if D:
                kk = [classify(r) for r in D]
                L.append(f'    {d[:11]:12s} n {len(D):3d}  ' + '  '.join(f'{k} {kk.count(k) / len(kk):5.1%}' for k in DIAGNOSTIC_CLASSES))
        if tag.startswith('catastrophic'):
            fb = [k for r, k in zip(sel, cls) if r['fallback']]
            L.append(f'  of which fallback episodes {len(fb)}: ' + '  '.join(f'{k} {fb.count(k)}' for k in DIAGNOSTIC_CLASSES))
        L.append('')
    L.append('## class mIoU (ten-domain mean, natural threshold): SH %.2f  P_R %.2f  P_Q (cross-fitted, uses query GT) %.2f' % tuple(
        np.mean([sc.miou([r for r in R if r['dataset'] == d], a) for d in doms]) for a in ('SH', 'PR', 'PQ')))
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'qfrac', 'rho', 'fallback', 'SH', 'PQ', 'PR', 'cut_F', 'cut_PQ', 'cut_PR',
            'cut_F_all', 'cut_PQ_all', 'cut_PR_all', 'scoring', 'diagnostic')
    with gzip.open(ROOT / 'evidence/research/w45_solvability_S_exact.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
