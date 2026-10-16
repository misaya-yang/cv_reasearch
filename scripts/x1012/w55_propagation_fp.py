"""W55: SH's propagation false positives (comparison c32 <= .5 says background, SH says foreground) -- which line adds
them (point line U, prototype line M, both), and would the point line still add them if it kept the absolute scale
of its input instead of min-max normalising it.  S half only; diagnostic.
Plan and readings: docs/research/x1012_tasks/W55_propagation_FP_PLAN.md (fixed before this file was written).

    nice -n 19 python scripts/x1012/w55_propagation_fp.py produce [workers]
    python scripts/x1012/w55_propagation_fp.py report
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

OUT = c.task_path('W55')
CATS = ('U_only', 'M_only', 'both')


def solve_abs(primary, log_e, L, lam):
    """w09.solve with the same weights and lambda, but on the un-normalised input (counterfactual only)."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from scipy.special import expit
    s = primary.ravel().astype(np.float64)
    a = (.1 + np.abs(2 * s - 1)) * expit(log_e.astype(np.float64) - np.median(log_e))
    a = a / a.mean()
    f, status = cg(sparse.diags(a) + lam * L, a * s, x0=s, rtol=1e-7, atol=1e-9, maxiter=2000)
    assert status == 0, status
    return f


def capture(b):
    import torch
    import w09_integration_lambda as w09, w19_simple_head as w19, w23_simple_pipeline as sh
    cap = {'solve': [], 'M': [], 'ind': []}
    ou, om, oi = w09.solve, w19.mean_solve, c.rc.independent

    def point(*a, **k):
        v = ou(*a, **k); cap['solve'].append((a, np.asarray(v).copy())); return v

    def proto(*a, **k):
        v = om(*a, **k); cap['M'].append(np.asarray(v).copy()); return v

    def indep(*a, **k):
        v = oi(*a, **k); cap['ind'].append(v); return v
    w09.solve, w19.mean_solve, c.rc.independent = point, proto, indep
    try:
        pre, _, rho = sh.simple_head(b, crf=False)
    finally:
        w09.solve, w19.mean_solve, c.rc.independent = ou, om, oi
    lab = c.rc.lab
    up = lambda x: lab.up(torch.from_numpy(np.asarray(x, np.float32).reshape(64, 64))).numpy().astype(np.float64)
    c32 = np.asarray(cap['ind'][0][4]['c32'], np.float64).ravel()
    if rho > .9:
        return dict(pre=pre, rho=rho, c32=c32)
    (primary, log_e, L, lam), u0 = cap['solve'][0]
    U = lab.minmax(lab.up(torch.from_numpy(u0.reshape(64, 64).astype(np.float32)))).numpy().astype(np.float64)
    M = up(cap['M'][0])
    assert np.array_equal(.5 * M + .5 * U > .5, pre), 'captured lines do not reproduce SH'
    U_abs = up(solve_abs(np.asarray(primary), log_e, L, lam))
    cmod = np.asarray(primary, np.float64).ravel()
    return dict(pre=pre, rho=rho, c32=c32, U=U, M=M, U_abs=U_abs, cmod_range=float(cmod.max() - cmod.min()),
                cmod_max=float(cmod.max()))


def one(b):
    import w31_demonstration as w31
    began = time.monotonic()
    x = capture(b)
    gt, valid = c.truth(b)                       # attribution only from here
    r = dict(w31.meta(b), gt=int(gt.sum()), rho=x['rho'], SH=c.iu(x['pre'], gt, valid),
             c32_max=float(x['c32'].max()), c32_fg_frac=float((x['c32'] > .5).mean()))
    if 'U' not in x:
        r['seconds'] = round(time.monotonic() - began, 2)
        return r
    cov = gt.reshape(64, 16, 64, 16).mean((1, 3))
    tok = ((cov <= .1) & (x['c32'].reshape(64, 64) <= .5)).repeat(16, 0).repeat(16, 1)   # pure background, comparison says bg
    pf = x['pre'] & ~gt & valid & tok
    u, m = x['U'] > .5, x['M'] > .5
    r.update(prop_fp=int(pf.sum()), fp=int((x['pre'] & ~gt & valid).sum()), fn=int((gt & ~x['pre'] & valid).sum()),
             U_only=int((pf & u & ~m).sum()), M_only=int((pf & m & ~u).sum()), both=int((pf & u & m).sum()),
             U_minmax=int((pf & u & ~(x['U_abs'] > .5)).sum()), U_involved=int((pf & u).sum()),
             cmod_range=x['cmod_range'], cmod_max=x['cmod_max'])
    assert r['U_only'] + r['M_only'] + r['both'] == r['prop_fp']
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
            if n % 100 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    from scipy.stats import spearmanr
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    same = sum(r['SH'][:2] == w23[r['index']]['SH'][:2] for r in R)
    for r in R:
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
    L = [f'n {len(R)}  check: A0 == W23 SH {same}/{len(R)} (need >= 1280); fallback {sum("U_only" not in r for r in R)}', '']
    lines = [r for r in R if 'U_only' in r]

    def block(tag, sel):
        sel = [r for r in sel if r['prop_fp'] > 0]
        if not sel:
            return
        ep = {k: np.mean([r[k] / r['prop_fp'] for r in sel]) for k in CATS}
        pool = {k: sum(r[k] for r in sel) / sum(r['prop_fp'] for r in sel) for k in CATS}
        ui = [r for r in sel if r['U_involved']]
        mm = np.mean([r['U_minmax'] / r['U_involved'] for r in ui]) if ui else float('nan')
        mmp = sum(r['U_minmax'] for r in ui) / max(sum(r['U_involved'] for r in ui), 1)
        L.append(f'{tag:28s} n {len(sel):4d}  ' + '  '.join(f'{k} {ep[k]:.1%} ({pool[k]:.1%})' for k in CATS) +
                 f'  | M involved {ep["M_only"] + ep["both"]:.1%}  | U-involved removed by keeping scale {mm:.1%} ({mmp:.1%})')
    L.append('## propagation FP pixels by line (episode mean; pooled in brackets), episodes with prop_fp > 0')
    cat = [r for r in lines if r['cat']]; non = [r for r in lines if not r['cat']]
    block('catastrophic', cat); block('non-catastrophic', non)
    for d in c.DOMAINS:
        block(f'  cat, {d}', [r for r in cat if r['dataset'] == d])
    L.append('')
    share = lambda r: r['prop_fp'] / max(r['fp'] + r['fn'], 1)
    for tag, sel in (('catastrophic', cat), ('non-catastrophic', non)):
        s = [share(r) for r in sel]
        for q in ('c32_fg_frac', 'c32_max', 'cmod_range'):
            rho, p = spearmanr(s, [r[q] for r in sel])
            L.append(f'P3 {tag:17s} Spearman(prop_FP share of errors, {q:11s}) = {rho:+.3f} (p {p:.1e}, n {len(sel)})')
    L.append(f'prop_FP share of all pixel errors: catastrophic {np.mean([share(r) for r in cat]):.1%}, '
             f'non-catastrophic {np.mean([share(r) for r in non]):.1%}')
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w55_propagation_fp_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ('cat', 'seconds')}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
