"""W57: replace the reference's own prior log-odds log(N_F/N_B) inside SH's point evidence c32 by a constant.
c = df/(df+db) = sigmoid(log(N_F/N_B) + log(m_F/m_B)); arm P_c uses pi0 = .088 (median S reference foreground
fraction, from reference masks only), P_h uses pi0 = .5.  Only c32 and its derived ind0_32/indn_32 change.
S half only.  Plan and readings: docs/research/x1012_tasks/W57_reference_prior_PLAN.md (fixed before this file).

    nice -n 19 python scripts/x1012/w57_reference_prior.py produce [workers]
    python scripts/x1012/w57_reference_prior.py report
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

OUT = c.task_path('W57')
ARMS = {'A0': None, 'P_c': .088, 'P_h': .5}


def recompute(r, q, y, fields, pi0):
    """r5 independent's c / ind0 / indn with the reference prior (pi0 None) or a constant prior pi0."""
    import torch
    import torch.nn.functional as F
    lab = c.rc.lab
    mu = .5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    kernel = torch.exp((sim - sim.max()) / .05)
    df, db = kernel @ y, kernel @ (1 - y)
    if pi0 is None:
        cc = df / (df + db).clamp_min(1e-30)
    else:
        mf, mb = df / y.sum().clamp_min(1e-6), db / (1 - y).sum().clamp_min(1e-6)
        cc = pi0 * mf / (pi0 * mf + (1 - pi0) * mb).clamp_min(1e-30)
    onehot = F.one_hot(torch.from_numpy(fields['regions64'].astype(np.int64)), 64).float()
    region = lambda v: onehot @ ((onehot.T @ v) / onehot.sum(0).clamp_min(1))
    sim_max = sim.max(1).values
    return dict(fields, c32=cc.numpy(), ind0_32=region(cc).numpy(), indn_32=region(cc * lab.minmax(sim_max)).numpy())


def one(b):
    import w23_simple_pipeline as sh
    import w31_demonstration as w31
    began = time.monotonic()
    oi = c.rc.independent
    out, check = {}, {}
    for arm, pi0 in ARMS.items():
        def patched(*a, **k):
            r_cpu, q_cpu, mask, y, fields = oi(*a, **k)
            new = recompute(r_cpu, q_cpu, y, fields, pi0)
            if pi0 is None:
                check['c32'] = float(np.abs(new['c32'] - fields['c32']).max())
                check['ind'] = float(max(np.abs(new['ind0_32'] - fields['ind0_32']).max(), np.abs(new['indn_32'] - fields['indn_32']).max()))
                new = fields                          # A0 is exactly the frozen fields
            out['refpi_' + arm] = float(y.mean())
            return r_cpu, q_cpu, mask, y, new
        c.rc.independent = patched
        try:
            pre, _, rho = sh.simple_head(b, crf=False)
        finally:
            c.rc.independent = oi
        out[arm] = pre
    gt, valid = c.truth(b)                       # scoring only from here
    r = dict(w31.meta(b), gt=int(gt.sum()), rho=rho, ref_frac=out['refpi_A0'], check=check,
             **{a: c.iu(out[a], gt, valid) for a in ARMS})
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def produce_all(workers):
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 100 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    same = sum(r['A0'][:2] == w23[r['index']]['SH'][:2] for r in R)
    L = [f"n {len(R)}  checks: max |c32 recomputed - frozen| {max(r['check']['c32'] for r in R):.2e}, ind {max(r['check']['ind'] for r in R):.2e} "
         f"(need < 1e-4); A0 == W23 SH {same}/{len(R)} (need >= 1280)", '']
    rng = np.random.default_rng(c.SEED)
    for a in ('P_c', 'P_h'):
        cells, v, sp, sn = [], [], 0, 0
        for d in c.DOMAINS:
            S = [r for r in R if r['dataset'] == d]
            x = score.miou(S, a) - score.miou(S, 'A0'); lo, hi = score.paired_ci(S, a, 'A0', rng)
            v.append(x); sp += lo > 0; sn += hi < 0; cells.append(f'{d[:6]} {x:+.2f}[{lo:+.1f},{hi:+.1f}]')
        L.append(f'{a} - A0 (pre-CRF, 10-domain): mean {np.mean(v):+.2f}  sig+ {sp} sig- {sn}\n   ' + '  '.join(cells))
    L.append('')
    iou = lambda v: v[0] / max(v[1], 1)
    m = np.array([np.log2((r['ref_frac'] + 1e-4) / (r['gt'] / 1048576 + 1e-4)) for r in R])
    L.append('log2(ref fraction / GT fraction) bin: n, mean IoU A0 / P_c / P_h')
    for lo, hi in ((-99, -3), (-3, -1.5), (-1.5, -.5), (-.5, .5), (.5, 1.5), (1.5, 3), (3, 99)):
        S = [r for r, x in zip(R, m) if lo <= x < hi]
        L.append(f'  [{lo:+5.1f},{hi:+5.1f}) n {len(S):4d}  ' + '  '.join(f'{a} {100 * np.mean([iou(r[a]) for r in S]):5.1f}' for a in ARMS))
    for tag, sel in (('> +1.5', m > 1.5), ('< -1.5', m < -1.5)):
        S = [r for r, s in zip(R, sel) if s]
        L.append(f'  mismatch {tag}: n {len(S)}  P_c - A0 {100 * np.mean([iou(r["P_c"]) - iou(r["A0"]) for r in S]):+.2f}  '
                 f'P_h - A0 {100 * np.mean([iou(r["P_h"]) - iou(r["A0"]) for r in S]):+.2f}')
    L.append('')
    N = [r for r in R if r['rho'] <= .9]
    D = sorted({r['dataset'] for r in N})
    X = np.array([[np.log2(r['ref_frac'] + 1e-4), np.log2(r['gt'] / 1048576 + 1e-4)] + [1. * (r['dataset'] == d) for d in D] for r in N])
    for a in ARMS:
        y = np.array([np.log2((r[a][2] + 1) / (r['gt'] + 1)) for r in N])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        bs = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][0] for i in [rng.integers(0, len(y), len(y)) for _ in range(300)]]
        L.append(f'{a:4s} log2(out/GT) = a*log2(ref) + b*log2(GT) + domain:  a {coef[0]:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]  b {coef[1]:+.3f}  (non-fallback n {len(N)})')
    L.append(f"seconds per episode median {np.median([r['seconds'] for r in R]):.1f}")
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w57_reference_prior_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'seconds'}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
