"""W41: where SH's errors are (common vs disagreement regions of the two actual fields) and the W39 A field exchange.
S half only.  Plan and readings: docs/research/x1012_tasks/W41_error_geography_PLAN.md.

    nice -n 19 python scripts/x1012/w41_error_geography.py produce [workers]
    python scripts/x1012/w41_error_geography.py report
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

OUT = c.task_path('W41')
NAT = ('U', 'M', 'X', 'Y', 'F', 'XM', 'UY', 'XY', 'Xt', 'Yt', 'XtM', 'UYt', 'XtYt')
EQ = tuple('k_' + a for a in NAT)
ORA = ('O_D', 'O_C', 'O_C0', 'O_C1', 'O_fb')
REG = ('C0', 'C1', 'D', 'fb')


def fields(b):
    """U = minmax(up(U0)), M = up(M') at 1024, the natural mask and rho (same capture as W37)."""
    import torch
    import w09_integration_lambda as w09, w19_simple_head as w19, w23_simple_pipeline as sh
    cap = {'U0': [], 'M': []}
    ou, om = w09.solve, w19.mean_solve

    def point(*a, **k):
        v = ou(*a, **k); cap['U0'].append(np.asarray(v).copy()); return v

    def proto(*a, **k):
        v = om(*a, **k); cap['M'].append(np.asarray(v).copy()); return v
    w09.solve, w19.mean_solve = point, proto
    try:
        pre, _, rho = sh.simple_head(b, crf=False)
    finally:
        w09.solve, w19.mean_solve = ou, om
    if rho > .9:
        return None, None, pre, rho
    lab = c.rc.lab
    U = lab.minmax(lab.up(torch.from_numpy(cap['U0'][0].reshape(64, 64).astype(np.float32)))).numpy().astype(np.float64)
    M = lab.up(torch.from_numpy(cap['M'][0])).numpy().astype(np.float64)
    assert np.array_equal(.5 * M + .5 * U > .5, pre), 'captured fields do not reproduce SH'
    return U, M, pre, rho


def exchange(recv, donor):
    """W39 A (1): donor order statistics placed at the receiver's ranks (ties by row-major coordinate), and the
    tie-preserving version (mean of donor order statistics over each receiver tie block).  Also tied pixel count."""
    r, d = recv.ravel(), np.sort(donor.ravel())
    order = np.argsort(r, kind='stable')
    x = np.empty_like(r); x[order] = d
    rs = r[order]
    ids = np.concatenate([[0], np.cumsum(rs[1:] != rs[:-1])])
    cnt = np.bincount(ids)
    xt = np.empty_like(r); xt[order] = (np.bincount(ids, d) / cnt)[ids]
    return x.reshape(recv.shape), xt.reshape(recv.shape), int(cnt[cnt > 1].sum())


def topk(f, k):
    """k highest values; among equal values at the cut, larger row-major index first (= stable ascending order)."""
    v = f.ravel()
    out = np.zeros(v.size, bool)
    if k <= 0:
        return out.reshape(f.shape)
    if k >= v.size:
        return ~out.reshape(f.shape)
    t = np.partition(v, v.size - k)[v.size - k]
    out[v > t] = True
    need = k - int(out.sum())
    if need > 0:
        out[np.flatnonzero(v == t)[-need:]] = True
    return out.reshape(f.shape)


def one(b):
    import torch
    import produce
    import w31_demonstration as w31
    began = time.monotonic()
    U, M, A, rho = fields(b)
    fb = U is None
    if fb:
        plus = A
    else:
        with c.rc.pc.inference_only(b):
            plus = produce.crf(b, A)
        torch.set_num_threads(1)
    k0 = int(A.sum())
    r = dict(w31.meta(b), rho=rho, fallback=fb, k0=k0)
    if not fb:
        X, Xt, tU = exchange(U, M)
        Y, Yt, tM = exchange(M, U)
        fl = {'U': U, 'M': M, 'X': X, 'Y': Y, 'F': .5 * (U + M), 'XM': .5 * (X + M), 'UY': .5 * (U + Y),
              'XY': .5 * (X + Y), 'Xt': Xt, 'Yt': Yt, 'XtM': .5 * (Xt + M), 'UYt': .5 * (U + Yt), 'XtYt': .5 * (Xt + Yt)}
        nat = {a: fl[a] > .5 for a in NAT}
        eq = {'k_' + a: topk(fl[a], k0) for a in NAT}
        assert np.array_equal(nat['F'], A)
        C0, C1 = (U <= .5) & (M <= .5), (U > .5) & (M > .5)
        reg = {'C0': C0, 'C1': C1, 'D': ~(C0 | C1), 'fb': np.zeros_like(A)}
        r.update(ties_U=tU, ties_M=tM, a2_violations=[int((A & C0).sum()), int((C1 & ~A).sum())],
                 tie_flip=[int((nat['X'] != nat['Xt']).sum()), int((nat['Y'] != nat['Yt']).sum())])
    else:
        nat = {a: A for a in NAT}; eq = {a: A for a in EQ}
        z = np.zeros_like(A)
        reg = {'C0': z, 'C1': z, 'D': z, 'fb': ~z}
    gt, valid = c.truth(b)                      # read only after every label-free output is fixed
    r['gt'] = int(gt.sum()); r['valid_frac'] = float(valid.mean())
    r['A'] = c.iu(A, gt, valid); r['SH+'] = c.iu(plus, gt, valid)
    for a in NAT:
        r[a] = c.iu(nat[a], gt, valid)
    for a in EQ:
        r[a] = c.iu(eq[a], gt, valid)
    swap = lambda R: (A & ~R) | (gt & R)
    r['O_D'] = c.iu(swap(reg['D']), gt, valid)
    r['O_C'] = c.iu(swap(reg['C0'] | reg['C1']), gt, valid)
    r['O_C0'] = c.iu(swap(reg['C0']), gt, valid)
    r['O_C1'] = c.iu(swap(reg['C1']), gt, valid)
    r['O_fb'] = c.iu(swap(reg['fb']), gt, valid)
    Av, Sv = A & valid, plus & valid
    add, rem = Sv & ~Av, Av & ~Sv
    err = (Av ^ gt)
    for k, R in reg.items():
        r['edit_' + k] = [int((add & gt & R).sum()), int((add & ~gt & R).sum()),
                          int((rem & gt & R).sum()), int((rem & ~gt & R).sum())]
        r['err_' + k] = [int((err & gt & R).sum()), int((err & ~gt & R).sum())]   # FN, FP of A in the region
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


def crf_attribution(D):
    """W39 A (6): per-region contribution of SH+ - A, slot-pooled with SH+'s final denominator; sums to the total."""
    slots = {}
    for r in D:
        s = slots.setdefault((r['fold'], r['cls']), {'I': 0, 'U': 0, **{k: np.zeros(4) for k in REG}})
        s['I'] += r['A'][0]; s['U'] += r['A'][1]
        for k in REG:
            s[k] += r['edit_' + k]
    out = {k: [] for k in REG}
    for s in slots.values():
        J = s['I'] / max(s['U'], 1)
        den = max(s['U'] + sum(s[k][1] - s[k][3] for k in REG), 1)
        for k in REG:
            a, b_, c_, d = s[k]
            out[k].append(100 * (a - c_ - J * (b_ - d)) / den)
    return {k: float(np.mean(v)) for k, v in out.items()}


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    nf = [r for r in R if not r['fallback']]
    same = sum(r['A'][:2] == w23[r['index']]['SH'][:2] for r in R)
    same_plus = sum(r['SH+'][:2] == w27[r['index']]['SH+'][:2] for r in R)
    viol = sum(sum(r['a2_violations']) for r in nf)
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    L = [f'n {len(R)} (fallback {len(R) - len(nf)})  checks: A == W23 SH {same}/{len(R)} (need >= 1280); '
         f'SH+ == W27 SH+ {same_plus}/{len(R)}; A2 violating pixels {viol} (need 0)',
         f'ties: U tied pixels median {np.median([r["ties_U"] for r in nf]):.0f}, M {np.median([r["ties_M"] for r in nf]):.0f}; '
         f'natural X vs Xt differ in {np.mean([r["tie_flip"][0] > 0 for r in nf]):.1%} of episodes, Y vs Yt {np.mean([r["tie_flip"][1] > 0 for r in nf]):.1%}; '
         f'valid < 1 in {np.mean([r["valid_frac"] < 1 for r in R]):.1%} of episodes']
    fn = {k: sum(r['err_' + k][0] for r in R) for k in REG}; fp = {k: sum(r['err_' + k][1] for r in R) for k in REG}
    tot = sum(fn.values()) + sum(fp.values())
    L += ['', '## pooled pixel errors of A by region (share of all errors)'] + \
         [f'{k:3s} FN {fn[k] / 1e6:6.2f}M ({fn[k] / tot:5.1%})  FP {fp[k] / 1e6:6.2f}M ({fp[k] / tot:5.1%})' for k in REG]

    def table(title, arms, base):
        rows = [f'## {title}', f'{"":12s}' + ''.join(f'{a:>8s}' for a in arms)]
        mean = {a: [] for a in arms}
        for d in doms:
            cells = []
            for a in arms:
                v = sc.miou(by[d], a) - sc.miou(by[d], base); mean[a].append(v); cells.append(f'{v:+8.2f}')
            rows.append(f'{d[:11]:12s}' + ''.join(cells))
        rows.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):+8.2f}' for a in arms))
        return rows, {a: float(np.mean(v)) for a, v in mean.items()}

    def ci(a, b):
        lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], a, b) for d in doms], axis=0), [2.5, 97.5])
        v = np.mean([sc.miou(by[d], a) - sc.miou(by[d], b) for d in doms])
        return v, lo, hi

    t1, ora = table('region capacities (oracle, pre-CRF): mIoU minus A', ORA, 'A')
    t2, _ = table('natural threshold (> .5): mIoU minus A (= F)', ('U', 'M', 'X', 'Y', 'XM', 'UY', 'XY', 'Xt', 'Yt', 'XtM', 'UYt', 'XtYt'), 'A')
    t3, _ = table('equal area (top k0 = |A|): mIoU minus A', EQ, 'A')
    L += [''] + t1 + [''] + t2 + [''] + t3
    att = {d: crf_attribution(by[d]) for d in doms}
    L += ['', '## CRF attribution (W39 A eq. 6): SH+ - A by region, slot-pooled, SH+ final denominator',
          f'{"":12s}' + ''.join(f'{k:>8s}' for k in REG) + '     sum   SH+-A']
    for d in doms:
        L.append(f'{d[:11]:12s}' + ''.join(f'{att[d][k]:+8.2f}' for k in REG) +
                 f'{sum(att[d].values()):+8.2f}{sc.miou(by[d], "SH+") - sc.miou(by[d], "A"):+8.2f}')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean([att[d][k] for d in doms]):+8.2f}' for k in REG))
    # readings
    oc, od = ora['O_C'], ora['O_D']
    geo = 'common errors dominate (O_C >= 2 O_D)' if oc >= 2 * od else 'disagreement dominates (O_D >= 2 O_C)' if od >= 2 * oc else 'both'
    L += ['', f'reading 2: O_C {oc:+.2f}  O_D {od:+.2f}  -> {geo}']
    res = {}
    for tag, (xm, uy) in (('primary', ('k_XM', 'k_UY')), ('tie-preserving', ('k_XtM', 'k_UYt'))):
        res[tag] = {f'{p}-{q}': ci(p, q) for p in (xm, uy) for q in ('k_U', 'k_M')}
        L.append(f'reading 3 {tag}: ' + '  '.join(f'{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]' for k, v in res[tag].items()))
    h2 = all(v[1] > 0 and v[0] > 0 for v in res['primary'].values())
    best = max(ci('k_U', 'A')[0], ci('k_M', 'A')[0])
    nat_ok = ci('A', 'U')[0] > 0 and ci('A', 'M')[0] > 0
    h1 = ci('k_XM', 'A')[0] <= best and ci('k_UY', 'A')[0] <= best and nat_ok
    flip = any(np.sign(res['primary'][k][0]) != np.sign(v[0]) for k, v in zip(res['primary'], res['tie-preserving'].values()))
    L.append(f'reading 3: H2 (order combination useful at equal area) {h2}; H1 (area/scale) {h1}; '
             f'natural F beats both lines {nat_ok}; tie-dependent sign flip {flip}')
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback', 'k0', 'valid_frac', 'A', 'SH+') + NAT + EQ + ORA \
        + tuple('edit_' + k for k in REG) + tuple('err_' + k for k in REG)
    with gzip.open(ROOT / 'evidence/research/w41_error_geography_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys if k in r}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
