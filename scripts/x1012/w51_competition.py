"""W51: SH's point evidence is the comparison c = sigmoid((F - B)/tau) of a query token's soft similarity to the
reference foreground (F) and background (B).  Attribute SH's pixel errors to the side of that comparison that failed.
S half only.  Plan and readings: docs/research/x1012_tasks/W51_competition_PLAN.md.

    nice -n 19 python scripts/x1012/w51_competition.py produce [workers]
    python scripts/x1012/w51_competition.py report
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
from w47_concept_consistency import pairs
from w48_reference_specificity import with_reference

OUT = c.task_path('W51')
TAU = .05
CATS = ('contamination', 'appearance', 'competitor', 'neither', 'prop_FN', 'prop_FP', 'mixed')


def auc(s, pos):
    from scipy.stats import rankdata
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return None
    rk = rankdata(s)
    return float((rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fb(r, q, y):
    """F, B of every query token; sigmoid((F - B)/tau) equals r5's c32 exactly."""
    import torch
    lab = c.rc.lab
    mu = .5 * (r.mean(0) + q.mean(0))
    s = (lab.unit(q - mu) @ lab.unit(r - mu).T).double() / TAU
    F = TAU * torch.logsumexp(s + torch.log(y.double())[None], 1)
    B = TAU * torch.logsumexp(s + torch.log((1 - y).double())[None], 1)
    return F.numpy(), B.numpy()


def anchor(cov, hi):
    """Pure-token set for the median reference (plan section 2), target side if hi else background side."""
    for t in ((.9, .5) if hi else (.1, .5)):
        m = cov >= t if hi else (cov <= t if t == .1 else cov < t)
        if m.any():
            return m
    m = np.zeros_like(cov, bool); m[np.argmax(cov) if hi else np.argmin(cov)] = True
    return m


def attribute(A0, gt, valid, F, B):
    cov = gt.reshape(64, 16, 64, 16).mean((1, 3)).ravel()
    fn = (gt & ~A0 & valid).reshape(64, 16, 64, 16).sum((1, 3)).ravel().astype(np.int64)
    fp = (A0 & ~gt & valid).reshape(64, 16, 64, 16).sum((1, 3)).ravel().astype(np.int64)
    pt, pb = cov >= .9, cov <= .1
    mix = ~pt & ~pb
    fmed, bmed = np.median(F[anchor(cov, True)]), np.median(B[anchor(cov, False)])
    win = F > B
    out = {'mixed': int(fn[mix].sum() + fp[mix].sum() + fp[pt].sum() + fn[pb].sum()),   # incl. sub-token edge pixels
           'prop_FN': int(fn[pt & win].sum()), 'contamination': int(fn[pt & ~win & (B >= bmed)].sum()),
           'appearance': int(fn[pt & ~win & (B < bmed)].sum()),
           'prop_FP': int(fp[pb & ~win].sum()), 'competitor': int(fp[pb & win & (F >= fmed)].sum()),
           'neither': int(fp[pb & win & (F < fmed)].sum())}
    assert sum(out.values()) == int(fn.sum() + fp.sum())
    pure = pt | pb
    aucs = {'A_F': auc(F[pure], pt[pure]), 'A_B': auc(-B[pure], pt[pure]), 'A_D': auc((F - B)[pure], pt[pure])}
    return out, aucs


def one(pair):
    import torch
    import w31_demonstration as w31
    import w42_ordered_cut as w42
    b, other = pair
    began = time.monotonic()
    variants = {'true': b}
    if other is not None:
        variants['mis'] = with_reference(b, other['row'], other['fit']['reference_raw'], 'reference')
    got = {}
    for a, bb in variants.items():
        keep, oi = [], c.rc.independent

        def hook(*x, **k):
            v = oi(*x, **k); keep.append(v); return v
        c.rc.independent = hook
        try:
            F_, A0, U, M, rho = w42.capture(bb)
        finally:
            c.rc.independent = oi
        r_, q_, _, y, f = keep[0]
        F, B = fb(r_, q_, y)
        with np.errstate(over='ignore'):
            err = float(np.max(np.abs(1 / (1 + np.exp((B - F) / TAU)) - f['c32'])))
        got[a] = (A0, F, B, err, U is None)
    gt, valid = c.truth(b)                       # scoring and attribution only from here
    r = dict(w31.meta(b), gt=int(gt.sum()))
    for a, (A0, F, B, err, fbk) in got.items():
        out, aucs = attribute(A0, gt, valid, F, B)
        r[a] = c.iu(A0, gt, valid); r['err_' + a] = out; r['auc_' + a] = aucs; r['c32_diff_' + a] = err; r['fb_' + a] = fbk
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
    todo = [p for p in pairs(w31.rows()) if str(p[0]['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 100 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    w45 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w45_solvability_S.jsonl.gz', 'rt'))}
    same = sum(r['true'][:2] == w23[r['index']]['SH'][:2] for r in R)
    worst = max(max(r['c32_diff_true'], r.get('c32_diff_mis', 0)) for r in R)
    for r in R:
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
        e = w45[r['index']]
        r['w45'] = 'representation' if e['cut_PQ'] < .5 else 'readout' if e['cut_F'] >= .5 else 'transfer' if e['cut_PR'] < .3 else 'other'
    L = [f'n {len(R)}  check: max |sigmoid((F-B)/tau) - c32| {worst:.2e} (need < 1e-4); true A0 == W23 SH {same}/{len(R)} (need >= 1280)', '']

    def shares(sel, a):
        E = [r['err_' + a] for r in sel if 'err_' + a in r]
        tot = [sum(e.values()) for e in E]
        ep = {k: np.mean([e[k] / t for e, t in zip(E, tot) if t]) for k in CATS}
        pool = {k: sum(e[k] for e in E) / max(sum(tot), 1) for k in CATS}
        return ep, pool, len(E)
    head = f'{"":34s}{"n":>5s}' + ''.join(f'{k[:11]:>12s}' for k in CATS)
    for a in ('true', 'mis'):
        L += [f'## reference = {a}: share of pixel errors (episode mean; pooled in brackets on the next line)', head]
        groups = [('catastrophic', [r for r in R if r['cat']]), ('non-catastrophic', [r for r in R if not r['cat']])]
        if a == 'true':
            groups += [(f'  cat, W45 {k}', [r for r in R if r['cat'] and r['w45'] == k]) for k in ('representation', 'readout', 'transfer', 'other')]
            groups += [(f'  cat, {d}', [r for r in R if r['cat'] and r['dataset'] == d]) for d in c.DOMAINS]
        else:
            groups = [(t, [r for r in s if 'mis' in r]) for t, s in groups]
            groups += [('catastrophic, same episodes, true ref', [r for r in R if r['cat'] and 'mis' in r])]
        for tag, sel in groups:
            aa = 'true' if tag.endswith('true ref') else a
            ep, pool, n = shares(sel, aa)
            if not n:
                continue
            L.append(f'{tag:34s}{n:5d}' + ''.join(f'{ep[k]:12.1%}' for k in CATS))
            if not tag.startswith('  cat, ') or tag.startswith('  cat, W45'):
                L.append(f'{"   pooled":34s}{"":5s}' + ''.join(f'{"(" + format(pool[k], ".1%") + ")":>12s}' for k in CATS))
        L.append('')
    for a in ('true', 'mis'):
        for tag, sel in (('catastrophic', [r for r in R if r['cat']]), ('non-catastrophic', [r for r in R if not r['cat']])):
            A = {k: [r['auc_' + a][k] for r in sel if 'auc_' + a in r and r['auc_' + a][k] is not None] for k in ('A_F', 'A_B', 'A_D')}
            L.append(f'AUC {a:4s} {tag:17s} median ' + '  '.join(f'{k} {np.median(v):.3f}' for k, v in A.items()) + f'  (n {len(A["A_F"])})')
    cat = [r for r in R if r['cat']]
    ep, _, _ = shares(cat, 'true')
    six = sorted(((ep[k], k) for k in CATS if k != 'mixed'), reverse=True)
    epm, _, _ = shares([r for r in cat if 'mis' in r], 'mis'); ept, _, _ = shares([r for r in cat if 'mis' in r], 'true')
    L += ['', f'reading 2: largest non-mixed category in catastrophic (true): {six[0][1]} {six[0][0]:.1%}, second {six[1][1]} {six[1][0]:.1%}'
          f' -> {"tie" if six[0][0] - six[1][0] < .05 else "single"}',
          f'reading 4: neither share, catastrophic with a mismatched class: true {ept["neither"]:.1%} vs mis {epm["neither"]:.1%}',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w51_competition_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ('cat', 'w45', 'seconds')}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
