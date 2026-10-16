"""W66 independent branch (sub-agent): look at attached regions on G, pre-register <= 5 GT-free region
quantities, test them once on the X half.

Plan: docs/research/x1012_tasks/W66_region_attribution_PLAN.md.  Regions, token arrays and labels are the main
session's (scripts/x1012/w66_regions.py, ../cv_data/a/x1013_W66/{regions,labels}); this file only reads them.
The main session's report.txt / results.json / quantities.jsonl are never read here.

Rules fixed before any board was drawn (this docstring was written first):
  sample    the 1,970 W65 G-field episodes (S half), W65 photo split (V viewing / X test).  T half and the
            new-photo holdout are never read.
  regions   OUT (attached band .15 <= G64 <= .5) and IN (N minus core K), as in w66_regions.masks().
  label     region member <=> mean token GT coverage >= .5 over labelled tokens; region counted iff n_lab >= 4
            (labels.jsonl).  GT is used only to group the boards and to score.
  boards    V half only.  rng = np.random.default_rng(2026101666); pools sorted by index string; draw, in this order,
            20 OUT member, 20 OUT non-member, 10 IN member, 10 IN non-member (without replacement within a pool).
            Row: reference + mask contour | query + GT contour | G heat (0-1) with K white, IN cyan, OUT orange |
            query RGB with the same K/IN/OUT contours (GT-free view) | c32 (reference fg posterior, 0-1) |
            log_e (reference explanation strength, per-episode min-max).
  explore   V half only; every candidate quantity tried is logged in explore_V.txt.
  measure   only after ../cv_data/a/x1013_W66/agent/preregistration.md exists; X half, one round, no changes.

    nice -n 19 python -B scripts/x1012/w66_agent.py select
    nice -n 19 python -B scripts/x1012/w66_agent.py boards
    nice -n 19 python -B scripts/x1012/w66_agent.py explore     # V half only
    nice -n 19 python -B scripts/x1012/w66_agent.py measure     # after preregistration; X half
"""
import json
import os
import sys
from pathlib import Path

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

W66 = c.OUT / 'x1013_W66'
OUT = W66 / 'agent'
EPISODES = c.OUT / 'x1013_W65/boards/episodes.jsonl'
SEED = 2026101666
CELL = 224


def episodes():
    return {e['index']: e for e in map(json.loads, EPISODES.read_text().splitlines())}


def labels():
    return {r['index']: r for r in map(json.loads, (W66 / 'labels.jsonl').read_text().splitlines())}


def bindings(want):
    return {str(b['index']): b for b in c.rows() if c.half(b) == 'S' and str(b['index']) in want}


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    E, L = episodes(), labels()
    rng = np.random.default_rng(SEED)
    sel = []
    for name, member, k in (('OUT', True, 20), ('OUT', False, 20), ('IN', True, 10), ('IN', False, 10)):
        pool = sorted((i for i in E if E[i]['split'] == 'V' and L[i][name]['n_lab'] >= 4
                       and (L[i][name]['cov'] >= .5) == member), key=str)
        pick = sorted(rng.choice(len(pool), k, replace=False))
        sel += [{'index': pool[j], 'region': name, 'member': member, 'dataset': E[pool[j]]['dataset'],
                 'cov': L[pool[j]][name]['cov'], 'n_lab': L[pool[j]][name]['n_lab'], 'pool': len(pool)} for j in pick]
    (OUT / 'selection.json').write_text(json.dumps(sel, indent=1))
    print(len(sel), flush=True)


# ---------------------------------------------------------------- drawing
def small(x, size=CELL):
    from PIL import Image
    return np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                      np.float64) / 255


def edge(mask):
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def up_tok(m64, size=CELL):
    return np.kron(m64.astype(np.uint8), np.ones((size // 64, size // 64), np.uint8)).astype(bool) if size % 64 == 0 \
        else small(m64.astype(np.float64), size) > .5


def down(mask, size=CELL):
    return small(mask.astype(np.float64), size) > .5


def heat(values, gray, cmap):
    import matplotlib
    return .78 * matplotlib.colormaps[cmap](np.clip(values, 0, 1))[..., :3] + .22 * gray[..., None]


def draw_regions(img, z):
    for name, col in (('OUT', (1, .55, 0)), ('IN', (0, .9, 1)), ('K', (1, 1, 1))):
        img[edge(up_tok(z[name]))] = col
    return img


def board_row(s, b):
    import w42_ordered_cut as w42
    from PIL import Image, ImageDraw, ImageFont
    with np.load(W66 / 'regions' / (s['index'] + '.npz')) as a:
        z = {k: a[k] for k in a.files}
    fields = c.load_fields(b)
    gt, valid = c.truth(b)
    rm = c.rc.lab.mask1024(b['row']['reference_mask_path']).numpy() > .5
    rr = small(w42.rgb(b['row']['reference_path'], b['row'].get('reference_crop')))
    qq = small(w42.rgb(b['row']['query_path'], b['row'].get('query_crop')))
    gray = qq.mean(2)
    c1 = rr.copy(); m = down(rm); c1[m] = .8 * c1[m] + .2 * np.array([1, .8, 0]); c1[edge(m)] = (1, .8, 0)
    c2 = qq.copy(); m = down(gt & valid); c2[m] = .8 * c2[m] + .2 * np.array([0, .9, 0]); c2[edge(m)] = (0, .9, 0)
    c3 = draw_regions(heat(small(np.kron(z['G64'], np.ones((16, 16))).clip(0, 1)), gray, 'viridis'), z)
    c4 = draw_regions(qq.copy(), z)
    c5 = draw_regions(heat(small(np.kron(fields['c32'].reshape(64, 64), np.ones((16, 16)))), gray, 'magma'), z)
    le = fields['log_e'].astype(np.float64).reshape(64, 64)
    le = (le - le.min()) / max(le.max() - le.min(), 1e-9)
    c6 = draw_regions(heat(small(np.kron(le, np.ones((16, 16)))), gray, 'cividis'), z)
    img = np.concatenate([c1, c2, c3, c4, c5, c6], 1)
    head = Image.new('RGB', (img.shape[1], 36), (255, 255, 255))
    font = ImageFont.load_default(size=13)
    R = z[s['region']]
    tag = f"{s['region']} {'MEMBER' if s['member'] else 'NOT'}"
    t1 = (f"{tag} | {s['dataset']} idx {s['index']} | region cov {s['cov']:.2f} n_lab {s['n_lab']} | "
          f"|K| {int(z['K'].sum())} |IN| {int(z['IN'].sum())} |OUT| {int(z['OUT'].sum())}")
    t2 = (f"G mean in region {float(z['G64'][R].mean()):.2f} | ref mask {100 * rm.mean():.1f}% | "
          f"cols: ref | query+GT | G (K white, IN cyan, OUT orange) | query | c32 | log_e")
    d = ImageDraw.Draw(head)
    d.text((4, 2), t1, fill=(0, 0, 0), font=font)
    d.text((4, 19), t2, fill=(0, 0, 0), font=font)
    return np.concatenate([np.asarray(head), (img * 255).astype(np.uint8)], 0)


def boards():
    from PIL import Image
    c.initialize()
    sel = json.loads((OUT / 'selection.json').read_text())
    B = bindings({s['index'] for s in sel})
    (OUT / 'boards').mkdir(exist_ok=True)
    groups = {}
    for s in sel:
        groups.setdefault(f"{s['region']}_{'member' if s['member'] else 'not'}", []).append(s)
    order = []
    for g, L in groups.items():
        for k in range(0, len(L), 5):
            rows = [board_row(s, B[s['index']]) for s in L[k:k + 5]]
            name = f'{g}_{k // 5 + 1:02d}.png'
            Image.fromarray(np.concatenate(rows, 0)).save(OUT / 'boards' / name)
            order += [{'board': name, 'index': s['index'], 'dataset': s['dataset']} for s in L[k:k + 5]]
            print(name, flush=True)
    (OUT / 'boards' / 'board_order.json').write_text(json.dumps(order, indent=1))


# ---------------------------------------------------------------- GT-free region quantities
# Every candidate tried on the V half is listed here (CANDIDATES) and logged in explore_V.txt.
# Sign: +1 means larger value predicts "member".
CANDIDATES = {
    'B0': 1,     # baseline from the plan: mean G64 over the region
    'A1': 1,     # rim analogy: max sim to reference INNER rim - max sim to reference OUTER rim
    'A2': 1,     # max sim to reference fg - max sim to reference bg (all)
    'A4': -1,    # max sim to reference outer rim - max sim to reference fg (all)
    'A5': 1,     # region best fg match - core best fg match
    'C1': 1,     # core's absolute best fg match (episode-level)
    'B1': 1,     # query edge location: adjacent-pair cosine (inner side) - (outer side)
    'B2': 1,     # prototype: cos(region, core) - cos(region, outer ring of region)
    'B3': 1,     # colour edge location: Lab distance (outer side) - (inner side)
}


def features(b):
    lab = c.rc.lab
    r = lab.debias(c.rc.raw(b['fit']['reference_raw'], {}))
    q = lab.debias(c.rc.raw(b['fit']['query_raw_identity'], {}))
    mu = .5 * (r.mean(0) + q.mean(0))
    return lab.unit(r - mu).numpy().astype(np.float64), lab.unit(q - mu).numpy().astype(np.float64), \
        q.numpy().astype(np.float64)


def reference_sets(b):
    from scipy import ndimage
    y = c.rc.lab.coverage64(c.rc.lab.mask1024(b['row']['reference_mask_path'])).numpy().astype(np.float64)
    M = (y >= min(.5, y.max())) & (y > 0)
    RI = M & ~ndimage.binary_erosion(M, iterations=2, border_value=1)
    if not RI.any():
        RI = M
    RO = ndimage.binary_dilation(M, iterations=3) & ~M & (y <= .1)
    for alt in ((~M) & (y <= .1), ~M):
        if not RO.any():
            RO = alt
    BG = (~M) & (y <= .1)
    if not BG.any():
        BG = ~M
    return {'FG': M.ravel(), 'RI': RI.ravel(), 'RO': RO.ravel(), 'BG': BG.ravel()}


def pairs(A, B):
    """4-adjacent token pairs (a in A, b in B) on the 64 grid, as flat index arrays."""
    idx = np.arange(4096).reshape(64, 64)
    P, Q = [], []
    for sa, sb in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        for x, y in ((sa, sb), (sb, sa)):
            m = A[x] & B[y]
            P.append(idx[x][m]); Q.append(idx[y][m])
    return np.concatenate(P), np.concatenate(Q)


def region_values(Z, inner, outer, K, z, rr, qr, qq, refs, core_fg):
    """Candidate quantities for one region Z; inner/outer = the token sets on its core side / far side."""
    from scipy import ndimage
    zf = Z.ravel()
    S = qr[zf] @ rr.T
    mx = lambda s: S[:, refs[s]].max(1) if refs[s].any() else np.full(len(S), np.nan)
    v = {'B0': float(z['G64'][Z].mean())}
    v['A1'] = float(np.nanmean(mx('RI') - mx('RO')))
    v['A2'] = float(np.nanmean(mx('FG') - mx('BG')))
    v['A4'] = float(np.nanmean(mx('RO') - mx('FG')))
    v['A5'] = float(np.nanmean(mx('FG')) - core_fg)
    v['C1'] = core_fg
    a, b_ = pairs(inner, Z)
    a2, b2 = pairs(Z, outer)
    cosp = lambda p, q_: (qq[p] * qq[q_]).sum(1)
    v['B1'] = float(cosp(a, b_).mean() - cosp(a2, b2).mean()) if len(a) and len(a2) else 0.0
    ring = ndimage.binary_dilation(Z, iterations=2) & outer
    unitv = lambda x: x / max(np.linalg.norm(x), 1e-12)
    pz = unitv(qq[zf].mean(0))
    v['B2'] = float(pz @ unitv(qq[K.ravel()].mean(0)) - pz @ unitv(qq[ring.ravel()].mean(0))) if ring.any() else 0.0
    L = z['lab']
    dl = lambda p, q_: np.linalg.norm(L[p] - L[q_], axis=1)
    v['B3'] = float(dl(a2, b2).mean() - dl(a, b_).mean()) if len(a) and len(a2) else 0.0
    return v


def one_episode(e, b, names):
    with np.load(W66 / 'regions' / (e['index'] + '.npz')) as a:
        z = {k: a[k] for k in a.files}
    K, N = z['K'], z['N']
    rec = {'index': e['index'], 'dataset': e['dataset'], 'photo': e['photo'], 'split': e['split']}
    if not K.any():
        return dict(rec, IN=None, OUT=None)
    rr, qr, qq = features(b)
    refs = reference_sets(b)
    core_fg = float((qr[K.ravel()] @ rr[refs['FG']].T).max(1).mean())
    for name, Z, inner, outer in (('IN', z['IN'], K, ~N), ('OUT', z['OUT'], N, ~N & ~z['OUT'])):
        if Z.sum() < 4:
            rec[name] = None
            continue
        v = region_values(Z, inner, outer, K, z, rr, qr, qq, refs, core_fg)
        rec[name] = {k: v[k] for k in names}
    return rec


def run(split, names, path):
    c.initialize()
    E = episodes()
    want = sorted((i for i in E if E[i]['split'] == split), key=str)
    B = bindings(set(want))
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open('a') as fh:
        for n, i in enumerate(want, 1):
            if i in done:
                continue
            fh.write(json.dumps(one_episode(E[i], B[i], names)) + '\n'); fh.flush()
            if n % 100 == 0:
                print(split, n, len(want), flush=True)


def auc(x, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(x, y, photos, reps=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    P = np.unique(photos)
    idx = {p: np.flatnonzero(photos == p) for p in P}
    out = []
    for _ in range(reps):
        s = np.concatenate([idx[p] for p in rng.choice(P, len(P))])
        out.append(auc(x[s], y[s]))
    return np.nanpercentile(out, [2.5, 97.5])


def table(path, names, signs, title, ci=True):
    from scipy.stats import rankdata
    Q = {r['index']: r for r in map(json.loads, path.read_text().splitlines())}
    L = labels()
    lines = [title]
    res = {}
    for reg in ('OUT', 'IN'):
        R = [(Q[k], L[k]) for k in sorted(Q, key=str) if Q[k][reg] is not None and L[k][reg]['n_lab'] >= 4]
        y = np.array([l[reg]['cov'] >= .5 for _, l in R])
        dom = np.array([q['dataset'] for q, _ in R])
        ph = np.array([q['photo'] for q, _ in R])
        valid = [d for d in sorted(set(dom)) if min(y[dom == d].sum(), (~y[dom == d]).sum()) >= 5]
        lines.append(f'\n## {reg}: regions {len(R)}, member {int(y.sum())}; valid domains: {", ".join(valid)}')
        lines.append(f'{"q":4s} {"AUC [95% CI]":>22s} {"noSUIM":>7s} {"dom+":>6s}  per-domain')
        res[reg] = {'n': len(R), 'member': int(y.sum()), 'valid_domains': valid}
        for m in names:
            x = np.nan_to_num(np.array([signs[m] * q[reg][m] for q, _ in R], np.float64))  # empty ref set -> 0
            a = auc(x, y)
            lo, hi = boot(x, y, ph) if ci else (np.nan, np.nan)
            ns = dom != 'suim'
            an = auc(x[ns], y[ns])
            per = {d: auc(x[dom == d], y[dom == d]) for d in valid}
            plus = sum(v > .5 for v in per.values())
            lines.append(f'{m:4s} {a:6.3f} [{lo:.3f},{hi:.3f}] {an:7.3f} {plus:2d}/{len(per):<3d}  '
                         + ' '.join(f'{d[:5]}={v:.2f}' for d, v in per.items()))
            res[reg][m] = {'auc': a, 'ci': [float(lo), float(hi)], 'auc_noSUIM': an, 'per_domain': per,
                           'domains_plus': plus}
    return '\n'.join(lines), res


def explore():
    path = OUT / 'explore_V.jsonl'
    run('V', list(CANDIDATES), path)
    text, _ = table(path, list(CANDIDATES), CANDIDATES,
                    f'V half exploration; candidates tried: {len(CANDIDATES) - 1} (+ baseline B0)', ci=True)
    (OUT / 'explore_V.txt').write_text(text + '\n')
    print(text)


def explore_extra():
    """V half only: the further candidates looked at after explore() (rank-mean combos, two shape descriptors,
    AUC within B0 quintiles).  Appended to explore_V.txt so every quantity looked at on V is on record."""
    from scipy import ndimage
    from scipy.stats import rankdata, spearmanr
    R = {r['index']: r for r in map(json.loads, (OUT / 'explore_V.jsonl').read_text().splitlines())}
    L, E = labels(), episodes()
    assert all(E[k]['split'] == 'V' for k in R)
    lines = ['\n# further candidates looked at on V (after the table above)']
    combos = {'OUT': [('B0', 'A1'), ('B0', '-C1'), ('B0', 'B1'), ('B0', 'A1', '-C1'), ('B0', 'A1', 'B1'),
                      ('B0', 'A1', 'B1', '-C1'), ('A1', 'B1'), ('A1', 'B1', '-C1')],
              'IN': [('B0', 'A2'), ('B0', 'C1'), ('B0', 'A1'), ('B0', 'C1', 'A2'), ('B0', 'C1', 'B2'), ('A2', 'C1'),
                     ('A1', 'C1', 'B2')]}
    n_extra = 0
    for reg in ('OUT', 'IN'):
        keys = [k for k in sorted(R, key=str) if R[k][reg] is not None and L[k][reg]['n_lab'] >= 4]
        y = np.array([L[k][reg]['cov'] >= .5 for k in keys])
        X = {m: np.nan_to_num(np.array([R[k][reg][m] for k in keys], float)) for m in CANDIDATES}
        lines.append(f'\n## {reg}: rank-mean combinations (sign-adjusted; "-" flips the listed sign)')
        for cb in combos[reg]:
            s = np.mean([rankdata((-1 if m[0] == '-' else 1) * CANDIDATES[m.lstrip('-')] * X[m.lstrip('-')])
                         for m in cb], 0)
            lines.append(f'  {"+".join(cb):18s} AUC {auc(s, y):.3f}')
            n_extra += 1
        q = np.quantile(X['B0'], [.2, .4, .6, .8]); st = np.digitize(X['B0'], q)
        lines.append(f'## {reg}: AUC within B0 quintiles (pooled), Spearman with B0')
        for m in CANDIDATES:
            x = CANDIDATES[m] * X[m]
            num = den = 0.
            for g in range(5):
                mm = st == g
                n1, n0 = y[mm].sum(), (~y[mm]).sum()
                if n1 and n0:
                    num += auc(x[mm], y[mm]) * n1 * n0; den += n1 * n0
            lines.append(f'  {m}: within-B0 AUC {num / den:.3f}  rho(B0) {spearmanr(X["B0"], x)[0]:+.2f}')
    # two OUT shape descriptors
    rows = []
    for k in sorted(R, key=str):
        with np.load(W66 / 'regions' / (k + '.npz')) as z:
            Z, N = z['OUT'], z['N']
            if Z.sum() < 4 or L[k]['OUT']['n_lab'] < 4:
                continue
            rows.append(((ndimage.binary_dilation(N) & Z).sum() / Z.sum(), np.log(Z.sum() / N.sum()),
                         L[k]['OUT']['cov'] >= .5))
    ring, size, y = map(np.array, zip(*rows))
    lines.append('\n## OUT shape descriptors')
    lines.append(f'  ringness (share of OUT tokens touching N), sign -: AUC {auc(-ring, y):.3f}; '
                 f'share with ringness < .3: {np.mean(ring < .3):.2f}')
    lines.append(f'  log |OUT|/|N|, sign -: AUC {auc(-size, y):.3f}')
    n_extra += 2
    lines.append(f'\ncandidate quantities looked at on V in total: {len(CANDIDATES) - 1} single + {n_extra} further = '
                 f'{len(CANDIDATES) - 1 + n_extra} (B0 is the plan baseline, not counted)')
    with (OUT / 'explore_V.txt').open('a') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ---------------------------------------------------------------- pre-registered test (X half, one round)
# Fixed in ../cv_data/a/x1013_W66/agent/preregistration.md before measure() first ran.
PREREG = ('A1', 'A2', 'B1', 'C1', 'K1')
PREREG_SIGN = {'OUT': {'B0': 1, 'A1': 1, 'A2': 1, 'B1': 1, 'C1': -1, 'K1': 1},
               'IN': {'B0': 1, 'A1': 1, 'A2': 1, 'B1': 1, 'C1': 1, 'K1': 1}}


def measure():
    from scipy.stats import rankdata
    assert (OUT / 'preregistration.md').exists(), 'pre-register first'
    path = OUT / 'measure_X.jsonl'
    run('X', ['B0', 'A1', 'A2', 'B1', 'C1'], path)
    Q = [json.loads(l) for l in path.read_text().splitlines()]
    L = labels()
    assert all(q['split'] == 'X' for q in Q)
    # K1 = rank mean of B0 and A1 over the scored regions of each type (computed per region type)
    for reg in ('OUT', 'IN'):
        S = [q for q in Q if q[reg] is not None and L[q['index']][reg]['n_lab'] >= 4]
        k1 = np.mean([rankdata(np.nan_to_num(np.array([q[reg][m] for q in S], float))) for m in ('B0', 'A1')], 0)
        for q, v in zip(S, k1):
            q[reg]['K1'] = float(v)
    tmp = OUT / 'measure_X_with_K1.jsonl'
    tmp.write_text(''.join(json.dumps(q) + '\n' for q in Q))
    out, res = [], {}
    for reg in ('OUT', 'IN'):
        text, r = table_one(tmp, reg, ('B0',) + PREREG, PREREG_SIGN[reg])
        out.append(text)
        b0 = r['B0']['auc']
        nv = len(r['valid_domains'])
        r['passed'] = [m for m in PREREG if r[m]['auc'] >= .75 and r[m]['auc'] >= b0 + .05
                       and r[m]['domains_plus'] > nv / 2]
        out.append(f"gate (AUC >= .75, >= B0 + .05, majority of valid domains): {r['passed'] or 'none'}")
        res[reg] = r
    text = 'X half, pre-registered quantities, one round\n' + '\n'.join(out)
    (OUT / 'measure_X.txt').write_text(text + '\n')
    (OUT / 'measure_X.json').write_text(json.dumps(res, indent=1))
    print(text)


def table_one(path, reg, names, signs):
    Q = {r['index']: r for r in map(json.loads, path.read_text().splitlines())}
    L = labels()
    R = [(Q[k], L[k]) for k in sorted(Q, key=str) if Q[k][reg] is not None and L[k][reg]['n_lab'] >= 4]
    y = np.array([l[reg]['cov'] >= .5 for _, l in R])
    dom = np.array([q['dataset'] for q, _ in R])
    ph = np.array([q['photo'] for q, _ in R])
    valid = [d for d in sorted(set(dom)) if min(y[dom == d].sum(), (~y[dom == d]).sum()) >= 5]
    lines = [f'\n## {reg}: regions {len(R)}, member {int(y.sum())}; photos {len(set(ph))}; valid domains: {", ".join(valid)}',
             f'{"q":4s} {"AUC [95% CI]":>22s} {"noSUIM":>7s} {"dom+":>6s}  per-domain']
    res = {'n': len(R), 'member': int(y.sum()), 'valid_domains': valid}
    for m in names:
        x = np.nan_to_num(np.array([signs[m] * q[reg][m] for q, _ in R], np.float64))
        a = auc(x, y)
        lo, hi = boot(x, y, ph)
        ns = dom != 'suim'
        an = auc(x[ns], y[ns])
        per = {d: auc(x[dom == d], y[dom == d]) for d in valid}
        plus = sum(v > .5 for v in per.values())
        lines.append(f'{m:4s} {a:6.3f} [{lo:.3f},{hi:.3f}] {an:7.3f} {plus:2d}/{len(per):<3d}  '
                     + ' '.join(f'{d[:5]}={v:.2f}' for d, v in per.items()))
        res[m] = {'auc': a, 'ci': [float(lo), float(hi)], 'auc_noSUIM': an, 'per_domain': per, 'domains_plus': plus}
    return '\n'.join(lines), res


if __name__ == '__main__':
    {'select': select, 'boards': boards, 'explore': explore, 'explore_extra': explore_extra,
     'measure': measure}[sys.argv[1]]()
