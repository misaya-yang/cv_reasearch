"""W42: reference-calibrated edge conductance + F-prefix chain min-cut in the boundary band (W39 D section 4).  S half only.
Plan and readings: docs/research/x1012_tasks/W42_ordered_cut_PLAN.md.  Arms (W39 D section 7.1): N natural, R eq. (7)-(10),
G generic RGB conductance with matched total, Uc = R's energy without the prefix chain, K = R's per-block counts placed
by (Uc label, F, coordinate).  R+ = CRF(R) as SH+ (fallback without CRF).

    nice -n 19 python scripts/x1012/w42_ordered_cut.py produce [workers]
    python scripts/x1012/w42_ordered_cut.py report
"""
import gzip, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W42')
ARMS = ('N', 'R', 'G', 'Uc', 'K')
REG = ('C0', 'C1', 'D', 'fb')
ETA, LAM, MARGIN = 1 / 16, 1 / 4, 2.0 ** -20
TOP = 2 ** 30                                   # integer capacity budget (scipy max-flow uses int32)


def capture(b):
    """F (pre-CRF field of the frozen SH, fallback = minmax(up(c32))), N, the two actual line fields, rho."""
    import torch
    import w09_integration_lambda as w09, w19_simple_head as w19, w23_simple_pipeline as sh
    cap = {'U0': [], 'M': [], 'ind': []}
    ou, om, oi = w09.solve, w19.mean_solve, c.rc.independent

    def point(*a, **k):
        v = ou(*a, **k); cap['U0'].append(np.asarray(v).copy()); return v

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
    if rho > .9:
        f = cap['ind'][0][4]
        F = lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))).numpy().astype(np.float64)
        U = M = None
    else:
        U = lab.minmax(lab.up(torch.from_numpy(cap['U0'][0].reshape(64, 64).astype(np.float32)))).numpy().astype(np.float64)
        M = lab.up(torch.from_numpy(cap['M'][0])).numpy().astype(np.float64)
        F = .5 * M + .5 * U
    assert np.array_equal(F > .5, pre), 'captured field does not reproduce SH'
    return F, pre, U, M, rho


def rgb(path, crop):
    from PIL import Image
    from ics.official_data import decoded_rgb
    return np.asarray(decoded_rgb(c.ASSETS / path, crop).resize((1024, 1024), Image.BILINEAR), np.float64) / 255


def edges(shape):
    """4-neighbour edges of a 1024x1024 grid as (p, q) flat index arrays (right, then down)."""
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    return (np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]),
            np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()]))


def bins(img, p, q):
    g = np.abs(img.reshape(-1, 3)[p] - img.reshape(-1, 3)[q])
    k = np.minimum((g * 4).astype(np.int64), 3)
    return k[:, 0] * 16 + k[:, 1] * 4 + k[:, 2]


def reference_model(b, P, Q):
    """Eq. (7): per-class (FG-FG, BG-BG, cross) 64-bin histograms of reference neighbour RGB differences."""
    row = b['row']
    img = rgb(row['reference_path'], row.get('reference_crop'))
    m = c.rc.lab.mask1024(row['reference_mask_path']).numpy().ravel()
    bb = bins(img, P, Q)
    mp, mq = m[P], m[Q]
    cls = {'ff': mp & mq, 'bb': ~mp & ~mq, 'x': mp != mq}
    h = {k: np.bincount(bb[v], minlength=64).astype(np.float64) for k, v in cls.items()}
    n = {k: int(v.sum()) for k, v in h.items()}
    if min(n.values()) == 0:
        return None, n, h
    p = {k: (1 - ETA) * h[k] / n[k] + ETA / 64 for k in h}
    same, cross = .5 * (p['ff'] + p['bb']), p['x']
    r = np.maximum(0, np.log(same / cross))
    return r / (1 + r), n, h


def solve(Bpix, N, a, ep, eq, w, chain):
    """Exact s-t min cut of eq. (9) with optional prefix chain (10); returns the binary mask (minimal source side).
    Capacities are scaled to integers; returns (mask, scale) or (None, scale) when the resolution would be < 1/256."""
    nb = np.flatnonzero(Bpix.ravel())
    P = len(nb)
    pos = np.full(Bpix.size, -1); pos[nb] = np.arange(P)
    Nf = N.ravel()
    s_cap, t_cap = np.where(Nf[nb], a, 0.), np.where(Nf[nb], 0., a)
    ip, iq = pos[ep], pos[eq]
    lw = LAM * w
    inner = (ip >= 0) & (iq >= 0)
    for x, y in ((ip, eq), (iq, ep)):                     # one endpoint in the band, the other fixed to N
        sel = (x >= 0) & (pos[y] < 0)
        np.add.at(s_cap, x[sel], np.where(Nf[y[sel]], lw[sel], 0.))
        np.add.at(t_cap, x[sel], np.where(Nf[y[sel]], 0., lw[sel]))
    finite = s_cap.sum() + t_cap.sum() + 2 * lw[inner].sum()
    scale = min(4096., TOP / (finite + 1))
    if scale < 256:
        return None, scale
    q = lambda v: np.maximum(np.rint(v * scale), 0).astype(np.int64)
    S, T = P, P + 1
    rows = [np.full(P, S), np.arange(P), ip[inner], iq[inner]]
    cols = [np.arange(P), np.full(P, T), iq[inner], ip[inner]]
    caps = [q(s_cap), q(t_cap), q(lw[inner]), q(lw[inner])]
    big = int(sum(int(x.sum()) for x in caps)) + 1
    if chain is not None:
        lo, hi = chain                                    # arcs lower-ranked -> higher-ranked (pos indices)
        rows.append(lo); cols.append(hi); caps.append(np.full(len(lo), big, np.int64))
    C = sparse.coo_matrix((np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))), shape=(P + 2, P + 2)).tocsr()
    C.sum_duplicates()
    assert C.data.max() < 2 ** 31 - 1
    C = C.astype(np.int32)
    flow = maximum_flow(C, S, T).flow
    R = (C.astype(np.int64) - flow.astype(np.int64)).tocsr()
    R.data = (R.data > 0).astype(np.int8); R.eliminate_zeros()
    reach = breadth_first_order(R, S, directed=True, return_predecessors=False)
    m = Nf.copy()
    m[nb] = False
    m[nb[reach[reach < P]]] = True
    return m.reshape(N.shape), scale


def one(b):
    import torch
    import produce
    import w31_demonstration as w31
    import w37_coverage as w37
    import w38_band_amount as w38
    began = time.monotonic()
    F, N, U, M, rho = capture(b)
    fb = U is None
    B = w37.band(w37.pool16(N.astype(np.float64)))
    Bpix = np.kron(B, np.ones((16, 16), bool))
    P, Q = edges(N.shape)
    touch = Bpix.ravel()[P] | Bpix.ravel()[Q]
    ep, eq = P[touch], Q[touch]
    wR, n_ref, h = reference_model(b, P, Q)
    row = b['row']
    qimg = rgb(row['query_path'], row.get('query_crop'))
    r = dict(w31.meta(b), rho=rho, fallback=fb, band_pixels=int(Bpix.sum()), ref_edges=n_ref)
    out = {'N': N}
    status = 'ok'
    if not Bpix.any():
        status = 'no_band'
    elif wR is None:
        status = 'ref_class_empty'
    if status == 'ok':
        qb = bins(qimg, ep, eq)
        seen = (h['ff'] + h['bb'] + h['x']) > 0
        r['unseen_bin_frac'] = float(np.mean(~seen[qb]))
        w = wR[qb]
        d = np.linalg.norm(qimg.reshape(-1, 3)[ep] - qimg.reshape(-1, 3)[eq], axis=1)
        sig = np.median(d[d > 0]) if (d > 0).any() else None
        v = np.exp(-d ** 2 / (2 * sig ** 2)) if sig is not None else np.ones_like(d)
        wG = v * (w.sum() / v.sum()) if w.sum() > 0 else np.zeros_like(v)
        a = np.abs(2 * F.ravel()[np.flatnonzero(Bpix.ravel())] - 1) + MARGIN
        # prefix chain: within each band block, F-descending stable order (same as W38 place)
        fbk, (ii, jj) = w38.blocks(F, B)
        order = np.argsort(-fbk, axis=1, kind='stable')
        loc = np.arange(256).reshape(16, 16)
        flat = np.stack([((16 * i + loc // 16) * 1024 + 16 * j + loc % 16).ravel() for i, j in zip(ii, jj)])
        ranked = np.take_along_axis(flat, order, axis=1)
        pos = np.full(N.size, -1); pos[np.flatnonzero(Bpix.ravel())] = np.arange(int(Bpix.sum()))
        chain = (pos[ranked[:, 1:]].ravel(), pos[ranked[:, :-1]].ravel())
        t0 = time.monotonic()
        for arm, ww, ch in (('R', w, chain), ('G', wG, chain), ('Uc', w, None)):
            m, scale = solve(Bpix, N, a, ep, eq, ww, ch)
            r['scale_' + arm] = scale
            out[arm] = m if m is not None else N
        r['cut_seconds'] = round(time.monotonic() - t0, 2)
        kR = w38.blocks(out['R'][..., None], B)[0][..., 0].sum(1)
        # R must be an F prefix in every band block
        rank = np.empty_like(order); rank[np.arange(len(order))[:, None], order] = np.arange(256)[None, :]
        Rb = w38.blocks(out['R'][..., None], B)[0][..., 0]
        r['prefix_violations'] = int((Rb != (rank < kR[:, None])).sum())
        ub = w38.blocks(out['Uc'][..., None], B)[0][..., 0].astype(np.int64)
        idx = np.broadcast_to(np.arange(256), ub.shape)
        kord = np.lexsort((idx, -fbk, -ub), axis=1)
        sel = np.zeros_like(ub, bool)
        np.put_along_axis(sel, kord, np.arange(256)[None, :] < kR[:, None], axis=1)
        K = N.copy()
        for t, (i, j) in enumerate(zip(ii, jj)):
            K[16 * i:16 * i + 16, 16 * j:16 * j + 16] = sel[t].reshape(16, 16)
        out['K'] = K
    else:
        for arm in ARMS[1:]:
            out[arm] = N
    r['status'] = status
    if fb:
        out['R+'] = out['R']
    else:
        with c.rc.pc.inference_only(b):
            out['R+'] = produce.crf(b, out['R'])
        torch.set_num_threads(1)
    gt, valid = c.truth(b)                      # read only after every label-free output is fixed
    r['gt'] = int(gt.sum())
    for arm in ARMS + ('R+',):
        r[arm] = c.iu(out[arm], gt, valid)
    kt = w38.blocks(gt[..., None], B)[0][..., 0].sum(1) if B.any() else np.zeros(0)
    for arm in ARMS:
        k = w38.blocks(out[arm][..., None], B)[0][..., 0].sum(1) if B.any() else np.zeros(0)
        r['count_l1_' + arm] = int(np.abs(k - kt).sum())
    if fb:
        reg = {'C0': np.zeros_like(N), 'C1': np.zeros_like(N), 'D': np.zeros_like(N), 'fb': np.ones_like(N)}
    else:
        C0, C1 = (U <= .5) & (M <= .5), (U > .5) & (M > .5)
        reg = {'C0': C0, 'C1': C1, 'D': ~(C0 | C1), 'fb': np.zeros_like(N)}
    Nv = N & valid
    err_fn, err_fp = gt & ~Nv, Nv & ~gt
    for k, Rg in reg.items():
        r['err_' + k] = [int((err_fn & Rg & Bpix).sum()), int((err_fp & Rg & Bpix).sum()),
                         int((err_fn & Rg & ~Bpix).sum()), int((err_fp & Rg & ~Bpix).sum())]   # FN/FP in band, off band
    for arm in ('R', 'G', 'Uc', 'K'):
        Av = out[arm] & valid
        add, rem = Av & ~Nv, Nv & ~Av
        r['edit_' + arm] = {k: [int((add & gt & Rg).sum()), int((add & ~gt & Rg).sum()),
                                int((rem & gt & Rg).sum()), int((rem & ~gt & Rg).sum())] for k, Rg in reg.items()}
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


def attribution(D, arm):
    """W39 A eq. (6) for arm - N by region (slot-pooled, the arm's final denominator)."""
    slots = {}
    for r in D:
        s = slots.setdefault((r['fold'], r['cls']), {'I': 0, 'U': 0, **{k: np.zeros(4) for k in REG}})
        s['I'] += r['N'][0]; s['U'] += r['N'][1]
        for k in REG:
            s[k] += r['edit_' + arm][k]
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
    for r in R:
        r['SH+'] = w27[r['index']]['SH+']
    same = sum(r['N'][:2] == w23[r['index']]['SH'][:2] for r in R)
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    st = {s: sum(r['status'] == s for r in R) for s in ('ok', 'no_band', 'ref_class_empty')}
    ok = [r for r in R if r['status'] == 'ok']
    low = sum(any(r.get('scale_' + a, 4096) < 256 for a in ('R', 'G', 'Uc')) for r in ok)
    L = [f'n {len(R)}  checks: N == W23 SH {same}/{len(R)} (need >= 1280); prefix violations {sum(r.get("prefix_violations", 0) for r in R)} (need 0)',
         f'status {st}; integer resolution below 1/256 (default N) in {low} episodes; median scale R {np.median([r["scale_R"] for r in ok]):.0f}',
         f'unseen query bins median {np.median([r["unseen_bin_frac"] for r in ok]):.3f}; band pixels median {np.median([r["band_pixels"] for r in R]):.0f}; '
         f'cut seconds median {np.median([r["cut_seconds"] for r in ok]):.2f}, max {max(r["cut_seconds"] for r in ok):.1f}']
    fn = {k: np.sum([r['err_' + k] for r in R], axis=0) for k in REG}
    common_band = fn['C0'][0] + fn['C1'][1]; common_all = common_band + fn['C0'][2] + fn['C1'][3]
    L += ['', '## errors of N by region: FN in band, FP in band, FN off band, FP off band (M pixels)'] + \
         [f'{k:3s} ' + '  '.join(f'{x / 1e6:6.2f}' for x in fn[k]) for k in REG] + \
         [f'common errors (C0 FN + C1 FP) inside the band: {common_band / max(common_all, 1):.1%}']

    def ci(a, b_):
        v = np.mean([sc.miou(by[d], a) - sc.miou(by[d], b_) for d in doms])
        lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], a, b_) for d in doms], axis=0), [2.5, 97.5])
        pos = sum(sc.miou(by[d], a) - sc.miou(by[d], b_) >= 0 for d in doms)
        return v, lo, hi, pos
    cols = ('R', 'G', 'Uc', 'K')
    L += ['', '## class mIoU minus N (pre-CRF); last column R+ - SH+ (post-CRF)', f'{"":12s}' + ''.join(f'{a:>8s}' for a in cols) + '  R+-SH+']
    for d in doms:
        L.append(f'{d[:11]:12s}' + ''.join(f'{sc.miou(by[d], a) - sc.miou(by[d], "N"):+8.2f}' for a in cols) +
                 f'{sc.miou(by[d], "R+") - sc.miou(by[d], "SH+"):+8.2f}')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean([sc.miou(by[d], a) - sc.miou(by[d], "N") for d in doms]):+8.2f}' for a in cols) +
             f'{np.mean([sc.miou(by[d], "R+") - sc.miou(by[d], "SH+") for d in doms]):+8.2f}')
    cl = {a: sum(r['count_l1_' + a] for r in R) for a in ARMS}
    L += ['', 'band count error sum|k - s| relative to N: ' + '  '.join(f'{a} {cl[a] / max(cl["N"], 1):.3f}' for a in ARMS)]
    L += ['', '## attribution of arm - N by region (W39 A eq. 6), ten-domain', f'{"":6s}' + ''.join(f'{k:>8s}' for k in REG)]
    for a in cols:
        at = [attribution(by[d], a) for d in doms]
        L.append(f'{a:6s}' + ''.join(f'{np.mean([x[k] for x in at]):+8.2f}' for k in REG))
    c_rn, c_rg, c_un, c_gn, c_ur, c_kr, c_ps = (ci('R', 'N'), ci('R', 'G'), ci('Uc', 'N'), ci('G', 'N'), ci('Uc', 'R'),
                                                ci('K', 'R'), ci('R+', 'SH+'))
    fmt = lambda t, v: f'{t} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}] ({v[3]}/10 >= 0)'
    L += ['', 'reading 2: ' + '; '.join(fmt(t, v) for t, v in (('R-N', c_rn), ('R-G', c_rg), ('G-N', c_gn), ('Uc-N', c_un),
                                                                   ('Uc-R', c_ur), ('K-R', c_kr)))]
    if c_rn[1] > 0 and c_rg[1] > 0:
        verdict = 'a: reference edge relations beyond generic smoothing'
    elif c_rn[1] > 0 and c_gn[1] > 0 and c_rg[1] <= 0 <= c_rg[2]:
        verdict = 'b: generic image-guided smoothing'
    elif max(c_rn[0], c_gn[0], c_un[0]) <= 0:
        verdict = 'd: R, G, Uc all <= N -> close this energy / conductance construction'
    else:
        verdict = 'mixed'
    pos_c = c_ur[1] > 0 and c_kr[1] > 0
    L += [f'reading 2 verdict: {verdict}; prefix too restrictive (Uc-R and K-R both CI > 0): {pos_c}',
          f'reading 3: {fmt("R+ - SH+", c_ps)} -> method validation plan: {bool(c_ps[0] >= 1 and c_ps[1] > 0 and c_ps[3] >= 7)}',
          f'reading 4: common errors inside band {common_band / max(common_all, 1):.1%} -> '
          + ('band methods address common errors' if common_band >= .5 * common_all else 'common errors mostly off band (entity level)'),
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback', 'status', 'band_pixels', 'ref_edges',
            'unseen_bin_frac', 'cut_seconds', 'scale_R', 'scale_G', 'scale_Uc', 'prefix_violations') + ARMS + ('R+',) \
        + tuple('count_l1_' + a for a in ARMS) + tuple('err_' + k for k in REG) + tuple('edit_' + a for a in cols)
    with gzip.open(ROOT / 'evidence/research/w42_ordered_cut_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys if k in r}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
