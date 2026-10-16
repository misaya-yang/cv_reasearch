"""W65 branch B: why did the closed label-free cut cues fail?  Anatomy on MFU's pre-CRF fused field G.

Sample: W14's S-half sample (`w14_headroom.subset`, cap 300/domain) restricted to rho <= .9 (1,970 episodes).
G = .5 up(T1 `mean`) + .5 minmax(up(VP_UN)), natural mask G > .5, exactly W14 `one()`.  The GT-optimal level t*
and area A* follow the distinct-value-group rule of `diagnostic_metrics.exact_best_iou` (ties are never split).

Phases, run in this order; the preregistration is written after `cues` and before `truth`:
  cues    label-free.  Each cue's suggested level on G.  Opens no query label and no valid/ignore mask.
  truth   scoring only.  t*, A*, natural IoU, IoU at every cue level, IoU-vs-area curves, J-vs-IoU across levels.
  report  relations, query-photo-grouped bootstrap, the preregistered mechanical (a)-(d) classification.

    nice -n 19 python -B scripts/x1012/w65_cues.py cues|truth|report

Cue sources.  The archive scripts d05/d08/d10/d17/d20/d25 run their whole experiment at import, so their cue functions
are copied verbatim below (marked); only the field they are applied to (G) and the episode loop are new.  Exceptions,
marked "rewritten": the sweep cut and the J prefix return a level on G instead of a token mask, and the FoRIS area uses
minmax(up(score)) as specified for W65 (d25 used up(minmax(score))).  J uses the cached tau=.05 fields: `c32` equals
d19/d20's fwdF0.05 and `back_f` equals backF0.05 (same debias, same joint centring, same softmax; see r5 `independent`).
Outputs only under ../cv_data/a/x1013_W65/cues/.  S half only.  CPU, one process, two threads.
"""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c  # noqa: E402  (sets the BLAS thread variables to 2)
import torch  # noqa: E402

OUT = c.OUT / 'x1013_W65' / 'cues'
N = 4096
ALPHA = np.arange(0.04, 0.97, 0.02)      # d17 / d19 level grid
QS = torch.tensor([0.5, 0.6, 0.7, 0.76, 0.81, 0.85, 0.88, 0.905, 0.925, 0.94, 0.953, 0.963, 0.971, 0.978, 0.984,
                   0.989, 0.993, 0.996, 0.998])   # d08 quantile levels
LEVEL_CUES = ('otsu', 'otsu3_hi', 'otsu3_lo', 'mser', 'robust_mid', 'J', 'sweep', 'contrast', 'inverse')
AREA_CUES = ('nat_area', 'sumG', 'ref_area', 'foris_area', 'fwd_area', 'inv_simpson')
CUES = LEVEL_CUES + AREA_CUES
lab = None


# ---------------------------------------------------------------- verbatim copies (d05_extent_rules.py)
def otsu(s, bins=128):
    h = torch.histc(s, bins=bins, min=float(s.min()), max=float(s.max()))
    p = h / h.sum()
    c_ = torch.linspace(float(s.min()), float(s.max()), bins)
    w0 = torch.cumsum(p, 0)
    m0 = torch.cumsum(p * c_, 0)
    mt = m0[-1]
    var = (mt * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-9)
    return float(c_[var.argmax()])


def otsu_multi(s, bins=128):
    """Three-class Otsu; return the upper threshold (foreground = top class)."""
    h = torch.histc(s, bins=bins, min=float(s.min()), max=float(s.max())).double()
    p = h / h.sum()
    c_ = torch.linspace(float(s.min()), float(s.max()), bins).double()
    P = torch.cumsum(p, 0)
    S = torch.cumsum(p * c_, 0)
    best, arg = -1.0, (0, 0)
    for i in range(4, bins - 8, 2):
        j = torch.arange(i + 2, bins - 2, 2)
        w0, w1, w2 = P[i], P[j] - P[i], 1 - P[j]
        m0, m1, m2 = S[i] / w0.clamp_min(1e-9), (S[j] - S[i]) / w1.clamp_min(1e-9), (S[-1] - S[j]) / w2.clamp_min(1e-9)
        v = w0 * m0 ** 2 + w1 * m1 ** 2 + w2 * m2 ** 2
        k = int(v.argmax())
        if float(v[k]) > best:
            best, arg = float(v[k]), (i, int(j[k]))
    return float(c_[arg[0]]), float(c_[arg[1]])


def stable(s, lo=0.25, hi=0.8, n=56):
    """MSER-style: level in [lo, hi] of the robust range where relative area changes least."""
    a, b = torch.quantile(s, 0.01), torch.quantile(s, 0.999)
    ts = a + (b - a) * torch.linspace(lo, hi, n)
    area = torch.stack([(s > t).float().mean() for t in ts]).clamp_min(1e-6)
    d = (area[:-4] - area[4:]) / area[2:-2]
    return float(ts[2:-2][d.argmin()])


# ---------------------------------------------------------------- verbatim copy (d20_bidir_sets.py)
def jprefix(rank_score, f, g):
    """Level of rank_score whose prefix maximises J; returns the threshold value."""
    order = np.argsort(-rank_score)
    a = np.cumsum(f[order]) / np.arange(1, len(order) + 1)
    c_ = np.cumsum(g[order])
    J = a * c_ / np.maximum(a + c_ - a * c_, 1e-9)
    k = int(np.argmax(J[4:])) + 4
    return rank_score[order[k]], J[k]


# ---------------------------------------------------------------- verbatim copies (d10_sweep_cut.py)
def knn_graph(x, k=16, sigma=0.1):
    """Symmetric kNN affinity with weights exp((cos-1)/sigma); dense (n, n), zero diagonal."""
    s = x @ x.T
    s.fill_diagonal_(-2)
    v, i = s.topk(k, dim=1)
    W = torch.zeros_like(s).scatter_(1, i, torch.exp((v - 1) / sigma))
    return torch.maximum(W, W.T)


def sweep(W, score):
    """Conductance of every prefix of the descending-score order. Returns (order, phi)."""
    order = torch.argsort(score, descending=True)
    Wp = W[order][:, order]
    deg = Wp.sum(1)
    vol = torch.cumsum(deg, 0)
    inside = torch.cumsum(2 * torch.tril(Wp, -1).sum(1), 0)
    cut = vol - inside
    phi = cut / torch.minimum(vol, vol[-1] - vol).clamp_min(1e-9)
    return order, phi


def pick_k(phi, lo=8, hi=int(0.7 * N)):
    """d10 `pick`, rewritten to return the prefix length index instead of the token mask."""
    return lo + int(phi[lo:hi].argmin())


# ---------------------------------------------------------------- verbatim copy (d17_boundary_contrast.py)
def profile(sn, dh, dv):
    """Mean edge dissimilarity and edge count of the level-alpha contour, for all ALPHA."""
    g = sn.reshape(64, 64)
    lo = np.concatenate([np.minimum(g[:, 1:], g[:, :-1]).ravel(), np.minimum(g[1:], g[:-1]).ravel()])
    hi = np.concatenate([np.maximum(g[:, 1:], g[:, :-1]).ravel(), np.maximum(g[1:], g[:-1]).ravel()])
    d = np.concatenate([dh.ravel(), dv.ravel()])
    act = (lo[None, :] <= ALPHA[:, None]) & (ALPHA[:, None] < hi[None, :])
    cnt = act.sum(1)
    return (act * d[None, :]).sum(1) / np.maximum(cnt, 1), cnt


# ---------------------------------------------------------------- verbatim copy (d08_inverse_likelihood.py)
def reverse_iou(qc, rc, fgq, cr, KF, KB):
    """Predict reference labels from the query hypothesis; token-level soft IoU with M_R."""
    if fgq.sum() < 1 or (~fgq).sum() < 1:
        return 0.0
    PF, PB = lab.kmeans(qc, KF, iters=4, w=fgq.float()), lab.kmeans(qc, KB, iters=4, w=(~fgq).float())
    pred = ((rc @ PF.T).max(1).values - (rc @ PB.T).max(1).values) > 0
    inter = (pred.float() * cr).sum()
    return float(inter / (pred.float().sum() + cr.sum() - inter).clamp_min(1e-6))


# ---------------------------------------------------------------- shared
def setup():
    global lab
    c.initialize()
    lab = c.rc.lab
    lab.DEV = torch.device('cpu')
    lab.basis.cache_clear()
    torch.set_num_threads(2)


def sample():
    import w14_headroom as w14
    return w14.subset([b for b in c.rows() if c.half(b) == 'S'], 300)


def fused(b, fields):
    """G at 1024 exactly as W14 `one()`, and the 64-grid G64 with up(G64) == G (bilinear up is affine-equivariant)."""
    with np.load(c.task_path('T1') / 'fields' / (c.key(b) + '.npz'), allow_pickle=False) as z:
        mean64 = z['mean'].astype(np.float32)
    up_u = c.up(np.asarray(fields['VP_UN'], np.float32))
    G = .5 * c.up(mean64).astype(np.float64) + .5 * lab.minmax(torch.from_numpy(up_u)).numpy().astype(np.float64)
    lo, hi = float(up_u.min()), float(up_u.max())
    G64 = .5 * mean64.astype(np.float64) + .5 * (fields['VP_UN'].astype(np.float64) - lo) / max(hi - lo, 1e-6)
    return G, G64


def topk_level(G, frac):
    """d25 `topk`: level of the k-th largest value, k = clip(frac * n, 1, n); mask is G >= level."""
    flat = G.ravel()
    k = int(min(max(frac * flat.size, 1), flat.size))
    return float(np.partition(flat, flat.size - k)[flat.size - k])


def cue_one(b):
    """Label-free.  Reads fields, the T1 mean field, cached O24 tokens, the reference mask and the FoRIS score."""
    from repo_methods import foris_score
    fields = c.load_fields(b)
    rec = {'index': str(b['index']), 'dataset': b['dataset'], 'photo': c.photo(b), 'rho': float(fields['reference_rho'][0])}
    if rec['rho'] > .9:
        return rec
    t0 = time.time()
    G, G64 = fused(b, fields)
    g64 = G64.ravel()
    f = torch.from_numpy(G.astype(np.float32)).flatten()[::16]
    lev, op, est = {}, {}, {}
    lev['otsu'] = otsu(f)
    lev['otsu3_lo'], lev['otsu3_hi'] = otsu_multi(f)
    lev['mser'] = stable(f)
    lev['robust_mid'] = float(.5 * (torch.quantile(f, .01) + torch.quantile(f, .995)))
    for k in ('otsu', 'otsu3_lo', 'otsu3_hi', 'mser', 'robust_mid'):
        op[k] = '>'
    # J(S): forward purity a = mean c32 over S, backward capture c = sum back_f over S (tau .05)
    cf, bf = fields['c32'].astype(np.float64), fields['back_f'].astype(np.float64)
    tJ, Jmax = jprefix(g64, cf, bf)
    lev['J'], op['J'] = float(tJ), '>='
    S = g64[None, :] > ALPHA[:, None]
    size = np.maximum(S.sum(1), 1)
    a, cc = (S * cf[None]).sum(1) / size, (S * bf[None]).sum(1)
    Jcurve = a * cc / np.maximum(a + cc - a * cc, 1e-9)
    # tokens: debiased O24, jointly centred (d08 / d10), debiased only (d17 edges, dump_edges.py)
    r = lab.debias(lab.feats(b, 'reference'))
    q = lab.debias(lab.feats(b, 'query'))
    mu = .5 * (r.mean(0) + q.mean(0))
    rcn, qcn = lab.unit(r - mu), lab.unit(q - mu)
    W = knn_graph(qcn, 16, .1)
    order, phi = sweep(W, torch.from_numpy(g64.astype(np.float32)))
    k = pick_k(phi)
    lev['sweep'], op['sweep'] = float(g64[int(order[k])]), '>='
    del W
    gq = q.reshape(64, 64, -1)
    dh = (1 - (gq[:, 1:] * gq[:, :-1]).sum(-1)).numpy().astype(np.float32)
    dv = (1 - (gq[1:] * gq[:-1]).sum(-1)).numpy().astype(np.float32)
    sn = (g64 - g64.min()) / max(g64.max() - g64.min(), 1e-6)
    B, _ = profile(sn, dh, dv)
    am = int(np.argmax(np.convolve(B, np.ones(3) / 3, mode='same')[6:44])) + 6
    lev['contrast'], op['contrast'] = float(g64.min() + ALPHA[am] * (g64.max() - g64.min())), '>'
    ref = lab.mask1024(b['row']['reference_mask_path'])
    cr = lab.coverage64(ref).flatten()
    s0 = torch.from_numpy(g64.astype(np.float32))
    ts = torch.quantile(s0, QS)
    Jinv = [reverse_iou(qcn, rcn, s0 > t, cr, 4, 16) for t in ts]
    lev['inverse'], op['inverse'] = float(ts[int(np.argmax(Jinv))]), '>'
    # area estimators (fraction of the whole query frame), cut by top-k on G
    est['nat_area'] = float((G > .5).mean())
    est['sumG'] = float(G.mean())
    est['ref_area'] = float(ref.float().mean())
    fs = c.up(foris_score(b))
    est['foris_area'] = float((lab.minmax(torch.from_numpy(fs)).numpy() > .5).mean())
    est['fwd_area'] = float((cf > .5).mean())
    p = bf / bf.sum()
    est['inv_simpson'] = float(1 / (p ** 2).sum() / 4096)
    for k in AREA_CUES:
        if k == 'nat_area':
            lev[k], op[k] = .5, '>'
        else:
            lev[k], op[k] = topk_level(G, est[k]), '>='
    rec.update(level=lev, op=op, estimate=est, J_curve=[float(x) for x in Jcurve], J_max=float(Jmax),
               G_max=float(G.max()), G_min=float(G.min()), seconds=round(time.time() - t0, 2))
    return rec


def run_cues(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    rows = sample()
    path = OUT / ('cues_smoke.jsonl' if limit else 'cues.jsonl')
    done = set()
    if path.exists():
        done = {json.loads(l)['index'] for l in path.read_text().splitlines()}
    todo = [b for b in rows if str(b['index']) not in done][:limit]
    t0 = time.time()
    with path.open('a') as fh:
        for n, b in enumerate(todo, 1):
            fh.write(json.dumps(cue_one(b)) + '\n')
            fh.flush()
            if n % 100 == 0 or limit:
                print(n, len(todo), round(time.time() - t0), flush=True)
    print('done', flush=True)


# ---------------------------------------------------------------- truth (scoring only; after the preregistration)
RGRID = np.round(np.arange(-6, 6.0001, .1), 2)   # log2 area ratio to the natural area, for counterfactual curves


def truth_one(b, cue):
    from scipy import ndimage
    from diagnostic_metrics import exact_best_iou
    fields = c.load_fields(b)
    G, _ = fused(b, fields)
    gt, valid = c.truth(b)
    g = gt & valid
    v, y = G[valid], g[valid]
    n, ng = int(v.size), int(y.sum())
    rec = {'index': cue['index'], 'n_valid': n, 'n_gt': ng, 'gt_frac': ng / 2 ** 20}
    if ng == 0:
        return rec
    order = np.argsort(-v, kind='stable')
    s, cy = v[order], np.cumsum(y[order], dtype=np.int64)
    asc = s[::-1]

    def count(t, op):
        return int(n - np.searchsorted(asc, t, side='right' if op == '>' else 'left'))

    def iou_k(k):
        i = int(cy[k - 1]) if k > 0 else 0
        return i / max(ng + k - i, 1), i

    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), n - 1]
    inter = cy[ends]
    union = ng + ends + 1 - inter
    curve = inter / np.maximum(union, 1)
    j = int(curve.argmax())
    best, kstar, tstar = float(curve[j]), int(ends[j] + 1), float(s[ends[j]])
    ref = exact_best_iou(G, gt, valid)
    assert abs(ref - best) < 1e-12, (ref, best)
    knat = count(.5, '>')
    nat, inat = iou_k(knat)
    near = ends[curve >= best - .02] + 1
    shrink = float(curve[ends + 1 <= knat].max()) if knat > 0 else 0.
    expand = float(curve[ends + 1 >= knat].max())
    rec.update(natural=100 * nat, oracle=100 * best, t_star=tstar, k_star=kstar, k_nat=knat,
               near_lo=int(near.min()), near_hi=int(near.max()), shrink_oracle=100 * shrink, expand_oracle=100 * expand,
               P_nat=inat / max(knat, 1), R_nat=inat / ng, P_star=int(cy[kstar - 1]) / kstar, R_star=int(cy[kstar - 1]) / ng)
    rec['at_star_x1.41'] = 100 * iou_k(min(int(round(kstar * 2 ** .5)), n))[0]
    rec['at_star_/1.41'] = 100 * iou_k(max(int(round(kstar / 2 ** .5)), 1))[0]
    rec['cue'] = {}
    for k in CUES:
        kk = count(cue['level'][k], cue['op'][k])
        rec['cue'][k] = [kk, 100 * iou_k(kk)[0]]
    rec['curve'] = [round(100 * iou_k(int(min(max(round(max(knat, 1) * 2 ** r), 1), n)))[0], 3) for r in RGRID]
    rec['alpha_iou'] = [round(100 * iou_k(count(a, '>'))[0], 3) for a in ALPHA]
    Jc, Ia = np.array(cue['J_curve']), np.array(rec['alpha_iou'])
    rec['J_r'] = float(np.corrcoef(Jc, Ia)[0, 1]) if Jc.std() > 0 and Ia.std() > 0 else None
    # shape of the optimal move: is the oracle level set growing / trimming the found object, or collecting pieces?
    st = np.ones((3, 3))
    natm, orm = (G > .5) & valid, (G >= tstar) & valid
    lab_o, n_o = ndimage.label(orm, structure=st)
    lab_n, n_n = ndimage.label(natm, structure=st)
    rec.update(comp_nat=int(n_n), comp_star=int(n_o), comp_gt=int(ndimage.label(g, structure=st)[1]))
    if kstar > knat:
        add = orm & ~natm
        touch = np.zeros(n_o + 1, bool)
        touch[np.unique(lab_o[natm])] = True
        conn = add & touch[lab_o]
        rec['move'] = {'added': int(add.sum()), 'attached': int(conn.sum()), 'added_hit': int((add & g).sum()),
                       'detached_hit': int((add & ~conn & g).sum())}
    elif kstar < knat:
        rem = natm & ~orm
        keep = np.zeros(n_n + 1, bool)
        keep[np.unique(lab_n[orm])] = True
        whole = rem & ~keep[lab_n]
        rec['move'] = {'removed': int(rem.sum()), 'whole_comp': int(whole.sum()), 'removed_fp': int((rem & ~g).sum()),
                       'whole_fp': int((whole & ~g).sum())}
    return rec


def run_truth():
    pre = OUT / 'preregistration.md'
    assert pre.exists(), 'write the preregistration first'
    setup()
    cues = {r['index']: r for r in map(json.loads, (OUT / 'cues.jsonl').read_text().splitlines())}
    path = OUT / 'truth.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(l)['index'] for l in path.read_text().splitlines()}
    seal = {'preregistration_sha256': hashlib.sha256(pre.read_bytes()).hexdigest(),
            'cues_sha256': hashlib.sha256((OUT / 'cues.jsonl').read_bytes()).hexdigest(), 'started': time.ctime()}
    if not done:
        (OUT / 'truth_seal.json').write_text(json.dumps(seal, indent=1) + '\n')
    rows = [b for b in sample() if str(b['index']) in cues and 'level' in cues[str(b['index'])]]
    todo = [b for b in rows if str(b['index']) not in done]
    t0 = time.time()
    with path.open('a') as fh:
        for n, b in enumerate(todo, 1):
            fh.write(json.dumps(truth_one(b, cues[str(b['index'])])) + '\n')
            fh.flush()
            if n % 200 == 0:
                print(n, len(todo), round(time.time() - t0), flush=True)
    print('done', flush=True)


# ---------------------------------------------------------------- report (relations, bootstrap, classification)
BINS = (0, .01, .05, .2, .5, 1.01)
BIN_NAMES = ('<1%', '1-5%', '5-20%', '20-50%', '>50%')
B_REPS, B_SEED = 1000, 2026101602


def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float('nan')


def spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def partial_spearman(x, y, z):
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    A = np.c_[rz, np.ones_like(rz)]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return _pearson(res(rx), res(ry))


def auc(score, label):
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 < 10 or n0 < 10:
        return float('nan')
    r = _rank(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_table():
    C = {r['index']: r for r in map(json.loads, (OUT / 'cues.jsonl').read_text().splitlines())}
    T = [r for r in map(json.loads, (OUT / 'truth.jsonl').read_text().splitlines())]
    E = [(C[t['index']], t) for t in T if 'natural' in t]
    D = {'n_dropped_empty_gt': sum('natural' not in t for t in T)}
    D['dom'] = np.array([cu['dataset'] for cu, _ in E])
    D['photo'] = np.array([cu['photo'] for cu, _ in E])
    for k in ('natural', 'oracle', 't_star', 'k_star', 'k_nat', 'gt_frac', 'near_lo', 'near_hi', 'shrink_oracle',
              'expand_oracle', 'P_nat', 'R_nat', 'P_star', 'R_star', 'at_star_x1.41', 'at_star_/1.41', 'comp_nat',
              'comp_star', 'comp_gt', 'n_valid'):
        D[k] = np.array([t[k] for _, t in E], float)
    D['J_r'] = np.array([np.nan if t['J_r'] is None else t['J_r'] for _, t in E])
    D['curve'] = np.array([t['curve'] for _, t in E])
    D['alpha_iou'] = np.array([t['alpha_iou'] for _, t in E])
    D['move'] = [t.get('move') for _, t in E]
    D['natf'] = np.array([cu['estimate']['nat_area'] for cu, _ in E])
    D['reff'] = np.array([cu['estimate']['ref_area'] for cu, _ in E])
    D['level'] = {k: np.array([cu['level'][k] for cu, _ in E]) for k in CUES}
    D['kc'] = {k: np.array([t['cue'][k][0] for _, t in E], float) for k in CUES}
    D['iouc'] = {k: np.array([t['cue'][k][1] for _, t in E]) for k in CUES}
    return D


def report():
    D = load_table()
    n = len(D['dom'])
    rng = np.random.default_rng(B_SEED)
    hr = D['oracle'] - D['natural']
    top = np.zeros(n, bool)
    top[np.argsort(-hr, kind='stable')[:int(round(.2 * n))]] = True
    yr = np.log2((D['k_star'] + 1) / (D['k_nat'] + 1))
    yt = D['t_star']
    lognat = np.log2(np.maximum(D['natf'], 2 ** -20))
    X = {}
    for k in CUES:
        if k == 'nat_area':
            X[k] = (lognat, lognat)
        else:
            X[k] = (D['level'][k], np.log2((D['kc'][k] + 1) / (D['k_nat'] + 1)))
    suim = D['dom'] == 'suim'
    G = {'all': np.ones(n, bool), 'noSUIM': ~suim, 'hr:top20': top, 'hr:rest': ~top, 'hr:top20 noSUIM': top & ~suim}
    for d in c.DOMAINS:
        if (D['dom'] == d).any():
            G['dom:' + d] = D['dom'] == d
    for nm, (lo, hi) in zip(BIN_NAMES, zip(BINS[:-1], BINS[1:])):
        G['gt:' + nm] = (D['gt_frac'] >= lo) & (D['gt_frac'] < hi)
    for nm, (lo, hi) in zip(BIN_NAMES, zip(BINS[:-1], BINS[1:])):
        G['nat:' + nm] = (D['natf'] >= lo) & (D['natf'] < hi)
    for nm, (lo, hi) in zip(BIN_NAMES, zip(BINS[:-1], BINS[1:])):
        G['ref:' + nm] = (D['reff'] >= lo) & (D['reff'] < hi)
    up, tie = D['k_star'] > D['k_nat'], D['k_star'] == D['k_nat']
    pcode = np.unique(D['photo'], return_inverse=True)[1]

    def stats(idx, full=False):
        out = {}
        lab_up, keep = up[idx], ~tie[idx]
        for k in CUES:
            xt, xr = X[k][0][idx], X[k][1][idx]
            s = {'rho_r': spearman(xr, yr[idx]), 'auc': auc(xr[keep], lab_up[keep]),
                 'delta': float(np.mean(D['iouc'][k][idx] - D['natural'][idx]))}
            s['rho_t'] = float('nan') if k == 'nat_area' else spearman(xt, yt[idx])
            if full:
                s['partial_r'] = partial_spearman(xr, yr[idx], lognat[idx]) if k != 'nat_area' else float('nan')
            out[k] = s
        return out

    def boot(mask, full=False):
        eps = np.flatnonzero(mask)
        codes, inv = np.unique(pcode[eps], return_inverse=True)
        P = len(codes)
        reps = []
        for _ in range(B_REPS):
            w = rng.multinomial(P, np.full(P, 1 / P))
            reps.append(stats(np.repeat(eps, w[inv]), full))
        return reps

    def ci(reps, k, m):
        v = np.array([r[k][m] for r in reps])
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) > 50 else [float('nan')] * 2

    R = {'n': n, 'n_dropped_empty_gt': D['n_dropped_empty_gt'], 'groups': {}}
    t0 = time.time()
    for g, m in G.items():
        full = g in ('all', 'noSUIM', 'hr:top20', 'hr:top20 noSUIM')
        idx = np.flatnonzero(m)
        if len(idx) < 20:
            R['groups'][g] = {'n': int(len(idx))}
            continue
        point, reps = stats(idx, full), boot(m, full)
        entry = {'n': int(len(idx)), 'n_up': int(up[idx].sum()), 'n_tie': int(tie[idx].sum()),
                 'natural': float(D['natural'][idx].mean()), 'oracle': float(D['oracle'][idx].mean()), 'cues': {}}
        for k in CUES:
            e = dict(point[k])
            for mm in list(point[k]):
                e[mm + '_ci'] = ci(reps, k, mm)
            xr = X[k][1][idx]
            y = yr[idx]
            if k != 'nat_area':
                agree = (np.sign(xr) == np.sign(y)) & (y != 0)
                e['sign_agree'] = float(agree.mean())
                e['overshoot'] = float(np.median(np.abs(xr[agree]) / np.abs(y[agree]))) if agree.sum() >= 10 else float('nan')
                e['med_abs_x'], e['med_abs_y'] = float(np.median(np.abs(xr))), float(np.median(np.abs(y)))
                e['frac_expand'] = float((xr > 0).mean())
                side = np.where(xr > 0, D['expand_oracle'][idx], np.where(xr < 0, D['shrink_oracle'][idx], D['natural'][idx]))
                e['dir_only'] = float(np.mean(side - D['natural'][idx]))
                e['dir_random'] = float(np.mean(.5 * (D['expand_oracle'][idx] + D['shrink_oracle'][idx]) - D['natural'][idx]))
                tgt = np.clip(np.sign(y) * np.abs(xr), RGRID[0], RGRID[-1])
                mag = np.array([np.interp(tt, RGRID, D['curve'][i]) for tt, i in zip(tgt, idx)])
                e['mag_only'] = float(np.mean(mag - D['natural'][idx]))
                e['recovered'] = e['delta'] / max(entry['oracle'] - entry['natural'], 1e-9)
            entry['cues'][k] = e
        R['groups'][g] = entry
        print(g, len(idx), round(time.time() - t0), flush=True)
    R['classification'] = classify(R)
    R['framework'] = framework(D, top, yr, hr, suim)
    (OUT / 'results.json').write_text(json.dumps(R, indent=1, allow_nan=True) + '\n')
    write_text_report(R)


def classify(R):
    """Mechanical rule fixed in preregistration.md (section 4)."""
    out = {}
    free = [g for g in R['groups'] if g.startswith(('dom:', 'nat:', 'ref:'))]
    gtdef = [g for g in R['groups'] if g.startswith(('gt:', 'hr:top20', 'hr:rest')) and 'noSUIM' not in g]

    def sig(g, k, sign, thr):
        e = R['groups'][g]
        if e.get('n', 0) < 40 or 'cues' not in e:
            return False
        r, (lo, hi) = e['cues'][k]['rho_r'], e['cues'][k]['rho_r_ci']
        return (r >= thr and lo > 0) if sign > 0 else (r <= -thr and hi < 0)

    for k in CUES:
        a = R['groups']['all']['cues'][k]
        r, (lo, hi) = a['rho_r'], a['rho_r_ci']
        why = []
        if r >= .15 and lo > 0 and a['auc_ci'][0] > .5:
            if k == 'nat_area':
                cls = 'direction signal (no amount of its own)'
            elif a['delta_ci'][0] > 0:
                cls = 'partly effective (not a failure)'
            elif a['overshoot'] > 1.5:
                cls = '(c)'
            else:
                cls = "(c') direction right, no overshoot, still no recovery"
        elif r <= -.15 and hi < 0:
            cls = 'reversed overall (outside a-d)'
        else:
            pos = [g for g in free if sig(g, k, 1, .2)]
            neg = [g for g in free if sig(g, k, -1, .2)]
            gd = [g for g in gtdef if sig(g, k, 1, .25) or sig(g, k, -1, .25)]
            why = {'free+': pos, 'free-': neg, 'gt-defined': gd}
            if pos and neg:
                cls = '(b)'
            elif gd:
                cls = '(d)'
            else:
                cls = '(a)'
        out[k] = {'class': cls, 'detail': why}
    return out


def framework(D, top, yr, hr, suim):
    """Is the per-episode cut a well-posed target?  Shape of the optimum and of the move (scoring only)."""
    F = {}
    for nm, m in (('top20', top), ('rest', ~top), ('all', np.ones_like(top))):
        width = np.log2(D['near_hi'][m] / np.maximum(D['near_lo'][m], 1))
        inside = (D['k_nat'][m] >= D['near_lo'][m]) & (D['k_nat'][m] <= D['near_hi'][m])
        F[nm] = {'n': int(m.sum()), 'headroom': float(hr[m].mean()), 'oracle_lt50': float((D['oracle'][m] < 50).mean()),
                 'near2_width_log2_median': float(np.median(width)), 'natural_within_2pts': float(inside.mean()),
                 'abs_yr_median': float(np.median(np.abs(yr[m]))), 't_star_quartiles': np.quantile(D['t_star'][m], [.25, .5, .75]).tolist(),
                 'loss_at_x1.41': float(np.mean(D['oracle'][m] - D['at_star_x1.41'][m])),
                 'loss_at_/1.41': float(np.mean(D['oracle'][m] - D['at_star_/1.41'][m])),
                 'P_nat': float(np.mean(D['P_nat'][m])), 'R_nat': float(np.mean(D['R_nat'][m])),
                 'P_star': float(np.mean(D['P_star'][m])), 'R_star': float(np.mean(D['R_star'][m]))}
        exp = [mv for mv, mm in zip(D['move'], m) if mm and mv and 'added' in mv]
        shr = [mv for mv, mm in zip(D['move'], m) if mm and mv and 'removed' in mv]
        if exp:
            F[nm]['expand'] = {'n': len(exp), 'attached_share': float(np.median([e['attached'] / max(e['added'], 1) for e in exp])),
                               'added_precision': float(np.median([e['added_hit'] / max(e['added'], 1) for e in exp])),
                               'detached_precision': float(np.median([e['detached_hit'] / max(e['added'] - e['attached'], 1) for e in exp if e['added'] > e['attached']] or [np.nan]))}
        if shr:
            F[nm]['shrink'] = {'n': len(shr), 'whole_component_share': float(np.median([e['whole_comp'] / max(e['removed'], 1) for e in shr])),
                               'removed_fp_share': float(np.median([e['removed_fp'] / max(e['removed'], 1) for e in shr]))}
    for nm, m in (('top20', top), ('all', np.ones_like(top))):
        for d in sorted(set(D['dom'])):
            mm = m & (D['dom'] == d)
            if mm.sum():
                F.setdefault(nm + '_dir_by_domain', {})[d] = [int(mm.sum()), float((yr[mm] > 0).mean())]
    ai = D['alpha_iou']
    best = int(np.argmax(ai.mean(0)))
    F['best_constant_in_sample'] = {'alpha': float(ALPHA[best]), 'mean_iou': float(ai[:, best].mean()),
                                    'natural': float(D['natural'].mean()), 'oracle': float(D['oracle'].mean())}
    F['J_cross_level_r'] = {'all': float(np.nanmean(D['J_r'])), 'median': float(np.nanmedian(D['J_r'])),
                            'by_domain': {d: float(np.nanmean(D['J_r'][D['dom'] == d])) for d in sorted(set(D['dom']))}}
    F['oracle_vs_gt_area'] = {nm: np.quantile(np.log2((D['k_star'][m] + 1) / (D['gt_frac'][m] * 2 ** 20 + 1)), [.1, .25, .5, .75, .9]).tolist()
                              for nm, m in (('top20', top), ('rest', ~top))}
    return F


def write_text_report(R):
    L = [f"W65 cues, n={R['n']} (dropped empty GT {R['n_dropped_empty_gt']}); photo-grouped bootstrap B={B_REPS}"]
    f2 = lambda x: 'nan' if x is None or not np.isfinite(x) else f'{x:+.2f}'
    for g in ('all', 'noSUIM', 'hr:top20', 'hr:top20 noSUIM', 'hr:rest'):
        e = R['groups'][g]
        L.append(f"\n## {g}  n={e['n']}  natural {e['natural']:.1f}  oracle {e['oracle']:.1f}  up {e['n_up']} tie {e['n_tie']}")
        L.append(f"{'cue':12s}{'rho_t':>16s}{'rho_r':>18s}{'partial':>8s}{'AUC':>18s}{'delta':>20s}{'rec%':>7s}{'agree':>6s}{'over':>6s}{'|x|':>6s}{'|y|':>6s}{'DO':>7s}{'rand':>7s}{'MO':>7s}")
        for k in CUES:
            s = e['cues'][k]
            L.append(f"{k:12s}{f2(s['rho_t'])} [{f2(s['rho_t_ci'][0])},{f2(s['rho_t_ci'][1])}]"
                     f"{f2(s['rho_r']):>7s} [{f2(s['rho_r_ci'][0])},{f2(s['rho_r_ci'][1])}]{f2(s.get('partial_r')):>7s}"
                     f"{s['auc']:7.2f} [{s['auc_ci'][0]:.2f},{s['auc_ci'][1]:.2f}]"
                     f"{s['delta']:+7.2f} [{s['delta_ci'][0]:+.2f},{s['delta_ci'][1]:+.2f}]"
                     + (f"{100 * s['recovered']:+7.0f}{s['sign_agree']:6.2f}{s['overshoot']:6.2f}{s['med_abs_x']:6.2f}{s['med_abs_y']:6.2f}"
                        f"{s['dir_only']:+7.2f}{s['dir_random']:+7.2f}{s['mag_only']:+7.2f}" if k != 'nat_area' else ''))
    L.append('\n## rho_r by subgroup (point; * = 95% CI excludes 0)')
    gs = [g for g in R['groups'] if g.startswith(('dom:', 'gt:', 'nat:', 'ref:', 'hr:')) and 'cues' in R['groups'][g]]
    L.append(f"{'cue':12s}" + ''.join(f"{g.split(':')[1][:7]:>8s}" for g in gs))
    L.append(f"{'n':12s}" + ''.join(f"{R['groups'][g]['n']:8d}" for g in gs))
    for k in CUES:
        cells = []
        for g in gs:
            s = R['groups'][g]['cues'][k]
            star = '*' if (s['rho_r_ci'][0] > 0 or s['rho_r_ci'][1] < 0) else ' '
            cells.append(f"{s['rho_r']:+7.2f}{star}")
        L.append(f"{k:12s}" + ''.join(cells))
    L.append('\n## delta IoU (cue - natural) by subgroup')
    L.append(f"{'cue':12s}" + ''.join(f"{g.split(':')[1][:7]:>8s}" for g in gs))
    for k in CUES:
        L.append(f"{k:12s}" + ''.join(f"{R['groups'][g]['cues'][k]['delta']:+8.2f}" for g in gs))
    L.append('\n## direction AUC by subgroup')
    for k in CUES:
        L.append(f"{k:12s}" + ''.join(f"{R['groups'][g]['cues'][k]['auc']:8.2f}" for g in gs))
    L.append('\n## classification (mechanical rule of the preregistration)')
    for k, v in R['classification'].items():
        L.append(f"{k:12s} {v['class']}   {v['detail']}")
    L.append('\n## framework\n' + json.dumps(R['framework'], indent=1))
    (OUT / 'report.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    phase = sys.argv[1]
    if phase == 'cues':
        run_cues(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif phase == 'truth':
        run_truth()
    elif phase == 'report':
        report()
