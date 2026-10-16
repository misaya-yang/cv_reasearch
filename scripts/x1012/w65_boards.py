"""W65 branch A: look at the extent errors of MFU's fused pre-CRF field G, then test GT-free observables.

Plan: docs/research/x1012_tasks/W65_extent_headroom_PLAN.md (sections 1-3 fixed before running).
Sample: the W14 S-half subset (w14_headroom.subset(S, 300)) with rho <= .9, i.e. the 1,970 episodes that have
G = .5 * up(T1 mean) + .5 * minmax(up(VP_UN)) (definition copied from w14_headroom.one()).  T half and the
new-photo holdout are never read.

Rules fixed before any per-episode result was looked at (this docstring was written first):
  split      sha256('2026101601:' + c.photo(b)) -> first 8 hex digits mod 2: 0 = 'V' (viewing half), 1 = 'X' (test
             half).  The same query photo always lands in the same half.
  oracle     exact best level of G inside `valid`, keeping score-tie groups whole (diagnostic_metrics.exact_best_iou
             rule); t* = lowest score value of the best prefix, best mask = G >= t*, natural mask = G > .5.
  headroom   100 * (best IoU - natural IoU).  "top 20%" = headroom >= 80th percentile of the half in question.
  direction  under = best area > natural area (should expand), over = best area < natural area (should shrink).
  boards     viewing half only.  rng = np.random.default_rng(2026101601); from the episodes sorted by index,
             30 random top-20% under, then 30 random top-20% over, then 20 random controls whose headroom is
             <= the 20th percentile of the viewing half.  Rows: reference + mask | query + GT |
             G heat map (fixed 0-1 scale) with the natural (white) and best (red) contours | FoRIS score heat map
             (per-episode min-max) | small panel: area fraction of {G > t} against t (log scale), white = .5,
             red = t*.  GT enters only the second column, the red contour/marker and the header.
  measure    only after ../cv_data/a/x1013_W65/boards/preregistration.md exists; GT-free quantities on the test
             half, one round (see measure()).

    nice -n 19 python -B scripts/x1012/w65_boards.py prepare     # oracle + split + selection (all 1,970)
    nice -n 19 python -B scripts/x1012/w65_boards.py boards      # 80 rows -> 16 PNGs
    nice -n 19 python -B scripts/x1012/w65_boards.py measure     # after preregistration; test half only

Viewing-half-only diagnostics (exploration; never read the test half):
    explore     25 candidate quantities -> explore_V.txt (used to choose what to pre-register)
    levels      GT diagnostic: IoU at fixed levels, gain of a known direction with a fixed step -> levels_V.txt
    size_check  after the test round: are the quantities proxies of target size? -> size_check_V.txt
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W65') / 'boards'
SEED = 2026101601
CELL = 256


def split(binding):
    return 'V' if int(hashlib.sha256(f'{SEED}:{c.photo(binding)}'.encode()).hexdigest()[:8], 16) % 2 == 0 else 'X'


def sample():
    import w14_headroom as w14
    rows = w14.subset([b for b in c.rows() if c.half(b) == 'S'], 300)
    return [b for b in rows if float(c.load_fields(b)['reference_rho'][0]) <= .9]


def field_G(binding, fields=None):
    """W14 one(): G = .5 * up(T1 mean) + .5 * minmax(up(VP_UN)), float64, 1024 x 1024."""
    import torch
    fields = fields if fields is not None else c.load_fields(binding)
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    vp = c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(fields['VP_UN'], np.float32)))).numpy().astype(np.float64)
    return .5 * m + .5 * vp


def exact_best(field, gt, valid):
    """diagnostic_metrics.exact_best_iou, extended to also return the level and area of the best prefix."""
    f, g = field[valid], gt[valid]
    order = np.argsort(-f, kind='stable')
    s = f[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    inter = np.cumsum(g[order], dtype=np.int64)[ends]
    pred = ends + 1
    iou = inter / np.maximum(int(g.sum()) + pred - inter, 1)
    j = int(np.argmax(iou))
    return float(iou[j]), float(s[ends[j]]), int(pred[j])


def one(binding):
    fields = c.load_fields(binding)
    G = field_G(binding, fields)
    gt, valid = c.truth(binding)
    g = gt & valid
    nat = (G > .5) & valid
    best, t, area = exact_best(G, g, valid)
    import diagnostic_metrics as dm
    check = dm.exact_best_iou(G, g, valid)
    assert abs(check - best) < 1e-12, (check, best)
    rm = c.rc.lab.mask1024(binding['row']['reference_mask_path']).numpy() > .5
    n_valid = int(valid.sum())
    return {'index': str(binding['index']), 'dataset': binding['dataset'], 'photo': c.photo(binding),
            'split': split(binding), 'rho': float(fields['reference_rho'][0]),
            'natural': 100 * int((nat & g).sum()) / max(int((nat | g).sum()), 1), 'best': 100 * best, 't': t,
            'natural_area': int(nat.sum()), 'best_area': area, 'gt_area': int(g.sum()), 'valid_area': n_valid,
            'ref_frac': float(rm.mean())}


def load_episodes():
    return [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    c.initialize()
    rows = sample()
    print('sample', len(rows), flush=True)
    done = set()
    if (OUT / 'episodes.jsonl').exists():
        done = {r['index'] for r in load_episodes()}
    with (OUT / 'episodes.jsonl').open('a') as fh:
        for n, b in enumerate((b for b in rows if str(b['index']) not in done), 1):
            fh.write(json.dumps(one(b)) + '\n')
            fh.flush()
            if n % 200 == 0:
                print(n, flush=True)
    E = load_episodes()
    assert len(E) == len(rows) == len({e['index'] for e in E})
    by_photo = {}
    for e in E:
        by_photo.setdefault(e['photo'], set()).add(e['split'])
    assert all(len(s) == 1 for s in by_photo.values())
    V = sorted((e for e in E if e['split'] == 'V'), key=lambda e: e['index'])
    h = np.array([e['best'] - e['natural'] for e in V])
    hi, lo = np.quantile(h, .8), np.quantile(h, .2)
    under = [e for e, x in zip(V, h) if x >= hi and e['best_area'] > e['natural_area']]
    over = [e for e, x in zip(V, h) if x >= hi and e['best_area'] < e['natural_area']]
    ctrl = [e for e, x in zip(V, h) if x <= lo]
    rng = np.random.default_rng(SEED)
    pick = lambda L, k: [L[i] for i in sorted(rng.choice(len(L), min(k, len(L)), replace=False))]
    sel = [dict(e, kind='under') for e in pick(under, 30)] + [dict(e, kind='over') for e in pick(over, 30)] + \
          [dict(e, kind='control') for e in pick(ctrl, 20)]
    (OUT / 'selection.json').write_text(json.dumps({'n_V': len(V), 'n_X': len(E) - len(V), 'top20_threshold_V': hi,
                                                    'control_threshold_V': lo, 'n_under_pool': len(under),
                                                    'n_over_pool': len(over), 'n_control_pool': len(ctrl),
                                                    'rows': sel}, indent=1))
    print('V', len(V), 'X', len(E) - len(V), 'pools', len(under), len(over), len(ctrl), flush=True)


def tint(rgb, mask, color):
    from scipy import ndimage
    x = rgb.copy()
    x[mask] = .55 * x[mask] + .45 * np.array(color, np.float64) / 255
    edge = mask & ~ndimage.binary_erosion(mask, iterations=3)
    x[edge] = np.array(color, np.float64) / 255
    return x


def small(x, size=CELL):
    from PIL import Image
    return np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                      np.float64) / 255


def contour(mask, size=CELL):
    from scipy import ndimage
    m = mask.reshape(size, mask.shape[0] // size, size, mask.shape[1] // size).mean((1, 3)) > .5
    return m & ~ndimage.binary_erosion(m)


def heat(values, gray, cmap):
    import matplotlib
    return .78 * matplotlib.colormaps[cmap](np.clip(values, 0, 1))[..., :3] + .22 * gray[..., None]


def area_panel(G, valid, t_star, w=128):
    """Area fraction of {G > t} for t in [0, 1], log10 scale from 1e-4 to 1; white line .5, red line t*."""
    img = np.full((CELL, w, 3), .12)
    v = np.sort(G[valid])
    ts = np.linspace(0, 1, CELL)
    frac = 1 - np.searchsorted(v, ts, side='right') / len(v)
    y = np.clip((np.log10(np.maximum(frac, 1e-4)) + 4) / 4, 0, 1)  # 0 at 1e-4, 1 at 100%
    for k in (1, 2, 3):  # grid at 10%, 1%, 0.1%
        img[:, int((1 - k / 4) * (w - 1))] = .3
    for i, (t, yy) in enumerate(zip(ts, y)):
        img[CELL - 1 - i, :max(int(yy * (w - 1)), 1)] = (.55, .75, .95)
    img[CELL - 1 - int(.5 * (CELL - 1))] = (1, 1, 1)
    if 0 <= t_star <= 1:
        img[CELL - 1 - int(t_star * (CELL - 1))] = (1, .15, .15)
    return img


def board_row(e):
    import w42_ordered_cut as w42
    import repo_methods as rmeth
    from PIL import Image, ImageDraw, ImageFont
    b = BY_INDEX[e['index']]
    G = field_G(b)
    gt, valid = c.truth(b)
    nat, best = (G > .5) & valid, (G >= e['t']) & valid
    rm = c.rc.lab.mask1024(b['row']['reference_mask_path']).numpy() > .5
    rr = w42.rgb(b['row']['reference_path'], b['row'].get('reference_crop'))
    qq = w42.rgb(b['row']['query_path'], b['row'].get('query_crop'))
    gray = small(qq).mean(2)
    g3 = heat(small(np.clip(G, 0, 1)), gray, 'viridis')
    cb, cn = contour(best), contour(nat)
    g3[cb] = (1, .15, .15)
    g3[cn] = (1, 1, 1)
    s = rmeth.foris_score(b)
    s_up = c.up(s)
    lo, hi = float(s.min()), float(s.max())
    f3 = heat(small((s_up - lo) / max(hi - lo, 1e-6)), gray, 'magma')
    cells = [small(tint(rr, rm, (255, 200, 0))), small(tint(qq, gt & valid, (0, 220, 0))), g3, f3,
             area_panel(G, valid, e['t'])]
    img = np.concatenate(cells, 1)
    head = Image.new('RGB', (img.shape[1], 38), (255, 255, 255))
    font = ImageFont.load_default(size=13)
    va = e['valid_area']
    t1 = (f"{e['kind'].upper()} | {e['dataset']} idx {e['index']} | IoU nat {e['natural']:.0f} -> best {e['best']:.0f}"
          f" (+{e['best'] - e['natural']:.0f}) | t* {e['t']:.2f}")
    t2 = (f"area%  ref {100 * e['ref_frac']:.1f}  GT {100 * e['gt_area'] / va:.1f}  nat {100 * e['natural_area'] / va:.1f}"
          f"  best {100 * e['best_area'] / va:.1f} | FoRIS range {lo:.2f}..{hi:.2f}")
    d = ImageDraw.Draw(head)
    d.text((4, 2), t1, fill=(0, 0, 0), font=font)
    d.text((4, 20), t2, fill=(0, 0, 0), font=font)
    return np.concatenate([np.asarray(head), (img * 255).astype(np.uint8)], 0)


def boards():
    from PIL import Image
    global BY_INDEX
    c.initialize()
    sel = json.loads((OUT / 'selection.json').read_text())['rows']
    BY_INDEX = {str(b['index']): b for b in sample()}
    order = {'under': 0, 'over': 1, 'control': 2}
    sel = sorted(sel, key=lambda e: (order[e['kind']], e['dataset'], e['index']))
    for kind in order:
        L = [e for e in sel if e['kind'] == kind]
        for k in range(0, len(L), 5):
            rows = [board_row(e) for e in L[k:k + 5]]
            Image.fromarray(np.concatenate(rows, 0)).save(OUT / f'{kind}_{k // 5 + 1:02d}.png')
            print(kind, k, flush=True)
    (OUT / 'board_order.json').write_text(json.dumps(
        [{'board': f"{e['kind']}_{[x['index'] for x in sel if x['kind'] == e['kind']].index(e['index']) // 5 + 1:02d}",
          'index': e['index'], 'dataset': e['dataset'], 'kind': e['kind']} for e in sel], indent=1))


def observables(b):
    """GT-free candidate quantities (inputs: G, its two halves, FoRIS score, reference mask, cached T0 fields)."""
    import torch
    from scipy import ndimage
    import repo_methods as rmeth
    fields = c.load_fields(b)
    with np.load(c.task_path('T1') / 'fields' / (c.key(b) + '.npz'), allow_pickle=False) as z:
        M = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    U = c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(fields['VP_UN'], np.float32)))).numpy().astype(np.float64)
    G = .5 * M + .5 * U
    valid = np.ones((1024, 1024), bool)  # the label ignore map is label-derived (e.g. VOC boundaries): not used
    v = G[valid]
    A = lambda t: int((v > t).sum())
    nat = (G > .5) & valid
    a5 = max(A(.5), 1)
    gmax = float(v.max())
    mid = (.5 + gmax) / 2
    out = {'peak': float(np.quantile(v, .995)), 'gmax': gmax, 'nat_frac': A(.5) / v.size,
           'Lsh': float(np.log((A(.35) + 1) / (A(.5) + 1))), 'Ush': float(np.log((A(.5) + 1) / (A(.65) + 1))),
           'core': A(mid) / a5}
    # component structure on a 256 grid (G is bilinear from 64^2, so 4x subsampling loses nothing visible)
    g4, n4 = G[2::4, 2::4], nat[2::4, 2::4]
    lab_, n = ndimage.label(n4, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(np.ones_like(g4), lab_, index=np.arange(1, n + 1))
        peaks = ndimage.maximum(g4, lab_, index=np.arange(1, n + 1))
        out['weak'] = float(sizes[peaks < mid].sum() / sizes.sum())
        out['ncomp'] = int((sizes >= 4).sum())
    else:
        out['weak'], out['ncomp'] = 0.0, 0
    lab35, n35 = ndimage.label(g4 > .35, structure=np.ones((3, 3)))
    touching = np.unique(lab35[n4])
    touching = touching[touching > 0]
    att = np.isin(lab35, touching)
    out['attached_sh'] = float(np.log((att.sum() + 1) / (n4.sum() + 1)))
    out['detached_sh'] = float(((lab35 > 0) & ~att).sum() / max((lab35 > 0).sum(), 1))
    rm = c.rc.lab.mask1024(b['row']['reference_mask_path'])
    cov = c.rc.lab.coverage64(rm).numpy().ravel()
    ref_frac = float(rm.float().mean())
    out['ref_nat'] = float(np.log((ref_frac + 1e-4) / (out['nat_frac'] + 1e-4)))
    out['ref_thin'] = float(((cov > 0) & (cov < 1)).sum() / max((cov > 0).sum(), 1))
    nat64 = nat.reshape(64, 16, 64, 16).mean((1, 3)).ravel() > .5
    reg = fields['regions64'].astype(np.int64)
    n_k = np.bincount(reg, minlength=64)
    in_k = np.bincount(reg, weights=nat64.astype(float), minlength=64)
    p_k = np.where(n_k > 0, in_k / np.maximum(n_k, 1), 0)
    expand = float(((n_k - in_k) * (p_k > .5)).sum())
    shrink = float((in_k * ((p_k <= .5) & (n_k > 0))).sum())
    out['region_vote'] = float(np.log((expand + 1) / (shrink + 1)))
    bf = fields['back_f'].astype(np.float64)
    out['back_out'] = float(bf[~nat64].sum() / max(bf.sum(), 1e-12))
    sm = fields['sim_max'].astype(np.float64).reshape(64, 64)
    n2 = nat64.reshape(64, 64)
    ring = ndimage.binary_dilation(n2, iterations=1) & ~n2
    out['sim_ring'] = float(sm[ring].mean() - sm[n2].mean()) if ring.any() and n2.any() else 0.0
    d_out = ndimage.distance_transform_edt(~nat[2::4, 2::4])
    band = (d_out > 0) & (d_out <= 4) & valid[2::4, 2::4]
    out['ring_G'] = float(g4[band].mean()) if band.any() else 0.0
    out['MU'] = float(np.log(((M > .5) & valid).sum() + 1) - np.log(((U > .5) & valid).sum() + 1))
    nb = fields['query_neighbors20'].astype(np.int64)
    inside_nb = nat64[nb].mean(1)  # per token: share of its 20 query-feature neighbours inside the natural mask
    rflat = ring.ravel()
    out['graph_expand'] = float(inside_nb[rflat].mean()) if rflat.any() else 0.0
    out['graph_shrink'] = float(1 - inside_nb[nat64].mean()) if nat64.any() else 0.0
    out['graph_dir'] = out['graph_expand'] - out['graph_shrink']
    bb = fields['back_b'].astype(np.float64)
    out['backb_in'] = float(bb[nat64].sum() / max(bb.sum(), 1e-12))
    out['back_dir'] = float(np.log(out['back_out'] + 1e-4) - np.log(out['backb_in'] + 1e-4))
    c32 = fields['c32'].astype(np.float64)
    out['c32_ring'] = float(c32[rflat].mean()) if rflat.any() else 0.0
    le = fields['log_e'].astype(np.float64)
    out['loge_ring'] = float(le[rflat].mean() - le[nat64].mean()) if rflat.any() and nat64.any() else 0.0
    s = rmeth.foris_score(b)
    fs = c.up(s).astype(np.float64)
    fs = (fs - fs.min()) / max(fs.max() - fs.min(), 1e-6)
    out['foris_rel'] = float(np.log(((fs > .5) & valid).sum() + 1) - np.log(nat.sum() + 1))
    return out


def auc(x, y):
    """P(x_under > x_over), ties counted half; y = 1 for under."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def explore():
    """Viewing half only: candidate quantities, to choose what to pre-register (never reads the test half)."""
    c.initialize()
    E = {e['index']: e for e in load_episodes() if e['split'] == 'V'}
    rows = [b for b in sample() if str(b['index']) in E]
    path = OUT / 'explore_V.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open('a') as fh:
        for n, b in enumerate((b for b in rows if str(b['index']) not in done), 1):
            fh.write(json.dumps(dict(observables(b), index=str(b['index']))) + '\n')
            fh.flush()
            if n % 200 == 0:
                print(n, flush=True)
    F = [dict(json.loads(l), **{k: E[json.loads(l)['index']][k] for k in ('dataset', 'natural', 'best', 'natural_area', 'best_area')})
         for l in path.read_text().splitlines()]
    h = np.array([f['best'] - f['natural'] for f in F])
    top = h >= np.quantile(h, .8)
    y = np.array([f['best_area'] > f['natural_area'] for f in F])
    doms = [d for d in c.DOMAINS if d != 'fundus']
    keys = [k for k in F[0] if k not in ('index', 'dataset', 'natural', 'best', 'natural_area', 'best_area')]
    print(f'{"quantity":12s}{"AUC top20":>10s}{"AUC all":>9s}  per-domain AUC (all V) ' + ' '.join(d[:5] for d in doms))
    for k in keys:
        x = np.array([f[k] for f in F], float)
        per = [auc(x[[f['dataset'] == d for f in F]], y[[f['dataset'] == d for f in F]]) for d in doms]
        print(f'{k:12s}{auc(x[top], y[top]):10.3f}{auc(x, y):9.3f}  ' + ' '.join(f'{p:5.2f}' for p in per))


LEVELS = np.round(np.arange(.10, .91, .05), 2)


def level_curve(b):
    G = field_G(b)
    gt, valid = c.truth(b)
    g = (gt & valid)[valid]
    v = G[valid]
    return {'index': str(b['index']),
            'iou': [100 * int(((v > t) & g).sum()) / max(int(((v > t) | g).sum()), 1) for t in LEVELS]}


def levels():
    """Viewing half only, GT diagnostic: IoU at fixed levels, to ask whether a known direction alone would help."""
    c.initialize()
    E = {e['index']: e for e in load_episodes() if e['split'] == 'V'}
    path = OUT / 'levels_V.jsonl'
    if not path.exists():
        with path.open('w') as fh:
            for b in (b for b in sample() if str(b['index']) in E):
                fh.write(json.dumps(level_curve(b)) + '\n')
    L = [json.loads(l) for l in path.read_text().splitlines()]
    nat = np.array([E[r['index']]['natural'] for r in L])
    best = np.array([E[r['index']]['best'] for r in L])
    I = np.array([r['iou'] for r in L])
    up = np.array([E[r['index']]['best_area'] > E[r['index']]['natural_area'] for r in L])
    h = best - nat
    top = h >= np.quantile(h, .8)
    i5 = int(np.where(LEVELS == .5)[0][0])
    assert np.allclose(I[:, i5], nat)
    print(f'V n={len(L)}; top20 n={top.sum()}: natural {nat[top].mean():.1f}, best {best[top].mean():.1f}, '
          f'best>=70 {np.mean(best[top] >= 70):.2f}, best<50 {np.mean(best[top] < 50):.2f}')
    print('step  | top20: right-direction gain, wrong-direction gain, share of headroom | all V: right, wrong')
    for k in (1, 2, 3, 4):
        right = np.where(up, I[:, i5 - k], I[:, i5 + k]) - nat
        wrong = np.where(up, I[:, i5 + k], I[:, i5 - k]) - nat
        print(f'{.05 * k:.2f}  | {right[top].mean():+6.1f} {wrong[top].mean():+6.1f} {right[top].sum() / h[top].sum():5.2f} |'
              f' {right.mean():+6.2f} {wrong.mean():+6.2f}')
    w = [(row >= b_ - 5).sum() * .05 for row, b_ in zip(I, best)]
    print('width of levels (step .05) within 5 IoU of the best, top20 median', np.median(np.array(w)[top]),
          'others', np.median(np.array(w)[~top]))


PREREG = ('back_out', 'sim_ring', 'graph_dir', 'MU')


def boot(x, y, photos, reps=2000):
    """Point AUC and photo-grouped bootstrap 95% percentile interval."""
    rng = np.random.default_rng(SEED)
    groups = {}
    for i, p in enumerate(photos):
        groups.setdefault(p, []).append(i)
    keys = list(groups)
    vals = []
    for _ in range(reps):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        a = auc(x[idx], y[idx])
        if np.isfinite(a):
            vals.append(a)
    return auc(x, y), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def measure():
    """Test half, one round, exactly as pre-registered in preregistration.md (written before this ran)."""
    from scipy.stats import rankdata
    assert (OUT / 'preregistration.md').exists(), 'pre-register first'
    c.initialize()
    E = {e['index']: e for e in load_episodes() if e['split'] == 'X'}
    path = OUT / 'measure_X.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open('a') as fh:
        for n, b in enumerate((b for b in sample() if str(b['index']) in E and str(b['index']) not in done), 1):
            o = observables(b)
            fh.write(json.dumps({'index': str(b['index']), **{k: o[k] for k in PREREG},
                                 'slope': o['Lsh'] + o['Ush']}) + '\n')
            fh.flush()
            if n % 200 == 0:
                print(n, flush=True)
    F = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(F) == len(E) == len({f['index'] for f in F})
    ep = [E[f['index']] for f in F]
    dom = np.array([e['dataset'] for e in ep])
    photos = [e['photo'] for e in ep]
    y = np.array([e['best_area'] > e['natural_area'] for e in ep])
    h = np.array([e['best'] - e['natural'] for e in ep])
    top = h >= np.quantile(h, .8)
    X = {k: np.array([f[k] for f in F], float) for k in PREREG + ('slope',)}
    X['combo'] = np.mean([rankdata(X[k]) for k in ('back_out', 'sim_ring', 'graph_dir')], 0)
    doms = [d for d in c.DOMAINS if d != 'fundus']
    nos = dom != 'suim'
    lines = [f'test half n={len(F)}, top20 n={top.sum()} (under {y[top].sum()}, over {(~y[top]).sum()}); '
             f'without SUIM top20 n={(top & nos).sum()}, all n={nos.sum()}', '',
             '| quantity | AUC top20 [95% CI] | AUC all [95% CI] | domains AUC>.5 (all, 9) | domains AUC>.5 (top20, defined) |'
             ' AUC top20 no SUIM | AUC all no SUIM |', '|---|---|---|---|---|---|---|']
    per_dom = {}
    for k in ('back_out', 'sim_ring', 'graph_dir', 'MU', 'combo', 'slope'):
        x = X[k]
        a_t = boot(x[top], y[top], [p for p, t in zip(photos, top) if t])
        a_a = boot(x, y, photos)
        pd_all = {d: auc(x[dom == d], y[dom == d]) for d in doms}
        pd_top = {d: auc(x[top & (dom == d)], y[top & (dom == d)]) for d in doms
                  if y[top & (dom == d)].sum() >= 3 and (~y[top & (dom == d)]).sum() >= 3}
        per_dom[k] = (pd_all, pd_top)
        n_all = sum(np.isfinite(v) and v > .5 for v in pd_all.values())
        n_top = sum(v > .5 for v in pd_top.values())
        n_t = boot(x[top & nos], y[top & nos], [p for p, t in zip(photos, top & nos) if t])
        n_a = boot(x[nos], y[nos], [p for p, t in zip(photos, nos) if t])
        lines.append(f'| {k} | {a_t[0]:.3f} [{a_t[1]:.3f}, {a_t[2]:.3f}] | {a_a[0]:.3f} [{a_a[1]:.3f}, {a_a[2]:.3f}] | '
                     f'{n_all}/9 | {n_top}/{len(pd_top)} | {n_t[0]:.3f} [{n_t[1]:.3f}, {n_t[2]:.3f}] | '
                     f'{n_a[0]:.3f} [{n_a[1]:.3f}, {n_a[2]:.3f}] |')
    lines += ['', 'per-domain AUC, all test half (n per domain) / top20 where both classes >= 3',
              f'{"quantity":10s}' + ''.join(f'{d[:9]:>11s}' for d in doms)]
    lines.append(f'{"n":10s}' + ''.join(f'{int((dom == d).sum()):>11d}' for d in doms))
    lines.append(f'{"n top20":10s}' + ''.join(f'{int((top & (dom == d)).sum()):>11d}' for d in doms))
    for k, (pa, pt) in per_dom.items():
        lines.append(f'{k:10s}' + ''.join(f'{pa[d]:>11.2f}' for d in doms))
        lines.append(f'{"  top20":10s}' + ''.join(f'{pt[d]:>11.2f}' if d in pt else f'{"-":>11s}' for d in doms))
    noi = dom != 'isic'  # descriptive only, added before the test half was computed (preregistration section 5)
    lines += ['', 'descriptive (preregistration section 5): without ISIC',
              f'top20 n={(top & noi).sum()} (under {(y & top & noi).sum()}), all n={noi.sum()}',
              '| quantity | AUC top20 no ISIC [95% CI] | AUC all no ISIC [95% CI] |', '|---|---|---|']
    for k in ('back_out', 'sim_ring', 'graph_dir', 'MU', 'combo', 'slope'):
        a_t = boot(X[k][top & noi], y[top & noi], [p for p, t in zip(photos, top & noi) if t])
        a_a = boot(X[k][noi], y[noi], [p for p, t in zip(photos, noi) if t])
        lines.append(f'| {k} | {a_t[0]:.3f} [{a_t[1]:.3f}, {a_t[2]:.3f}] | {a_a[0]:.3f} [{a_a[1]:.3f}, {a_a[2]:.3f}] |')
    s = boot(X['slope'], top, photos)
    lines += ['', f'additional (outside the plan reading): slope as detector of top-20% headroom vs rest, AUC '
                  f'{s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]; without SUIM {auc(X["slope"][nos], top[nos]):.3f}']
    (OUT / 'measure_X.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def size_check():
    """Viewing half only, after the test round: is a quantity just a proxy of the (unknown) target size?"""
    from scipy.stats import spearmanr
    E = {e['index']: e for e in load_episodes()}
    F = [json.loads(l) for l in (OUT / 'explore_V.jsonl').read_text().splitlines()]
    ep = [E[f['index']] for f in F]
    gt = np.log10(np.array([e['gt_area'] / e['valid_area'] for e in ep]) + 1e-5)
    y = np.array([e['best_area'] > e['natural_area'] for e in ep])
    h = np.array([e['best'] - e['natural'] for e in ep])
    top = h >= np.quantile(h, .8)
    band = np.digitize(gt, np.quantile(gt, [1 / 3, 2 / 3]))
    print(f'GT size as a direction "predictor" (oracle): AUC all {auc(gt, y):.3f}, top20 {auc(gt[top], y[top]):.3f}; '
          f'under rate by GT tertile {[round(float(y[band == i].mean()), 2) for i in range(3)]}')
    for k in ('back_out', 'sim_ring', 'graph_dir', 'MU', 'nat_frac'):
        x = np.array([f[k] for f in F])
        print(f'{k:9s} Spearman with log GT size {spearmanr(x, gt).correlation:+.2f}; direction AUC within GT-size tertiles '
              f'{[round(auc(x[band == i], y[band == i]), 3) for i in range(3)]}')


if __name__ == '__main__':
    {'prepare': prepare, 'boards': boards, 'explore': explore, 'levels': levels, 'measure': measure,
     'size_check': size_check}[sys.argv[1]]()
