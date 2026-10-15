"""W21: a cheaper replacement for the final CRF of the FoRIS-free head Xp.  (S half)

Motivation (W19): the official CRF costs ~1.0 s of the 1.41 s head and adds only +0.19 (S) / +0.26 (T).
crf_refine builds a ~10 px band around the binary mask (core .95, band .5), runs a fully connected
permutohedral CRF over the whole 1024^2 image, and only keeps its decision inside the band.
Refinements of the same pre-CRF mask / field (parameters fixed before running, no tuning):
  Xp       bilinear field > .5, no refinement
  Xp+      official CRF at 1024 (W19 head)
  C640     the same official CRF run at 640 (INSID3's crf_size), band 6 px; the band decision is upsampled
  GFband   colour guided filter (He et al., r = 8, eps = 1e-3) of crf_refine's band probability map
           (core .95 / outside .05 / band .5); only band pixels may change, as in crf_refine
  GFsoft   the same guided filter of the soft fused field (.5 M + .5 U0, bilinear 1024), then > .5
Each refinement is timed (perf_counter, inside 3 parallel 2-thread workers; the chosen one is re-timed alone).
Reading fixed before running: a refinement replaces the CRF if its ten-domain mean is >= Xp+'s - 0.1 with
no domain significantly below Xp+, at <= 1/3 of the CRF's time; then T-half check.

    python scripts/x1012/w21_refine.py produce [workers]
    python scripts/x1012/w21_refine.py report
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

OUT = c.task_path('W21')
ARMS = ('Xp', 'Xp+', 'C640', 'GFband', 'GFsoft')
R_GF, EPS = 8, 1e-3


def box(x, r):
    """Mean over a (2r+1)^2 window clipped at the border (integral image, O(1) per pixel); x: (C, H, W)."""
    import torch
    def along(y, dim):
        n = y.shape[dim]
        cum = torch.cat([torch.zeros_like(y.narrow(dim, 0, 1)), y.cumsum(dim)], dim)
        hi = torch.clamp(torch.arange(n) + r + 1, max=n)
        lo = torch.clamp(torch.arange(n) - r, min=0)
        return cum.index_select(dim, hi) - cum.index_select(dim, lo), (hi - lo).to(y.dtype)
    s, nh = along(x, 1)
    s, nw = along(s, 2)
    return s / (nh[:, None] * nw[None, :])


def guided(I, p, r=R_GF, eps=EPS):
    """Colour guided filter (He, Sun, Tang 2013). I: (3, H, W) in [0, 1]; p: (H, W)."""
    import torch
    p = p[None]
    mI, mp = box(I, r), box(p, r)
    cov_Ip = box(I * p, r) - mI * mp
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    var = {ij: box(I[ij[0]:ij[0] + 1] * I[ij[1]:ij[1] + 1], r)[0] - mI[ij[0]] * mI[ij[1]] for ij in idx}
    a, b, cc, d, e, f = (var[(0, 0)] + eps, var[(0, 1)], var[(0, 2)], var[(1, 1)] + eps, var[(1, 2)], var[(2, 2)] + eps)
    # inverse of the symmetric 3x3 [[a b c] [b d e] [c e f]] via the adjugate
    A, B, C = d * f - e * e, cc * e - b * f, b * e - cc * d
    D, E, Fm = a * f - cc * cc, b * cc - a * e, a * d - b * b
    det = a * A + b * B + cc * C
    inv = torch.stack([torch.stack([A, B, C]), torch.stack([B, D, E]), torch.stack([C, E, Fm])]) / det
    coef = torch.einsum('ijhw,jhw->ihw', inv, cov_Ip)
    bias = mp[0] - (coef * mI).sum(0)
    return (box(coef, r) * I).sum(0) + box(bias[None], r)[0]


def band_map(mask, band_px=10, p_core=.95):
    import torch
    import torch.nn.functional as F
    r = max(1, band_px // 2)
    ys, xs = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing='ij')
    kernel = ((xs ** 2 + ys ** 2) <= r * r).float()[None, None]
    y = F.conv2d(mask.float()[None, None], kernel, padding=r)[0, 0]
    dil, ero = y > 0, y == float(kernel.sum())
    prob = torch.where(ero, torch.tensor(p_core), torch.tensor(1 - p_core))
    prob[dil ^ ero] = .5
    return prob, dil ^ ero, ero


def one(binding):
    import torch
    import torch.nn.functional as F
    import produce
    from w09_integration_lambda import solve
    from w19_simple_head import heads, mean_graph, mean_solve
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index']), 'sec': {}}
    if produce.fallback(fields):
        with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
            t_mask = c.unpack(z['T'])
        for a in ARMS:
            r[a] = c.iu(t_mask, gt, valid)
        return r
    mm = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32)))).numpy().astype(np.float64)
    u0 = mm(solve(fields['c_mod'], fields['log_e'], c.laplacian(fields), 16).reshape(64, 64))
    q, _, _, guide, cheap = heads(binding, fields)
    soft = .5 * c.up(mean_solve(cheap['p'].reshape(64, 64), guide, mean_graph(q))).astype(np.float64) + .5 * u0
    mask = soft > .5
    masks = {'Xp': mask}
    with c.rc.pc.inference_only(binding):
        t = time.perf_counter(); masks['Xp+'] = produce.crf(binding, mask); r['sec']['Xp+'] = time.perf_counter() - t
        from ics.official_data import decoded_rgb
        from utils.refinement import crf_refine
        from utils.data import denormalize
        host = produce.CRF_HOST
        host.set_target(decoded_rgb(c.ASSETS / binding['row']['query_path'], binding['row'].get('query_crop')))
        image = host._tgt_image.clone()
        host._tgt_image = host._orig_tgt_size = None
        t = time.perf_counter()
        small = F.interpolate(image[None], size=(640, 640), mode='bilinear', align_corners=False)
        m640 = F.interpolate(torch.from_numpy(mask).float()[None, None], size=(640, 640), mode='nearest')[0, 0] > .5
        out = crf_refine(host._crf, 6, host._crf_p_core, small, m640)
        up = F.interpolate(out.float()[None, None], size=(1024, 1024), mode='nearest')[0, 0].numpy() > .5
        _, band, core = band_map(torch.from_numpy(mask))
        masks['C640'] = np.where(band.numpy(), up, core.numpy())
        r['sec']['C640'] = time.perf_counter() - t
        rgb = denormalize(image).clamp(0, 1).float()
    t = time.perf_counter()
    prob, band, core = band_map(torch.from_numpy(mask))
    filt = guided(rgb, prob)
    masks['GFband'] = np.where(band.numpy(), (filt > .5).numpy(), core.numpy())
    r['sec']['GFband'] = time.perf_counter() - t
    t = time.perf_counter()
    masks['GFsoft'] = (guided(rgb, torch.from_numpy(soft).float()) > .5).numpy()
    r['sec']['GFsoft'] = time.perf_counter() - t
    for a in ARMS:
        r[a] = c.iu(masks[a], gt, valid)
    return r


def produce_all(workers):
    import w13_plugin as w13
    OUT.mkdir(parents=True, exist_ok=True)
    rows = w13.subset([b for b in c.rows() if c.half(b) == 'S'], 150)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    with gzip.open(ROOT / 'evidence/research/w19_simple_head_S.jsonl.gz', 'rt') as f:
        W19 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    same = np.mean([e['Xp+'][:2] == W19[e['index']]['Xp+'][:2] for e in R])
    print('n', len(R), ' Xp+ identical to W19:', same)
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]
    print(f'\n## class mIoU minus Xp+ (official CRF)\n{"arm":7s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0  sig-')
    for a in ('Xp', 'C640', 'GFband', 'GFsoft'):
        v, cells, sn = [], [], 0
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            x = score.miou(L, a) - score.miou(L, 'Xp+'); lo, hi = score.paired_ci(L, a, 'Xp+', rng)
            v.append(x); sn += hi < 0; cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]')
        print(f'{a:7s}' + ''.join(f'{s:>17s}' for s in cells) + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}{sn:6d}')
    T = [e for e in R if e['sec']]
    print('\nmean seconds per refinement (3 parallel workers, 2 threads each):',
          {a: round(float(np.mean([e['sec'][a] for e in T])), 3) for a in ('Xp+', 'C640', 'GFband', 'GFsoft')})
    with gzip.open(ROOT / 'evidence/research/w21_refine_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'sec') + ARMS}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
