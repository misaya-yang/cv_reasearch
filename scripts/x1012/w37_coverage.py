"""W37: how much is the boundary band worth if each band token had the right coverage, and how much do two label-free
coverage estimates recover?  S half only.  Plan and readings: docs/research/x1012_tasks/W37_coverage_PLAN.md.

Only pixels in the 16x16 blocks of boundary-band tokens of the natural mask N = F > .5 are changed.  Coverage sources
in the band: O (GT 16x16 mean, capacity diagnostic), C (kernel transfer c32), U (linear unmixing of the debiased query
tokens between the core mean and the background mean).  Reconstructions: R1 bilinear coverage > .5, R2 per-patch
top-k by F, R3 per-patch top-k by a label-free colour log-odds.  Primary deployable arm U3; U3+ adds SH's CRF.

    nice -n 19 python scripts/x1012/w37_coverage.py produce [workers]
    python scripts/x1012/w37_coverage.py report
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

OUT = c.task_path('W37')
ARMS = ('nat',) + tuple(f'{s}{k}' for s in 'OCU' for k in (1, 2, 3)) + ('U3+',)


def init():
    c.initialize()
    import torch
    torch.set_num_threads(1)


def capture(b):
    """F, natural mask and the pieces of the frozen simple_head (same capture as W31 K0)."""
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
    _, q, _, _, f = cap['ind'][0]
    c32 = f['c32'].reshape(64, 64).astype(np.float64)
    if rho > .9:
        F = lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))).numpy().astype(np.float64)
    else:
        u0 = lab.minmax(lab.up(torch.from_numpy(cap['U0'][0].reshape(64, 64).astype(np.float32)))).numpy()
        m = lab.up(torch.from_numpy(cap['M'][0])).numpy()
        F = .5 * m.astype(np.float64) + .5 * u0.astype(np.float64)
    assert np.array_equal(F > .5, pre), 'captured field does not reproduce SH'
    return F, pre, rho, c32, q.numpy().astype(np.float64)


def pool16(x):
    return x.reshape(64, 16, 64, 16).mean((1, 3))


def band(nT):
    maj = nT >= .5
    p = np.pad(maj, 1, mode='edge')
    win = np.stack([p[i:i + 64, j:j + 64] for i in range(3) for j in range(3)])
    return (win.any(0) & ~win.all(0)) | ((nT > 0) & (nT < 1))


def colour_logodds(b, F):
    from PIL import Image
    from ics.official_data import decoded_rgb
    row = b['row']
    rgb = np.asarray(decoded_rgb(c.ASSETS / row['query_path'], row.get('query_crop')).resize((1024, 1024), Image.BILINEAR))
    idx = (rgb[..., 0] // 16).astype(np.int64) * 256 + (rgb[..., 1] // 16) * 16 + rgb[..., 2] // 16
    fg, bg = F > .75, F < .25
    if not fg.any() or not bg.any():
        return None
    hf = np.bincount(idx[fg], minlength=4096) + 1.
    hb = np.bincount(idx[bg], minlength=4096) + 1.
    return (np.log(hf / hf.sum()) - np.log(hb / hb.sum()))[idx]


def unmix(q, F, c32):
    t = pool16(F).ravel()
    core, back = t >= .75, t <= .25
    if core.sum() < 4 or back.sum() < 4:
        return c32
    a, bb = q[core].mean(0), q[back].mean(0)
    d = a - bb
    return np.clip(((q - bb) @ d) / max(float(d @ d), 1e-12), 0, 1).reshape(64, 64)


def rebuild(N, nT, B, pi, how, score):
    """Replace only band blocks; how in {1: bilinear, 2/3: per-patch top-k by `score`}."""
    import torch
    out = N.copy()
    full = np.where(B, pi, nT)
    if how == 1:
        up = c.rc.lab.up(torch.from_numpy(full.astype(np.float32))).numpy() > .5
        blk = np.repeat(np.repeat(B, 16, 0), 16, 1)
        out[blk] = up[blk]
        return out
    for i, j in zip(*np.nonzero(B)):
        k = int(round(256 * float(pi[i, j])))
        s = score[16 * i:16 * i + 16, 16 * j:16 * j + 16].ravel()
        blk = np.zeros(256, bool)
        if k > 0:
            blk[np.argsort(-s, kind='stable')[:k]] = True
        out[16 * i:16 * i + 16, 16 * j:16 * j + 16] = blk.reshape(16, 16)
    return out


def one(b):
    import produce
    import w31_demonstration as w31
    began = time.monotonic()
    F, N, rho, c32, q = capture(b)
    nT = pool16(N.astype(np.float64))
    B = band(nT)
    col = colour_logodds(b, F)
    col = F if col is None else col
    masks = {'nat': N}
    srcs = {'C': c32, 'U': unmix(q, F, c32)}
    gt, valid = c.truth(b)                      # read only after all label-free sources are fixed
    srcs['O'] = pool16(gt.astype(np.float64))
    for s, pi in srcs.items():
        masks[f'{s}1'] = rebuild(N, nT, B, pi, 1, None)
        masks[f'{s}2'] = rebuild(N, nT, B, pi, 2, F)
        masks[f'{s}3'] = rebuild(N, nT, B, pi, 3, col)
    if rho > .9:
        masks['U3+'] = masks['U3']
    else:
        with c.rc.pc.inference_only(b):
            masks['U3+'] = produce.crf(b, masks['U3'])
    r = dict(w31.meta(b), gt=int(gt.sum()), rho=rho, fallback=rho > .9, band_tokens=int(B.sum()),
             band_pixel_fraction=float(np.repeat(np.repeat(B, 16, 0), 16, 1).mean()), seconds=0.)
    for a in ARMS:
        r[a] = c.iu(masks[a], gt, valid)
    # calibration of c against true coverage on GT boundary patches (0 < pi_T < 1)
    piT = srcs['O']; edge = (piT > 0) & (piT < 1)
    r['calib'] = np.histogram2d(piT[edge], np.clip(c32[edge], 0, 1), bins=10, range=[[0, 1], [0, 1]])[0].astype(int).tolist()
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


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
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    for r in R:
        r['SH+'] = w27[r['index']]['SH+']
    same = sum(r['nat'][:2] == w23[r['index']]['SH'][:2] for r in R)
    doms = list(c.DOMAINS)
    rng = lambda: np.random.default_rng(c.SEED)
    L = [f'n {len(R)}  implementation check: nat == W23 SH (pre-CRF) {same}/{len(R)} (need >= 1280)',
         f'band pixel fraction: median {np.median([r["band_pixel_fraction"] for r in R]):.3f}',
         '', '## class mIoU minus nat (pre-CRF); U3+ minus SH+ (post-CRF, W27)', f'{"":12s}' + ''.join(f'{a:>9s}' for a in ARMS[1:])]
    mean = {a: [] for a in ARMS[1:]}
    ci_u3 = []
    for d in doms:
        D = [r for r in R if r['dataset'] == d]
        cells = []
        for a in ARMS[1:]:
            base = 'SH+' if a == 'U3+' else 'nat'
            v = sc.miou(D, a) - sc.miou(D, base); mean[a].append(v); cells.append(f'{v:+9.2f}')
        ci_u3.append(sc.paired_ci(D, 'U3', 'nat', rng()))
        L.append(f'{d[:11]:12s}' + ''.join(cells) + f'   U3 CI [{ci_u3[-1][0]:+.2f},{ci_u3[-1][1]:+.2f}]')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):+9.2f}' for a in ARMS[1:]))
    cap = max(np.mean(mean[a]) for a in ('O1', 'O2', 'O3'))
    u3 = np.mean(mean['U3']); pos = sum(x >= 0 for x in mean['U3'])
    allR = R
    lo, hi = np.percentile(np.mean([np.array(_boot(sc, [r for r in allR if r['dataset'] == d], 'U3', 'nat')) for d in doms], axis=0), [2.5, 97.5])
    L += ['', f'reading 1 capacity max(O1,O2,O3) - nat = {cap:+.2f}  (>= +5 continue; < +2 close; else thin-structure only, O3-O1 = '
          + ', '.join(f'{d[:6]} {mean["O3"][k] - mean["O1"][k]:+.2f}' for k, d in enumerate(doms) if d in ('fundus', 'deepglobe_road')) + ')',
          f'reading 2 U3 - nat = {u3:+.2f}, ten-domain paired CI [{lo:+.2f},{hi:+.2f}], domains >= 0: {pos}/10  '
          f'-> pass: {bool(u3 >= 1.0 and lo > 0 and pos >= 7)}',
          f'U3+ - SH+ (post-CRF) ten-domain = {np.mean(mean["U3+"]):+.2f}']
    H = np.sum([np.array(r['calib']) for r in R if 'calib' in r], axis=0)
    L += ['', '## calibration of c on GT boundary patches: rows true coverage decile, columns c decile (row-normalised %)']
    for k, row in enumerate(H):
        L.append(f'pi {k / 10:.1f}-{(k + 1) / 10:.1f} ' + ' '.join(f'{100 * x / max(row.sum(), 1):5.1f}' for x in row))
    L.append(f'\nseconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w37_coverage_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback',
                                                  'band_tokens', 'band_pixel_fraction') + ARMS}) + '\n')


def _boot(sc, records, a, b):
    """Per-draw domain differences with the same photo bootstrap as score.paired_ci (for the ten-domain mean CI)."""
    from scipy import sparse
    rng = np.random.default_rng(c.SEED)
    photos = list(dict.fromkeys(r['photo'] for r in records))
    slots = list(dict.fromkeys((r['fold'], r['cls']) for r in records))
    p = [photos.index(r['photo']) for r in records]; s = [slots.index((r['fold'], r['cls'])) for r in records]
    w = np.stack([np.bincount(rng.integers(len(photos), size=len(photos)), minlength=len(photos)) for _ in range(500)])
    pres = sparse.csr_matrix((np.ones(len(records)), (p, s)), shape=(len(photos), len(slots)))
    rep = w @ pres > 0
    out = []
    for arm in (a, b):
        i = sparse.csr_matrix(([r[arm][0] for r in records], (p, s)), shape=(len(photos), len(slots)))
        u = sparse.csr_matrix(([r[arm][1] for r in records], (p, s)), shape=(len(photos), len(slots)))
        v = 100 * (w @ i) / np.maximum(w @ u, 1)
        out.append((v * rep).sum(1) / rep.sum(1))
    return out[0] - out[1]


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
