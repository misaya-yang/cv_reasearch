"""W38: can the image give the amount (coverage) of each boundary-band token that the field cannot?  S half only.
Plan and readings: docs/research/x1012_tasks/W38_band_amount_PLAN.md.  Same episodes, field, natural mask and band as
W37 (functions imported).  Amount per band token: nat, crf (SH's CRF on N), rgb (max colour separation along the
F order, gated by the episode median separation), rgb_all (ungated), true (GT, capacity).  Reconstruction = W37 R2.

    nice -n 19 python scripts/x1012/w38_band_amount.py produce [workers]
    python scripts/x1012/w38_band_amount.py report
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

OUT = c.task_path('W38')
ARMS = ('nat', 'crfA', 'rgb', 'rgb_all', 'true')
EST = ('crfA', 'rgb', 'rgb_all')


def blocks(x, B):
    """(64*16, 64*16, ...) -> (number of band tokens, 256, ...) in row-major token order."""
    ii, jj = np.nonzero(B)
    if not len(ii):
        return np.zeros((0, 256) + x.shape[2:], x.dtype), (ii, jj)
    return np.stack([x[16 * i:16 * i + 16, 16 * j:16 * j + 16].reshape(256, *x.shape[2:]) for i, j in zip(ii, jj)]), (ii, jj)


def place(N, F, B, k):
    """Band blocks take the k highest-F pixels (stable order); elsewhere N."""
    out = N.copy()
    fb, (ii, jj) = blocks(F, B)
    order = np.argsort(-fb, axis=1, kind='stable')
    sel = np.zeros_like(fb, dtype=bool)
    rank = np.empty_like(order); rows = np.arange(len(order))[:, None]
    rank[rows, order] = np.arange(256)[None, :]
    sel = rank < np.asarray(k)[:, None]
    for t, (i, j) in enumerate(zip(ii, jj)):
        out[16 * i:16 * i + 16, 16 * j:16 * j + 16] = sel[t].reshape(16, 16)
    return out


def one(b):
    import produce
    from PIL import Image
    from ics.official_data import decoded_rgb
    import w31_demonstration as w31
    import w37_coverage as w37
    began = time.monotonic()
    F, N, rho, _, _ = w37.capture(b)
    nT = w37.pool16(N.astype(np.float64))
    B = w37.band(nT)
    row = b['row']
    rgb = np.asarray(decoded_rgb(c.ASSETS / row['query_path'], row.get('query_crop')).resize((1024, 1024), Image.BILINEAR),
                     np.float64) / 255
    if rho > .9:
        crf = N
    else:
        with c.rc.pc.inference_only(b):
            crf = produce.crf(b, N)
    fb, _ = blocks(F, B)
    cb, _ = blocks(rgb, B)
    order = np.argsort(-fb, axis=1, kind='stable')
    cs = np.take_along_axis(cb, order[:, :, None], axis=1)
    S1 = np.cumsum(cs, axis=1); T = S1[:, -1:, :]
    k = np.arange(1, 256)[None, :]
    mt = S1[:, :255] / k[..., None]; mr = (T - S1[:, :255]) / (256 - k)[..., None]
    sep = (k * (256 - k) / 256 ** 2) * ((mt - mr) ** 2).sum(-1)
    kstar = sep.argmax(1) + 1 if len(sep) else np.zeros(0, int); smax = sep.max(1) if len(sep) else np.zeros(0)
    kn = blocks(N[..., None], B)[0][..., 0].sum(1)
    amounts = {'nat': kn, 'crfA': blocks(crf[..., None], B)[0][..., 0].sum(1),
               'rgb': np.where(smax >= np.median(smax), kstar, kn) if len(smax) else kn, 'rgb_all': kstar}
    masks = {a: place(N, F, B, amounts[a]) for a in amounts}
    gt, valid = c.truth(b)                      # read only after every label-free amount is fixed
    amounts['true'] = blocks(gt[..., None], B)[0][..., 0].sum(1)
    masks['true'] = place(N, F, B, amounts['true'])
    y = (amounts['true'] - kn).astype(np.float64)
    r = dict(w31.meta(b), gt=int(gt.sum()), rho=rho, fallback=rho > .9, band_tokens=int(B.sum()),
             gate_used=float(np.mean(smax >= np.median(smax))) if len(smax) else 0.)
    for a in ARMS:
        r[a] = c.iu(masks[a], gt, valid)
    for e in EST:
        x = (amounts[e] - kn).astype(np.float64)
        r['stat_' + e] = [len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()]
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


def pearson(st):
    n, sx, sy, sxy, sxx, syy = np.sum(st, axis=0)
    cov = sxy / n - sx * sy / n ** 2
    return cov / np.sqrt(max(sxx / n - (sx / n) ** 2, 1e-12) * max(syy / n - (sy / n) ** 2, 1e-12))


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    for r in R:
        r['SH+'] = w27[r['index']]['SH+']
    same = sum(r['nat'][:2] == w23[r['index']]['SH'][:2] for r in R)
    doms = list(c.DOMAINS)
    rng = np.random.default_rng(c.SEED)
    rhos = {}
    for e in EST:
        st = np.array([r['stat_' + e] for r in R if r['stat_' + e][0] > 0])
        boot = [pearson(st[rng.integers(0, len(st), len(st))]) for _ in range(500)]
        rhos[e] = (pearson(st), *np.percentile(boot, [2.5, 97.5]))
    L = [f'n {len(R)}  implementation check: nat == W23 SH (pre-CRF) {same}/{len(R)} (need >= 1280)',
         f'gate used k* on {100 * np.mean([r["gate_used"] for r in R]):.1f}% of band tokens (episode mean)', '',
         '## local T8 correlation rho(k_est - k_nat, k_true - k_nat) over band tokens, episode-cluster bootstrap 95%']
    L += [f'{e:8s} rho {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]' for e, v in rhos.items()]
    L += ['', '## class mIoU minus nat (pre-CRF); SH+ column = W27 SH+ minus nat', f'{"":12s}' + ''.join(f'{a:>9s}' for a in ARMS[1:] + ('SH+',))]
    mean = {a: [] for a in ARMS[1:] + ('SH+',)}
    for d in doms:
        D = [r for r in R if r['dataset'] == d]
        cells = []
        for a in ARMS[1:] + ('SH+',):
            v = sc.miou(D, a) - sc.miou(D, 'nat'); mean[a].append(v); cells.append(f'{v:+9.2f}')
        lo, hi = sc.paired_ci(D, 'rgb', 'nat', np.random.default_rng(c.SEED))
        L.append(f'{d[:11]:12s}' + ''.join(cells) + f'   rgb CI [{lo:+.2f},{hi:+.2f}]')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):+9.2f}' for a in mean))
    lo, hi = np.percentile(np.mean([w37._boot(sc, [r for r in R if r['dataset'] == d], 'rgb', 'nat') for d in doms], axis=0), [2.5, 97.5])
    g = np.mean(mean['rgb']); pos = sum(x >= 0 for x in mean['rgb'])
    ok = rhos['rgb'][0] >= .25 and rhos['rgb'][1] > 0 and g >= 1.0 and lo > 0 and pos >= 7 and rhos['rgb'][0] > rhos['crfA'][0]
    L += ['', f'reading: rho_rgb {rhos["rgb"][0]:+.3f} (>= .25, lower > 0), rgb - nat {g:+.2f} CI [{lo:+.2f},{hi:+.2f}] '
              f'domains >= 0 {pos}/10, rho_rgb > rho_crf ({rhos["crfA"][0]:+.3f}) -> pass: {bool(ok)}',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w38_band_amount_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback',
                                                  'band_tokens', 'gate_used') + ARMS + tuple('stat_' + e for e in EST)}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
