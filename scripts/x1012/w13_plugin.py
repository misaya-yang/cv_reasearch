"""W13: how much of MFU's gain depends on FoRIS-specific parts?  (selection half S only)

Motivation: MFU = rho>.9 ? T : CRF([.5 M + .5 U > .5]).  The M line is the FoRIS score refined by
MEAN; U borrows FoRIS's vote V and seed-cluster prior P; the CRF is FoRIS's.  The user asks whether
the win is a property of fusing two integration scales, or of these particular FoRIS parts.
Pre-stated arms (rho > .9 keeps T for every arm, as MFU does):
  FU   = [.5 F + .5 U > .5]      F = minmax(up(FoRIS score)): fuse U with FoRIS itself instead of MEAN
  M0U  = [.5 M + .5 U0 > .5]     U0 = our solve on c_mod only (no FoRIS vote, no FoRIS prior)
  U0   = minmax(up(U0)) > .5     the FoRIS-free single line
  G    = [.5 M + .5 U > .5]      MFU before its CRF (identity check against E1 MFU after CRF)
each also after the same official CRF (suffix +).  Readings:
  FU+ >= FoRIS in every domain and close to MFU  -> the gain is the fusion, MEAN is not needed;
  M0U+ close to MFU                              -> U does not need FoRIS's V/P inside the fusion;
  U0 vs candidate                                -> what V/P buy for the single line.
Offline: T0 fields, T1 MEAN field, cached FoRIS score; no encoder.  T half is not read.

    python scripts/x1012/w13_plugin.py produce [workers]
    python scripts/x1012/w13_plugin.py report
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W13')
ARMS = ('G', 'FU', 'FU+', 'M0U', 'M0U+', 'U0', 'U0+')


def subset(rows, cap):
    """Seeded per-domain sample of at most `cap` episodes, fixed before any result is read (diagnostics need
    no full pass; the full S half took ~1 h per arm set and saturated the host)."""
    rng, out = np.random.default_rng(c.SEED), []
    for d in c.DOMAINS:
        L = sorted((b for b in rows if b['dataset'] == d), key=lambda b: str(b['index']))
        out += [L[i] for i in sorted(rng.choice(len(L), min(cap, len(L)), replace=False))]
    return out


def one(binding):
    import torch
    import produce
    from w09_integration_lambda import solve
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index'])}
    with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
        t_mask = c.unpack(z['T'])
    if produce.fallback(fields):
        for a in ARMS:
            r[a] = c.iu(t_mask, gt, valid)
        return r
    mm = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32)))).numpy().astype(np.float64)
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    u = mm(fields['VP_UN'])
    u0 = mm(solve(fields['c_mod'], fields['log_e'], c.laplacian(fields), 16).reshape(64, 64))
    f = mm(np.load(binding['score64']['path'])[binding['score64']['key']])
    masks = {'G': .5 * m + .5 * u > .5, 'FU': .5 * f + .5 * u > .5, 'M0U': .5 * m + .5 * u0 > .5, 'U0': u0 > .5}
    with c.rc.pc.inference_only(binding):
        for a in ('FU', 'M0U', 'U0'):
            masks[a + '+'] = produce.crf(binding, masks[a])
    for a in ARMS:
        r[a] = c.iu(masks[a], gt, valid)
    return r


def produce_all(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = subset([b for b in c.rows() if c.half(b) == 'S'], 150)
    done = set()
    if (OUT / 'episodes.jsonl').exists():
        done = {json.loads(l)['index'] for l in (OUT / 'episodes.jsonl').read_text().splitlines()}
    todo = [b for b in rows if str(b['index']) not in done]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, (OUT / 'episodes.jsonl').open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 250 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    keep = {str(b['index']) for b in subset([b for b in c.rows() if c.half(b) == 'S'], 150)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines()) if r['index'] in keep]
    print('n', len(R), 'S only', all(r['split'] == 'S' for r in R))
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]

    def table(title, ref, rows):
        print(f'\n## {title}\n{"arm":10s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0  sig+  sig-')
        for a in rows:
            v, cells, sp, sn = [], [], 0, 0
            for d in c.DOMAINS:
                L = [r for r in R if r['dataset'] == d]
                x = score.miou(L, a) - score.miou(L, ref)
                lo, hi = score.paired_ci(L, a, ref, rng)
                sp += lo > 0; sn += hi < 0; v.append(x)
                cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]')
            print(f'{a:10s}' + ''.join(f'{s:>17s}' for s in cells) + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}{sp:6d}{sn:6d}')

    table('vs complete FoRIS', 'foris', ('MFU', 'FU+', 'M0U+', 'C1', 'U0+', 'M+', 'FU', 'M0U', 'G', 'candidate', 'U0'))
    table('vs MFU', 'MFU', ('FU+', 'M0U+'))
    table('U0 vs candidate (what FoRIS vote and prior buy for the single line)', 'candidate', ('U0',))
    print(f'\n## mean episode-IoU difference vs complete FoRIS by GT area\n{"arm":10s}{"<2%":>8s}{"2-20%":>8s}{">20%":>8s}')
    bands = {b: [r for r in R if score.area_band(r) == b] for b in ('<2%', '2-20%', '>20%')}
    for a in ('MFU', 'FU+', 'M0U+', 'C1', 'U0+', 'M+'):
        print(f'{a:10s}' + ''.join(f'{np.mean([score.episode_iou(r, a) - score.episode_iou(r, "foris") for r in L]):+8.2f}'
                                   for L in bands.values()))
    with gzip.open(ROOT / 'evidence/research/w13_plugin_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt') + ARMS}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
