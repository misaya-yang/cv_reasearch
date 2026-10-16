"""W49 correction check: C_g (bilinear up-sampling of the TRUE 64-grid coverage, best threshold) is only one field of
the readout family "64x64 scalar field -> bilinear -> one threshold", so it is a LOWER bound of that family's best,
not an upper bound.  Here: directly optimise a free 64x64 field (Adam on a soft IoU, threshold fixed at .5, which is
w.l.o.g. for a free field) and report the hard IoU it reaches.  Gradient descent gives a lower bound too.
Episodes: every S episode with C64 < .5, plus a seeded sample of 150 others.  S half only; GT is the target here.

    nice -n 19 python scripts/x1012/w49_field_opt.py produce [workers]
    python scripts/x1012/w49_field_opt.py report
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

OUT = c.task_path('W49') / 'field_opt.jsonl'


def optimise(gt):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    g = torch.from_numpy(gt.astype(np.float32))
    y = g.reshape(64, 16, 64, 16).mean((1, 3))
    f = y.clone().requires_grad_(True)                       # start from the true coverage (the C64 field)
    opt = torch.optim.Adam([f], lr=.05)
    up = lambda x: F.interpolate(x[None, None], size=(1024, 1024), mode='bilinear', align_corners=False)[0, 0]
    best = 0.
    for it in range(400):
        k = 10 * (20 ** (it / 399))                          # sharpen the surrogate from 10 to 200
        p = torch.sigmoid(k * (up(f) - .5))
        inter = (p * g).sum()
        loss = 1 - inter / (p.sum() + g.sum() - inter)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 49 or it == 399:
            with torch.no_grad():
                m = up(f) > .5
                i = float((m & (g > .5)).sum()); u = float((m | (g > .5)).sum())
                best = max(best, i / max(u, 1))
    return best


def one(b):
    import w31_demonstration as w31
    gt, valid = c.truth(b)
    return dict(w31.meta(b), gt=int(gt.sum()), opt64=optimise(gt))


def produce_all(workers):
    import w31_demonstration as w31
    w49 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w49_token_resolution_S.jsonl.gz', 'rt'))}
    rows = w31.rows()
    low = [b for b in rows if w49[str(b['index'])]['C64'] < .5]
    rest = [b for b in rows if w49[str(b['index'])]['C64'] >= .5]
    pick = low + [rest[i] for i in sorted(np.random.default_rng(2026101504).choice(len(rest), 150, replace=False))]
    done = {json.loads(l)['index'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [b for b in pick if str(b['index']) not in done]
    print('C64 < .5:', len(low), ' sample:', 150, ' todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool, \
            OUT.open('a') as fh:
        for r in pool.map(one, todo, chunksize=2):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done', flush=True)


def report():
    w49 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w49_token_resolution_S.jsonl.gz', 'rt'))}
    R = [json.loads(l) for l in OUT.read_text().splitlines()]
    for r in R:
        r['C64'] = w49[r['index']]['C64']
    for tag, sel in (('C64 < .5', [r for r in R if r['C64'] < .5]), ('C64 >= .5 (sample)', [r for r in R if r['C64'] >= .5])):
        a, b = np.array([r['C64'] for r in sel]), np.array([r['opt64'] for r in sel])
        print(f'{tag:20s} n {len(sel):4d}  mean C64 {a.mean():.3f}  mean optimised {b.mean():.3f}  '
              f'reach >= .5: {np.mean(b >= .5):.1%}  optimised > C64: {np.mean(b > a + 1e-6):.1%}')
        for d in c.DOMAINS:
            s = [r for r in sel if r['dataset'] == d]
            if s:
                print(f'   {d:15s} n {len(s):3d}  C64 {np.mean([r["C64"] for r in s]):.3f}  opt {np.mean([r["opt64"] for r in s]):.3f}')
    with gzip.open(ROOT / 'evidence/research/w49_field_opt_S.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in R)


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
