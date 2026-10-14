"""W16: on the large-gap episodes, are M and U wrong in the same direction?  (S half, GT diagnostic, W15 sample)

Motivation: W15 closed query-only grouping; the extent sits in G's ranking but the level is off by >= 2x in
~20 % of episodes, and level-selection cues are closed (TOKEN_READOUT_CEILING_20261012.md).  The last cheap
place the level could hide is the disagreement between the two lines that G averages.
Reading fixed before running: if on the top-gap episodes best-of{M, U} (natural levels) recovers < 1/3 of the
G gap, and the two lines' area errors share their sign in most of them, no line-choice rule can reach the gap
and the threshold headroom is closed for our readouts; otherwise line reliability is the next question.

    python scripts/x1012/w16_line_agreement.py
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W16')


def one(binding):
    import torch
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    g = gt & valid
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
    r = {'index': str(binding['index']), 'gt': int(g.sum())}
    for k, mask in (('M', m > .5), ('U', u > .5), ('G', .5 * m + .5 * u > .5)):
        mask &= valid
        r[k] = [int((mask & g).sum()), int((mask | g).sum()), int(mask.sum())]
    return r


def main():
    import w15_query_extent as w15
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'episodes.jsonl').exists():
        R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    else:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
            R = list(pool.map(one, w15.sample(), chunksize=4))
        (OUT / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in R))
    W = {r['index']: r for r in map(json.loads, w15.W14.read_text().splitlines()) if 'G' in r}
    cut = np.quantile([w['G']['oracle'] - w['G']['natural'] for w in W.values()], .8)
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    for e in R:
        e['orc'] = W[e['index']]['G']['oracle']
        e['top'] = W[e['index']]['G']['oracle'] - W[e['index']]['G']['natural'] >= cut
    for nm, L in (('top-gap', [e for e in R if e['top']]), ('rest', [e for e in R if not e['top']]), ('all', R)):
        G, M, U = (np.array([iou(e[k]) for e in L]) for k in ('G', 'M', 'U'))
        O = np.array([e['orc'] for e in L])
        best = np.maximum(M, U)
        lm = np.log2(np.array([max(e['M'][2], 1) / e['gt'] for e in L]))
        lu = np.log2(np.array([max(e['U'][2], 1) / e['gt'] for e in L]))
        lg = np.log2(np.array([max(e['G'][2], 1) / e['gt'] for e in L]))
        big = np.abs(lg) >= 1
        print(f'{nm:8s} n={len(L):3d}  G {G.mean():5.1f}  M {M.mean():5.1f}  U {U.mean():5.1f}  best(M,U) {best.mean():5.1f}  '
              f'oracle {O.mean():5.1f}  share of G gap recovered by best(M,U) {(best - G).mean() / max((O - G).mean(), 1e-9):.2f}')
        print(f'{"":8s} |log2 area err| >= 1 for G: {big.mean():.2f}; among them M and U err in the same direction: '
              f'{np.mean(np.sign(lm[big]) == np.sign(lu[big])):.2f}; both also >= 1 octave off: {np.mean((np.abs(lm[big]) >= 1) & (np.abs(lu[big]) >= 1)):.2f}; '
              f'under-coverage share {np.mean(lg[big] < 0):.2f}')


if __name__ == '__main__':
    main()
