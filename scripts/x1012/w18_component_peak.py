"""W18: does a relative-peak rule remove distractor components without removing true instances?  (S half)

Motivation (W17): distractor components of G > .5 peak at a median 0.62 of the strongest true component and
only 15 % reach 0.9; missed instances sit at background level (axis closed).  Whole-distractor FP is worth
+2.2 class mIoU at perfect removal (W14, after CRF).  At test time the strongest component is unknown, so the
rule uses the strongest predicted component: drop every connected component of G > .5 whose peak is below
tau x the episode's maximum component peak.  tau in {.6, .7, .8} (pre-CRF, W15 sample).
Reading fixed before running: best tau >= +0.5 class mIoU mean over domains with >= 7 of 9 domains >= 0 ->
test the rule inside MFU (before the CRF) on a fresh S subset, then T; otherwise close the distractor axis.

    python scripts/x1012/w18_component_peak.py
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

OUT = c.task_path('W18')
TAUS = (.6, .7, .8)


def one(binding):
    import torch
    from scipy import ndimage
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    g = gt & valid
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
        m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
    u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
    f = .5 * m + .5 * u
    nat = (f > .5) & valid
    lab, n = ndimage.label(nat, np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    px = ndimage.sum(np.ones_like(f), lab, idx) if n else np.zeros(0)
    ov = ndimage.sum(g.astype(np.float64), lab, idx) if n else np.zeros(0)
    pk = ndimage.maximum(f, lab, idx) if n else np.zeros(0)
    return {'index': str(binding['index']), 'I': int((nat & g).sum()), 'U': int((nat | g).sum()),
            'comps': [[int(a), int(b), float(p)] for a, b, p in zip(px, ov, pk)]}


def main():
    import w15_query_extent as w15
    import score
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'episodes.jsonl').exists():
        R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    else:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
            R = list(pool.map(one, w15.sample(), chunksize=4))
        (OUT / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in R))
    E1 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt'))}
    R = [dict(E1[r['index']], **r) for r in R]
    for e in R:
        e['nat'] = [e['I'], e['U']]
        top = max([p for _, _, p in e['comps']] or [1])
        for t in TAUS:
            drop = [(a, b) for a, b, p in e['comps'] if p < t * top]
            e[f'tau{t}'] = [e['I'] - sum(b for _, b in drop), e['U'] - sum(a - b for a, b in drop)]
    doms = [d for d in c.DOMAINS if any(e['dataset'] == d for e in R)]
    print(f'{"":8s}' + ''.join(f'{d[:6]:>8s}' for d in doms) + '    mean  >=0')
    for t in TAUS:
        v = [score.miou([e for e in R if e['dataset'] == d], f'tau{t}') - score.miou([e for e in R if e['dataset'] == d], 'nat') for d in doms]
        print(f'tau {t:<4}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}{sum(x >= 0 for x in v):5d}')
    comps = [(b > 0, p / max([q for _, _, q in e['comps']]), a) for e in R for a, b, p in e['comps'] if len(e['comps']) > 1]
    for t in TAUS:
        tr = [a for ok, r, a in comps if ok and r < t]; fa = [a for ok, r, a in comps if not ok and r < t]
        print(f'tau {t}: removed true-touching comps {len(tr)} ({sum(tr)} px), removed distractors {len(fa)} ({sum(fa)} px)')


if __name__ == '__main__':
    main()
