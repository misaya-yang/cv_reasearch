"""W14 follow-up: split interior / boundary errors into FN and FP from the stored counterfactual I/U, and
describe where the fused-field threshold gap sits.  Reads ../cv_data/a/x1013_W14/episodes.jsonl (seeded S subset).

    python scripts/x1012/w14_split.py
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT + '/scripts/x1013'); sys.path.insert(0, ROOT + '/scripts/x1012')
import common as c, score
import w14_headroom as w
E1 = {r['index']: r for r in map(json.loads, gzip.open(ROOT + '/evidence/research/x1013_E1_iu.jsonl.gz', 'rt'))}
keep = {str(b['index']) for b in w.subset([b for b in c.rows() if c.half(b) == 'S'], 300)}
R = [dict(E1[r['index']], W=r) for r in map(json.loads, open(str(w.OUT / 'episodes.jsonl'))) if r['index'] in keep]
D = c.DOMAINS; names = [d[:6] for d in D]
def parts(a):
    (I, U), (Ii, Ui), (Ib, Ub) = a['mask'], a['interior'], a['boundary']
    return dict(int_fn=Ii - I, int_fp=U - Ui, bnd_fn=Ib - I, bnd_fp=U - Ub, I=I, U=U)
for arm in ('MFU', 'FoRIS'):
    for e in R:
        p = parts(e['W'][arm]); e['_p'] = p
        e['_m'] = [p['I'], p['U']]
        e['_ifn'] = [p['I'] + p['int_fn'], p['U']]; e['_ifp'] = [p['I'], p['U'] - p['int_fp']]
        e['_bfn'] = [p['I'] + p['bnd_fn'], p['U']]; e['_bfp'] = [p['I'], p['U'] - p['bnd_fp']]
    print(f'\n## {arm}: gain from fixing one half of an error type (class mIoU)\n{"":12s}' + ''.join(f'{d:>8s}' for d in names) + '    mean')
    base = [score.miou([e for e in R if e['dataset'] == d], '_m') for d in D]
    for k, nm in (('_ifn', 'interior FN'), ('_ifp', 'interior FP'), ('_bfn', 'boundary FN'), ('_bfp', 'boundary FP')):
        v = [score.miou([e for e in R if e['dataset'] == d], k) - b for d, b in zip(D, base)]
        print(f'{nm:12s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}')
    print('px / GT')
    for k in ('int_fn', 'int_fp', 'bnd_fn', 'bnd_fp'):
        v = [100 * sum(e['_p'][k] for e in R if e['dataset'] == d) / max(sum(e['gt'] for e in R if e['dataset'] == d), 1) for d in D]
        print(f'{k:12s}' + ''.join(f'{x:8.1f}' for x in v))
    if arm == 'MFU':
        band = lambda e: '<2%' if e['gt'] / 2 ** 20 < .02 else ('2-20%' if e['gt'] / 2 ** 20 <= .2 else '>20%')
        iou = lambda x: 100 * x[0] / max(x[1], 1)
        print(f'\nby band (mean episode IoU gain){"":4s}' + ''.join(f'{k:>12s}' for k in ('int FN', 'int FP', 'bnd FN', 'bnd FP')))
        for b in ('<2%', '2-20%', '>20%'):
            L = [e for e in R if band(e) == b and e['gt'] > 0]
            print(f'{b:8s}{len(L):6d}' + ''.join(f'{np.mean([iou(e[k]) - iou(e["_m"]) for e in L]):+12.2f}' for k in ('_ifn', '_ifp', '_bfn', '_bfp')))
G = [e for e in R if 'G' in e['W']]
gap = np.array([e['W']['G']['oracle'] - e['W']['G']['natural'] for e in G])
under = np.array([e['W']['G']['oracle_area'] > e['W']['G']['natural_area'] for e in G])
print(f'\n## G gap direction: share of total gap where the oracle mask is LARGER than natural (under-coverage): {gap[under].sum() / gap.sum():.2f}')
top = gap >= np.quantile(gap, .8)
print(f'top-20% gap episodes: n={top.sum()}, mean gap {gap[top].mean():.1f}, natural IoU {np.mean([G[i]["W"]["G"]["natural"] for i in np.where(top)[0]]):.1f} '
      f'vs rest {np.mean([G[i]["W"]["G"]["natural"] for i in np.where(~top)[0]]):.1f}; under-coverage share {under[top].mean():.2f}')
band = lambda e: '<2%' if e['gt'] / 2 ** 20 < .02 else ('2-20%' if e['gt'] / 2 ** 20 <= .2 else '>20%')
for nm, f in (('band', band), ('comp', lambda e: min(e['W']['n_gt_comp'], 4))):
    vals = sorted({f(e) for e in G}, key=str)
    print(nm, {v: (round(float(np.mean([f(G[i]) == v for i in np.where(top)[0]])), 2), round(float(np.mean([f(e) == v for e in G])), 2)) for v in vals}, '(top share, all share)')
ratio = np.array([e['W']['G']['natural_area'] / max(e['gt'], 1) for e in G])
print('natural area / GT area in top-20%: quartiles', np.round(np.quantile(ratio[top], [.1, .25, .5, .75, .9]), 2))
print('natural area / GT area in rest:    quartiles', np.round(np.quantile(ratio[~top], [.1, .25, .5, .75, .9]), 2))
