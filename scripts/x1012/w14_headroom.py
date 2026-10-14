"""W14: where is the ~10-point oracle headroom?  Error anatomy of the final masks (selection half S).

Motivation: per-episode GT-optimal levels put the candidate at 62.8 vs 52.4 natural (T1, S half), FoRIS
61.2 vs 50.1.  Earlier work closed constant / Otsu / area-estimator / supervised-fusion cuts
(TOKEN_READOUT_CEILING_20261012.md), so this does not test another cut.  It asks which pixels the
natural masks get wrong, so that the next mechanism targets the dominant error type:
  distractor  FP in predicted components that touch no GT pixel          (wrong object / region)
  missed      GT components (8-conn) with no predicted pixel               (unfound instances)
  boundary    errors within 16 px (one token) of the GT boundary
  interior    the rest: attached over-extension far from the boundary, holes / partial extent
For each type the counterfactual "fix only this type" gives its IoU cost.  Also, on MFU's fused field
G = .5 M + .5 U (rho <= .9), the per-episode GT-optimal level t* and its direction.
Diagnostic only (uses GT); S half; no encoder.

    python scripts/x1012/w14_headroom.py produce [workers]
    python scripts/x1012/w14_headroom.py report
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
import common as c

OUT = c.task_path('W14')
TYPES = ('distractor', 'missed', 'boundary', 'interior')
BAND = 16


def anatomy(pred, gt, valid, near, gt_lab, n_gt, near8):
    """I/U of the mask and of the four single-type counterfactual fixes."""
    from scipy import ndimage
    p, g = pred & valid, gt & valid
    plab, n_p = ndimage.label(p, structure=np.ones((3, 3)))
    touch = np.zeros(n_p + 1, bool)
    touch[np.unique(plab[p & g])] = True
    distractor = p & ~touch[plab]
    hit = np.zeros(n_gt + 1, bool)
    hit[np.unique(gt_lab[p & g])] = True
    missed = g & ~hit[gt_lab] & (gt_lab > 0)
    err = p ^ g
    boundary = err & near & ~distractor & ~missed
    interior = err & ~near & ~distractor & ~missed
    out = {'mask': p}
    out['distractor'] = p & ~distractor
    out['missed'] = p | missed
    out['boundary'] = np.where(boundary, g, p)
    out['interior'] = np.where(interior, g, p)
    out['boundary8'] = np.where(err & near8 & ~distractor & ~missed, g, p)
    res = {}
    for k, m in out.items():
        i = int((m & g).sum())
        res[k] = [i, int((m | g).sum())]
    res['px'] = {t: int(x.sum()) for t, x in zip(TYPES, (distractor, missed, boundary, interior))}
    res['n_pred_comp'], res['n_distractor_comp'] = int(n_p), int((~touch[1:]).sum())
    res['n_missed_inst'] = int(n_gt - hit[1:].sum())
    return res


def subset(rows, cap):
    """Seeded per-domain sample of at most `cap` episodes, fixed before any result is read (diagnostics need
    no full pass; the full S half took ~1 h per arm set and saturated the host)."""
    rng, out = np.random.default_rng(c.SEED), []
    for d in c.DOMAINS:
        L = sorted((b for b in rows if b['dataset'] == d), key=lambda b: str(b['index']))
        out += [L[i] for i in sorted(rng.choice(len(L), min(cap, len(L)), replace=False))]
    return out


def oracle(values, labels):
    order = np.argsort(-values, kind='stable')
    cum = np.cumsum(labels[order], dtype=np.int64)
    G = int(labels.sum())
    k = np.arange(1, len(values) + 1)
    iou = cum / (G + k - cum)
    j = int(iou.argmax())
    return float(iou[j]), float(values[order][j]), j + 1


def one(binding):
    import torch
    from scipy import ndimage
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    g = gt & valid
    gt_lab, n_gt = ndimage.label(g, structure=np.ones((3, 3)))
    dist_in = ndimage.distance_transform_edt(g)
    dist_out = ndimage.distance_transform_edt(~g)
    near = (np.where(g, dist_in, dist_out) <= BAND)
    near8 = (np.where(g, dist_in, dist_out) <= 8)
    r = {'index': str(binding['index']), 'n_gt_comp': int(n_gt), 'rho': float(fields['reference_rho'][0])}
    with np.load(c.prediction_path('E1', binding), allow_pickle=False) as z:
        mfu = c.unpack(z['MFU'])
    r['MFU'] = anatomy(mfu, gt, valid, near, gt_lab, n_gt, near8)
    r['FoRIS'] = anatomy(c.rc.lab.arm_cli(binding, 'B').numpy().astype(bool), gt, valid, near, gt_lab, n_gt, near8)
    if r['rho'] <= .9 and g.any():
        mm = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32)))).numpy().astype(np.float64)
        with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as z:
            m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
        field = .5 * m + .5 * mm(fields['VP_UN'])
        v, lab = field[valid], g[valid]
        nat = v > .5
        i = int((nat & lab).sum())
        best, t, k = oracle(v, lab)
        r['G'] = {'natural': 100 * i / max(int((nat | lab).sum()), 1), 'oracle': 100 * best, 't': t,
                  'natural_area': int(nat.sum()), 'oracle_area': k}
    return r


def produce(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = subset([b for b in c.rows() if c.half(b) == 'S'], 300)
    done = set()
    if (OUT / 'episodes.jsonl').exists():
        done = {json.loads(l)['index'] for l in (OUT / 'episodes.jsonl').read_text().splitlines()}
    todo = [b for b in rows if str(b['index']) not in done]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, (OUT / 'episodes.jsonl').open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 500 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    keep = {str(b['index']) for b in subset([b for b in c.rows() if c.half(b) == 'S'], 300)}
    R = [dict(E1[r['index']], **{'W': r}) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines()) if r['index'] in keep]
    print('n', len(R))

    def pooled(L, arm, k):
        for e in L:
            e['_x'] = e['W'][arm][k]
        return score.miou(L, '_x')

    names = [d[:6] for d in c.DOMAINS]
    for arm in ('MFU', 'FoRIS'):
        print(f'\n## {arm}: class mIoU of the mask and gain from fixing one error type\n{"":12s}' + ''.join(f'{d:>8s}' for d in names) + '    mean')
        base = [pooled([e for e in R if e['dataset'] == d], arm, 'mask') for d in c.DOMAINS]
        print(f'{"mask":12s}' + ''.join(f'{x:8.2f}' for x in base) + f'{np.mean(base):8.2f}')
        for t in TYPES + ('boundary8',):
            v = [pooled([e for e in R if e['dataset'] == d], arm, t) - b for d, b in zip(c.DOMAINS, base)]
            print(f'{"+fix " + t:12s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}')
        print(f'{"error px/GT":12s}' + ''.join(f'{d:>8s}' for d in names))
        for t in TYPES:
            v = [100 * sum(e['W'][arm]['px'][t] for e in R if e['dataset'] == d) / max(sum(e['gt'] for e in R if e['dataset'] == d), 1) for d in c.DOMAINS]
            print(f'{t:12s}' + ''.join(f'{x:8.1f}' for x in v))
    band = lambda e: '<2%' if e['gt'] / 2 ** 20 < .02 else ('2-20%' if e['gt'] / 2 ** 20 <= .2 else '>20%')
    print('\n## MFU: mean episode-IoU gain from fixing one error type, by GT area band and by GT component count')
    groups = [('<2%', lambda e: band(e) == '<2%'), ('2-20%', lambda e: band(e) == '2-20%'), ('>20%', lambda e: band(e) == '>20%'),
              ('1 comp', lambda e: e['W']['n_gt_comp'] == 1), ('2-3 comp', lambda e: 2 <= e['W']['n_gt_comp'] <= 3),
              ('4+ comp', lambda e: e['W']['n_gt_comp'] >= 4)]
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    print(f'{"group":10s}{"n":>6s}{"IoU":>8s}' + ''.join(f'{t:>12s}' for t in TYPES + ('boundary8',)))
    for nm, f in groups:
        L = [e for e in R if f(e) and e['gt'] > 0]
        print(f'{nm:10s}{len(L):6d}{np.mean([iou(e["W"]["MFU"]["mask"]) for e in L]):8.1f}'
              + ''.join(f'{np.mean([iou(e["W"]["MFU"][t]) - iou(e["W"]["MFU"]["mask"]) for e in L]):+12.2f}' for t in TYPES + ('boundary8',)))
    print('\n## share of episodes with GT components >= 2, and with a missed instance / distractor component (MFU)')
    for d in c.DOMAINS:
        L = [e for e in R if e['dataset'] == d and e['gt'] > 0]
        print(f'{d:15s} multi {np.mean([e["W"]["n_gt_comp"] >= 2 for e in L]):.2f}  missed-inst {np.mean([e["W"]["MFU"]["n_missed_inst"] > 0 for e in L]):.2f}'
              f'  distractor {np.mean([e["W"]["MFU"]["n_distractor_comp"] > 0 for e in L]):.2f}')
    G = [e for e in R if 'G' in e['W']]
    print(f'\n## fused field G (rho <= .9, n={len(G)}): per-episode GT-optimal level t* and headroom')
    print(f'{"domain":15s}{"natural":>9s}{"oracle":>8s}{"gap":>7s}{"t*<.4":>7s}{".4-.6":>7s}{"t*>.6":>7s}{"gap from t*<.5":>16s}')
    for d in c.DOMAINS:
        L = [e['W']['G'] for e in G if e['dataset'] == d]
        if not L:
            continue
        nat, orc, t = (np.array([x[k] for x in L]) for k in ('natural', 'oracle', 't'))
        gap = orc - nat
        print(f'{d:15s}{nat.mean():9.1f}{orc.mean():8.1f}{gap.mean():7.1f}{np.mean(t < .4):7.2f}{np.mean((t >= .4) & (t <= .6)):7.2f}'
              f'{np.mean(t > .6):7.2f}{gap[t < .5].sum() / max(gap.sum(), 1e-9):16.2f}')
    gap = np.array([e['W']['G']['oracle'] - e['W']['G']['natural'] for e in G])
    s = np.sort(gap)[::-1]
    print('\nconcentration of the G gap: top 10% / 20% / 50% of episodes hold',
          [round(float(s[:int(len(s) * q)].sum() / s.sum()), 2) for q in (.1, .2, .5)], 'of the total')


if __name__ == '__main__':
    produce(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
