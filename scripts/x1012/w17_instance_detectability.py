"""W17: are missed instances and distractor components separable in the fused field?  (S half, GT diagnostic)

Motivation: W15/W16 closed the shared level (extent) headroom.  W14 leaves whole missed instances (iSAID +12.3,
COCO/LVIS/PACO/SUIM +2-5) and whole distractor components (COCO/LVIS ~+4) as the other error mass.  The global
readout (minmax, one level) can drop a weaker instance under the strongest one's half-max.
Per GT component (8-conn) of each episode in the W15 sample: is it hit by G > .5, its area, and the peak of G in it
relative to the background distribution (non-GT valid pixels) and to the episode's strongest true component.
Per distractor component of G > .5: its peak relative to the same references.
Reading fixed before running:
  >= 50 % of missed components (by GT pixels) peak above the background 99th percentile -> a local-peak readout
     (each candidate gets its own extent) is worth a rule test; otherwise missed instances are at background
     level and the axis is closed for our fields.
  distractor peaks vs hit-component peaks: if distractors peak as high as true components (median ratio >= .9),
     whole-component removal needs evidence other than the field.
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

OUT = c.task_path('W17')


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
    bg = f[valid & ~g]
    q = np.quantile(bg, [.95, .99, .999]) if bg.size else np.ones(3)
    glab, n = ndimage.label(g, np.ones((3, 3)))
    comps = []
    for k in range(1, n + 1):
        cm = glab == k
        comps.append({'px': int(cm.sum()), 'hit': bool((cm & nat).any()), 'peak': float(f[cm].max())})
    plab, npred = ndimage.label(nat, np.ones((3, 3)))
    touch = set(np.unique(plab[nat & g]).tolist())
    dist = [{'px': int((plab == k).sum()), 'peak': float(f[plab == k].max())} for k in range(1, npred + 1) if k not in touch]
    return {'index': str(binding['index']), 'bg_q': q.tolist(), 'comps': comps, 'distractors': dist}


def main():
    import w15_query_extent as w15
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'episodes.jsonl').exists():
        R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    else:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
            R = list(pool.map(one, w15.sample(), chunksize=4))
        (OUT / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in R))
    import gzip
    E1 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt'))}
    print(f'{"domain":15s}{"missed comps":>13s}{"missed GT px%":>14s}{">bg p95":>9s}{">p99":>7s}{">p99.9":>8s}{">0.5":>6s}'
          f'{"peak/top":>10s}{"distr":>7s}{"distr peak/top":>16s}{"distr>top*.9":>14s}')
    for d in [*c.DOMAINS, 'all']:
        L = [r for r in R if d == 'all' or E1[r['index']]['dataset'] == d]
        if not L:
            continue
        miss, w, rel, dpk = [], [], [], []
        gt_px = sum(cc['px'] for r in L for cc in r['comps'])
        for r in L:
            hits = [cc['peak'] for cc in r['comps'] if cc['hit']]
            top = max(hits) if hits else max([cc['peak'] for cc in r['comps']] or [1])
            for cc in r['comps']:
                if not cc['hit']:
                    miss.append([cc['peak'] > x for x in r['bg_q']] + [cc['peak'] > .5]); w.append(cc['px']); rel.append(cc['peak'] / max(top, 1e-9))
            dpk += [x['peak'] / max(top, 1e-9) for x in r['distractors']]
        miss, w = np.array(miss, float).reshape(-1, 4), np.array(w, float)
        share = (w[:, None] * miss).sum(0) / max(w.sum(), 1) if len(w) else np.zeros(4)
        print(f'{d:15s}{len(w):13d}{100 * w.sum() / max(gt_px, 1):14.1f}' + ''.join(f'{x:{k}.2f}' for x, k in zip(share, (9, 7, 8, 6)))
              + f'{(np.median(rel) if rel else np.nan):10.2f}{len(dpk):7d}{(np.median(dpk) if dpk else np.nan):16.2f}'
              f'{(np.mean(np.array(dpk) >= .9) if dpk else np.nan):14.2f}')
    print('shares are by GT pixels of missed components; peak/top = missed-component peak over the strongest hit component')


if __name__ == '__main__':
    main()
