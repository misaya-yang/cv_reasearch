"""W11: descriptive ablation of the frozen E1 arms on the x1013 report half T.

Motivation: E2 froze MFU from S and scored it once on T (ten domains >= complete FoRIS).
The paper needs the same held-out half for the single-line controls -- does the candidate
alone still miss large targets and MEAN alone miss fine structure on T, and is the fusion
the only arm positive in every size band?  Selection is already frozen: whatever this shows,
the main method stays MFU (plan 20261013C, H2).  Reads sealed E1/T1/T0 masks only; no encoder.

    python scripts/x1012/w11_t_ablation.py produce [workers]
    python scripts/x1012/w11_t_ablation.py report
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
import common as c
import score

OUT = c.task_path('W11')
E1_ARMS = ('C1', 'C2', 'UFraw', 'UFraw+', 'Ugm+', 'UF+', 'M+', 'MFU')
ROWS = ('candidate', 'C1', 'T', 'M', 'M+', 'UFraw', 'UFraw+', 'UF+', 'Ugm+', 'MFU')


def one(binding):
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index'])}
    with np.load(c.prediction_path('E1', binding), allow_pickle=False) as z:
        for a in E1_ARMS:
            r[a] = c.iu(c.unpack(z[a]), gt, valid)
    with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
        r['M'] = c.iu(c.unpack(z['mean']), gt, valid)
        r['T'] = c.iu(c.unpack(z['T']), gt, valid)
    with np.load(c.prediction_path('T0', binding), allow_pickle=False) as z:
        r['candidate'] = c.iu(c.unpack(z['candidate']), gt, valid)
    return r


def e2_records():
    with gzip.open(c.task_path('E2') / 'iu.jsonl.gz', 'rt') as f:
        return {r['index']: r for r in map(json.loads, f)}


def produce(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    E2 = e2_records()
    todo = [b for b in c.rows() if str(b['index']) in E2]
    assert len(todo) == len(E2) == 6861
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, (OUT / 'episodes.jsonl').open('w') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=8), 1):
            fh.write(json.dumps(r) + '\n')
            if n % 500 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    E2 = e2_records()
    R = [dict(E2[r['index']], **{k: v for k, v in r.items() if k not in ('MFU', 'UFraw')}, _check=r)
         for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    bad = sum(r['_check'][a][:2] != r[a][:2] for r in R for a in ('MFU', 'UFraw'))
    print(f'n {len(R)} (T only: {all(r["split"] == "T" for r in R)}); E1 MFU/UFraw vs E2 records mismatches: {bad}')
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]

    def table(title, ref, rows, ci=True):
        print(f'\n## {title}\n{"arm":10s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0  sig+  sig-')
        for a in rows:
            v, cells, sp, sn = [], [], 0, 0
            for d in c.DOMAINS:
                L = [r for r in R if r['dataset'] == d]
                x = score.miou(L, a) - score.miou(L, ref)
                lo, hi = score.paired_ci(L, a, ref, rng) if ci else (np.nan, np.nan)
                sp += lo > 0; sn += hi < 0; v.append(x)
                cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]' if ci else f'{x:+17.2f}')
            print(f'{a:10s}' + ''.join(f'{s:>17s}' for s in cells)
                  + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}{sp:6d}{sn:6d}')

    print('\n## absolute class mIoU\n' + f'{"arm":10s}' + ''.join(f'{d:>8s}' for d in names) + '    mean')
    for a in ('foris', 'foris_noCRF') + ROWS:
        v = [score.miou([r for r in R if r['dataset'] == d], a) for d in c.DOMAINS]
        print(f'{a:10s}' + ''.join(f'{x:8.2f}' for x in v) + f'{np.mean(v):8.2f}')
    table('vs complete FoRIS (500 photo-bootstrap, 95% CI)', 'foris', ROWS)
    table('MFU vs its single lines (contribution of the other line)', 'C1', ('MFU',))
    table('', 'M+', ('MFU',))
    table('UFraw vs FoRIS without CRF (equal area, neither has CRF)', 'foris_noCRF', ('UFraw',))
    print(f'\n## mean episode-IoU difference vs complete FoRIS by GT area\n{"arm":10s}{"<2%":>8s}{"2-20%":>8s}{">20%":>8s}')
    bands = {b: [r for r in R if score.area_band(r) == b] for b in ('<2%', '2-20%', '>20%')}
    print(f'{"n":10s}' + ''.join(f'{len(L):8d}' for L in bands.values()))
    for a in ROWS:
        print(f'{a:10s}' + ''.join(f'{np.mean([score.episode_iou(r, a) - score.episode_iou(r, "foris") for r in L]):+8.2f}'
                                   for L in bands.values()))
    with gzip.open(ROOT / 'evidence/research/w11_t_ablation.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'foris', 'foris_noCRF') + ROWS}) + '\n')


if __name__ == '__main__':
    produce(int(sys.argv[2]) if len(sys.argv) > 2 else 4) if sys.argv[1] == 'produce' else report()
