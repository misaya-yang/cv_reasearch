"""W43: are SH's entity-level errors whole wrong/missed components (selection) or errors inside mostly-correct components
(splitting)?  S half only.  Plan and readings: docs/research/x1012_tasks/W43_entity_units_PLAN.md.

    nice -n 19 python scripts/x1012/w43_entity_units.py produce [workers]
    python scripts/x1012/w43_entity_units.py report
"""
import gzip, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W43')
ORA = ('O_del', 'O_add', 'O_sel')
EDGES = (0, .1, .3, .5, .9, 1.0001)
EIGHT = np.ones((3, 3), bool)


def one(b):
    import w31_demonstration as w31
    import w37_coverage as w37
    import w42_ordered_cut as w42
    began = time.monotonic()
    F, N, U, M, rho = w42.capture(b)
    fb = U is None
    Bpix = np.kron(w37.band(w37.pool16(N.astype(np.float64))), np.ones((16, 16), bool))
    lab_n, k_n = ndimage.label(N, EIGHT)
    r = dict(w31.meta(b), rho=rho, fallback=fb, n_comp=int(k_n))
    gt, valid = c.truth(b)                      # read only after the label-free mask and units are fixed
    lab_g, k_g = ndimage.label(gt, EIGHT)
    r['g_comp'] = int(k_g); r['gt'] = int(gt.sum())
    r['N'] = c.iu(N, gt, valid)
    J = r['N'][0] / max(r['N'][1], 1)
    area = np.bincount(lab_n.ravel(), minlength=k_n + 1)
    tp = np.bincount(lab_n.ravel(), weights=gt.ravel(), minlength=k_n + 1)
    pur = tp / np.maximum(area, 1)
    drop = np.zeros(k_n + 1, bool); drop[1:] = pur[1:] < J / (1 + J)
    hit = np.bincount(lab_g.ravel(), weights=N.ravel(), minlength=k_g + 1) > 0
    missed = np.zeros(k_g + 1, bool); missed[1:] = ~hit[1:]
    dele, add = drop[lab_n], missed[lab_g]
    r['O_del'] = c.iu(N & ~dele, gt, valid)
    r['O_add'] = c.iu(N | add, gt, valid)
    r['O_sel'] = c.iu((N & ~dele) | add, gt, valid)
    r['n_drop'] = int(drop.sum()); r['n_missed'] = int(missed[1:].sum())
    if fb:
        C0 = C1 = np.zeros_like(N)
    else:
        C0, C1 = (U <= .5) & (M <= .5), (U > .5) & (M > .5)
    off = ~Bpix
    pbin = np.digitize(pur, EDGES) - 1                   # component purity bin 0..4
    fp, fn = N & ~gt & valid, gt & ~N
    for tag, R_ in (('cfp_off', C1 & off), ('fp_all', np.ones_like(N))):
        sel = fp & R_
        r[tag] = np.bincount(pbin[lab_n[sel]], minlength=5).astype(int).tolist()
    for tag, R_ in (('cfn_off', C0 & off), ('fn_all', np.ones_like(N))):
        sel = fn & R_
        r[tag] = [int((sel & add).sum()), int((sel & ~add).sum())]   # in missed GT components, in hit components
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


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    same = sum(r['N'][:2] == w23[r['index']]['SH'][:2] for r in R)
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    L = [f'n {len(R)}  check: N == W23 SH {same}/{len(R)} (need >= 1280)',
         f'N components per episode: median {np.median([r["n_comp"] for r in R]):.0f}, mean {np.mean([r["n_comp"] for r in R]):.1f}; '
         f'GT components median {np.median([r["g_comp"] for r in R]):.0f}; dropped by O_del mean {np.mean([r["n_drop"] for r in R]):.2f}; '
         f'missed GT components mean {np.mean([r["n_missed"] for r in R]):.2f}', '']
    names = ['[0,.1)', '[.1,.3)', '[.3,.5)', '[.5,.9)', '[.9,1]']
    for tag, title in (('cfp_off', 'off-band common FP'), ('fp_all', 'all FP')):
        v = np.sum([r[tag] for r in R], axis=0); t = max(v.sum(), 1)
        L.append(f'{title:20s} by component purity: ' + '  '.join(f'{n} {x / t:5.1%}' for n, x in zip(names, v)) +
                 f'   (< .5: {v[:3].sum() / t:.1%}; total {t / 1e6:.2f}M)')
    for tag, title in (('cfn_off', 'off-band common FN'), ('fn_all', 'all FN')):
        v = np.sum([r[tag] for r in R], axis=0); t = max(v.sum(), 1)
        L.append(f'{title:20s} in missed GT components {v[0] / t:5.1%}, attached to detected components {v[1] / t:5.1%} (total {t / 1e6:.2f}M)')
    L += ['', '## oracle capacities, class mIoU minus N (pre-CRF)', f'{"":12s}' + ''.join(f'{a:>8s}' for a in ORA) + '   cfp_off<.5  cfn_off missed']
    mean = {a: [] for a in ORA}
    for d in doms:
        cells = []
        for a in ORA:
            v = sc.miou(by[d], a) - sc.miou(by[d], 'N'); mean[a].append(v); cells.append(f'{v:+8.2f}')
        fpv = np.sum([r['cfp_off'] for r in by[d]], axis=0); fnv = np.sum([r['cfn_off'] for r in by[d]], axis=0)
        L.append(f'{d[:11]:12s}' + ''.join(cells) + f'   {fpv[:3].sum() / max(fpv.sum(), 1):9.1%}  {fnv[0] / max(fnv.sum(), 1):9.1%}')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):+8.2f}' for a in ORA))
    lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], 'O_sel', 'N') for d in doms], axis=0), [2.5, 97.5])
    g = np.mean(mean['O_sel'])
    L += ['', f'reading 2: O_sel {g:+.2f} [{lo:+.2f},{hi:+.2f}] vs 0.5 * O_C (W41 +16.48) = +8.24 -> '
              + ('selection of whole components carries most entity-level capacity' if g >= 8.24 else
                 'errors mostly inside mostly-correct components: need splitting / growing'),
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback', 'n_comp', 'g_comp', 'n_drop', 'n_missed',
            'N') + ORA + ('cfp_off', 'fp_all', 'cfn_off', 'fn_all')
    with gzip.open(ROOT / 'evidence/research/w43_entity_units_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
