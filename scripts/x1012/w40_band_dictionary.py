"""W40: transfer band coverage from the reference's own boundary band (population-matched kernel transfer).  S half only.
Plan and readings: docs/research/x1012_tasks/W40_band_dictionary_PLAN.md.  Same episodes, field, natural mask and query
band as W37/W38.  Amount per query band token: nat, C (c32, all reference tokens), Bd (same kernel, dictionary restricted
to the reference band), true (GT).  Reconstruction = W37 R2 (k highest-F pixels per band block).

    nice -n 19 python scripts/x1012/w40_band_dictionary.py produce [workers]
    python scripts/x1012/w40_band_dictionary.py report
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

OUT = c.task_path('W40')
ARMS = ('nat', 'C2', 'Bd2', 'true')
EST = ('C', 'Bd')


def one(b):
    import torch
    import w31_demonstration as w31
    import w37_coverage as w37
    import w38_band_amount as w38
    began = time.monotonic()
    got, oi = [], c.rc.independent

    def keep(*a, **k):
        v = oi(*a, **k); got.append(v); return v
    c.rc.independent = keep
    try:
        F, N, rho, c32, _ = w37.capture(b)
    finally:
        c.rc.independent = oi
    r, q, _, y, _ = got[0]
    lab = c.rc.lab
    mu = .5 * (r.mean(0) + q.mean(0))
    sim = (lab.unit(q - mu) @ lab.unit(r - mu).T).double()
    y = y.double()
    w = torch.exp((sim - sim.max()) / .05)
    cc = (w @ y) / (w.sum(1)).clamp_min(1e-30)
    gap = float(np.abs(cc.numpy() - c32.ravel()).max())
    BR = w37.band(y.reshape(64, 64).numpy()).ravel()
    if BR.sum() >= 4:
        s = sim[:, torch.from_numpy(BR)]
        wb = torch.exp((s - s.max(1, keepdim=True).values) / .05)
        bd = ((wb @ y[torch.from_numpy(BR)]) / wb.sum(1)).numpy().reshape(64, 64)
    else:
        bd = c32
    nT = w37.pool16(N.astype(np.float64))
    B = w37.band(nT)
    kn = w38.blocks(N[..., None], B)[0][..., 0].sum(1)
    kq = lambda pi: np.rint(256 * np.clip(pi[B], 0, 1)).astype(np.int64)   # row-major, same order as blocks()
    amounts = {'nat': kn, 'C2': kq(c32), 'Bd2': kq(bd)}
    masks = {a: w38.place(N, F, B, amounts[a]) for a in amounts}
    gt, valid = c.truth(b)                      # read only after every label-free amount is fixed
    amounts['true'] = w38.blocks(gt[..., None], B)[0][..., 0].sum(1)
    masks['true'] = w38.place(N, F, B, amounts['true'])
    kt = amounts['true']
    y_ = (kt - kn).astype(np.float64)
    r_ = dict(w31.meta(b), gt=int(gt.sum()), rho=rho, fallback=rho > .9, band_tokens=int(B.sum()),
              ref_band_tokens=int(BR.sum()), c_gap=gap, l1_nat=int(np.abs(kn - kt).sum()))
    for a in ARMS:
        r_[a] = c.iu(masks[a], gt, valid)
    for e, a in zip(EST, ('C2', 'Bd2')):
        x = (amounts[a] - kn).astype(np.float64)
        r_['stat_' + e] = [len(x), x.sum(), y_.sum(), (x * y_).sum(), (x * x).sum(), (y_ * y_).sum()]
        r_['l1_' + e] = int(np.abs(amounts[a] - kt).sum())
    r_['seconds'] = round(time.monotonic() - began, 2)
    return r_


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
    import w38_band_amount as w38
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w37r = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w37_coverage_S.jsonl.gz', 'rt'))}
    same = sum(r['nat'][:2] == w23[r['index']]['SH'][:2] for r in R)
    sameC = sum(r['C2'][:2] == w37r[r['index']]['C2'][:2] for r in R) if 'C2' in next(iter(w37r.values())) else -1
    doms = list(c.DOMAINS)
    rng = np.random.default_rng(c.SEED)
    idx = [rng.integers(0, len(R), len(R)) for _ in range(500)]
    L = [f'n {len(R)}  implementation check: nat == W23 SH (pre-CRF) {same}/{len(R)} (need >= 1280); '
         f'C2 == W37 C2 {sameC}/{len(R)}; max |c recomputed - c32| {max(r["c_gap"] for r in R):.2e}',
         f'reference band tokens median {np.median([r["ref_band_tokens"] for r in R]):.0f}; '
         f'Bd fell back to C in {sum(r["ref_band_tokens"] < 4 for r in R)} episodes', '',
         '## pooled absolute count error ratio L1(est)/L1(nat) over band tokens; local rho; episode-cluster bootstrap 95%']
    l1n = np.array([r['l1_nat'] for r in R], float)
    for e in EST:
        le = np.array([r['l1_' + e] for r in R], float)
        st = np.array([r['stat_' + e] for r in R])
        bl = [le[i].sum() / l1n[i].sum() for i in idx]
        keep = st[:, 0] > 0
        br = [w38.pearson(st[i][keep[i]]) for i in idx]
        L.append(f'{e:4s} L1 ratio {le.sum() / l1n.sum():.3f} [{np.percentile(bl, 2.5):.3f},{np.percentile(bl, 97.5):.3f}]'
                 f'   rho {w38.pearson(st[keep]):+.3f} [{np.percentile(br, 2.5):+.3f},{np.percentile(br, 97.5):+.3f}]')
    L += ['', '## class mIoU minus nat (pre-CRF)', f'{"":12s}' + ''.join(f'{a:>9s}' for a in ARMS[1:] + ('Bd2-C2',))]
    mean = {a: [] for a in ARMS[1:] + ('Bd2-C2',)}
    for d in doms:
        D = [r for r in R if r['dataset'] == d]
        cells = []
        for a in ARMS[1:]:
            v = sc.miou(D, a) - sc.miou(D, 'nat'); mean[a].append(v); cells.append(f'{v:+9.2f}')
        v = sc.miou(D, 'Bd2') - sc.miou(D, 'C2'); mean['Bd2-C2'].append(v); cells.append(f'{v:+9.2f}')
        lo, hi = sc.paired_ci(D, 'Bd2', 'nat', np.random.default_rng(c.SEED))
        L.append(f'{d[:11]:12s}' + ''.join(cells) + f'   Bd2 CI [{lo:+.2f},{hi:+.2f}]')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean(mean[a]):+9.2f}' for a in mean))
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], 'Bd2', 'nat') for d in doms], axis=0), [2.5, 97.5])
    g = np.mean(mean['Bd2']); pos = sum(x >= 0 for x in mean['Bd2']); dc = np.mean(mean['Bd2-C2'])
    ok = g >= 1.0 and lo > 0 and pos >= 7
    lc = sum(r['l1_C'] for r in R); lb = sum(r['l1_Bd'] for r in R)
    I = {a: sum(r[a][0] for r in R) for a in ARMS}; U = {a: sum(r[a][1] for r in R) for a in ARMS}
    L += [''] + [f'{a:5s} pooled I {I[a] / 1e6:7.2f}M  U {U[a] / 1e6:7.2f}M  errors {(U[a] - I[a]) / 1e6:7.2f}M' for a in ARMS]
    L += ['', f'reading: Bd2 - nat {g:+.2f} CI [{lo:+.2f},{hi:+.2f}] domains >= 0 {pos}/10 -> pass: {bool(ok)}',
          f'cause split (if not pass): Bd2 - C2 {dc:+.2f} (>= +1.0 ?); count L1 Bd {lb} vs C {lc} (report only)',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w40_band_dictionary_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback',
                                                  'band_tokens', 'ref_band_tokens', 'c_gap', 'l1_nat', 'l1_C', 'l1_Bd')
                                + ARMS + tuple('stat_' + e for e in EST)}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
