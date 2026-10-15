"""W23: the FoRIS-free head as one standalone function, its accuracy, and its directly measured runtime.

Method SH (frozen DINOv3 O24 features of one reference and the query; reference mask; query RGB for the CRF):
  1  debiased unit features r, q; reference coverage y on the 64x64 grid; rho = share of FG mass off pure tokens
  2  kernel transfer c = soft FG share among exp(sim/.05)-weighted reference tokens        (point evidence)
     rho > .9 -> return minmax(up(c)) > .5                                                  (as MFU)
  3  U0 = minmax(up(solve(c_mod)))  c_mod = c x novelty^s (max-J choice), lambda-16 solve on the query kNN graph L
  4  M' = rank-guided lambda-16 solve of p on the SAME graph L (W22 'Ash'),
     p = q.mu_fg - .55 q.mu_bg_orth (one FG prototype vs hard-negative BG prototype)       (prototype evidence)
  5  official CRF( [.5 M' + .5 U0 > .5] )        (CRF640 is a speed option with equal accuracy, W21)
Everything in 1-3 is the frozen r5_candidate code (rc.independent / rc.modulate / rc.graph); unlike W19's Xp, p and
the guide use the same always-debiased features as U instead of FoRIS Part-1's conditional debias.
Readings fixed before running:
  S: SH+ vs W19 Xp+ within 0.1 in mean, no domain significant  -> faithful implementation of the selected head.
  T (one shot, same T subset as W19): report SH+ - FoRIS and SH+ - MFU per domain.
  time: SH+ measured end to end per call (in-memory O24, CPU, 2 threads, mask + RGB load, CRF) on the W19 timing
        episodes, next to a fresh FoRIS / MFU run of w19_time.py.

    python scripts/x1012/w23_simple_pipeline.py produce S|T [workers]
    python scripts/x1012/w23_simple_pipeline.py report S|T
    python scripts/x1012/w23_simple_pipeline.py time
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

OUT = {h: c.task_path('W23' + h) for h in 'ST'}


def simple_head(binding, crf=True):
    """Return (mask before CRF, final mask, rho).  Reads the reference mask, the O24 features via rc.raw, the RGB."""
    import torch
    import torch.nn.functional as F
    import produce
    from w09_integration_lambda import solve
    from w19_simple_head import mean_solve
    rc, lab = c.rc, c.rc.lab
    r, q, _, y, f = rc.independent(binding, {})
    rho = float(f['reference_rho'][0])
    if rho > .9:
        mask = (lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))) > .5).numpy()
        return mask, mask, rho
    c_mod, _ = rc.modulate(torch.from_numpy(f['c32']), torch.from_numpy(f['regions64'].astype(np.int64)),
                           f['ind0_32'], f['indn_32'], f['sim_max'], f['back_f'])
    L = rc.graph(q, f['query_neighbors20']).tocsr()
    u0 = lab.minmax(lab.up(torch.from_numpy(solve(c_mod.numpy(), f['log_e'], L, 16).reshape(64, 64).astype(np.float32)))).numpy()
    cov = y.numpy()
    qn, rn = F.normalize(q.float(), dim=1), F.normalize(r.float(), dim=1)
    fi = np.flatnonzero(cov >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov == cov.max())
    guide = (qn @ F.normalize(rn[fi].mean(0), dim=0)).numpy()
    fg = torch.from_numpy(cov >= .5 if (cov >= .5).any() else cov == cov.max())
    mu_fg = F.normalize(rn[fg].mean(0), dim=0)
    bg = rn[~fg]
    if len(bg):
        hard = bg[(bg @ mu_fg).topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0); mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg
        mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p = (qn @ mu_fg - .55 * (qn @ mu_bg)).numpy()
    m = lab.up(torch.from_numpy(mean_solve(p.reshape(64, 64), guide, L))).numpy()
    mask = .5 * m.astype(np.float64) + .5 * u0.astype(np.float64) > .5
    if not crf:
        return mask, mask, rho
    with rc.pc.inference_only(binding):
        final = produce.crf(binding, mask)
    return mask, final, rho


def one(binding):
    gt, valid = c.truth(binding)
    pre, final, rho = simple_head(binding)
    return {'index': str(binding['index']), 'SH': c.iu(pre, gt, valid), 'SH+': c.iu(final, gt, valid), 'rho': rho}


def rows_for(half):
    import w13_plugin as w13
    return w13.subset([b for b in c.rows() if c.half(b) == half], 150)


def produce_all(half, workers):
    out = OUT[half]
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_for(half)
    path = out / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for r in pool.map(one, todo, chunksize=4):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done', flush=True)


def report(half):
    import score
    rng = np.random.default_rng(c.SEED)
    if half == 'S':
        with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
            base = {r['index']: r for r in map(json.loads, f)}
        with gzip.open(ROOT / 'evidence/research/w19_simple_head_S.jsonl.gz', 'rt') as f:
            W = {r['index']: r for r in map(json.loads, f)}
    else:
        with gzip.open(ROOT / 'evidence/research/w11_t_ablation.jsonl.gz', 'rt') as f:
            base = {r['index']: r for r in map(json.loads, f)}
        with gzip.open(ROOT / 'evidence/research/w19_simple_head_T.jsonl.gz', 'rt') as f:
            W = {r['index']: r for r in map(json.loads, f)}
    R = [dict(base[r['index']], **{'Xp+': W[r['index']]['Xp+']}, **r) for r in map(json.loads, (OUT[half] / 'episodes.jsonl').read_text().splitlines())]
    print(half, 'n', len(R))
    for a, ref in (('SH+', 'Xp+'), ('SH+', 'foris'), ('SH+', 'MFU'), ('SH', 'foris')):
        v, cells, sp, sn = [], [], 0, 0
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            x = score.miou(L, a) - score.miou(L, ref); lo, hi = score.paired_ci(L, a, ref, rng)
            sp += lo > 0; sn += hi < 0; v.append(x); cells.append(f'{d[:6]} {x:+.2f}[{lo:+.1f},{hi:+.1f}]')
        print(f'{a} - {ref}: mean {np.mean(v):+.2f}  >=0 {sum(x >= 0 for x in v)}  sig+ {sp}  sig- {sn}\n   ' + '  '.join(cells))
    with gzip.open(ROOT / f'evidence/research/w23_simple_pipeline_{half}.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'SH', 'SH+', 'rho')}) + '\n')


def time_it():
    import torch
    rows = json.loads((c.task_path('W19') / 'time_bindings.json').read_text())
    c.initialize()
    torch.set_num_threads(2)
    store = {}
    original = c.rc.raw
    c.rc.raw = lambda metadata, counters: store[metadata['payload_path']]
    out = []
    for i, b in enumerate([rows[0], rows[1]] + rows):
        store.clear()
        for name in ('reference_raw', 'query_raw_identity'):
            path = b['fit'][name]['payload_path']
            with np.load(path, allow_pickle=False) as z:
                store[path] = torch.from_numpy(z['O/24'].copy())
        t = time.perf_counter(); simple_head(b, crf=True); full = time.perf_counter() - t
        t = time.perf_counter(); simple_head(b, crf=False); bare = time.perf_counter() - t
        if i >= 2:
            out.append({'index': str(b['index']), 'dataset': b['dataset'], 'SH+': full, 'SH': bare})
            print(f"{b['dataset']:15s} SH+ {full:.3f}  SH(no CRF) {bare:.3f}", flush=True)
    c.rc.raw = original
    (c.task_path('W23S') / 'timing.json').write_text(json.dumps(out, indent=1))
    a = np.array([o['SH+'] for o in out]); b_ = np.array([o['SH'] for o in out])
    print(f'mean SH+ {a.mean():.3f} (median {np.median(a):.3f})   SH without CRF {b_.mean():.3f} (median {np.median(b_):.3f})')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'produce':
        produce_all(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == 'report':
        report(sys.argv[2])
    else:
        time_it()
