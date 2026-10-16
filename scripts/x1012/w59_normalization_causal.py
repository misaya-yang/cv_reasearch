"""W59 normalization intervention. Produce all predictions before scoring.

    python -B scripts/x1012/w59_normalization_causal.py produce 2
    python -B scripts/x1012/w59_normalization_causal.py score
"""
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'scripts/x1013')]
import common as c

OUT = c.task_path('W59')
BASE_ARMS = ('A0', 'Nout', 'AbsFix', 'AbsWeight')
ARMS = tuple(a + s for a in BASE_ARMS for s in ('', '+'))
DEADLINE = datetime.fromisoformat('2026-10-16T05:46:30+00:00').timestamp()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def init():
    global THREAD_LIMIT
    c.initialize()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    THREAD_LIMIT = threadpool_limits(limits=1)


def solve_raw(primary, log_e, lap, weights_primary):
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from scipy.special import expit
    x = np.asarray(primary, np.float64).ravel()
    w = np.asarray(weights_primary, np.float64).ravel()
    a = (.1 + np.abs(2 * w - 1)) * expit(np.asarray(log_e, np.float64) - np.median(log_e))
    a /= a.mean()
    value, status = cg(sparse.diags(a) + 16 * lap, a * x, x0=x, rtol=1e-7, atol=1e-9, maxiter=2000)
    assert status == 0 and np.isfinite(value).all()
    return value


def save_episode(out, b, masks, extras, seconds, fields=None):
    import w31_demonstration as w31
    p = out / 'predictions' / (c.key(b) + '.npz')
    arrays = {a: c.pack(m) for a, m in masks.items()}
    arrays.update(fields or {})
    c.save_npz(p, **arrays)
    return dict(w31.meta(b), **extras, prediction=str(p), prediction_sha256=sha(p),
                seconds=round(seconds, 4), query_GT_reads=0)


def one(b):
    if time.time() >= DEADLINE:
        raise TimeoutError('Authorized window ended; no further inference')
    import torch
    import w09_integration_lambda as w09
    import w19_simple_head as w19
    import w23_simple_pipeline as sh
    import produce
    start = time.monotonic()
    cap = {}
    original_u, original_m = w09.solve, w19.mean_solve
    def point(*args, **kwargs):
        value = original_u(*args, **kwargs)
        cap['args'], cap['v'] = args, value
        return value
    def proto(*args, **kwargs):
        value = original_m(*args, **kwargs)
        cap['m'] = value
        return value
    w09.solve, w19.mean_solve = point, proto
    try:
        with c.rc.pc.inference_only(b):
            pre, final, rho = sh.simple_head(b, crf=True)
    finally:
        w09.solve, w19.mean_solve = original_u, original_m
    masks = {'A0': pre, 'A0+': final}
    extra = dict(rho=rho, fallback=rho > .9)
    if rho > .9:
        for a in BASE_ARMS[1:]:
            masks[a], masks[a + '+'] = pre.copy(), final.copy()
    else:
        lab = c.rc.lab
        up = lambda x: lab.up(torch.from_numpy(np.asarray(x, np.float32).reshape(64, 64))).numpy().astype(np.float64)
        primary, log_e, lap, lam = cap['args']
        assert lam == 16
        x = np.asarray(primary, np.float64).ravel()
        lo, span = float(x.min()), float(x.max() - x.min())
        v, m = up(cap['v']), up(cap['m'])
        u = lab.minmax(torch.from_numpy(v.astype(np.float32))).numpy().astype(np.float64)
        assert np.array_equal(.5 * m + .5 * u > .5, pre)
        fixed = up(solve_raw(x, log_e, lap, (x - lo) / max(span, 1e-12)))
        affine = lo + span * v
        extra['affine_error'] = float(np.max(np.abs(fixed - affine)))
        assert extra['affine_error'] < 2e-5
        # The exact fixed-A counterfactual follows from G_A 1 = 1.
        raw_weight = up(solve_raw(x, log_e, lap, x))
        extra.update(cmod_min=lo, cmod_range=span,
                     normalized_fixed_error=float(np.max(np.abs(lab.minmax(torch.from_numpy(affine.astype(np.float32))).numpy() - u))))
        for a, field in (('Nout', v), ('AbsFix', affine), ('AbsWeight', raw_weight)):
            mk = .5 * m + .5 * field > .5
            with c.rc.pc.inference_only(b):
                crf = produce.crf(b, mk)
            masks[a], masks[a + '+'] = mk, crf
    return save_episode(OUT, b, masks, extra, time.monotonic() - start)


def run_produce(out, arms, fn, workers):
    import w31_demonstration as w31
    assert 1 <= workers <= 2
    out.mkdir(exist_ok=True, parents=True)
    lock = (out / 'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Already running; no duplicate launched', flush=True)
        return
    rows = w31.rows()
    ids = [str(b['index']) for b in rows]
    assert len(rows) == 1285 and len(set(ids)) == 1285 and all(c.half(b) == 'S' for b in rows)
    source = Path(sys.argv[0]).resolve()
    config = dict(arms=list(arms), n=len(rows), indices=ids,
                  bindings_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  sources={str(p.relative_to(ROOT)): sha(p) for p in {source, Path(__file__), ROOT/'scripts/x1012/r5_candidate.py', ROOT/'scripts/x1012/w23_simple_pipeline.py'}},
                  workers=workers, threads_per_worker=1, mps=False, deadline_utc='2026-10-16T05:46:30Z')
    cp = out / 'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Frozen run configuration changed'
    else:
        write(cp, config)
    import psutil
    write(out / 'execution.json', dict(pid=os.getpid(), create_time=psutil.Process().create_time(), script=str(source),
                                     started_utc=datetime.now(timezone.utc).isoformat(), workers=workers, status='running'))
    path = out / 'episodes.jsonl'
    records = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    done = {r['index'] for r in records}
    assert len(done) == len(records) and done <= set(ids)
    todo = [b for b in rows if str(b['index']) not in done]
    start = time.monotonic()
    print('todo', len(todo), 'workers', workers, 'one thread each; MPS disabled', flush=True)
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, path.open('a') as f:
            for n, r in enumerate(pool.map(fn, todo, chunksize=1), 1):
                f.write(json.dumps(r) + '\n'); f.flush(); records.append(r)
                if n % 50 == 0:
                    print(len(records), len(rows), 'episode_s', r['seconds'], 'elapsed_s', round(time.monotonic()-start, 1), flush=True)
        assert {r['index'] for r in records} == set(ids)
        assert all(sha(r['prediction']) == r['prediction_sha256'] for r in records)
        write(out / 'complete.json', dict(n=len(records), config_sha256=sha(cp), episodes_sha256=sha(path),
                                         completed_utc=datetime.now(timezone.utc).isoformat(), query_GT_reads=0))
        print('complete; all predictions sealed before scoring', flush=True)
    except BaseException as e:
        write(out / 'failure.json', dict(type=type(e).__name__, message=str(e), records=len(records),
                                        utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        state = json.loads((out / 'execution.json').read_text())
        state.update(status='complete' if (out/'complete.json').exists() else 'partial_or_failed', ended_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'execution.json', state)
        lock.close()


def summarize(out, arms, records):
    import score
    rng = np.random.default_rng(2026101605)
    by_domain = {d: [r for r in records if r['dataset'] == d] for d in c.DOMAINS}
    summary = {'n': len(records), 'arms': list(arms), 'comparisons': {}, 'by_w58': {}}
    lines = [f'n {len(records)}; paired original S; predictions sealed before GT; CPU only']
    for a in arms:
        if a in ('A0', 'A0+'):
            continue
        base = 'A0+' if a.endswith('+') else 'A0'
        cells, ds = [], {}
        for d, sel in by_domain.items():
            delta = score.miou(sel, a) - score.miou(sel, base)
            lo, hi = score.paired_ci(sel, a, base, rng)
            ds[d] = dict(delta=delta, ci=[lo, hi], value=score.miou(sel, a))
            cells.append(f'{d[:6]} {delta:+.2f}[{lo:+.2f},{hi:+.2f}]')
        mean = float(np.mean([v['delta'] for v in ds.values()]))
        edits = {k: sum(r['edit_'+a][k] for r in records) for k in ('add_tp', 'add_fp', 'del_tp', 'del_fp')}
        summary['comparisons'][a] = dict(reference=base, mean_delta=mean, domains=ds, pooled_edits=edits)
        lines += [f'{a} - {base}: ten-domain mean {mean:+.3f}', '  ' + '  '.join(cells), '  pooled edits '+json.dumps(edits)]
        groups = {}
        for group in sorted({r['w58'] for r in records}):
            sel = [r for r in records if r['w58'] == group and r['catastrophic']]
            if sel:
                groups[group] = dict(n=len(sel), mean_episode_delta=float(100*np.mean([r[a][0]/max(r[a][1],1)-r[base][0]/max(r[base][1],1) for r in sel])))
        summary['by_w58'][a] = groups
    summary['identity_A0'] = sum(r['identity_A0'] for r in records)
    summary['identity_A0_plus'] = sum(r['identity_A0_plus'] for r in records)
    assert summary['identity_A0'] == summary['identity_A0_plus'] == 1285
    summary['median_produce_seconds'] = float(np.median([r['seconds'] for r in records]))
    lines += [f'A0 / A0+ identity {summary["identity_A0"]}/{summary["identity_A0_plus"]} of 1285',
              f'produce median {summary["median_produce_seconds"]:.3f} seconds/episode']
    write(out / 'summary.json', summary)
    (out / 'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return summary


def score_run(out, arms, evidence, extra_scorer=None):
    import w31_demonstration as w31
    import plan_common as pc
    complete = json.loads((out/'complete.json').read_text())
    assert complete['n'] == 1285 and complete['episodes_sha256'] == sha(out/'episodes.jsonl')
    assert complete['config_sha256'] == sha(out/'config.json')
    meta = {r['index']: r for r in map(json.loads, (out/'episodes.jsonl').read_text().splitlines())}
    with gzip.open(ROOT/'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt') as f:
        original = {r['index']: r for r in map(json.loads, f)}
    with gzip.open(ROOT/'evidence/research/w58_score_repair_cases.jsonl.gz', 'rt') as f:
        groups = {r['index']: r for r in map(json.loads, f)}
    records = []
    for b in w31.rows():
        key = str(b['index']); r = dict(meta[key])
        assert sha(r['prediction']) == r['prediction_sha256']
        gt, valid = pc.truth(b)
        with np.load(r['prediction'], allow_pickle=False) as z:
            masks = {a: c.unpack(z[a]) & valid for a in arms}
            for a, mk in masks.items():
                r[a] = c.iu(mk, gt, valid)
                base = masks['A0+' if a.endswith('+') else 'A0']
                add, delete = mk & ~base, base & ~mk
                r['edit_'+a] = dict(add_tp=int((add & gt).sum()), add_fp=int((add & ~gt).sum()),
                                   del_tp=int((delete & gt).sum()), del_fp=int((delete & ~gt).sum()))
            if extra_scorer:
                r.update(extra_scorer(z, gt, valid))
        r.update(gt=int(gt.sum()), identity_A0=r['A0'] == original[key]['SH'],
                 identity_A0_plus=r['A0+'] == original[key]['SH+'], w58=groups[key]['diagnostic'], catastrophic=groups[key]['catastrophic'])
        records.append(r)
    assert len(records) == 1285
    with gzip.open(ROOT/'evidence/research'/evidence, 'wt') as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    summary = summarize(out, arms, records)
    write(out/'scored.json', dict(n=len(records), evidence=evidence, sha256=sha(ROOT/'evidence/research'/evidence),
                                  scored_utc=datetime.now(timezone.utc).isoformat()))
    return records, summary


if __name__ == '__main__':
    if sys.argv[1] == 'produce':
        run_produce(OUT, ARMS, one, int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1] == 'score':
        score_run(OUT, ARMS, 'w59_normalization_causal_S.jsonl.gz')
