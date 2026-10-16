"""One preregistered W67 trial. Metadata preparation and GT-free production only.

The scorer is separate. No model fitting, new encoding, layer/resolution search,
or modification of frozen SH primitives. See W67_correspondence_validation_PLAN.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import fcntl
import hashlib
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import common as c
import w67_correspondence_algebra as algebra

OUT = c.task_path('W67_validation')
DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim', 'isic')
ARMS = ('SH', 'Repair', 'Null')
BETA, RADIUS, NULL_SEED = 4., 4, 2026101607
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: datetime.now(timezone.utc).isoformat()
write = c.write


def read_records(path, key='index'):
    return {r[key]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def photo_order(seed, dataset, photo):
    return hashlib.sha256(f'{seed}/{dataset}/{photo}'.encode()).hexdigest()


def clean_binding(b):
    return {k: b[k] for k in ('index', 'dataset', 'episode_id', 'row', 'fit', '_split') if k in b}


def prepare():
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/'manifest.json'
    if dest.exists():
        print('Retain manifest', sha(dest)); return
    ref_path, switch_path = c.task_path('W61'), c.task_path('W63')
    refs = json.loads((ref_path/'manifest.json').read_text())['episodes']
    switches = json.loads((switch_path/'manifest.json').read_text())['episodes']
    excluded = {c.photo(b) for b in json.loads((c.task_path('W64_refgraph')/'manifest.json').read_text())['episodes']}
    old = {name: read_records(c.task_path(name)/'episodes.jsonl', 'pair_id' if name == 'W63' else 'index')
           for name in ('W60', 'W61', 'W63')}
    jobs = []

    def append(panel, unit, member, b, old_name, old_key, before, after, half=None):
        record = old[old_name][str(old_key)]
        assert c.half(b) == 'S'
        jobs.append(dict(job_id=f'{panel}_{unit}_{member}', panel=panel, unit=unit,
                         member=member, half=half, binding=clean_binding(b),
                         index=str(b['index']), dataset=b['dataset'], photo=c.photo(b),
                         fold=b['row']['fold'], cls=b['row']['global_class_id'],
                         baseline=dict(path=record['prediction'], sha256=record['prediction_sha256'],
                                       before=before, after=after)))

    for d in DOMAINS:
        selected = sorted([x for x in refs if x['dataset'] == d],
                          key=lambda x: photo_order(2026101618, d, x['photo']))[:10]
        assert len(selected) == 10 and len({x['photo'] for x in selected}) == 10
        for x in selected:
            excluded.add(x['photo'])
            for j, b in enumerate(x['bindings']):
                append('reference', f'{d}_{x["index"]}', f'R{j}', b, 'W61', x['index'], f'R{j}_SH', f'R{j}_SH+')
    for x in switches:
        excluded.add(x['photo'])
        for member, b in zip(('A', 'B'), x['bindings']):
            append('switch', x['pair_id'], member, b, 'W63', x['pair_id'], member, member+'+')
    shortages = {}
    source = w31.rows()
    for d in DOMAINS:
        by_photo = {}
        for b in sorted(source, key=lambda b: (str(b.get('episode_id', b['index'])), str(b['index']))):
            if b['dataset'] == d and c.photo(b) not in excluded:
                by_photo.setdefault(c.photo(b), b)
        selected = sorted(by_photo, key=lambda p: photo_order(2026101617, d, p))[:30]
        if len(selected) < 30:
            shortages[d] = len(selected)
        for j, p in enumerate(selected):
            b = by_photo[p]
            append('main', f'{d}_{j:03d}', 'one', b, 'W60', b['index'], 'A0', 'A0+', 'AB'[j % 2])
    jobs.sort(key=lambda x: (('main', 'reference', 'switch').index(x['panel']), x['job_id']))
    assert len({x['job_id'] for x in jobs}) == len(jobs)
    write(dest, dict(n=len(jobs), selection='metadata only; original S; no GT or scores used',
                     seeds=dict(main=2026101617, references=2026101618), shortages=shortages,
                     counts=dict(Counter(x['panel'] for x in jobs)), episodes=jobs))
    print('Prepared', len(jobs), 'tasks', dict(Counter(x['panel'] for x in jobs)), 'shortages', shortages)


def init():
    global _LIMIT, _ZERO_CHECKED
    import torch
    from threadpoolctl import threadpool_limits
    os.nice(19)
    c.initialize()
    for name in c.THREAD_VARS:
        os.environ[name] = '1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _LIMIT = threadpool_limits(limits=1)
    _ZERO_CHECKED = set()


def check_operator():
    rng = np.random.default_rng(NULL_SEED)
    t = rng.random((12, 20)); t /= t.sum(1, keepdims=True)
    perm = rng.permutation(20); inv = np.argsort(perm)
    relation = algebra.centered_proximity((4, 5), 2)
    expected = algebra.grid_neighbors((3, 4)) @ t @ relation[np.ix_(perm, perm)]
    actual = algebra.message_grid(t[:, inv], (3, 4), (4, 5), 2)[:, perm]
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-12
    return dict(permuted_fast_operator_error=error, query_GT_reads=0)


def kernel_inputs(r, q, y, fields):
    import torch
    lab = c.rc.lab
    mu = .5*(r.mean(0)+q.mean(0))
    sim = lab.unit(q-mu) @ lab.unit(r-mu).T
    k = torch.exp((sim-sim.max())/.05)
    df, db = k@y, k@(1-y)
    error = float(np.max(np.abs((df/(df+db).clamp_min(1e-30)).numpy()-fields['c32'])))
    assert error < 1e-6
    return k.numpy().astype(np.float64), lab.minmax(sim.max(1).values), error


def repaired_fields(kernel, novelty, y, fields, null=False, beta=BETA):
    import torch
    import torch.nn.functional as F
    if beta == 0:
        return fields, dict(zero=True)
    mass = kernel.sum(1, keepdims=True)
    prior = kernel/mass
    if null:
        perm = np.random.default_rng(NULL_SEED).permutation(kernel.shape[1])
        message = algebra.message_grid(prior[:, np.argsort(perm)], (64, 64), (64, 64), RADIUS)[:, perm]
    else:
        message = algebra.message_grid(prior, (64, 64), (64, 64), RADIUS)
    posterior = np.exp(beta*message) * prior
    posterior /= posterior.sum(1, keepdims=True)
    positive = (posterior > 0) & (prior > 0)
    kl = float(np.sum(posterior[positive]*np.log(posterior[positive]/prior[positive]))/len(prior))
    cc = torch.from_numpy((posterior@y.numpy()).astype(np.float32))
    k1 = posterior*mass
    mass_error = float(np.max(np.abs(k1.sum(1)-mass[:, 0])/mass[:, 0]))
    assert mass_error < 1e-12
    reverse = k1 / np.maximum(k1.sum(0, keepdims=True), np.finfo(np.float64).tiny)
    yy = y.numpy().astype(np.float64)
    bf = reverse@yy/max(float(yy.sum()), 1e-6)
    bb = reverse@(1-yy)/max(float((1-yy).sum()), 1e-6)
    labels = torch.from_numpy(fields['regions64'].astype(np.int64))
    hot = F.one_hot(labels, 64).float()
    region = lambda v: hot@((hot.T@v)/hot.sum(0).clamp_min(1))
    ff = dict(fields, c32=cc.numpy(), ind0_32=region(cc).numpy(),
              indn_32=region(cc*novelty).numpy(), back_f=bf.astype(np.float16), back_b=bb.astype(np.float16))
    assert all(np.isfinite(ff[k]).all() for k in ('c32', 'ind0_32', 'indn_32', 'back_f', 'back_b'))
    return ff, dict(row_mass_relative_error=mass_error, kl=kl,
                    message_abs_mean=float(np.mean(np.abs(message))), message_abs_max=float(np.max(np.abs(message))),
                    c_abs_mean=float(np.mean(np.abs(ff['c32']-fields['c32']))),
                    c_abs_max=float(np.max(np.abs(ff['c32']-fields['c32']))))


def field_from_arrays(z, arm, fallback):
    import torch
    lab = c.rc.lab
    if fallback:
        value = lab.minmax(lab.up(torch.from_numpy(z['c64']).reshape(64, 64))).numpy()
        return value.astype(np.float64), value, value
    u = lab.minmax(lab.up(torch.from_numpy(z[arm+'_u64']))).numpy()
    m = lab.up(torch.from_numpy(z[arm+'_m64'])).numpy()
    return .5*u.astype(np.float64)+.5*m.astype(np.float64), u, m


def head(binding, inputs):
    import w09_integration_lambda as w09
    import w19_simple_head as w19
    import w23_simple_pipeline as sh
    import produce
    captured = {}
    oi, ou, om = c.rc.independent, w09.solve, w19.mean_solve
    def point(*a, **k):
        v = ou(*a, **k); captured['u64'] = np.asarray(v).reshape(64, 64).astype(np.float32); return v
    def proto(*a, **k):
        v = om(*a, **k); captured['m64'] = np.asarray(v).astype(np.float32); return v
    c.rc.independent = lambda *a, **k: inputs
    w09.solve, w19.mean_solve = point, proto
    start = time.perf_counter()
    try:
        pre, _, rho = sh.simple_head(binding, crf=False)
    finally:
        c.rc.independent, w09.solve, w19.mean_solve = oi, ou, om
    bare_seconds = time.perf_counter()-start
    start = time.perf_counter()
    post = pre if rho > .9 else produce.crf(binding, pre)
    return pre, post, captured, dict(head_pre_seconds=bare_seconds, crf_seconds=time.perf_counter()-start)


def one(item):
    import torch
    started = time.perf_counter()
    b = item['binding']; arrays = {}; diagnostics = {}; timings = {}
    with c.rc.pc.inference_only(b), torch.inference_mode():
        start = time.perf_counter()
        inputs = c.rc.independent(b, {})
        independent_seconds = time.perf_counter()-start
        r, q, mask, y, fields = inputs
        fallback = float(fields['reference_rho'][0]) > .9
        k, novelty, c_error = (None, None, 0.) if fallback else kernel_inputs(r, q, y, fields)
        for arm in ARMS:
            start = time.perf_counter()
            ff, diag = (fields, dict(unchanged=True)) if arm == 'SH' or fallback else repaired_fields(k, novelty, y, fields, null=arm == 'Null')
            repair_seconds = time.perf_counter()-start
            pre, post, cap, elapsed = head(b, (r, q, mask, y, ff))
            arrays[arm], arrays[arm+'+'] = c.pack(pre), c.pack(post)
            if fallback:
                arrays['c64'] = fields['c32'].reshape(64, 64)
            else:
                arrays.update({arm+'_'+key: value for key, value in cap.items()})
                if arm != 'SH':
                    assert np.array_equal(arrays[arm+'_m64'], arrays['SH_m64'])
            field, u, m = field_from_arrays(arrays, arm, fallback)
            assert np.array_equal(field > .5, pre), 'Stored small fields do not reconstruct exact output'
            if arm == 'SH':
                arrays['common_positive'] = c.pack((u > .5) & (m > .5))
                arrays['common_negative'] = c.pack((u <= .5) & (m <= .5))
            diagnostics[arm] = diag
            timings[arm] = dict(repair_seconds=repair_seconds, **elapsed)
        zero_check = fallback not in _ZERO_CHECKED
        if zero_check:
            zero, _ = repaired_fields(k, novelty, y, fields, beta=0)
            assert zero is fields
            pre, post, _, _ = head(b, (r, q, mask, y, zero))
            assert np.array_equal(c.pack(pre), arrays['SH']) and np.array_equal(c.pack(post), arrays['SH+'])
            _ZERO_CHECKED.add(fallback)
    # Old predictions are inspected only after all new outputs for this task exist.
    old = item['baseline']
    assert sha(old['path']) == old['sha256']
    with np.load(old['path'], allow_pickle=False) as z:
        assert np.array_equal(z[old['before']], arrays['SH']), f'Original SH mismatch: {item["job_id"]}'
        assert np.array_equal(z[old['after']], arrays['SH+']), f'Original SH+ mismatch: {item["job_id"]}'
    path = OUT/'predictions'/(item['job_id']+'.npz')
    c.save_npz(path, **arrays)
    return dict(**{k: item[k] for k in ('job_id', 'panel', 'unit', 'member', 'half', 'index', 'dataset', 'photo', 'fold', 'cls')},
                rho=float(fields['reference_rho'][0]), fallback=fallback, c_error=c_error,
                zero_strength_checked=zero_check, baseline_pixel_identity=True,
                query_GT_reads=0, diagnostics=diagnostics, timings=timings,
                independent_seconds=independent_seconds, seconds=time.perf_counter()-started,
                prediction=str(path), prediction_sha256=sha(path))


def source_paths():
    return [Path(__file__), Path(algebra.__file__), ROOT/'scripts/x1012/w23_simple_pipeline.py',
            ROOT/'scripts/x1012/r5_candidate.py', ROOT/'scripts/x1012/w09_integration_lambda.py',
            ROOT/'scripts/x1012/w19_simple_head.py', ROOT/'scripts/x1013/common.py',
            ROOT/'scripts/x1013/produce.py', ROOT/'scripts/x1012/plan_common.py',
            ROOT/'docs/research/x1012_tasks/W67_correspondence_validation_PLAN.md']


def produce(workers=2):
    import psutil
    assert workers in (1, 2)
    manifest = json.loads((OUT/'manifest.json').read_text()); items = manifest['episodes']
    lock = (OUT/'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = dict(n=len(items), manifest_sha256=sha(OUT/'manifest.json'), beta=BETA,
                  radius=RADIUS, null_seed=NULL_SEED, workers=workers, worker_threads=1, mps=False,
                  sources={str(p.relative_to(ROOT)): sha(p) for p in source_paths()})
    cp = OUT/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config, 'Configuration changed; do not mix predictions'
    else:
        write(cp, config)
    write(OUT/'operator_verification.json', check_operator())
    process = psutil.Process()
    write(OUT/'execution.json', dict(pid=os.getpid(), create_time=process.create_time(), status='running', started_utc=utc()))
    path = OUT/'episodes.jsonl'
    records = list(read_records(path, 'job_id').values()) if path.exists() else []
    done = {r['job_id'] for r in records}; todo = [x for x in items if x['job_id'] not in done]
    prior_cpu = {}
    print('W67 CPU only', len(items), 'tasks; remaining', len(todo), flush=True)
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, path.open('a') as f, (OUT/'resources.jsonl').open('a') as resource:
            for r in pool.map(one, todo, chunksize=1):
                records.append(r); f.write(json.dumps(r)+'\n'); f.flush()
                if len(records) % 10 == 0 or len(records) == len(items):
                    now = time.monotonic(); total_cpu = 0.; rss = 0
                    for p in [process]+process.children(recursive=True):
                        try:
                            t = p.cpu_times(); value = t.user+t.system; rss += p.memory_info().rss
                            if p.pid in prior_cpu:
                                before, stamp = prior_cpu[p.pid]; total_cpu += (value-before)/max(now-stamp, 1e-9)
                            prior_cpu[p.pid] = (value, now)
                        except psutil.NoSuchProcess:
                            pass
                    cpu = 100*total_cpu/psutil.cpu_count()
                    sample = dict(n=len(records), utc=utc(), tree_cpu_machine_percent=cpu,
                                  rss_gib=rss/2**30, mps=0, method='interval CPU time; first sample warmup')
                    resource.write(json.dumps(sample)+'\n'); resource.flush()
                    assert cpu <= 50, f'CPU cap exceeded: {cpu}'
                    print(len(records), '/', len(items), 'task_s', round(r['seconds'], 2), 'treeCPU%', round(cpu, 2), flush=True)
        assert len(records) == len(items) and len({r['job_id'] for r in records}) == len(items)
        assert all(sha(r['prediction']) == r['prediction_sha256'] for r in records)
        assert all(sha(ROOT/p) == digest for p, digest in config['sources'].items())
        write(OUT/'complete.json', dict(n=len(items), episodes_sha256=sha(path), config_sha256=sha(cp),
                                       query_GT_reads=0, model_parameter_updates=0, encoder_calls=0, utc=utc()))
    except BaseException as e:
        write(OUT/'failure.json', dict(type=type(e).__name__, message=str(e), n=len(records), utc=utc()))
        raise
    finally:
        write(OUT/'execution.json', dict(pid=os.getpid(), status='complete' if (OUT/'complete.json').exists() else 'partial_or_failed', utc=utc()))
        lock.close()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    elif sys.argv[1] == 'verify': print(json.dumps(check_operator()))
    elif sys.argv[1] == 'produce': produce(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
