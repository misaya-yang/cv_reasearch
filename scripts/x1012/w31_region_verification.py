"""One preregistered W31 K0 round, revised plan: S only, four selectors.

fields -> check-fields -> encode -> select -> check -> report.
All candidate pools, four score vectors and selections are sealed before query
labels are opened. W27's sealed SH masks supply a label-free identity check.
No T/holdout, CRF, encoder fallback, tuning, or W30 code path exists here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1013'), str(ROOT / 'scripts/x1012')]
import common as c
import score as sc
import w23_simple_pipeline as sh
import w27_sh_single_line as w27

OUT = c.task_path('W31_K0_revised')
PLAN = ROOT / 'docs/research/x1012_tasks/W31_region_verification_PLAN.md'
REPORT = ROOT / 'docs/research/x1012_tasks/W31_region_verification_REPORT.md'
EVIDENCE = ROOT / 'evidence/research/w31_region_verification_K0_S.jsonl.gz'
WEIGHTS = c.ASSETS / 'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'
LADDER = (-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.)
SIZE, N, BATCH = 448, 1024 ** 2, 10
SELECTORS = ('J', 'pool', 'masked', 'crop')
ARMS = ('natural',) + SELECTORS + ('oracle',)
LABELS = {'natural': '自然 SH', 'J': 'J', 'pool': '缓存池化',
          'masked': '遮背景重编码', 'crop': '同框不遮背景裁剪', 'oracle': '候选 oracle'}
sha = w27.sha


def read_records(name):
    p = OUT / name
    result = {}
    if p.exists():
        for line in p.read_text().splitlines():
            e = json.loads(line)
            assert e['index'] not in result, f'duplicate index in {p}'
            result[e['index']] = e
    return result


def write(name, value):
    c.write(OUT / name, value)


def initialize():
    c.initialize()


def sample():
    bs = sh.rows_for('S')
    assert len(bs) == 1285 and all(c.half(b) == 'S' for b in bs)
    assert dict(Counter(b['dataset'] for b in bs)) == w27.EXPECTED
    return bs


def meta(b):
    return w27.metadata(b)


def manifest():
    bs = sample()
    sources = [ROOT / 'scripts/x1012' / n for n in
               ('r5_candidate.py', 'w23_simple_pipeline.py', 'w27_sh_single_line.py',
                'w09_integration_lambda.py', 'w19_simple_head.py', 'lab.py', 'plan_common.py')]
    sources += sorted((ROOT / 'scripts/x1013').glob('*.py'))
    sources += sorted((c.RUNTIME / 'runtime').rglob('*.py'))
    return {'code_sha256': sha(__file__), 'plan_sha256_at_start': sha(PLAN),
            'sources': {str(p): sha(p) for p in sources}, 'weights_sha256': sha(WEIGHTS),
            'basis_sha256': sha(c.ASSETS / 'native_assets/positional_basis.pt'),
            'W27_predictions_sha256': sha(c.task_path('W27') / 'predictions.jsonl'),
            'indices': [str(b['index']) for b in bs],
            'bindings_sha256': hashlib.sha256(json.dumps(
                [{'row': b['row'], 'fit': b['fit']} for b in bs], sort_keys=True).encode()).hexdigest(),
            'ladder': list(LADDER), 'size': SIZE, 'dtype': 'float32', 'batch': BATCH,
            'selectors': list(SELECTORS), 'split': 'S', 'T_read': False,
            'new_photo_holdout_read': False, 'started_unix_ns': time.time_ns()}


def verify():
    m = c.read(OUT / 'manifest.json')
    assert m['code_sha256'] == sha(__file__), 'Code changed after run began'
    # The result section may be appended later; the frozen copy is immutable.
    assert m['plan_sha256_at_start'] == sha(OUT / 'plan_frozen.md')
    for p, h in m['sources'].items():
        assert sha(p) == h, f'frozen source changed: {p}'
    assert sha(WEIGHTS) == m['weights_sha256']
    assert sha(c.task_path('W27') / 'predictions.jsonl') == m['W27_predictions_sha256']


def snapshot():
    import psutil
    return {'unix_ns': time.time_ns(), 'cpu_percent': psutil.cpu_percent(interval=.1),
            'available_gib': psutil.virtual_memory().available / 1024 ** 3,
            'rss_gib': psutil.Process().memory_info().rss / 1024 ** 3}


def masks_from_field(field):
    natural = field > .5
    a0 = int(natural.sum())
    order = np.argsort(-field.ravel(), kind='stable')
    # Tie preference is specified in the plan, including after clipping/dedup.
    masks, ss, seen = [], [], {}
    for s in sorted(LADDER, key=lambda x: (abs(x), x)):
        k = min(N, max(0, int(np.floor(a0 * 2 ** s + .5))))
        if k in seen:
            continue
        mk = np.zeros(N, bool)
        mk[order[:k]] = True
        mk = mk.reshape(1024, 1024)
        if s == 0:
            assert np.array_equal(mk, natural)
        masks.append(mk)
        ss.append(s)
        seen[k] = len(masks) - 1
    assert ss[0] == 0 and np.array_equal(masks[0], natural)
    return masks, ss


def fields_one(b):
    import torch
    import torch.nn.functional as F
    import w09_integration_lambda as w09
    import w19_simple_head as w19
    assert c.half(b) == 'S'
    cap = {}
    ou, om, oi = w09.solve, w19.mean_solve, c.rc.independent

    def point(*a, **k):
        v = ou(*a, **k)
        cap['u'] = np.asarray(v).copy()
        return v

    def proto(*a, **k):
        v = om(*a, **k)
        cap['m'] = np.asarray(v).copy()
        return v

    def independent(*a, **k):
        v = oi(*a, **k)
        cap['rq'] = v
        return v

    t0 = time.perf_counter()
    with c.rc.pc.inference_only(b):
        w09.solve, w19.mean_solve, c.rc.independent = point, proto, independent
        try:
            pre, _, rho = sh.simple_head(b, crf=False)
        finally:
            w09.solve, w19.mean_solve, c.rc.independent = ou, om, oi
        r, q, _, y, f = cap.pop('rq')
        if rho > .9:
            masks, ss, js, ps = [pre], [0.], [0.], [0.]
        else:
            u = c.rc.lab.minmax(c.rc.lab.up(torch.from_numpy(
                cap['u'].reshape(64, 64).astype(np.float32)))).numpy()
            m = c.rc.lab.up(torch.from_numpy(cap['m'])).numpy()
            field = .5 * m.astype(np.float64) + .5 * u.astype(np.float64)
            assert np.array_equal(field > .5, pre)
            masks, ss = masks_from_field(field)
            ref = F.normalize(y.float() @ r.float() / y.sum().clamp_min(1e-9), dim=0)
            js, ps = [], []
            for mk in masks:
                cov = c.rc.lab.coverage64(torch.from_numpy(mk).float()).flatten()
                pooled = F.normalize(cov @ q.float() / cov.sum().clamp_min(1e-9), dim=0)
                ps.append(float(pooled @ ref))
                js.append(float(c.rc.quality(mk, f['c32'], f['back_f'].astype(np.float64))))
    # W27 is a previously sealed prediction, not a query annotation.
    wp = c.task_path('W27') / 'predictions' / (c.key(b) + '.npz')
    with np.load(wp, allow_pickle=False) as z:
        assert np.array_equal(pre, c.unpack(z['SH'])), 'natural differs from W27'
    p = OUT / 'candidates' / (c.key(b) + '.npz')
    c.save_npz(p, m=np.stack([c.pack(mk) for mk in masks]))
    return dict(meta(b), rho=rho, fallback=rho > .9, s=ss,
                areas=[int(mk.sum()) for mk in masks], J=js, pool=ps,
                payload=str(p), sha256=sha(p), W27_xor=0, raw_cache_requests=2,
                encoder_calls=0, query_GT_reads=0, seconds=time.perf_counter() - t0)


def fields(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'manifest.json').exists():
        (OUT / 'plan_frozen.md').write_text(PLAN.read_text())
        write('manifest.json', manifest())
    verify()
    stamp = time.time_ns()
    top = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True, check=True).stdout
    (OUT / f'top_{stamp}.txt').write_text(top)
    done = read_records('fields.jsonl')
    for e in done.values():
        assert sha(e['payload']) == e['sha256']
    todo = [b for b in sample() if str(b['index']) not in done]
    t0, resources = time.perf_counter(), [snapshot()]
    print(f'fields S1285, saved={len(done)}, todo={len(todo)}, workers={workers}, torch_threads=2', flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, (OUT / 'fields.jsonl').open('a') as stream:
        for n, e in enumerate(pool.map(fields_one, todo, chunksize=1), 1):
            stream.write(json.dumps(e, allow_nan=False) + '\n')
            stream.flush()
            if n % 50 == 0 or n == len(todo):
                resources.append(snapshot())
                write(f'fields_resources_{stamp}.json', resources)
                print(f'fields {len(done)+n}/1285 elapsed={time.perf_counter()-t0:.1f}s '
                      f'available={resources[-1]["available_gib"]:.1f}GiB', flush=True)
    write(f'fields_time_{stamp}.json', {'seconds': time.perf_counter() - t0, 'n': len(todo),
                                      'workers': workers, 'torch_threads': 2})
    check_fields()


def check_fields():
    verify()
    records = read_records('fields.jsonl')
    assert len(records) == 1285
    expected = {str(b['index']) for b in sample()}
    assert set(records) == expected
    active = Counter(e['dataset'] for e in records.values() if not e['fallback'])
    assert {d: active[d] for d in c.DOMAINS} == w27.EXPECTED_ACTIVE
    for e in records.values():
        assert e['W27_xor'] == e['encoder_calls'] == e['query_GT_reads'] == 0
        assert sha(e['payload']) == e['sha256']
        assert e['s'][0] == 0 and len(e['s']) == len(set(e['areas']))
    check = {'passed': True, 'n': 1285, 'active': sum(active.values()),
             'natural_W27_pixel_exact': 1285, 'query_GT_reads': 0,
             'fields_sha256': sha(OUT / 'fields.jsonl'), 'unix_ns': time.time_ns()}
    write('fields_check.json', check)
    print('Field check PASS: natural W27 pixel exact 1285/1285; active1090; query_GT_reads=0', flush=True)


def render(img, mask, masked):
    """Exact tight bbox, shorter axis zero-padded to a square; no dilation/margin."""
    import torch
    import torch.nn.functional as F
    ys, xs = torch.nonzero(mask, as_tuple=True)
    assert len(ys)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    tile = img[:, y0:y1, x0:x1]
    if masked:
        tile = tile * mask[None, y0:y1, x0:x1]
    top, left = (side - h) // 2, (side - w) // 2
    canvas = torch.zeros((3, side, side), dtype=torch.float32)
    canvas[:, top:top+h, left:left+w] = tile
    return F.interpolate(canvas[None], size=(SIZE, SIZE), mode='bilinear',
                         align_corners=False, antialias=True)[0]


def mps_busy():
    busy = []
    # macOS can raise an uncaught SystemError while psutil fetches protected
    # processes' argv. ps supplies the same read-only inventory safely.
    output = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,args='],
                                     text=True, errors='replace')
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        executable = cmd.split()[0]
        if int(pid) == os.getpid() or 'python' not in Path(executable).name.lower():
            continue
        if 'w30_confirm.py' in cmd or ('w31_demonstration.py' in cmd and 'encode' in cmd.split()):
            busy.append({'pid': int(pid), 'command': cmd})
    return busy


def encode():
    import torch
    import timm
    from safetensors.torch import load_file
    from ics.official_data import decoded_rgb
    verify()
    assert c.read(OUT / 'fields_check.json')['passed']
    assert not mps_busy(), f'Another MPS/W30 process is active: {mps_busy()}'
    initialize()
    t0 = time.perf_counter()
    model = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
    model.load_state_dict(load_file(str(WEIGHTS)), strict=True)
    model = model.eval().float().to('mps')
    host, _ = c.rc.pc.host_cpu()
    recs = read_records('fields.jsonl')
    binds = {str(b['index']): b for b in sample()}
    done = read_records('embeddings.jsonl')
    for e in done.values():
        assert sha(e['payload']) == e['sha256']
    todo = [i for i in recs if not recs[i]['fallback'] and i not in done]
    stamp, total_views, model_seconds = time.time_ns(), 0, 0.
    resources = [snapshot()]
    print(f'encode S active1090, saved={len(done)}, todo={len(todo)}, MPS FP32 batch<={BATCH}', flush=True)
    with (OUT / 'embeddings.jsonl').open('a') as stream, torch.inference_mode():
        for n, i in enumerate(todo, 1):
            if mps_busy():
                raise RuntimeError('W30/another encoder started; saved progress, release MPS')
            b, e = binds[i], recs[i]
            assert sha(e['payload']) == e['sha256']
            start = time.perf_counter()
            with c.rc.pc.inference_only(b):
                row = b['row']
                q = host._transform(decoded_rgb(c.ASSETS / row['query_path'], row.get('query_crop')))
                r = host._transform(decoded_rgb(c.ASSETS / row['reference_path'], row.get('reference_crop')))
                rm = c.rc.lab.mask1024(row['reference_mask_path'])
                with np.load(e['payload'], allow_pickle=False) as z:
                    ms = [torch.from_numpy(c.unpack(x)) for x in z['m']]
                # Empty natural means all candidates are empty. Preserve it without encoding.
                if not ms[0].any():
                    emb = {a: np.zeros((len(ms) + 1, 1024), np.float32) for a in ('masked', 'crop')}
                    views_n, forward = 0, 0.
                else:
                    assert rm.any()
                    views = []
                    for masked in (True, False):
                        views.append(render(r, rm, masked))
                        views.extend(render(q, m, masked) for m in ms)
                    prep = time.perf_counter() - start
                    outputs, forward = [], 0.
                    for j in range(0, len(views), BATCH):
                        x = torch.stack(views[j:j+BATCH]).to('mps')
                        torch.mps.synchronize()
                        tf = time.perf_counter()
                        tok = model.forward_features(x)
                        vec = torch.nn.functional.normalize(tok[:, 0].float(), dim=1).cpu().numpy()
                        torch.mps.synchronize()
                        forward += time.perf_counter() - tf
                        outputs.append(vec)
                        del x, tok
                    vectors = np.concatenate(outputs)
                    count = len(ms) + 1
                    emb = {'masked': vectors[:count], 'crop': vectors[count:]}
                    views_n = len(views)
                    assert vectors.shape == (views_n, 1024) and np.isfinite(vectors).all()
                    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6)
            p = OUT / 'embeddings' / (c.key(b) + '.npz')
            c.save_npz(p, **emb)
            record = {'index': i, 'payload': str(p), 'sha256': sha(p), 'views': views_n,
                      'model_seconds': forward, 'total_seconds': time.perf_counter() - start,
                      'query_GT_reads': 0}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            total_views += views_n
            model_seconds += forward
            if n % 10 == 0 or n == len(todo):
                resources.append(snapshot())
                resources[-1].update(mps_allocated_gib=torch.mps.current_allocated_memory() / 1024 ** 3,
                                     mps_driver_gib=torch.mps.driver_allocated_memory() / 1024 ** 3)
                write(f'encode_resources_{stamp}.json', resources)
                print(f'encoded {len(done)+n}/1090, elapsed={time.perf_counter()-t0:.1f}s, '
                      f'views={total_views}, available={resources[-1]["available_gib"]:.1f}GiB', flush=True)
            if n == 10:
                write('prefix_timing.json', {'episodes': 10, 'views': total_views,
                      'elapsed_seconds_including_model_load': time.perf_counter() - t0,
                      'model_seconds': model_seconds, 'definition_changed': False})
    write(f'encode_time_{stamp}.json', {'seconds': time.perf_counter() - t0, 'n': len(todo),
                                      'views': total_views, 'model_seconds': model_seconds})
    print('All revised K0 embeddings saved; no query GT read.', flush=True)


def pick(values, ss):
    assert len(values) == len(ss) and np.isfinite(values).all()
    return min(range(len(ss)), key=lambda k: (-values[k], abs(ss[k]), ss[k]))


def select():
    verify()
    fields, embs = read_records('fields.jsonl'), read_records('embeddings.jsonl')
    assert {i for i, e in fields.items() if not e['fallback']} == set(embs)
    selected = []
    for i, e in fields.items():
        assert sha(e['payload']) == e['sha256']
        values = {a: e[a] for a in ('J', 'pool')}
        if e['fallback']:
            values.update(masked=[0.], crop=[0.])
        else:
            assert sha(embs[i]['payload']) == embs[i]['sha256']
            with np.load(embs[i]['payload'], allow_pickle=False) as z:
                for a in ('masked', 'crop'):
                    values[a] = (z[a][1:] @ z[a][0]).astype(float).tolist()
        sels = {a: pick(values[a], e['s']) for a in SELECTORS}
        assert not e['fallback'] or all(k == 0 for k in sels.values())
        selected.append({'index': i, 'scores': values, 'selected': sels})
    p = OUT / 'selections.json'
    payload = json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == payload
    else:
        p.write_text(payload)
    seal = {'selections_sha256': sha(p), 'fields_sha256': sha(OUT / 'fields.jsonl'),
            'embeddings_sha256': sha(OUT / 'embeddings.jsonl'), 'n': len(fields),
            'sealed_unix_ns': time.time_ns(), 'T_read': False, 'new_photo_holdout_read': False,
            'query_GT_reads_before_seal': 0}
    if (OUT / 'sealed.json').exists():
        assert c.read(OUT / 'sealed.json')['selections_sha256'] == seal['selections_sha256']
    else:
        write('sealed.json', seal)
    print('Sealed all four selections on S1285. Query GT reads=0.', flush=True)


def score_one(job):
    from scipy.stats import spearmanr
    b, e, sel = job
    assert c.half(b) == 'S' and sha(e['payload']) == e['sha256']
    with np.load(e['payload'], allow_pickle=False) as z:
        ms = [c.unpack(x) for x in z['m']]
    gt, valid = c.truth(b)
    ius = [c.iu(mk, gt, valid) for mk in ms]
    vals = np.array([100 * x[0] / max(x[1], 1) for x in ius])
    oracle = pick(vals, e['s'])
    selection = {'natural': 0, **sel['selected'], 'oracle': oracle}
    r = dict(meta(b), fallback=e['fallback'], rho=e['rho'], gt=int(gt.sum()),
             natural_area=int((ms[0] & valid).sum()), ius=ius,
             **{a: ius[k] for a, k in selection.items()})
    r['selected_s'] = {a: e['s'][k] for a, k in selection.items()}
    r['oracle_at_min'] = e['areas'][oracle] == min(e['areas'])
    r['oracle_at_max'] = e['areas'][oracle] == max(e['areas'])
    r['within_episode_rank'] = {}
    for a in SELECTORS:
        v = sel['scores'][a]
        r['within_episode_rank'][a] = (float(spearmanr(v, vals).statistic)
             if len(v) > 1 and np.ptp(v) > 0 and np.ptp(vals) > 0 else None)
    r['FP_FN'] = {a: {'FP': ius[k][2] - ius[k][0], 'FN': r['gt'] - ius[k][0]}
                  for a, k in selection.items()}
    return r


def check(workers):
    verify()
    seal = c.read(OUT / 'sealed.json')
    assert seal['n'] == 1285 and seal['query_GT_reads_before_seal'] == 0
    for filename, key in (('selections.json', 'selections_sha256'), ('fields.jsonl', 'fields_sha256'),
                          ('embeddings.jsonl', 'embeddings_sha256')):
        assert sha(OUT / filename) == seal[key]
    fields = read_records('fields.jsonl')
    selected = {e['index']: e for e in c.read(OUT / 'selections.json')}
    done = read_records('episodes.jsonl')
    jobs = [(b, fields[str(b['index'])], selected[str(b['index'])]) for b in sample()
            if str(b['index']) not in done]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, (OUT / 'episodes.jsonl').open('a') as stream:
        for n, r in enumerate(pool.map(score_one, jobs, chunksize=4), 1):
            stream.write(json.dumps(r, allow_nan=False) + '\n')
            stream.flush()
            if n % 200 == 0 or n == len(jobs):
                print(f'identity scoring {len(done)+n}/1285', flush=True)
    records = read_records('episodes.jsonl')
    with gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt') as stream:
        old = {e['index']: e for e in map(json.loads, stream)}
    equal = sum(r['natural'][:2] == old[i]['SH'][:2] for i, r in records.items())
    ok = len(records) == equal == 1285
    result = {'passed': ok, 'natural_IU_equal_W27': equal, 'n': len(records),
              'natural_pixel_equal_W27': 1285, 'selections_sealed_before_GT': True,
              'episodes_sha256': sha(OUT / 'episodes.jsonl'),
              'seconds': time.perf_counter() - t0, 'T_read': False, 'new_photo_holdout_read': False}
    write('checks.json', result)
    assert ok, 'Stop: natural I/U identity check failed'
    print('All implementation checks PASS; natural I/U exact W27 1285/1285. Results may now be analysed.', flush=True)


def scope(records):
    rng = np.random.default_rng(c.SEED)
    domains, boot = {}, {}
    comparisons = [(a, 'natural') for a in ARMS if a != 'natural'] + [
        ('masked', a) for a in ('J', 'pool', 'crop', 'oracle')]
    for d in c.DOMAINS:
        rows = [r for r in records if r['dataset'] == d]
        if not rows:
            continue
        boot[d] = w27.bootstrap_domain(rows, ARMS, rng)
        scores = {a: sc.miou(rows, a) for a in ARMS}
        effects = {}
        for a, b in comparisons:
            lo, hi = np.percentile(boot[d][a] - boot[d][b], [2.5, 97.5]).tolist()
            effects[f'{a}-{b}'] = {'pp': scores[a] - scores[b], 'lo': lo, 'hi': hi}
        domains[d] = {'n': len(rows), 'scores': scores, 'effects': effects,
                      'episode_IoU_mean': {a: float(np.mean([sc.episode_iou(r, a) for r in rows])) for a in ARMS}}
    means = {a: float(np.mean([v['scores'][a] for v in domains.values()])) for a in ARMS}
    mean_boot = {a: np.mean([v[a] for v in boot.values()], axis=0) for a in ARMS}
    effects = {}
    for a, b in comparisons:
        lo, hi = np.percentile(mean_boot[a] - mean_boot[b], [2.5, 97.5]).tolist()
        effects[f'{a}-{b}'] = {'pp': means[a] - means[b], 'lo': lo, 'hi': hi}
    return {'n': len(records), 'domains': domains, 'mean': {'scores': means, 'effects': effects},
            'episode_IoU_mean': {a: float(np.mean([sc.episode_iou(r, a) for r in records])) for a in ARMS}}


def diagnostics(rows):
    active = [r for r in rows if not r['fallback']]
    out = {'active_n': len(active), 'selection_s': {}, 'within_episode_spearman_median': {},
           'regret_episode_IoU_pp': {}, 'coverage_groups': {}}
    for a in ARMS:
        out['selection_s'][a] = dict(Counter(str(r['selected_s'][a]) for r in active))
        regret = [sc.episode_iou(r, 'oracle') - sc.episode_iou(r, a) for r in active]
        out['regret_episode_IoU_pp'][a] = {'mean': float(np.mean(regret)), 'median': float(np.median(regret))}
    for a in SELECTORS:
        values = [r['within_episode_rank'][a] for r in active if r['within_episode_rank'][a] is not None]
        out['within_episode_spearman_median'][a] = {'n': len(values), 'median': float(np.median(values)) if values else None}
    out['oracle_at_window_min'] = sum(r['oracle_at_min'] for r in active)
    out['oracle_at_window_max'] = sum(r['oracle_at_max'] for r in active)
    for name, predicate in (('under', lambda r: r['natural_area'] < r['gt']),
                            ('over', lambda r: r['natural_area'] > r['gt']),
                            ('equal', lambda r: r['natural_area'] == r['gt'])):
        group = [r for r in active if predicate(r)]
        out['coverage_groups'][name] = {'n': len(group), 'FP_FN_change_vs_natural': {
            a: {k: sum(r['FP_FN'][a][k] - r['FP_FN']['natural'][k] for r in group) for k in ('FP', 'FN')}
            for a in ARMS if a != 'natural'}}
    return out


def cell(v):
    return f'{v["pp"]:+.2f} [{v["lo"]:+.2f}, {v["hi"]:+.2f}]'


def report():
    verify()
    checks = c.read(OUT / 'checks.json')
    assert checks['passed'] and checks['episodes_sha256'] == sha(OUT / 'episodes.jsonl')
    rows = list(read_records('episodes.jsonl').values())
    full = scope(rows)
    active = scope([r for r in rows if not r['fallback']])
    e = full['mean']['effects']
    gates = {'vs_natural': e['masked-natural']['pp'] >= 1 and e['masked-natural']['lo'] > 0,
             'vs_J': e['masked-J']['pp'] >= .5 and e['masked-J']['lo'] > 0,
             'increment_vs_pool': e['masked-pool']['lo'] > 0,
             'increment_vs_crop': e['masked-crop']['lo'] > 0}
    passed = all(gates.values())
    timings = {key: [c.read(p) for p in sorted(OUT.glob(pattern))] for key, pattern in (
        ('fields_runs', 'fields_time_*.json'), ('encoding_runs', 'encode_time_*.json'))}
    embs = list(read_records('embeddings.jsonl').values())
    timings.update(encoding_views=sum(e['views'] for e in embs),
                   encoding_views_per_arm=sum(e['views'] for e in embs) // 2,
                   model_seconds=sum(e['model_seconds'] for e in embs), scoring_seconds=checks['seconds'])
    result = {'checks': checks, 'full': full, 'active': active, 'gates': gates,
              'pass_all_four': passed, 'diagnostics': diagnostics(rows), 'timings': timings,
              'T_read': False, 'new_photo_holdout_read': False,
              'code_sha256': sha(__file__), 'selections_sha256': sha(OUT / 'selections.json')}
    write('report.json', result)
    with gzip.open(EVIDENCE, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# W31 修正版 K0：候选范围验证结果', '',
             '固定定义和读法见 [W31计划](W31_region_verification_PLAN.md)。仅W27同批S1285例，'
             '其中1090非回退例；195回退例原样保留。不读T或新照片留出，不改W30，不做CRF。', '',
             '## 1 实现检查', '',
             '候选自然掩码与W27逐像素一致1285/1285；四个选择向量及选择索引封存后才打开S查询GT；'
             '自然掩码整数I/U再次复现W27 1285/1285。全部检查PASS后才分析结果。', '',
             '## 2 十域主比较', '',
             '绝对分数为(fold,class)池化I/U后槽等权mIoU，差值单位pp，附500次查询照片成组配对95%区间，'
             'seed=c.SEED。十域均值逐域等权。各臂使用同一候选池；遮背景重编码是预定主方法。', '',
             '| 域 | 自然SH | J−SH | 缓存池化−SH | 遮背景重编码−SH | 同框裁剪−SH | oracle−SH |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        v = full['domains'][d]
        lines.append(f'| {name} | {v["scores"]["natural"]:.2f} | ' + ' | '.join(
            cell(v['effects'][f'{a}-natural']) for a in ('J', 'pool', 'masked', 'crop', 'oracle')) + ' |')
    lines.append(f'| 十域均值 | {full["mean"]["scores"]["natural"]:.2f} | ' + ' | '.join(
        cell(e[f'{a}-natural']) for a in ('J', 'pool', 'masked', 'crop', 'oracle')) + ' |')
    lines += ['', '## 3 遮背景验证的增量', '', '| 域 | 相对J | 相对缓存池化 | 相对同框裁剪 |', '|---|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        lines.append(f'| {name} | ' + ' | '.join(cell(full['domains'][d]['effects'][f'masked-{a}'])
                    for a in ('J', 'pool', 'crop')) + ' |')
    lines.append('| 十域均值 | ' + ' | '.join(cell(e[f'masked-{a}']) for a in ('J', 'pool', 'crop')) + ' |')
    lines += ['', '## 4 非回退九域', '', '| 臂 | 绝对mIoU | 相对自然SH pp [95%区间] |', '|---|---:|---:|']
    for a in ARMS:
        v = active['mean']
        lines.append(f'| {LABELS[a]} | {v["scores"][a]:.2f} | ' +
                     (cell(v['effects'][f'{a}-natural']) if a != 'natural' else '—') + ' |')
    lines += ['', '完整逐域非回退结果见外置report.json。Fundus非回退例数为0，不纳入九域均值。', '',
              '## 5 oracle和选择诊断', '',
              'oracle按逐例IoU选择候选，再报告同一槽池化指标；不是域级mIoU严格最大值。自然掩码在候选中。', '',
              f'全1285例的逐例IoU均值：自然SH {full["episode_IoU_mean"]["natural"]:.2f}，'
              f'oracle {full["episode_IoU_mean"]["oracle"]:.2f}；非回退1090例：'
              f'{active["episode_IoU_mean"]["natural"]:.2f} / {active["episode_IoU_mean"]["oracle"]:.2f}。', '',
              '```json', json.dumps(result['diagnostics'], ensure_ascii=False, indent=2), '```', '',
              '边界计数只表示本九候选池最优位于边缘，不知道窗口外最优，也不作额外候选搜索。'
              'FP/FN收支为组内像素总数，不等同于mIoU的线性分解。', '', '## 6 预定读法', '']
    for key, value in gates.items():
        lines.append(f'- {key}: {"PASS" if value else "FAIL"}')
    conclusion = ('四项门槛全部通过，下一步仅做第4节S任务粒度干预；K1/K2未执行，W30仍为SH+。' if passed else
                  'K0未通过。停止本固定CLS范围验证构造，不调填充、编码层、尺寸或候选密度；'
                  '不进入任务粒度干预或K1/K2，W30仍按冻结SH+执行。候选容量与选择能力分开解释。')
    lines += ['', conclusion, '', '该结果仅检验本固定构造，不能证明所有重新编码范围验证均无效；'
              '不能把控制臂更好直接替换为主方法，更不能用本S结果宣称独立泛化或新的范围原理成立。', '',
              '## 7 实际成本与复现', '', '```json', json.dumps(timings, ensure_ascii=False, indent=2), '```', '',
              '编码臂各自包含每例参考和去重后候选，未跨例缓存参考；未保存整图新O24特征。'
              '相同候选的两种渲染各编码一次，单MPS进程、FP32、batch≤10、torch2线程。'
              'CPU字段阶段最多2worker×2线程。另一个会话的旧W31定义不同，未混用它的掩码/嵌入/选择或成绩。', '',
              f'外置断点与结果：`{OUT}`。紧凑逐例证据：`{EVIDENCE.relative_to(ROOT)}`。'
              f'代码SHA256：`{sha(__file__)}`。']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(conclusion, flush=True)
    for key, v in e.items():
        print(key, cell(v), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fields', 'check-fields', 'encode', 'select', 'check', 'report'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4
    if args.command in ('fields', 'check'):
        globals()[args.command](args.workers)
    else:
        globals()[args.command.replace('-', '_')]()
