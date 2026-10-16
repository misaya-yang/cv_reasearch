"""W70 fixed information probe, not a segmentation modification.

Prepare from old S accounting; seal label-free layer scores before evaluating
query labels. One CPU encoder / two threads. No feature cache or new masks.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
import fcntl
import gzip
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import common as c

OUT = c.task_path('W70_midlayer')
DOMAINS = ('coco', 'pascal_part', 'paco_part', 'isic')
LAYERS = (19, 20, 23)
BASIS = c.OUT/'fundus_layer_resolution50_20261013/basis/1024.pt'
MODEL = c.ASSETS/'demo4_cache/models/dinov3-vitl16-timm'
ENCODER_SOURCE = c.OUT/'x1013_official/frozen/encoder.py'
PLAN = ROOT/'docs/research/x1012_tasks/W70_midlayer_attribution_PLAN.md'
NULL_SEED = 2026101670
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: datetime.now(timezone.utc).isoformat()
write = c.write


def read_jsonl(path, key='index'):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return {str(r[key]): r for r in map(json.loads, stream)}


def order(seed, value):
    return hashlib.sha256(f'{seed}/{value}'.encode()).hexdigest()


def clean_binding(b):
    return {k: b[k] for k in ('index', 'dataset', 'row', 'fit', '_split') if k in b}


def task(b, job_id, baseline, group=None, repeat=None):
    assert c.half(b) == 'S'
    return dict(job_id=job_id, index=str(b['index']), dataset=b['dataset'], photo=c.photo(b),
                group=group, repeat=repeat, binding=clean_binding(b), baseline=baseline)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        print('Retained', sha(path)); return
    groups = []; excluded = set()
    switches = json.loads((c.task_path('W63')/'manifest.json').read_text())['episodes']
    old_switch = read_jsonl(c.task_path('W63')/'episodes.jsonl', 'pair_id')
    for domain in ('pascal_part', 'paco_part'):
        selected = sorted([x for x in switches if x['dataset'] == domain],
                          key=lambda x: order(2026101672, x['pair_id']))[:3]
        assert len(selected) == 3
        for x in selected:
            old = old_switch[x['pair_id']]; members = []
            for label, b in zip(('A', 'B'), x['bindings']):
                members.append(task(b, 'switch_'+x['pair_id']+'_'+label,
                                    dict(path=old['prediction'], sha256=old['prediction_sha256'], key=label)))
            groups.append(dict(group_id='switch_'+x['pair_id'], panel='switch', members=members))
            excluded.add((domain, x['photo']))
    refs = json.loads((c.task_path('W61')/'manifest.json').read_text())['episodes']
    old_ref = read_jsonl(c.task_path('W61')/'episodes.jsonl')
    for domain in DOMAINS:
        eligible = [x for x in refs if x['dataset'] == domain and (domain, x['photo']) not in excluded]
        x = min(eligible, key=lambda x: order(2026101673, domain+'/'+x['photo']))
        old = old_ref[x['index']]; members = []
        for j, b in enumerate(x['bindings']):
            members.append(task(b, f'reference_{domain}_{x["index"]}_R{j}',
                                dict(path=old['prediction'], sha256=old['prediction_sha256'], key=f'R{j}_SH')))
        groups.append(dict(group_id='reference_'+domain+'_'+x['index'], panel='reference', members=members))
        excluded.add((domain, x['photo']))

    binds = {str(b['index']): b for b in c.rows() if c.half(b) == 'S'}
    candidates = []
    for name, evidence in (('W27', 'w27_sh_single_line_S'), ('W28', 'w28_sh_second_batch_S')):
        records = read_jsonl(ROOT/f'evidence/research/{evidence}.jsonl.gz')
        preds = read_jsonl(c.task_path(name)/'predictions.jsonl')
        for index, r in records.items():
            if r['dataset'] not in DOMAINS or r['rho'] > .9:
                continue
            i, u, p = r['SH']; g = r['gt']
            precision, recall, jiou = i/max(p, 1), i/max(g, 1), i/max(u, 1)
            category = ('add' if precision >= .75 and recall <= .6 else
                        'remove' if recall >= .75 and precision <= .6 else
                        'correct' if jiou >= .75 else None)
            if category:
                candidates.append(dict(record=r, pred=preds[index], category=category, source=name,
                                       precision=precision, recall=recall, iou=jiou))
    used = set(excluded); availability = {}; shortages = {}
    for domain in DOMAINS:
        for category in ('add', 'remove', 'correct'):
            by_photo = {}
            for x in sorted(candidates, key=lambda x: x['record']['index']):
                r = x['record']; key = (domain, r['photo'])
                if r['dataset'] == domain and x['category'] == category and key not in used:
                    by_photo.setdefault(r['photo'], x)
            picked = sorted(by_photo, key=lambda photo: order(2026101671, domain+'/'+photo))[:6]
            availability[domain+'/'+category] = len(by_photo)
            if len(picked) < 6:
                shortages[domain+'/'+category] = len(picked)
            for n, photo in enumerate(picked):
                x = by_photo[photo]; r, p = x['record'], x['pred']; b = binds[r['index']]
                assert c.photo(b) == photo
                baseline = dict(path=p['prediction'], sha256=p['payload_sha256'], key='SH',
                                expected_iu=r['SH'], expected_gt=r['gt'])
                member = task(b, f'main_{domain}_{category}_{n:02d}', baseline, category, 'AB'[n % 2])
                member['old_selection'] = {k: x[k] for k in ('precision', 'recall', 'iou', 'source')}
                groups.append(dict(group_id=member['job_id'], panel='main', members=[member]))
                used.add((domain, photo))
    groups.sort(key=lambda x: (('main', 'reference', 'switch').index(x['panel']), x['group_id']))
    tasks = [m for g in groups for m in g['members']]
    assert len({m['job_id'] for m in tasks}) == len(tasks)
    counts = dict(Counter(g['panel'] for g in groups))
    write(path, dict(n_tasks=len(tasks), n_groups=len(groups), panel_group_counts=counts,
                     layer_indices_zero_based=LAYERS, availability=availability, shortages=shortages,
                     selection='Old S accounting and metadata only; GT-defined diagnostic strata, no new layer signal',
                     groups=groups))
    print('Frozen', len(tasks), 'tasks /', len(groups), 'image groups', counts, 'shortages', shortages, flush=True)


def initialize(load_encoder=True):
    global ENCODER, BASES, LIMITER
    import torch
    from threadpoolctl import threadpool_limits
    os.nice(19)
    c.initialize()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    LIMITER = threadpool_limits(limits=2)
    if not load_encoder:
        return
    spec = importlib.util.spec_from_file_location('w70_frozen_encoder', ENCODER_SOURCE)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ENCODER = module.TimmDINOv3(str(MODEL)).eval().requires_grad_(False)
    saved = torch.load(BASIS, map_location='cpu', weights_only=True)
    assert saved['size'] == 1024
    BASES = {layer: saved['bases'][layer].float() for layer in LAYERS}
    assert all(b.shape == (1024, 500) and torch.isfinite(b).all() for b in BASES.values())


def encode_role(b, role, cache):
    import torch
    from PIL import Image
    row = b['row']; path = c.ASSETS/row[role+'_path']; crop = row.get(role+'_crop')
    key = (str(path), json.dumps(crop))
    start = time.perf_counter()
    fresh = key not in cache
    if fresh:
        with Image.open(path) as image:
            image = image.convert('RGB')
            if crop is not None:
                image = image.crop(tuple(crop))
            value = c.rc.host.transform(image)[None]
        with torch.inference_mode():
            maps = ENCODER.get_intermediate_layers(value, n=list(LAYERS), reshape=True)
        cache[key] = {layer: fmap[0].permute(1, 2, 0).reshape(4096, 1024).contiguous()
                      for layer, fmap in zip(LAYERS, maps)}
    maps = cache[key]
    raw = c.rc.raw(b['fit']['reference_raw' if role == 'reference' else 'query_raw_identity'], {})
    relative = float((maps[23]-raw).norm()/raw.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(maps[23], raw, dim=1).min())
    assert relative < 1e-3 and cosine > .9999, (role, relative, cosine)
    return maps, dict(role=role, fresh_encoder_call=fresh, seconds=time.perf_counter()-start,
                      last_layer_relative_l2=relative, last_layer_min_cosine=cosine)


def layer_cue(rraw, qraw, basis, sets):
    import torch
    import torch.nn.functional as F
    unit = lambda x: F.normalize(x.float(), dim=-1)
    r, q = unit(rraw), unit(qraw)
    r, q = unit(r-(r@basis)@basis.T), unit(q-(q@basis)@basis.T)
    mu = .5*(r.mean(0)+q.mean(0)); r, q = unit(r-mu), unit(q-mu)
    if not sets['RI'].any() or not sets['RO'].any():
        return None, None
    perm = np.random.default_rng(NULL_SEED).permutation(4096)
    def compare(inner, outer):
        a, b = r[torch.as_tensor(inner)], r[torch.as_tensor(outer)]
        chunks = []
        for start in range(0, 4096, 256):
            qq = q[start:start+256]
            chunks.append((qq@a.T).max(1).values-(qq@b.T).max(1).values)
        return torch.cat(chunks).numpy().reshape(64, 64)
    real = compare(np.flatnonzero(sets['RI']), np.flatnonzero(sets['RO']))
    null = compare(perm[sets['RI']], perm[sets['RO']]) if basis is BASES[19] else None
    return real, null


def one_group(group):
    import torch
    import w42_ordered_cut as capture
    import w66_regions as regions
    import w66_agent as rim
    started = time.perf_counter(); cache = {}; result = []
    for item in group['members']:
        b = item['binding']; old = item['baseline']; t = time.perf_counter()
        with c.rc.pc.inference_only(b), torch.inference_mode():
            field, pre, _, _, rho = capture.capture(b)
            assert sha(old['path']) == old['sha256']
            with np.load(old['path'], allow_pickle=False) as z:
                assert np.array_equal(c.pack(pre), z[old['key']]), ('SH identity failure', item['job_id'])
            b0 = regions.pool(field)
            K, N, IN, OUTmask, F = regions.masks(b0)
            sets = rim.reference_sets(b)
            r, rtrace = encode_role(b, 'reference', cache)
            q, qtrace = encode_role(b, 'query', cache)
            arrays = dict(B0=b0, K=K, N=N, IN=IN, OUT=OUTmask, far=F, SH=c.pack(pre))
            missing = not sets['RI'].any() or not sets['RO'].any()
            if not missing:
                for layer in LAYERS:
                    signal, null = layer_cue(r[layer], q[layer], BASES[layer], sets)
                    arrays[f'S{layer}'] = signal
                    if null is not None:
                        arrays['P19'] = null
                assert all(np.isfinite(arrays[name]).all() for name in ('B0', 'S19', 'S20', 'S23', 'P19'))
            destination = OUT/'signals'/(item['job_id']+'.npz')
            c.save_npz(destination, **arrays)
        result.append(dict(**{k: item[k] for k in ('job_id', 'index', 'dataset', 'photo', 'group', 'repeat')},
                           panel=group['panel'], unit=group['group_id'], rho=rho, missing_reference_rim=missing,
                           reference_rim_counts={s: int(sets[s].sum()) for s in ('RI', 'RO')},
                           baseline_pixel_identity=True, query_GT_file_reads=0, encoder_trace=[rtrace, qtrace],
                           signal_path=str(destination), signal_sha256=sha(destination), seconds=time.perf_counter()-t))
    return result, time.perf_counter()-started


def sources():
    return [Path(__file__), PLAN, ENCODER_SOURCE, BASIS, MODEL/'config.json', MODEL/'model.safetensors',
            ROOT/'scripts/x1012/w42_ordered_cut.py', ROOT/'scripts/x1012/w66_regions.py',
            ROOT/'scripts/x1012/w66_agent.py', ROOT/'scripts/x1012/w23_simple_pipeline.py',
            ROOT/'scripts/x1012/r5_candidate.py', ROOT/'scripts/x1012/plan_common.py',
            ROOT/'scripts/x1013/common.py', ROOT/'scripts/x1012/w09_integration_lambda.py',
            ROOT/'scripts/x1012/w19_simple_head.py']


def produce():
    import psutil
    manifest = json.loads((OUT/'manifest.json').read_text())
    lock = (OUT/'run.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = dict(manifest_sha256=sha(OUT/'manifest.json'), n_tasks=manifest['n_tasks'], layers=list(LAYERS),
                  threads=2, processes=1, mps=False, null_seed=NULL_SEED,
                  sources_sha256={str(p): sha(p) for p in sources()})
    path = OUT/'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Frozen inputs changed'
    else:
        write(path, config)
    ledger = OUT/'signals.jsonl'; records = list(read_jsonl(ledger, 'job_id').values()) if ledger.exists() else []
    done = {r['job_id'] for r in records}
    write(OUT/'execution.json', dict(pid=os.getpid(), status='running', started_utc=utc()))
    initialize()
    proc = psutil.Process(); stamp = time.monotonic(); ct = proc.cpu_times(); cpu_before = ct.user+ct.system
    print('Fixed information probe', manifest['n_tasks'], 'tasks; CPU2; MPS0', flush=True)
    try:
        with ledger.open('a') as stream, (OUT/'resources.jsonl').open('a') as resources:
            for group in manifest['groups']:
                ids = {m['job_id'] for m in group['members']}
                if ids <= done:
                    continue
                assert not ids & done, 'Partial group: retain evidence and investigate without mixing outputs'
                out, elapsed = one_group(group)
                for record in out:
                    stream.write(json.dumps(record)+'\n'); stream.flush(); records.append(record); done.add(record['job_id'])
                now = time.monotonic(); ct = proc.cpu_times(); cpu_now = ct.user+ct.system
                cpu = 100*(cpu_now-cpu_before)/max(now-stamp, 1e-9)/psutil.cpu_count()
                sample = dict(n=len(records), utc=utc(), cpu_machine_percent=cpu, rss_gib=proc.memory_info().rss/2**30, mps=0)
                resources.write(json.dumps(sample)+'\n'); resources.flush()
                assert cpu <= 50, f'CPU cap exceeded: {cpu}'
                stamp, cpu_before = now, cpu_now
                print(len(records), '/', manifest['n_tasks'], group['group_id'], f'{elapsed:.1f}s', f'CPU {cpu:.1f}%', flush=True)
        assert len(records) == manifest['n_tasks'] and len(done) == len(records)
        assert all(sha(r['signal_path']) == r['signal_sha256'] for r in records)
        assert all(sha(p) == digest for p, digest in config['sources_sha256'].items())
        write(OUT/'complete.json', dict(n_tasks=len(records), signals_sha256=sha(ledger), config_sha256=sha(path),
                                       encoder_calls=sum(t['fresh_encoder_call'] for r in records for t in r['encoder_trace']),
                                       query_GT_file_reads=0, model_parameter_updates=0, candidate_segmentation_masks=0, utc=utc()))
    except BaseException as exc:
        write(OUT/'failure.json', dict(error=type(exc).__name__, message=str(exc), n=len(records), utc=utc()))
        raise
    finally:
        write(OUT/'execution.json', dict(pid=os.getpid(), status='complete' if (OUT/'complete.json').exists() else 'failed_or_partial', utc=utc()))
        lock.close()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    elif sys.argv[1] == 'produce': produce()
