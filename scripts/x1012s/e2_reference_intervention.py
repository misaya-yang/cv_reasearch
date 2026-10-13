"""E2: cache-only reference-content/mass/centre interventions, IND0 endpoint.

prepare freezes all B1 common-K3 queries and an order-only 20/domain pilot.
produce --cohort pilot is resumable, never reads query labels or old scores.
score opens query GT only after every requested natural prediction is sealed.
No encoder, FoRIS replay, clustering rerun, or mutation of B1/P0 is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'scripts')]

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import b1_kshot as b1
import plan_report as pr
from raw_feature_cache import tensor_hash

lab, pc = b1.lab, b1.pc
B1 = lab.ASSETS / 'a/x1012_B1'
P0 = lab.ASSETS / 'a/x1012_P0/fields'
OUT = lab.ASSETS / 'a/x1012s_E2'
PLAN = ROOT / 'docs/research/SINGLE_REFERENCE_SOLVE_PLAN_20261012.md'
TEMPERATURE = .05
ARMS = ('FG', 'BG', 'FGBG', 'CENTER', 'MASS', 'FGBG_NATIVE_MASS')
BASE = 'K1/IND0'
K3 = 'K3/IND0'
TOL = 1e-5


def setup(device='cpu'):
    pc.setup(device)
    torch.set_num_threads(2)


def check_binding(b):
    refs, row = b['references'][:3], b['row']
    assert 3 in b['available_k'] and len(refs) == 3
    photos = [r['photo'] for r in refs]
    assert len(set(photos)) == 3
    assert str((lab.ASSETS / row['query_path']).resolve()) not in photos
    assert all(r['dataset'] == b['dataset'] and str(r['fold']) == str(row['fold'])
               and str(r['class_id']) == str(row['global_class_id']) for r in refs)
    for key in ('payload_path', 'payload_sha256', 'input_sha256', 'tensor_sha256'):
        assert refs[0]['raw'][key] == b['fit']['reference_raw'][key], ('R1 binding', key)
    for key in ('mask_path', 'mask_hash', 'crop'):
        if key in refs[0]['view'] or 'reference_' + key in row:
            assert refs[0]['view'].get(key) == row.get('reference_' + key), ('R1 view', key)


def prepare():
    assert PLAN.is_file(), 'Root execution plan must exist before freezing E2'
    if (OUT / 'config.json').exists():
        check_config()
        print({'state': 'ALREADY_PREPARED', 'output': str(OUT)}, flush=True)
        return
    assert not OUT.exists() or not any(OUT.iterdir()), 'Preserve existing E2 directory'
    seal, cfg = pc.read(B1 / 'sealed.json'), pc.read(B1 / 'config.json')
    assert seal['state'] == 'ALL_KSHOT_NATURAL_PREDICTIONS_SEALED'
    assert pc.sha(B1 / 'manifest.json') == seal['manifest_sha256']
    assert pc.sha(B1 / 'config.json') == seal['config_sha256']
    assert cfg['device'] == 'mps'
    rows = [b for b in pc.read(B1 / 'manifest.json') if 3 in b['available_k']]
    receipts = {r['index']: r for r in seal['records']}
    p0seal = pc.read(P0 / 'sealed.json')
    assert p0seal['state'] == 'STANDARD_FIELDS_SEALED'
    assert pc.sha(P0 / 'manifest.json') == p0seal['manifest_sha256']
    p0records = {r['index']: r for r in p0seal['records']}
    label_sources = {}
    for b in pc.read(P0 / 'manifest.json'):
        label_sources.setdefault(b['fit']['query_raw_identity']['tensor_sha256'],
                                 p0records[pc.key(b)])
    pilot, counts = [], Counter()
    sources = {}
    for b in rows:
        check_binding(b)
        key = pc.key(b)
        sources[key] = {'B1': receipts[key],
                        'labels': label_sources[b['fit']['query_raw_identity']['tensor_sha256']]}
        if counts[b['dataset']] < 20:
            pilot.append(key)
            counts[b['dataset']] += 1
    assert len(rows) == 4585 and len(pilot) == 200
    files = [Path(__file__), Path(b1.__file__), Path(lab.__file__), Path(pc.__file__), Path(pr.__file__),
             ROOT / 'scripts/raw_feature_cache.py', lab.BASIS]
    # Existing frozen source identity matters because the cached K1 is the control.
    for path in files[1:]:
        if str(path) in cfg['source_sha256']:
            assert pc.sha(path) == cfg['source_sha256'][str(path)]
    source_files = [B1 / n for n in ('manifest.json', 'config.json', 'sealed.json')]
    source_files += [P0 / n for n in ('manifest.json', 'config.json', 'sealed.json')]
    OUT.mkdir(parents=True, exist_ok=True)
    pc.write(OUT / 'manifest.json', rows)
    pc.write(OUT / 'pilot_keys.json', pilot)
    pc.write(OUT / 'source_receipts.json', sources)
    pc.write(OUT / 'config.json', {
        'state': 'FROZEN', 'task': 'E2', 'n_full': len(rows), 'n_pilot': len(pilot),
        'pilot_selection': 'first 20 B1 common-K3 records per dataset, original manifest order; no score selection',
        'pilot_counts': dict(counts), 'device': 'mps', 'cpu_threads': 2, 'query_block': 256,
        'temperature': TEMPERATURE, 'endpoint': 'IND0 only; fixed cached query K64 region labels',
        'readout': 'FP32 bilinear1024; minmax; strict >0.5',
        'arms': list(ARMS), 'role_mass': 'soft support sum(y1), sum(1-y1); union role weights renormalized to these totals',
        'centre': 'mu1=.5*(mean(R1)+mean(Q)); CENTER uses mu3 with R1 content and mass unchanged',
        'novelty': 'not used; no INDN causal attribution',
        'reproduction_max_abs_tolerance': TOL,
        'query_GT_in_inference': False, 'new_encoder_calls': 0,
        'plan_path': str(PLAN), 'plan_sha256_at_freeze': pc.sha(PLAN),
        'source_sha256': {str(p): pc.sha(p) for p in files + source_files},
        'baseline_report_path': str(B1 / 'common_K3/report.json'),
        'baseline_report_sha256': pc.sha(B1 / 'common_K3/report.json'),
        'manifest_sha256': pc.sha(OUT / 'manifest.json'),
        'pilot_keys_sha256': pc.sha(OUT / 'pilot_keys.json'),
        'source_receipts_sha256': pc.sha(OUT / 'source_receipts.json')})
    print({'state': 'PREPARED', 'pilot_n': len(pilot), 'full_n': len(rows), 'counts': dict(counts)}, flush=True)


def check_config():
    cfg = pc.read(OUT / 'config.json')
    for path, digest in cfg['source_sha256'].items():
        assert pc.sha(path) == digest, 'E2 frozen dependency changed: ' + path
    for name in ('manifest', 'pilot_keys', 'source_receipts'):
        assert pc.sha(OUT / (name + '.json')) == cfg[name + '_sha256']
    return cfg


def selected(cohort):
    rows = pc.read(OUT / 'manifest.json')
    if cohort == 'pilot':
        keys = set(pc.read(OUT / 'pilot_keys.json'))
        rows = [b for b in rows if pc.key(b) in keys]
    return rows


def coverage(ref):
    view = ref['view']
    with Image.open(view['mask_path']) as image:
        array = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
    assert tensor_hash(array) == view['mask_hash'], 'Reference annotation binding changed'
    full = F.interpolate(torch.from_numpy(array)[None, None].float(), size=(1024, 1024), mode='nearest')[0, 0]
    return lab.coverage64(full).flatten().to(lab.DEV)


def role_log_mass(reference, query, y, mu):
    rc, qc = lab.unit(reference - mu), lab.unit(query - mu)
    fg, bg, post = [], [], []
    for offset in range(0, len(query), 256):
        sim = qc[offset:offset + 256] @ rc.T
        maximum = sim.max(1).values
        kernel = torch.exp((sim - maximum[:, None]) / TEMPERATURE)
        f, b = kernel @ y, kernel @ (1 - y)
        fg.append(f.clamp_min(1e-30).log() + maximum / TEMPERATURE)
        bg.append(b.clamp_min(1e-30).log() + maximum / TEMPERATURE)
        # Exact B1 K1 arithmetic for control reproduction.
        post.append(f / kernel.sum(1).clamp_min(1e-30))
    return torch.cat(fg), torch.cat(bg), torch.cat(post)


def rescale(log_values, source_mass, target_mass):
    if target_mass <= 0:
        return torch.full_like(log_values, -torch.inf)
    assert source_mass > 0, 'Cannot create a missing reference role by reweighting'
    return log_values + np.log(target_mass / source_mass)


def posterior(log_f, log_b):
    return torch.sigmoid(log_f - log_b)


def intervention_fields(roles, centre_role, masses):
    """Only support statistics enter this function, never query annotations."""
    f1, b1log, c1 = roles[0]
    fs = torch.logsumexp(torch.stack([x[0] for x in roles]), dim=0)
    bs = torch.logsumexp(torch.stack([x[1] for x in roles]), dim=0)
    mf1, mb1 = masses[0]
    mfs, mbs = np.sum(masses, axis=0)
    fp, bp = rescale(fs, mfs, mf1), rescale(bs, mbs, mb1)
    # Empty legal R1 role: fixed-mass interventions retain that absence exactly.
    f1fixed = rescale(f1, mf1, mf1)
    b1fixed = rescale(b1log, mb1, mb1)
    out = {'FG': posterior(fp, b1fixed), 'BG': posterior(f1fixed, bp),
           'FGBG': posterior(fp, bp), 'CENTER': centre_role[2],
           'FGBG_NATIVE_MASS': posterior(fs, bs)}
    eligible = mf1 > 0 and mb1 > 0
    if eligible:
        out['MASS'] = posterior(rescale(f1, mf1, mfs / 3), rescale(b1log, mb1, mbs / 3))
        alpha_log = np.log((mfs / mbs) / (mf1 / mb1))
        # Mass changes alone are a monotone token-odds transformation; after
        # region averaging this need not preserve the rendered ranking.
        bridged = torch.sigmoid((fp - bp) + alpha_log)
        bridge_error = float((bridged - out['FGBG_NATIVE_MASS']).abs().max())
    else:
        out['MASS'] = c1.clone()  # explicitly ineligible, never pretend to add absent content
        bridge_error = None
    repeated_f = torch.logsumexp(torch.stack([f1, f1, f1]), dim=0) - np.log(3)
    repeated_b = torch.logsumexp(torch.stack([b1log, b1log, b1log]), dim=0) - np.log(3)
    repeated = posterior(rescale(repeated_f, mf1, mf1), rescale(repeated_b, mb1, mb1))
    if mf1 == 0:
        repeated = torch.zeros_like(c1)
    elif mb1 == 0:
        repeated = torch.ones_like(c1)
    repeat_error = float((repeated - c1).abs().max())
    assert repeat_error <= TOL, ('Repeat-R1 invariant', repeat_error)
    if bridge_error is not None:
        assert bridge_error <= TOL, ('Fixed/native mass bridge', bridge_error)
    assert all(torch.isfinite(v).all() for v in out.values())
    return out, {'mass_only_eligible': bool(eligible), 'repeat_R1_max_abs': repeat_error,
                 'mass_bridge_max_abs': bridge_error,
                 'R1_role_mass': [float(mf1), float(mb1)],
                 'pooled_role_mass': [float(mfs), float(mbs)],
                 'fixed_role_weight_sums': [float(mf1), float(mb1)]}


def average_regions(values, labels):
    oh = F.one_hot(labels, num_classes=64).to(values.dtype)
    return oh @ ((oh.T @ values) / oh.sum(0).clamp_min(1))


def compute(b, sources):
    tick, counters = time.monotonic(), {}
    refs = b['references'][:3]
    check_binding(b)
    with b1.legal_inference(b, [r['view']['mask_path'] for r in refs]):
        labels_source = sources['labels']
        assert pc.sha(labels_source['path']) == labels_source['sha256']
        with np.load(labels_source['path'], allow_pickle=False) as cache:
            labels = torch.from_numpy(cache['regions64'].copy().astype(np.int64)).to(lab.DEV)
        assert labels.shape == (4096,) and labels.min() >= 0 and labels.max() < 64
        old = sources['B1']
        assert pc.sha(old['field_path']) == old['field_sha256']
        assert pc.sha(old['prediction_path']) == old['prediction_sha256']
        with np.load(old['field_path'], allow_pickle=False) as z:
            old_field = z[BASE].copy()
        with np.load(old['prediction_path'], allow_pickle=False) as z:
            old_mask = pc.unpack(z[BASE])
        query = lab.debias(b1.read_raw(b['fit']['query_raw_identity'], counters).to(lab.DEV))
        references = [lab.debias(b1.read_raw(r['raw'], counters).to(lab.DEV)) for r in refs]
        ys = [coverage(r) for r in refs]
        masses = [(float(y.sum()), float((1 - y).sum())) for y in ys]
        mu1 = .5 * (references[0].mean(0) + query.mean(0))
        mu3 = .5 * (torch.stack([r.mean(0) for r in references]).mean(0) + query.mean(0))
        roles = [role_log_mass(r, query, y, mu1) for r, y in zip(references, ys)]
        centre = role_log_mass(references[0], query, ys[0], mu3)
        field1 = average_regions(roles[0][2], labels).cpu().numpy()
        reproduction_error = float(np.max(np.abs(field1 - old_field)))
        reproduction_pixels = int(np.count_nonzero(pc.readout(field1) != old_mask))
        assert reproduction_error <= TOL, ('K1 field reproduction', pc.key(b), reproduction_error)
        # Pixel-exact equality is expected from the same frozen arithmetic/device.
        assert reproduction_pixels == 0, ('K1 mask reproduction', pc.key(b), reproduction_pixels)
        token_fields, diagnostics = intervention_fields(roles, centre, masses)
        payload = {}
        for arm in ARMS:
            field = average_regions(token_fields[arm], labels).cpu().numpy().astype(np.float32)
            payload['field/' + arm] = field
            payload['mask/' + arm] = pc.pack(pc.readout(field))
        diagnostics.update(K1_field_max_abs=reproduction_error, K1_mask_xor_pixels=reproduction_pixels,
                           centre_L2=float((mu3 - mu1).norm()),
                           reference_role_masses=[[float(x) for x in m] for m in masses],
                           source_K64_sha256=labels_source['sha256'], query_GT_reads=0,
                           new_encoder_calls=0)
    receipt = {'index': pc.key(b), 'dataset': b['dataset'], 'seconds': time.monotonic() - tick,
               'diagnostics': diagnostics, **counters}
    payload['receipt_json'] = np.frombuffer(json.dumps(receipt, allow_nan=False).encode(), dtype=np.uint8)
    return payload, receipt


def artifact_path(key):
    return OUT / 'data' / (key + '.npz')


def receipt_path(key):
    return OUT / 'records' / (key + '.json')


def valid_receipt(key):
    path, rec = artifact_path(key), receipt_path(key)
    if not path.exists():
        assert not rec.exists(), 'Receipt without E2 prediction artifact'
        return None
    if rec.exists():
        r = pc.read(rec)
        assert pc.sha(path) == r['artifact_sha256'], 'E2 artifact changed'
        return r
    # Recover an atomic artifact written just before an interrupted receipt write.
    with np.load(path, allow_pickle=False) as z:
        r = json.loads(z['receipt_json'].tobytes())
    assert r['index'] == key
    r.update(artifact_path=str(path), artifact_sha256=pc.sha(path))
    pc.write(rec, r)
    return r


def produce(cohort):
    cfg = check_config()
    assert torch.backends.mps.is_available(), 'No mixed-device fallback'
    setup('mps')
    rows = selected(cohort)
    sources = pc.read(OUT / 'source_receipts.json')
    for name in ('data', 'records', cohort):
        (OUT / name).mkdir(exist_ok=True)
    start, receipts = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        key = pc.key(b)
        r = valid_receipt(key)
        if r is None:
            result, r = compute(b, sources[key])
            path = artifact_path(key)
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, **result)
            temporary.replace(path)
            r.update(artifact_path=str(path), artifact_sha256=pc.sha(path))
            pc.write(receipt_path(key), r)
            del result
            lab.free()
        receipts.append(r)
        if i <= 3 or i % 20 == 0:
            print({'task': 'E2', 'cohort': cohort, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - start, 2),
                   'K1_field_max_abs': r['diagnostics']['K1_field_max_abs'],
                   'K1_mask_xor_pixels': r['diagnostics']['K1_mask_xor_pixels']}, flush=True)
    pc.write(OUT / cohort / 'sealed.json', {
        'state': 'ALL_E2_NATURAL_PREDICTIONS_SEALED', 'cohort': cohort, 'n': len(rows),
        'keys': [pc.key(b) for b in rows], 'config_sha256': pc.sha(OUT / 'config.json'),
        'manifest_sha256': cfg['manifest_sha256'], 'records': receipts,
        'invocation_wall_seconds': time.monotonic() - start,
        'episode_seconds_sum': sum(r['seconds'] for r in receipts),
        'logical_payload_bytes': sum(r['logical_payload_bytes'] for r in receipts),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_encoder_calls': 0, 'query_GT_reads': 0, 'baseline_replays': 0})
    print({'state': 'SEALED', 'cohort': cohort, 'n': len(rows),
           'seconds': round(time.monotonic() - start, 2)}, flush=True)


def worker_init():
    setup('cpu')


def score_one(job):
    b, r, baseline, sources = job
    start = time.monotonic()
    gt, valid = pc.truth(b)
    with np.load(sources['B1']['prediction_path'], allow_pickle=False) as z:
        base_mask = pc.unpack(z[BASE]) & valid
    record = {k: baseline[k] for k in ('index', 'episode_id', 'dataset', 'fold', 'class_id', 'query_photo')}
    record['metrics'] = {a: baseline['metrics'][a] for a in
                         ('FoRIS', BASE, K3, BASE + '/oracle_threshold', K3 + '/oracle_threshold')}
    transitions, thresholds = {}, {}
    with np.load(r['artifact_path'], allow_pickle=False) as z:
        for arm in ARMS:
            pred = pc.unpack(z['mask/' + arm])
            record['metrics'][arm] = pr.metric(pred, gt, valid)
            oracle, threshold = b1.threshold_oracle(z['field/' + arm], gt, valid)
            record['metrics'][arm + '/oracle_threshold'] = pr.metric(oracle, gt, valid)
            thresholds[arm] = threshold
            added, removed = pred & ~base_mask & valid, base_mask & ~pred & valid
            transitions[arm] = {'added_TP': int((added & gt).sum()), 'added_FP': int((added & ~gt).sum()),
                                'removed_TP': int((removed & gt).sum()), 'removed_FP': int((removed & ~gt).sum())}
    record.update(transitions_vs_K1=transitions, oracle_thresholds=thresholds,
                  diagnostics=r['diagnostics'], scoring_seconds=time.monotonic() - start)
    return record


def summarize(records, bootstrap=1000):
    from scipy.sparse import csr_matrix
    rng, output = np.random.default_rng(20261012), {}
    names = (BASE, K3, *ARMS)
    contrasts = {'FG_content': ('FG', BASE), 'BG_content': ('BG', BASE),
                 'FGBG_content': ('FGBG', BASE), 'centre_R1': ('CENTER', BASE),
                 'mass_R1': ('MASS', BASE), 'mass_with_pool': ('FGBG_NATIVE_MASS', 'FGBG'),
                 'centre_with_pool': (K3, 'FGBG_NATIVE_MASS')}
    for dataset in lab.ALL_DOMAINS:
        rs = [r for r in records if r['dataset'] == dataset]
        if not rs:
            continue
        photos = sorted({r['query_photo'] for r in rs})
        slots = sorted({(str(r['fold']), str(r['class_id'])) for r in rs})
        pi, si = {x: i for i, x in enumerate(photos)}, {x: i for i, x in enumerate(slots)}
        p = [pi[r['query_photo']] for r in rs]
        s = [si[(str(r['fold']), str(r['class_id']))] for r in rs]
        shape = (len(photos), len(slots))
        presence = csr_matrix((np.ones(len(rs)), (p, s)), shape=shape)
        weights = np.stack([np.bincount(rng.integers(0, len(photos), len(photos)), minlength=len(photos))
                            for _ in range(bootstrap)])
        represented = (weights @ presence) > 0
        modes = {}
        for suffix, mode in (('', 'natural'), ('/oracle_threshold', 'strict_oracle')):
            table, boot = {}, {}
            for arm in names:
                metrics = [r['metrics'][arm + suffix] for r in rs]
                inter = csr_matrix(([m['I'] for m in metrics], (p, s)), shape=shape)
                union = csr_matrix(([m['U'] for m in metrics], (p, s)), shape=shape)
                vals = 100 * (weights @ inter) / np.maximum(weights @ union, 1)
                boot[arm] = (vals * represented).sum(1) / represented.sum(1)
                point = float(np.mean(100 * np.asarray(inter.sum(0)) / np.maximum(np.asarray(union.sum(0)), 1)))
                table[arm] = {'class_miou': point, 'episode_mean_iou': float(np.mean([m['IoU'] for m in metrics])),
                              'FN_GT': sum(m['FN'] for m in metrics) / max(sum(m['GT'] for m in metrics), 1),
                              'FP_GT': sum(m['FP'] for m in metrics) / max(sum(m['GT'] for m in metrics), 1)}
                if suffix == '' and arm in ARMS:
                    table[arm]['transitions_vs_K1'] = {k: sum(r['transitions_vs_K1'][arm][k] for r in rs)
                                                     for k in ('added_TP', 'added_FP', 'removed_TP', 'removed_FP')}
            effects = {}
            for label, (arm, baseline) in contrasts.items():
                effects[label] = {'contrast': [arm, baseline],
                                  'delta_pp': table[arm]['class_miou'] - table[baseline]['class_miou'],
                                  'paired_query_photo_bootstrap95_pp': np.percentile(boot[arm] - boot[baseline], [2.5, 97.5]).tolist()}
            effects['FG_BG_interaction'] = {
                'contrast': 'FGBG-FG-BG+K1',
                'delta_pp': table['FGBG']['class_miou'] - table['FG']['class_miou'] - table['BG']['class_miou'] + table[BASE]['class_miou'],
                'paired_query_photo_bootstrap95_pp': np.percentile(boot['FGBG'] - boot['FG'] - boot['BG'] + boot[BASE], [2.5, 97.5]).tolist()}
            modes[mode] = {'arms': table, 'effects': effects}
        output[dataset] = {'n': len(rs), 'photo_n': len(photos), 'class_slot_n': len(slots),
                           'mass_only_eligible_n': sum(r['diagnostics']['mass_only_eligible'] for r in rs), **modes}
    return output


def score(cohort, workers):
    cfg = check_config()
    setup('cpu')
    seal_path = OUT / cohort / 'sealed.json'
    seal = pc.read(seal_path)
    assert seal['state'] == 'ALL_E2_NATURAL_PREDICTIONS_SEALED'
    assert seal['config_sha256'] == pc.sha(OUT / 'config.json')
    rows, receipts = selected(cohort), {r['index']: r for r in seal['records']}
    assert [pc.key(b) for b in rows] == seal['keys']
    sources = pc.read(OUT / 'source_receipts.json')
    for b in rows:
        key = pc.key(b)
        assert pc.sha(receipts[key]['artifact_path']) == receipts[key]['artifact_sha256']
        old = sources[key]['B1']
        assert pc.sha(old['prediction_path']) == old['prediction_sha256']
    # Every natural artifact is verified before opening old GT-derived scores or query GT.
    assert pc.sha(cfg['baseline_report_path']) == cfg['baseline_report_sha256']
    baseline_report = pc.read(cfg['baseline_report_path'])
    baseline = {r['index']: r for r in baseline_report['records']}
    assert set(baseline) == {pc.key(b) for b in pc.read(OUT / 'manifest.json')}
    start = time.monotonic()
    score_dir = OUT / cohort / 'scores'
    score_dir.mkdir(exist_ok=True)
    records, pending = [], []
    for b in rows:
        key = pc.key(b)
        path = score_dir / (key + '.json')
        if path.exists():
            records.append(pc.read(path))
        else:
            pending.append((b, receipts[key], baseline[key], sources[key]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as pool:
        for record in pool.map(score_one, pending, chunksize=1):
            pc.write(score_dir / (record['index'] + '.json'), record)
            records.append(record)
            if len(records) <= 3 or len(records) % 20 == 0:
                print({'task': 'E2-score', 'completed': len(records), 'n': len(rows),
                       'seconds': round(time.monotonic() - start, 2)}, flush=True)
    lookup = {r['index']: r for r in records}
    records = [lookup[pc.key(b)] for b in rows]
    summary = summarize(records)
    seconds = time.monotonic() - start
    controls = {'K1_max_abs': max(r['diagnostics']['K1_field_max_abs'] for r in records),
                'K1_total_mask_xor': sum(r['diagnostics']['K1_mask_xor_pixels'] for r in records),
                'repeat_R1_max_abs': max(r['diagnostics']['repeat_R1_max_abs'] for r in records),
                'mass_bridge_max_abs': max((r['diagnostics']['mass_bridge_max_abs'] or 0) for r in records)}
    cost = {'production_episode_seconds': seal['episode_seconds_sum'], 'scoring_wall_seconds': seconds,
            'seconds_per_case_production': seal['episode_seconds_sum'] / len(rows),
            'projected_full_production_minutes': seal['episode_seconds_sum'] / len(rows) * cfg['n_full'] / 60,
            'new_encoder_calls': 0, 'baseline_replays': 0,
            'logical_payload_GiB': seal['logical_payload_bytes'] / 2**30, 'peak_RSS_bytes': seal['peak_RSS_bytes']}
    result = {'state': 'COMPLETE', 'task': 'E2', 'cohort': cohort, 'n': len(rows), 'summary': summary,
              'controls': controls, 'cost': cost, 'records': records, 'seal_sha256': pc.sha(seal_path),
              'interpretation': 'Reference intervention diagnostic, not a deployable single-reference method; pilot is not domain-level evidence.'}
    pc.write(OUT / cohort / 'report.json', result)
    lines = ['# E2 参考干预诊断', '', f'状态：{cohort}，{len(rows)} 条固定记录。每域按 B1 共同 K3 原顺序前20条；未按结果筛样本。',
             '', '仅为机制诊断，不是域级成果，不把 K3 交付为单参考方法。自然预测先封存，GT只在后续评分和严格阈值诊断使用。',
             '固定 K1 中心及 FG/BG 总软权重质量，固定现有 query K64；主端点 IND0，CLI1024观察(fold,class)槽池化I/U后等权。',
             '', '|域|n|K1 natural|K3 natural|FG Δ|BG Δ|FG+BG Δ|中心单改 Δ|质量单改 Δ|池化后质量 Δ|池化后中心 Δ|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, s in summary.items():
        a, e = s['natural']['arms'], s['natural']['effects']
        vals = [e[k]['delta_pp'] for k in ('FG_content', 'BG_content', 'FGBG_content', 'centre_R1', 'mass_R1', 'mass_with_pool', 'centre_with_pool')]
        lines.append(f"|{d}|{s['n']}|{a[BASE]['class_miou']:.3f}|{a[K3]['class_miou']:.3f}|" + '|'.join(f'{v:+.3f}' for v in vals) + '|')
    lines += ['', '|域|K1 strict oracle|K3 strict oracle|FG Δ|BG Δ|FG+BG Δ|中心单改 Δ|质量单改 Δ|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d, s in summary.items():
        a, e = s['strict_oracle']['arms'], s['strict_oracle']['effects']
        vals = [e[k]['delta_pp'] for k in ('FG_content', 'BG_content', 'FGBG_content', 'centre_R1', 'mass_R1')]
        lines.append(f"|{d}|{a[BASE]['class_miou']:.3f}|{a[K3]['class_miou']:.3f}|" + '|'.join(f'{v:+.3f}' for v in vals) + '|')
    lines += ['', 'FG/BG联合效果允许交互；最后两列是固定路径的条件差，不能解释成唯一、顺序无关的原因贡献。',
              '质量变化在token层是单调odds映射，但与后续区域均值不交换，故最终strict oracle可以变化。',
              f'控制核对：`{json.dumps(controls)}`。', f'成本：`{json.dumps(cost)}`。',
              f'逐例I/U、TP/FP纠错收支、照片配对bootstrap区间与资格标记：`{OUT / cohort / "report.json"}`。']
    report_path = ROOT / 'docs/research/x1012s_tasks/E2.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n')
    print({'state': 'COMPLETE', 'cohort': cohort, 'n': len(rows), 'controls': controls, 'cost': cost,
           'report': str(report_path)}, flush=True)


def self_test():
    setup('cpu')
    # Synthetic support-only algebra check, no research cache or GT is opened.
    generator = torch.Generator().manual_seed(5)
    r = [lab.unit(torch.randn(12, 8, generator=generator)) for _ in range(3)]
    q = lab.unit(torch.randn(16, 8, generator=generator))
    ys = [torch.rand(12, generator=generator) for _ in range(3)]
    mu = .5 * (r[0].mean(0) + q.mean(0))
    roles = [role_log_mass(a, q, y, mu) for a, y in zip(r, ys)]
    masses = [(float(y.sum()), float((1-y).sum())) for y in ys]
    out, diagnostics = intervention_fields(roles, roles[0], masses)
    rc, qc = lab.unit(torch.cat(r) - mu), lab.unit(q - mu)
    native = torch.softmax((qc @ rc.T) / TEMPERATURE, dim=1) @ torch.cat(ys)
    assert torch.allclose(out['FGBG_NATIVE_MASS'], native, atol=TOL, rtol=0)
    assert diagnostics['repeat_R1_max_abs'] < TOL
    print({'state': 'SYNTHETIC_ALGEBRA_CHECK_PASSED', 'diagnostics': diagnostics}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce', 'score', 'self-test'))
    parser.add_argument('--cohort', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'produce':
        produce(args.cohort)
    elif args.mode == 'score':
        assert 1 <= args.workers <= 2
        score(args.cohort, args.workers)
    else:
        self_test()
