"""Single-reference R1 VP_UN with optional R2 pixel geometry, from O/24 only.

`reproduce` checks ten old natural outputs without reading query annotations.
`freeze` binds a new manifest and parameters; `produce` seals natural predictions.
There is deliberately no scorer here. Run from the repository root, or provide
CV_RESEARCH_ROOT when executing the frozen copy outside the checkout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import types

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.special import expit
from scipy.sparse.linalg import cg
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get('CV_RESEARCH_ROOT', Path.cwd())).resolve()
ASSETS = REPO.parent / 'cv_data'
DEFAULT_OUT = ASSETS / 'a/x1012_R5_candidate'
DEFAULT_MANIFEST = ASSETS / 'a/x1012_R5_validation/manifest.json'
STRENGTHS = tuple(float(x) for x in np.linspace(0, 1, 9))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_runtime(out):
    """Copy only the required runtime; concurrent cleanup cannot replace it."""
    runtime = Path(out) / 'runtime'
    mapping = {name: REPO / 'scripts/x1012' / name for name in
               ('lab.py', 'plan_common.py', 'plan_report.py', 'r2_within_patch.py')}
    mapping['pixel.py'] = ASSETS / 'a/x1012_R2/frozen/4_diagnose.py'
    for relative, source in (
        ('official/models/foris.py', '12_foris.py'),
        ('official/utils/data.py', '13_data.py'),
        ('official/utils/clustering.py', '14_clustering.py'),
        ('official/utils/refinement.py', '15_refinement.py'),
    ):
        mapping[relative] = ASSETS / 'a/x1012_C1/frozen' / source
    lock = runtime / 'runtime.json'
    if not lock.exists():
        runtime.mkdir(parents=True, exist_ok=True)
        entries = []
        for relative, source in mapping.items():
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append({'relative': relative, 'source': str(source), 'sha256': sha(target)})
        for relative in ('official/models/__init__.py', 'official/utils/__init__.py'):
            (runtime / relative).write_text('')
        write(lock, {'repo_root': str(REPO), 'sources': entries,
                     'basis': str(ASSETS / 'native_assets/positional_basis.pt'),
                     'basis_sha256': sha(ASSETS / 'native_assets/positional_basis.pt')})
    config = read(lock)
    assert all(sha(runtime / e['relative']) == e['sha256'] for e in config['sources'])
    assert sha(config['basis']) == config['basis_sha256']
    return runtime


def initialize(out, device='cpu', threads=2, readout_rule='midrange', use_pixel=True):
    global lab, pc, geometry, pixel, host, SETTINGS
    runtime = Path(out) / 'runtime'
    sys.path.insert(0, str(runtime))
    lab = importlib.import_module('lab')
    lab.ROOT, lab.ASSETS = REPO, ASSETS
    lab.BASIS = ASSETS / 'native_assets/positional_basis.pt'
    lab.DEV = torch.device('cpu')
    pc = importlib.import_module('plan_common')
    pc.ROOT, pc.ASSETS = REPO, ASSETS
    sys.path[:0] = [str(runtime), str(runtime / 'official')]
    geometry = importlib.import_module('r2_within_patch')
    pixel = types.ModuleType('r5_pixel')
    # The preserved source derives REPO from __file__; its source file need not
    # remain in the checkout for that path-to-root calculation to be valid.
    pixel.__file__ = str(REPO / 'evidence/local/autonomous_20261010/physical_pixel_evidence/frozen/diagnose.py')
    exec(compile((runtime / 'pixel.py').read_text(), str(runtime / 'pixel.py'), 'exec'), pixel.__dict__)
    from models.foris import FoRIS
    from utils.data import build_transform
    # Bind only the two declared official functions. No FoRIS/model constructor,
    # full prediction, or CRF is available through this object.
    host = types.SimpleNamespace(tau=.6, _tgt_image=None, transform=build_transform(1024))
    host.locate = types.MethodType(FoRIS._locate_candidates, host)
    host.prior = types.MethodType(FoRIS._build_seed_cluster_prior, host)
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    SETTINGS = {'out': str(out), 'device': device, 'threads': threads,
                'readout_rule': readout_rule, 'use_pixel': use_pixel}


def raw(metadata, counters):
    payload = Path(metadata['payload_path'])
    entry = Path(metadata.get('entry_path', payload.parent / 'entry.json'))
    info = read(entry)
    if metadata.get('entry_sha256'):
        assert sha(entry) == metadata['entry_sha256']
    assert payload.name == info['file'] and sha(payload) == info['file_sha256']
    if metadata.get('payload_sha256'):
        assert metadata['payload_sha256'] == info['file_sha256']
    with np.load(payload, allow_pickle=False) as archive:
        value = archive['O/24'].copy()
    feature = info['features']['O/24']
    assert value.dtype == np.float32 and value.shape == (4096, 1024) and np.isfinite(value).all()
    assert array_hash(value) == feature['tensor_sha256']
    counters['raw_requests'] = counters.get('raw_requests', 0) + 1
    counters['logical_payload_bytes'] = counters.get('logical_payload_bytes', 0) + payload.stat().st_size
    return torch.from_numpy(value)


def quality(mask, c, back):
    native = mask.reshape(64, 16, 64, 16).mean(axis=(1, 3)).reshape(-1) > .5
    if not native.any():
        return 0.
    precision, capture = float(np.mean(c[native])), float(np.sum(back[native]))
    return precision * capture / max(precision + capture - precision * capture, 1e-12)


def modulate(c, labels, ind0, indn, sim_max, back_f):
    """Exact old max-J rule: FP16 side fields and FP32 endpoint masks."""
    novelty = sim_max.astype(np.float32)
    novelty = (novelty - novelty.min()) / max(float(novelty.max() - novelty.min()), 1e-6)
    onehot = F.one_hot(labels, 64).float()
    endpoint = {0.: pc.readout(ind0), 1.: pc.readout(indn)}
    values, tokens = [], []
    for strength in STRENGTHS:
        value = torch.from_numpy(c.numpy() * np.power(novelty, strength))
        field = onehot @ ((onehot.T @ value) / onehot.sum(0).clamp_min(1))
        mask = endpoint[strength] if strength in endpoint else pc.readout(field)
        tokens.append(value)
        values.append(quality(mask, c.numpy(), back_f.astype(np.float64)))
    selected = int(np.argmax(values))
    return tokens[selected], {'selected_s': STRENGTHS[selected], 'J_grid': values}


def independent(binding, counters):
    """Recompute the single-reference P0 tensors without any baseline input."""
    r_raw, q_raw = raw(binding['fit']['reference_raw'], counters), raw(binding['fit']['query_raw_identity'], counters)
    r_cpu, q_cpu = lab.debias(r_raw), lab.debias(q_raw)
    mask = lab.mask1024(binding['row']['reference_mask_path']).float()
    y_cpu = lab.coverage64(mask).flatten()
    device = torch.device(SETTINGS['device'])
    if device.type == 'mps':
        lab.DEV = device
        lab.basis.cache_clear()
        r, q = lab.debias(r_raw.to(device)), lab.debias(q_raw.to(device))
        lab.DEV = torch.device('cpu')
        lab.basis.cache_clear()
    else:
        r, q = r_cpu, q_cpu
    y = y_cpu.to(device)
    mu = .5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    maximum = sim.max()
    kernel = torch.exp((sim - maximum) / .05)
    df, db = kernel @ y, kernel @ (1 - y)
    explanation = df + db
    c = df / explanation.clamp_min(1e-30)
    log_e = explanation.clamp_min(1e-30).log() + maximum / .05
    q_unit = lab.unit(q)
    centroids = lab.kmeans(q_unit, 64, iters=6)
    labels = (q_unit @ centroids.T).argmax(1)
    onehot = F.one_hot(labels, 64).float()
    region = lambda v: onehot @ ((onehot.T @ v) / onehot.sum(0).clamp_min(1))
    sim_max = sim.max(1).values
    ind0, indn = region(c), region(c * lab.minmax(sim_max))
    reverse = torch.softmax(sim / .05, dim=0)
    back_f = reverse @ y / y.sum().clamp_min(1e-6)
    back_b = reverse @ (1 - y) / (1 - y).sum().clamp_min(1e-6)
    # Match P0's exhaustive cosine neighbors; R1 evaluates graph weights with
    # CPU-debiased q even when P0 neighbor identities originated on MPS.
    similarity = q_unit @ q_unit.T - torch.eye(4096, device=device) * 1e9
    neighbors = similarity.topk(20, dim=1).indices.cpu().numpy().astype(np.uint16)
    rho = float(1 - (y * (y >= .9)).sum() / y.sum().clamp_min(1e-9))
    fields = {'c32': c.cpu().numpy(), 'log_e': log_e.cpu().numpy().astype(np.float16),
              'sim_max': sim_max.cpu().numpy().astype(np.float16),
              'back_f': back_f.cpu().numpy().astype(np.float16), 'back_b': back_b.cpu().numpy().astype(np.float16),
              'ind0_32': ind0.cpu().numpy(), 'indn_32': indn.cpu().numpy(),
              'regions64': labels.cpu().numpy().astype(np.uint16), 'query_neighbors20': neighbors,
              'reference_rho': np.asarray([rho], np.float32)}
    return r_cpu, q_cpu, mask, y_cpu, fields


def graph(q, neighbors):
    q = F.normalize(q, dim=1)
    ids = torch.as_tensor(neighbors.astype(np.int64))
    values = torch.empty((4096, 20), dtype=torch.float32)
    for start in range(0, 4096, 256):
        values[start:start + 256] = (q[start:start + 256, None] * q[ids[start:start + 256]]).sum(-1)
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), neighbors.ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    w = w / max(float(np.asarray(w.sum(1)).ravel().mean()), 1e-8)
    return sparse.diags(np.asarray(w.sum(1)).ravel()) - w


def solve(primary, explanation, laplacian):
    s = lab.minmax(torch.as_tensor(primary, dtype=torch.float32).reshape(-1)).numpy()
    a = (.1 + np.abs(2 * s - 1)) * explanation
    a = (a / a.mean()).astype(np.float64)
    y = s.astype(np.float64)
    h, rhs = sparse.diags(a) + 16 * laplacian, a * y
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    value, status = cg(h, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    residual = float(np.linalg.norm(h @ value - rhs) / max(np.linalg.norm(rhs), 1e-12))
    assert status == 0 and np.isfinite(value).all() and residual < 2e-7
    return value.reshape(64, 64).astype(np.float32), {'iterations': iterations[0], 'relative_residual': residual}


def vp_additions(binding, r, q, mask, y):
    row = binding['row']
    with Image.open(ASSETS / row['query_path']) as opened:
        image = np.asarray(opened.convert('RGB')).copy()
    if row.get('query_crop') is not None:
        x0, y0, x1, y1 = row['query_crop']
        image = image[y0:y1, x0:x1]
    assert array_hash(image) == row['query_rgb_hash']
    host._tgt_image = host.transform(Image.fromarray(image))
    ref_map, q_map = r.T.reshape(1, 1, 1024, 64, 64).contiguous(), q.T.reshape(1, 1024, 64, 64).contiguous()
    mask = mask[None, None]
    prototype = lab.unit((r * y[:, None]).sum(0))
    candidates, vote = host.locate(ref_map, q_map, mask, prototype, 1, 64, 64)
    dense = torch.zeros((64, 64), dtype=torch.bool)
    dense[::2, ::2] = True
    prior = host.prior(ref_map, q_map, mask, candidates | dense, 1, 64, 64)
    host._tgt_image = None
    return .20 * (vote.numpy() - .5), .25 * (prior.numpy() - .5)


def natural_readout(field, rule):
    if rule == 'midrange':
        return pc.readout(field)
    raise ValueError('Unfrozen readout rule: ' + rule)


def pixel_geometry(binding, field, baseline):
    label = lab.mask1024(binding['row']['reference_mask_path']).numpy()
    train, holdout = pixel.source_split()
    descriptor = pixel.descriptor(pixel.rgb(binding['row'], 'reference'))
    beta, fit = geometry.fit_pairwise(geometry.pairs(descriptor, label, train))
    reference_semantic = lab.up(lab.coverage64(torch.from_numpy(label))).numpy()
    alpha, selection = geometry.select_alpha(reference_semantic, descriptor @ beta.astype(np.float32), label, holdout)
    del descriptor
    counts = geometry.blocks(baseline).sum(1)
    semantic = geometry.standardize_within(lab.up(torch.from_numpy(field)).numpy())
    residual = geometry.standardize_within(pixel.descriptor(pixel.rgb(binding['row'], 'query')) @ beta.astype(np.float32))
    prediction = geometry.rerank(semantic, residual, counts, alpha, baseline)
    assert np.array_equal(geometry.blocks(prediction).sum(1), counts)
    return prediction, {'beta': beta.tolist(), 'fit': fit, 'selection': selection,
                        'all_patch_areas_preserved': True, 'alpha_zero_exact_identity': bool(alpha != 0 or np.array_equal(prediction, baseline))}


def infer(binding):
    started, counters, detail = time.monotonic(), {}, {}
    with pc.inference_only(binding):
        r, q, mask, y, fields = independent(binding, counters)
        rho = float(fields['reference_rho'][0])
        if rho > .9:
            field = fields['c32'].reshape(64, 64)
            detail['raw_T_fallback'] = True
        else:
            c_mod, selection = modulate(torch.from_numpy(fields['c32']), torch.from_numpy(fields['regions64'].astype(np.int64)),
                                       fields['ind0_32'], fields['indn_32'], fields['sim_max'], fields['back_f'])
            addition_v, addition_p = vp_additions(binding, r, q, mask, y)
            # Preserve R1's arithmetic order: addition_V + addition_P, then c_mod.
            primary = c_mod.numpy().reshape(64, 64) + (addition_v + addition_p)
            log_e = fields['log_e'].astype(np.float64)
            explanation = expit(log_e - np.median(log_e)).astype(np.float32)
            field, solver = solve(primary, explanation, graph(q, fields['query_neighbors20']))
            fields.update(c_mod=c_mod.numpy(), addition_V=addition_v, addition_P=addition_p,
                          primary=primary, explanation=explanation)
            detail.update(modulation=selection, solver=solver, raw_T_fallback=False)
        fields['VP_UN'] = field
        baseline = natural_readout(field, SETTINGS['readout_rule'])
        predictions = {'UN_identity': pc.pack(baseline)}
        if SETTINGS['use_pixel']:
            prediction, detail['geometry'] = pixel_geometry(binding, field, baseline)
            predictions['UN_pixel'] = pc.pack(prediction)
    detail.update(counters, seconds=time.monotonic() - started, reference_rho=rho,
                  reference_count=1, new_encoder_calls=0, inference_query_GT_reads=0,
                  FoRIS_score_consumed=False, complete_FoRIS_calls=0)
    if SETTINGS['device'] == 'mps':
        torch.mps.empty_cache()
    return predictions, fields, detail


def produce_one(job):
    binding, destination = job
    out, name = Path(destination), pc.key(binding)
    record_path = out / 'records' / (name + '.json')
    if record_path.exists():
        record = read(record_path)
        for kind in ('prediction', 'field'):
            assert sha(record[kind + '_path']) == record[kind + '_sha256']
        return record
    predictions, fields, detail = infer(binding)
    pred_path, field_path = out / 'predictions' / (name + '.npz'), out / 'fields' / (name + '.npz')
    assert not pred_path.exists() and not field_path.exists(), 'Preserve incomplete output for inspection'
    np.savez_compressed(pred_path, **predictions)
    np.savez_compressed(field_path, **fields)
    record = {'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
              'prediction_path': str(pred_path), 'prediction_sha256': sha(pred_path),
              'field_path': str(field_path), 'field_sha256': sha(field_path), **detail}
    write(record_path, record)
    return record


def run_rows(rows, out, args):
    for folder in ('predictions', 'fields', 'records'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.out, args.device, 2, args.readout, args.pixel)) as pool:
        for i, record in enumerate(pool.map(produce_one, ((b, str(out)) for b in rows), chunksize=1), 1):
            records.append(record)
            if i <= 10 or i % 25 == 0:
                print({'phase': args.mode, 'completed': i, 'n': len(rows), 'seconds': round(time.monotonic() - started, 1)}, flush=True)
    return records, time.monotonic() - started


def reproduce(args):
    prepare_runtime(args.out)
    initialize(args.out, args.device, 2, args.readout, True)
    rows = read(ASSETS / 'a/x1012_R1/manifest.json')
    rows = [next(b for b in rows if b['dataset'] == domain) for domain in lab.ALL_DOMAINS]
    destination = args.out / ('reproduction_' + args.device)
    write(destination / 'manifest.json', rows)
    records, seconds = run_rows(rows, destination, args)
    comparisons = []
    for binding, record in zip(rows, records):
        name = pc.key(binding)
        old_r2 = read(ASSETS / 'a/x1012_R2/records' / (name + '.json'))
        with np.load(record['field_path'], allow_pickle=False) as z, np.load(ASSETS / 'a/x1012_R1/fields' / (name + '.npz'), allow_pickle=False) as old:
            diff = np.abs(z['VP_UN'] - old['VP_UN'])
            result = {'index': name, 'dataset': binding['dataset'], 'field_max_abs': float(diff.max()), 'field_mean_abs': float(diff.mean())}
        with np.load(record['prediction_path'], allow_pickle=False) as z, np.load(ASSETS / 'a/x1012_R1/predictions' / (name + '.npz'), allow_pickle=False) as r1, np.load(ASSETS / 'a/x1012_R3/predictions' / (name + '.npz'), allow_pickle=False) as r3:
            result['R1_mask_xor'] = int(np.count_nonzero(pc.unpack(z['UN_identity']) ^ pc.unpack(r1['VP_UN'])))
            result['R3_pixel_mask_xor'] = int(np.count_nonzero(pc.unpack(z['UN_pixel']) ^ pc.unpack(r3['UN_pixel'])))
        result.update(beta_max_abs=float(np.max(np.abs(np.asarray(record['geometry']['beta']) - np.asarray(old_r2['beta'])))),
                      alpha=record['geometry']['selection']['selected'], old_alpha=old_r2['selection']['selected'],
                      seconds=record['seconds'])
        comparisons.append(result)
    report = {'state': 'TEN_OLD_NATURAL_REPRODUCTIONS_COMPLETE', 'query_GT_reads': 0, 'n': len(rows),
              'device': args.device, 'seconds': seconds, 'source_sha256': sha(__file__), 'comparisons': comparisons}
    write(destination / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def freeze(args):
    runtime = prepare_runtime(args.out)
    rows = read(args.manifest)
    assert len(rows) == 500 and len({str(b['index']) for b in rows}) == 500
    config = {'state': 'CANDIDATE_FROZEN_BEFORE_NEW_NATURAL_OUTPUTS', 'n': len(rows),
              'method': 'R1_VP_UN', 'geometry': 'R2_reference_only' if args.pixel else None,
              'readout': args.readout, 'device': args.device, 'threads_per_worker': 2,
              'reference_count': 1, 'new_encoder_calls': 0, 'query_GT_in_inference': False,
              'runtime_sha256': sha(runtime / 'runtime.json'), 'manifest_sha256': sha(args.manifest),
              'source_sha256': sha(__file__)}
    target = args.out / 'config.json'
    if target.exists():
        assert read(target) == config, 'Frozen candidate changed'
    else:
        shutil.copyfile(args.manifest, args.out / 'manifest.json')
        (args.out / 'frozen').mkdir(exist_ok=True)
        shutil.copyfile(__file__, args.out / 'frozen/r5_candidate.py')
        write(target, config)
    print(config, flush=True)


def produce(args):
    config = read(args.out / 'config.json')
    assert config['source_sha256'] == sha(__file__)
    assert config['manifest_sha256'] == sha(args.out / 'manifest.json')
    assert config['runtime_sha256'] == sha(args.out / 'runtime/runtime.json')
    assert config['device'] == args.device and config['readout'] == args.readout
    assert bool(config['geometry']) == args.pixel
    prepare_runtime(args.out)
    rows = read(args.out / 'manifest.json')
    records, seconds = run_rows(rows, args.out, args)
    write(args.out / 'sealed.json', {'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(rows),
          'records': records, 'manifest_sha256': sha(args.out / 'manifest.json'),
          'config_sha256': sha(args.out / 'config.json'), 'wall_seconds': seconds,
          'case_seconds_sum': sum(r['seconds'] for r in records), 'workers': args.workers,
          'threads_per_worker': 2, 'reference_count': 1, 'new_encoder_calls': 0, 'inference_query_GT_reads': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reproduce', 'freeze', 'produce'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--readout', default='midrange')
    parser.add_argument('--pixel', action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    arguments.out = arguments.out.resolve()
    if arguments.device == 'mps':
        assert arguments.workers == 1, 'One MPS process only'
    globals()[arguments.mode](arguments)
