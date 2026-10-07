#!/usr/bin/env python3
"""PLAN C's two fixed Pro repairs, cached features only; encoder forwards = 0.

Server execution (the author of this file does not execute experiments):
  python scripts/bench_pro_repairs.py --a-run A_six_levels_v6 \
    --rgb-transform-source /path/to/FoRIS/utils/data.py --out C_run --phase all
  python scripts/bench_pro_repairs.py [same arguments] --phase score
  python scripts/bench_pro_repairs.py --a-run A_six_levels_v6 \
    --rgb-transform-source /path/to/FoRIS/utils/data.py --out C_holdout \
    --holdout-run B_directory --fresh-result C_run/report.json --phase all

Exactly three arms: card1.mode_return, card1.global.control, and
card2.pure_bg.control. The last arm uses existing NORMAL reference features:
it is NOT the card's all-layer attention intervention. No model is loaded.
The current PLAN's explicit two-card exception defines this execution scope.
No sweep, full-FG subspace projection, additional baseline call, or new arm.

A's actual frozen source closure is mandatory. A's saved baseline_evidence /
baseline_unit / baseline_rcg are used as-is. B uses its actual FP32 fields,
with rcg explicitly interpreted as RCG_s2. If B did not save auxiliary
reference statistics, the SAME frozen native Part2 prefix reconstructs them
once per case and verifies its s2 against B's FP32 s2; this is not an encoder
or a recomputation of the comparison baseline/graph.

Timing uses the first five ordered cases, saves their complete fields, and
reuses those fields in the one 600-case run. Query truth is never indexed
during timing/inference; it is first read by the score phase after the seal.
Scores are binary 64x64 positions (class-summed I/U), not original-size IoU.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

SERVER = Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
ARMS = ('card1.mode_return', 'card1.global.control', 'card2.pure_bg.control')
N = 600
_BENCH = None
_CONFIG = None
_PREFIX = None
_C_WORKER = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def document(path):
    return json.loads(Path(path).read_text())


def require_hash(path, digest):
    if sha(path) != digest:
        raise ValueError('Bound file changed: ' + str(path))


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def load_frozen_a(a_run, hashes):
    """Import only A's recorded code, never the changing repository's recipe."""
    global _BENCH
    source = Path(a_run) / 'source'
    for name, digest in hashes.items():
        require_hash(source / name, digest)
    for name in list(sys.modules):
        if name == 'ics' or name.startswith('ics.'):
            del sys.modules[name]
    sys.path.insert(0, str(source / 'src'))
    path = source / 'scripts/bench_evidence.py'
    spec = importlib.util.spec_from_file_location('_pro_actual_A_bench', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.assert_recipe()
    _BENCH = module
    return module


def resize_source_binding(path):
    """Check the actual FoRIS transform source without importing/encoding it."""
    path = Path(path)
    tree = ast.parse(path.read_text())
    function = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == 'build_transform'), None)
    if function is None:
        raise ValueError('Actual FoRIS build_transform function required')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
    named = lambda n, name: isinstance(n.func, ast.Attribute) and n.func.attr == name
    resizes = [n for n in calls if named(n, 'Resize')]
    if len(resizes) != 1:
        raise ValueError('Exactly one actual square Resize required')
    resize_call = resizes[0]
    if resize_call.args:
        if len(resize_call.args) != 1 or resize_call.keywords:
            raise ValueError('Resize interpolation/support overrides are not supported')
        size_node = resize_call.args[0]
    else:
        if len(resize_call.keywords) != 1 or resize_call.keywords[0].arg != 'size':
            raise ValueError('Exactly the source size keyword and default interpolation required')
        size_node = resize_call.keywords[0].value
    bound_square = (isinstance(size_node, ast.Tuple) and len(size_node.elts) == 2
                    and all(isinstance(v, ast.Name) and v.id == 'image_size' for v in size_node.elts)
                    and any(arg.arg == 'image_size' for arg in function.args.args))
    if not bound_square:
        try:
            bound_square = tuple(ast.literal_eval(size_node)) == (1024, 1024)
        except (ValueError, TypeError):
            bound_square = False
    if not bound_square:
        raise ValueError('FoRIS square image_size bound to this1024 cache required; no overrides')
    if not any(named(n, 'ToTensor') for n in calls) or not any(named(n, 'Normalize') for n in calls):
        raise ValueError('FoRIS Resize/ToTensor/Normalize source chain required')
    from PIL import Image
    import torchvision.transforms as transforms
    resize = transforms.Resize((1024, 1024))
    if resize.interpolation != transforms.InterpolationMode.BILINEAR:
        raise ValueError('Installed torchvision default interpolation is not bound BILINEAR')
    return dict(**bind(path), function_source_sha256=hashlib.sha256(
        ast.get_source_segment(path.read_text(), function).encode()).hexdigest(),
        actual_transform='default torchvision Resize((1024,1024)) on PIL RGB, then ToTensor/Normalize',
        support_transform='class binary PNG as PIL mode F -> same PIL BILINEAR1024; any nonzero in16x16 excludes patch',
        pillow_version=Image.__version__, interpolation='BILINEAR', size=[1024, 1024],
        antialias_support='PIL resizing support, including downsampling filter footprint; no nearest approximation')


def field_path(provider, directory, key, receipt):
    local = Path(provider) / directory / (key + '.npz')
    path = local if local.exists() else Path(receipt['field_path'])
    require_hash(path, receipt['field_sha256'])
    return path


def load_aux(path):
    import numpy as np
    required = ('s2', 'mu_fg', 'mu_bg_hard', 'fg_prototypes')
    with np.load(path, allow_pickle=False) as pack:
        values = {name: pack[name].copy() for name in required}
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Actual finite FP32 native Part2 auxiliaries required')
    if (values['s2'].shape != (64, 64) or values['mu_fg'].shape != (1024,)
            or values['mu_bg_hard'].shape != (1024,) or values['fg_prototypes'].ndim != 2
            or values['fg_prototypes'].shape[1] != 1024 or not len(values['fg_prototypes'])):
        raise ValueError('Native Part2 auxiliary shapes changed')
    return values


def known_reference(row, cov, normalized_r, auxiliary):
    """Exact source FG/pool plus source RGB support; never reads query labels."""
    import io
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    fg, receipt = _BENCH.native_foreground(row, _CONFIG['annotation_root'], cov)
    annotation = Path(_CONFIG['annotation_root']) / Path(row['support']).with_suffix('.png')
    data = annotation.read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt['reference_annotation_sha256']:
        raise ValueError('Reference PNG changed between source FG and support construction')
    with Image.open(io.BytesIO(data)) as image:
        labels = np.asarray(image)
        if labels.ndim != 2 or labels.dtype.kind not in 'uib':
            raise ValueError('Actual original integer annotation PNG required')
        target = (labels == int(row['c']) + 1).astype(np.float32)
    rgb = Path(_CONFIG['rgb_root']) / row['support']
    with Image.open(rgb) as image:
        if image.size != (target.shape[1], target.shape[0]):
            raise ValueError('Actual reference RGB and original annotation geometry differ')
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Single original RGB photo required')
    # Mode F preserves the nonzero support of the resampling kernel instead of
    # rounding tiny foreground contributions to zero in an 8-bit resized mask.
    bilinear = getattr(Image, 'Resampling', Image).BILINEAR
    support = np.asarray(Image.fromarray(target, mode='F').resize((1024, 1024),
                         resample=bilinear), np.float32)
    if not np.isfinite(support).all() or support.min() < 0:
        raise ValueError('PIL bilinear class support invalid')
    contaminated = (support > 0).reshape(64, 16, 64, 16).any(axis=(1, 3)).reshape(-1)
    background_indices = torch.nonzero(~fg, as_tuple=False).flatten()
    if len(background_indices):
        background = normalized_r[background_indices]
        # This is the original pool, BEFORE pure-patch filtering. No reselection
        # of a top20% among the pure patches, no majority-cov FG approximation.
        hard = (background @ torch.from_numpy(auxiliary['mu_fg'])).topk(
            max(1, int(.2 * len(background)))).indices
        hard_indices = background_indices[hard].numpy()
        computed_b = F.normalize(normalized_r[hard_indices].mean(0), dim=0).numpy()
        diagnostic = float(np.max(np.abs(computed_b - auxiliary['mu_bg_hard'])))
        if diagnostic > 1e-6:
            raise ValueError('Reconstructed original hard BG pool disagrees with saved native mean')
    else:
        hard_indices = np.empty(0, np.int64)
        diagnostic = float(np.max(np.abs(auxiliary['mu_bg_hard'])))
        if diagnostic != 0:
            raise ValueError('Empty source BG but nonzero saved native hard mean')
    indices = hard_indices[(cov.reshape(-1)[hard_indices] == 0) & ~contaminated[hard_indices]]
    receipt.update(original_rgb=bind(rgb), original_hw=list(target.shape),
                   RGB_support_array_sha256=_BENCH.array_sha(support),
                   RGB_contaminated_patch_array_sha256=_BENCH.array_sha(contaminated),
                   native_hard_pool_indices=hard_indices.tolist(), pure_J_indices=indices.tolist(),
                   hard_pool_size=len(hard_indices), pure_J_size=len(indices),
                   reconstructed_hard_mean_max_abs_error=diagnostic,
                   pure_J_array_sha256=_BENCH.array_sha(indices),
                   J_rule='original native hard BG pool intersect exact cov==0 and PIL BILINEAR RGB support==0',
                   all_layer_candidate_J_if_later_authorized='must reuse these exact indices; currently NOT_EXECUTED')
    return indices, receipt


def card1_evidence(q_unit, auxiliary, *, global_control=False):
    """No re-unit of g or p-prime, including the global arithmetic-mean control."""
    import torch
    p = torch.from_numpy(auxiliary['fg_prototypes'])
    if len(p) == 1:
        return auxiliary['s2'].copy(), dict(fallback='K1_native_s2', prototypes=1)
    b = torch.from_numpy(auxiliary['mu_bg_hard'])  # already unit BEFORE source orthogonalization
    c = b @ p.T
    if global_control:
        c = c.mean().expand_as(c)
    g = b[None, :] - c[:, None] * p
    p_prime = p - .55 * g
    value = .07 * torch.logsumexp((q_unit @ p_prime.T) / .07, dim=1)
    return value.reshape(64, 64).numpy(), dict(fallback=None, prototypes=len(p),
        c_min=float(c.min()), c_max=float(c.max()), c_mean=float(c.mean()),
        g_or_p_prime_reunit=False, preorthogonal_unit_b=True,
        global_control=global_control)


def pure_bg_evidence(q_unit, r_unit, auxiliary, indices):
    import torch
    if not len(indices):
        return auxiliary['s2'].copy(), dict(fallback='empty_J_native_s2', pure_J_size=0,
                                            all_layer_attention_intervention='NOT_EXECUTED')
    mu = torch.from_numpy(auxiliary['mu_fg'])
    mean = r_unit[indices].mean(0)
    orth = mean - torch.dot(mean, mu) * mu
    n = orth / orth.norm().clamp_min(1e-8)
    p = torch.from_numpy(auxiliary['fg_prototypes'])
    foreground = .07 * torch.logsumexp((q_unit @ p.T) / .07, dim=1)
    value = foreground - .55 * (q_unit @ n)
    return value.reshape(64, 64).numpy(), dict(fallback=None, pure_J_size=len(indices),
        orthogonal_residual_norm=float(orth.norm()), reference_feature_source='existing NORMAL processed cache',
        mean_before_projection='arithmetic mean of original per-token unit reference features in J',
        all_layer_attention_intervention='NOT_EXECUTED', added_encoder_forwards=0)


def initialize_worker(config):
    global _CONFIG, _PREFIX
    _CONFIG = config
    _PREFIX = None
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(config['threads'])
    import torch
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    load_frozen_a(config['a_run'], config['a_source_hashes'])


def initialize_frozen_worker(config):
    """Bind the saved implementation once, without pickling600 receipts per job."""
    global _C_WORKER
    path = Path(config['out'])/'source/scripts/bench_pro_repairs.py'
    require_hash(path, config['script_sha256'])
    spec = importlib.util.spec_from_file_location('_pro_frozen_C_worker', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.initialize_worker(config)
    _C_WORKER = module


def frozen_dispatch(index, row, requested):
    """Worker methods execute C's saved bytes, not the live entry-point body."""
    if _C_WORKER is None:
        raise RuntimeError('Frozen C worker was not initialized')
    return _C_WORKER.compute_case(index, row, requested)


def compute_case(index, row, requested):
    global _PREFIX
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.methods import rcg, stage_bank
    wall, cpu = time.monotonic(), time.process_time()
    try:
        q, r, provenance = _BENCH.features(Path(row['feature_export']))
        pack, packet_sha = _BENCH.packet(Path(row['packet_export']))
        entry = _CONFIG['native_receipts'][row['key']]
        if (entry['features']['feature_sha256'] != provenance['feature_sha256']
                or entry['packet_sha256'] != packet_sha):
            raise ValueError('Feature/packet differs from A/B accepted native field')
        aux_seconds = 0.
        if _CONFIG['cohort'] == 'fresh600':
            source = field_path(_CONFIG['a_run'], 'acceptance2_fields', row['key'], entry)
            auxiliary = load_aux(source)
            aux_binding = bind(source)
        else:
            # B stored only the endpoint, not p/b. Reconstruct the SAME native
            # source statistics, never rerun its graph or invent cached p/b.
            source = field_path(_CONFIG['holdout_run'], 'fields', row['key'], entry)
            with np.load(source, allow_pickle=False) as z:
                saved_s2, reference = z['s2'].copy(), z['ref'].copy()
            if not np.array_equal(reference, pack['cov']):
                raise ValueError('B FP32 reference coverage differs from original packet')
            start = time.monotonic()
            if _PREFIX is None:
                _PREFIX = _BENCH.part2_prefix()
            fg, _ = _BENCH.native_foreground(row, _CONFIG['annotation_root'], pack['cov'])
            head = _BENCH.native_evidence(fg, _PREFIX[0])
            reconstructed = head(q, r, pack['cov'])
            auxiliary = dict(head.last_aux, s2=saved_s2)
            if (saved_s2.dtype != np.float32 or not np.array_equal(reconstructed, saved_s2)):
                raise ValueError('B auxiliary reconstruction is not exactly its frozen FP32 native s2')
            aux_seconds = time.monotonic() - start
            aux_binding = dict(**bind(source), native_aux_reconstructed=True,
                               reconstructed_s2_array_sha256=_BENCH.array_sha(reconstructed),
                               prefix_source_sha256=_PREFIX[1], baseline_RCG_calls=0)
        with torch.inference_mode():
            q_unit = F.normalize(torch.from_numpy(q), dim=1)
            r_unit = F.normalize(torch.from_numpy(r), dim=1)
            if 'card2.pure_bg.control' in requested:
                indices, reference_receipt = known_reference(row, pack['cov'], r_unit, auxiliary)
                original_receipt = entry.get('reference_mask', entry.get('reference_source', {}))
                if original_receipt.get('reference_annotation_sha256') != reference_receipt['reference_annotation_sha256']:
                    raise ValueError('Reference original PNG differs from accepted A/B prompt')
                if original_receipt.get('foreground_array_sha256') != reference_receipt['foreground_array_sha256']:
                    raise ValueError('Source foreground/native hard pool differs from accepted A/B prompt')
            else:
                indices, reference_receipt = None, None
            common_wall, common_cpu = time.monotonic() - wall, time.process_time() - cpu
            values, timings, solvers, heads = {}, {}, {}, {}
            for arm in requested:
                start, start_cpu = time.monotonic(), time.process_time()
                if arm.startswith('card1.'):
                    evidence, head_info = card1_evidence(q_unit, auxiliary,
                                                       global_control=arm == 'card1.global.control')
                else:
                    evidence, head_info = pure_bg_evidence(q_unit, r_unit, auxiliary, indices)
                direct = stage_bank.unit(torch.from_numpy(evidence)).numpy()
                evidence_wall, evidence_cpu = time.monotonic() - start, time.process_time() - start_cpu
                start, start_cpu = time.monotonic(), time.process_time()
                graph, solver = rcg.predict(q, r, pack['cov'], evidence, device='cpu')
                timings[arm] = dict(evidence_unit_wall_seconds=evidence_wall,
                    evidence_unit_CPU_seconds=evidence_cpu, RCG_wall_seconds=time.monotonic()-start,
                    RCG_CPU_seconds=time.process_time()-start_cpu)
                values.update({arm + '.evidence': evidence, arm + '.unit': direct, arm + '.rcg': graph})
                solvers[arm], heads[arm] = solver, head_info
        if any(v.shape != (64, 64) or v.dtype != np.float32 or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Every arm must output a complete finite FP32 physical64 field')
        path = Path(_CONFIG['out']) / 'fields' / (row['key'] + '.npz')
        if path.exists():
            raise ValueError('Refuse to overwrite an existing case field without its sealed receipt')
        _BENCH.write_npz(path, values)
        receipt = dict(state='complete', index=index, key=row['key'], arms=list(requested),
            field_path=str(path), field_sha256=sha(path),
            field_array_hashes={name: _BENCH.array_sha(v) for name, v in values.items()},
            features=provenance, packet_sha256=packet_sha, native_auxiliary=aux_binding,
            reference=reference_receipt, native_aux_reconstruction_wall_seconds=aux_seconds,
            common_wall_seconds=common_wall, common_CPU_seconds=common_cpu,
            timings=timings, solvers=solvers, heads=heads,
            worker_wall_seconds=time.monotonic()-wall, worker_CPU_seconds=time.process_time()-cpu,
            query_GT_indexed_in_worker=False, new_encoder_forwards=0, new_baseline_RCG_calls=0)
    except Exception as error:
        receipt = dict(state='failed', index=index, key=row['key'], error=repr(error),
            traceback=traceback.format_exc(), worker_wall_seconds=time.monotonic()-wall,
            worker_CPU_seconds=time.process_time()-cpu, query_GT_indexed_in_worker=False,
            new_encoder_forwards=0, fallback='none except original K1/emptyJ contracts')
    write(Path(_CONFIG['out']) / 'receipts' / (row['key'] + '.json'), receipt)
    return receipt


def read_receipt(out, row, required):
    path = Path(out) / 'receipts' / (row['key'] + '.json')
    if not path.exists():
        return None
    receipt = document(path)
    if receipt.get('state') != 'complete':
        raise ValueError('Preserved failed case; stop rather than automatically rerun: ' + row['key'])
    if receipt.get('key') != row['key'] or not set(required) <= set(receipt['arms']):
        raise ValueError('Existing complete case has a different arm contract')
    require_hash(receipt['field_path'], receipt['field_sha256'])
    return receipt


def run_pool(config, rows, indices, requested):
    todo = [i for i in indices if read_receipt(config['out'], rows[i], requested) is None]
    start = time.monotonic()
    if todo:
        context = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['workers'], mp_context=context,
                initializer=initialize_frozen_worker, initargs=(config,)) as pool:
            iterator = iter(todo)
            pending = {}
            for _ in range(min(len(todo), config['workers'] * 2)):
                i = next(iterator)
                pending[pool.submit(frozen_dispatch, i, rows[i], requested)] = i
            complete = len(indices) - len(todo)
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    i = pending.pop(future)
                    receipt = future.result()
                    complete += 1
                    write(Path(config['out'])/'progress.json', dict(state='C_INFERENCE_NO_GT',
                          completed=complete, scheduled=len(indices), last_key=rows[i]['key']))
                    if receipt['state'] != 'complete':
                        for task in pending:
                            task.cancel()
                        raise ValueError('Case failed; completed fields retained: ' + repr(receipt))
                    i = next(iterator, None)
                    if i is not None:
                        pending[pool.submit(frozen_dispatch, i, rows[i], requested)] = i
    return dict(wall_seconds=time.monotonic()-start, newly_computed_cases=len(todo),
                reused_complete_cases=len(indices)-len(todo))


def timing_phase(config, rows):
    import numpy as np
    path = Path(config['out']) / 'timing.json'
    if path.exists():
        result = document(path)
        if result['config_sha256'] != config['config_sha256']:
            raise ValueError('Existing timing contract differs')
        for row in rows[:5]:
            read_receipt(config['out'], row, config['eligible_arms'])
        return result
    pool = run_pool(config, rows, list(range(5)), config['eligible_arms'])
    receipts = [read_receipt(config['out'], r, config['eligible_arms']) for r in rows[:5]]
    # Five cases cannot occupy16workers. Record observed occupancy; extrapolate
    # to the declared32-CPU budget without calling the projection a measurement.
    estimates = {}
    for arm in config['eligible_arms']:
        walls = [r['common_wall_seconds'] + r['timings'][arm]['evidence_unit_wall_seconds']
                 + r['timings'][arm]['RCG_wall_seconds'] for r in receipts]
        cpus = [r['common_CPU_seconds'] + r['timings'][arm]['evidence_unit_CPU_seconds']
                + r['timings'][arm]['RCG_CPU_seconds'] for r in receipts]
        estimates[arm] = dict(first5_case_wall_seconds=walls, first5_case_CPU_seconds=cpus,
            projected600_CPU_seconds=float(np.mean(cpus)*N),
            projected600_wall_seconds=max(float(np.max(walls))*math.ceil(N/config['workers']),
                float(np.mean(cpus))*N/(config['workers']*config['threads'])) + pool['wall_seconds'])
    costs = {}
    if 'card1.mode_return' in config['eligible_arms']:
        bundle = ['card1.mode_return', 'card1.global.control']
        estimates_wall = sum(estimates[a]['projected600_wall_seconds'] for a in bundle)
        costs['card1_with_required_global_control'] = dict(projected600_wall_seconds=estimates_wall,
                                                           within20minutes=estimates_wall<=1200)
    if 'card2.pure_bg.control' in config['eligible_arms']:
        costs['card2_zero_encoding_control'] = dict(
            projected600_wall_seconds=estimates['card2.pure_bg.control']['projected600_wall_seconds'],
            within20minutes=estimates['card2.pure_bg.control']['projected600_wall_seconds']<=1200)
    active = [a for a in config['eligible_arms'] if costs[
        'card1_with_required_global_control' if a.startswith('card1.') else 'card2_zero_encoding_control']['within20minutes']]
    result = dict(state='TIMING_PASS' if active else 'SKIPPED_COST', config_sha256=config['config_sha256'],
        fixed_first5_keys=[r['key'] for r in rows[:5]], workers=config['workers'],
        threads_per_worker=config['threads'], available_CPUs=config['workers']*config['threads'],
        first5_max_active_workers=min(5, config['workers']), observed_pool=pool,
        estimates=estimates, cost_gate=costs, active_arms=active, skipped_arms=[a for a in config['eligible_arms'] if a not in active],
        uncertainty='600 projections from fixed5, not measured600 cost; content/parallel contention unknown; max observed per-case wave and CPU-capacity bound plus cold pool cost',
        timing_case_fields_reused_in_full_inference=True, new_encoder_forwards=0)
    write(path, result)
    return result


def inference_phase(config, rows, timing):
    active = timing['active_arms']
    if not active:
        write(Path(config['out'])/'report.json', dict(state='CLOSED_COST', timing=timing,
                                                      all_layer_card2='NOT_EXECUTED', new_encoder_forwards=0))
        return False
    seal_path = Path(config['out'])/'inference_seal.json'
    if seal_path.exists():
        seal = document(seal_path)
        if (seal['config_sha256']!=config['config_sha256'] or seal['active_arms']!=active
                or seal['ordered_keys']!=[r['key'] for r in rows]):
            raise ValueError('An existing inference seal has another fixed contract')
        for row in rows:
            read_receipt(config['out'],row,active)
            bound = seal['predictions'][row['key']]
            require_hash(bound['field_path'],bound['field_sha256'])
            require_hash(Path(config['out'])/'receipts'/(row['key']+'.json'),bound['receipt_sha256'])
        return True  # Preserve original measured costs; never reseal a no-op as inference.
    pool = run_pool(config, rows, list(range(N)), active)
    receipts = [read_receipt(config['out'], r, active) for r in rows]
    if any(r is None for r in receipts):
        raise ValueError('Cannot seal an incomplete600 run')
    seals = {r['key']: dict(field_path=r['field_path'], field_sha256=r['field_sha256'],
                            receipt_sha256=sha(Path(config['out'])/'receipts'/(r['key']+'.json'))) for r in receipts}
    write(seal_path, dict(state='C_INFERENCE_COMPLETE',
        config_sha256=config['config_sha256'], n=N, active_arms=active, predictions=seals,
        ordered_keys=[r['key'] for r in rows], inference_pool=pool, new_encoder_forwards=0,
        new_baseline_RCG_calls=0, query_GT_indexed_for_inference=False))
    return True


def score_phase(config, rows):
    import numpy as np
    from ics.experiment import photo_groups
    out = Path(config['out'])
    seal = document(out/'inference_seal.json')
    if seal['config_sha256'] != config['config_sha256'] or seal['ordered_keys'] != [r['key'] for r in rows]:
        raise ValueError('Inference seal differs from score cohort/source')
    active = seal['active_arms']
    arrays = {(a, b): np.zeros((N, 2), np.int64) for a in active+['baseline'] for b in ('raw', 'rcg')}
    # FIRST queryGT member read in this script. No truth is passed to a worker.
    if config['cohort'] == 'fresh600':
        with np.load(config['tokens']['path'], allow_pickle=False) as pack:
            saved = {name: pack[name].copy() for name in ('truth', 'rcg')}
    else:
        truth, rcg_s2 = [], []
        for row in rows:
            entry = config['native_receipts'][row['key']]
            source = field_path(config['holdout_run'], 'fields', row['key'], entry)
            with np.load(source, allow_pickle=False) as pack:
                truth.append(pack['truth'].copy()); rcg_s2.append(pack['rcg'].copy())
        saved = dict(truth=np.array(truth), rcg=np.array(rcg_s2))
    target, wrong, comparable, failure_info = _BENCH.failure_sets(saved)
    failure_info['selection_field_origin'] = ('original fresh storedRCG-on-final-score' if config['cohort']=='fresh600'
                                             else 'B fixed FP32 RCG_s2, not fresh historical RCG-on-final-score')
    winners, region_means = {key: 0 for key in arrays}, {key: [] for key in arrays}
    for index, row in enumerate(rows):
        output = seal['predictions'][row['key']]
        require_hash(out/'receipts'/(row['key']+'.json'), output['receipt_sha256'])
        require_hash(output['field_path'], output['field_sha256'])
        if config['cohort'] == 'fresh600':
            entry = config['ladder_receipts'][row['key']]
            source = field_path(config['a_run'], 'ladder_fields', row['key'], entry)
            with np.load(source, allow_pickle=False) as pack:
                baseline = dict(raw=pack['baseline_unit'].copy(), rcg=pack['baseline_rcg'].copy())
                evidence = pack['baseline_evidence'].copy()
            from ics.methods import stage_bank
            import torch
            if not np.array_equal(stage_bank.unit(torch.from_numpy(evidence)).numpy(), baseline['raw']):
                raise ValueError('A saved baseline raw/unit binding changed')
        else:
            entry = config['native_receipts'][row['key']]
            source = field_path(config['holdout_run'], 'fields', row['key'], entry)
            with np.load(source, allow_pickle=False) as pack:
                evidence, graph = pack['s2'].copy(), pack['rcg'].copy()
            from ics.methods import stage_bank
            import torch
            baseline = dict(raw=stage_bank.unit(torch.from_numpy(evidence)).numpy(), rcg=graph)
        fields = {('baseline', backend): value for backend, value in baseline.items()}
        with np.load(output['field_path'], allow_pickle=False) as pack:
            for arm in active:
                for backend in ('raw', 'rcg'):
                    fields[arm, backend] = pack[arm+('.unit' if backend=='raw' else '.rcg')].copy()
        for key, field in fields.items():
            if field.shape!=(64,64) or field.dtype!=np.float32 or not np.isfinite(field).all():
                raise ValueError('Scoring requires original finite FP32 fields')
            arrays[key][index] = _BENCH.counts(field>.5, saved['truth'][index])
            if comparable[index]:
                flat = field.reshape(-1)
                tmean = float(flat[target[index]].mean(dtype=np.float64))
                wmean = float(flat[wrong[index]].mean(dtype=np.float64))
                winners[key] += tmean > wmean
                region_means[key].append(dict(key=row['key'], target_mean=tmean, wrong_region_mean=wmean))
    classes = np.array([row['c'] for row in rows])
    groups = photo_groups(rows)
    number = int(groups.max())+1
    draws = np.random.RandomState(0).randint(number, size=(2000, number))
    weights = np.stack([np.bincount(draw, minlength=number) for draw in draws])[:, groups]
    samples = {key: _BENCH.bootstrap_samples(value, classes, weights) for key, value in arrays.items()}
    scores = {key: _BENCH.miou(value, classes) for key, value in arrays.items()}
    table = []
    for arm in active:
        row = dict(arm=arm)
        for backend in ('raw', 'rcg'):
            key, base = (arm, backend), ('baseline', backend)
            delta = samples[key]-samples[base]
            row[backend] = dict(class_mIoU=scores[key], baseline_class_mIoU=scores[base],
                gain_vs_matched_s2=scores[key]-scores[base], paired_ci95=np.percentile(delta,[2.5,97.5]).tolist(),
                failing_target_mean_above_wrong_region=int(winners[key]),
                baseline_failing_target_mean_above_wrong_region=int(winners[base]),
                comparable_failure_count=failure_info['comparable_count'])
            control = ('card1.global.control', backend)
            if arm=='card1.mode_return' and control in scores:
                row[backend]['vs_global_control'] = dict(gain=scores[key]-scores[control],
                    paired_ci95=np.percentile(samples[key]-samples[control],[2.5,97.5]).tolist())
        if arm=='card1.global.control':
            row['gate'] = dict(role='required strong control, not independent primary candidate', passed=None)
        else:
            graph = row['rcg']
            conditions = dict(graph_gain_above_1pp=graph['gain_vs_matched_s2']>1,
                paired_lower_above_0=graph['paired_ci95'][0]>0,
                failing_images_not_worse=graph['failing_target_mean_above_wrong_region']>=graph['baseline_failing_target_mean_above_wrong_region'])
            if arm=='card1.mode_return':
                conditions['higher_than_global_control'] = graph['vs_global_control']['gain']>0
            if config['cohort']=='holdout600':
                # PLAN C's repetition gate is CI lower>0; the other fixed
                # fresh accuracy/failure gates remain reported. The card1
                # author's required global-control falsifier still applies.
                passed = conditions['paired_lower_above_0']
                if arm=='card1.mode_return':
                    passed = passed and conditions['higher_than_global_control']
            else:
                passed = all(conditions.values())
            row['gate'] = dict(passed=bool(passed), conditions=conditions,
                decision='passes frozen screening/replication gate' if passed else 'closed fixed card; no variant',
                rule='holdout paired CI lower>0; card1 must still beat global control' if config['cohort']=='holdout600' else 'gain>1pp; CI lower>0; failures>=matched baseline; card1>global')
        table.append(row)
    _BENCH.write_npz(out/'counts.npz', {a+'.'+b:v for (a,b),v in arrays.items()})
    np.save(out/'bootstrap_photo_draws.npy', draws)
    write(out/'failure_region_means.json', {a+'.'+b:v for (a,b),v in region_means.items()})
    report = dict(state='C_SCORED_STOP', n=N, cohort=config['cohort'], table=table,
        config_sha256=config['config_sha256'], inference_seal_sha256=sha(out/'inference_seal.json'),
        A_source_hashes=config['a_source_hashes'], script_sha256=config['script_sha256'],
        timing=document(out/'timing.json'), failure_region_protocol=failure_info,
        observed_inference_pool=seal['inference_pool'],
        baseline_failing_target_mean_above_wrong_region={b:int(winners['baseline',b]) for b in ('raw','rcg')},
        bootstrap=dict(draws=2000, seed=0, rng='RandomState(0)', groups=number,
            unit='connected support/query photographs', all_arms_and_baselines_share_draws=True,
            class_denominator='only classes present by resampled row mass, frozen A metric'),
        truth_and_readout='binary64 positions, truth>.5; direct frozen stage_bank.unit>.5; unchanged rcg.predict output>.5',
        exposure='reused development data; no independent confirmation claim',
        native_cache_stage='FoRIS Part1 processed FP16 cache; not raw DINOv3',
        baseline_source='A saved per-case ladder baseline_evidence/unit/rcg' if config['cohort']=='fresh600' else 'B actual FP32 s2 and RCG_s2 episode fields',
        new_encoder_forwards=0, new_baseline_RCG_calls=0, all_layer_card2='NOT_EXECUTED; D report then human decision',
        originality='FoRIS evidence-formula repairs, not our own replacement evidence principle')
    write(out/'report.json', report)
    lines = ['| arm | direct mIoU; gain [paired95%CI] | RCG mIoU; gain [paired95%CI] | target>wrong direct/RCG | decision |',
             '|---|---|---|---|---|']
    for row in table:
        cells = []
        for backend in ('raw','rcg'):
            v = row[backend]; lo, hi = v['paired_ci95']
            cells.append(f"{v['class_mIoU']:.4f}; {v['gain_vs_matched_s2']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        gate = row['gate']
        decision = 'required control' if row['arm']=='card1.global.control' else ('passed' if gate['passed'] else 'closed')
        lines.append(f"| {row['arm']} | {cells[0]} | {cells[1]} | {row['raw']['failing_target_mean_above_wrong_region']}/{row['rcg']['failing_target_mean_above_wrong_region']} of {failure_info['comparable_count']} | {decision} |")
    lines += ['', 'Position-level screening only; baseline and candidate use the same fixed graph/readout.',
              'Card2 here is normal-feature pure-BG control. All-layer reencoding was not executed.',
              'These are FoRIS formula repairs; passing does not establish an original evidence method.']
    (out/'table.md').write_text('\n'.join(lines)+'\n')
    return report


def make_config(args):
    """Input/source identity checks only; no inference or query-label indexing."""
    a = document(args.a_run/'report.json')
    if a.get('state')!='LADDER_COMPLETE' or a.get('n')!=N:
        raise ValueError('Completed actual A600 six-level report required')
    bench = load_frozen_a(args.a_run, a['source_hashes'])
    source = bench.rows(args.input_manifest)
    fresh = bench.rows(args.fresh/'rows.json')
    fresh_inputs = dict(rows_sha256=sha(args.fresh/'rows.json'), tokens_sha256=sha(args.fresh/'tokens.npz'),
                        input_manifest_sha256=sha(args.input_manifest))
    if a['input_hashes']!=fresh_inputs or a['ordered600_keys']!=[r['key'] for r in fresh]:
        raise ValueError('A source cohort/manifest is not the supplied fresh600')
    aux_report_path = args.a_run/'acceptance2_report.json'
    aux_report = document(aux_report_path)
    if (aux_report['source_hashes']!=a['source_hashes'] or aux_report['input_hashes']!=fresh_inputs
            or not aux_report['grade']['continue_ladder']):
        raise ValueError('A auxiliary fields were not accepted under actual frozen closure')
    config = dict(a_run=str(args.a_run.resolve()), a_source_hashes=a['source_hashes'],
        A_report=bind(args.a_run/'report.json'), A_aux_report=bind(aux_report_path),
        A_config=bind(args.a_run/'config.json'), input_manifest=bind(args.input_manifest),
        original_fresh_rows=bind(args.fresh/'rows.json'), original_fresh_tokens=bind(args.fresh/'tokens.npz'),
        annotation_root=str(args.annotation_root.resolve()), rgb_root=str(args.rgb_root.resolve()),
        RGB_transform=resize_source_binding(args.rgb_transform_source),
        workers=args.workers, threads=args.threads, out=str(args.out.resolve()),
        script_sha256=sha(__file__), new_encoder_forwards=0,
        eligible_arms=list(ARMS), cohort='fresh600', holdout_run=None, native_receipts=aux_report['receipts'])
    levels = {r['level']:r for r in a['table']}
    bg_gain = levels['L4']['rcg']['class_mIoU']-levels['L1']['rcg']['class_mIoU']
    config['card2_background_gate'] = dict(measured_A_L4_minus_L1_graph_pp=bg_gain,
                                          at_least2pp=bg_gain>=2, reencoding_authorized=False)
    if bg_gain<2:
        raise ValueError('Current PLAN expects A background gain>=2; no pureBG trial below gate')
    if args.holdout_run is None:
        rows = bench.exact_join(fresh, source)
        ladder_path = args.a_run/'ladder_receipts.json'
        records = document(ladder_path)
        if len(records)!=N or any(r is None or r.get('state')!='complete' for r in records):
            raise ValueError('A600 saved baseline ladder fields are incomplete')
        config.update(ladder_receipts={r['key']:r for r in records}, A_ladder_receipts=bind(ladder_path),
                      tokens=bind(args.fresh/'tokens.npz'), rows=bind(args.fresh/'rows.json'))
    else:
        if args.fresh_result is None:
            raise ValueError('--holdout-run requires actual --fresh-result; never automatic new cohort selection')
        result = document(args.fresh_result)
        if result.get('state')!='C_SCORED_STOP' or result.get('cohort')!='fresh600':
            raise ValueError('Actual completed fresh C report required before repetition')
        if result.get('A_source_hashes')!=a['source_hashes'] or result.get('script_sha256')!=config['script_sha256']:
            raise ValueError('Holdout must repeat exactly the same frozen C/A recipe')
        passed = [r['arm'] for r in result['table'] if r['arm']!='card1.global.control' and r['gate']['passed']]
        if not passed:
            raise ValueError('No fresh card passed; holdout remains NOT_RUN, cards closed')
        eligible = [arm for arm in ARMS if arm in passed or (arm=='card1.global.control' and 'card1.mode_return' in passed)]
        b = document(args.holdout_run/'report.json')
        if (b.get('state')!='HOLDOUT_TOKENS_COMPLETE' or b.get('n')!=N or b.get('rcg_input')!='s2'
                or b.get('inference_source_hashes') is None):
            raise ValueError('B actual600 complete native-s2/RCG_s2 provider required')
        for name, digest in b['inference_source_hashes'].items():
            if a['source_hashes'].get(name)!=digest:
                raise ValueError('B native evidence/graph is not actual frozen A source: '+name)
        if b['source_binding']['files']['input_manifest']['sha256']!=sha(args.input_manifest):
            raise ValueError('B was generated from another1200 input manifest')
        rows = bench.rows(args.holdout_run/'rows.json')
        if len(rows)!=N or len({r['key'] for r in rows})!=N:
            raise ValueError('Actual B600 unique rows required; no fabricated601st case')
        if {tuple(r[k] for k in bench.IDENTITY) for r in rows}&{tuple(r[k] for k in bench.IDENTITY) for r in fresh}:
            raise ValueError('B exact identities overlap fresh600')
        index = {r['key']:r for r in source}
        for row in rows:
            if row['key'] not in index or any(row[k]!=index[row['key']][k] for k in bench.IDENTITY):
                raise ValueError('B exact episode differs from1200 manifest')
        records = document(args.holdout_run/'receipts.json')
        if len(records)!=N or any('error' in r for r in records):
            raise ValueError('B native field receipts incomplete')
        config.update(cohort='holdout600', holdout_run=str(args.holdout_run.resolve()), eligible_arms=eligible,
                      native_receipts={r['key']:r for r in records}, ladder_receipts={},
                      tokens=bind(args.holdout_run/'tokens.npz'), rows=bind(args.holdout_run/'rows.json'),
                      B_report=bind(args.holdout_run/'report.json'), B_receipts=bind(args.holdout_run/'receipts.json'),
                      fresh_C_result=bind(args.fresh_result))
    if len(rows)!=N or set(config['native_receipts'])!={r['key'] for r in rows}:
        raise ValueError('Complete exact600 native endpoint/auxiliary receipts required')
    for row in rows:
        for name in ('feature_export','packet_export'):
            row[name] = str(bench.resolve(row[name], args.input_base or args.input_manifest.parent).resolve())
    config['ordered_rows'] = rows
    return config


def verify_config(config):
    for name in ('A_report','A_aux_report','A_config','A_ladder_receipts','input_manifest','original_fresh_rows',
                 'original_fresh_tokens','tokens','rows','B_report','B_receipts','fresh_C_result'):
        if name in config:
            item = config[name]
            require_hash(item['path'],item['sha256'])
    require_hash(config['RGB_transform']['path'], config['RGB_transform']['sha256'])
    for name, digest in config['a_source_hashes'].items():
        require_hash(Path(config['a_run'])/'source'/name,digest)
    require_hash(Path(config['out'])/'source/scripts/bench_pro_repairs.py',config['script_sha256'])
    require_hash(__file__,config['script_sha256'])


def run(args):
    # Thread limits are installed BEFORE NumPy/Torch/frozen imports.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]=str(args.threads)
    config = make_config(args)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'config.json'
    if path.exists():
        if document(path)!=config:
            raise ValueError('Refuse source/input/arm changes in an existing C run')
    else:
        if any(args.out.iterdir()):
            raise ValueError('New C run needs an empty output directory')
        write(path,config)
        source = args.out/'source/scripts/bench_pro_repairs.py'
        source.parent.mkdir(parents=True)
        source.write_bytes(Path(__file__).read_bytes())
        (args.out/'fields').mkdir()
        (args.out/'receipts').mkdir()
        write(args.out/'manifest.json',dict(state='SOURCE_READY_NOT_RUN',
            source=bind(source), actual_A_source_hashes=config['a_source_hashes'], arms=config['eligible_arms'],
            zero_encoder_forwards=True, card2_full_attention='NOT_EXECUTED', source_only_author_delivery=True,
            invocation='--phase timing -> infer -> score, or all; --holdout-run B_DIR only after fresh gates pass',
            count=600, workers=args.workers, threads_per_worker=args.threads,
            baseline='saved A ladder fields / actual B FP32 RCG_s2; no new baseline solve'))
    config=dict(config,config_sha256=sha(path))
    rows=config['ordered_rows']
    verify_config(config)
    try:
        timing = timing_phase(config,rows) if args.phase in ('timing','all') else document(args.out/'timing.json')
        if timing['config_sha256']!=config['config_sha256']:
            raise ValueError('Timing report is not bound to this exact C source/input')
        if args.phase in ('infer','all') and not inference_phase(config,rows,timing):
            return 2
        if args.phase in ('score','all'):
            score_phase(config,rows)
        verify_config(config)
        write(args.out/'progress.json',dict(state='STOP_AFTER_'+args.phase.upper(),new_encoder_forwards=0))
        return 0
    except Exception as error:
        write(args.out/'error_stop.json',dict(state='FAILED_STOP',error=repr(error),traceback=traceback.format_exc(),
            fields_preserved=True,automatic_rerun=False,new_encoder_forwards=0,card2_full_attention='NOT_EXECUTED'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a-run',type=Path,required=True,help='Actual server A_six_levels_v6 with frozen source and600 field files')
    parser.add_argument('--fresh',type=Path,default=SERVER/'claude_order_fresh600')
    parser.add_argument('--input-manifest',type=Path,default=SERVER/'confirm1200_conditional_v1/inputs/manifest.json')
    parser.add_argument('--input-base',type=Path)
    parser.add_argument('--annotation-root',type=Path,default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    parser.add_argument('--rgb-root',type=Path,default=Path('/root/demo4_cache/data/COCO2014'))
    parser.add_argument('--rgb-transform-source',type=Path,required=True,help='Actual FoRIS utils/data.py build_transform bytes')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=('timing','infer','score','all'),default='all')
    parser.add_argument('--holdout-run',type=Path,help='Completed actual B600 provider; repeats only fresh-passing fixed cards')
    parser.add_argument('--fresh-result',type=Path,help='Actual fresh C report.json, required with --holdout-run')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=16 or not 1<=args.threads<=2 or args.workers*args.threads>32:
        parser.error('At most16 CPU workers x2 threads,32 total')
    if args.fresh_result is not None and args.holdout_run is None:
        parser.error('--fresh-result is only for the explicitly requested B repetition')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
