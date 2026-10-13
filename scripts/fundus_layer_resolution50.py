"""Paired 50-task Fundus pilot: original U/T and complete FoRIS, 3 layers x 2 sizes.

All predictions precede scoring. Layers share one fresh forward per RGB/size.
The 2048 extension changes grid dimensions, not U's numerical parameters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.special import expit
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/fundus_layer_resolution50_20261013'
WEIGHTS = ASSETS / 'demo4_cache/models/dinov3-vitl16-timm'
PARENT = ASSETS / 'a/x1012s2_confirm1000/manifest.json'
LAYERS = (19, 20, 23)
SIZES = (1024,)  # User cancelled2048 while the1024 cohort was still running.
SEED = 2026101303


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key(b):
    return str(b['index'])


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'config.json').exists():
        c = read(OUT / 'config.json')
        assert c['source_sha256'] == sha(__file__)
        assert c['manifest_sha256'] == sha(OUT / 'manifest.json')
        return read(OUT / 'manifest.json'), c
    pool = [b for b in read(PARENT) if b['dataset'] == 'fundus']
    chosen = np.random.default_rng(SEED).choice(len(pool), 50, replace=False)
    rows = [pool[int(i)] for i in sorted(chosen)]
    assert len(rows) == len({key(b) for b in rows}) == 50
    assert all(b['row']['reference_size_hw'] == b['row']['query_size_hw'] == [2048, 2048] for b in rows)
    write(OUT / 'manifest.json', rows)
    c = {
        'state': 'FIXED_50_PAIRED_TASKS_BEFORE_NEW_PREDICTIONS', 'n': 50,
        'selection': 'seed2026101303 sample without replacement from the100 existing Fundus confirmation pairs; no score/case selection',
        'development_exposure': True, 'independent_confirmation': False,
        'layer_indices_zero_based': LAYERS, 'input_sizes': SIZES, 'methods': ['T', 'U', 'FoRIS'],
        'encoding': 'fresh frozen DINOv3-L16 FP32 MPS; RGBs encoded one at a time; layers19/20/23 from the same forward',
        'positional_basis': 'same native normalized-black-image/column-normalize/center/CPU torch.linalg.svd recipe,rank500,separately for each size/layer; no query or labels',
        'U': 'original fixed rho>.9 T fallback; otherwise original9strength max-J modulation,V=.20,P=.25,mutual20NN graph,lambda16,reliability weights; dimensions generalized to64or128 grid',
        'T': 'unconditional positional projection,common R/Q mean centering,temp.05 soft-coverage kernel,bilinear to input size,minmax,strict>.5',
        'FoRIS': 'unchanged official complete semantic pipeline plus original input-size-scaled CRF; same fresh features and positional basis as T/U',
        'primary_evaluation': 'common original2048 grid for every arm; native predictions nearest-resized to original2048 before I/U',
        'secondary_evaluation': 'commonCLI1024 grid; every native prediction and query GT nearest-resized to1024',
        'metric': 'sum I/sum U single foreground class; also episode mean IoU; paired canonical query-photo bootstrap1000repeats,seed2026101304',
        'timing': 'shared R+Q preprocessing/encoding/CPU transfer plus each method actual head; basis/model setup and artifact writes separate; all three intermediate outputs extracted together',
        'training': False, 'query_GT_in_inference': False, 'CPU_threads': 2, 'GPU_workers': 1,
        'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT / 'manifest.json'),
        'parent_manifest_sha256': sha(PARENT), 'weights_sha256': sha(WEIGHTS / 'model.safetensors'),
        'rationale': 'FoRIS published Fundus layer/resolution sweeps motivate testing our fixed U/T. Higher paired accuracy supports this representation change; no gain rejects it for the current readout, without retuning.',
    }
    write(OUT / 'config.json', c)
    (OUT / 'frozen').mkdir(exist_ok=True)
    shutil.copyfile(__file__, OUT / 'frozen/fundus_layer_resolution50.py')
    print({'state': c['state'], 'n': 50, 'layers': LAYERS, 'sizes': SIZES}, flush=True)
    return rows, c


@contextmanager
def inference_only(b):
    import builtins
    original = builtins.open, io.open, Image.open
    row = b['row']
    allowed = {Path(row['reference_mask_path']).resolve(), (ASSETS / row['reference_annotation_path']).resolve()}
    denied = {Path(row['query_mask_path']).resolve(), (ASSETS / row['query_annotation_path']).resolve()} - allowed
    def wrap(fn):
        def guard(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() in denied:
                raise RuntimeError('Query GT forbidden in natural inference')
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Image.open = map(wrap, original)
    try:
        with torch.inference_mode():
            yield
    finally:
        builtins.open, io.open, Image.open = original


def normalized(x):
    return F.normalize(x, dim=-1)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def readout(field, size):
    x = F.interpolate(torch.as_tensor(field).float()[None, None], (size, size), mode='bilinear', align_corners=False)[0, 0]
    return (minmax(x) > .5).numpy()


class MapReplay(torch.nn.Module):
    def bind(self, inputs, maps):
        self.values = {hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest(): m for v, m in zip(inputs, maps)}

    def get_intermediate_layers(self, inputs, n=1, reshape=True):
        assert n == 1 and reshape and inputs.device.type == 'cpu'
        return [torch.stack([self.values[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()] for x in inputs])]


def modules():
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/x1012'), str(ASSETS / 'third_party/foris_official')]
    import plan_common as pc
    native = pc.extension('ics_permutohedral_m4', ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so')
    prepared = pc.extension('ics_permutohedral_m4_prepared_v1', ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')
    class Layer(torch.nn.Module):
        def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
            super().__init__()
            self.bilateral, self.parameters_ = bilateral, (theta_alpha, theta_beta, theta_gamma)
        def forward(self, value, image):
            value = value.detach().float().permute(0, 2, 3, 1).contiguous()
            image = image.detach().float().permute(0, 2, 3, 1).contiguous()
            return native.forward(value, image, self.bilateral, *self.parameters_).permute(0, 3, 1, 2).contiguous()
    shim = types.ModuleType('PermutohedralFiltering')
    shim.PermutohedralLayer = Layer
    sys.modules['PermutohedralFiltering'] = shim
    sys.path.insert(0, str(ASSETS / 'third_party/crf_source/src'))
    from models.foris import FoRIS
    from utils.data import build_transform
    from ics.data import TimmDINOv3
    from ics.m4_crf_cached import enable_on_crf
    return FoRIS, build_transform, TimmDINOv3, enable_on_crf, prepared


def encode(encoder, value):
    torch.mps.synchronize()
    began = time.perf_counter()
    x = value[None].to('mps')
    maps = encoder.get_intermediate_layers(x, n=list(LAYERS), reshape=True)
    torch.mps.synchronize()
    outputs = [m[0].cpu().contiguous() for m in maps]
    torch.mps.synchronize()
    elapsed = time.perf_counter() - began
    del x, maps
    torch.mps.empty_cache()
    assert len(outputs) == 3
    return outputs, elapsed


def make_bases(encoder, transform, size):
    path = OUT / 'basis' / f'{size}.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)['bases']
    black = transform(Image.new('RGB', (size, size), 0))
    maps, cost = encode(encoder, black)
    bases, times = {}, {}
    for layer, fmap in zip(LAYERS, maps):
        began = time.perf_counter()
        matrix = F.normalize(fmap[None], dim=1)[0].flatten(1)
        matrix = matrix - matrix.mean(1, keepdim=True)
        u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        bases[layer] = u[:, :500].contiguous()
        times[str(layer)] = time.perf_counter() - began
        del matrix, u
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'bases': bases, 'size': size, 'layers': list(LAYERS), 'source': 'native_normalized_black'}, path)
    write(path.with_suffix('.json'), {'size': size, 'encoder_seconds': cost, 'svd_seconds': times, 'sha256': sha(path)})
    print({'phase': 'basis_ready', 'size': size, 'encoder_seconds': cost, 'svd_seconds': times}, flush=True)
    return bases


def kernel(r, q, cov, need_reverse):
    """Exact kernel definition with blockwise matrix evaluation for bounded memory."""
    mu = .5 * (r.mean(0) + q.mean(0))
    rr, qq = normalized(r - mu), normalized(q - mu)
    simmax = torch.empty(len(q))
    maximum = torch.tensor(-float('inf'))
    for lo in range(0, len(q), 512):
        s = qq[lo:lo + 512] @ rr.T
        simmax[lo:lo + len(s)] = s.max(1).values
        maximum = torch.maximum(maximum, s.max())
    c, log_e = torch.empty(len(q)), torch.empty(len(q))
    reverse_den = torch.zeros(len(r)) if need_reverse else None
    for lo in range(0, len(q), 512):
        s = qq[lo:lo + 512] @ rr.T
        k = torch.exp((s - maximum) / .05)
        fg, bg = k @ cov, k @ (1 - cov)
        evidence = fg + bg
        c[lo:lo + len(s)] = fg / evidence.clamp_min(1e-30)
        log_e[lo:lo + len(s)] = evidence.clamp_min(1e-30).log() + maximum / .05
        if need_reverse:
            reverse_den += k.sum(0)
    back = None
    if need_reverse:
        back = torch.empty(len(q))
        for lo in range(0, len(q), 512):
            k = torch.exp(((qq[lo:lo + 512] @ rr.T) - maximum) / .05)
            back[lo:lo + len(k)] = (k / reverse_den.clamp_min(1e-30)) @ cov / cov.sum().clamp_min(1e-6)
    return c, simmax, log_e, back


def u_nonfallback(r, q, cov, mask, image, c, simmax, log_e, back, host, size, grid):
    import lab
    unit = normalized(q)
    labels = (unit @ lab.kmeans(unit, 64, iters=6).T).argmax(1)
    onehot = F.one_hot(labels, 64).float()
    region = lambda x: onehot @ ((onehot.T @ x) / onehot.sum(0).clamp_min(1))
    novelty = simmax.numpy().astype(np.float16).astype(np.float32)
    novelty = (novelty - novelty.min()) / max(float(novelty.max() - novelty.min()), 1e-6)
    back = back.numpy().astype(np.float16).astype(np.float64)
    strengths, scores, candidates = np.linspace(0, 1, 9), [], []
    # Original R1 endpoint quality uses pre-FP16 side-field readouts. Its
    # selected modulation tokens still use the FP16 novelty side field.
    endpoints = {0.: readout(region(c).reshape(grid, grid), size),
                 1.: readout(region(c * minmax(simmax)).reshape(grid, grid), size)}
    for strength in strengths:
        value = torch.from_numpy(c.numpy() * np.power(novelty, strength))
        area_field = region(value).reshape(grid, grid)
        area_mask = endpoints[strength] if strength in endpoints else readout(area_field, size)
        active = area_mask.reshape(grid, 16, grid, 16).mean(axis=(1, 3)).ravel() > .5
        if active.any():
            precision, capture = float(c.numpy()[active].mean()), float(back[active].sum())
            score = precision * capture / max(precision + capture - precision * capture, 1e-12)
        else:
            score = 0.
        scores.append(score)
        candidates.append(value)
    chosen = int(np.argmax(scores))
    modified = candidates[chosen]
    rf, qf = r.T.reshape(1, 1, 1024, grid, grid).contiguous(), q.T.reshape(1, 1024, grid, grid).contiguous()
    proto = normalized((r * cov[:, None]).sum(0))
    host._tgt_image = image
    try:
        picks, votes = host._locate_candidates(rf, qf, mask[None, None].float(), proto, 1, grid, grid)
        dense = torch.zeros((grid, grid), dtype=torch.bool)
        dense[::2, ::2] = True
        prior = host._build_seed_cluster_prior(rf, qf, mask[None, None].float(), picks | dense, 1, grid, grid)
    finally:
        host._tgt_image = None
    primary = modified.numpy().reshape(grid, grid) + (.20 * (votes.numpy() - .5) + .25 * (prior.numpy() - .5))
    neighbors = np.empty((len(q), 20), np.uint16)
    values = np.empty((len(q), 20), np.float32)
    for lo in range(0, len(q), 512):
        sim = unit[lo:lo + 512] @ unit.T
        ids = torch.arange(len(sim))
        sim[ids, ids + lo] -= 1e9
        v, ix = sim.topk(20, dim=1)
        neighbors[lo:lo + len(sim)] = ix.numpy()
        # Preserve original graph's edge dot-product evaluation rather than
        # substituting the numerical GEMM values used only to select neighbors.
        values[lo:lo + len(sim)] = (unit[lo:lo + len(sim), None] * unit[ix]).sum(-1).numpy()
    distance = np.maximum(1 - values, 0)
    weights = np.exp(-distance / np.maximum(distance[:, -1:], 1e-6)).ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(len(q)), 20), neighbors.ravel())), shape=(len(q), len(q)))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    w /= max(float(np.asarray(w.sum(1)).ravel().mean()), 1e-8)
    laplacian = sparse.diags(np.asarray(w.sum(1)).ravel()) - w
    s = minmax(torch.from_numpy(primary).float().flatten()).numpy()
    reliability = expit(log_e.numpy().astype(np.float16).astype(np.float64) - np.median(log_e.numpy().astype(np.float16).astype(np.float64))).astype(np.float32)
    a = (.1 + np.abs(2 * s - 1)) * reliability
    a = (a / a.mean()).astype(np.float64)
    h, rhs = sparse.diags(a) + 16 * laplacian, a * s.astype(np.float64)
    z, status = cg(h, rhs, x0=s.astype(np.float64), rtol=1e-7, atol=1e-9, maxiter=300)
    assert status == 0 and np.isfinite(z).all()
    return readout(z.reshape(grid, grid).astype(np.float32), size), {'selected_s': float(strengths[chosen]), 'J_grid': scores}


def our_prediction(ref, query, basis, ref_mask, image, host, size):
    grid = size // 16
    r_raw, q_raw = ref.flatten(1).T, query.flatten(1).T
    r, q = normalized(r_raw), normalized(q_raw)
    r, q = normalized(r - (r @ basis) @ basis.T), normalized(q - (q @ basis) @ basis.T)
    cov = F.avg_pool2d(ref_mask[None, None].float(), 16)[0, 0].flatten()
    rho = float(1 - (cov * (cov >= .9)).sum() / cov.sum().clamp_min(1e-9))
    c, simmax, log_e, back = kernel(r, q, cov, rho <= .9)
    t = readout(c.reshape(grid, grid), size)
    if rho > .9:
        return {'T': t, 'U': t}, {'reference_rho': rho, 'T_fallback': True}
    u, detail = u_nonfallback(r, q, cov, ref_mask, image, c, simmax, log_e, back, host, size, grid)
    return {'T': t, 'U': u}, {'reference_rho': rho, 'T_fallback': False, **detail}


def run():
    rows, config = prepare()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    FoRIS, build_transform, Encoder, enable, backend = modules()
    tick = time.perf_counter()
    encoder = Encoder(str(WEIGHTS)).to('mps').eval().requires_grad_(False)
    torch.mps.synchronize()
    write(OUT / 'model_setup.json', {'seconds': time.perf_counter() - tick, 'device': 'MPS', 'weights_sha256': config['weights_sha256']})
    all_records, wall = [], time.perf_counter()
    for size in SIZES:
        transform = build_transform(size)
        bases = make_bases(encoder, transform, size)
        replay = MapReplay()
        original = FoRIS._build_positional_basis
        FoRIS._build_positional_basis = lambda this, device: bases[23].clone()
        try:
            host = FoRIS(encoder=replay, image_size=size, svd_components=500, tau=.6,
                         mask_refiner='crf', resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
        finally:
            FoRIS._build_positional_basis = original
        enable(host._crf, backend)
        for number, b in enumerate(rows, 1):
            record_path = OUT / 'records' / f'{size}_{key(b)}.json'
            if record_path.exists():
                record = read(record_path)
                assert sha(record['prediction_path']) == record['prediction_sha256']
                all_records.append(record)
                continue
            with inference_only(b):
                started = time.perf_counter()
                with Image.open(ASSETS / b['row']['reference_path']) as f:
                    ref_img = f.convert('RGB')
                with Image.open(ASSETS / b['row']['query_path']) as f:
                    query_img = f.convert('RGB')
                with Image.open(b['row']['reference_mask_path']) as f:
                    ref_mask_native = torch.from_numpy(np.asarray(f.convert('L')).copy() > 0)
                inputs = [transform(ref_img), transform(query_img)]
                mask = F.interpolate(ref_mask_native[None, None].float(), (size, size), mode='nearest')[0, 0] > .5
                load_seconds = time.perf_counter() - started
                r_maps, r_seconds = encode(encoder, inputs[0])
                q_maps, q_seconds = encode(encoder, inputs[1])
                masks, details = {}, {}
                for layer, ref, query in zip(LAYERS, r_maps, q_maps):
                    host.positional_basis = bases[layer]
                    replay.bind(inputs, (ref, query))
                    began = time.perf_counter()
                    ours, detail = our_prediction(ref, query, bases[layer], mask, inputs[1], host, size)
                    detail['our_head_seconds'] = time.perf_counter() - began
                    masks.update({f'{method}_L{layer}': np.packbits(value) for method, value in ours.items()})
                    began = time.perf_counter()
                    host.set_reference(ref_img, ref_mask_native)
                    host.set_target(query_img)
                    final = host.segment().numpy()
                    detail['FoRIS_head_seconds'] = time.perf_counter() - began
                    masks[f'FoRIS_L{layer}'] = np.packbits(final)
                    details[str(layer)] = detail
                output = OUT / 'predictions' / f'{size}_{key(b)}.npz'
                save_npz(output, **masks)
            record = {'index': key(b), 'size': size, 'dataset': 'fundus', 'query_photo': b['row']['query_photo_id'],
                      'prediction_path': str(output), 'prediction_sha256': sha(output), 'layers': details,
                      'shared_R_Q_encode_seconds': r_seconds + q_seconds, 'RGB_load_preprocess_seconds': load_seconds,
                      'new_encoder_calls': 2, 'query_GT_reads': 0, 'seconds_including_artifacts': time.perf_counter() - started}
            write(record_path, record)
            all_records.append(record)
            write(OUT / 'progress.json', {'phase': 'natural_predictions', 'size': size, 'completed_in_size': number,
                                         'n': 50, 'total_size_tasks': len(all_records), 'elapsed_seconds': time.perf_counter() - wall,
                                         'last_shared_encoder_seconds': r_seconds + q_seconds,
                                         'last_heads': details})
            print({'phase': 'natural', 'size': size, 'completed': number, 'n': 50,
                   'encoder_seconds': round(r_seconds + q_seconds, 3),
                   'rho': details['23']['reference_rho'], 'elapsed': round(time.perf_counter() - wall, 1)}, flush=True)
            del r_maps, q_maps, ref, query, inputs, masks, final
        del host, replay, bases
    write(OUT / 'predictions_complete.json', {'state': 'ALL_REQUESTED_NATURAL_OUTPUTS_COMPLETE', 'n_pairs': 50,
                                            'records': all_records, 'query_GT_reads': 0,
                                            'source_sha256': sha(__file__), 'wall_seconds': time.perf_counter() - wall})
    del encoder
    torch.mps.empty_cache()
    score()


def resize_bool(mask, size):
    return F.interpolate(torch.from_numpy(mask)[None, None].float(), (size, size), mode='nearest')[0, 0].numpy() > .5


def score():
    rows, config = prepare()
    complete = read(OUT / 'predictions_complete.json')
    assert len(complete['records']) == len(rows) * len(SIZES)
    rm = {(r['size'], r['index']): r for r in complete['records']}
    metrics, timing = {}, {}
    for b in rows:
        with Image.open(b['row']['query_mask_path']) as f:
            gt_native = np.asarray(f.convert('L')) > 0
        assert gt_native.shape == (2048, 2048)
        targets = {'original2048': gt_native, 'CLI1024': resize_bool(gt_native, 1024)}
        for size in SIZES:
            record = rm[(size, key(b))]
            assert sha(record['prediction_path']) == record['prediction_sha256']
            with np.load(record['prediction_path'], allow_pickle=False) as z:
                for layer in LAYERS:
                    for method in ('T', 'U', 'FoRIS'):
                        arm = f'{method}_{size}_L{layer}'
                        mask = np.unpackbits(z[f'{method}_L{layer}'])[:size * size].reshape(size, size).astype(bool)
                        for frame, gt in targets.items():
                            pred = mask if mask.shape == gt.shape else resize_bool(mask, gt.shape[0])
                            i, u = int((pred & gt).sum()), int((pred | gt).sum())
                            metrics.setdefault(frame, {}).setdefault(arm, []).append({'index': key(b), 'photo': b['row']['query_photo_id'], 'I': i, 'U': u, 'IoU': 100 * i / max(u, 1)})
                        if method != 'T':
                            head = record['layers'][str(layer)]['our_head_seconds' if method == 'U' else 'FoRIS_head_seconds']
                            timing.setdefault(arm, []).append(record['RGB_load_preprocess_seconds'] + record['shared_R_Q_encode_seconds'] + head)
    rng = np.random.default_rng(2026101304)
    photos = sorted({b['row']['query_photo_id'] for b in rows})
    pm = {p: i for i, p in enumerate(photos)}
    weights = rng.multinomial(len(photos), np.full(len(photos), 1 / len(photos)), size=1000)
    summary, bootstrap, contrasts = {}, {}, {}
    for frame, arms in metrics.items():
        summary[frame], bootstrap[frame], contrasts[frame] = {}, {}, {}
        for arm, records in arms.items():
            counts = np.zeros((len(photos), 2), np.float64)
            for r in records:
                counts[pm[r['photo']]] += (r['I'], r['U'])
            resampled = weights @ counts
            boot = 100 * resampled[:, 0] / np.maximum(resampled[:, 1], 1)
            bootstrap[frame][arm] = boot
            summary[frame][arm] = {'n': 50, 'class_miou': 100 * counts[:, 0].sum() / max(counts[:, 1].sum(), 1),
                                   'episode_mean_iou': float(np.mean([r['IoU'] for r in records])),
                                   'absolute_bootstrap95': np.quantile(boot, [.025, .975]).tolist()}
        for method in ('T', 'U', 'FoRIS'):
            for size in SIZES:
                for layer in LAYERS:
                    arm = f'{method}_{size}_L{layer}'
                    references = [f'{method}_1024_L23', f'FoRIS_{size}_L{layer}']
                    if size == 2048:
                        references.append(f'{method}_1024_L{layer}')
                    for base in set(references) - {arm}:
                        contrasts[frame][arm + '_minus_' + base] = {
                            'delta_pp': summary[frame][arm]['class_miou'] - summary[frame][base]['class_miou'],
                            'paired_photo_bootstrap95_pp': np.quantile(bootstrap[frame][arm] - bootstrap[frame][base], [.025, .975]).tolist()}
    cost = {a: {'mean_seconds': float(np.mean(v)), 'median_seconds': float(np.median(v)), 'p95_seconds': float(np.quantile(v, .95))} for a, v in timing.items()}
    fallback = {str(size): sum(r['layers']['23']['T_fallback'] for r in complete['records'] if r['size'] == size) for size in SIZES}
    result = {'state': 'COMPLETE_50_PAIRED_LAYER_RESOLUTION_PILOT', 'n': 50, 'query_photo_n': len(photos),
              'summary': summary, 'contrasts': contrasts, 'timing': cost, 'fallback_n': fallback,
              'timing_scope': config['timing'], 'config_sha256': sha(OUT / 'config.json'),
              'manifest_sha256': sha(OUT / 'manifest.json'), 'prediction_complete_sha256': sha(OUT / 'predictions_complete.json'),
              'new_encoder_calls_for_dataset': sum(r['new_encoder_calls'] for r in complete['records']),
              'query_GT_reads_in_inference': 0, 'training': False, 'development_pilot': True, 'records': metrics}
    write(OUT / 'report.json', result)
    lines = ['# Fundus50：三层×两分辨率的同配对试验', '', '每行同50个配对；层索引0起。主指标统一原图2048网格，另列CLI1024。此批配对已曝光，为开发诊断。', '',
             '|输入分辨率|层|U/T主mIoU|FoRIS主mIoU|U/T CLI1024|FoRIS CLI1024|U整链秒|FoRIS整链秒|',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size in SIZES:
        for layer in LAYERS:
            u, f = f'U_{size}_L{layer}', f'FoRIS_{size}_L{layer}'
            lines.append(f'|{size}|{layer}|{summary["original2048"][u]["class_miou"]:.4f}|{summary["original2048"][f]["class_miou"]:.4f}|{summary["CLI1024"][u]["class_miou"]:.4f}|{summary["CLI1024"][f]["class_miou"]:.4f}|{cost[u]["mean_seconds"]:.3f}|{cost[f]["mean_seconds"]:.3f}|')
    lines += ['', 'U回退T的数量：' + '，'.join(f'{size}={fallback[str(size)]}/50' for size in SIZES) + '。', '',
              '准确率全部预测完成后评分；每层/分辨率使用同样原生黑图SVD位置去偏流程，未复用末层位置基座。未训练、未调参数。',
              '编码时间是一次同时取三层的真实前向与回传成本；每个方法只加自身实际头部耗时，其他层/方法的计算不计入该方法。模型/位置基座初始化、产物写盘另计。',
              '全部逐例I/U、照片簇区间、基座与计时在相邻JSON。用户取消了2048，实际只完成1024的50例三层对照。该试验不修改现行T0–T9计划，也不继续旧组合搜索。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    compact = {k: v for k, v in result.items() if k != 'records'}
    write(ROOT / 'evidence/research/fundus_layer_resolution50_20261013.json', compact)
    (ROOT / 'docs/research/FUNDUS_LAYER50_RESULT_20261013.md').write_text('\n'.join(lines) + '\n')
    print({'state': result['state'], 'main_frame': summary['original2048'], 'timing': cost}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'score'), nargs='?', default='run')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'score':
        score()
    else:
        run()
