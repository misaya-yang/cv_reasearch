#!/usr/bin/env python3
"""One finite full-FoRIS stopping experiment; existing assets, no training.

All variants use the same original score, continuous interpolation, source
CRF and original-size evaluator. Features are discarded after each episode.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARMS = ('foris_public', 'midpoint_replay', 'scalar_contrast', 'dino_boundary', 'rgb_boundary')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def scalar_contrast_height(field):
    """Exactly Claude's frozen ring2/61-level scalar control, returning t."""
    import numpy as np
    from scipy import ndimage
    best, value = .5, -1.
    for threshold in np.linspace(.3, .9, 61):
        mask = field > threshold
        if mask.sum() < 4 or mask.sum() > .9*mask.size:
            continue
        ring = ndimage.binary_dilation(mask, iterations=2) & ~mask
        if ring.any():
            contrast = field[mask].mean()-field[ring].mean()
            if contrast > value:
                value, best = contrast, float(threshold)
    return best


def run(args):
    if not args.allow_gpu or os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise RuntimeError('Own finite GPU resource guard required')
    plan = json.loads(args.plan.read_text())
    guard = json.loads(args.guard_state.read_text())
    if guard.get('state') != 'GPU_RUNNING' or guard.get('last_event', {}).get('stage') != plan['stage_name']:
        raise RuntimeError('Wrong owned stage')
    parent = json.loads(Path(plan['parent_manifest']).read_text())
    if digest(plan['parent_manifest']) != plan['parent_manifest_sha256']:
        raise ValueError('Frozen source/data manifest drift')
    for path, sha in plan['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Prepared source drift')
    for item in parent['assets']:
        current = Path(item['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('Existing asset drift')
    if args.out.exists():
        raise ValueError('Fresh own output required; no repeated matrix')
    args.out.mkdir(parents=True)
    report = dict(schema='query_boundary_experiment_v1', state='RUNNING', arms=list(ARMS),
                  records=[], scope=plan['scope'], expected_rows=len(parent['frozen_episodes']),
                  expected_folds=sorted({r['fold'] for r in parent['frozen_episodes']}),
                  features_written=False, training=False, downloads=False)
    start = time.monotonic()
    def write():
        report['elapsed_seconds'] = time.monotonic()-start
        save(args.out/'report.json', report)
    write()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        sys.path.insert(0, str(BASE))
        sys.path.insert(0, parent['foris_root'])
        sys.path.insert(0, '/root/autodl-tmp/demo4')
        from models.foris import FoRIS
        from icx.common import TimmDINOv3
        from tics.native_assets import reuse_native_basis
        from tics.native_decision_trace import trace_native_decisions
        from native_rice_core_experiment import capture_native
        from query_boundary_cut import minimum_mean_boundary_cut, grid_edges
        if not torch.cuda.is_available():
            raise RuntimeError('No real CUDA; no model allocation')
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.45)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        encoder = TimmDINOv3().cuda().eval().requires_grad_(False)
        with reuse_native_basis(FoRIS, parent['projection_basis']['path']):
            host = FoRIS(encoder=encoder, image_size=1024, svd_components=500, tau=.6,
                         mask_refiner='crf', resize_to_orig_size=False, device='cuda').eval().requires_grad_(False)
        def edge_weights(features):
            features = F.normalize(features, dim=0)
            horizontal = (features[:, :, :-1]*features[:, :, 1:]).sum(0).flatten()
            vertical = (features[:, :-1]*features[:, 1:]).sum(0).flatten()
            return torch.cat([horizontal, vertical]).double().cpu().numpy()
        def packed(mask):
            bits = np.packbits(mask.reshape(-1)).tobytes()
            return dict(shape=list(mask.shape), codec='zlib_np_packbits_big',
                        data=base64.b64encode(zlib.compress(bits, 6)).decode())
        with torch.no_grad():
            for row in parent['frozen_episodes']:
                began = time.monotonic()
                support = Image.open(Path(parent['data_root'])/row['support']).convert('RGB')
                query = Image.open(Path(parent['data_root'])/row['query']).convert('RGB')
                support_gold = torch.from_numpy((np.asarray(Image.open(
                    Path(parent['annotation_root'])/Path(row['support']).with_suffix('.png')))==row['c']+1).copy())
                host.set_reference(support, support_gold)
                host.set_target(query)
                query_rgb = host._tgt_image[None].clone()  # Original NCHW CRF interface
                with capture_native(host) as packet, trace_native_decisions(host) as trace:
                    native = host.segment().reshape(1024, 1024).bool().clone()
                identity = f"{row['fold']}:{row['e']}"
                baseline_bits = np.packbits(native.cpu().numpy().reshape(-1)).tobytes()
                expected_mask = plan['native_mask_sha256'].get(identity)
                if plan['require_all_native_identity'] and expected_mask is None:
                    raise RuntimeError('Missing required existing complete control mask')
                if expected_mask is not None and hashlib.sha256(baseline_bits).hexdigest() != expected_mask:
                    raise RuntimeError('Complete public source differs from the existing matched mask')
                score = trace['maps']['part4_score'].cuda().float()
                if score.shape != (64, 64):
                    raise RuntimeError('Native score grid drift')
                normalized = (score-score.min())/(score-score.min()).max().clamp_min(1e-6)
                field = score.double().cpu().numpy()
                sn = (field-field.min())/max(float(field.max()-field.min()), 1e-6)
                query_features = packet['_extract_features'][0, -1]
                native_affinity = edge_weights(query_features)
                # Ordinary colour-boundary control: maximize mean squared
                # RGB jump; no kernel bandwidth and no black-pixel cosine collapse.
                image01 = host._transform(query).cuda()
                from utils.data import denormalize
                rgb_grid = F.interpolate(denormalize(image01)[None].clamp(0, 1), (64, 64), mode='area')[0]
                rgb_horizontal = -((rgb_grid[:, :, :-1]-rgb_grid[:, :, 1:])**2).sum(0).flatten()
                rgb_vertical = -((rgb_grid[:, :-1]-rgb_grid[:, 1:])**2).sum(0).flatten()
                rgb_affinity = torch.cat([rgb_horizontal, rgb_vertical]).double().cpu().numpy()
                dino = minimum_mean_boundary_cut(field, native_affinity)
                rgb = minimum_mean_boundary_cut(field, rgb_affinity)
                thresholds = dict(midpoint_replay=.5, scalar_contrast=scalar_contrast_height(sn),
                                  dino_boundary=dino['normalized_threshold'], rgb_boundary=rgb['normalized_threshold'])
                model_predictions = {'foris_public': native}
                continuous = F.interpolate(normalized[None, None], (1024, 1024),
                                            mode='bilinear', align_corners=False)[0, 0]
                for arm, threshold in thresholds.items():
                    mask = continuous > threshold
                    model_predictions[arm] = host._finalize_mask(mask, query_rgb).reshape(1024, 1024).bool().clone()
                if not torch.equal(model_predictions['midpoint_replay'], native):
                    raise RuntimeError('Same .5 score/upsampling/NCHW CRF replay differs')
                original_size = (query.height, query.width)
                originals = {arm: (F.interpolate(mask[None, None].float(), original_size,
                    mode='bilinear', align_corners=False)[0, 0] > .5).cpu().numpy().copy()
                    for arm, mask in model_predictions.items()}
                frozen = {arm: mask.cpu().numpy().copy() for arm, mask in model_predictions.items()}
                torch.cuda.synchronize()
                # FIRST query annotation pixel access, after every variant freezes.
                truth = np.asarray(Image.open(Path(parent['annotation_root'])/Path(row['query']).with_suffix('.png'))) == row['c']+1
                model_truth = F.interpolate(torch.from_numpy(truth.copy()).cuda()[None, None].float(),
                                           (1024, 1024), mode='nearest')[0, 0].bool().cpu().numpy()
                def iu(mask, target):
                    return [int((mask & target).sum()), int((mask | target).sum())]
                ledger = {arm: dict(recovered_fn=int((~originals['foris_public'] & mask & truth).sum()),
                                   lost_tp=int((originals['foris_public'] & ~mask & truth).sum()),
                                   added_fp=int((~originals['foris_public'] & mask & ~truth).sum()),
                                   removed_fp=int((originals['foris_public'] & ~mask & ~truth).sum()))
                          for arm, mask in originals.items() if arm != 'foris_public'}
                # Small decision/edge trace, never frozen feature maps.
                trace_path = args.out/f"trace_{row['fold']}_{row['e']}_{row['c']}.npz"
                np.savez_compressed(trace_path, score=sn, dino_affinity=native_affinity,
                    rgb_affinity=rgb_affinity, truth_coverage=model_truth.reshape(64, 16, 64, 16).mean((1, 3)))
                dino.pop('mask'); rgb.pop('mask')
                report['records'].append(dict(**row,
                    original_iu={arm: iu(mask, truth) for arm, mask in originals.items()},
                    model_iu={arm: iu(mask, model_truth) for arm, mask in frozen.items()},
                    prediction_bits={arm: packed(mask) for arm, mask in frozen.items()},
                    ledger=ledger, thresholds=thresholds, dino_cut=dino, rgb_cut=rgb,
                    exact_complete_public_baseline=True, midpoint_replay_exact=True,
                    query_GT_after_all_predictions=True, full_query_RGB_CRF_NCHW=True,
                    trace_archive=str(trace_path.name), trace_sha256=digest(trace_path),
                    elapsed_seconds=time.monotonic()-began))
                write()
                print(json.dumps(dict(fold=row['fold'], e=row['e'], original_iu=report['records'][-1]['original_iu'],
                                      thresholds=thresholds)), flush=True)
                del packet, trace, query_features, model_predictions
        report['state'] = 'COMPLETED'
        write()
    except Exception as exc:
        report.update(state='ERROR', error=type(exc).__name__+': '+str(exc))
        write()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--guard-state', type=Path, required=True)
    parser.add_argument('--allow-gpu', action='store_true')
    run(parser.parse_args())
