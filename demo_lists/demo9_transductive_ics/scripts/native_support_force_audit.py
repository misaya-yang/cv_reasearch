#!/usr/bin/env python3
"""Finite support-only diagnosis of the frozen E3 pilot, no metric retraining.

Stream the original ten paired inference images; never open query masks or
write features. Diagnose the support force before the density/host interface.
--self-check is CPU-only and cannot initialize a DINO model or CUDA context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--prepared-plan', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--global-checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from unittest.mock import patch
    sys.path.insert(0, str(HERE.parent))
    if args.self_check:
        # tics/__init__ historically auto-discovers DEV; prevent CUDA probing.
        with patch.object(torch.cuda, 'is_available', side_effect=lambda: False):
            from tics.reference_force_diagnostics import self_check
            answer = self_check()
        if args.out:
            if args.out.exists():
                raise ValueError('Preserve previous checks')
            args.out.write_text(json.dumps(answer, indent=2) + '\n')
        print(json.dumps(answer)); return
    if not all((args.prepared_plan, args.checkpoint, args.global_checkpoint, args.out)):
        parser.error('Explicit frozen cohort/checkpoints/new output required')
    if os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise RuntimeError('Only a CPU-preflighted resource guard may execute')
    if args.out.exists():
        raise ValueError('Fresh audit output required; old queues stay retired')
    plan = json.loads(args.prepared_plan.read_text())
    rows = plan['rows']['inference']
    if len(rows) != 10 or plan['schema'] != 'demo9_native_metric_acquisition_v1':
        raise ValueError('Only the frozen E3 ten-task diagnostic is declared')
    source = Path(plan['demo4_root']) / 'INSID3/models/insid3.py'
    if sha(source) != plan['native_host_source_sha256']:
        raise ValueError('Native host source drift')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU; do not load the encoder')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from tics.native_assets import reuse_native_basis
    from tics.reference_metric import ReferenceMetric, local_triplets, difference_design
    from tics.reference_force_diagnostics import force_matrix, geometry, response
    from train_reference_metric import load_deployment
    sys.path.insert(0, str(Path(args.prepared_root) / 'scripts'))
    import _paths
    sys.path.insert(0, str(Path(_paths.DEMO4) / 'INSID3'))
    from models.insid3 import INSID3
    from utils.data import load_image
    sys.path.insert(0, _paths.DEMO4)
    from icx.common import TimmDINOv3
    import numpy as np
    from PIL import Image
    basis_path = Path(plan['fixed_basis']['path'])
    contract = {**plan['expected_feature_contract'], 'projection_id': 'sha256:' + sha(basis_path),
                'encoding': 'original SQ-paired native BF16 feature extraction; native normalization/debias; FP32 serialization'}
    selected, metadata = load_deployment(args.checkpoint, contract, 'cuda')
    global_model, global_metadata = load_deployment(args.global_checkpoint, contract, 'cuda')
    if metadata['variant'] != 'protected' or global_metadata['variant'] != 'fixed_global':
        raise ValueError('Frozen selected protected and independently learned global arms required')
    if global_model.config != selected.config:
        raise ValueError('Matched objective/configuration required')
    report = dict(state='RUNNING', records=[], query_GT_opened=False, feature_cache_written=False,
                  training=False, new_segmentation_score=False,
                  scope='Existing E3 ten development/inference tasks; support-only causal diagnosis, not a new test score',
                  checkpoint_sha256=sha(args.checkpoint), global_checkpoint_sha256=sha(args.global_checkpoint),
                  selected_epoch=metadata['selected_epoch'], config=metadata['config'],
                  source_sha256={str(p): sha(p) for p in (Path(__file__), HERE.parent / 'tics/reference_force_diagnostics.py', source)},
                  prepared_plan_sha256=sha(args.prepared_plan), feature_contract=contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.out)
    start = time.monotonic(); save()
    try:
        torch.manual_seed(0)
        with torch.inference_mode():
            encoder = TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(INSID3, basis_path) as receipt:
                host = INSID3(encoder=encoder, image_size=1024, svd_components=500, tau=.6,
                              merge_threshold=.2, mask_refiner='bilinear', resize_to_orig_size=False,
                              device='cuda').eval().requires_grad_(False)
            report['basis_receipt'] = receipt
            dictionaries = dict(selected=selected.directions(),
                                initialized_random=ReferenceMetric(1024, selected.config, 2048).cuda().directions(),
                                independently_learned_global=global_model.directions())
            sum_c = torch.zeros(1024, 1024, device='cuda', dtype=torch.float64)
            energies, centered_energies = [], []
            for row in rows:
                images = [Image.open(Path(plan['data_root']) / row[role]).convert('RGB') for role in ('support', 'query')]
                si, qi = [load_image(image, host._transform, 'cuda')[0] for image in images]
                # Query RGB preserves the original B2 encoding context; its annotation is NEVER opened.
                sm = torch.from_numpy((np.asarray(Image.open(Path(plan['annotation_root']) / Path(row['support']).with_suffix('.png'))) == row['classid'] + 1).copy()).cuda()
                raw = host._extract_features(torch.cat((si, qi), 0).unsqueeze(0))
                debiased = host._debias_features(F.normalize(raw, p=2, dim=2))
                grid = tuple(debiased.shape[-2:])
                if grid != (64, 64) or debiased.shape[2] != 1024:
                    raise ValueError('Original full-coordinate grid changed')
                # Same FP32 serialization-equivalent support and normalization as the original metric.
                f = F.normalize(debiased[0, 0].flatten(1).T.float(), dim=-1)
                coverage = F.interpolate(sm.float()[None, None], grid, mode='area')[0, 0].flatten()
                triples, mass = local_triplets(f, coverage, grid, selected.config)
                if not len(triples):
                    report['records'].append(dict(e=row['e'], c=row['classid'], state='NO_LEGAL_TRIPLETS', pure_fg=int((coverage >= .9).sum()), pure_bg=int((coverage <= .1).sum())))
                    save(); continue
                margins, _ = difference_design(f, dictionaries['selected'], triples)
                c, _ = force_matrix(f, triples, mass, margins, selected.config)
                r = response(f, coverage, grid, triples, mass, dictionaries['selected'], selected.config, c)
                controls = {name: geometry(c, u) for name, u in dictionaries.items() if name != 'selected'}
                report['records'].append(dict(e=row['e'], c=row['classid'], support=row['support'], query=row['query'], state='COMPLETED', selected=r, dictionary_geometry_controls=controls))
                sum_c += c
                energies.append(r['force_projection']['total_energy']); centered_energies.append(r['force_projection']['centered_energy'])
                torch.cuda.synchronize()
                report['elapsed_s'] = time.monotonic() - start; save()
                print(json.dumps(dict(count=len(report['records']), elapsed_s=report['elapsed_s'], g0=r['g0_norm'], preferred_rms=r['preferred32_rms'], alpha=r['alpha'])), flush=True)
                del si, qi, sm, raw, debiased, f, coverage, triples, mass, c
            if energies:
                mean = sum_c / len(energies)
                centered = mean - torch.trace(mean) / len(mean) * torch.eye(len(mean), device=mean.device)
                report['across_task_force'] = dict(valid_tasks=len(energies), mean_force_energy=float(mean.square().sum()),
                    mean_task_energy=sum(energies) / len(energies), mean_centered_force_energy=float(centered.square().sum()),
                    mean_task_centered_energy=sum(centered_energies) / len(energies))
        report['state'] = 'COMPLETED'; report['elapsed_s'] = time.monotonic() - start
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(); save()
    except BaseException as error:
        report.update(state='ERROR', error=repr(error), elapsed_s=time.monotonic() - start); save(); raise


if __name__ == '__main__':
    main()
