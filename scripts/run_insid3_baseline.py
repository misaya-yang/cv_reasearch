#!/usr/bin/env python3
"""Replay frozen episodes with complete upstream INSID3 or FoRIS baselines.

MPS adapts only encoder placement. Public set_reference/set_target/segment and
bilinear/1024 CLI defaults remain intact; query labels are never opened here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO/'scripts'))
from frozen_dino_plan import locked_run, prediction_file, read_rows, write
from ics.official_data import file_hash, load_inputs


def build_model(assets, device, baseline='insid3'):
    import torch
    from ics.data import TimmDINOv3
    from ics.native_basis import reuse_native_basis
    if baseline == 'foris':
        from ics.m4_crf import install
        from ics.foris import build_host
        install(assets/'third_party/crf_source', assets/'runtime/macos/crf')
        host = build_host(dict(projection_basis=str(assets/'native_assets/positional_basis.pt')),
                          'cpu', str(assets/'third_party/foris_official'),
                          weights=str(assets/'demo4_cache/models/dinov3-vitl16-timm'))
    else:
        source = assets/'third_party/INSID3'
        sys.path.insert(0, str(source))
        from models.insid3 import INSID3
        if Path(sys.modules['models.insid3'].__file__).resolve() != source/'models/insid3.py':
            raise ValueError('INSID3 source shadowed by another models package')
        encoder = TimmDINOv3(assets/'demo4_cache/models/dinov3-vitl16-timm').eval().requires_grad_(False)
        with reuse_native_basis(INSID3, assets/'native_assets/positional_basis.pt'):
            host = INSID3(encoder, image_size=1024, svd_components=500, tau=.6,
                          merge_threshold=.2, mask_refiner='bilinear', crf_size=640,
                          resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
    host.encoder.to(device)

    def extract(imgs):
        b, t = imgs.shape[:2]
        maps = host.encoder.get_intermediate_layers(
            imgs.reshape(b*t, *imgs.shape[2:]).to(device), n=1, reshape=True)[0]
        return maps.cpu().reshape(b, t, *maps.shape[1:])
    host._extract_features = extract
    return host


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--datasets', default='coco,lvis,pascal_part,paco_part,suim')
    p.add_argument('--split', choices=['smoke', 'dev', 'val', 'official'], required=True)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    p.add_argument('--baseline', choices=['insid3', 'foris'], default='insid3')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.assets, a.manifest, a.out = (x.resolve() for x in (a.assets, a.manifest, a.out))
    import numpy as np
    import torch
    from run_m4_baselines import render
    torch.set_num_threads(2)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    rows = [r for r in read_rows(a.manifest) if r['dataset'] in a.datasets.split(',')]
    if not rows or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('Require a nonempty unique draw-ID cohort')
    prepared_path = a.manifest.parent/'prepared.json'
    prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else None
    required_state = 'OFFICIAL_MANIFEST_FROZEN' if a.split == 'official' else 'FRESH_MANIFEST_FROZEN'
    if a.split != 'smoke' and (not prepared or prepared['state'] != required_state):
        raise ValueError('Require the correct frozen official-loader manifest')
    if a.split in ('dev', 'val') and prepared['split_role'] != a.split:
        raise ValueError('Prepared split differs')
    source = a.assets/('third_party/INSID3' if a.baseline == 'insid3' else 'third_party/foris_official')
    arm = 'insid3' if a.baseline == 'insid3' else 'foris.crf'
    files = [Path(__file__), REPO/'scripts/frozen_dino_plan.py', REPO/'scripts/run_m4_baselines.py',
             *sorted((REPO/'src/ics').rglob('*.py')), *sorted(source.rglob('*.py'))]
    if a.baseline == 'foris':
        files += sorted((a.assets/'third_party/crf_source/src/CRF').glob('*.py'))
        cpu_source = a.assets/'third_party/crf_source/src/PermutohedralFiltering/source/cpu'
        files += sorted(cpu_source.rglob('*.cpp'))+sorted(cpu_source.rglob('*.h'))
    config = dict(candidate_id=arm+' baseline', split_role=a.split, arms=[arm],
                  assets=str(a.assets), datasets=a.datasets.split(','), prepared_protocol=prepared,
                  branch=subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip(),
                  commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  manifest_sha256=file_hash(a.manifest), encoder_device=a.device, encoder_dtype='float32',
                  mask_refiner='bilinear (official CLI default)' if a.baseline == 'insid3' else 'original FoRIS CRF CPU, unchanged parameters/10 iterations', image_size=1024, svd_components=500,
                  tau=.6, merge_threshold=.2 if a.baseline == 'insid3' else None, resize_to_orig_size=False,
                  weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
                  basis_sha256=file_hash(a.assets/'native_assets/positional_basis.pt'),
                  source_sha256={str(f): file_hash(f) for f in files}, query_mask_in_inference=False)
    with locked_run(a.out):
        if (a.out/'sealed.json').exists():
            old = json.loads((a.out/'config.json').read_text())
            if not a.resume or any(old[k] != config[k] for k in ('manifest_sha256', 'datasets', 'split_role')):
                raise ValueError('Existing sealed run requires original inputs and --resume')
            print('Existing sealed INSID3 predictions retained', flush=True)
            return
        if (a.out/'config.json').exists():
            old = json.loads((a.out/'config.json').read_text())
            if not a.resume or {k:v for k,v in old.items() if k != 'commit'} != {k:v for k,v in config.items() if k != 'commit'}:
                raise ValueError('Cannot resume with a different implementation/input')
        else:
            write(a.out/'config.json', config); write(a.out/'manifest.json', rows)
            for f in files:
                dest = a.out/'source'/('repo' if f.is_relative_to(REPO) else 'assets')/f.relative_to(REPO if f.is_relative_to(REPO) else a.assets)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, dest)
        progress = a.out/'inference.jsonl'
        done = read_rows(progress, repair_partial_last=True) if progress.exists() else []
        completed = {r['episode_id'] for r in done}
        for row in done:
            if file_hash(a.out/'predictions'/row['prediction_file']) != row['prediction_sha256']:
                raise ValueError('Committed INSID3 prediction changed')
        (a.out/'predictions').mkdir(exist_ok=True)
        host = build_model(a.assets, a.device, a.baseline)
        with torch.inference_mode(), progress.open('a', buffering=1) as log:
            for row in rows:
                if row['episode_id'] in completed:
                    continue
                start = time.monotonic()
                r, mask, q = load_inputs(row, a.assets)
                sampling_rng = np.random.get_state()
                try:
                    host.set_reference(r, mask); host.set_target(q)
                    prediction = host.segment().reshape(1024, 1024).bool().cpu().numpy()
                finally:
                    if hasattr(host, 'reset_state'):
                        host.reset_state()
                    else:
                        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
                if not all(np.array_equal(x, y) for x, y in zip(sampling_rng, np.random.get_state())):
                    raise RuntimeError('INSID3 inference altered official sampling RNG')
                if prediction.shape != (1024, 1024):
                    raise ValueError('Expected official INSID3 1024 output frame')
                shape = (q.height, q.width)
                file = prediction_file(row)
                np.savez_compressed(a.out/'predictions'/file, original_hw=np.array(shape),
                                    **{'cli/'+arm: np.packbits(prediction),
                                       'original/'+arm: np.packbits(render(prediction, shape))})
                record = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'],
                              prediction_file=file, prediction_sha256=file_hash(a.out/'predictions'/file),
                              inference_seconds=time.monotonic()-start)
                log.write(json.dumps(record)+'\n'); completed.add(row['episode_id'])
                print(json.dumps(dict(n=len(completed), total=len(rows), seconds=record['inference_seconds'])), flush=True)
        write(a.out/'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=len(rows),
               manifest_sha256=file_hash(a.out/'manifest.json'), config_sha256=file_hash(a.out/'config.json'),
               inference_index_sha256=file_hash(progress), query_labels_opened=False,
               utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
