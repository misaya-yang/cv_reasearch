#!/usr/bin/env python3
"""Run the fixed C2 minimum self test only after its representation qualifies.

Q=R is encoded as an ordinary two-image pair. The scorer receives features and
three reference role indices, never the complete mask or a self-test flag.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from ics.foris import build_host, observe
from ics.official_data import array_hash, decoded_rgb, file_hash
from ics.representations import observe_branches, source_reference_roles, unit_tokens
from frozen_dino_plan import locked_run, write
from frozen_unary_candidates import first_context_roles, minimal_context

DATASETS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim')


def qualified_choice(decision):
    candidate = decision.get('candidates', {}).get('C2', {})
    if candidate.get('passed') is not True:
        return None
    choice = candidate.get('representation')
    if choice not in ('C2concat/raw', 'C2concat/deb'):
        raise ValueError('Require the already selected fixed16/24 Q/K representation')
    return choice


def evaluate_reference(host, row, assets, bases, choice, device):
    image = decoded_rgb(assets/row['reference_path'], row.get('reference_crop'))
    if array_hash(np.asarray(image)) != row['reference_rgb_hash']:
        raise ValueError('Frozen reference RGB changed')
    with Image.open(row['reference_mask_path']) as source:
        raw_mask = (np.asarray(source.convert('L')) > 0).astype(np.uint8)
    if array_hash(raw_mask) != row['reference_mask_hash']:
        raise ValueError('Frozen reference annotation changed')
    if not raw_mask.any():
        return None
    host.set_reference(image, torch.from_numpy(raw_mask.astype(bool)))
    host.set_target(image)
    try:
        if not host._ref_masks.any():
            return None
        reference_mask = host._ref_masks.unsqueeze(1)
        inputs = torch.cat([host._ref_images, host._tgt_image[None]], dim=0).to(device)
        with observe_branches(host.encoder, layers=(16, 24), branches='OQK') as bank:
            maps = host.encoder.get_intermediate_layers(inputs, n=1, reshape=True)[0].cpu()
        pair = F.normalize(maps, dim=1)[None]
        processed = host._part1_positional_debias(pair, reference_mask, 1)
        with observe(host) as got:
            host._part2_background_suppression(processed, reference_mask, 1, 64, 64)
        foreground, background = source_reference_roles(got['reference_features'][0, 0], got['reference_fg'][0])
        if not len(background):
            return None
        # This is the same rounded-and-unitized last output used by locked RCG.
        app = F.normalize(processed[0].flatten(2).transpose(1, 2).half().float(), dim=-1)
        coverage = F.interpolate(reference_mask.float(), (64, 64), mode='area')[0, 0].numpy()
        roles = first_context_roles(app[0], foreground.numpy(), background.numpy(), coverage)
        qk = {}
        for layer in (16, 24):
            for kind in ('Q', 'K'):
                key = f'{kind}/{layer}'
                qk[key] = unit_tokens(bank[key], bases[key] if choice.endswith('/deb') else None)
        result = minimal_context(app[0], app[1], {k:v[0] for k,v in qk.items()},
                                 {k:v[1] for k,v in qk.items()}, roles, num_heads=16)
        # Evaluation is separate from the scorer; its only labels are the
        # supplied reference mask, downsampled by the source's own rule.
        truth = got['reference_fg'][0].numpy().ravel().astype(bool)
        scores = {}
        for name, key in [('real', 'probability'), ('direct', 'direct_probability'), ('wrong', 'wrong_probability')]:
            prediction = result[key].cpu().numpy() > .5
            scores[name] = float((prediction & truth).sum()/max(int((prediction | truth).sum()), 1))
        payload = {k:result[k].cpu().numpy() for k in ('probability', 'direct_probability', 'wrong_probability')}
        info = dict(episode_id=row['episode_id'], reference_rgb_hash=row['reference_rgb_hash'],
                    reference_mask_hash=row['reference_mask_hash'], reference_photo_id=row.get('reference_photo_id'),
                    roles=roles, query_anchor=result['query_anchor'], wrong_anchor=result['wrong_anchor'],
                    native_last_output_debiased=bool(host.should_debiass), gt_foreground_patches=int(truth.sum()),
                    scores=scores)
        return payload, info
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--basis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    a = p.parse_args()
    a.assets, a.out = a.assets.resolve(), a.out.resolve()
    decision = json.loads(a.decision.read_text())
    choice = qualified_choice(decision)
    if choice is None:
        print('C2 representation did not qualify; no reference encoding or candidate execution', flush=True)
        return
    prepared = json.loads((a.manifest.parent/'prepared.json').read_text())
    if prepared.get('state') != 'FRESH_MANIFEST_FROZEN' or prepared.get('seed') != 1:
        raise ValueError('Self check uses the fresh seed1 development references')
    rows = json.loads(a.manifest.read_text())
    if set(r['dataset'] for r in rows) != set(DATASETS):
        raise ValueError('Require the five complete development cohorts')
    config = dict(choice=choice, decision_sha256=file_hash(a.decision), manifest_sha256=file_hash(a.manifest),
                  code_sha256=file_hash(Path(__file__)), scorer_sha256=file_hash(REPO/'scripts/frozen_unary_candidates.py'),
                  weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
                  native_basis_sha256=file_hash(a.assets/'native_assets/positional_basis.pt'),
                  own_bases_sha256=file_hash(a.basis/'complete.json'), encoder_device=a.device,
                  commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  input='reference RGB/mask, query RGB is the same image; no query annotation reads',
                  self_gt='actual FoRIS stage2 64-grid reference label including source fallback',
                  confidence_threshold=.5, minimum_mean_iou=.98, minimum_wrong_drop=.05)
    torch.set_num_threads(2)
    with locked_run(a.out):
        if (a.out/'report.json').exists():
            old = json.loads((a.out/'config.json').read_text())
            if {k:v for k,v in old.items() if k != 'commit'} != {k:v for k,v in config.items() if k != 'commit'}:
                raise ValueError('Existing self check belongs to different inputs/specification')
            print('Existing C2 self check retained', flush=True)
            return
        if (a.out/'config.json').exists():
            raise ValueError('Incomplete self check requires inspection; do not silently mix attempts')
        write(a.out/'config.json', config)
        complete = json.loads((a.basis/'complete.json').read_text())
        bases = {}
        for layer in (16, 24):
            for kind in ('Q', 'K'):
                key = f'{kind}/{layer}';entry=complete['bases'][key];path=a.basis/entry['path']
                if file_hash(path) != entry['sha256']:
                    raise ValueError('Own branch basis changed')
                bases[key] = torch.load(path, map_location='cpu', weights_only=True)['basis']
        host = build_host(dict(projection_basis=str(a.assets/'native_assets/positional_basis.pt')),
                          'cpu', str(a.assets/'third_party/foris_official'),
                          weights=str(a.assets/'demo4_cache/models/dinov3-vitl16-timm'), mask_refiner='bilinear')
        host.encoder.to(a.device)
        (a.out/'predictions').mkdir(exist_ok=True)
        report = dict(state='COMPLETE', choice=choice, datasets={}, candidate_enabled=False,
                      role='reference self sanity, not query segmentation gain')
        with torch.inference_mode():
            for name in DATASETS:
                selected, skipped = [], []
                for row in (r for r in rows if r['dataset'] == name):
                    started = time.monotonic()
                    result = evaluate_reference(host, row, a.assets, bases, choice, a.device)
                    if result is None:
                        skipped.append(row['episode_id']);continue
                    payload, info = result;info['seconds']=time.monotonic()-started
                    file = a.out/'predictions'/f'{name}_{len(selected)}.npz';np.savez_compressed(file, **payload)
                    info.update(prediction_file=file.name, prediction_sha256=file_hash(file));selected.append(info)
                    print(json.dumps(dict(dataset=name, n=len(selected), scores=info['scores'])), flush=True)
                    if len(selected) == 5:
                        break
                means = {key:float(np.mean([r['scores'][key] for r in selected])) if selected else None
                         for key in ('real', 'direct', 'wrong')}
                passed = len(selected) == 5 and means['real'] >= .98 and means['real']-means['wrong'] >= .05
                report['datasets'][name] = dict(n=len(selected), means=means, passed=passed, references=selected, skipped=skipped)
        report['passed'] = all(v['passed'] for v in report['datasets'].values())
        report['utc'] = datetime.now(timezone.utc).isoformat()
        write(a.out/'sealed.json', dict(state='REFERENCE_SELF_PREDICTIONS_SEALED', n=sum(v['n'] for v in report['datasets'].values()),
                                      config_sha256=file_hash(a.out/'config.json'), query_annotations_opened=False))
        write(a.out/'report.json', report)
        print(json.dumps(dict(passed=report['passed'], means={n:d['means'] for n,d in report['datasets'].items()})), flush=True)


if __name__ == '__main__':
    main()
