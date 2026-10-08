#!/usr/bin/env python3
"""Prepare 24-layer O/Q/K/V positional bases using one normalized black image."""
import argparse
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import torch
from torchvision.transforms.functional import normalize
from ics.data import TimmDINOv3
from ics.official_data import file_hash
from ics.representations import observe_branches, positional_basis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    config = dict(source_input='normalized_black_image', image_size=1024, max_components=500,
                  QK='after QK norm before RoPE; prefix removed', O='block output with model.norm',
                  V='value projection before attention', unit_normalize_before_center=True,
                  zero_rank='exactly constant unit tokens: no subtraction', device=a.device,
                  weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
                  observer_sha256=file_hash(REPO/'src/ics/representations.py'))
    config_path = a.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Basis directory belongs to a different implementation/input')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    if (a.out/'complete.json').exists():
        print('Branch bases already complete', flush=True)
        return
    encoder = TimmDINOv3(a.assets/'demo4_cache/models/dinov3-vitl16-timm').to(a.device).eval()
    black = normalize(torch.zeros(1, 3, 1024, 1024),
                      mean=[.485, .456, .406], std=[.229, .224, .225]).to(a.device)
    start = time.monotonic()
    with torch.inference_mode(), observe_branches(encoder) as tapped:
        encoder.get_intermediate_layers(black, n=1, reshape=False)
    del encoder, black
    if a.device == 'mps':
        torch.mps.empty_cache()
    if len(tapped) != 96 or any(x.shape != (1, 4096, 1024) for x in tapped.values()):
        raise ValueError('Expected all 96 branch/layer black representations')
    elapsed_forward = time.monotonic()-start
    receipts = {}
    for key in sorted(tapped, key=lambda k: (int(k.split('/')[1]), k[0])):
        path = a.out/(key.replace('/', '_')+'.pt')
        if path.exists():
            doc = torch.load(path, map_location='cpu', weights_only=True)
            if doc['branch'] != key:
                raise ValueError('Basis branch differs')
        else:
            at = time.monotonic()
            basis, info = positional_basis(tapped[key])
            doc = dict(branch=key, basis=basis, **info, svd_seconds=time.monotonic()-at)
            temporary = path.with_suffix('.tmp')
            torch.save(doc, temporary)
            temporary.replace(path)
        del tapped[key]
        receipts[key] = dict(path=path.name, sha256=file_hash(path), rank=doc['rank_used'],
                             zero_rank=doc['zero_rank'], svd_seconds=doc['svd_seconds'])
        print(json.dumps(dict(branch=key, rank=doc['rank_used'], seconds=doc['svd_seconds'])), flush=True)
    report = dict(state='COMPLETE', n=96, config=config, bases=receipts,
                  forward_seconds=elapsed_forward, total_seconds=time.monotonic()-start)
    temporary = a.out/'complete.tmp'
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(a.out/'complete.json')
    print(json.dumps(dict(state='COMPLETE', n=96, seconds=report['total_seconds'])), flush=True)


if __name__ == '__main__':
    main()
