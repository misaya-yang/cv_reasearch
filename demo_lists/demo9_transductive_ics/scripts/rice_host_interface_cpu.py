#!/usr/bin/env python3
"""Actual installed FoRIS decoder with a synthetic CPU encoder, no DINO load."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foris-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('Preserve existing CPU receipts')
    torch.set_num_threads(1)
    with patch('torch.cuda.is_available', return_value=False):
        sys.path.insert(0, args.foris_root)
        import models.foris as source
        from native_rice_core_experiment import capture_native, frozen_native_replay, exact_clustering_cache, digest
        class Encoder(torch.nn.Module):
            def get_intermediate_layers(self, rgb, n=1, reshape=True):
                a = F.avg_pool2d(rgb, 16)
                return [torch.cat((a, a.square(), a[:, :2] - a[:, 1:3]), dim=1)]
        original = source.FoRIS._build_positional_basis
        source.FoRIS._build_positional_basis = lambda this, dev: torch.eye(8)[:, :2]
        try:
            host = source.FoRIS(Encoder(), image_size=64, svd_components=2,
                mask_refiner='bilinear', resize_to_orig_size=False, device='cpu').eval()
        finally: source.FoRIS._build_positional_basis = original
        generator = torch.Generator().manual_seed(2055)
        records = []
        for index in range(10):
            s, q = torch.randn(1, 3, 64, 64, generator=generator), torch.randn(3, 64, 64, generator=generator)
            mask = torch.zeros(1, 64, 64, dtype=torch.bool); mask[:, 16:48, 16:48] = True
            before = {k: v for k, v in host.__dict__.items() if callable(v)}
            with torch.inference_mode(), exact_clustering_cache(source) as ca:
                with capture_native(host) as native:
                    prediction = host.predict(s, mask, q).clone()
                raw, part1, part2 = [native[k] for k in ('_extract_features', '_part1_positional_debias', '_part2_background_suppression')]
                with frozen_native_replay(host, raw, part1, part2):
                    replay = host.predict(s, mask, q)
                assert torch.equal(prediction, replay)
                p = torch.linspace(.1, .9, 16).reshape(4, 4)
                new = (p - .5, p, 1 - p, part2[3], part2[4])
                with frozen_native_replay(host, raw, part1, new):
                    inserted = host.predict(s, mask, q)
                assert inserted.shape == prediction.shape and torch.isfinite(inserted).all()
                assert {k: v for k, v in host.__dict__.items() if callable(v)} == before
                records.append(dict(index=index, actual_FoRIS_source=True, native_replay_pixel_exact=True,
                    inserted_full_decoder_ran=True, cache=ca.copy()))
        receipt = dict(state='CPU_RICE_HOST_INTERFACE_PASSED', source_sha256=digest(Path(args.foris_root) / 'models/foris.py'),
            records=records, synthetic_encoder=True, pretrained_DINO=False, CRF_executed=False,
            CUDA_initialized=torch.cuda.is_initialized())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt) + '\n')
        print(json.dumps(dict(state=receipt['state'], cases=len(records), actual_source=True,
            pretrained_DINO=False, CUDA_initialized=receipt['CUDA_initialized'])))


if __name__ == '__main__': main()
