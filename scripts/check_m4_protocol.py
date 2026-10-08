#!/usr/bin/env python3
"""Check official sampling identity, RGB-only replay and fine-readout geometry."""
from pathlib import Path
import argparse
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    a = p.parse_args()
    import numpy as np
    import torch
    from ics.official_data import (array_hash, build_dataset, load_inputs,
                                   record_episode, reset_sampling_seed, sample_observed)
    from ics import fine_readout
    torch.set_num_threads(2)
    ds = build_dataset('coco', 0, a.assets)
    reset_sampling_seed(); plain = ds[0]; state0 = np.random.get_state()
    reset_sampling_seed(); observed, _ = sample_observed(ds, 0); state1 = np.random.get_state()
    for role in ('tgt_img',):
        assert array_hash(np.asarray(plain[role])) == array_hash(np.asarray(observed[role]))
    assert array_hash(np.asarray(plain['ref_imgs'][0])) == array_hash(np.asarray(observed['ref_imgs'][0]))
    assert torch.equal(plain['tgt_mask'], observed['tgt_mask'])
    assert torch.equal(plain['ref_masks'][0], observed['ref_masks'][0])
    assert all(np.array_equal(x, y) for x, y in zip(state0, state1))
    with tempfile.TemporaryDirectory() as temporary:
        reset_sampling_seed()
        row = record_episode(ds, 0, 0, a.assets, Path(temporary))
        row['query_mask_path'] = '/intentionally-unavailable-query-label.png'
        rgb, reference_mask, query = load_inputs(row, a.assets)
        assert reference_mask.shape == (rgb.height, rgb.width)
        assert array_hash(np.asarray(query)) == row['query_rgb_hash']
    # The existing original-size finalizer is exactly the same bool->bilinear
    # transform used by the new runner, including non-square output shapes.
    sys.path.insert(0, str(a.assets/'third_party/foris_official'))
    from utils.refinement import upsample_mask
    from run_m4_baselines import render
    rng = np.random.RandomState(19)
    pixels = rng.rand(17, 13) > .4
    for shape in ((31, 21), (5, 29), (1024, 1024)):
        source = upsample_mask(torch.from_numpy(pixels), *shape).numpy()
        assert np.array_equal(source, render(pixels, shape))
    # Constant unit features isolate the published spatial geometry. Check the
    # first 73 fine positions against a separate scalar Gaussian calculation.
    fine = torch.zeros(16384, 1024); fine[:, 0] = 1
    q = torch.zeros(4096, 1024); q[:, 0] = 1
    constant = fine_readout.field(fine, q, np.full((64, 64), .625, np.float32))
    assert np.max(np.abs(constant-.625)) < 2e-7
    coarse = np.linspace(-.2, 1.2, 4096, dtype=np.float32).reshape(64, 64)
    out = fine_readout.field(fine, q, coarse)
    assert out.min() >= coarse.min()-1e-6 and out.max() <= coarse.max()+1e-6
    golden = []
    for index in range(73):
        fy, fx = divmod(index, 128)
        cy, cx = int(np.rint(fy/2-.25)), int(np.rint(fx/2-.25))
        numerator = denominator = 0.
        for y in range(cy-2, cy+3):
            for x in range(cx-2, cx+3):
                if not (0 <= y < 64 and 0 <= x < 64):
                    continue
                dy, dx = (16*y+8-(8*fy+4))/16, (16*x+8-(8*fx+4))/16
                w = np.exp(-(dy*dy+dx*dx)/(2*1.25**2))
                numerator += w*coarse[y, x]; denominator += w
        golden.append(numerator/denominator)
    assert np.max(np.abs(out.ravel()[:73]-golden)) < 1e-6
    print('PASS: official first draw/RNG, inference excludes query labels, source finalizer, fine constants/bounds/geometry')


if __name__ == '__main__':
    main()
