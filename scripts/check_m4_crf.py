#!/usr/bin/env python3
"""Check channel/layout safety and CRF filter invariants on a nonsquare grid."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[2] / 'cv_data')
    a = p.parse_args()
    import torch
    from ics.m4_crf import install
    cls, receipt = install(a.assets / 'third_party/crf_source', a.assets / 'runtime/macos/crf')
    torch.manual_seed(0)
    image = torch.rand(1, 3, 17, 13)
    x, y = torch.rand(1, 2, 17, 13), torch.rand(1, 2, 17, 13)
    constant = torch.tensor([.25, .75])[None, :, None, None].expand_as(x).contiguous()
    errors = {}
    with torch.inference_mode():
        for bilateral in (False, True):
            layer = cls(bilateral, 12., .03, 4.)
            out = layer(x, image)
            assert out.shape == x.shape and torch.isfinite(out).all()
            const_error = float((layer(constant, image) - constant).abs().max())
            linear_error = float((layer(x+y, image)-layer(x, image)-layer(y, image)).abs().max())
            assert const_error < 1e-5 and linear_error < 1e-5
            assert out.min() >= x.min()-1e-5 and out.max() <= x.max()+1e-5
            errors[str(bilateral)] = dict(constant_max_error=const_error, linear_max_error=linear_error)
        # Colour separation should block mixing of the two regions even with
        # a broad spatial kernel; this checks interleaved image channels.
        colours = torch.zeros_like(image); colours[:, :, :, 6:] = 1
        step = colours[:, :1].expand(1, 2, 17, 13).contiguous()
        separated = cls(True, 24., .01, 4.)(step, colours)
        boundary_error = float((separated-step).abs().max())
        assert boundary_error < 1e-5
        errors['colour_boundary_max_error'] = boundary_error
        from CRF import DenseGaussianCRF, FrankWolfeParams
        crf = DenseGaussianCRF(classes=2, alpha=12., beta=.03, gamma=4.,
            spatial_weight=3., bilateral_weight=20., compatibility=1., init='potts',
            solver='fw', iterations=10, params=FrankWolfeParams(scheme='fixed',
                stepsize=1., regularizer='l2', lambda_=1., lambda_learnable=False,
                x0_weight=0., x0_weight_learnable=False)).eval()
        logits = torch.cat([-5*step[:, :1]+2.5, 5*step[:, :1]-2.5], 1)
        refined = crf(colours, logits)
        assert refined.shape == logits.shape and torch.isfinite(refined).all()
        errors['solver_finite'] = True
    output = dict(status='PASSED', backend=receipt, errors=errors,
                  scope='small CPU filter/solver invariants; not CUDA parity or full-resolution validation')
    dest = Path(__file__).resolve().parents[1] / 'outputs/m4/environment'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'crf_smoke.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
