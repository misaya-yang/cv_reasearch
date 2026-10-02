#!/usr/bin/env python3
"""Ten CPU fake-encoder/interface/geometry cases, NEVER a DINO gain test."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
import torch.nn.functional as F
from tics.reference_views import (FrozenEncoderAdapter, support_background_views,
    patch_lattice_views, inverse_coordinates, shift_support_annotation,
    fit_support_view_weights, combine_view_outputs, background_holdout_factory)


class PoolEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, image):
        self.calls += 1
        return F.avg_pool2d(image, 16)


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected explicit guard, got an accepted invalid input')


def run():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    cases = []
    max_affine_error = 0.
    for seed in range(10):
        torch.manual_seed(seed)
        b, h, w = 2, 128, 128
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        fg = (((yy//16 + xx//16) % 2) == 0)[None, None].expand(b, 1, h, w).clone()
        image = torch.rand(b, 3, h, w, dtype=torch.float64)*.25
        image[:, :1] = torch.where(fg, .85, .15)
        original = image.clone()
        encoder = PoolEncoder().eval()
        background = support_background_views(image, fg, encoder)
        assert encoder.calls == 3 and len(background.maps) == 3
        assert torch.equal(image, original)
        for value in background.rgb:
            assert torch.equal(value[fg.expand_as(image)], image[fg.expand_as(image)])
        for bi in range(b):
            for ci in range(3):
                assert torch.equal(background.rgb[2][bi, ci][~fg[bi, 0]].sort().values,
                                   image[bi, ci][~fg[bi, 0]].sort().values)
        expected_mean = image.mean((-2, -1), keepdim=True).expand_as(image)
        assert torch.equal(background.rgb[1][~fg.expand_as(image)], expected_mean[~fg.expand_as(image)])
        # Coordinate-affine RGB: all valid back-sampled average-pool features
        # must agree with original lattice exactly to interpolation roundoff.
        affine = torch.stack((yy.double(), xx.double(), yy.double()+2*xx.double()), 0)[None].repeat(b, 1, 1, 1)
        lattice_encoder = PoolEncoder().eval()
        lattice = patch_lattice_views(affine, lattice_encoder)
        assert lattice_encoder.calls == 4 and lattice.maps[0].shape == (b, 3, 8, 8)
        for offset, value, fmap, valid in zip(lattice.offsets, lattice.rgb, lattice.maps, lattice.valid):
            dy, dx = offset
            assert torch.equal(value[..., :h-dy, :w-dx], affine[..., dy:, dx:])
            coords, pixel_valid = inverse_coordinates((h, w), offset)
            restored = coords + torch.tensor(offset)
            assert torch.equal(restored[..., 0], yy.double()) and torch.equal(restored[..., 1], xx.double())
            assert pixel_valid.sum() == (h-dy)*(w-dx)
            shifted_fg, shifted_valid = shift_support_annotation(fg, offset)
            assert torch.equal(shifted_fg[..., :h-dy, :w-dx], fg[..., dy:, dx:])
            assert shifted_valid.sum() == b*(h-dy)*(w-dx)
            assert not shifted_fg[~shifted_valid].any()
            if dy:
                assert not valid[..., 0, :].any() and not valid[..., -1, :].any()
            if dx:
                assert not valid[..., :, 0].any() and not valid[..., :, -1].any()
            err = (fmap-lattice.maps[0]).abs()[valid.expand_as(fmap)].max().item()
            max_affine_error = max(max_affine_error, err)
            assert err <= 1e-12
        # Equal-budget probability/feature combining includes original view,
        # preserving original-only border evidence rather than inventing BG.
        combined, valid = combine_view_outputs(lattice.maps, lattice)
        assert valid.all() and (combined-lattice.maps[0]).abs().max() <= 1e-12
        label = fg[..., ::16, ::16]
        callback_calls = [0]
        def support_recipe(feature, train_fg, train_bg, held_valid):
            callback_calls[0] += 1
            assert not (train_fg & held_valid).any()
            assert not (train_bg & held_valid).any()
            assert not (train_fg & train_bg).any()
            channel = feature[:, :1]
            fg_mean = (channel*train_fg).flatten(1).sum(1)/train_fg.flatten(1).sum(1).clamp_min(1)
            bg_mean = (channel*train_bg).flatten(1).sum(1)/train_bg.flatten(1).sum(1).clamp_min(1)
            midpoint = (fg_mean+bg_mean)[:, None, None, None]/2
            # Fixed classifier; training centroids only, no held labels/GT
            # available inside this callback.
            return torch.sigmoid(8*(channel-midpoint)).float()
        factory = background_holdout_factory(image, fg, encoder)
        def audited_factory(held):
            result = factory(held)
            protected_mask = F.interpolate(held.float(), size=(h, w), mode='nearest').bool()
            protected = protected_mask.expand_as(image)
            for value in result.rgb:
                assert torch.equal(value[protected], image[protected])
            # Rendering itself is invariant to every withheld annotation bit.
            alternate = support_background_views(image, fg ^ protected_mask,
                PoolEncoder().eval(), protected_pixels=protected_mask)
            assert all(torch.equal(x, y) for x, y in zip(result.rgb, alternate.rgb))
            return result
        weights = fit_support_view_weights(background, label, support_recipe, region_view_factory=audited_factory)
        assert weights.valid.all() and weights.eligible_regions.all()
        assert callback_calls[0] == 3*4
        assert torch.allclose(weights.weights.sum(1), torch.ones(b))
        assert torch.equal(weights.uniform_weights, torch.full((b, 3), 1/3))
        assert weights.metadata['query_labels_used'] is False
        # No balanced holdout -> explicit ineligible uniform, no fake score.
        assert encoder.calls == 15 and weights.metadata['calibration_encoder_calls'] == 12
        origin_support = patch_lattice_views(image, PoolEncoder().eval())
        origin_weights = fit_support_view_weights(origin_support, label, support_recipe)
        assert origin_weights.valid.all() and origin_weights.eligible_regions.all()
        assert torch.allclose(origin_weights.weights.sum(1), torch.ones(b))
        assert origin_weights.uniform_weights.shape == (b, 4)
        assert origin_weights.metadata['calibration_encoder_calls'] == 0
        invalid = fit_support_view_weights(background, torch.ones_like(label), support_recipe, region_view_factory=audited_factory)
        assert not invalid.valid.any() and torch.equal(invalid.weights, invalid.uniform_weights)
        assert invalid.heldout_brier.isnan().all()
        expect_error(lambda: support_background_views(image, fg.float(), encoder))
        expect_error(lambda: fit_support_view_weights(background, label, support_recipe))
        expect_error(lambda: support_background_views(image, torch.ones_like(fg), encoder))
        expect_error(lambda: patch_lattice_views(image, lambda x: x))
        expect_error(lambda: FrozenEncoderAdapter(PoolEncoder().train())(image))
        expect_error(lambda: FrozenEncoderAdapter(lambda x: x.flatten(2))(image))
        cases.append(dict(seed=seed,foreground_rgb_exact=True,background_histogram_exact=True,
                          original_rgb_untouched=True,coordinate_roundtrip_exact=True,
                          reflected_label_exclusion=True,local_patch_border_exclusion=True,
                          same_budget_uniform_control=True,support_holdouts_label_disjoint=True,
                          heldout_rgb_not_label_rendered=True,background_encoder_calls=3,
                          background_calibration_encoder_calls=12,lattice_encoder_calls=4))
    assert 'query' not in inspect.signature(fit_support_view_weights).parameters
    return dict(state='PASSED',scope='CPU fake encoder + interface/geometry only; no DINO/task gain',
                cases=cases,cases_count=10,max_affine_feature_alignment_error=max_affine_error,
                cuda_initialized=torch.cuda.is_initialized(),no_weights_loaded=True,
                no_downloads=True,methods=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    args = parser.parse_args()
    report = run()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
