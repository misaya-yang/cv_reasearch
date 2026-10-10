"""Independent reference-focus physical measure and expanded-risk validation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METHOD = REPO/'src/ics/methods/reference_focus_head.py'
EXPECTED_SHA = '37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_method():
    assert sha(METHOD) == EXPECTED_SHA
    spec = importlib.util.spec_from_file_location('independent_reference_focus', METHOD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expanded_original_fit(features, sample, strength):
    foreground = torch.tensor(sample['diagnostics']['foreground_occurrence_ids'])
    background = torch.tensor(sample['diagnostics']['background_occurrence_ids'])
    ids = torch.cat((foreground, background))
    targets = torch.cat((torch.ones(len(foreground)), -torch.ones(len(background)))).double()
    x = features[ids].double()
    z = torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), dim=1)
    lhs = z.T @ z/len(z)
    lhs[:-1, :-1] += strength*torch.eye(x.shape[1], dtype=torch.float64)
    theta = torch.linalg.solve(lhs, z.T @ targets/len(z))
    return theta, ids, targets


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(4139)
    m = load_method()
    y, x = torch.meshgrid(torch.arange(1024), torch.arange(1024), indexing='ij')
    masks = {'irregular_binary': (((x % 31) < 4) & ((y % 47) < 8)).double(),
             'single_pixel': torch.zeros((1024, 1024), dtype=torch.float64)}
    masks['single_pixel'][405, 271] = 1
    boxes = ((0, 0, 512, 512), (240, 384, 752, 896), (512, 512, 1024, 1024))
    physical, fits = {}, {}
    features = F.normalize(torch.randn(8192, 9), dim=1)
    for mask_name, mask in masks.items():
        for box in boxes:
            key = mask_name+'/'+','.join(map(str, box))
            measure = m._reference_measure(mask, box)
            sample = m._role_quadrature(measure)
            x0, y0, x1, y1 = box
            pixel_view_weights = torch.ones((1024, 1024), dtype=torch.float64)
            pixel_view_weights[y0:y1, x0:x1] = .5
            # Direct pixel sums under each observed view, independent of token coverage formulas.
            fg_whole = F.avg_pool2d((mask*pixel_view_weights)[None, None], 16, stride=16)[0, 0].reshape(-1)*256
            fg_focus = F.avg_pool2d((mask[y0:y1, x0:x1]*.5)[None, None], 8, stride=8)[0, 0].reshape(-1)*64
            bg_whole = F.avg_pool2d(((1-mask)*pixel_view_weights)[None, None], 16, stride=16)[0, 0].reshape(-1)*256
            bg_focus = F.avg_pool2d(((1-mask[y0:y1, x0:x1])*.5)[None, None], 8, stride=8)[0, 0].reshape(-1)*64
            expected_fg, expected_bg = torch.cat((fg_whole, fg_focus)), torch.cat((bg_whole, bg_focus))
            assert torch.equal(expected_fg, measure['foreground']) and torch.equal(expected_bg, measure['background'])
            # Independent half-pixel integer cumulative mass reproduces both128 midpoint samples.
            quantiles = {}
            for name, mass in (('foreground', expected_fg), ('background', expected_bg)):
                half_pixel_mass = (mass.numpy()*2).astype(np.int64)
                assert np.array_equal(half_pixel_mass, mass.numpy()*2)
                cumulative = np.cumsum(half_pixel_mass, dtype=np.int64)
                positions = (2*np.arange(128)+1)*int(cumulative[-1])/256
                ids = np.searchsorted(cumulative, positions, side='left')
                assert np.array_equal(ids, sample['diagnostics'][name+'_occurrence_ids'])
                quantiles[name] = dict(count=len(ids), exact_independent_ids=True,
                                       duplicate_occurrences=len(ids)-len(np.unique(ids)))
            weighted_prior = float(expected_fg.sum()/(expected_fg+expected_bg).sum())
            check = dict(direct_pixel_foreground_vector_exact=True, direct_pixel_background_vector_exact=True,
                         foreground_mass_error=float((expected_fg.sum()-mask.sum()).abs()),
                         background_mass_error=float((expected_bg.sum()-(1-mask).sum()).abs()),
                         total_physical_area=float(measure['physical_area'].sum()),
                         raw_physical_prior_error=abs(weighted_prior-float(mask.mean())), quantiles=quantiles,
                         role_risk_weights=[float(sample['foreground_risk'].sum()), float(sample['background_risk'].sum())],
                         weighted_target_mean=float((sample['weights']*sample['target']).sum()))
            assert check['foreground_mass_error'] == check['background_mass_error'] == check['raw_physical_prior_error'] == 0
            assert check['total_physical_area'] == 1024**2 and check['role_risk_weights'] == [.5, .5]
            fit = m._fit(features, measure, sample)
            theta, expanded_ids, labels = expanded_original_fit(features, sample, m.RIDGE_STRENGTH)
            pred = features[expanded_ids].double() @ theta[:-1]+theta[-1]
            original_loss = (pred-labels).square().mean()
            fit_check = dict(coefficient_max_error=float((theta[:-1]-fit['coefficient']).abs().max()),
                             intercept_error=float((theta[-1]-fit['bias']).abs()),
                             original_expanded_binary_loss_error=abs(float(original_loss)-fit['diagnostics']['sample_binary_role_loss']),
                             collapsed_binary_loss_error=fit['diagnostics']['binary_collapsed_loss_absolute_error'],
                             coefficient_stationarity_l2=fit['diagnostics']['coefficient_stationarity_l2'],
                             intercept_stationarity_abs=abs(fit['diagnostics']['intercept_stationarity']))
            assert max(fit_check.values()) < 1e-11, fit_check
            physical[key], fits[key] = check, fit_check
    box, mask = boxes[1], masks['irregular_binary']
    whole = torch.randn(4096, 9)+.4
    derived = m._derived_focus(whole, box)
    query = torch.randn(4096, 9)+.4
    corners = [torch.randn(4096, 9)+.4 for _ in range(4)]
    projection = torch.eye(9)
    projection[:3, :3] = 0
    pure_case = {}
    for apd in (False, True):
        result = m.fit_predict(whole, derived, mask, box, query, corners, apply_apd=apd, projection=projection)
        repeat = m.fit_predict(whole, derived, mask, box, query, corners, apply_apd=apd, projection=projection)
        exact = {view: np.array_equal(result['fields']['actual.'+view], result['fields']['derived.'+view])
                 for view in ('global', 'local4', 'equal')}
        same = all(np.array_equal(value, repeat['fields'][name]) for name, value in result['fields'].items())
        assert all(exact.values()) and same
        json.dumps(result['diagnostics'], allow_nan=False)
        pure_case[str(apd)] = dict(actual_equal_derived=exact, all6_repeat_exact=same, shapes_correct=all(
            value.shape == (128, 128) and value.dtype == np.float32 for value in result['fields'].values()))
    # Full1024-channel fixture preserves old study's normalization strides and arithmetic.
    q1024 = torch.randn(4096, 1024)
    corners1024 = [torch.randn(4096, 1024) for _ in range(4)]
    full_projection = torch.eye(1024)
    full_projection[:500, :500] = 0
    parity = {}
    for apd in (False, True):
        old_global_raw = F.interpolate(q1024.reshape(64, 64, 1024).permute(2, 0, 1)[None], (128, 128),
                                       mode='bilinear', align_corners=False)[0].permute(1, 2, 0).reshape(-1, 1024)
        old_global = F.normalize(old_global_raw, dim=1)
        old_local = torch.cat((torch.cat((corners1024[0].reshape(64,64,1024), corners1024[1].reshape(64,64,1024)), dim=1),
                               torch.cat((corners1024[2].reshape(64,64,1024), corners1024[3].reshape(64,64,1024)), dim=1)), dim=0).reshape(-1,1024)
        old_local = F.normalize(old_local, dim=1)
        if apd:
            old_global, old_local = (F.normalize(value @ full_projection.T, dim=1) for value in (old_global, old_local))
        new = m._query_representations(q1024, corners1024, apd, full_projection)
        check = dict(global_exact=torch.equal(old_global, new['global_features']),
                     local4_exact=torch.equal(old_local, new['local4_features']),
                     global_max_error=float((old_global-new['global_features']).abs().max()),
                     local4_max_error=float((old_local-new['local4_features']).abs().max()),
                     old_global_raw_stride=list(old_global_raw.stride()))
        assert check['global_exact'] and check['local4_exact'], check
        parity[str(apd)] = check
    report = dict(status='PASSED_INDEPENDENT_REFERENCE_FOCUS_PREFLIGHT_NO_GT', method_sha256=sha(METHOD),
                  validator_sha256=sha(__file__), physical=physical, expanded_primal_dual=fits,
                  actual_equals_derived_pure_case=pure_case, original_query_representation_byte_parity=parity,
                  query_GT_reads=0, real_feature_arrays_read=0, encoder_calls=0,
                  source_score_semantics='raw physical role masses preserved before equal-role risk; signed score is not area occupancy/posterior.',
                  matched_contrast='Identical labels, physical areas, sampling IDs, weights, targets, lambda and query features; real versus derived focus feature values differ.',
                  elapsed_seconds=time.perf_counter()-started,
                  remaining=['Runner static review once written; actual replay and strong-baseline input identity after all outputs are sealed.',
                             'Source loss/purity is reconstruction evidence, not query confidence or performance.'])
    (OUT/'reference_focus_preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], physical_cases=len(physical),
                         max_coefficient_error=max(r['coefficient_max_error'] for r in fits.values()),
                         max_intercept_error=max(r['intercept_error'] for r in fits.values()),
                         pure_case=pure_case, prior_query_parity=parity, seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
