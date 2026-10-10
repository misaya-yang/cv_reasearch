"""Small no-GT fixtures for the frozen fixed-count ranking experiment."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/'src'))
from ics.methods import reference_rank_transfer as method
from ics.methods.autonomous_context_transport import _role_sample, _ridge

torch.set_num_threads(1)
rng = np.random.default_rng(20261010)
checks = {}
baseline = np.zeros((1024,1024), bool)
baseline.ravel()[[0,3,5,8]] = True
score = np.full((1024,1024), -1., np.float64)
score.ravel()[:9] = [2,1,1,1,1,1,0,0,0]
got, info = method.topk_at_baseline_count(score, baseline)
assert np.array_equal(np.flatnonzero(got), [0,1,3,5])
checks['cutoff_tie_baseline_then_rowmajor'] = True
for constant in (0., 2., -3.):
    got, _ = method.topk_at_baseline_count(np.full_like(score, constant), baseline)
    assert np.array_equal(got, baseline)
checks['constant_field_exact_baseline'] = True
for value in (False, True):
    old = np.full_like(baseline, value)
    got, _ = method.topk_at_baseline_count(score, old)
    assert np.array_equal(got, old)
checks['empty_full_exact_baseline'] = True
for _ in range(6):
    field = rng.standard_normal((64,64)).astype(np.float32)
    pixel = F.interpolate(torch.from_numpy(field)[None,None],(1024,1024),
        mode='bilinear',align_corners=False)[0,0].numpy()
    base = pixel > .5
    got, _ = method.topk_at_baseline_count(pixel, base)
    assert np.array_equal(got, base)
checks['author_style_FP32_own_score_mask_identity_6'] = True
for _ in range(4):
    pixels = rng.integers(-8,9,baseline.shape).astype(np.float64)
    old = rng.random(baseline.shape) < .17
    got, diag = method.topk_at_baseline_count(pixels,old)
    affine, _ = method.topk_at_baseline_count(2*pixels+.25,old)
    assert np.array_equal(got,affine) and int(got.sum()) == int(old.sum())
    assert diag['added_pixels'] == diag['removed_pixels']
checks['positive_affine_invariant_and_exact_K_4'] = True

r = F.normalize(torch.from_numpy(rng.standard_normal((4096,16)).astype(np.float32)),dim=1)
q = F.normalize(torch.from_numpy(rng.standard_normal((4096,16)).astype(np.float32)),dim=1)
c = rng.choice(np.array([0,.0625,.25,.5,.75,.9375,1],np.float32),4096)
old = rng.random(baseline.shape) < .125
pred = method.fit_predict(r,c,q,old)
sample = _role_sample(c)
x = r[sample['ids']].double()
delta = sample['positive'] @ x-sample['negative'] @ x
expected = (q.double() @ delta).numpy().reshape(64,64)
assert np.array_equal(pred['fields64']['prototype'],expected)
checks['prototype_exact_same_IDs_and_role_masses'] = True
fit = _ridge(x,sample)
a,y = sample['weights'],sample['target']
mean_x,mean_y = (a[:,None]*x).sum(0)/a.sum(), (a*y).sum()/a.sum()
xc,yc = x-mean_x,y-mean_y
primal = torch.linalg.solve(xc.T @ (a[:,None]*xc)+.01*torch.eye(x.shape[1],dtype=torch.float64),
                            xc.T @ (a*yc))
assert float((primal-fit['coefficient']).abs().max()) < 1e-12
assert fit['diagnostics']['binary_collapsed_loss_absolute_error'] < 1e-12
assert fit['diagnostics']['weighted_stationarity_l2'] < 1e-12
checks['independent_weighted_primal_stationarity_and_binary_loss'] = True
for field in pred['fields64'].values():
    assert field.dtype == np.float64 and field.shape == (64,64) and np.isfinite(field).all()
for mask in pred['masks1024'].values():
    assert mask.dtype == np.bool_ and int(mask.sum()) == int(old.sum())
json.dumps(pred['diagnostics'],allow_nan=False)
checks['finite_typed_outputs_JSON_diagnostics'] = True
constant = torch.zeros_like(r); constant[:,0] = 1
null = method.fit_predict(constant,c,q,old)
for mask in null['masks1024'].values():
    assert np.array_equal(mask,old)
for coverage in (np.zeros(4096,np.float32),np.ones(4096,np.float32)):
    null = method.fit_predict(r,coverage,q,old)
    assert null['diagnostics']['fit_skipped'] == 'missing_reference_role'
    assert all(np.array_equal(mask,old) for mask in null['masks1024'].values())
checks['null_direction_and_missing_role_exact_baseline'] = True
for value in (False,True):
    full = np.full_like(old,value)
    null = method.fit_predict(r,c,q,full)
    assert null['diagnostics']['fit_skipped'] == 'empty_or_full_baseline'
    assert all(np.array_equal(mask,full) for mask in null['masks1024'].values())
checks['empty_full_predict_skip_fit'] = True
for invalid in (np.full_like(score,np.nan),np.full_like(score,np.inf)):
    try:
        method.topk_at_baseline_count(invalid,old)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite field must fail, not silently select pixels')
checks['nonfinite_rejected'] = True

# One practical whole-O24 dimensional fixture, no model or data file read.
r = F.normalize(torch.from_numpy(rng.standard_normal((4096,1024)).astype(np.float32)),dim=1)
q = F.normalize(torch.from_numpy(rng.standard_normal((4096,1024)).astype(np.float32)),dim=1)
started = time.perf_counter()
large = method.fit_predict(r,c,q,old)
wall = time.perf_counter()-started
assert all(np.isfinite(field).all() for field in large['fields64'].values())
assert all(int(mask.sum()) == int(old.sum()) for mask in large['masks1024'].values())
path = REPO/'src/ics/methods/reference_rank_transfer.py'
helper = REPO/'src/ics/methods/autonomous_context_transport.py'
result = dict(checks=checks,
    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    fixture_seconds=wall,fixture='4096x1024 unit FP32 R/Q; two heads and FP64 1024 readouts',
    CPU_threads=1,query_GT_reads=0,encoder_calls=0,real_case_predictions=0,
    caveat='Synthetic fixture timing; actual CRF and cache IO are runner costs')
out = Path(__file__).with_suffix('.json')
assert not out.exists(), 'Do not replace a finished validation'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,allow_nan=False))
