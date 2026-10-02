#!/usr/bin/env python3
"""CPU-only algebra, leakage and ten-episode synthetic CLI checks; no GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPARED_ROOT = Path(os.environ.get('DEMO9_PREPARED_ROOT', str(ROOT)))
spec = importlib.util.spec_from_file_location('inverse_route_probe', HERE / 'inverse_route_probe.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(ROOT))
from tics import ImageSet


def algebra_checks():
    torch.manual_seed(6)
    dtype = torch.float64
    # Five donor votes including repeated routes and an entirely unvisited atom.
    values = [torch.rand(16, dtype=dtype) for _ in range(7)]
    atoms = [torch.randint(0, 8, (16,)) for _ in range(7)]
    B = m.route_matrix(values, atoms, 9)
    assert torch.allclose(B.sum(1), torch.ones(16, dtype=dtype))
    assert torch.equal(B[:, 8], torch.zeros(16, dtype=dtype))
    z0 = (torch.randint(0, 2, (9,)) * 2 - 1).to(dtype)
    y = (torch.randint(0, 2, (16,)) * 2 - 1).to(dtype)
    v = torch.linspace(.5, 1.5, 16, dtype=dtype)
    delta, Bw, gram, residual = m.ridge_delta(B, z0, y, v)
    primal = torch.linalg.solve(Bw.T @ Bw + m.effective_mu(gram) * torch.eye(9, dtype=dtype), Bw.T @ residual)
    assert torch.allclose(delta, primal, atol=1e-10, rtol=1e-10)
    scaled_delta, _, _, _ = m.ridge_delta(B, z0, y, v * 512)
    assert torch.allclose(delta, scaled_delta, atol=1e-9)
    assert delta[8] == 0
    assert (residual - Bw @ delta).square().sum() <= residual.square().sum()
    error = torch.randn(9, dtype=dtype)
    observable, shrunk, signal, _ = m.error_projections(Bw, gram, error, mu=m.effective_mu(gram))
    reference = torch.linalg.pinv(Bw, rtol=1e-3) @ (Bw @ error)
    assert torch.allclose(observable, reference, atol=1e-9)
    assert torch.allclose(Bw @ observable, signal, atol=1e-9)
    assert observable[8] == 0 and shrunk[8] == 0
    assert shrunk.norm() <= observable.norm() + 1e-10
    assert 0 <= m.energy_fraction(observable, error) <= 1 + 1e-10
    assert m.transpose_vote(B, y, v, z0)[8] == z0[8]
    return {'dual_primal_max_abs': float((delta - primal).abs().max()),
            'projection_max_abs': float((observable - reference).abs().max())}


def cache_fixture():
    torch.manual_seed(4)
    n, p, c, h = 20, 64, 12, 8
    f = F.normalize(torch.randn(n, p, c), dim=-1)
    lab = torch.arange(p).reshape(1, -1).repeat(n, 1) // 16
    gt = (f[:, :, 0] > 0).reshape(n, h, h)
    po = [F.normalize(torch.stack([f[i, lab[i] == k].mean(0) for k in range(4)]), dim=-1) for i in range(n)]
    return dict(fold=0, cls=[0] * n, names=[f'image_{i}' for i in range(n)], fq=f.half(), lab=lab,
                Po=po, gt64=gt, gt_bits=np.packbits(gt.numpy().reshape(n, -1), axis=1), S=h)


def interface_checks(D):
    legal = dict(D)
    legal['c'] = 0
    legal['gt64'] = torch.zeros_like(D['gt64'])
    legal['gt64'][0] = D['gt64'][0]
    legal['gt_bits'] = np.zeros_like(D['gt_bits'])
    legal['gt_bits'][0] = D['gt_bits'][0]
    s = ImageSet(legal)
    p1 = {j: (torch.arange(s.P) + j) % 3 == 0 for j in range(s.n)}
    p1[0] = s.gt64[0]
    donors = list(range(2, s.n))
    z0, maps, _, _ = m.label_atoms(s.lab, p1, donors)
    rebuilt = m.masks_from_margins(z0, p1, maps)
    assert all(torch.equal(rebuilt[j], p1[j]) for j in donors)
    predictions, info = m.repair(s, p1)
    altered = {j: p.clone() for j, p in p1.items()}
    altered[1] = ~altered[1]
    predictions2, info2 = m.repair(s, altered)
    assert torch.equal(info['B'], info2['B']) and torch.equal(info['delta'], info2['delta'])
    assert all(torch.equal(predictions[k], predictions2[k]) for k in predictions if k != '1shot')
    # Corrupt illegal GT in the inference object: deployable computation must not read it.
    s.gt64[1:] = True
    predictions3, info3 = m.repair(s, p1)
    assert torch.equal(info['delta'], info3['delta'])
    assert all(torch.equal(predictions[k], predictions3[k]) for k in predictions)
    assert 1 not in maps
    for mask in (torch.zeros(64, dtype=torch.bool), torch.ones(64, dtype=torch.bool), D['gt64'][0].flatten()):
        ps, y, v = m.sample_gold_patches(mask, 8, 32)
        assert len(ps) == 32 and len(torch.unique(ps)) == 32
        assert torch.isclose(v.sum(), torch.tensor(1.0))
        if (y > 0).any() and (y < 0).any():
            assert torch.isclose(v[y > 0].sum(), v[y < 0].sum())
    return {'atom_noop_exact': True, 'query_excluded_from_fit': True,
            'illegal_GT_mutation_invariant': True, 'sampling_class_balance': True}


def cli_check(D):
    with tempfile.TemporaryDirectory(prefix='inverse_route_probe_check_') as folder:
        cache, out = Path(folder) / 'cache.pt', Path(folder) / 'out.json'
        torch.save(D, cache)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run([sys.executable, str(HERE / 'inverse_route_probe.py'), '--file', str(cache),
            '--out', str(out), '--prepared-root', str(PREPARED_ROOT), '--limit', '10', '--seconds', '120'],
            env=env, capture_output=True, text=True, timeout=130)
        assert proc.returncode == 0, proc.stderr
        report = json.loads(out.read_text())
        assert report['state'] == 'COMPLETED' and report['episodes'] == 10
        assert all(r['diagnostic']['zero_route_delta_max'] == 0 for r in report['records'])
        assert len(report['miou']) == 11
        # CLI must reject runs outside the fixed pre-registered cap.
        bad = subprocess.run([sys.executable, str(HERE / 'inverse_route_probe.py'), '--file', str(cache),
            '--out', str(out), '--limit', '41'], env=env, capture_output=True, text=True)
        assert bad.returncode != 0
        return {'synthetic_CPU_episodes': 10, 'elapsed_s': report['elapsed_s'],
                'reported_rows': sorted(report['miou']), 'over_40_rejected': True}


if __name__ == '__main__':
    fixture = cache_fixture()
    print(json.dumps(dict(algebra=algebra_checks(), interface=interface_checks(fixture), cli=cli_check(fixture)), indent=2))
