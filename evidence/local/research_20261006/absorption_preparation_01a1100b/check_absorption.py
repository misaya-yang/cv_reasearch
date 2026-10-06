"""Independent first-hit calculation and complete cached-feature context witness."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from ics.methods.reference_absorption import absorbing_values, predict


def feature_witness(dimension=16, reflected=False, neutral_stem=False):
    focal = np.array([2**-.5, 2**-.5, 0.0, 0.0])
    if neutral_stem:
        first = np.array([np.sin(t)*focal + np.cos(t)*np.array([0, 0, 1, 0])
                          for t in np.linspace(0, np.pi/4, 24)])
        junction = first[-1]
        fg = np.array([1, 0, 0, 0])
        theta = np.arccos(junction @ fg)
        middle = np.array([(np.sin(theta-t)*junction + np.sin(t)*fg)/np.sin(theta)
                           for t in np.linspace(0, theta, 13)[1:]])
        pole = np.array([0, 0, 0, 1])
        last_a = np.array([np.cos(t)*junction + np.sin(t)*pole
                           for t in np.linspace(0, np.pi/2, 15)[1:]])
        last_b = np.array([[0, np.sin(t), 0, np.cos(t)] for t in np.linspace(0, np.pi/2, 15)[1:]])
        last = np.concatenate((last_a, last_b))
    else:
        first = np.array([[np.cos(t), np.sin(t), 0, 0] for t in np.linspace(0, np.pi/4, 17)])
        middle = np.array([np.cos(t)*focal + np.sin(t)*np.array([0, 0, 1, 0])
                           for t in np.linspace(0, np.pi/2, 25)[1:]])
        last = np.array([[0, np.sin(t), np.cos(t), 0] for t in np.linspace(0, np.pi/2, 24)[1:]])
    modes = np.concatenate((first, middle, last))
    assert len(modes) == 64
    if reflected:
        modes = modes[:, [1, 0, 2, 3]]
    q = np.zeros((4096, dimension), dtype=np.float32)
    q[:, :4] = np.repeat(modes, 64, axis=0)
    r = np.zeros_like(q)
    r[:2048, 0] = 1
    r[2048:, 1] = 1
    cov = np.zeros((64, 64))
    cov[:32] = 1
    base = np.full((64, 64), .5)
    return q, r, cov, base


def main():
    # Independent absorbing Markov iteration on an explicit path.
    weight = np.zeros((6, 6))
    for i in range(5):
        weight[i, i+1] = weight[i+1, i] = 1
    labels = np.array([1, np.nan, np.nan, np.nan, np.nan, 0])
    value, active, info = absorbing_values(weight, labels)
    np.testing.assert_allclose(value, np.linspace(1, 0, 6), atol=1e-14)
    expected = np.full(6, .5)
    expected[[0, -1]] = [1, 0]
    transition = weight / weight.sum(1, keepdims=True)
    for _ in range(500):
        expected[1:-1] = (transition @ expected)[1:-1]
    np.testing.assert_allclose(value, expected, atol=1e-13)
    # A component with a single reference class does not force every token FG.
    single, single_active, _ = absorbing_values(weight, np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan]))
    assert not single_active.any()
    q, r, cov, base = feature_witness()
    output = predict(q, r, cov, base)
    reflected = predict(*feature_witness(reflected=True))
    focus = np.arange(16*64, 17*64)
    assert output['info']['reference_anchors'] <= 32 and output['info']['graph_nodes'] <= 96
    assert np.isfinite(output['field']).all()
    values = {key: float(output[key].ravel()[focus].mean()) for key in output if key != 'info'}
    mirrored = {key: float(reflected[key].ravel()[focus].mean()) for key in reflected if key != 'info'}
    for key in ('nearest_control', 'kernel_control'):
        np.testing.assert_allclose(values[key], .5, atol=1e-7)
        np.testing.assert_allclose(values[key], mirrored[key], atol=1e-7)
    # Prediction changes while these focal features and reference labels do not.
    assert abs(values['field'] - mirrored['field']) > .1
    stem = predict(*feature_witness(neutral_stem=True))
    stem_reflected = predict(*feature_witness(reflected=True, neutral_stem=True))
    stem_values = {key: float(stem[key].ravel()[:64].mean()) for key in stem if key != 'info'}
    stem_mirrored = {key: float(stem_reflected[key].ravel()[:64].mean())
                     for key in stem_reflected if key != 'info'}
    for key in ('nearest_control', 'kernel_control', 'one_hop_control', 'one_step_control', 'component_control'):
        np.testing.assert_allclose(stem_values[key], .5, atol=1e-7)
    assert stem_values['field'] > .6 and stem_mirrored['field'] < .4
    # Declare the unchanged focal group background: the same pathway is harmful.
    negative_false_additions = int((output['field'].ravel()[focus] > .5).sum())
    missing = predict(q, r, np.ones_like(cov), base)
    np.testing.assert_array_equal(missing['field'], base)
    # One default full-dimensional episode; not a real data or cohort benchmark.
    began = time.perf_counter()
    full = predict(*feature_witness(dimension=1024))
    elapsed = time.perf_counter() - began
    np.testing.assert_allclose(full['field'], output['field'], atol=1e-7)
    source = ROOT / 'src/ics/methods/reference_absorption.py'
    report = dict(kind='mathematical_and_synthetic_only', real_episodes=0,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  six_node_path_maximum_error=float(np.max(np.abs(value - expected))),
                  one_class_component_abstains=True, missing_reference_class_keeps_base=True,
                  same_focal_q_and_reference_context_witness=dict(foreground_path_context=values,
                                                                  background_path_context=mirrored),
                  first_witness_one_step_also_distinguishes=True,
                  neutral_stem_multi_step_witness=dict(foreground_path_context=stem_values,
                                                      background_path_context=stem_mirrored),
                  background_identity_counterexample_false_additions=negative_false_additions,
                  complete_candidate_info=output['info'],
                  full_4096x1024_synthetic_candidate_seconds=elapsed,
                  full_dimension_info=full['info'], no_gpu=True, no_server=True,
                  real_gain='unmeasured', real_runtime='unmeasured')
    Path(__file__).with_name('check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
