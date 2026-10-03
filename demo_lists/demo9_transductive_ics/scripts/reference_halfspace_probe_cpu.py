#!/usr/bin/env python3
"""CPU algebra checks for a proposed observation, not segmentation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_halfspace_probe import reference_axis, project_query_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2132)
    support = torch.randn(3, 12, 5, generator=generator, dtype=torch.float64)
    fg = torch.arange(12) < 4
    bg = torch.arange(12) >= 6
    values = torch.randn(2, 3, 15, 5, generator=generator, dtype=torch.float64)
    axis = reference_axis(support, fg, bg)
    old_support, old_values = support.clone(), values.clone()
    receipt = []

    def project(x, a, branch):
        return project_query_values(x, a, branch, query_index=1, prefix_tokens=3)

    def close(x, y):
        torch.testing.assert_close(x, y, atol=2e-13, rtol=2e-13)

    swapped = reference_axis(support, bg, fg)
    close(swapped.center, axis.center)
    close(swapped.direction, -axis.direction)
    for branch, other in [('plus', 'minus'), ('minus', 'plus')]:
        close(project(values, axis, branch), project(values, swapped, other))
    receipt.append(dict(name='reference_label_swap_exchanges_probes', passed=True))

    bias = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    biased_axis = reference_axis(support+bias[:, None], fg, bg)
    for branch in ('plus', 'minus'):
        close(project(values+bias[None, :, None], biased_axis, branch),
              project(values, axis, branch)+bias[None, :, None])
    receipt.append(dict(name='common_value_bias_equivariance', passed=True))

    rotation = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64)).Q
    rotated_axis = reference_axis(support@rotation, fg, bg)
    for branch in ('plus', 'minus'):
        close(project(values@rotation, rotated_axis, branch),
              project(values, axis, branch)@rotation)
    receipt.append(dict(name='orthogonal_head_coordinates_equivariance', passed=True))

    second = torch.randn(values.shape, generator=generator, dtype=torch.float64)
    distances = []
    for branch in ('plus', 'minus'):
        result = project(values, axis, branch)
        close(project(result, axis, branch), result)
        alpha = ((result[1, :, 3:]-axis.center[:, None])*axis.direction[:, None]).sum(-1)
        assert (alpha <= 2e-13).all() if branch == 'plus' else (alpha >= -2e-13).all()
        old_dist = torch.linalg.vector_norm(values-second, dim=-1)
        new_dist = torch.linalg.vector_norm(result-project(second, axis, branch), dim=-1)
        assert (new_dist <= old_dist+2e-13).all()
        assert torch.equal(result[0], values[0])
        assert torch.equal(result[:, :, :3], values[:, :, :3])
        distances.append(float((new_dist-old_dist).max()))
    assert torch.equal(support, old_support) and torch.equal(values, old_values)
    receipt.append(dict(name='idempotent_nonexpansive_halfspace_projections',
                        passed=True, maximum_distance_increase=max(distances)))
    receipt.append(dict(name='prefix_support_and_original_inputs_unchanged', passed=True))

    zero_axis = reference_axis(torch.ones_like(support), fg, bg)
    empty_axis = reference_axis(support, torch.zeros_like(fg), bg)
    for disabled in (zero_axis, empty_axis):
        for branch in ('plus', 'minus'):
            assert project(values, disabled, branch) is values
    receipt.append(dict(name='degenerate_and_missing_labels_are_exact_noops', passed=True))

    # Source attention probabilities include both prefix and patch keys.
    # Erased values are zero at prefix keys; no patch-only renormalization.
    attention = torch.randn(3, 15, 15, generator=generator, dtype=torch.float64).softmax(-1)
    for branch in ('plus', 'minus'):
        result = project(values, axis, branch)
        erased = values[1]-result[1]
        immediate_delta = attention@values[1]-attention@result[1]
        close(immediate_delta, attention@erased)
    receipt.append(dict(name='immediate_response_exactly_reconstructible_from_native_attention', passed=True,
                        new_information_at_this_stage=False))

    # Centering both class means on ANY pooled mean makes the two
    # centered directions antiparallel. Their ordinary erasure operators
    # are identical, for any mixture weight strictly between zero and one.
    mu_f = support[:, fg].mean(1)
    mu_b = support[:, bg].mean(1)
    for mixture in (.1, .5, .9):
        common = mixture*mu_f+(1-mixture)*mu_b
        uf = torch.nn.functional.normalize(mu_f-common, dim=-1)
        ub = torch.nn.functional.normalize(mu_b-common, dim=-1)
        close(uf[:, :, None]*uf[:, None, :], ub[:, :, None]*ub[:, None, :])
    plus = project(values, axis, 'plus')
    minus = project(values, axis, 'minus')
    assert not torch.allclose(plus, minus)
    receipt.append(dict(name='old_centered_two_erasure_degeneracy_reproduced_new_probes_distinct', passed=True))

    # The downstream nonlinearity can expose a distinction lost in the
    # FINAL scalar: F(x)=relu(x1+x2), (1,0) and (0,1) both yield1.
    # FG-side suppression along e1 gives final responses1 and0.
    # Native hidden states already distinguish them; this is NOT extra
    # Shannon information over the hidden state or original RGB.
    inputs = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    original = inputs.sum(-1).relu()
    changed = inputs.clone()
    changed[:, 0] -= changed[:, 0].clamp_min(0)
    response = original-changed.sum(-1).relu()
    assert torch.equal(original, torch.ones(2, dtype=torch.float64))
    assert torch.equal(response, torch.tensor([1., 0.], dtype=torch.float64))
    receipt.append(dict(name='final_embedding_collision_response_distinguishes_toy', passed=True,
                        response=response.tolist(),
                        scope='Existence toy only; native hidden-state control already distinguishes inputs'))

    paths = [Path(__file__), Path(__file__).resolve().parents[1]/'tics/reference_halfspace_probe.py']
    result = dict(state='CPU_HALFSPACE_PROBE_ALGEBRA_PASSED', checks=receipt,
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  CUDA_initialized=torch.cuda.is_initialized(), real_task_gain_measured=False,
                  encoder_hook_ready=False, full_method_selected=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(state=result['state'], checks=len(receipt),
                          CUDA_initialized=result['CUDA_initialized'],
                          real_task_gain_measured=False)))


if __name__ == '__main__':
    main()
