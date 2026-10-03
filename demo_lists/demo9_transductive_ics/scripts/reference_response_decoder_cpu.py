#!/usr/bin/env python3
"""CPU implementation checks only; no DINO, real task training or gain claim."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tics.reference_response_decoder import (
    ReferenceResponseDecoder, logits_to_mask, mask_to_coverage,
    prefix_attention, segmentation_loss, support_attention, upsample_logits,
)


def dense_attention(query, support, coverage):
    """Independent small verifier; explicitly selects nonzero role anchors."""
    roles = []
    for weights in (coverage, 1-coverage):
        batches = []
        for b in range(query.shape[0]):
            valid = weights[b] > 0
            if not valid.any():
                batches.append(torch.zeros_like(query[b]))
                continue
            scores = query[b] @ support[b, valid].T/math.sqrt(query.shape[-1])
            probability = (scores+weights[b, valid].log()[None]).softmax(dim=-1)
            batches.append(probability @ support[b, valid])
        roles.append(torch.stack(batches))
    return tuple(roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=BASE/'results/native_membership_v1/response_decoder_cpu_v3.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('CPU receipt already exists; refusing overwrite: '+str(args.out))
    torch.set_num_threads(1)
    torch.manual_seed(31027)
    generator = torch.Generator().manual_seed(31027)
    started = time.monotonic()
    assert not torch.cuda.is_initialized()
    checks = []

    def record(name, **evidence):
        checks.append(dict(name=name, passed=True, **evidence))

    def reject(action):
        try:
            action()
        except (ValueError, TypeError):
            return
        raise AssertionError('Expected explicit input rejection')

    # Fractional support supervision is preserved, not converted into seeds.
    gold = torch.tensor([[[[1., 0., 1., 0.], [0., 0., 1., 1.],
                           [0., 0., 0., 0.], [1., 1., 1., 1.]]]])
    coverage = mask_to_coverage(gold, (2, 2))
    torch.testing.assert_close(coverage, torch.tensor([[[[.25, .75], [.5, .5]]]]))
    assert torch.equal(mask_to_coverage(gold.bool(), (2, 2)), coverage)
    record('support_mask_area_downsample_preserves_fractional_coverage')

    query = torch.randn(2, 9, 128, generator=generator, dtype=torch.float64)
    support = torch.randn(2, 11, 128, generator=generator, dtype=torch.float64)
    weights = torch.rand(2, 11, generator=generator, dtype=torch.float64)
    weights[:, 0], weights[:, 1] = 0., 1.
    expected = dense_attention(query, support, weights)
    max_difference = 0.
    for chunk in (1, 4, 32):
        observed = support_attention(query, support, weights, query_chunk_size=chunk)
        for actual, target in zip(observed, expected):
            torch.testing.assert_close(actual, target, atol=2e-12, rtol=2e-12)
            max_difference = max(max_difference, float((actual-target).abs().max()))
    record('query_chunk_attention_matches_independent_full_support_dense',
           chunks=[1, 4, 32], maximum_absolute_error=max_difference)

    grad_query = query.clone().requires_grad_()
    grad_support = support.clone().requires_grad_()
    contexts = support_attention(grad_query, grad_support, weights, query_chunk_size=4)
    sum(x.square().mean() for x in contexts).backward()
    chunk_gradients = (grad_query.grad.clone(), grad_support.grad.clone())
    dense_query = query.clone().requires_grad_()
    dense_support = support.clone().requires_grad_()
    sum(x.square().mean() for x in dense_attention(dense_query, dense_support, weights)).backward()
    gradient_error = 0.
    for actual, target in zip(chunk_gradients, (dense_query.grad, dense_support.grad)):
        torch.testing.assert_close(actual, target, atol=2e-12, rtol=2e-12)
        gradient_error = max(gradient_error, float((actual-target).abs().max()))
    assert (chunk_gradients[1].abs().sum(dim=-1) > 0).all()
    record('checkpoint_chunk_backward_matches_dense_all_support_gradients',
           maximum_absolute_error=gradient_error, support_tokens_with_gradient=22)

    fg, bg = support_attention(query, support, weights, query_chunk_size=4)
    swapped_fg, swapped_bg = support_attention(query, support, 1-weights, query_chunk_size=4)
    torch.testing.assert_close(fg, swapped_bg, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(bg, swapped_fg, atol=2e-12, rtol=2e-12)
    for all_fg in (False, True):
        role_weights = torch.full_like(weights, float(all_fg))
        observed = support_attention(query, support, role_weights, query_chunk_size=4)
        absent = observed[1] if all_fg else observed[0]
        assert torch.equal(absent, torch.zeros_like(absent))
        for actual, target in zip(observed, dense_attention(query, support, role_weights)):
            torch.testing.assert_close(actual, target, atol=2e-12, rtol=2e-12)
    record('coverage_complement_exchanges_contexts_and_missing_roles_are_finite_zero')

    prefix_query = torch.randn(2, 9, 64, generator=generator, dtype=torch.float64)
    prefix_bank = torch.randn(2, 5, 64, generator=generator, dtype=torch.float64)
    prefix_expected = (prefix_query @ prefix_bank.transpose(1, 2)/math.sqrt(64)).softmax(-1) @ prefix_bank
    prefix_error = 0.
    for chunk in (1, 4, 32):
        observed = prefix_attention(prefix_query, prefix_bank, query_chunk_size=chunk)
        torch.testing.assert_close(observed, prefix_expected, atol=2e-12, rtol=2e-12)
        prefix_error = max(prefix_error, float((observed-prefix_expected).abs().max()))
    moved = prefix_bank.clone()
    moved[:, 0, 0] += 2
    moved[:, 1, 0] -= 2
    torch.testing.assert_close(moved.mean(dim=1), prefix_bank.mean(dim=1), atol=2e-16, rtol=2e-16)
    nonmean_delta = float((prefix_attention(prefix_query, moved)-prefix_expected).abs().max())
    assert nonmean_delta > 0
    prefix_q_grad = prefix_query.clone().requires_grad_()
    prefix_p_grad = prefix_bank.clone().requires_grad_()
    prefix_attention(prefix_q_grad, prefix_p_grad, query_chunk_size=4).square().mean().backward()
    dense_q_grad = prefix_query.clone().requires_grad_()
    dense_p_grad = prefix_bank.clone().requires_grad_()
    ((dense_q_grad @ dense_p_grad.transpose(1, 2)/math.sqrt(64)).softmax(-1) @ dense_p_grad).square().mean().backward()
    prefix_gradient_error = 0.
    for actual, expected in ((prefix_q_grad.grad, dense_q_grad.grad), (prefix_p_grad.grad, dense_p_grad.grad)):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        prefix_gradient_error = max(prefix_gradient_error, float((actual-expected).abs().max()))
    assert (prefix_p_grad.grad.abs().sum(dim=-1) > 0).all()
    empty_bank = prefix_bank[:, :0]
    empty_context = prefix_attention(prefix_query, empty_bank, query_chunk_size=4)
    assert torch.equal(empty_context, torch.zeros_like(prefix_query))
    record('prefix_attention_matches_dense_all_tokens_gradients_not_mean_and_empty_zero',
           chunks=[1, 4, 32], prefix_tokens_with_gradient=10,
           maximum_absolute_error=prefix_error, maximum_gradient_error=prefix_gradient_error,
           same_mean_bank_maximum_context_difference=nonmean_delta)

    def fixture():
        features = dict(
            support_final=torch.randn(2, 8, 3, 4, generator=generator),
            support_middle=torch.randn(2, 32, 3, 4, generator=generator),
            support_coverage=torch.rand(2, 1, 3, 4, generator=generator),
            query_final=torch.randn(2, 8, 4, 5, generator=generator),
            query_middle=torch.randn(2, 32, 4, 5, generator=generator),
            support_prefix=torch.randn(2, 5, 32, generator=generator),
            query_prefix=torch.randn(2, 5, 32, generator=generator),
            prefix_immediate_plus=torch.randn(2, 5, 8, generator=generator),
            prefix_immediate_minus=torch.randn(2, 5, 8, generator=generator),
        )
        for name in ('immediate_plus', 'immediate_minus', 'response_plus', 'response_minus'):
            features[name] = torch.randn(2, 8, 4, 5, generator=generator)
        return features

    inputs = fixture()
    old = {name: value.clone() for name, value in inputs.items()}
    model = ReferenceResponseDecoder(8, 32, query_chunk_size=7)
    inputs['query_final'].requires_grad_()
    prefix_fields = ('support_prefix', 'query_prefix', 'prefix_immediate_plus', 'prefix_immediate_minus')
    for name in prefix_fields:
        inputs[name].requires_grad_()
    logits = model(**inputs)
    assert logits.shape == (2, 1, 4, 5) and torch.isfinite(logits).all()
    labels = (torch.rand(2, 1, 8, 10, generator=generator) > .5).float()
    loss = segmentation_loss(logits, labels)
    loss['loss'].backward()
    assert (inputs['query_final'].grad.abs().sum(dim=1) > 0).all()
    gradient_norms = {}
    for name in ('native_final', 'native_middle', 'immediate_plus', 'immediate_minus',
                 'response_plus', 'response_minus'):
        gradient = getattr(model, name)[1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        gradient_norms[name] = float(gradient.norm())
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for name in old:
        assert torch.equal(inputs[name].detach(), old[name])
    record('full_query_grid_and_actual_gradients_reach_all_six_streams',
           output_grid=[2, 1, 4, 5], query_patches_with_gradient=40,
           projection_gradient_norms=gradient_norms, inputs_unchanged=True)

    prefix_input_norms = {}
    for name in prefix_fields:
        gradient = inputs[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert (gradient.abs().sum(dim=-1) > 0).all()
        if name in ('support_prefix', 'query_prefix'):
            assert (gradient.reshape(2, 5, 4, 8).abs().sum(dim=-1) > 0).all()
        prefix_input_norms[name] = float(gradient.norm())
    assert model.prefix_projection[1].weight.grad.abs().sum() > 0
    assert model.prefix_query.weight.grad.abs().sum() > 0
    prefix_output_deltas = {}
    with torch.no_grad():
        baseline = model(**inputs)
        for name in prefix_fields:
            changed = dict(inputs)
            changed[name] = inputs[name].detach().clone()
            changed[name][:, 2, 0] += 2
            delta = float((model(**changed)-baseline).abs().max())
            assert delta > 0
            prefix_output_deltas[name] = delta
    record('head_consumes_each_required_prefix_stream_and_every_native_QKV_prefix_token',
           tokens_with_gradient_per_stream=10, native_channel_blocks_with_gradient=4,
           input_gradient_norms=prefix_input_norms, changed_token_maximum_logit_deltas=prefix_output_deltas)

    static_inputs = {name: value.detach() for name, value in inputs.items()
                     if name not in ('response_plus', 'response_minus')}
    static_model = ReferenceResponseDecoder(8, 32, query_chunk_size=7)
    static_model.load_state_dict(model.state_dict())
    static_logits = static_model(**static_inputs, static=True)
    duplicated = dict(static_inputs, response_plus=static_inputs['immediate_plus'],
                      response_minus=static_inputs['immediate_minus'])
    with torch.no_grad():
        torch.testing.assert_close(static_logits, static_model(**duplicated), atol=0., rtol=0.)
    segmentation_loss(static_logits, labels)['loss'].backward()
    static_gradient_norms = {}
    for name in ('response_plus', 'response_minus'):
        gradient = getattr(static_model, name)[1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        static_gradient_norms[name] = float(gradient.norm())
    count = lambda value: sum(p.numel() for p in value.parameters())
    assert count(model) == count(static_model)
    record('static_control_same_architecture_duplicate_immediate_active_trainable_response_slots',
           parameters_per_arm=count(model), response_projection_gradient_norms=static_gradient_norms,
           prefix_inputs_same_for_both_arms=True)

    with torch.no_grad():
        empty_prefix_inputs = dict(static_inputs)
        empty_prefix_inputs['support_prefix'] = static_inputs['support_prefix'][:, :0]
        assert torch.isfinite(static_model(**empty_prefix_inputs, static=True)).all()
        for name in ('query_prefix', 'prefix_immediate_plus', 'prefix_immediate_minus'):
            empty_prefix_inputs[name] = static_inputs[name][:, :0]
        assert torch.isfinite(static_model(**empty_prefix_inputs, static=True)).all()
    record('head_explicit_empty_support_and_query_prefix_banks_are_finite_zero_context')

    with torch.no_grad():
        changed = dict(static_inputs, support_coverage=1-static_inputs['support_coverage'])
        delta = float((static_model(**changed, static=True)-static_model(**static_inputs, static=True)).abs().max())
        assert delta > 0
        for fraction in (0., 1.):
            missing = dict(static_inputs, support_coverage=torch.full_like(inputs['support_coverage'], fraction))
            assert torch.isfinite(static_model(**missing, static=True)).all()
    record('support_role_change_affects_decoder_and_empty_fg_bg_remain_finite',
           role_change_maximum_logit_difference=delta, no_logit_antisymmetry_required=True)

    small = torch.tensor([[[[-.2, .8], [-.3, -.1]]]])
    torch.testing.assert_close(upsample_logits(small, (5, 7)),
                               F.interpolate(small, (5, 7), mode='bilinear', align_corners=False),
                               atol=0., rtol=0.)
    assert torch.equal(logits_to_mask(small, (5, 7)), upsample_logits(small, (5, 7)) > 0)
    assert not torch.equal(logits_to_mask(small, (5, 7)),
                           F.interpolate((small > 0).float(), (5, 7), mode='bilinear', align_corners=False) > .5)
    target = torch.zeros(1, 1, 5, 7)
    loss_order = segmentation_loss(small, target)
    torch.testing.assert_close(loss_order['bce'],
                               F.binary_cross_entropy_with_logits(upsample_logits(small, (5, 7)), target))
    assert logits_to_mask(small).shape == (1, 1, 1024, 1024)
    record('logits_bilinear_before_zero_threshold_and_both_losses_use_label_grid',
           inference_grid=[1024, 1024], source_minmax=False, CRF_executed=False)

    for target_value in (0., 1.):
        constant_logits = torch.full((1, 1, 2, 3), -40. if target_value else 40., requires_grad=True)
        target = torch.full((1, 1, 4, 6), target_value)
        components = segmentation_loss(constant_logits, target)
        assert all(torch.isfinite(value) for value in components.values())
        components['loss'].backward()
        assert torch.isfinite(constant_logits.grad).all()
    record('empty_foreground_and_all_foreground_loss_and_gradients_are_finite')

    # None must reproduce the original loss and gradient exactly, not merely
    # converge to the same target under a new normalization.
    void_logits = torch.tensor([[[[-.8, .4, 1.2], [.2, -.5, .6]]],
                                [[[.7, -.3, .1], [-1., .9, -.2]]]])
    void_labels = (torch.arange(48).reshape(2, 1, 4, 6) % 3 == 0).float()
    none_logits = void_logits.clone().requires_grad_()
    legacy_logits = void_logits.clone().requires_grad_()
    none_loss = segmentation_loss(none_logits, void_labels, valid_mask=None)
    legacy_resized = F.interpolate(legacy_logits, (4, 6), mode='bilinear', align_corners=False)
    legacy_bce = F.binary_cross_entropy_with_logits(legacy_resized, void_labels)
    legacy_probability = legacy_resized.sigmoid()
    legacy_intersection = (legacy_probability*void_labels).sum(dim=(1, 2, 3))
    legacy_union = (legacy_probability+void_labels-legacy_probability*void_labels).sum(dim=(1, 2, 3))
    legacy_iou = 1-((legacy_intersection+1e-6)/(legacy_union+1e-6)).mean()
    for actual, expected in ((none_loss['bce'], legacy_bce),
                             (none_loss['soft_iou_loss'], legacy_iou),
                             (none_loss['loss'], legacy_bce+legacy_iou)):
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    none_loss['loss'].backward()
    (legacy_bce+legacy_iou).backward()
    assert torch.equal(none_logits.grad, legacy_logits.grad)
    record('valid_mask_none_preserves_legacy_loss_and_gradients_exactly')

    valid = torch.ones_like(void_labels, dtype=torch.bool)
    valid[0, :, :, ::2] = False
    valid[1, :, 1:] = False
    changed_labels = torch.where(valid, void_labels, 1-void_labels)
    first_logits = void_logits.clone().requires_grad_()
    second_logits = void_logits.clone().requires_grad_()
    before = (void_logits.clone(), void_labels.clone(), changed_labels.clone(), valid.clone())
    first_loss = segmentation_loss(first_logits, void_labels, valid_mask=valid)
    second_loss = segmentation_loss(second_logits, changed_labels, valid_mask=valid)
    for key in first_loss:
        torch.testing.assert_close(first_loss[key], second_loss[key], atol=0., rtol=0.)
    first_loss['loss'].backward()
    second_loss['loss'].backward()
    assert torch.equal(first_logits.grad, second_logits.grad)
    resized = F.interpolate(void_logits, (4, 6), mode='bilinear', align_corners=False)
    expected_bce = F.binary_cross_entropy_with_logits(resized[valid], void_labels[valid])
    torch.testing.assert_close(first_loss['bce'], expected_bce, atol=0., rtol=0.)
    expected_ious = []
    for b in range(2):
        p, y = resized[b][valid[b]].sigmoid(), void_labels[b][valid[b]]
        expected_ious.append((p*y).sum().add(1e-6)/(p+y-p*y).sum().add(1e-6))
    torch.testing.assert_close(first_loss['soft_iou_loss'], 1-torch.stack(expected_ious).mean())
    for actual, expected in zip((void_logits, void_labels, changed_labels, valid), before):
        assert torch.equal(actual, expected)
    record('void_label_flips_leave_losses_gradients_exact_and_inputs_unchanged',
           valid_pixels_per_image=[12, 6], BCE_normalization='all valid pixels',
           softIoU_normalization='valid pixels per image, then batch mean',
           invalid_pixels=30, maximum_loss_difference=0., maximum_gradient_difference=0.)

    one_invalid_image = valid.clone()
    one_invalid_image[1] = False
    reject(lambda: segmentation_loss(void_logits, void_labels, valid_mask=one_invalid_image))
    reject(lambda: segmentation_loss(void_logits, void_labels, valid_mask=torch.zeros_like(valid)))
    reject(lambda: segmentation_loss(void_logits, void_labels, valid_mask=valid.float()))
    reject(lambda: segmentation_loss(void_logits, void_labels, valid_mask=valid[:, :, :, :-1]))
    reject(lambda: segmentation_loss(void_logits, void_labels,
                                     valid_mask=torch.empty(valid.shape, dtype=torch.bool, device='meta')))
    record('valid_mask_rejects_any_all_invalid_image_wrong_dtype_shape_device', rejected_cases=5)

    signature = inspect.signature(ReferenceResponseDecoder.forward)
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    for name in ('query_gt', 'query_labels', 'class_id', 'target_area'):
        reject(lambda name=name: model(**inputs, **{name: labels}))
    record('prediction_api_rejects_query_labels_class_id_and_target_area')

    for name in prefix_fields:
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == inspect.Parameter.empty
        missing = {key: value for key, value in inputs.items() if key != name}
        reject(lambda missing=missing: model(**missing))
    prefix_bad_cases = [
        dict(inputs, support_prefix=inputs['support_prefix'][:, :, :-1]),
        dict(inputs, query_prefix=inputs['query_prefix'][:1]),
        dict(inputs, prefix_immediate_plus=inputs['prefix_immediate_plus'][:, :-1]),
        dict(inputs, prefix_immediate_minus=inputs['prefix_immediate_minus'].double()),
        dict(inputs, query_prefix=torch.full_like(inputs['query_prefix'], float('nan'))),
    ]
    for bad in prefix_bad_cases:
        reject(lambda bad=bad: model(**bad))
    record('all_four_prefix_fields_required_and_channel_token_dtype_finite_validation', rejected_cases=9)

    bad_cases = [
        dict(inputs, query_middle=inputs['query_middle'][:, :, :-1]),
        dict(inputs, support_middle=inputs['support_middle'][:, :-1]),
        dict(inputs, immediate_plus=inputs['immediate_plus'].double()),
        dict(inputs, support_coverage=torch.full_like(inputs['support_coverage'], 1.01)),
        dict(inputs, support_coverage=inputs['support_coverage'][:, :, :-1]),
        dict(inputs, query_final=torch.full_like(inputs['query_final'], float('nan'))),
        dict(inputs, response_minus=torch.full_like(inputs['response_minus'], float('inf'))),
    ]
    for bad in bad_cases:
        reject(lambda bad=bad: model(**bad))
    reject(lambda: model(**static_inputs))
    reject(lambda: model(**inputs, static=True))
    reject(lambda: support_attention(query, support, weights, query_chunk_size=0))
    reject(lambda: support_attention(query, support, weights[:, :-1]))
    reject(lambda: mask_to_coverage(torch.full_like(gold, -.1), (2, 2)))
    reject(lambda: segmentation_loss(small, torch.ones(1, 1, 0, 1)))
    reject(lambda: segmentation_loss(small, torch.full((1, 1, 2, 2), float('nan'))))
    record('finite_shape_dtype_coverage_missing_stream_and_empty_image_validation', rejected_cases=14)

    # Tiny SAME-IMAGE supervised feature fixture, explicitly not transfer.
    # Coordinates are fixed inputs; the label merely defines a synthetic task.
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 6), torch.linspace(-1, 1, 8), indexing='ij')
    native = torch.stack((xx, yy, xx*yy, xx.square(), yy.square(),
                          torch.sin(2*xx), torch.cos(2*yy), torch.ones_like(xx)))[None]
    middle = torch.cat((native, native.square(), native*.5, torch.sin(native)), dim=1)
    target = ((xx > -.25) & (yy > -.5)).float()[None, None]
    fit_inputs = dict(support_final=native, support_middle=middle,
                      support_coverage=mask_to_coverage(target, (6, 8)),
                      query_final=native.clone(), query_middle=middle.clone(),
                      immediate_plus=.1*native, immediate_minus=-.2*native,
                      response_plus=native.square(), response_minus=torch.sin(native),
                      support_prefix=middle.flatten(2).transpose(1, 2)[:, :5],
                      query_prefix=middle.flatten(2).transpose(1, 2)[:, -5:],
                      prefix_immediate_plus=.1*native.flatten(2).transpose(1, 2)[:, -5:],
                      prefix_immediate_minus=-.2*native.flatten(2).transpose(1, 2)[:, -5:])
    fit_model = ReferenceResponseDecoder(8, 32, query_chunk_size=16)
    optimizer = torch.optim.Adam(fit_model.parameters(), lr=.002)
    with torch.no_grad():
        initial = segmentation_loss(fit_model(**fit_inputs), target)
        initial_loss = float(initial['loss'])
    for _ in range(48):
        optimizer.zero_grad(set_to_none=True)
        fit_loss = segmentation_loss(fit_model(**fit_inputs), target)['loss']
        assert torch.isfinite(fit_loss)
        fit_loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_logits = fit_model(**fit_inputs)
        final_loss = float(segmentation_loss(final_logits, target)['loss'])
        correct_patches = int(((final_logits > 0) == target.bool()).sum())
    assert final_loss < initial_loss*.5
    assert correct_patches == target.numel()
    record('controlled_same_image_supervised_fixture_fit_implementation_only',
           optimizer='Adam', learning_rate=.002, fixed_steps=48,
           initial_BCE_plus_softIoU=initial_loss, final_BCE_plus_softIoU=final_loss,
           correct_patches=correct_patches, total_patches=target.numel(),
           cross_image_validation=False, real_segmentation_benefit=False)

    actual = ReferenceResponseDecoder(1024, 4096)
    formula_count = 462*1024+132*4096+993537
    assert count(actual) == formula_count
    assert not torch.cuda.is_initialized()
    source = BASE/'tics/reference_response_decoder.py'
    report = dict(
        schema='reference_response_decoder_cpu_v3', state='PASS', seed=31027,
        checks=checks, passed=len(checks), torch_version=torch.__version__,
        architecture=dict(final_channels=1024, middle_channels=4096,
                          stream_channels=64, fusion_input_channels=768, fusion_channels=128,
                          prefix_input_channels=4096+2*1024, prefix_context_channels=64,
                          runtime_expected_prefix_tokens=5,
                          spatial_residual_blocks=3, convolutions_per_block=2,
                          group_norm_groups=8, query_chunk_size=256,
                          parameters=formula_count,
                          parameter_formula='462*final_channels+132*middle_channels+993537'),
        CPU_only=True, CUDA_initialized=False, DINO_executed=False,
        encoder_calls=0, CRF_executed=False, model_binary_saved=False,
        elapsed_seconds=time.monotonic()-started,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            'Synthetic CPU implementation evidence only; no real image, frozen encoder or cross-image benefit.',
            'Runtime adapter, exact original refiner and photo/class-disjoint task training are not tested here.',
            'Head consumes full selected-layer patch/prefix QKV inputs and immediate prefix responses, not all encoder hidden states.',
            'Immediate output differences are not asserted equivalent to full QKV or the full native hidden state.',
            'Chunk checkpointing bounds attention-score allocations but adds backward recomputation.',
        ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(state=report['state'], passed=report['passed'],
                          parameters=formula_count, CPU_only=True,
                          receipt=str(args.out), elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
