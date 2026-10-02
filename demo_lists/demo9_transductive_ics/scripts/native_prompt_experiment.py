#!/usr/bin/env python3
"""E5: support-only existing-prefix prompt fitting on the actual frozen encoder.

Requires the external CUDA/resource guard. No downloads, feature replay, native
weight updates, query-label fitting, dynamic precision/resolution/OOM retries.
Final maps use one matched full-anchor FROST-style decoder without APD; the
unchanged complete INSID3 prediction is a separately labelled strong baseline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from tics.reference_prompt import ReferencePrefixPrompt, fit_reference_prompt
from tics.readout_baselines import frost_style_dense_readout
from tics.native_assets import reuse_native_basis
from global_representation_probe import fixed_selection, summary, pixel_ledger


@contextmanager
def checkpoint_frozen_blocks(backbone, enabled=True):
    """Predeclared memory policy, non-reentrant checkpoint of actual block forward.

    Injection pre-hook runs once before first block; checkpoint recomputation
    calls its ORIGINAL bound forward on the already injected input. All other
    arguments (including actual RoPE tensors) remain untouched. No handcrafted
    Eva math. Evaluation/no-grad calls follow original forwards directly.
    """
    originals = [(block, block.forward, 'forward' in block.__dict__) for block in backbone.blocks]
    try:
        if enabled:
            for block, original, _ in originals:
                def wrapped(this, *args, _original=original, **kwargs):
                    x = args[0] if args else kwargs.get('x')
                    if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad:
                        return checkpoint(_original, *args, use_reentrant=False,
                                          preserve_rng_state=True, **kwargs)
                    return _original(*args, **kwargs)
                block.forward = MethodType(wrapped, block)
        yield
    finally:
        for block, original, was_instance in originals:
            if enabled:
                if was_instance:
                    block.forward = original
                else:
                    del block.forward


def spatial_regions(grid, device):
    """Fixed 4x4 contiguous tile groups, independent of every label/GT score."""
    height, width = grid
    y = torch.arange(height, device=device) * 4 // height
    x = torch.arange(width, device=device) * 4 // width
    return (4 * y[:, None] + x[None]).flatten()


def direct_features(backbone, images, *, cuda=True):
    """Actual gradient-enabled timm final norm path; no decorated extractor."""
    with torch.autocast('cuda' if cuda else 'cpu', dtype=torch.bfloat16, enabled=cuda):
        maps = backbone.forward_intermediates(images, indices=1, norm=True,
                                             output_fmt='NCHW', intermediates_only=True)[0]
    if maps.ndim != 4:
        raise RuntimeError('Actual encoder did not return BCHW final norm maps')
    return maps.float()


def compact_fit(value):
    return {key: item for key, item in value.items() if key not in ('history', 'coefficients')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', default='/root/autodl-tmp/demo9_transductive_ics')
    parser.add_argument('--prepared-manifest')
    parser.add_argument('--projection-basis', help='Optional typed native normalized-black positional basis; avoids repeat SVD')
    parser.add_argument('--out')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--checkpoint-blocks', choices=('on', 'off'), default='on')
    parser.add_argument('--selfcheck', '--self-check', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(args.out)
        return
    if not args.out or not args.prepared_manifest:
        parser.error('--out and CPU-prepared --prepared-manifest are required')
    if args.steps < 1 or args.limit < 1:
        parser.error('Positive steps/limit required; actual optimization must be tested')
    # The outer guard is responsible for owner/memory/time/provider shutdown.
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Do not construct prompt/backbone inside inference_mode; support backward requires normal tensors')
    if os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise RuntimeError('Launch only through CUDA/resource guard with DEMO9_CUDA_GUARD=1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no automatic CPU/model-loading fallback')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'report.json').exists():
        raise RuntimeError('Fresh output required; existing evidence is preserved')
    files = [Path(__file__), HERE.parent / 'tics/reference_prompt.py',
             HERE.parent / 'tics/readout_baselines.py', HERE.parent / 'tics/native_assets.py',
             HERE / 'global_representation_probe.py']
    report = dict(state='PREPARING', args=vars(args), seed=0, records=[],
        source_sha256={str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
        contract=dict(encoder='Existing frozen DINOv3 ViT-L/16 L24 C1024 prefix5, native BF16',
            fitting='B1 legal original support only, fixed4x4 spatial regions; independent held validation',
            final='Ordered B2 support/query plus B1 horizontally flipped support for every reader',
            memory='Predeclared non-reentrant actual-block checkpoint during differentiable fit only',
            reader='Same full FG/BG anchors, support-only whiten/LOO bandwidth, native original query geometry/RGB; no APD',
            controls='Unchanged full INSID3; zero/native and fixed-prefix matched fitting budget',
            exposure='All branch masks frozen before evaluator opens query annotation',
            fallback='No legal support region split or degenerate anchors -> explicit native reader/host fallback'),
        scope='Fixed development cohort; CPU fixture acceptance is not real DINO/gain evidence')
    def save():
        report['class_miou'] = summary(report['records'])
        temporary = out / 'report.tmp'
        temporary.write_text(json.dumps(report, allow_nan=False))
        temporary.replace(out / 'report.json')
    save()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC', '.3')))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    started = time.monotonic()
    current_episode = None
    try:
        sys.path.insert(0, str(Path(args.prepared_root) / 'scripts'))
        import _paths
        sys.path.insert(0, _paths.DEMO4)
        from icx.common import TimmDINOv3, coco_episodes, DEV
        from utils.data import build_transform, load_image, load_mask, downsample_mask
        from models.insid3 import INSID3
        from PIL import Image
        episodes, _, base = coco_episodes(args.fold, 400, shot=1, seed=0)
        selected = fixed_selection(episodes, args.limit)
        frozen = [dict(e=e, c=c, support=refs[0], query=q) for e, (c, q, refs) in selected]
        manifest_path = Path(args.prepared_manifest)
        manifest = json.loads(manifest_path.read_text())
        if frozen != manifest['frozen_episodes'] or manifest['state'] != 'PREPARED_ASSETS':
            raise RuntimeError('Prepared exact IDs/state differ; encoder not yet constructed')
        report.update(frozen_episodes=frozen, prepared_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        save()
        with torch.no_grad():
            encoder = TimmDINOv3().to(DEV).eval()
        backbone = encoder.m
        for parameter in backbone.parameters():
            if torch.is_inference(parameter):
                raise RuntimeError('Backbone parameter is an inference tensor')
            parameter.requires_grad_(False)
        if len(backbone.blocks) != 24 or int(getattr(backbone, 'num_prefix_tokens', -1)) != 5:
            raise RuntimeError('Expected actual 24-layer five-prefix frozen Eva backbone')
        report['actual_forward_intermediates_sha256'] = hashlib.sha256(
            inspect.getsource(type(backbone).forward_intermediates).encode()).hexdigest()
        report['torch_version'] = torch.__version__
        parameter_versions = [(parameter, parameter._version) for parameter in backbone.parameters()]
        call_log = []
        context = ['host_initialization']
        original_get = encoder.get_intermediate_layers
        # Count host calls as well as explicitly counted direct calls; values unchanged.
        def counted_get(this, images, *positional, **keywords):
            call_log.append(dict(path='native_wrapper', stage=context[0], batch=int(images.shape[0]), grad=torch.is_grad_enabled()))
            return original_get(images, *positional, **keywords)
        encoder.get_intermediate_layers = MethodType(counted_get, encoder)
        try:
            with torch.no_grad(), reuse_native_basis(INSID3, args.projection_basis) as basis_receipt:
                host = INSID3(encoder=encoder, image_size=1024, svd_components=500, tau=.6,
                              merge_threshold=.2, mask_refiner='bilinear', resize_to_orig_size=False, device=DEV).eval()
            if torch.is_inference(host.positional_basis):
                raise RuntimeError('Host positional basis is an inference tensor')
            report['projection_basis'] = basis_receipt or dict(source_input='normalized_black_image',
                state='ORIGINAL_INSID3_CONSTRUCTOR_BASIS')
            report['host_initialization_calls'] = list(call_log)
            transform = build_transform(1024)
            report['state'] = 'RUNNING'
            save()
            def encode(images):
                call_log.append(dict(path='direct_actual_backbone', stage=context[0], batch=int(images.shape[0]), grad=torch.is_grad_enabled()))
                return direct_features(backbone, images)
            for e, (c, query_name, refs) in selected:
                current_episode = e
                call_start = len(call_log)
                torch.cuda.reset_peak_memory_stats()
                episode_started = time.monotonic()
                support_pil = Image.open(Path(base) / refs[0]).convert('RGB')
                query_pil = Image.open(Path(base) / query_name).convert('RGB')
                support_gt = torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN) /
                    str(Path(refs[0]).with_suffix('.png')))) == c + 1).copy())
                support = load_image(support_pil, transform, DEV)[0]
                query = load_image(query_pil, transform, DEV)[0]
                support_mask = load_mask(support_gt, 1024, DEV)
                flipped = support.flip(-1)
                flipped_mask = support_mask.flip(-1)
                context[0] = 'native_reference_final'
                with torch.no_grad():
                    native = encode(torch.cat((support, query)))
                    native_flip = encode(flipped)
                if native.shape != (2, 1024, 64, 64):
                    raise RuntimeError('Expected actual 1024 input ->64x64 C1024 features')
                grid = tuple(native.shape[-2:])
                masks = [downsample_mask(mask.unsqueeze(1), *grid).reshape(*grid).bool()
                         if mask.any() else torch.zeros(grid, dtype=torch.bool, device=DEV)
                         for mask in (support_mask, flipped_mask)]
                region_ids = spatial_regions(grid, DEV)
                rgb = torch.from_numpy(np.asarray(query_pil).copy()).to(DEV).permute(2, 0, 1).float() / 255
                rgb_grid = F.interpolate(rgb[None], grid, mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
                predictions, reader_info, fits, costs = {}, {}, {}, {}
                context[0] = 'insid3_native'
                with torch.no_grad():
                    host_prediction = (host.predict_mask(support, support_mask, query).reshape(1024, 1024).bool()
                                       if support_mask.any() else torch.zeros((1024, 1024), dtype=torch.bool, device=DEV))
                predictions['insid3_native'] = host_prediction
                def decode(name, maps, flip_maps):
                    if any(not mask.any() or mask.all() for mask in masks):
                        predictions[name] = host_prediction.clone()
                        reader_info[name] = dict(state='DEGENERATE_ANCHORS_NATIVE_HOST_FALLBACK')
                        return
                    prediction, info = frost_style_dense_readout([maps[0], flip_maps[0]], masks,
                        maps[1], native[1], rgb_grid, float(support_gt.float().mean()), (1024, 1024))
                    predictions[name] = prediction.clone()
                    reader_info[name] = dict(state=info.get('state', 'DENSE'), sigma=info['sigma'],
                        loo_margin=info['loo_margin'], candidate_patches=int(info['candidate_grid'].sum()),
                        W_source='arm_full_support_only', sigma_source='arm_support_LOO',
                        geometry_source='original_native_SQ_query_final', APD=False)
                with torch.no_grad():
                    decode('native_dense', native, native_flip)
                for mode in ('native', 'adaptive', 'fixed'):
                    name = 'prompt_' + mode
                    context[0] = name + '_fit_B1'
                    prompt = ReferencePrefixPrompt(backbone, 5, 1024, rank=32,
                                                   relative_radius=.1, seed=4101)
                    fit_started = time.monotonic()
                    with checkpoint_frozen_blocks(backbone, args.checkpoint_blocks == 'on'):
                        fit = fit_reference_prompt(prompt,
                            lambda image: encode(image)[0].flatten(1).T, support, masks[0].flatten(),
                            region_ids, steps=args.steps, mode=mode)
                    for key, value in compact_fit(fit).items():
                        if isinstance(value, float) and not math.isfinite(value):
                            raise RuntimeError('Nonfinite support fitting statistic: ' + key)
                    if not torch.isfinite(prompt.coefficients).all():
                        raise RuntimeError('Nonfinite prompt coefficients')
                    torch.cuda.synchronize()
                    fit_seconds = time.monotonic() - fit_started
                    # All modes zero-fallback when region split is invalid (fixed is not applied).
                    output_mode = mode if fit['state'] == 'FROZEN_FOR_QUERY' else 'native'
                    context[0] = name + '_final_B2_then_B1'
                    with torch.no_grad(), prompt.activate(mode=output_mode):
                        maps = encode(torch.cat((support, query)))
                        flip_maps = encode(flipped)
                    if mode == 'native' or output_mode == 'native':
                        if not torch.equal(maps, native) or not torch.equal(flip_maps, native_flip):
                            raise RuntimeError('Zero-prompt actual native final maps differ')
                    with torch.no_grad():
                        decode(name, maps, flip_maps)
                    if mode == 'native' and not torch.equal(predictions[name], predictions['native_dense']):
                        raise RuntimeError('Matched native prompt reader mask differs from native_dense')
                    fits[name] = compact_fit(fit)
                    fits[name]['output_mode'] = output_mode
                    costs[name] = dict(fit_seconds=fit_seconds, support_feature_calls=fit.get('feature_calls', 0),
                        backward_steps=fit.get('steps', 0), final_forward_calls=2, fit_batch=1, final_batches=[2, 1],
                        checkpoint_blocks=args.checkpoint_blocks, prompt_trainable_parameters=32)
                    if any(parameter._version != version for parameter, version in parameter_versions):
                        raise RuntimeError('Frozen backbone parameter version changed')
                    if prompt._handle is not None or any(parameter.grad is not None for parameter in backbone.parameters()):
                        raise RuntimeError('Prompt hook leaked or frozen backbone received gradients')
                    del prompt, maps, flip_maps
                # Freeze every inference output; no query annotation has been opened.
                predictions = {name: prediction.detach().clone() for name, prediction in predictions.items()}
                torch.cuda.synchronize()
                frozen_time = time.monotonic() - episode_started
                row = dict(e=e, c=c, support=refs[0], query=query_name, iu={}, original_iu={}, pixels={},
                    fit=fits, fit_cost=costs, reader_info=reader_info, encoder_calls=call_log[call_start:],
                    outputs_frozen_before_query_gt=True, predictions_frozen_s=frozen_time,
                    native_prompt_noop_exact=True, backbone_versions_unchanged=True,
                    frozen_backbone_has_no_gradients=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                # Save label-free inference/fit receipt before evaluator reads any query GT.
                report['pending_frozen_episode'] = row
                save()
                query_gt = torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN) /
                    str(Path(query_name).with_suffix('.png')))) == c + 1).copy())
                truth = load_mask(query_gt, 1024, DEV)[0].bool()
                for name, prediction in predictions.items():
                    row['iu'][name] = [int((prediction & truth).sum()), int((prediction | truth).sum())]
                    original = F.interpolate(prediction[None, None].float(), tuple(query_gt.shape),
                                             mode='bilinear', align_corners=False)[0, 0] > .5
                    gold = query_gt.to(DEV)
                    row['original_iu'][name] = [int((original & gold).sum()), int((original | gold).sum())]
                    row['pixels'][name] = pixel_ledger(predictions['native_dense'], prediction, truth)
                report.pop('pending_frozen_episode', None)
                report['records'].append(row)
                report['elapsed_s'] = time.monotonic() - started
                save()
                print(json.dumps(dict(e=e, count=len(report['records']), scores=report['class_miou'],
                    fit_states={name: value['state'] for name, value in fits.items()},
                    calls=len(row['encoder_calls']), peak_bytes=row['peak_allocated_bytes'])), flush=True)
                del support, query, flipped, support_mask, flipped_mask, native, native_flip, masks, predictions
            report.update(state='COMPLETED', elapsed_s=time.monotonic() - started)
            save()
        finally:
            del encoder.get_intermediate_layers
    except BaseException as error:
        report.update(state='ERROR', error=repr(error), failed_episode=current_episode,
                      elapsed_s=time.monotonic() - started)
        if torch.cuda.is_initialized():
            report['failure_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['failure_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save()
        raise


def selfcheck(out=None):
    """Actual CPU autograd/SDPA in prepared fake blocks; no real-DINO assertion."""
    from reference_prompt_cpu import FakeEvaBlock
    from torch import nn
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = nn.Parameter(torch.randn(1, 5, 24) * .3)
            self.blocks = nn.ModuleList([FakeEvaBlock(), FakeEvaBlock()])
            self.norm = nn.LayerNorm(24)
        def forward_intermediates(self, images, indices, norm, output_fmt, intermediates_only):
            assert indices == 1 and norm and output_fmt == 'NCHW' and intermediates_only
            x = torch.cat((self.prefix.expand(len(images), -1, -1), images), dim=1)
            for block in self.blocks:
                x = block(x, rope=None)
            return [self.norm(x)[:, 5:].transpose(1, 2).reshape(len(x), 24, 4, 4)]
    torch.set_num_threads(1)
    cases = []
    for seed in range(10):
        torch.manual_seed(seed)
        model = Fixture().eval().requires_grad_(False)
        support = torch.randn(1, 16, 24)
        labels = torch.arange(16) % 2
        regions = torch.arange(16) // 4
        query = torch.randn(1, 16, 24)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        pointers = [block.forward.__func__ for block in model.blocks]
        with torch.no_grad():
            native = direct_features(model, query, cuda=False)
        fits = {}
        coefficients = {}
        # Checkpoint and uncheckpointed adaptive solutions should match.
        for mode, enabled in [('adaptive', False), ('adaptive', True), ('native', True), ('fixed', True)]:
            prompt = ReferencePrefixPrompt(model, 5, 24, rank=32)
            with checkpoint_frozen_blocks(model, enabled):
                result = fit_reference_prompt(prompt,
                    lambda value: direct_features(model, value, cuda=False)[0].flatten(1).T,
                    support, labels, regions, steps=2, mode=mode)
            assert result['state'] == 'FROZEN_FOR_QUERY' and result['feature_calls'] == 5
            assert result['gradient_norm'] > 0 if mode == 'adaptive' else True
            with torch.no_grad(), prompt.activate(mode=mode):
                output = direct_features(model, query, cuda=False)
            if mode == 'native':
                assert torch.equal(native, output)
            assert all(parameter.grad is None for parameter in model.parameters())
            assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
            assert all(block.forward.__func__ is pointer for block, pointer in zip(model.blocks, pointers))
            assert all(not block._forward_pre_hooks for block in model.blocks)
            key = mode + ('_checkpoint' if enabled else '_plain')
            coefficients[key] = prompt.coefficients.detach().clone()
            fits[key] = compact_fit(result)
        assert torch.equal(coefficients['adaptive_plain'], coefficients['adaptive_checkpoint'])
        try:
            with checkpoint_frozen_blocks(model):
                raise RuntimeError('cleanup fixture')
        except RuntimeError:
            pass
        assert all('forward' not in block.__dict__ for block in model.blocks)
        cases.append(dict(seed=seed, fit=fits))
    # Actual production decoder composition on the tiny fixture output.
    with torch.no_grad():
        support_maps = direct_features(model, support, cuda=False)
        prediction, reader = frost_style_dense_readout(
            [support_maps[0], support_maps[0].flip(-1)],
            [labels.reshape(4, 4).bool(), labels.reshape(4, 4).bool().flip(-1)],
            native[0], native[0], torch.rand(4, 4, 3), .5, (16, 16))
    assert prediction.shape == (16, 16) and prediction.dtype == torch.bool
    # Predetermined fixed coefficients must also zero-fallback for invalid splits.
    prompt = ReferencePrefixPrompt(model, 5, 24, rank=32)
    invalid = fit_reference_prompt(prompt,
        lambda value: direct_features(model, value, cuda=False)[0].flatten(1).T,
        support, torch.zeros_like(labels), regions, steps=2, mode='fixed')
    assert invalid['state'] == 'NO_LEGAL_REGION_SPLIT'
    with torch.no_grad(), prompt.activate(mode='native'):
        assert torch.equal(direct_features(model, query, cuda=False), native)
    assert spatial_regions((64, 64), 'cpu').shape == (4096,)
    assert spatial_regions((64, 64), 'cpu').unique().numel() == 16
    # Reuse context leaves a normal frozen tensor and restores the original method.
    class BasisHost:
        svd_components = 500
        def _build_positional_basis(self, device):
            raise AssertionError('Native SVD should not rerun in reuse fixture')
    original_builder = BasisHost._build_positional_basis
    with tempfile.TemporaryDirectory(prefix='demo9_prompt_basis_cpu_') as directory:
        basis_path = Path(directory) / 'native_basis.pt'
        torch.save(dict(state='NATIVE_BASIS_FROZEN', source_input='normalized_black_image',
                        basis=torch.eye(1024)[:, :500].contiguous()), basis_path)
        with torch.no_grad(), reuse_native_basis(BasisHost, basis_path) as receipt:
            host_basis = BasisHost()._build_positional_basis('cpu')
        assert receipt['source_input'] == 'normalized_black_image'
        assert BasisHost._build_positional_basis is original_builder
        assert not torch.is_inference(host_basis) and not host_basis.requires_grad
        differentiable_input = torch.ones(1, 1024, requires_grad=True)
        (differentiable_input @ host_basis).square().mean().backward()
        assert torch.isfinite(differentiable_input.grad).all()
    assert all(not torch.is_inference(parameter) for parameter in model.parameters())
    result = dict(state='CPU_SELFCHECK_PASSED', cases=cases, cases_count=10,
        grad_only_prompt=True, backbone_unchanged=True, native_noop_exact=True,
        checkpoint_adaptive_exact=True, hooks_and_forwards_restored=True,
        dense_decoder_composition_checked=True, invalid_split_zero_fallback_checked=True,
        native_basis_reuse_restored=True, native_basis_normal_tensor_backward_checked=True,
        support_forward_calls_per_fit=5, backward_steps_per_fit=2, fit_cases=40,
        torch_version=torch.__version__,
        source_sha256={str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in
            (Path(__file__), HERE.parent / 'tics/reference_prompt.py',
             HERE.parent / 'tics/readout_baselines.py', HERE.parent / 'tics/native_assets.py',
             HERE / 'reference_prompt_cpu.py')},
        gradient_feature_path='Direct final-norm actual-shaped forward_intermediates',
        cuda_initialized=torch.cuda.is_initialized(), query_gt_used=False,
        scope='Prepared fake Eva-shaped CPU SDPA blocks; no timm weights/GPU/task gain claim')
    if out:
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
