#!/usr/bin/env python3
"""Bounded CPU equivalence test against byte-identical pinned Meta SAM modules.

This is a correctness harness, not a performance benchmark. Official modules are
copied unchanged after hash verification; only package scaffolding is authored.
The reference calls the original MaskDecoder.forward/predict_masks. The candidate
reuses its Parameter objects and sparse/head modules, with projected factors and a
dense companion for the two image writes and LayerNorm statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIN = 'dca509fe793f601edb92606367a655c15ac00fdf'


def materialize_official():
    provenance = json.loads((ROOT / 'source_audit/source_provenance.json').read_text())
    selected = []
    package = HERE / 'official_sam'
    package.mkdir(exist_ok=True)
    (package / '__init__.py').write_text('# Local package scaffolding only.\n')
    for basename in ('common', 'transformer', 'mask_decoder'):
        record = next(r for r in provenance['files'] if r['project'] == 'sam' and
                      r['source_path'] == f'segment_anything/modeling/{basename}.py')
        src = ROOT / 'source_audit' / record['saved_path']
        data = src.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == record['sha256']
        assert record['revision_commit'] == PIN
        dest = package / f'{basename}.py'
        dest.write_bytes(data)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
        selected.append({**record, 'executed_copy': str(dest.relative_to(HERE)),
                         'executed_copy_sha256': digest})
    (HERE / 'executed_source_provenance.json').write_text(json.dumps({
        'copied_unchanged': True, 'package_scaffolding_only': True,
        'source_safety_inspection': 'All three complete sources inspected before import: only torch, math, typing, local common imports; no subprocess, network, filesystem side effects, eval/exec or dynamic source loading.',
        'files': selected}, indent=2) + '\n')
    sys.path.insert(0, str(HERE))
    transformer = importlib.import_module('official_sam.transformer')
    decoder = importlib.import_module('official_sam.mask_decoder')
    common = importlib.import_module('official_sam.common')
    return transformer, decoder, common


class Trace:
    """Non-mutating hooks record the actual official forward's module outputs."""
    def __init__(self, model, official):
        self.data = {}
        self.handles = []
        self.attn_type = official[0].Attention
        selected_types = (nn.Linear, nn.LayerNorm, nn.ConvTranspose2d, nn.GELU,
                          nn.ReLU, official[2].LayerNorm2d,
                          official[2].MLPBlock, official[1].MLP,
                          official[0].Attention, official[0].TwoWayAttentionBlock,
                          official[0].TwoWayTransformer)
        for name, module in model.named_modules():
            if name and isinstance(module, selected_types):
                self.handles.append(module.register_forward_hook(self.make_hook(name), with_kwargs=True))

    def put(self, name, value):
        if isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.put(f'{name}.{i}', item)
        else:
            assert name not in self.data, f'duplicate stage {name}'
            self.data[name] = value.detach()

    def make_hook(self, name):
        def hook(module, args, kwargs, output):
            self.put(name + '.output', output)
            if isinstance(module, nn.LayerNorm):
                self.put(name + '.input', args[0])
                if name.endswith('.norm4'):
                    _, mean, rstd = torch.native_layer_norm(args[0], module.normalized_shape,
                                                          module.weight, module.bias, module.eps)
                    self.put(name + '.mean', mean)
                    self.put(name + '.rstd', rstd)
            if isinstance(module, self.attn_type):
                q = self.data[name + '.q_proj.output']
                k = self.data[name + '.k_proj.output']
                qh = module._separate_heads(q, module.num_heads)
                kh = module._separate_heads(k, module.num_heads)
                logits = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
                self.put(name + '.logits', logits)
                self.put(name + '.probabilities', torch.softmax(logits, dim=-1))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


class SharedCache:
    """Image-specific transformed bases and fixed PE projections, built once."""
    def __init__(self, model, image, dense_prompt, pe):
        assert image.shape[0] == 1 and pe.shape[0] == 1
        assert dense_prompt.shape[0] == 1, 'Bounded test covers shared dense prompt only'
        self.model = model
        self.h, self.w = image.shape[-2:]
        self.pe = pe.flatten(2).transpose(1, 2)
        self.bases = [(image + dense_prompt).flatten(2).transpose(1, 2)]
        for block in model.transformer.layers:
            old = self.bases[-1]
            self.bases.append((old - old.mean(-1, keepdim=True)) * block.norm4.weight)
        self.projections = {}
        for i, block in enumerate(model.transformer.layers):
            for role in ('k_proj', 'v_proj'):
                self.add(f'transformer.layers.{i}.cross_attn_token_to_image.{role}',
                         getattr(block.cross_attn_token_to_image, role), i, role == 'k_proj')
            self.add(f'transformer.layers.{i}.cross_attn_image_to_token.q_proj',
                     block.cross_attn_image_to_token.q_proj, i, True)
        for role in ('k_proj', 'v_proj'):
            self.add(f'transformer.final_attn_token_to_image.{role}',
                     getattr(model.transformer.final_attn_token_to_image, role),
                     len(model.transformer.layers), role == 'k_proj')
        conv = model.output_upscaling[0]
        assert conv.kernel_size == (2, 2) and conv.stride == (2, 2)
        assert conv.padding == (0, 0) and conv.output_padding == (0, 0)
        assert conv.dilation == (1, 1) and conv.groups == 1
        # ConvTranspose weight [input_channel, output_channel, dy, dx].
        # Map output order [dy, dx, output_channel], then interleave spatially.
        self.conv_matrix = conv.weight.permute(0, 2, 3, 1).reshape(conv.in_channels, -1)
        self.conv_base = self.bases[-1] @ self.conv_matrix
        self.conv_bias = conv.bias.repeat(4)

    def add(self, name, linear, index, uses_pe):
        self.projections[name] = {
            'base': F.linear(self.bases[index], linear.weight, None),
            'pe': F.linear(self.pe, linear.weight, None) if uses_pe else None,
        }


class FactorState:
    def __init__(self, cache, batch):
        e = cache.bases[0]
        self.cache = cache
        self.level = 0
        self.s = e.new_ones(batch, e.shape[1], 1)
        self.u = e.new_empty(batch, e.shape[1], 0)
        self.v = e.new_empty(batch, 0, e.shape[-1])
        self.dense = e.expand(batch, -1, -1)
        self.has_constant = False
        self.widths = []
        self.residuals = []

    def reconstruct(self):
        return self.s * self.cache.bases[self.level] + self.u @ self.v

    def project(self, linear, name, trace):
        cached = self.cache.projections[name]
        out = self.s * cached['base']
        if self.u.shape[-1]:
            out = out + self.u @ F.linear(self.v, linear.weight, None)
        if cached['pe'] is not None:
            out = out + cached['pe']  # Position is never multiplied by row scale.
        if linear.bias is not None:
            out = out + linear.bias
        trace.put(name + '.output', out)
        return out

    def write_and_normalize(self, a_u, a_v, constant, norm, name, trace):
        update = a_u @ a_v + constant[:, None, :]
        pre = self.dense + update
        normalized, mean, rstd = torch.native_layer_norm(pre, norm.normalized_shape,
                                                       norm.weight, norm.bias, norm.eps)
        if self.has_constant:
            # Last column remains exactly one immediately after a LayerNorm.
            assert torch.equal(self.u[..., -1], torch.ones_like(self.u[..., -1]))
            old_v = torch.cat((self.v[:, :-1], self.v[:, -1:] + constant[:, None]), dim=1)
            u = torch.cat((self.u, a_u), dim=-1)
            v = torch.cat((old_v, a_v), dim=1)
        else:
            one = self.s.new_ones(*self.s.shape)
            u = torch.cat((self.u, a_u, one), dim=-1)
            v = torch.cat((self.v, a_v, constant[:, None]), dim=1)
        self.s = self.s * rstd
        self.u = torch.cat((u * rstd, torch.ones_like(rstd)), dim=-1)
        centered_v = (v - v.mean(-1, keepdim=True)) * norm.weight
        self.v = torch.cat((centered_v, norm.bias[None, None].expand(v.shape[0], 1, -1)), dim=1)
        self.has_constant = True
        self.level += 1
        self.dense = normalized
        self.widths.append(self.u.shape[-1])
        reconstruction = self.reconstruct()
        self.residuals.append(metrics(normalized, reconstruction))
        trace.put(name + '.input', pre)
        trace.put(name + '.output', normalized)
        trace.put(name + '.mean', mean)
        trace.put(name + '.rstd', rstd)
        return update


def read_attention(attn, q_input, state, name, trace):
    q = attn.q_proj(q_input)  # Official sparse module, hooks record it.
    k = state.project(attn.k_proj, name + '.k_proj', trace)
    v = state.project(attn.v_proj, name + '.v_proj', trace)
    qh = attn._separate_heads(q, attn.num_heads)
    kh = attn._separate_heads(k, attn.num_heads)
    vh = attn._separate_heads(v, attn.num_heads)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
    probabilities = torch.softmax(logits, dim=-1)
    out = attn.out_proj(attn._recombine_heads(probabilities @ vh))
    trace.put(name + '.logits', logits)
    trace.put(name + '.probabilities', probabilities)
    trace.put(name + '.output', out)
    return out


def write_attention(attn, sparse, query_pe, state, name, trace):
    q = state.project(attn.q_proj, name + '.q_proj', trace)
    k = attn.k_proj(sparse + query_pe)
    v = attn.v_proj(sparse)
    qh = attn._separate_heads(q, attn.num_heads)
    kh = attn._separate_heads(k, attn.num_heads)
    vh = attn._separate_heads(v, attn.num_heads)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
    probabilities = torch.softmax(logits, dim=-1)
    batch, heads, n, token_count = probabilities.shape
    head_width = vh.shape[-1]
    # Move official output projection onto each head's sparse projected values.
    out_weight = attn.out_proj.weight.reshape(-1, heads, head_width).permute(1, 2, 0)
    c = vh @ out_weight
    a_u = probabilities[..., :-1].permute(0, 2, 1, 3).reshape(batch, n, heads * (token_count - 1))
    a_v = (c[..., :-1, :] - c[..., -1:, :]).reshape(batch, heads * (token_count - 1), -1)
    constant = c[..., -1, :].sum(dim=1) + attn.out_proj.bias
    update = a_u @ a_v + constant[:, None, :]
    trace.put(name + '.logits', logits)
    trace.put(name + '.probabilities', probabilities)
    trace.put(name + '.out_proj.output', update)
    trace.put(name + '.output', update)
    return a_u, a_v, constant


def factored_predict(model, cache, sparse_prompt, trace):
    batch = sparse_prompt.shape[0]
    output_tokens = torch.cat((model.iou_token.weight, model.mask_tokens.weight), dim=0)
    tokens = torch.cat((output_tokens[None].expand(batch, -1, -1), sparse_prompt), dim=1)
    q = tokens
    state = FactorState(cache, batch)
    for i, block in enumerate(model.transformer.layers):
        name = f'transformer.layers.{i}'
        if block.skip_first_layer_pe:
            q = block.self_attn(q=q, k=q, v=q)  # Deliberately no residual or PE.
        else:
            q_with_pe = q + tokens
            q = q + block.self_attn(q=q_with_pe, k=q_with_pe, v=q)
        q = block.norm1(q)
        q = block.norm2(q + read_attention(block.cross_attn_token_to_image, q + tokens,
                                          state, name + '.cross_attn_token_to_image', trace))
        q = block.norm3(q + block.mlp(q))
        a_u, a_v, constant = write_attention(block.cross_attn_image_to_token, q, tokens, state,
                                            name + '.cross_attn_image_to_token', trace)
        state.write_and_normalize(a_u, a_v, constant, block.norm4, name + '.norm4', trace)
        trace.put(name + '.output', (q, state.dense))
    q = model.transformer.norm_final_attn(q + read_attention(
        model.transformer.final_attn_token_to_image, q + tokens, state,
        'transformer.final_attn_token_to_image', trace))
    trace.put('transformer.output', (q, state.dense))

    # Exact linear map for nonoverlapping first deconvolution, followed by
    # official per-child 64-channel LayerNorm2d, GELUs and second deconvolution.
    pre = state.s * cache.conv_base + state.u @ (state.v @ cache.conv_matrix) + cache.conv_bias
    child_channels = model.output_upscaling[0].out_channels
    pre = pre.reshape(batch, cache.h, cache.w, 2, 2, child_channels)
    pre = pre.permute(0, 5, 1, 3, 2, 4).reshape(batch, child_channels, 2 * cache.h, 2 * cache.w)
    trace.put('output_upscaling.0.output', pre)
    upscaled = pre
    for module in model.output_upscaling[1:]:
        upscaled = module(upscaled)
    mask_tokens = q[:, 1:1 + model.num_mask_tokens, :]
    hyper = torch.stack([mlp(mask_tokens[:, i]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], dim=1)
    b, c, h, w = upscaled.shape
    masks = (hyper @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
    iou = model.iou_prediction_head(q[:, 0, :])
    return masks, iou, state


def metrics(reference, candidate):
    assert reference.shape == candidate.shape
    delta = (candidate - reference).abs()
    floor = 1e-12 if reference.dtype == torch.float64 else 1e-6
    rel = delta / reference.abs().clamp_min(floor)
    scale = reference.abs().max().item()
    return {
        'shape': list(reference.shape), 'max_abs': delta.max().item(),
        'mean_abs': delta.mean().item(), 'max_rel_clamped': rel.max().item(),
        'mean_rel_clamped': rel.mean().item(), 'relative_denominator_floor': floor,
        'max_abs_reference': scale,
        'normalized_linf': delta.max().item() / max(scale, floor),
        'rms_error': delta.square().mean().sqrt().item(),
        'finite_reference': bool(torch.isfinite(reference).all()),
        'finite_candidate': bool(torch.isfinite(candidate).all()),
    }


def decision_metrics(masks, proposed_masks, iou, proposed_iou):
    differences = (masks > 0) != (proposed_masks > 0)
    max_error = (masks - proposed_masks).abs().max()
    sorted_iou = iou.sort(dim=-1, descending=True).values
    return {
        'binary_mask_threshold': 0.0,
        'binary_mask_disagreements': int(differences.sum()),
        'mask_logits_count': masks.numel(),
        'minimum_abs_mask_logit': masks.abs().min().item(),
        'reference_logits_within_max_abs_error_of_zero': int((masks.abs() <= max_error).sum()),
        'max_abs_ref_logit_among_disagreements': masks[differences].abs().max().item() if bool(differences.any()) else None,
        'argmax_all_four_iou_disagreements': int((iou.argmax(-1) != proposed_iou.argmax(-1)).sum()),
        'argmax_multimask_iou_disagreements': int((iou[:, 1:].argmax(-1) != proposed_iou[:, 1:].argmax(-1)).sum()),
        'all_four_top_two_iou_margin_min': (sorted_iou[:, 0] - sorted_iou[:, 1]).min().item(),
        'note': 'SAM MaskDecoder slices token0 for single output and tokens1:4 for multi output. Argmax diagnostics are additional downstream sensitivity checks, not its selection rule.',
    }


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_case(official, seed, dtype_name, grid, tokens):
    start = time.perf_counter()
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(seed)
    model = official[1].MaskDecoder(transformer_dim=256,
                transformer=official[0].TwoWayTransformer(depth=2, embedding_dim=256,
                    num_heads=8, mlp_dim=2048)).to(device='cpu', dtype=dtype).eval()
    # Official initialization has beta=0 and gamma=1. Deliberately use nontrivial
    # affine values to test trained-shape semantics without any checkpoint.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, official[2].LayerNorm2d)):
                module.weight.copy_(1 + 0.15 * torch.randn_like(module.weight))
                module.bias.copy_(0.15 * torch.randn_like(module.bias))
    image = torch.randn(1, 256, grid, grid, dtype=dtype)
    pe = torch.randn_like(image)
    sparse = torch.randn(2, tokens - 5, 256, dtype=dtype)
    # Same dense embedding (a no-mask-style shared channel vector) for both prompts.
    dense = torch.randn(1, 256, 1, 1, dtype=dtype).expand(1, 256, grid, grid)
    args = dict(image_embeddings=image, image_pe=pe, sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense.expand(2, -1, -1, -1))
    ref_trace = Trace(model, official)
    ref_start = time.perf_counter()
    ref_masks, ref_iou = model.predict_masks(**args)
    ref_elapsed = time.perf_counter() - ref_start
    ref_trace.close()

    cache_start = time.perf_counter()
    cache = SharedCache(model, image, dense, pe)
    cache_elapsed = time.perf_counter() - cache_start
    candidate_trace = Trace(model, official)
    candidate_start = time.perf_counter()
    masks, iou, state = factored_predict(model, cache, sparse, candidate_trace)
    candidate_elapsed = time.perf_counter() - candidate_start
    candidate_trace.close()

    missing = sorted(set(ref_trace.data) - set(candidate_trace.data))
    extra = sorted(set(candidate_trace.data) - set(ref_trace.data))
    assert not missing and not extra, (missing, extra)
    comparisons = {name: metrics(ref_trace.data[name], candidate_trace.data[name]) for name in ref_trace.data}
    comparisons['all_four_masks'] = metrics(ref_masks, masks)
    comparisons['all_four_iou'] = metrics(ref_iou, iou)
    # Both real public forward slicing modes, no replacement reference.
    slicing = {}
    for mode in (False, True):
        native_masks, native_iou = model(**args, multimask_output=mode)
        selection = slice(1, None) if mode else slice(0, 1)
        slicing[str(mode)] = {
            'official_forward_matches_predict_masks': torch.equal(native_masks, ref_masks[:, selection]) and torch.equal(native_iou, ref_iou[:, selection]),
            'mask_error': metrics(native_masks, masks[:, selection]),
            'iou_error': metrics(native_iou, iou[:, selection]),
        }
    tolerance = 1e-10 if dtype == torch.float64 else 5e-5
    failed = [name for name, m in comparisons.items() if m['max_abs'] > tolerance or not m['finite_candidate'] or not m['finite_reference']]
    for level, m in enumerate(state.residuals):
        if m['max_abs'] > tolerance:
            failed.append(f'factor_reconstruction_after_layer_{level}')
    expected_widths = [8 * (tokens - 1) + 2, 2 * (8 * (tokens - 1) + 1) + 1]
    assert state.widths == expected_widths
    worst = max(comparisons, key=lambda k: comparisons[k]['max_abs'])
    result = {
        'seed': seed, 'dtype': dtype_name, 'image_grid': [grid, grid], 'N': grid * grid,
        'prompts': 2, 'T': tokens, 'D': 256, 'H': 8, 'MLP_width': 2048,
        'model_state_sha256': parameter_hash(model),
        'model_mode': 'eval, inference_mode, CPU, random weights, nondefault LayerNorm affine',
        'shared_dense_prompt': 'one spatially constant channel vector, repeated across two prompts',
        'transformer_ln_eps': [block.norm4.eps for block in model.transformer.layers],
        'upscaler_ln_eps': model.output_upscaling[1].eps,
        'factor_widths_after_LN': state.widths,
        'factor_reconstruction_errors': state.residuals,
        'stage_count': len(comparisons), 'stage_errors': comparisons,
        'absolute_tolerance': tolerance, 'passed': not failed,
        'failed_stages': failed, 'worst_stage': worst,
        'worst_max_abs': comparisons[worst]['max_abs'],
        'decisions': decision_metrics(ref_masks, masks, ref_iou, iou),
        'slicing_modes': slicing,
        'execution_budget_seconds_only': {'reference_with_hooks': ref_elapsed,
             'cache_build': cache_elapsed, 'candidate_with_hooks': candidate_elapsed,
             'whole_case': time.perf_counter() - start},
        'timing_warning': 'Diagnostic hook overhead, uncompiled CPU execution, no warm-up: these durations are an execution budget record, not throughput/speedup evidence.',
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='One small FP64 case for implementation diagnosis only')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    official = materialize_official()
    configurations = [(s, dt, 8, 7) for dt in ('float64', 'float32') for s in (0, 1, 2)]
    configurations += [(0, dt, 8, 9) for dt in ('float64', 'float32')]
    configurations += [(0, dt, 64, 7) for dt in ('float64', 'float32')]
    if args.quick:
        configurations = configurations[:1]
    output = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'official_pin': PIN, 'torch': torch.__version__, 'python': platform.python_version(),
        'platform': platform.platform(), 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
        'scope': 'Random-weight official SAM MaskDecoder; projected-factor implementation with dense companion; no SAM2, pretrained checkpoint, dense-mask initial rank16, low precision, GPU, timing claim or implicit-attention variant.',
        'cases': [],
    }
    output_name = 'quick_results.json' if args.quick else 'results.json'
    with torch.inference_mode():
        for seed, dtype, grid, tokens in configurations:
            result = run_case(official, seed, dtype, grid, tokens)
            output['cases'].append(result)
            (HERE / output_name).write_text(json.dumps(output, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ('seed', 'dtype', 'N', 'T', 'passed', 'worst_stage', 'worst_max_abs', 'failed_stages', 'factor_widths_after_LN')}), flush=True)
            if not result['passed']:
                raise AssertionError('Fixed predeclared tolerance failed; inspect first diverging stage, do not tune tolerance')
    output['all_passed'] = all(c['passed'] for c in output['cases'])
    output['total_cases'] = len(output['cases'])
    output['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / output_name).write_text(json.dumps(output, indent=2) + '\n')
    print(f'Wrote {HERE / output_name}', flush=True)


if __name__ == '__main__':
    main()
