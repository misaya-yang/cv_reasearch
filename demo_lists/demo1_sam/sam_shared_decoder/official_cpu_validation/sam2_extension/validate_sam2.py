#!/usr/bin/env python3
"""Separate bounded SAM2 official decoder extension; no checkpoints or installs."""
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
PARENT = HERE.parent
ROOT = PARENT.parent
sys.path.insert(0, str(PARENT))
from validate_official import Trace, SharedCache, FactorState, write_attention, metrics, parameter_hash

PIN = '2b90b9f5ceec907a1c18123530e92e794ad901a4'


def materialize_sam2():
    provenance = json.loads((ROOT / 'source_audit/source_provenance.json').read_text())
    paths = ['sam2/modeling/sam/transformer.py', 'sam2/modeling/sam/mask_decoder.py',
             'sam2/modeling/sam2_utils.py', 'sam2/modeling/position_encoding.py',
             'sam2/utils/misc.py', 'LICENSE']
    selected = []
    for path in paths:
        r = next(r for r in provenance['files'] if r['project'] == 'sam2' and r['source_path'] == path)
        data = (ROOT / 'source_audit' / r['saved_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r['sha256'] and r['revision_commit'] == PIN
        target = HERE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == r['sha256']
        selected.append({**r, 'executed_copy': str(target.relative_to(HERE)), 'copy_is_identical': True})
    for package in ('sam2', 'sam2/modeling', 'sam2/modeling/sam', 'sam2/utils'):
        (HERE / package / '__init__.py').write_text('# Minimal local package scaffolding only.\n')
    (HERE / 'executed_source_provenance.json').write_text(json.dumps({
        'copied_unchanged': True,
        'safety_inspection': 'All complete source files read before import. Import-time dependencies are torch/numpy/PIL/tqdm and standard library, already installed. Video file IO, threads, decord/cv2 imports, CUDA extension and CUDA allocations occur only in unrelated functions/classes never called by this decoder test. Plain torch.device(cuda) default objects do not initialize a GPU. No network or subprocess operations are invoked.',
        'package_scaffolding': 'Only blank __init__.py files; no decoder or dependency definitions modified or stubbed.',
        'files': selected}, indent=2) + '\n')
    sys.path.insert(0, str(HERE))
    return (importlib.import_module('sam2.modeling.sam.transformer'),
            importlib.import_module('sam2.modeling.sam.mask_decoder'),
            importlib.import_module('sam2.modeling.sam2_utils'))


class Trace2(Trace):
    def __init__(self, model, official):
        self.data = {}
        self.handles = []
        self.attn_type = official[0].Attention
        selected_types = (nn.Linear, nn.LayerNorm, nn.ConvTranspose2d, nn.Conv2d,
            nn.GELU, nn.ReLU, official[2].LayerNorm2d, official[2].MLP,
            official[0].Attention, official[0].TwoWayAttentionBlock, official[0].TwoWayTransformer)
        for name, module in model.named_modules():
            if name and isinstance(module, selected_types):
                self.handles.append(module.register_forward_hook(self.make_hook(name), with_kwargs=True))

    def put(self, name, value):
        # SAM2 MLP reuses its activation Module at each hidden layer.
        if isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self.put(f'{name}.{i}', item)
        else:
            resolved, occurrence = name, 1
            while resolved in self.data:
                occurrence += 1
                resolved = f'{name}.call{occurrence}'
            self.data[resolved] = value.detach()

    def make_hook(self, name):
        parent_hook = super().make_hook(name)
        def hook(module, args, kwargs, output):
            parent_hook(module, args, kwargs, output)
            if name in ('output_upscaling.1', 'output_upscaling.4'):
                self.put(name + '.input', args[0])  # Explicit high-res skip boundary.
        return hook


def read_attention2(attn, q_input, state, name, trace):
    q = attn.q_proj(q_input)
    k = state.project(attn.k_proj, name + '.k_proj', trace)
    v = state.project(attn.v_proj, name + '.v_proj', trace)
    qh = attn._separate_heads(q, attn.num_heads)
    kh = attn._separate_heads(k, attn.num_heads)
    vh = attn._separate_heads(v, attn.num_heads)
    # Keep the official backend for reads. Writes must expose stochastic weights
    # to build factors; their explicit softmax is compared with native SDPA output.
    attended = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0)
    out = attn.out_proj(attn._recombine_heads(attended))
    diagnostic_logits = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
    trace.put(name + '.logits', diagnostic_logits)
    trace.put(name + '.probabilities', torch.softmax(diagnostic_logits, dim=-1))
    trace.put(name + '.output', out)
    return out


def factored_predict2(model, cache, sparse_prompt, highres, trace):
    assert model.pred_obj_scores and model.use_high_res_features
    batch = sparse_prompt.shape[0]
    output_tokens = torch.cat((model.obj_score_token.weight, model.iou_token.weight,
                               model.mask_tokens.weight), dim=0)
    tokens = torch.cat((output_tokens[None].expand(batch, -1, -1), sparse_prompt), dim=1)
    q = tokens
    state = FactorState(cache, batch)
    for i, block in enumerate(model.transformer.layers):
        name = f'transformer.layers.{i}'
        if block.skip_first_layer_pe:
            q = block.self_attn(q=q, k=q, v=q)
        else:
            qpe = q + tokens
            q = q + block.self_attn(q=qpe, k=qpe, v=q)
        q = block.norm1(q)
        q = block.norm2(q + read_attention2(block.cross_attn_token_to_image, q + tokens,
                                           state, name + '.cross_attn_token_to_image', trace))
        q = block.norm3(q + block.mlp(q))
        au, av, constant = write_attention(block.cross_attn_image_to_token, q, tokens, state,
                                           name + '.cross_attn_image_to_token', trace)
        state.write_and_normalize(au, av, constant, block.norm4, name + '.norm4', trace)
        trace.put(name + '.output', (q, state.dense))
    q = model.transformer.norm_final_attn(q + read_attention2(
        model.transformer.final_attn_token_to_image, q + tokens, state,
        'transformer.final_attn_token_to_image', trace))
    trace.put('transformer.output', (q, state.dense))

    pre = state.s * cache.conv_base + state.u @ (state.v @ cache.conv_matrix) + cache.conv_bias
    pre = pre.reshape(batch, cache.h, cache.w, 2, 2, 64)
    pre = pre.permute(0, 5, 1, 3, 2, 4).reshape(batch, 64, 2 * cache.h, 2 * cache.w)
    trace.put('output_upscaling.0.output', pre)
    _, ln1, act1, dc2, act2 = model.output_upscaling
    feat_s0, feat_s1 = highres
    upscaled = act1(ln1(pre + feat_s1))
    upscaled = act2(dc2(upscaled) + feat_s0)
    mask_tokens = q[:, 2:2 + model.num_mask_tokens, :]
    hyper = torch.stack([mlp(mask_tokens[:, i]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], dim=1)
    b, c, h, w = upscaled.shape
    masks = (hyper @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
    iou = model.iou_prediction_head(q[:, 1, :])
    obj = model.pred_obj_score_head(q[:, 0, :])
    return (masks, iou, mask_tokens, obj), state


def select_outputs(model, outputs, multimask):
    masks, iou, tokens, obj = outputs
    if multimask:
        masks, iou = masks[:, 1:], iou[:, 1:]
    elif model.dynamic_multimask_via_stability:
        masks, iou = model._dynamic_multimask_via_stability(masks, iou)
    else:
        masks, iou = masks[:, :1], iou[:, :1]
    tokens = tokens[:, 1:] if multimask and model.use_multimask_token_for_obj_ptr else tokens[:, :1]
    return masks, iou, tokens, obj


def run_case(official, dtype_name, grid, tokens, seed=0):
    start = time.perf_counter()
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype_name)
    model = official[1].MaskDecoder(transformer_dim=256,
        transformer=official[0].TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048),
        use_high_res_features=True, iou_prediction_use_sigmoid=True,
        dynamic_multimask_via_stability=True, pred_obj_scores=True, pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True).to(dtype=dtype, device='cpu').eval()
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, official[2].LayerNorm2d)):
            module.weight.copy_(1 + .15 * torch.randn_like(module.weight))
            module.bias.copy_(.15 * torch.randn_like(module.bias))
    image = torch.randn(1, 256, grid, grid, dtype=dtype)
    pe = torch.randn_like(image)
    sparse = torch.randn(2, tokens - 6, 256, dtype=dtype)
    dense = torch.randn(1, 256, 1, 1, dtype=dtype).expand(1, 256, grid, grid)
    # Already-projected shared feature caches, exactly the decoder's public input.
    highres = [torch.randn(1, 32, 4 * grid, 4 * grid, dtype=dtype),
               torch.randn(1, 64, 2 * grid, 2 * grid, dtype=dtype)]
    args = dict(image_embeddings=image, image_pe=pe, sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense.expand(2, -1, -1, -1), repeat_image=True,
                high_res_features=highres)
    ref_trace = Trace2(model, official)
    reference = model.predict_masks(**args)
    ref_trace.close()
    cache = SharedCache(model, image, dense, pe)
    candidate_trace = Trace2(model, official)
    proposed, state = factored_predict2(model, cache, sparse, highres, candidate_trace)
    candidate_trace.close()
    assert set(ref_trace.data) == set(candidate_trace.data), (set(ref_trace.data) - set(candidate_trace.data), set(candidate_trace.data) - set(ref_trace.data))
    errors = {name: metrics(ref_trace.data[name], candidate_trace.data[name]) for name in ref_trace.data}
    labels = ('all_four_masks', 'all_four_sigmoid_iou', 'all_four_mask_tokens', 'object_score_logits')
    for label, ref, cand in zip(labels, reference, proposed):
        errors[label] = metrics(ref, cand)
    selections = {}
    # Primary configured dynamic fallback and both public modes. Small cases
    # additionally exercise ordinary single-mask slicing and fixed object ptr token.
    variants = [(True, True)]
    if grid == 8:
        variants += [(False, False)]
    for dynamic, multi_ptr in variants:
        model.dynamic_multimask_via_stability = dynamic
        model.use_multimask_token_for_obj_ptr = multi_ptr
        for multi in (False, True):
            native = model(**args, multimask_output=multi)
            cand = select_outputs(model, proposed, multi)
            ref_sel = select_outputs(model, reference, multi)
            key = f'dynamic_{dynamic}_multi_ptr_{multi_ptr}_multimask_{multi}'
            selections[key] = {label: metrics(ref, pred) for label, ref, pred in zip(labels, native, cand)}
            selections[key]['official_forward_matches_predict_selection_exactly'] = all(torch.equal(a, b) for a, b in zip(native, ref_sel))
    model.dynamic_multimask_via_stability = True
    ref_stability = model._get_stability_scores(reference[0][:, :1])
    cand_stability = model._get_stability_scores(proposed[0][:, :1])
    errors['single_mask_stability_scores'] = metrics(ref_stability, cand_stability)
    threshold = model.dynamic_multimask_stability_thresh
    threshold_disagreements = {}
    for t in (0.0, -.05, .05):
        threshold_disagreements[str(t)] = int(((reference[0] > t) != (proposed[0] > t)).sum())
    decisions = {
        'binary_threshold_disagreements': threshold_disagreements,
        'dynamic_stability_threshold': threshold,
        'reference_stability_scores': ref_stability.tolist(),
        'candidate_stability_scores': cand_stability.tolist(),
        'dynamic_fallback_disagreements': int(((ref_stability >= threshold) != (cand_stability >= threshold)).sum()),
        'reference_dynamic_fallback_count': int((ref_stability < threshold).sum()),
        'best_multimask_iou_index_disagreements': int((reference[1][:, 1:].argmax(-1) != proposed[1][:, 1:].argmax(-1)).sum()),
        'object_present_at_zero_disagreements': int(((reference[3] > 0) != (proposed[3] > 0)).sum()),
        'minimum_abs_mask_logit': reference[0].abs().min().item(),
        'note': 'Finite random sample diagnostics only. Decoder receives random prompt embeddings and projected high-res features; no PromptEncoder, ImagePredictor, encoder, memory conditioning or real image is run.',
    }
    tolerance = 1e-10 if dtype == torch.float64 else 5e-5
    failed = [name for name, m in errors.items() if m['max_abs'] > tolerance or not m['finite_candidate']]
    for key, selection in selections.items():
        if not selection['official_forward_matches_predict_selection_exactly']:
            failed.append(key + '.reference_selection')
        for label in labels:
            if selection[label]['max_abs'] > tolerance:
                failed.append(key + '.' + label)
    for i, err in enumerate(state.residuals):
        if err['max_abs'] > tolerance:
            failed.append(f'factor_reconstruction.{i}')
    assert state.widths == [8 * (tokens - 1) + 2, 2 * (8 * (tokens - 1) + 1) + 1]
    worst = max(errors, key=lambda key: errors[key]['max_abs'])
    return {
        'seed': seed, 'dtype': dtype_name, 'N': grid * grid, 'T': tokens, 'prompts': 2,
        'D': 256, 'H': 8, 'depth': 2, 'MLP_width': 2048, 'model_state_sha256': parameter_hash(model),
        'model_options': {'use_high_res_features': True, 'iou_prediction_use_sigmoid': True,
            'pred_obj_scores': True, 'pred_obj_scores_mlp': True, 'primary_dynamic_multimask_via_stability': True,
            'primary_use_multimask_token_for_obj_ptr': True},
        'factor_widths_after_LN': state.widths, 'factor_reconstruction_errors': state.residuals,
        'transformer_ln_eps': [block.norm4.eps for block in model.transformer.layers],
        'upscaler_ln_eps': model.output_upscaling[1].eps,
        'stage_count': len(errors), 'stage_errors': errors, 'selection_errors': selections,
        'decisions': decisions, 'absolute_tolerance': tolerance, 'passed': not failed,
        'failed_stages': failed, 'worst_stage': worst, 'worst_max_abs': errors[worst]['max_abs'],
        'execution_budget_seconds_only': time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    official = materialize_sam2()
    cases = [(dt, 8, tokens) for dt in ('float64', 'float32') for tokens in (8, 9)] + [('float32', 64, 8)]
    if args.quick:
        cases = cases[:1]
    name = 'quick_results.json' if args.quick else 'results.json'
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'official_pin': PIN,
        'torch': torch.__version__, 'python': platform.python_version(), 'device': 'cpu',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_sam_harness_sha256': hashlib.sha256((PARENT / 'validate_official.py').read_bytes()).hexdigest(),
        'scope': 'Separate SAM2 official random-weight decoder suite; native SDPA reference, high-res additions, object-score MLP, sigmoid IoU, four actual hypernetworks, dynamic stability selection. No pretrained/GPU/low-precision/runtime-speed claim. Inputs are synthetic random embeddings, not encoded prompts or images.',
        'attention_diagnostics_note': 'Attention logits/probabilities recorded by hooks are explicit recomputations from the actual projected q/k, not instrumentation inside native SDPA. All attention OUTPUTS are those returned by the actual official native-SDPA forward.',
        'cases': []}
    with torch.inference_mode():
        for dtype, grid, tokens in cases:
            result = run_case(official, dtype, grid, tokens)
            report['cases'].append(result)
            (HERE / name).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ('dtype', 'N', 'T', 'passed', 'worst_stage', 'worst_max_abs', 'factor_widths_after_LN', 'failed_stages')}), flush=True)
            if not result['passed']:
                raise AssertionError('Predeclared tolerance failed, inspect implementation without changing tolerance')
    report['all_passed'] = all(c['passed'] for c in report['cases'])
    report['total_cases'] = len(report['cases'])
    (HERE / name).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {HERE / name}', flush=True)


if __name__ == '__main__':
    main()
