#!/usr/bin/env python3
"""Small independent Eva-protocol fixtures; NOT installed timm/DINO proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_response_runtime import collect_reference_responses, decoder_arguments, finalize_decoder_logits
from tics.reference_halfspace_probe import reference_axis
from tics.reference_response_decoder import ReferenceResponseDecoder


def apply_rot_embed_cat(x, rope, half=False):
    # Fixture-only signed coordinate rotation, not pretrained DINO RoPE.
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class EvaAttention(nn.Module):
    def __init__(self, channels, heads, prefix, variant):
        super().__init__()
        self.num_heads = heads
        self.num_prefix_tokens = prefix
        self.rotate_half = True
        self.scale = (channels//heads)**-.5
        self.qkv_bias_separate = variant % 3 == 1
        self.q_bias = None
        if variant % 3 == 2:
            self.qkv = None
            self.q_proj = nn.Linear(channels, channels)
            self.k_proj = nn.Linear(channels, channels)
            self.v_proj = nn.Linear(channels, channels)
        else:
            self.qkv = nn.Linear(channels, 3*channels, bias=variant % 3 == 0)
            if variant % 3 == 1:
                self.q_bias = nn.Parameter(torch.randn(channels)*.1)
                self.k_bias = nn.Parameter(torch.randn(channels)*.1)
                self.v_bias = nn.Parameter(torch.randn(channels)*.1)
        dimension = channels//heads
        self.q_norm = nn.LayerNorm(dimension)
        self.k_norm = nn.LayerNorm(dimension)
        self.fused_attn = variant % 2 == 0
        self.norm = nn.LayerNorm(channels) if variant % 2 else nn.Identity()
        self.gate = nn.Linear(channels, channels) if variant % 2 else None
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(.2)
        self.test_intervention = None
        self.last_source_v = None

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        # This reference implements the operations independently, without
        # calling the adapter's eva_components/native_heads/project_heads.
        batch, tokens, channels = x.shape
        if self.qkv is None:
            pieces = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
        else:
            if self.q_bias is None:
                joined = self.qkv(x)
            else:
                bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                joined = self.qkv(x)+bias if self.qkv_bias_separate else F.linear(x, self.qkv.weight, bias)
            pieces = joined.chunk(3, dim=-1)
        q, k, v = [p.reshape(batch, tokens, self.num_heads, -1).transpose(1, 2) for p in pieces]
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            ng = self.num_prefix_tokens
            q = torch.cat((q[:, :, :ng], apply_rot_embed_cat(q[:, :, ng:], rope, half=True)), dim=2)
            k = torch.cat((k[:, :, :ng], apply_rot_embed_cat(k[:, :, ng:], rope, half=True)), dim=2)
        self.last_source_v = v.detach().clone()
        if self.test_intervention is not None:
            axis, branch = self.test_intervention
            start = self.num_prefix_tokens
            original = v[1, :, start:]
            alpha = ((original-axis.center[:, None])*axis.direction[:, None]).sum(-1)
            scalar = torch.maximum(alpha, torch.zeros_like(alpha)) if branch == 'plus' else torch.minimum(alpha, torch.zeros_like(alpha))
            v = v.clone()
            v[1, :, start:] = original-scalar[..., None]*axis.direction[:, None]
        if self.fused_attn:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                    dropout_p=0., is_causal=is_causal)
        else:
            logits = (q*self.scale)@k.transpose(-2, -1)
            if is_causal:
                logits = logits.masked_fill(~torch.ones(tokens, tokens, dtype=torch.bool).tril(), float('-inf'))
            if attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else logits+attn_mask
            output = logits.softmax(-1)@v
        output = self.norm(output.transpose(1, 2).reshape(batch, tokens, channels))
        if self.gate is not None:
            output = output*self.gate(x).sigmoid()
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, channels, prefix, variant):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = EvaAttention(channels, 4, prefix, variant)
        self.scale = nn.Parameter(torch.rand(channels)*.4+.1)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 2*channels), nn.GELU(), nn.Linear(2*channels, channels))

    def forward(self, x, rope=None):
        result = self.attn(self.norm1(x), rope=rope)
        result.mul_(self.scale)  # Test observer copies before an in-place LayerScale.
        x = x+result
        return x+self.mlp(self.norm2(x))


class TinyEncoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.prefix_count = 2
        self.embed = nn.Linear(3, 16)
        self.prefix = nn.Parameter(torch.randn(1, self.prefix_count, 16))
        self.blocks = nn.ModuleList([Block(16, self.prefix_count, variant+i) for i in range(3)])
        self.norm = nn.LayerNorm(16)
        self.rotary = variant % 2 == 0

    def forward(self, images):
        batch, _, height, width = images.shape
        x = self.embed(images.flatten(2).transpose(1, 2))
        x = torch.cat((self.prefix.expand(batch, -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x, rope=True if self.rotary else None)
        x = self.norm(x)[:, self.prefix_count:]
        return x.transpose(1, 2).reshape(batch, 16, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing CPU receipt')
    torch.set_num_threads(1)
    records = []
    source_paths = [Path(__file__), Path(__file__).resolve().parents[1]/'tics/reference_response_runtime.py']
    result = dict(state='RUNNING_CPU_FIXTURES', records=records,
                  actual_installed_timm=False, pretrained_model_loaded=False,
                  real_task_gain_measured=False)
    try:
        with torch.no_grad():
            for case in range(10):
                torch.manual_seed(2140+case)
                model = TinyEncoder(case).double().eval().requires_grad_(False)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                images = torch.randn(2, 3, 4, 4, dtype=torch.float64)
                coverage = torch.zeros(4, 4, dtype=torch.float64)
                coverage[:2] = 1
                selected = model.blocks[1].attn
                original_forward = selected.forward.__func__
                baseline = model(images).clone()
                packed = collect_reference_responses(model, model, images, coverage,
                    grid=(4, 4), layer=1, allow_cpu_fixture=True)
                features = packed['features']
                assert torch.equal(features['query_final'][0], baseline[1])
                assert torch.equal(features['support_final'][0], baseline[0])
                assert features['query_middle'].shape == (1, 64, 4, 4)
                assert packed['audit']['native_replica_exact']
                assert selected.forward.__func__ is original_forward
                # Complete collection -> the ACTUAL full-grid head signature.
                decoder = ReferenceResponseDecoder(16, 64).double().eval()
                dynamic_logits = decoder(**decoder_arguments(packed))
                static_logits = decoder(**decoder_arguments(packed, static=True))
                assert dynamic_logits.shape == static_logits.shape == (1, 1, 4, 4)
                assert torch.isfinite(dynamic_logits).all() and torch.isfinite(static_logits).all()
                calls = []
                def source_finalizer_fixture(mask, rgb):
                    calls.append((mask.detach().clone(), rgb))
                    return mask.clone()
                refined = finalize_decoder_logits(dynamic_logits, images[1:2], source_finalizer_fixture)
                assert len(calls) == 1 and calls[0][1] is not None
                expected_binary = F.interpolate(dynamic_logits, (4, 4), mode='bilinear', align_corners=False)[0, 0] > 0
                assert torch.equal(refined, expected_binary)
                axis = reference_axis(selected.last_source_v[0, :, 2:],
                                      coverage.flatten() >= .9, coverage.flatten() <= .1)
                manual_errors = []
                for branch in ('plus', 'minus'):
                    selected.test_intervention = (axis, branch)
                    manual = model(images).clone()
                    selected.test_intervention = None
                    expected = baseline[1:2]-manual[1:2]
                    torch.testing.assert_close(features['query_response_'+branch], expected, rtol=0, atol=0)
                    assert torch.equal(manual[0], baseline[0])
                    manual_errors.append(float((features['query_response_'+branch]-expected).abs().max()))
                swapped = collect_reference_responses(model, model, images, 1-coverage,
                    grid=(4, 4), layer=1, allow_cpu_fixture=True)['features']
                assert torch.equal(features['query_response_plus'], swapped['query_response_minus'])
                assert torch.equal(features['query_response_minus'], swapped['query_response_plus'])
                for k, value in model.state_dict().items():
                    assert torch.equal(value, before[k])
                # A callback failure after a real selected-layer call must
                # remove only our forward override and our block observer.
                old_hook_count = len(model.blocks[1]._forward_hooks)
                counter = [0]
                def failing_encode(batch):
                    counter[0] += 1
                    value = model(batch)
                    if counter[0] == 2:
                        raise RuntimeError('deliberate CPU callback failure')
                    return value
                try:
                    collect_reference_responses(model, failing_encode, images, coverage,
                        grid=(4, 4), layer=1, allow_cpu_fixture=True)
                    raise AssertionError('Callback error swallowed')
                except RuntimeError as exc:
                    assert 'deliberate CPU callback failure' in str(exc)
                assert selected.forward.__func__ is original_forward
                assert len(model.blocks[1]._forward_hooks) == old_hook_count
                assert 'forward' not in selected.__dict__
                records.append(dict(case=case, exact_native_and_reference=True,
                    independent_explicit_probe_max_error=max(manual_errors),
                    label_swap_exact=True, parameters_and_buffers_unchanged=True,
                    original_forward_restored_after_error=True,
                    actual_dense_head_bridge_passed=True, finalizer_called_once=True,
                    CRF_fixture_only=True))

            model = TinyEncoder(1).double().eval().requires_grad_(False)
            # Same number of identical FG/BG values ensures exact zero contrast,
            # rather than asserting an approximate cancellation is a no-op.
            images = torch.zeros(2, 3, 4, 4, dtype=torch.float64)
            baseline = model(images)
            zero = collect_reference_responses(model, model, images, coverage,
                grid=(4, 4), layer=1, allow_cpu_fixture=True)
            assert zero['audit']['enabled_heads'] == 0
            assert not zero['features']['query_response_plus'].any()
            assert not zero['features']['query_response_minus'].any()
            assert torch.equal(zero['features']['query_final'], baseline[1:2])
            decoder = ReferenceResponseDecoder(16, 64).double().eval()
            assert torch.equal(decoder(**decoder_arguments(zero)),
                               decoder(**decoder_arguments(zero, static=True)))
            result['zero_contrast_whole_encoder_noop_exact'] = True
            try:
                collect_reference_responses(model, lambda batch: baseline, images, coverage,
                    grid=(4, 4), layer=1, allow_cpu_fixture=True)
                raise AssertionError('Cached encoder callback accepted')
            except RuntimeError as exc:
                assert 'Absent or repeated selected block' in str(exc)
            try:
                collect_reference_responses(model, model, images, coverage, grid=(4, 4), layer=1)
                raise AssertionError('Fixture enabled without explicit CPU permission')
            except RuntimeError as exc:
                assert 'Unsupported source' in str(exc)
            result.update(state='CPU_RESPONSE_RUNTIME_FIXTURES_PASSED',
                          cached_callback_rejected=True, default_fixture_rejected=True)
    except BaseException as exc:
        result.update(state='ERROR_CPU_RESPONSE_RUNTIME_FIXTURE', error=repr(exc))
        raise
    finally:
        result.update(CUDA_initialized=torch.cuda.is_initialized(),
                      source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(state=result['state'], cases=len(records),
                             actual_installed_timm=False, real_task_gain_measured=False,
                             CUDA_initialized=result['CUDA_initialized'])))


if __name__ == '__main__':
    main()
