#!/usr/bin/env python3
"""Actual already-installed timm modules, CPU only, no pretrained model/download.

Local absence of timm is PARTIAL/exit2, never a passing source check. Run this
on the existing CPU preparation environment before any paid GPU execution.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-timm-version', default='1.0.30')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Keep old CPU source receipt')
    report = dict(state='RUNNING_CPU_INSTALLED_MODULES', cases=[], CUDA_initialized=False,
                  pretrained_model_loaded=False, real_task_gain_measured=False,
                  downloads_allowed=False, expected_timm_version=args.expected_timm_version)
    exit_code = 0
    try:
        if importlib.util.find_spec('timm') is None:
            report.update(state='PARTIAL_INSTALLED_TIMM_UNAVAILABLE', actual_installed_timm=False)
            exit_code = 2
        else:
            import torch
            from torch import nn
            import timm
            from timm.models.eva import EvaBlock
            from timm.layers import LayerNorm
            from functools import partial
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            # Only factories/constants: this DOES NOT execute old scientific
            # readout queues or old attention-modification experiments.
            from native_attention_readout_installed_cpu import make_attention, make_rope, DIM, PREFIX, GRID, HEADS
            from tics.reference_response_runtime import (
                attention_source_hash, collect_reference_responses, decoder_arguments)
            from tics.reference_response_decoder import ReferenceResponseDecoder
            if timm.__version__ != args.expected_timm_version:
                raise RuntimeError('Installed timm differs from the declared original source version')
            if torch.cuda.is_initialized():
                raise RuntimeError('CUDA was initialized before CPU source acceptance')
            torch.set_num_threads(1)
            norm = partial(LayerNorm, eps=1e-5)
            _, rope = make_rope()

            class ActualBlockContainer(nn.Module):
                def __init__(self, variant):
                    super().__init__()
                    self.embed = nn.Linear(3, DIM)
                    self.prefix = nn.Parameter(torch.randn(1, PREFIX, DIM))
                    self.blocks = nn.ModuleList([EvaBlock(dim=DIM, num_heads=HEADS,
                        qkv_bias=False, qkv_fused=True, mlp_ratio=1.5,
                        num_prefix_tokens=PREFIX, attn_type='eva', rotate_half=True,
                        norm_layer=norm, init_values=.2, attn_drop=0., proj_drop=0.,
                        drop_path=0., device='cpu') for _ in range(3)])
                    # Exercise actual source qkv/bias/norm/gate configurations.
                    self.blocks[1].attn = make_attention(variant)
                    self.norm = norm(DIM)
                    self.double().eval().requires_grad_(False)

                def forward(self, images):
                    patch = self.embed(images.flatten(2).transpose(1, 2))
                    value = torch.cat((self.prefix.expand(images.shape[0], -1, -1), patch), dim=1)
                    for block in self.blocks:
                        value = block(value, rope=rope)
                    value = self.norm(value)[:, PREFIX:]
                    return value.transpose(1, 2).reshape(images.shape[0], DIM, *GRID)

            report.update(actual_installed_timm=True, timm_version=timm.__version__,
                          torch_version=torch.__version__, source_hashes={},
                          scope='Real installed Eva modules in a small CPU container, NOT pretrained DINO/Flash/task evidence')
            with torch.no_grad():
                for case in range(10):
                    torch.manual_seed(2160+case)
                    model = ActualBlockContainer(case)
                    images = torch.randn(2, 3, *GRID, dtype=torch.float64)
                    coverage = torch.zeros(GRID, dtype=torch.float64)
                    coverage[:GRID[0]//2] = 1
                    original = model(images).clone()
                    parameters = {name: value.clone() for name, value in model.state_dict().items()}
                    attention = model.blocks[1].attn
                    source_sha = attention_source_hash(attention)
                    native_forward = attention.forward.__func__
                    packet = collect_reference_responses(model, model, images, coverage,
                        grid=GRID, layer=1, expected_attention_sha=source_sha)
                    assert packet['audit']['actual_installed_timm']
                    assert torch.equal(packet['features']['support_final'][0], original[0])
                    assert torch.equal(packet['features']['query_final'][0], original[1])
                    swapped = collect_reference_responses(model, model, images, 1-coverage,
                        grid=GRID, layer=1, expected_attention_sha=source_sha)
                    assert torch.equal(packet['features']['query_response_plus'],
                                       swapped['features']['query_response_minus'])
                    assert attention.forward.__func__ is native_forward
                    assert all(torch.equal(value, parameters[name]) for name, value in model.state_dict().items())
                    decoder = ReferenceResponseDecoder(DIM, 4*DIM).double().eval()
                    logits = decoder(**decoder_arguments(packet))
                    static = decoder(**decoder_arguments(packet, static=True))
                    assert logits.shape == static.shape == (1, 1, *GRID)
                    assert torch.isfinite(logits).all() and torch.isfinite(static).all()
                    try:
                        collect_reference_responses(model, model, images, coverage,
                            grid=GRID, layer=1, expected_attention_sha='0'*64)
                        raise AssertionError('Changed original source hash accepted')
                    except RuntimeError as exc:
                        assert 'source differs' in str(exc)
                    report['cases'].append(dict(case=case, source_sha=source_sha,
                        native_and_reference_output_exact=True, label_swap_exact=True,
                        whole_dense_head_and_prefix_control_executed=True,
                        parameters_preserved=True, source_hash_drift_rejected=True))
                report.update(state='CPU_INSTALLED_RESPONSE_MODULES_PASSED',
                              CUDA_initialized=torch.cuda.is_initialized())
                assert not report['CUDA_initialized']
                report['source_hashes']['EvaAttention.forward'] = source_sha
                report['source_hashes']['EvaBlock.forward'] = hashlib.sha256(
                    inspect.getsource(EvaBlock.forward).encode()).hexdigest()
    except BaseException as exc:
        report.update(state='ERROR_CPU_INSTALLED_RESPONSE_MODULES', error=repr(exc))
        exit_code = 1
    finally:
        paths = [Path(__file__), Path(__file__).resolve().parents[1]/'tics/reference_response_runtime.py',
                 Path(__file__).resolve().parents[1]/'tics/reference_response_decoder.py',
                 Path(__file__).resolve().parents[1]/'tics/reference_halfspace_probe.py',
                 Path(__file__).with_name('native_attention_readout.py'),
                 Path(__file__).with_name('native_attention_readout_installed_cpu.py')]
        report['implementation_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(state=report['state'], cases=len(report['cases']),
                             pretrained_model_loaded=False, real_task_gain_measured=False,
                             CUDA_initialized=report['CUDA_initialized'])))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
