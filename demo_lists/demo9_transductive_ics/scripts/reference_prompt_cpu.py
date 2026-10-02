#!/usr/bin/env python3
"""CPU-only real autograd/SDPA fixtures for the prefix-prompt candidate.

Not a DINO checkpoint, installed-Eva reproduction, or segmentation experiment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1] / "tics/reference_prompt.py"
spec = importlib.util.spec_from_file_location("reference_prompt", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ReferencePrefixPrompt, fit_reference_prompt = module.ReferencePrefixPrompt, module.fit_reference_prompt


class FakeEvaBlock(nn.Module):
    """Eva-shaped pre-norm attention/residual/MLP, real CPU SDPA backward."""
    def __init__(self, dim=24, heads=3):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.gamma_1 = nn.Parameter(torch.full((dim,), .4))
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
    def forward(self, x, rope=None, attn_mask=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, channels // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        update = attention.transpose(1, 2).reshape(batch, tokens, channels)
        x = x + self.gamma_1 * self.proj(update)
        return x + self.mlp(self.norm2(x))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(1, 3, 24) * .3)
        self.blocks = nn.ModuleList([FakeEvaBlock(), FakeEvaBlock()])
        self.norm = nn.LayerNorm(24)
    def forward(self, patches):
        x = torch.cat((self.prefix.expand(len(patches), -1, -1), patches), dim=1)
        for block in self.blocks:
            x = block(x, rope=None)
        return self.norm(x)[:, 3:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = []
    for seed in range(4):
        torch.manual_seed(seed)
        model = FakeBackbone().train()
        snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
        flags = [parameter.requires_grad for parameter in model.parameters()]
        training_flags = [submodule.training for submodule in model.modules()]
        support = torch.randn(1, 16, 24)
        labels = torch.arange(16) % 2
        support[0, :, 0] += (labels * 2 - 1) * .5
        regions = torch.arange(16) // 4
        query = torch.randn(1, 16, 24)  # No query labels exist.
        model.eval()
        baseline = model(query).detach().clone()
        model.train()
        reports = {}
        for mode in ("adaptive", "native", "fixed"):
            prompt = ReferencePrefixPrompt(model, 3, 24, rank=4, relative_radius=.1)
            result = fit_reference_prompt(prompt, lambda value: model(value)[0], support,
                                          labels, regions, steps=8, mode=mode)
            assert result["state"] == "FROZEN_FOR_QUERY"
            assert result["feature_calls"] == 17 and result["validation_excluded_from_train"]
            assert result["selected_validation"] <= result["initial_validation"] + 1e-6
            assert result["prefix_delta_norm"] <= .1 * result["native_prefix_norm"] + 1e-6
            assert all(parameter.grad is None for parameter in model.parameters())
            assert not prompt.coefficients.requires_grad
            with prompt.activate(mode=mode):
                transferred = model(query).detach()
                assert not transferred.requires_grad
            if mode == "adaptive":
                assert result["gradient_norm"] > 0
            if mode == "native":
                assert torch.equal(transferred, baseline) and result["gradient_norm"] == 0
            assert prompt._handle is None and not model.blocks[0]._forward_pre_hooks
            assert [parameter.requires_grad for parameter in model.parameters()] == flags
            assert [submodule.training for submodule in model.modules()] == training_flags
            assert all(torch.equal(model.state_dict()[name], value) for name, value in snapshot.items())
            reports[mode] = dict(result, query_output_changed=not torch.equal(transferred, baseline))
        assert len({reports[mode]["feature_calls"] for mode in reports}) == 1
        # Explicit exception rollback and keyword-x handling.
        prompt = ReferencePrefixPrompt(model, 3, 24, rank=4)
        try:
            with prompt.activate(train_prompt=True):
                raise RuntimeError("fixture failure")
        except RuntimeError:
            pass
        assert prompt._handle is None and not model.blocks[0]._forward_pre_hooks
        with prompt.activate(mode="native"):
            first = torch.cat((model.prefix, support), 1)
            assert model.blocks[0](x=first, rope=None).shape == first.shape
        # A detached/cached extractor is rejected rather than falsely 'optimized'.
        try:
            fit_reference_prompt(prompt, lambda value: model(value)[0].detach(), support,
                                 labels, regions, steps=1)
            raise AssertionError("Detached feature adapter accepted")
        except RuntimeError as error:
            assert "Gradient-disabled" in str(error)
        assert prompt._handle is None and not model.blocks[0]._forward_pre_hooks
        assert all(torch.equal(model.state_dict()[name], value) for name, value in snapshot.items())
        cases.append(dict(seed=seed, reports=reports))
    result = dict(state="CPU_FIXTURES_PASSED", fixtures=len(cases), arms_per_fixture=3,
                  support_calls_per_arm=17, prefix_bound=.1,
                  backbone_parameters_unchanged=True, gradient_only_prompt=True,
                  legal_support_validation_only=True, query_labels_used=False,
                  hook_cleanup_on_success_and_exception=True,
                  detached_feature_adapter_rejected=True, cases=cases,
                  scope="Tiny CPU fake Eva-shaped blocks; no DINO/GPU/real-data quality claim")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
