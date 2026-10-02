"""Support-conditioned low-dimensional prompts in EXISTING global tokens.

Candidate: fixed-seed orthonormal prefix basis; no offline-trained dictionary,
extra tokens, pretrained weights, backbone update, or query-label interface.
This changes the encoder trajectory and is distinct from side attention readout.
Eva-compatible injection is a reversible first-block forward pre-hook.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ReferencePrefixPrompt:
    def __init__(self, backbone: nn.Module, prefix_count: int, channels: int,
                 rank: int = 32, relative_radius: float = .1, seed: int = 4101,
                 basis: Tensor | None = None):
        if not hasattr(backbone, "blocks") or len(backbone.blocks) == 0:
            raise ValueError("Backbone must expose its actual first transformer block")
        if prefix_count < 1 or not 1 <= rank <= prefix_count * channels or relative_radius < 0:
            raise ValueError("Invalid prefix/rank/radius")
        self.backbone, self.prefix_count, self.channels = backbone, prefix_count, channels
        self.rank, self.relative_radius, self.seed = rank, relative_radius, seed
        parameter = next(backbone.parameters())
        if basis is None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            matrix = torch.randn(prefix_count * channels, rank, generator=generator, dtype=torch.float64)
            orthogonal = torch.linalg.qr(matrix, mode="reduced").Q
            basis = orthogonal.T.reshape(rank, prefix_count, channels).float()
        if basis.shape != (rank, prefix_count, channels):
            raise ValueError("Basis must be [rank,prefix_count,channels]")
        flat = basis.double().reshape(rank, -1)
        if not torch.allclose(flat @ flat.T, torch.eye(rank, dtype=flat.dtype, device=flat.device),
                              atol=1e-5, rtol=1e-5):
            raise ValueError("Basis directions must be orthonormal")
        self.basis = basis.detach().to(device=parameter.device, dtype=torch.float32).clone()
        self.coefficients = nn.Parameter(torch.zeros(rank, device=parameter.device), requires_grad=False)
        self.native_prefix_norm = None
        self._handle = None
        self.last_gradient_norm = 0.0

    def _override(self, module, args, kwargs, mode):
        positional = bool(args)
        x = args[0] if positional else kwargs.get("x")
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("Eva first block must receive x[B,N,C]")
        if x.shape[1] <= self.prefix_count:
            raise ValueError("No patch tokens after existing prefixes")
        if self.native_prefix_norm is None:
            self.native_prefix_norm = x[:, :self.prefix_count].detach().float().flatten(1).norm(dim=1).mean()
        scale = self.relative_radius * self.native_prefix_norm / math.sqrt(self.rank)
        values = self.coefficients.tanh()
        # Native matched-budget control retains a zero-gradient graph.
        if mode == "native":
            values = values * 0
        delta = torch.einsum("r,rpc->pc", values, self.basis) * scale
        prefix = x[:, :self.prefix_count] + delta.to(x)[None]
        updated = torch.cat((prefix, x[:, self.prefix_count:]), dim=1)
        if positional:
            return (updated, *args[1:]), kwargs
        replaced = dict(kwargs, x=updated)
        return args, replaced

    @contextmanager
    def activate(self, *, mode: str = "adaptive", train_prompt: bool = False):
        """Freeze/eval backbone temporarily; restore flags and remove hook always.

        No backbone parameter value is assigned. mode='fixed' uses frozen
        predetermined coefficients; mode='native' injects exactly zero.
        feature_fn must call a gradient-enabled actual backbone path, not an
        inference_mode/no_grad cached-feature wrapper, during support fitting.
        """
        if mode not in ("adaptive", "native", "fixed") or self._handle is not None:
            raise ValueError("Invalid mode or concurrent prompt context")
        flags = [(parameter, parameter.requires_grad) for parameter in self.backbone.parameters()]
        training = [(module, module.training) for module in self.backbone.modules()]
        coefficient_flag = self.coefficients.requires_grad
        try:
            self.backbone.eval()
            for parameter, _ in flags:
                parameter.requires_grad_(False)
            self.coefficients.requires_grad_(train_prompt)
            self._handle = self.backbone.blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: self._override(module, args, kwargs, mode), with_kwargs=True)
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            self.coefficients.requires_grad_(coefficient_flag)
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for module, flag in training:
                module.training = flag

    def perturbation_norm(self) -> float:
        if self.native_prefix_norm is None:
            return 0.0
        delta = torch.einsum("r,rpc->pc", self.coefficients.detach().tanh(), self.basis)
        return float(delta.norm() * self.relative_radius * self.native_prefix_norm / math.sqrt(self.rank))


def _valid_region_folds(labels: Tensor, regions: Tensor, min_anchors: int = 2):
    legal = (labels == 0) | (labels == 1)
    identifiers = torch.unique(regions[legal], sorted=True).tolist()
    def both(index, minimum):
        return all(int(((labels == value) & index).sum()) >= minimum for value in (0, 1))
    for validation_id in identifiers:
        validation = legal & (regions == validation_id)
        train_pool = legal & ~validation
        if not both(validation, 1) or not both(train_pool, min_anchors):
            continue
        folds = []
        for train_id in identifiers:
            if train_id == validation_id:
                continue
            held = train_pool & (regions == train_id)
            anchors = train_pool & ~held
            if both(held, 1) and both(anchors, min_anchors):
                folds.append((held, anchors))
        if len(folds) >= 2:
            return validation_id, validation, train_pool, folds
    return None


def _discriminative_loss(features: Tensor, labels: Tensor, targets: Tensor,
                         anchors: Tensor, temperature: float):
    features = F.normalize(features.float(), dim=1)
    positive = F.normalize(features[anchors & (labels == 1)].mean(0), dim=0)
    negative = F.normalize(features[anchors & (labels == 0)].mean(0), dim=0)
    logits = (features[targets] @ positive - features[targets] @ negative) / temperature
    truth = labels[targets].float()
    point_loss = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    return .5 * (point_loss[truth == 1].mean() + point_loss[truth == 0].mean())


def fit_reference_prompt(prompt: ReferencePrefixPrompt, feature_fn, support, support_labels: Tensor,
                         region_ids: Tensor, *, steps: int = 20, learning_rate: float = .05,
                         proximal: float = .01, temperature: float = .1,
                         mode: str = "adaptive", min_improvement: float = 1e-6) -> dict:
    """Only supplied support labels enter optimization/validation; no query input.

    feature_fn(support) -> [P,C] must remain differentiable w.r.t. the prompt.
    Region IDs are fixed spatial groups, selected without quality/GT ranking.
    Validation labels never become train anchors or training targets.
    Native/fixed controls use the same feature calls/backward budget, no update.
    """
    if steps < 0 or proximal < 0 or temperature <= 0:
        raise ValueError("Invalid fitting configuration")
    labels = support_labels.flatten().to(prompt.coefficients.device)
    regions = region_ids.flatten().to(labels.device)
    if labels.shape != regions.shape:
        raise ValueError("Label/region grid mismatch")
    split = _valid_region_folds(labels, regions)
    with torch.no_grad():
        prompt.coefficients.zero_()
        if mode == "fixed":
            prompt.coefficients.copy_(torch.linspace(-.25, .25, prompt.rank, device=labels.device))
    if split is None:
        return dict(state="NO_LEGAL_REGION_SPLIT", mode=mode, steps=0, coefficients=prompt.coefficients.tolist())
    validation_id, validation, train_pool, folds = split
    calls = 0
    def features():
        nonlocal calls
        output = feature_fn(support)
        calls += 1
        if output.shape != (len(labels), prompt.channels):
            raise ValueError("feature_fn must return [P,C] aligned with support labels")
        return output
    with prompt.activate(mode=mode, train_prompt=True), torch.enable_grad():
        optimizer = torch.optim.Adam([prompt.coefficients], lr=learning_rate)
        with torch.no_grad():
            initial = float(_discriminative_loss(features(), labels, validation, train_pool, temperature))
        best, best_step = initial, 0
        best_coefficients = prompt.coefficients.detach().clone()
        history = []
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            embedding = features()
            if not embedding.requires_grad:
                raise RuntimeError("Gradient-disabled feature path; do not use cached/no_grad encoder wrappers")
            training_loss = torch.stack([
                _discriminative_loss(embedding, labels, targets, anchors, temperature)
                for targets, anchors in folds]).mean()
            loss = training_loss + proximal * prompt.coefficients.square().mean()
            if not loss.requires_grad:
                raise RuntimeError("Gradient-disabled feature path; do not use cached/no_grad encoder wrappers")
            loss.backward()
            if prompt.coefficients.grad is None or not torch.isfinite(prompt.coefficients.grad).all():
                raise RuntimeError("Prompt gradient missing/nonfinite")
            prompt.last_gradient_norm = float(prompt.coefficients.grad.norm())
            if mode == "adaptive":
                optimizer.step()
            with torch.no_grad():
                current = float(_discriminative_loss(features(), labels, validation, train_pool, temperature))
            if current < best - min_improvement:
                best, best_step = current, step
                best_coefficients = prompt.coefficients.detach().clone()
            history.append(dict(step=step, training=float(training_loss.detach()), validation=current))
        with torch.no_grad():
            prompt.coefficients.copy_(best_coefficients)
    prompt.coefficients.requires_grad_(False)
    return dict(state="FROZEN_FOR_QUERY", mode=mode, steps=steps, feature_calls=calls,
                selected_step=best_step, validation_region=validation_id,
                initial_validation=initial, selected_validation=best,
                train_regions=len(folds), validation_excluded_from_train=True,
                gradient_norm=prompt.last_gradient_norm,
                prefix_delta_norm=prompt.perturbation_norm(),
                native_prefix_norm=float(prompt.native_prefix_norm),
                coefficients=prompt.coefficients.detach().cpu().tolist(), history=history,
                scope="Support-only proxy fitting; no cross-image/DINO quality claim")
