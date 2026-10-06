"""Native-unary conditioned, class-explicit global correspondences.

Each query has two distributions over reference tokens, one per known source
role. Joint correspondence evidence updates a fixed host prior; outputs are not
mask combinations. No model load, class id, or query label enters this module.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from . import structure as base

CONFIG = {
    'name': 'structure_conditioned_v2',
    'temperature': 0.05,
    'neighbors': 12,
    'iterations': 4,
    'host_prior_strength': 1.0,
    'message_strength': 1.0,
    'extra_encoder_forwards': 0,
    'layers': ['cached_final'],
    'selection': 'unit_prior_and_compatibility_coefficients_frozen_before_v2_GT',
    'reference_graph': 'same_as_frozen_structure_v1',
}


def _host(score, device):
    s = torch.as_tensor(score, dtype=torch.float32, device=device)
    if s.ndim != 2 or not torch.isfinite(s).all():
        raise ValueError('Finite two-dimensional native raw score required')
    return (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)


def _prepare(q, r, cov, score, device):
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    r = torch.as_tensor(r, dtype=torch.float32, device=device)
    host = _host(score, device)
    shape, p = tuple(host.shape), host.reshape(-1)
    c = torch.as_tensor(cov, dtype=torch.float32, device=device).reshape(-1)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError('Matching two-dimensional token feature matrices required')
    if len(q) != p.numel() or len(r) != c.numel():
        raise ValueError('Token counts must match native grid/reference coverage')
    if not all(torch.isfinite(x).all().item() for x in (q, r, c)):
        raise ValueError('Nonfinite input')
    if c.min() < 0 or c.max() > 1 or c.sum() <= 0 or (1 - c).sum() <= 0:
        raise ValueError('Coverage in [0,1] with both source roles required')
    q = torch.nn.functional.normalize(q, dim=1)
    r = torch.nn.functional.normalize(r, dim=1)
    affinity = q @ r.T / CONFIG['temperature']
    # Soft coverage retains boundary reference tokens in both known roles.
    conditional = [
        (affinity + torch.where(weight > 0, weight.log(), -torch.inf)[None]).softmax(1)
        for weight in (c, 1 - c)
    ]
    source_priors = [c / c.sum(), (1 - c) / (1 - c).sum()]
    return q, r, p, conditional, source_priors, shape, affinity


def _message(graph_q, graph_r, joint):
    return torch.sparse.mm(graph_r, torch.sparse.mm(graph_q, joint).T.contiguous()).T.contiguous()


def _run(q, r, cov, score, device, mode, strength=1.0):
    started = time.perf_counter()
    if mode == 'independent' or strength == 0:
        field = _host(score, device).cpu().numpy().astype(np.float32)
        return field, {'config': dict(CONFIG), 'mode': mode, 'zero_interaction_equals': 'native_pre_unary',
                       'query_gt_used': False, 'extra_encoder_forwards': 0,
                       'elapsed_seconds': time.perf_counter() - started}
    q, r, p0, conditionals, source_priors, shape, _ = _prepare(q, r, cov, score, device)
    aq, ar = base._graph(q), base._graph(r)
    nulls = [torch.sparse.mm(ar, prior[:, None]).T.clamp_min(1e-20) for prior in source_priors]
    nulls = [null / null.sum() for null in nulls]
    p = p0
    initial = [p0[:, None] * conditionals[0], (1 - p0[:, None]) * conditionals[1]]
    joint = initial
    eps = torch.finfo(torch.float32).eps
    # Finite extrema permit a subsequent compatibility update; independent and
    # zero-strength controls preserve the exact unclipped native-pre field.
    log_prior_odds = torch.logit(p0.clamp(eps, 1 - eps))
    history = [{'iteration': 0, 'foreground_mass': float(p.mean().item())}]
    for iteration in range(CONFIG['iterations']):
        messages = [_message(aq, ar, j) for j in joint]
        if mode == 'diffusion':
            # Same correspondence operators and iteration count, without
            # class-conditional likelihood feedback or a posterior role update.
            normalizer = (messages[0].sum(1) + messages[1].sum(1)).clamp_min(1e-20)
            joint = [(j0 + msg / normalizer[:, None]) / 2
                     for j0, msg in zip(initial, messages)]
            p = joint[0].sum(1)
            log_bayes = torch.zeros_like(p)
        else:
            updated_conditionals, evidences = [], []
            for conditional, msg, null in zip(conditionals, messages, nulls):
                # Condition on the role BEFORE computing its source-token
                # evidence. Query-neighborhood role mass is not itself a vote.
                role_mass = msg.sum(1, keepdim=True)
                normalized = msg / role_mass.clamp_min(1e-20)
                ratio = torch.where(role_mass > 1e-20, normalized / null, torch.ones_like(msg))
                unnormalized = conditional * ratio
                evidence = unnormalized.sum(1).clamp_min(1e-20)
                updated_conditionals.append(unnormalized / evidence[:, None])
                evidences.append(evidence)
            log_bayes = evidences[0].log() - evidences[1].log()
            p = (CONFIG['host_prior_strength'] * log_prior_odds + strength * log_bayes).sigmoid()
            joint = [p[:, None] * updated_conditionals[0],
                     (1 - p[:, None]) * updated_conditionals[1]]
        history.append({'iteration': iteration + 1,
                        'foreground_mass': float(p.mean().item()),
                        'mean_log_bayes': float(log_bayes.mean().item()),
                        'mean_abs_log_bayes': float(log_bayes.abs().mean().item())})
    return p.reshape(shape).cpu().numpy().astype(np.float32), {
        'config': dict(CONFIG), 'mode': mode, 'history': history,
        'query_gt_used': False, 'extra_encoder_forwards': 0,
        'correspondence_state_mib': sum(x.numel() * x.element_size() for x in joint) / 2**20,
        'elapsed_seconds': time.perf_counter() - started,
    }


@torch.inference_mode()
def predict(q, r, cov, score, *, device='cpu', extras=None):
    return _run(q, r, cov, score, device, 'conditional_compatibility', CONFIG['message_strength'])


@torch.inference_mode()
def control(q, r, cov, score, *, device='cpu', extras=None):
    return _run(q, r, cov, score, device, 'independent')


@torch.inference_mode()
def diffusion_control(q, r, cov, score, *, device='cpu', extras=None):
    return _run(q, r, cov, score, device, 'diffusion')


@torch.inference_mode()
def zero_interaction_control(q, r, cov, score, *, device='cpu', extras=None):
    return _run(q, r, cov, score, device, 'conditional_compatibility', 0.0)


@torch.inference_mode()
def kernel_control(q, r, cov, score, *, device='cpu', extras=None):
    """Host odds times one class-balanced kernel likelihood ratio, both unit weight."""
    started = time.perf_counter()
    q, r, p, _, priors, shape, affinity = _prepare(q, r, cov, score, device)
    log_evidence = [torch.logsumexp(affinity + torch.where(v > 0, v.log(), -torch.inf)[None], dim=1)
                    for v in priors]
    eps = torch.finfo(torch.float32).eps
    log_ratio = log_evidence[0] - log_evidence[1]
    output = (torch.logit(p.clamp(eps, 1 - eps)) + log_ratio).sigmoid()
    return output.reshape(shape).cpu().numpy().astype(np.float32), {
        'config': dict(CONFIG), 'mode': 'native_prior_times_single_kernel_ratio',
        'kernel_strength': 1.0, 'host_prior_strength': 1.0,
        'query_gt_used': False, 'extra_encoder_forwards': 0,
        'mean_abs_log_ratio': float(log_ratio.abs().mean().item()),
        'elapsed_seconds': time.perf_counter() - started,
    }


additional_controls = {'diffusion': diffusion_control, 'kernel': kernel_control}
