"""Reference-constrained latent appearance slots with noisy native observations.

This is a fixed candidate, not an established method. No query labels enter inference.
Slot foreground identity and image membership are jointly inferred; there are no
foreground-area priors, trimaps, candidate-mask voting, or fitted covariance matrices.
"""
from __future__ import annotations

import time
import numpy as np
import torch

CONFIG = {
    "name": "latent_reference_constrained_slots_v1",
    "slots": 16,
    "temperature": 0.07,
    "native_log_likelihood_weight": 0.25,
    "reference_effective_mass_ratio": 1.0,
    "em_iterations": 5,
    "initial_lloyd_iterations": 3,
    "beta_pseudocount": 0.5,
    "extra_encoder_forwards": 0,
    "layers": "cached final DINOv3 layer only",
}


def _normalize(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)


def _initialize(q, k):
    # Farthest-point initialization uses every query token without native-mask seeds.
    first = int(torch.argmax(q @ _normalize(q.mean(0, keepdim=True))[0]).item())
    chosen = [first]
    nearest = q @ q[first]
    for _ in range(1, k):
        idx = int(torch.argmin(nearest).item())
        chosen.append(idx)
        nearest = torch.maximum(nearest, q @ q[idx])
    centers = q[chosen].clone()
    for _ in range(CONFIG["initial_lloyd_iterations"]):
        assignment = (q @ centers.T).argmax(1)
        sums = torch.zeros_like(centers).index_add_(0, assignment, q)
        nonempty = torch.linalg.vector_norm(sums, dim=1) > 1e-8
        centers[nonempty] = _normalize(sums[nonempty])
    return centers


def _run(q, r, cov, score, *, device, iterations, slots):
    start = time.perf_counter()
    dev = torch.device(device)
    score_array = np.asarray(score, dtype=np.float32)
    if score_array.ndim != 2:
        raise ValueError("score must be an HxW native response field")
    h, w = score_array.shape
    with torch.inference_mode():
        q = torch.as_tensor(np.asarray(q, dtype=np.float32), device=dev)
        r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=dev)
        c = torch.as_tensor(np.asarray(cov, dtype=np.float32).reshape(-1), device=dev)
        native = torch.as_tensor(score_array.reshape(-1), device=dev)
        if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
            raise ValueError("q and r must be NxD feature matrices with matching D")
        if q.shape[0] != h * w or r.shape[0] != c.numel():
            raise ValueError("feature counts must match score and reference coverage")
        for name, array in (("q", q), ("r", r), ("cov", c), ("score", native)):
            if not bool(torch.isfinite(array).all()):
                raise ValueError(f"{name} contains nonfinite values")
        if bool((c < 0).any() or (c > 1).any()):
            raise ValueError("cov must lie in [0,1]")
        score_clipped_fraction = float(((native < 0) | (native > 1)).float().mean())
        if float(c.sum()) <= 0 or float((1-c).sum()) <= 0:
            raise ValueError("reference must contain both foreground and background")
        q, r = _normalize(q), _normalize(r)
        centers = _initialize(q, min(slots, q.shape[0]))
        temperature = CONFIG["temperature"]
        pseudo = CONFIG["beta_pseudocount"]
        source_mass = q.shape[0] * CONFIG["reference_effective_mass_ratio"]
        # Equal total reference FG/BG mass removes reference object-area bias.
        rf_weight = c * (0.5 * source_mass / c.sum())
        rb_weight = (1-c) * (0.5 * source_mass / (1-c).sum())
        source_membership = torch.softmax((r @ centers.T) / temperature, 1)
        f_mass = (source_membership * rf_weight[:, None]).sum(0)
        b_mass = (source_membership * rb_weight[:, None]).sum(0)
        theta = (f_mass+pseudo) / (f_mass+b_mass+2*pseudo)
        initial_theta = theta.clone()
        native = native.clamp(1e-4, 1-1e-4)
        lf = CONFIG["native_log_likelihood_weight"] * torch.log(native)
        lb = CONFIG["native_log_likelihood_weight"] * torch.log1p(-native)

        def query_e_step():
            sim = (q @ centers.T) / temperature
            fg = sim + torch.log(theta)[None, :] + lf[:, None]
            bg = sim + torch.log1p(-theta)[None, :] + lb[:, None]
            joint = torch.softmax(torch.cat((fg, bg), 1), 1)
            return joint[:, :centers.shape[0]], joint[:, centers.shape[0]:]

        qf, qb = query_e_step()
        initial_field = qf.sum(1)
        history = []
        for step in range(iterations):
            rsim = (r @ centers.T) / temperature
            rf = torch.softmax(rsim + torch.log(theta)[None, :], 1) * rf_weight[:, None]
            rb = torch.softmax(rsim + torch.log1p(-theta)[None, :], 1) * rb_weight[:, None]
            f_mass = qf.sum(0) + rf.sum(0)
            b_mass = qb.sum(0) + rb.sum(0)
            new_theta = (f_mass+pseudo) / (f_mass+b_mass+2*pseudo)
            new_centers = _normalize((qf+qb).T @ q + (rf+rb).T @ r)
            history.append({"step": step+1,
                            "mean_identity_change": float((new_theta-theta).abs().mean()),
                            "mean_center_cosine": float((new_centers*centers).sum(1).mean())})
            theta, centers = new_theta, new_centers
            qf, qb = query_e_step()
        field = qf.sum(1)
        info = {"config": dict(CONFIG), "iterations_run": iterations,
                "slots_run": int(centers.shape[0]), "device": str(dev),
                "reference_effective_mass": float(source_mass),
                "score_clipped_fraction": score_clipped_fraction,
                "initial_slot_identity": initial_theta.cpu().tolist(),
                "final_slot_identity": theta.cpu().tolist(),
                "slot_identity_flips": int(((theta >= .5) != (initial_theta >= .5)).sum()),
                "initial_to_final_token_flips": int(((field >= .5) != (initial_field >= .5)).sum()),
                "initial_foreground_fraction": float((initial_field >= .5).float().mean()),
                "final_foreground_fraction": float((field >= .5).float().mean()),
                "history": history}
        result = field.reshape(h, w).cpu().numpy().astype(np.float32)
    info["elapsed_seconds"] = time.perf_counter()-start
    return result, info


def predict(q, r, cov, score, *, device="cpu", extras=None):
    """Five fixed EM updates; return complete token probability field and audit."""
    return _run(q, r, cov, score, device=device,
                iterations=CONFIG["em_iterations"], slots=CONFIG["slots"])


def control(q, r, cov, score, *, device="cpu", extras=None):
    """Identical inputs/initial slots, one readout with no joint EM updates."""
    return _run(q, r, cov, score, device=device, iterations=0, slots=CONFIG["slots"])


def two_slot_control(q, r, cov, score, *, device="cpu", extras=None):
    """Simple two-slot mixture with the same reference/native terms and EM budget."""
    return _run(q, r, cov, score, device=device,
                iterations=CONFIG["em_iterations"], slots=2)


additional_controls = {"two_slot_em": two_slot_control}
