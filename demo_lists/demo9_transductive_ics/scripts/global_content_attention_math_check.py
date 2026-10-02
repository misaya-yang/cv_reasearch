#!/usr/bin/env python3
"""NumPy/CPU audit of the native-trajectory LOCAL patch->global readout.

Synthetic algebra only: no model, timm, torch, server, weights or GPU required.
The delta API below takes native row LSE explicitly. Public PyTorch SDPA does
not expose it. Numerically unsafe rows deliberately recompute full logits here;
this is a correctness fallback, not evidence of a five-token runtime cost.
Native outputs/states are never overwritten by this diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def logsumexp(x):
    maximum = np.max(x, axis=-1, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    with np.errstate(divide="ignore"):
        result = safe_maximum + np.log(np.exp(x - safe_maximum).sum(-1, keepdims=True))
    return np.where(finite, result, -np.inf)


def attention(logits, values):
    z = logsumexp(logits)
    with np.errstate(invalid="ignore"):
        probability = np.exp(logits - z)
    probability = np.nan_to_num(probability, nan=0.0)
    return probability @ values


def chunk_attention(logits, values, chunk=3):
    return np.concatenate([attention(logits[..., lo:lo + chunk, :], values)
                           for lo in range(0, logits.shape[-2], chunk)], axis=-2)


def local_logits(q, k, qr, kr, global_count, scale, bias=None):
    """Only patch queries are returned; native global-query outputs are unused."""
    result = (qr[..., global_count:, :] @ np.swapaxes(kr, -1, -2)) * scale
    result[..., :global_count] = (
        q[..., global_count:, :] @ np.swapaxes(k[..., :global_count, :], -1, -2)
    ) * scale
    return result if bias is None else result + bias


def lift2d(q, k, v, qr, kr, global_count, scale, bias=None):
    """Patch Q=[qr,q]; global K=[0,k], patch K=[kr,0]; V=[v,0]."""
    qlift = np.concatenate([qr[..., global_count:, :], q[..., global_count:, :]], -1)
    zero = np.zeros_like(k)
    first = kr.copy()
    first[..., :global_count, :] = 0
    second = zero.copy()
    second[..., :global_count, :] = k[..., :global_count, :]
    klift = np.concatenate([first, second], -1)
    vlift = np.concatenate([v, np.zeros_like(v)], -1)
    logits = (qlift @ np.swapaxes(klift, -1, -2)) * scale
    if bias is not None:
        logits = logits + bias
    return attention(logits, vlift)[..., :v.shape[-1]], logits


def naive_delta(native_u, native_z, old_global, new_global, global_values):
    """Literal identity; intentionally unguarded, used to expose cancellation."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a = np.exp(old_global - native_z)
        b = np.exp(new_global - native_z)
        r = 1 - a.sum(-1, keepdims=True) + b.sum(-1, keepdims=True)
        numerator_form = (native_u - a @ global_values + b @ global_values) / r
        residual_form = native_u + ((b - a) @ global_values
                                    - (b - a).sum(-1, keepdims=True) * native_u) / r
    return numerator_form, residual_form


def guarded_delta(native_u, native_z, old_logits, new_global, values, global_count):
    """Exact algebra with a numerical guard and an explicit recompute fallback.

    old_logits are supplied ONLY so the CPU fallback can reconstruct lost patch
    contributions. An implementation holding only U/z/global K/V cannot recover
    tiny patch numerators already rounded away in U.
    """
    old_global = old_logits[..., :global_count]
    if np.array_equal(old_global, new_global):
        return native_u.copy(), {"exact_no_change": True, "fallback_rows": 0}
    if global_count == 0:
        return native_u.copy(), {"exact_no_change": True, "fallback_rows": 0}
    with np.errstate(invalid="ignore"):
        new_relative = new_global - native_z
        old_relative = old_global - native_z
    shift = np.maximum(0.0, np.max(new_relative, axis=-1, keepdims=True))
    # Under a huge positive replacement logit, b itself can overflow. Scaling
    # both numerator and denominator by exp(-shift) avoids constructing b.
    factor = np.exp(-shift)
    a = np.nan_to_num(np.exp(old_relative), nan=0.0)
    bscaled = np.nan_to_num(np.exp(new_relative - shift), nan=0.0)
    old_global_mass = a.sum(-1, keepdims=True)
    rscaled = factor * (1 - old_global_mass) + bscaled.sum(-1, keepdims=True)
    numerator = factor * native_u - (factor * a) @ values[..., :global_count, :]
    numerator += bscaled @ values[..., :global_count, :]
    unsafe = ((old_global_mass >= 1 - 1e-10) | ~np.isfinite(rscaled)
              | (rscaled <= 1e-12) | ~np.isfinite(native_z))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / rscaled
    if unsafe.any():
        changed_logits = old_logits.copy()
        changed_logits[..., :global_count] = new_global
        reference = chunk_attention(changed_logits, values)
        result = np.where(unsafe, reference, result)
    return result, {"exact_no_change": False, "fallback_rows": int(unsafe.sum()),
                    "exp_shift_max": float(np.max(shift))}


def rotate(x, phases, global_count):
    """Two-dimensional interleaved rotations; phases encode actual frequencies."""
    result = x.copy()
    pairs = x[..., global_count:, :].reshape(*x[..., global_count:, :].shape[:-1], -1, 2)
    cosine, sine = np.cos(phases), np.sin(phases)
    result[..., global_count:, :] = np.stack(
        [pairs[..., 0] * cosine - pairs[..., 1] * sine,
         pairs[..., 0] * sine + pairs[..., 1] * cosine], -1).reshape(
             x[..., global_count:, :].shape)
    return result


def max_error(x, y):
    return float(np.max(np.abs(x - y)))


def run():
    rng = np.random.default_rng(1729)
    shape = (2, 2, 9, 8)
    q, k, v = [rng.normal(size=shape) for _ in range(3)]
    g, d = 3, shape[-1]
    scale = d ** -0.5
    periods = np.array([0.9, 2.1])
    coords = rng.normal(size=(shape[-2] - g, 2))
    phases = (2 * np.pi * coords[..., None] / periods).reshape(shape[-2] - g, d // 2)
    delta = (2 * np.pi * np.array([0.37, -0.61])[:, None] / periods).reshape(d // 2)
    qr, kr = rotate(q, phases, g), rotate(k, phases, g)
    native = (qr[..., g:, :] @ np.swapaxes(kr, -1, -2)) * scale
    modified = local_logits(q, k, qr, kr, g, scale)
    u, z = attention(native, v), logsumexp(native)
    target = attention(modified, v)
    cases = []

    def record(name, ok, **details):
        cases.append({"name": name, "passed": bool(ok), **details})

    error = max_error(chunk_attention(modified, v), target)
    record("direct_equals_query_chunks", error < 1e-12, max_error=error)

    lifted, lifted_logits = lift2d(q, k, v, qr, kr, g, scale)
    error = max(max_error(lifted, target), max_error(lifted_logits, modified))
    wrong_scale = attention(lifted_logits / np.sqrt(2), v)
    split = (attention(modified[..., :g], v[..., :g, :])
             + attention(modified[..., g:], v[..., g:, :]))
    wrong_scale_difference = max_error(wrong_scale, target)
    split_difference = max_error(split, target)
    record("lift2d_joint_softmax_original_scale", error < 1e-12
           and wrong_scale_difference > 1e-4 and split_difference > 1e-4,
           max_error=error, wrong_scale_difference=wrong_scale_difference,
           separate_softmax_difference=split_difference)

    first, second = naive_delta(u, z, native[..., :g], modified[..., :g], v[..., :g, :])
    guarded, audit = guarded_delta(u, z, native, modified[..., :g], v, g)
    error = max(max_error(first, target), max_error(second, target), max_error(guarded, target))
    record("ordinary_delta_both_identities", error < 1e-12, max_error=error, **audit)

    unchanged, audit = guarded_delta(u, z, native, native[..., :g], v, g)
    record("no_change_returns_exact_native_u", np.array_equal(unchanged, u), **audit)

    huge_global = modified[..., :g].copy()
    huge_global[..., 0] = 1000.0
    huge_full = native.copy()
    huge_full[..., :g] = huge_global
    stable, audit = guarded_delta(u, z, native, huge_global, v, g)
    error = max_error(stable, attention(huge_full, v))
    raw, _ = naive_delta(u, z, native[..., :g], huge_global, v[..., :g, :])
    record("huge_b_exp_shift_avoids_overflow", error < 1e-12 and not np.isfinite(raw).all(),
           max_error=error, naive_has_nonfinite=bool(not np.isfinite(raw).all()), **audit)

    # exp(-40) patch mass is already lost when stored U is rounded to float64.
    nearly_all_global = np.array([[40.0, 40.0, 0.0, 0.0]])
    values = np.array([[2.0, -1.0], [2.0, -1.0], [-3.0, 4.0], [-1.0, 2.0]])
    near_u, near_z = attention(nearly_all_global, values), logsumexp(nearly_all_global)
    replacement = np.array([[-40.0, -40.0]])
    changed = nearly_all_global.copy()
    changed[..., :2] = replacement
    near_target = attention(changed, values)
    unstable, unstable_residual = naive_delta(near_u, near_z, nearly_all_global[..., :2], replacement, values[:2])
    stable, audit = guarded_delta(near_u, near_z, nearly_all_global, replacement, values, 2)
    error = max_error(stable, near_target)
    unstable_error = max_error(unstable, near_target)
    record("near_all_global_mass_detects_lost_patch_numerator", error < 1e-12
           and unstable_error > 0.1 and audit["fallback_rows"] == 1,
           max_error=error, naive_numerator_error=unstable_error,
           naive_residual_error=max_error(unstable_residual, near_target), **audit)

    shifted_qr, shifted_kr = rotate(q, phases + delta, g), rotate(k, phases + delta, g)
    shifted_local = local_logits(q, k, shifted_qr, shifted_kr, g, scale)
    shifted_native = (shifted_qr[..., g:, :] @ np.swapaxes(shifted_kr, -1, -2)) * scale
    conditional_error = max_error(shifted_local, modified)
    positive_control = max_error(shifted_native[..., :g], native[..., :g])
    record("fixed_native_qk_conditional_origin_test", conditional_error < 1e-12
           and positive_control > 1e-3, local_logits_error=conditional_error,
           native_patch_to_global_logits_change=positive_control,
           claim="Frozen Q/K only; not a whole native-trajectory invariance claim")

    # The LOCAL branch does not update hidden states. At layer2, the native
    # Q/K/V inherit layer1's origin sensitivity: a positive counterexample.
    x = rng.normal(size=(1, 1, shape[-2], d))
    weights = [rng.normal(size=(d, d)) * 0.2 for _ in range(6)]

    def native_layer(x0, angle, weights0):
        qq, kk, vv = [x0 @ w for w in weights0]
        qrot, krot = rotate(qq, angle, g), rotate(kk, angle, g)
        logits = (qrot @ np.swapaxes(krot, -1, -2)) * scale
        return x0 + attention(logits, vv)

    state = native_layer(x, phases, weights[:3])
    shifted_state = native_layer(x, phases + delta, weights[:3])

    def local_layer_readout(x0, angle):
        qq, kk, vv = [x0 @ w for w in weights[3:]]
        return attention(local_logits(qq, kk, rotate(qq, angle, g),
                                      rotate(kk, angle, g), g, scale), vv)

    local_difference = max_error(local_layer_readout(state, phases),
                                 local_layer_readout(shifted_state, phases + delta))
    record("native_trajectory_two_layer_positive_counterexample", local_difference > 1e-5,
           local_layer2_output_change=local_difference,
           interpretation="Upstream native position dependence survives local readout")

    identical_values = np.broadcast_to(np.array([0.5, -0.2, 1.7, 0.0, 0.1, 0.2, 0.3, -0.7]), v.shape)
    native_identical = attention(native, identical_values)
    changed_identical, _ = guarded_delta(native_identical, z, native, modified[..., :g], identical_values, g)
    error = max_error(changed_identical, native_identical)
    record("global_values_equal_u_imply_no_effect", error < 1e-12, max_error=error)

    bias = np.zeros(modified.shape[-2:])
    bias[:, 0] = -np.inf
    bias[::2, 4] = -np.inf
    masked_native, masked_changed = native + bias, modified + bias
    masked_u, masked_z = attention(masked_native, v), logsumexp(masked_native)
    masked_delta, _ = guarded_delta(masked_u, masked_z, masked_native, masked_changed[..., :g], v, g)
    masked_lift, _ = lift2d(q, k, v, qr, kr, g, scale, bias)
    zero_lift, _ = lift2d(q, k, v, qr, kr, 0, scale)
    zero_native = attention((qr @ np.swapaxes(kr, -1, -2)) * scale, v)
    error = max(max_error(masked_delta, attention(masked_changed, v)),
                max_error(masked_lift, attention(masked_changed, v)),
                max_error(zero_lift, zero_native))
    record("shared_mask_and_prefix_zero", error < 1e-12, max_error=error)

    return {"mechanism": "Local patch-query -> global-key readout on native Q/K/V; no state feedback",
            "scope": "Synthetic NumPy float64 CPU algebra; no model score, GPU or runtime claim",
            "seed": 1729, "cases": cases, "passed": sum(c["passed"] for c in cases),
            "total": len(cases), "all_passed": all(c["passed"] for c in cases),
            "normalizer_requirement": "Native row LSE required; public SDPA does not supply it",
            "numerical_limit": "LSE stabilizes denominator but cannot reconstruct a rounded-away patch numerator; guarded fallback recomputes",
            "whole_model_origin_invariance_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional small JSON result file")
    args = parser.parse_args()
    # Stress cases intentionally evaluate overflowing/unstable naive formulas.
    # Some BLAS builds retain those FP flags until later finite matmuls. The
    # case assertions and allow_nan=False still reject invalid guarded results.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = run()
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.write_text(serialized + "\n")
    print(serialized)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
