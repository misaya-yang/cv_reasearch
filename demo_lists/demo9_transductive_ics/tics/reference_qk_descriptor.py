"""Frozen source-role appearance-bin conditional native-QK descriptors.

Implements SINGLE_REFERENCE_QK_METHOD.md (2026-10-04), including its matched
3072-wide local/shared-bank attribution pair. It does not encode images, change
weights/attention/Part1, read QueryGT or postprocess masks. The caller captures
last-block post-QK-normalization/post-RoPE Q/K and supplies actual Part1 unit z.
All host branches must recompute complete FoRIS Part2--4 after this injection.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

HEADS = 16
HEAD_WIDTH = 64
FEATURE_WIDTH = 1024
MAX_BINS = 64
TAU = .6
ASSIGNMENT_TEMPERATURE = .07
NATIVE_SCALE = 1/math.sqrt(HEAD_WIDTH)

CARD4 = [
    "Assumption: per-image native Q responses to common source-role appearance-conditioned key means add transferable discrimination beyond final appearance.",
    "Prediction: complete paired mask gain must exceed native and constant-c/final-z/q-only controls; native FN with zero own assignment can read other common FG bins and should gain context FG margin.",
    "Match: retain this specific explanation only if complete mask gain and intended FN/FP changes survive the same source dictionary, complete host and matched local/shared-bank comparison.",
    "Mismatch: empty roles/insufficient common bins explicitly retain native; no gain or gains explained by cheaper controls reject necessity of this conditional-QK mechanism. No layer/head/temperature search.",
]


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(name+" must be a finite floating tensor")


def _unit(value, name):
    _finite(value, name)
    tolerance = 64*torch.finfo(value.dtype).eps
    if not bool((value.norm(dim=-1)-1).abs().le(tolerance).all()):
        raise ValueError(name+" requires caller-supplied unit channel vectors")


def _mean_unit(values, name):
    if not len(values):
        raise ValueError(name+" cannot have zero members")
    mean = values.mean(dim=0)
    norm = mean.norm()
    if not bool(torch.isfinite(mean).all()) or not bool(norm > 0):
        raise ValueError(name+" has a nonfinite/zero mean direction")
    return mean/norm


def _default_cluster(tokens, *, tau):
    """Same specified clamped-Gram/average-linkage metric, not a source replay.

    Production must pass the actual public FoRIS clustering callable. This
    standalone implementation is for mathematical CPU fixtures only.
    """
    if len(tokens) == 1:
        return torch.zeros(1, dtype=torch.long, device=tokens.device)
    from sklearn.cluster import AgglomerativeClustering
    distance = (1-(tokens@tokens.T).clamp(-1, 1)).detach().cpu().numpy()
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed",
        linkage="average", distance_threshold=1-tau).fit_predict(distance)
    return torch.as_tensor(labels, dtype=torch.long, device=tokens.device)


def build_reference_dictionary(z_source, source_role, *, cluster_fn=None):
    """Rolewise public clustering, then deterministic within-role capacity merge.

    Every prototype is recomputed from ORIGINAL member z vectors, never an
    unweighted mean of normalized prototypes. Equal merge distances choose the
    lexicographically smallest pair of cluster minimum original patch IDs.
    """
    _unit(z_source, "source Part1 z")
    if z_source.ndim != 2 or z_source.shape[1] != FEATURE_WIDTH or not len(z_source):
        raise ValueError("source z must retain N x 1024 physical patch vectors")
    if not isinstance(source_role, torch.Tensor) or source_role.dtype != torch.bool or source_role.numel() != len(z_source):
        raise ValueError("Exact public Stage2 bool Source roles required")
    role = source_role.reshape(-1).to(z_source.device)
    if not bool(role.any()) or not bool((~role).any()):
        return dict(state="NATIVE_FALLBACK", reason="SOURCE_ROLE_EMPTY",
                    prototypes=None, roles=None, member_ids=(), audit=dict(
                        source_FG=int(role.sum()), source_BG=int((~role).sum())))
    cluster = _default_cluster if cluster_fn is None else cluster_fn
    if not callable(cluster):
        raise TypeError("Source clustering callable required")
    groups = []
    for is_fg in (True, False):
        ids = torch.nonzero(role == is_fg, as_tuple=False).flatten()
        tokens = z_source[ids]
        labels = cluster(tokens, tau=TAU)
        labels = torch.as_tensor(labels, device=z_source.device)
        if labels.ndim != 1 or len(labels) != len(ids) or labels.dtype not in (torch.int32, torch.int64) or not bool((labels >= 0).all()):
            raise ValueError("Source clustering returned invalid original member labels")
        for label in labels.unique(sorted=True):
            members = tuple(sorted(int(i) for i in ids[labels == label].detach().cpu().tolist()))
            if not members:
                raise ValueError("Empty original source cluster is not silently repaired")
            groups.append(dict(members=members, role=is_fg))
    groups.sort(key=lambda group: group["members"][0])
    initial_count = len(groups)
    prototypes = torch.stack([_mean_unit(z_source[list(group["members"])], "source cluster") for group in groups])
    roles = torch.tensor([group["role"] for group in groups], device=z_source.device, dtype=torch.bool)
    active = torch.ones(initial_count, device=z_source.device, dtype=torch.bool)
    merge_records = []
    if initial_count > MAX_BINS:
        distances = 1-(prototypes@prototypes.T).clamp(-1, 1)
        ordered = torch.arange(initial_count, device=z_source.device)
        legal = (ordered[:, None] < ordered[None, :]) & (roles[:, None] == roles[None, :])
        distances = distances.masked_fill(~legal, torch.inf)
        count = initial_count
        while count > MAX_BINS:
            index = int(distances.reshape(-1).argmin())
            left, right = divmod(index, initial_count)
            distance = float(distances[left, right])
            if not math.isfinite(distance) or not bool(active[left]) or not bool(active[right]) or bool(roles[left] != roles[right]):
                raise RuntimeError("No legal within-role capacity merge; no cross-role fallback")
            members = tuple(sorted(groups[left]["members"]+groups[right]["members"]))
            merge_records.append(dict(left_min=groups[left]["members"][0],
                                      right_min=groups[right]["members"][0],
                                      role="FG" if groups[left]["role"] else "BG",
                                      cosine_distance=distance,
                                      merged_member_count=len(members)))
            groups[left] = dict(members=members, role=groups[left]["role"])
            groups[right] = dict(members=(), role=groups[right]["role"])
            prototypes[left] = _mean_unit(z_source[list(members)], "capacity-merged source cluster")
            active[right] = False
            distances[right, :] = torch.inf
            distances[:, right] = torch.inf
            refreshed = 1-(prototypes@prototypes[left]).clamp(-1, 1)
            valid = active & (roles == roles[left])
            distances[left, :] = torch.where(valid & (ordered > left), refreshed, torch.inf)
            distances[:, left] = torch.where(valid & (ordered < left), refreshed, torch.inf)
            count -= 1
    retained = torch.nonzero(active, as_tuple=False).flatten()
    result_prototypes = prototypes[retained]
    result_roles = roles[retained]
    members = tuple(groups[int(i)]["members"] for i in retained.detach().cpu().tolist())
    if len(members) > MAX_BINS or not bool(result_roles.any()) or not bool((~result_roles).any()) or sum(map(len, members)) != len(z_source):
        raise RuntimeError("Source dictionary lost members/roles or exceeded capacity")
    return dict(state="READY", prototypes=result_prototypes, roles=result_roles,
                member_ids=members, audit=dict(initial_source_clusters=initial_count,
                    retained_source_clusters=len(members), tau=TAU,
                    capacity=MAX_BINS, merge_tie_rule="smallest original member ID pair",
                    clustering="actual_public_callable" if cluster_fn is not None else "standalone_specified_average_linkage",
                    all_source_members_retained=True, rolewise_only=True,
                    capacity_merges=merge_records))


def soft_assignments(z_pair, prototypes):
    """Identical S/Q cosine gate and exponential weights; no role one-hot."""
    _unit(z_pair, "paired Part1 z")
    _unit(prototypes, "source dictionary prototypes")
    cosine = torch.einsum("bnd,kd->bnk", z_pair, prototypes)
    weights = torch.where(cosine >= TAU,
                          ((cosine-1)/ASSIGNMENT_TEMPERATURE).exp(), 0.)
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("Nonfinite frozen soft assignment")
    return weights


def _exp_l2(logits):
    _finite(logits, "conditional response logits")
    response = (logits-logits.max(dim=-1, keepdim=True).values).exp()
    norm = response.norm(dim=-1, keepdim=True)
    if not bool((norm > 0).all()) or not bool(torch.isfinite(norm).all()):
        raise RuntimeError("Invalid conditional response norm")
    return response/norm


def _pack_heads(response, width):
    if response.ndim != 4 or response.shape[1] != HEADS or response.shape[-1] > width:
        raise ValueError("Response must retain all 16 heads and fit the frozen bin width")
    padded = F.pad(response, (0, width-response.shape[-1]))
    descriptor = padded.permute(0, 2, 1, 3).reshape(response.shape[0], response.shape[2], HEADS*width)/math.sqrt(HEADS)
    _unit(descriptor, "packed conditional descriptor")
    return descriptor


def conditional_qk_context(q, k, weights, *, attention_scale=NATIVE_SCALE):
    """Exact exp(mean logit) readout with self subtraction, no NxN attention.

    Algebra uses q dot B minus w*(q dot k_i), avoiding a BxHxNxKxd tensor.
    Each bin must have at least two nonzero weights in each image.
    """
    if q.shape != k.shape or q.ndim != 4 or q.shape[0] != 2 or q.shape[1] != HEADS or q.shape[-1] != HEAD_WIDTH:
        raise ValueError("Post-norm/post-RoPE physical q/k must be 2x16xNx64")
    _finite(q, "native q")
    _finite(k, "native k")
    if weights.shape[:2] != (2, q.shape[2]) or weights.ndim != 3 or not 2 <= weights.shape[-1] <= MAX_BINS:
        raise ValueError("At least two frozen common bins are required")
    if q.dtype != k.dtype or q.device != k.device or weights.dtype != q.dtype or weights.device != q.device:
        raise ValueError("Common arithmetic precision/device is required")
    if float(attention_scale) != NATIVE_SCALE:
        raise ValueError("Frozen checkpoint native attention scale must be 1/sqrt64")
    _finite(weights, "common assignments")
    if not bool((weights >= 0).all()) or not bool(((weights > 0).sum(dim=1) >= 2).all()):
        raise ValueError("Each common bin needs >=2 nonzero members on both images")
    masses = weights.sum(dim=1)
    banks = torch.einsum("bnc,bhnd->bhcd", weights, k)
    denominator = masses[:, None, :]-weights
    if not bool((denominator > 0).all()):
        raise RuntimeError("Leave-self denominator is not positive; no repair")
    numerator = torch.einsum("bhnd,bhcd->bhnc", q, banks)
    self_logit = (q*k).sum(dim=-1)
    logits = (numerator-weights[:, None, :, :]*self_logit[:, :, :, None])/denominator[:, None, :, :]
    logits = logits*attention_scale
    context = _pack_heads(_exp_l2(logits), MAX_BINS)
    return context, dict(masses=masses, key_sums=banks,
                        key_means=banks/masses[:, None, :, None],
                        leave_self_denominator=denominator,
                        logits=logits)


def centroid_z_context(z_pair, weights):
    """Same bins, same leave-self; one final-z response repeated into16 slots."""
    _unit(z_pair, "paired Part1 z")
    masses = weights.sum(dim=1)
    denominator = masses[:, None, :]-weights
    if not bool((denominator > 0).all()):
        raise RuntimeError("Same-bin final-z leave-self denominator failed")
    sums = torch.einsum("bnc,bnd->bcd", weights, z_pair)
    numerator = torch.einsum("bnd,bcd->bnc", z_pair, sums)
    numerator = numerator-weights*(z_pair*z_pair).sum(dim=-1, keepdim=True)
    logits = numerator/denominator/ASSIGNMENT_TEMPERATURE
    response = _exp_l2(logits)[:, None, :, :].expand(-1, HEADS, -1, -1)
    return _pack_heads(response, MAX_BINS)


def local_shared_bank_contexts(q, key_means, *, attention_scale=NATIVE_SCALE):
    """Section8 matched pair: NO self exclusion, both pad2K to128/head.

    Local uses [B_X,B_X]; shared uses [B_S,B_Q] for both images. This 3072-wide
    pair is separate from the default 2048-wide leave-self method.
    """
    _finite(q, "native q")
    _finite(key_means, "conditional key means")
    if q.ndim != 4 or q.shape[:2] != (2, HEADS) or q.shape[-1] != HEAD_WIDTH or key_means.ndim != 4 or key_means.shape[:2] != (2, HEADS) or key_means.shape[-1] != HEAD_WIDTH or not 2 <= key_means.shape[2] <= MAX_BINS:
        raise ValueError("Full fixed native heads and common key-bank coordinates required")
    if q.dtype != key_means.dtype or q.device != key_means.device or float(attention_scale) != NATIVE_SCALE:
        raise ValueError("Matched pair requires common precision/native scale")
    local = torch.cat((key_means, key_means), dim=2)
    shared = torch.cat((key_means[0], key_means[1]), dim=1)[None].expand(2, -1, -1, -1)
    local_logits = torch.einsum("bhnd,bhcd->bhnc", q, local)*attention_scale
    shared_logits = torch.einsum("bhnd,bhcd->bhnc", q, shared)*attention_scale
    return (_pack_heads(_exp_l2(local_logits), 2*MAX_BINS),
            _pack_heads(_exp_l2(shared_logits), 2*MAX_BINS))


def _paired_feature(z_pair, context):
    feature = torch.cat((z_pair, context), dim=-1)/math.sqrt(2)
    _unit(feature, "equal-weight concatenated feature")
    return dict(source=feature[0], query=feature[1])


def build_qk_bundle(z_source, z_query, q, k, source_role, *,
                    attention_scale=NATIVE_SCALE, cluster_fn=None):
    """All fixed arms from one original S/Q forward, all predictions still absent.

    Features[arm] is {'source':NxC,'query':NxC}. On legitimate fallback features
    is empty and ALL arms retain native; a caller must keep this case in scores.
    Intermediate tensors stay in RAM; this module writes no files or features.
    """
    for value, name in ((z_source, "source Part1 z"), (z_query, "query Part1 z")):
        _unit(value, name)
        if value.ndim != 2 or value.shape[1] != FEATURE_WIDTH or not len(value) or value.dtype not in (torch.float32, torch.float64):
            raise ValueError("Actual Part1 unit Nx1024 FP32/FP64 features required")
    if z_source.shape != z_query.shape or z_source.device != z_query.device or z_source.dtype != z_query.dtype:
        raise ValueError("Original same-batch S/Q feature grids/precision required")
    _finite(q, "captured native q")
    _finite(k, "captured native k")
    if q.shape != k.shape or q.shape != (2, HEADS, len(z_source), HEAD_WIDTH) or q.device != z_source.device or k.device != z_source.device:
        raise ValueError("Exactly2x16xphysical-patchNx64 native Q/K required; remove5prefix upstream")
    if q.dtype != z_source.dtype or k.dtype != z_source.dtype or float(attention_scale) != NATIVE_SCALE:
        raise ValueError("Same native floating arithmetic and frozen scale required; no silent casting")
    audit = dict(method="source-role appearance conditional native QK mean",
        heads=HEADS, head_width=HEAD_WIDTH, patch_feature_width=FEATURE_WIDTH,
        native_attention_scale=float(attention_scale), tau=TAU,
        assignment_temperature=ASSIGNMENT_TEMPERATURE, max_bins=MAX_BINS,
        common_bin_nonzero_members_per_image_minimum=2,
        self_exclusion="only i=j in default and centroid-z; none in3072pair",
        new_encoder_calls=0, changed_attention=False, changed_weights=False,
        caller_native_postnorm_postRoPE_capture_required=True,
        injection="after original Part1; full Part2--4 recomputed per branch",
        original_basis_reused_on_augmented_features=False,
        QueryGT_used=False, training=False, mask_quality_verified=False,
        arithmetic_dtype=str(z_source.dtype), fallback_applies_all_arms=True,
        constant_c_definition="all1024coordinates 1/sqrt1024, shared by allpatches")
    dictionary = build_reference_dictionary(z_source, source_role, cluster_fn=cluster_fn)
    if dictionary["state"] != "READY":
        return dict(state="NATIVE_FALLBACK", features={}, audit={**audit,
            "fallback_reason":dictionary["reason"], "dictionary":dictionary["audit"]},
            dictionary=dictionary, assignments=None, mechanism=None)
    z_pair = torch.stack((z_source, z_query))
    weights_all = soft_assignments(z_pair, dictionary["prototypes"])
    nonzero_counts = (weights_all > 0).sum(dim=1)
    common = (nonzero_counts >= 2).all(dim=0)
    common_ids = torch.nonzero(common, as_tuple=False).flatten()
    common_roles = dictionary["roles"][common]
    dictionary = {**dictionary, "common_ids":common_ids,
                  "common_roles":common_roles, "nonzero_counts":nonzero_counts}
    if len(common_ids) < 2 or not bool(common_roles.any()):
        reason = "INSUFFICIENT_COMMON_BINS" if len(common_ids) < 2 else "NO_COMMON_SOURCE_FG_BIN"
        return dict(state="NATIVE_FALLBACK", features={}, audit={**audit,
            "fallback_reason":reason, "common_bins":len(common_ids),
            "common_FG_bins":int(common_roles.sum()), "dictionary":dictionary["audit"]},
            dictionary=dictionary, assignments=None, mechanism=None)
    weights = weights_all[:, :, common]
    qk_context, qk_fields = conditional_qk_context(q, k, weights, attention_scale=attention_scale)
    final_context = centroid_z_context(z_pair, weights)
    qnorm = q.norm(dim=-1, keepdim=True)
    if not bool((qnorm > 0).all()) or not bool(torch.isfinite(qnorm).all()):
        raise RuntimeError("q-only control has zero/nonfinite head direction; no repair")
    qonly = (q/qnorm).permute(0, 2, 1, 3).reshape(2, len(z_source), FEATURE_WIDTH)/math.sqrt(HEADS)
    _unit(qonly, "q-only normalized context")
    constant = torch.full_like(z_pair, 1/math.sqrt(FEATURE_WIDTH))
    local_context, shared_context = local_shared_bank_contexts(q, qk_fields["key_means"], attention_scale=attention_scale)
    features = {"main":_paired_feature(z_pair, qk_context),
                "constant_c":_paired_feature(z_pair, constant),
                "centroid_z":_paired_feature(z_pair, final_context),
                "qonly":_paired_feature(z_pair, qonly),
                "duplicate_z":_paired_feature(z_pair, z_pair),
                "local_bank":_paired_feature(z_pair, local_context),
                "shared_bank":_paired_feature(z_pair, shared_context)}
    return dict(state="READY", features=features,
        audit={**audit, "common_bins":len(common_ids),
               "common_FG_bins":int(common_roles.sum()),
               "common_BG_bins":int((~common_roles).sum()),
               "dictionary":dictionary["audit"],
               "source_query_bank_exact_equal":bool(torch.equal(qk_fields["key_means"][0], qk_fields["key_means"][1])),
               "feature_widths":{name:fields["source"].shape[1] for name,fields in features.items()}},
        dictionary=dictionary, assignments=dict(source=weights[0], query=weights[1]),
        mechanism=dict(qk_context=dict(source=qk_context[0], query=qk_context[1]),
            centroid_context=dict(source=final_context[0], query=final_context[1]),
            query_assignment_zero=(weights[1] == 0).all(dim=1),
            common_foreground_exists=True,
            key_means=qk_fields["key_means"],
            qk_logits=qk_fields["logits"],
            query_common_FG_assignment_zero=(weights[1][:, common_roles] == 0).all(dim=1)))


def _server_cpu_check():
    """Five bounded mathematical checks, callable ONLY on the preparation server."""
    import sys
    from pathlib import Path
    if not sys.platform.startswith("linux") or not str(Path(__file__).resolve()).startswith("/root/"):
        raise RuntimeError("SERVER-only numerical check; local use is AST inspection only")
    if torch.cuda.is_initialized():
        raise RuntimeError("This CPU contract must not initialize/use CUDA")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(6040)
    # Two known appearance groups, four patches each; no pretrained model/GT.
    z = torch.zeros((8, FEATURE_WIDTH), dtype=torch.float64)
    z[:4, 0] = 1
    z[4:, 1] = 1
    z[:, 2] = torch.linspace(.01, .08, 8, dtype=torch.float64)
    z = F.normalize(z, dim=-1)
    role = torch.arange(8) < 4
    q = torch.randn((2, HEADS, 8, HEAD_WIDTH), dtype=torch.float64, generator=generator)
    k = torch.randn((2, HEADS, 8, HEAD_WIDTH), dtype=torch.float64, generator=generator)
    def known_role_cluster(tokens, *, tau):
        if tau != TAU:
            raise AssertionError("Frozen clustering threshold changed")
        return torch.zeros(len(tokens), dtype=torch.long)
    bundle = build_qk_bundle(z, z.clone(), q, k, role, cluster_fn=known_role_cluster)
    assert bundle["state"] == "READY" and bundle["audit"]["common_bins"] == 2
    for name, fields in bundle["features"].items():
        expected = 3072 if name in ("local_bank", "shared_bank") else 2048
        assert fields["source"].shape == (8, expected) and fields["query"].shape == (8, expected)
        _unit(fields["source"], name)
        _unit(fields["query"], name)
    checks = ["all_frozen_arms_unit_shapes_and_source_query_shared_assignment"]
    weights = torch.stack((bundle["assignments"]["source"], bundle["assignments"]["query"]))
    # Literal exclusion construction versus factorized algebra on a single head.
    literal = []
    for i in range(8):
        rows = []
        for column in range(2):
            w = weights[0, :, column].clone()
            w[i] = 0
            mean = (w[:, None]*k[0, 0]).sum(0)/w.sum()
            rows.append((q[0, 0, i]*mean).sum()*NATIVE_SCALE)
        literal.append(torch.stack(rows))
    literal = torch.stack(literal)
    assert torch.allclose(literal, bundle["mechanism"]["qk_logits"][0, 0], atol=2e-13, rtol=0)
    literal_z = []
    for i in range(8):
        rows = []
        for column in range(2):
            w = weights[0, :, column].clone()
            w[i] = 0
            mean = (w[:, None]*z).sum(0)/w.sum()
            rows.append((z[i]*mean).sum()/ASSIGNMENT_TEMPERATURE)
        literal_z.append(torch.stack(rows))
    expected_centroid = _pack_heads(_exp_l2(torch.stack(literal_z))[None, None].expand(2, HEADS, -1, -1), MAX_BINS)
    assert torch.allclose(expected_centroid[0], bundle["mechanism"]["centroid_context"]["source"], atol=2e-13, rtol=0)
    checks.append("leave_only_self_factorized_QK_and_final_z_match_literal_weighted_means")
    repeated_keys = torch.stack((k[0], k[0]))
    means = torch.einsum("bnc,bhnd->bhcd", weights, repeated_keys)/weights.sum(1)[:, None, :, None]
    local, shared = local_shared_bank_contexts(q, means)
    assert torch.equal(local, shared)
    checks.append("section8_same_banks_identical_matched3072_context_no_self_exclusion")
    for invalid_role in (torch.ones(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)):
        fallback = build_qk_bundle(z, z, q, k, invalid_role, cluster_fn=known_role_cluster)
        assert fallback["state"] == "NATIVE_FALLBACK" and fallback["features"] == {}
    no_common = build_qk_bundle(z, -z, q, k, role, cluster_fn=known_role_cluster)
    assert no_common["state"] == "NATIVE_FALLBACK"
    bad = q.clone()
    bad[0, 0, 0, 0] = torch.nan
    try:
        build_qk_bundle(z, z, bad, k, role, cluster_fn=known_role_cluster)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite input silently became fallback")
    checks.append("legitimate_role_common_bin_fallbacks_and_nonfinite_error")
    # Dimension-wise function-equivalent Q/K rescaling preserves the readout.
    half_axis = torch.linspace(.8, 1.2, HEAD_WIDTH//2, dtype=torch.float64)
    axis = torch.cat((half_axis, half_axis))  # RoPE paired coordinates share scale.
    original, _ = conditional_qk_context(q, k, weights)
    rescaled, _ = conditional_qk_context(q/axis, k*axis, weights)
    assert torch.allclose(original, rescaled, atol=2e-13, rtol=0)
    checks.append("QK_bilinear_reparameterization_readout_invariance_not_quality")
    return dict(state="SERVER_QK_DESCRIPTOR_FIVE_CPU_CONTRACTS_PASSED",
                checks=checks, CUDA_initialized=torch.cuda.is_initialized(),
                pretrained_model=False, QueryGT=False, actual_native_capture_verified=False,
                complete_host_mask_parity_verified=False, quality_gain_verified=False)


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve previous receipt; use a new output path")
    receipt = _server_cpu_check()
    receipt["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps({key:receipt[key] for key in ("state", "CUDA_initialized")}))
