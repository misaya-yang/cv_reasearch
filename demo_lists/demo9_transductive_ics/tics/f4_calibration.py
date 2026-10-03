"""Independent F4-004 and fixed query-core deletion control G.

Only legal source endpoints calibrate Part4. Stage2 is untouched. G reads the
original native encoder tensor in the EXACT old R241 B2/token layout and edits
the native final working mask, never applying a second CRF. No query labels,
encoder invocation, fitting, data access or file writes occur in this module.
"""
import torch
import torch.nn.functional as F

UNIT_TOLERANCE = 1e-5  # Numeric domain validation, not a task rejection threshold.
CARD004 = [
    "Assumption: query-extrema BG normalization mis-scales Part4; legal original hard-bank endpoints provide useful FG/BG role separation.",
    "Prediction: all original Stage2/Part3/NN/seed/group tensors remain exact; only Part4 BG normalization and its resulting score/mask may change. No gain is inferred from source geometry.",
    "Match: retain only measured full-original-image benefit over complete A and G; check gained/lost TP/FP and bS-aS without tuning endpoints on queryGT.",
    "Mismatch: bS<=aS/domain unsupported returns native; an active field with no net benefit rejects this fixed calibration. No gap floor, quantile substitution or extra rejection gate.",
]
CARD_G = [
    "Assumption: the already measured fixed native-query double-centre reader remains a fair cheap control in the same cohort and complete-host precision.",
    "Prediction: G is a working-grid subset of native, adds no pixels, performs no second CRF, and must replay frozen old R241 querycore bits where matched inputs exist.",
    "Match: report only complete original-image I/U and TP/FP deletion; its prior DEV241 +.456pp does not transfer to F4 methods or an independent cohort.",
    "Mismatch: absent foreground/background pool returns native; implementation mismatch invalidates comparison, not the old measured control. Do not tune zero or reselect the representation.",
]


def anchored_background(raw_bg, foreground_modes, residual_bg, hard_bank, original_sbn):
    """Exact 004 field: clip((n-aS)/(bS-aS),0,1) iff bS>aS.

    Original hard20% unit global/cluster means are required. No all-token extrema,
    alternate background bank, projection-renormalization or denominator floor.
    Unsupported numerical domains preserve the ORIGINAL field by identity.
    """
    tensors = (raw_bg, foreground_modes, residual_bg, hard_bank, original_sbn)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("Actual source tensors required")
    if raw_bg.shape != original_sbn.shape or raw_bg.ndim != 2:
        raise ValueError("Matching original source grid fields required")
    if foreground_modes.ndim != 2 or residual_bg.ndim != 1 or hard_bank.ndim != 2:
        raise ValueError("FG modes KxD, original r0 D, hard-bank LxD required")
    dim = residual_bg.numel()
    if foreground_modes.shape[1] != dim or hard_bank.shape[1] != dim:
        raise ValueError("Source coordinate mismatch")
    if any(t.device != raw_bg.device or t.dtype != raw_bg.dtype for t in tensors):
        raise ValueError("No mixed source precision/device is permitted")
    meta = dict(definition="sourceFG maximum/source originalhard20 mean-mode maximum",
                aS=None, bS=None, gap=None, active=False, denominator_floor=None,
                Stage2_changed=False, source_only_endpoints=True)
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        return original_sbn, {**meta, "state": "NATIVE_UNSUPPORTED_NONFINITE_DOMAIN"}
    if not len(foreground_modes) or not len(hard_bank):
        return original_sbn, {**meta, "state": "NATIVE_MISSING_SOURCE_ROLE_MODES"}
    norms = (foreground_modes.norm(dim=1), residual_bg.norm().reshape(1), hard_bank.norm(dim=1))
    if any(not bool((value - 1).abs().le(UNIT_TOLERANCE).all()) for value in norms):
        return original_sbn, {**meta, "state": "NATIVE_UNSUPPORTED_NONUNIT_SOURCE_DOMAIN"}
    a = (foreground_modes @ residual_bg).max()
    b = (hard_bank @ residual_bg).max()
    meta.update(aS=float(a), bS=float(b), gap=float(b-a))
    if not bool(b > a):
        return original_sbn, {**meta, "state": "NATIVE_NO_SEPARATED_SOURCE_ENDPOINTS"}
    output = ((raw_bg - a) / (b - a)).clamp(0, 1)
    if not bool(torch.isfinite(output).all()):
        return original_sbn, {**meta, "state": "NATIVE_UNSUPPORTED_NONFINITE_CALIBRATION"}
    return output, {**meta, "state": "SOURCE_ANCHORED_PART4_BG", "active": True,
                    "clipped_low_count": int((raw_bg <= a).sum()),
                    "clipped_high_count": int((raw_bg >= b).sum())}


def _stage_identity(native, replay):
    keys = ("part2_score", "part2_sf", "part2_sbn", "part2_mu_fg", "part2_tgt_denoised",
            "part3_score", "part3_candidate_vote", "part3_seed_prior", "candidate_hard",
            "candidate_vote", "seed_prior", "seed_cluster_labels", "semantic_cluster_labels")
    checked = []
    for key in keys:
        if key not in native["stages"] or key not in replay["stages"]:
            raise ValueError("Actual source stage evidence missing: " + key)
        if not torch.equal(native["stages"][key], replay["stages"][key]):
            raise RuntimeError("004 changed a forbidden upstream stage: " + key)
        checked.append(key)
    if not torch.equal(native["part1"], replay["part1"]):
        raise RuntimeError("004 changed source Part1 coordinates")
    return checked


def run004(ctx, evidence=None):
    """Arm F; independent from all Stage2 F4 conditional-bank interventions."""
    if evidence is None:
        from .f4_conditional import source_evidence
        evidence = source_evidence(ctx)
    if evidence.get("state") != "SUPPORTED":
        result = ctx.replay()
        result["audit"] = {**result.get("audit", {}), "arm": "F4-004",
                           "state": "NATIVE_SOURCE_EVIDENCE_UNSUPPORTED",
                           "evidence_state": evidence.get("state"),
                           "reason": evidence.get("metadata", {}).get("reason"),
                           "extra_encoder_calls": 0, "Stage2_changed": False}
        return result
    if not bool(ctx.host.use_raw_target_scoring):
        result = ctx.replay()
        result["audit"] = {**result.get("audit", {}), "arm": "F4-004",
                           "state": "NATIVE_UNSUPPORTED_GATED_BG_COORDINATES",
                           "extra_encoder_calls": 0, "Stage2_changed": False}
        return result
    field, audit = anchored_background(evidence["raw_bg_map"], evidence["fg_prototypes"],
        evidence["residual_bg"], evidence["hard_bank"], evidence["original_sbn"])
    if not torch.equal(evidence["original_sbn"], ctx.native["stages"]["part2_sbn"]):
        raise RuntimeError("004 evidence is not from the current exact native Stage2")
    if not audit["active"]:
        result = ctx.replay()
    else:
        calls = []
        # Invoke the actual original CLASS body to avoid recursing into the replay
        # Part4 wrapper. Its original SDP/SR methods and clustering cache remain in
        # scope, so both helpers are recomputed from field with unchanged geometry.
        body = type(ctx.host)._part4_semantic_consistency_correction
        def replace(original_result, ctx, score, sf, sbn, cand_soft, seed_prior, tgt_feat):
            if calls:
                raise RuntimeError("Expected exactly one public Part4 call")
            if not torch.equal(sbn, evidence["original_sbn"]):
                raise RuntimeError("004 received changed Stage2 BG normalization")
            calls.append(True)
            return body(ctx.host, score, sf=sf, sbn=field, cand_soft=cand_soft,
                        seed_prior=seed_prior, tgt_feat=tgt_feat)
        result = ctx.replay(part4_override=replace)
        if len(calls) != 1:
            raise RuntimeError("004 complete source Part4 intervention missing")
    unchanged = _stage_identity(ctx.native, result)
    result["audit"] = {**result.get("audit", {}), **audit, "arm": "F4-004",
                       "unchanged_source_stages": unchanged, "Part4_only": True,
                       "extra_encoder_calls": 0, "query_GT_used": False,
                       "BG_bank": "same original hard20 global/unitcluster means",
                       "no_BG_map_as_direct_mask": True}
    return result


def query_core_field(native_work_mask, raw_native):
    """Exact original R241 querycore arithmetic including full B2 token layout."""
    if native_work_mask.ndim != 2 or native_work_mask.dtype != torch.bool:
        raise ValueError("Native final working-grid Boolean mask required")
    if raw_native.ndim != 5 or tuple(raw_native.shape[:2]) != (1, 2):
        raise ValueError("Actual original native B2 map 1x2xDxHxW required")
    if raw_native.dtype not in (torch.float32, torch.float64):
        raise ValueError("Do not mix an old reduced-precision cache into G")
    grid = tuple(raw_native.shape[-2:])
    # R241 used tokens(raw_final): [2,N,D], THEN normalized both image streams;
    # keep this exact layout rather than renormalizing ctx.features['final_raw'].
    pair = raw_native[0].flatten(2).transpose(1, 2)
    query = F.normalize(pair.float(), p=2, dim=-1)[1]
    if not bool(torch.isfinite(query).all()):
        raise ValueError("Nonfinite original query features")
    inside = F.interpolate(native_work_mask[None, None].float(), size=grid,
                           mode="area")[0, 0] > .5
    pool = inside.flatten()
    audit = dict(grid=list(grid), threshold=0., foreground_pool=int(pool.sum()),
                 background_pool=int((~pool).sum()), feature_role="original_finalraw_pre_Part1",
                 representation="exact old R241 fullB2 normalized token layout",
                 no_second_CRF=True, operation_grid="native final working mask",
                 original_mapping="common old evaluator bilinear binary mask >.5")
    if not bool(pool.any()) or bool(pool.all()):
        return None, {**audit, "state": "NATIVE_MISSING_QUERY_POOL"}
    foreground = F.normalize(query[pool].mean(0), dim=0)
    background = F.normalize(query[~pool].mean(0), dim=0)
    margin = (query @ (foreground-background)).reshape(grid)
    return margin, {**audit, "state": "QUERY_DOUBLE_CENTRE_ZERO_DELETION"}


def query_core_remove(ctx):
    """Arm G: edit final working mask, retaining the native source score as metadata."""
    native = ctx.native
    margin, audit = query_core_field(native["mask"], ctx.raw_native)
    if margin is None:
        output = native["mask"].clone()
    else:
        enlarged = F.interpolate(margin[None, None], size=native["mask"].shape,
                                 mode="bilinear", align_corners=False)[0, 0]
        output = native["mask"] & (enlarged > 0)
    if bool((output & ~native["mask"]).any()):
        raise RuntimeError("G must only delete on the final working grid")
    return dict(mask=output, score=native["score"].clone(), stages=native["stages"],
                part1=native["part1"], margin=margin,
                audit={**audit, "arm": "query_core_remove", "query_GT_used": False,
                       "extra_encoder_calls": 0, "added_work_pixels": 0,
                       "deleted_work_pixels": int((native["mask"] & ~output).sum()),
                       "score_role": "unchanged native Part4score; authoritative G prediction is editedmask",
                       "original_size_evaluation_owned_by_common_runner": True})


ARMS = {"F": dict(id="F4-004", function=run004, card4lines=CARD004),
        "G": dict(id="query_core_remove", function=query_core_remove, card4lines=CARD_G)}
