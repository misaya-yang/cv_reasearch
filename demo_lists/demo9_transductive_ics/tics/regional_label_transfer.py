"""Fixed region-observation pixel label transfer, no encoder or Query labels.

Source area is normalized separately inside FG and BG. All twelve S4 x Q3
observation pairs have equal mass. Scores are log density FG minus log density
BG, with strict zero cut and no minmax. CLS/template routing are conventional
operators under test, not asserted contributions or calibrated probabilities.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TAU_L = .07
TAU_G = .07
ARMS = ("raw_local", "regional_local", "old_global_local", "main",
        "source_cls_shuffled", "cls_constant", "global_only",
        "route_top1", "route_top4")


def _normalise(value, name, device, dtype):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.device != device or value.dtype != dtype:
        raise ValueError(name+" must use the common floating arithmetic/device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name+" contains nonfinite input")
    norm = value.norm(dim=-1, keepdim=True)
    if not bool((norm > 0).all()) or not bool(torch.isfinite(norm).all()):
        raise ValueError(name+" has an undefined cosine direction")
    return value/norm


def _ids(value, shape, maximum, device, name):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64) or value.shape != shape:
        raise ValueError(name+" needs exact integer lookup IDs")
    value = value.to(device=device, dtype=torch.long)
    if not bool(((value >= 0) & (value < maximum)).all()):
        raise ValueError(name+" is outside its observed global bank")
    return value


def _prepare(inputs):
    roles = inputs["source_roles"]
    area = inputs["source_area"]
    if not isinstance(roles, torch.Tensor) or roles.ndim != 1 or roles.dtype != torch.bool or not len(roles):
        raise ValueError("Exact nonempty Source anchor roles required")
    if not isinstance(area, torch.Tensor) or area.shape != roles.shape or not bool(torch.isfinite(area).all()) or not bool((area >= 0).all()):
        raise ValueError("Actual Source anchor area must be finite and nonnegative")
    fg_area = float(area[roles.to(area.device)].sum())
    bg_area = float(area[(~roles).to(area.device)].sum())
    if fg_area <= 0:
        raise ValueError("Source foreground empty is an invalid task")
    if bg_area <= 0:
        return None, dict(state="SOURCE_BACKGROUND_EMPTY", SourceFG_area=fg_area,
                          SourceBG_area=bg_area, all_arms_native_fallback=True)
    source = inputs["source_local"]
    query = inputs["query_local"]
    if not isinstance(source, torch.Tensor) or source.dtype not in (torch.float32, torch.float64) or source.ndim != 3 or source.shape[:2] != (4, len(roles)):
        raise ValueError("Source local must retain4 views x all physical anchors")
    device, dtype = source.device, source.dtype
    if not isinstance(query, torch.Tensor) or query.ndim != 3 or query.shape[0] != 3 or query.shape[1] < 1 or query.shape[2] != source.shape[2]:
        raise ValueError("Query local must retain3 views x every physical Query point")
    a, n, c = len(roles), query.shape[1], source.shape[2]
    roles = roles.to(device)
    area = area.to(device=device, dtype=dtype)
    values = dict(source_local=_normalise(source, "Source local", device, dtype),
                  query_local=_normalise(query, "Query local", device, dtype))
    shapes = {"source_raw":(a, c), "query_raw":(n, c),
              "source_global":(83, c), "query_global":(81, c),
              "source_old_global":(83, c), "query_old_global":(81, c)}
    for key, shape in shapes.items():
        if inputs[key].shape != shape:
            raise ValueError(key+" violates the same-observation template contract")
        values[key] = _normalise(inputs[key], key, device, dtype)
    values.update(source_template_ids=_ids(inputs["source_template_ids"], (4, a), 83, device, "Source templates"),
                  query_template_ids=_ids(inputs["query_template_ids"], (3, n), 81, device, "Query templates"),
                  roles=roles, area=area, n=n, a=a, c=c, device=device, dtype=dtype)
    sid, qid = values["source_template_ids"], values["query_template_ids"]
    if (not bool((sid[0] == 0).all()) or not bool((qid[0] == 0).all()) or
        not bool(((sid[1] >= 1) & (sid[1] <= 16)).all()) or
        not bool(((qid[1] >= 1) & (qid[1] <= 16)).all()) or
        not bool(((sid[2] >= 17) & (sid[2] <= 80)).all()) or
        not bool(((qid[2] >= 17) & (qid[2] <= 80)).all()) or
        not bool(torch.equal(sid[3], torch.where(roles, 81, 82)))):
        raise ValueError("Physical-point lookup must preserve whole/coarse/fine/own-role views")
    values["role_indices"] = tuple(torch.nonzero((roles == label) & (area > 0), as_tuple=False).flatten() for label in (True, False))
    values["log_weights"] = tuple(area[index].log()-area[index].sum().log() for index in values["role_indices"])
    return values, dict(state="READY", SourceFG_area=fg_area, SourceBG_area=bg_area,
                        SourceFG_anchor_count=int(roles.sum()), SourceBG_anchor_count=int((~roles).sum()),
                        actual_area_normalized_separately_per_role=True,
                        ignored_zero_area_anchors=int((area == 0).sum()))


def source_cls_permutation(seed=6050):
    """Keep whole0; permute observed CLS only inside coarse/fine/role groups."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.arange(83, dtype=torch.long)
    for start, stop in ((1, 17), (17, 81), (81, 83)):
        permutation[start:stop] = torch.arange(start, stop)[torch.randperm(stop-start, generator=generator)]
    return permutation


def _raw_score(p, query_chunk):
    output = torch.empty(p["n"], device=p["device"], dtype=p["dtype"])
    for start in range(0, p["n"], query_chunk):
        query = p["query_raw"][start:start+query_chunk]
        kernel = (query@p["source_raw"].T-1)/TAU_L
        densities = [torch.logsumexp(kernel[:, index]+weights, dim=1)
                     for index, weights in zip(p["role_indices"], p["log_weights"])]
        output[start:start+query_chunk] = densities[0]-densities[1]
    return output


def _regional_scores(p, query_chunk, permutation):
    # Only small observed template banks form a dense matrix. No NxN attention
    # or persistent Query x Source patch kernel is constructed.
    g = (p["query_global"]@p["source_global"].T-1)/TAU_G
    old_g = (p["query_old_global"]@p["source_old_global"].T-1)/TAU_G
    shuffled_g = g[:, permutation.to(p["device"])]
    names = ("regional_local", "old_global_local", "main", "source_cls_shuffled", "global_only", "route_top1", "route_top4")
    output = {name:torch.empty(p["n"], device=p["device"], dtype=p["dtype"]) for name in names}
    routes = {}
    route_sizes = {}
    for v in range(4):
        for role_id, index in enumerate(p["role_indices"]):
            templates = p["source_template_ids"][v, index].unique(sorted=True)
            # IDs sorted first; stable descending sort breaks exact cosine ties
            # by smallest observed template ID, not arbitrary topk order.
            ranking = torch.argsort(g[:, templates], dim=1, descending=True, stable=True)
            for count in (1, 4):
                routes[v, role_id, count] = templates[ranking[:, :min(count, len(templates))]]
                route_sizes[v, role_id, count] = min(count, len(templates))
    for start in range(0, p["n"], query_chunk):
        stop = min(start+query_chunk, p["n"])
        width = stop-start
        accum = {name:[torch.full((width,), -torch.inf, dtype=p["dtype"], device=p["device"]) for _ in range(2)] for name in names}
        for u in range(3):
            qid = p["query_template_ids"][u, start:stop]
            query = p["query_local"][u, start:stop]
            for v in range(4):
                local = (query@p["source_local"][v].T-1)/TAU_L
                sid = p["source_template_ids"][v]
                global_actual = g[qid][:, sid]
                global_old = old_g[qid][:, sid]
                global_shuffle = shuffled_g[qid][:, sid]
                for role_id, (index, weights) in enumerate(zip(p["role_indices"], p["log_weights"])):
                    local_role = local[:, index]
                    fields = dict(regional_local=local_role,
                        old_global_local=local_role+global_old[:, index],
                        main=local_role+global_actual[:, index],
                        source_cls_shuffled=local_role+global_shuffle[:, index],
                        global_only=global_actual[:, index])
                    for name, field in fields.items():
                        term = torch.logsumexp(field+weights, dim=1)-math.log(12)
                        accum[name][role_id] = torch.logaddexp(accum[name][role_id], term)
                    for count in (1, 4):
                        selected_templates = routes[v, role_id, count][qid]
                        selected = (sid[index][None, :, None] == selected_templates[:, None, :]).any(dim=2)
                        selected_area = (selected*p["area"][index]).sum(dim=1)
                        if not bool((selected_area > 0).all()):
                            raise RuntimeError("Template route has no legal selected role area")
                        routed_weights = p["area"][index].log()[None, :]-selected_area.log()[:, None]
                        term = torch.logsumexp((local_role+routed_weights).masked_fill(~selected, -torch.inf), dim=1)-math.log(12)
                        name = "route_top"+str(count)
                        accum[name][role_id] = torch.logaddexp(accum[name][role_id], term)
        for name in names:
            output[name][start:stop] = accum[name][0]-accum[name][1]
    # Constant CLS is EXACTLY the same local ratio: its common kernel factor
    # cancels from both role densities. Do not introduce rounding-only changes.
    output["cls_constant"] = output["regional_local"].clone()
    source_constant = bool((p["source_global"] == p["source_global"][:1]).all())
    query_constant = bool((p["query_global"] == p["query_global"][:1]).all())
    if source_constant:
        output["global_only"].zero_()
        output["source_cls_shuffled"] = output["main"].clone()
    if source_constant and query_constant:
        output["main"] = output["regional_local"].clone()
        output["source_cls_shuffled"] = output["regional_local"].clone()
    return output, dict(route_selected_templates={"source_view%d_role%s_top%d"%(v, "FG" if role_id == 0 else "BG", count):size
        for (v, role_id, count), size in route_sizes.items()},
        constant_Source_CLS=source_constant, constant_Query_CLS=query_constant,
        template_bank_matrix_shape=[81,83], constant_CLS_exact_local_degeneracy=True)


@torch.no_grad()
def run_transfer_suite(inputs, query_masked_inputs=None, *, include_ssp=False,
                       query_chunk=128, shuffle_seed=6050):
    """All fixed five-group comparisons; outputs are N scalar maps, not masks GT.

    Query masked packet replaces only query_local/query_global. Source/raw/IDs/
    areas/old-global observations remain identical. Means average LOG-RATIOS
    with fixed .5/.5, not minmax-normalized fields or fitted weights.
    """
    if include_ssp:
        raise ValueError("SSP is held diagnostic, not in the frozen five-group core")
    if not isinstance(query_chunk, int) or not 1 <= query_chunk <= 128:
        raise ValueError("Query streaming chunk must be1..128")
    p, audit = _prepare(inputs)
    if p is None:
        return dict(state="SOURCE_BACKGROUND_EMPTY", scores={}, probabilities={}, masks={}, diagnostics={}, audit=audit)
    permutation = source_cls_permutation(shuffle_seed)
    scores, stream_audit = _regional_scores(p, query_chunk, permutation)
    scores["raw_local"] = _raw_score(p, query_chunk)
    # Strong same-observation controls: separate semantic/local evidence can
    # explain a gain without a correspondence-conditioned product kernel.
    scores["global_plus_local"] = scores["global_only"] + scores["regional_local"]
    scores["global_plus_route4"] = scores["global_only"] + scores["route_top4"]
    if query_masked_inputs is not None:
        replacement = dict(inputs)
        allowed = {"query_local", "query_global"}
        if not allowed <= set(query_masked_inputs):
            raise ValueError("G5 requires actual masked Query local and CLS observations")
        for key in allowed:
            replacement[key] = query_masked_inputs[key]
        masked, masked_audit = _prepare(replacement)
        if masked is None:
            raise RuntimeError("Query observation control changed Source role validity")
        alternate, alternate_audit = _regional_scores(masked, query_chunk, permutation)
        scores["masked_main"] = alternate["main"]
        scores["masked_local"] = alternate["regional_local"]
        scores["mean_main_views"] = .5*(scores["main"]+scores["masked_main"])
        scores["mean_local_views"] = .5*(scores["regional_local"]+scores["masked_local"])
        stream_audit["G5_Source_fixed"] = True
        stream_audit["G5_average"] = "fixed half natural-Query plus half masked-Query log-density-ratios"
        stream_audit["G5_masked_stream"] = alternate_audit
    if any(not bool(torch.isfinite(score).all()) for score in scores.values()):
        raise RuntimeError("Nonfinite log density ratio; no clipping/minmax fallback")
    diagnostics = dict(g=scores["global_only"], l=scores["regional_local"],
                       r=scores["route_top4"], h=scores["main"],
                       h_minus_l=scores["main"]-scores["regional_local"],
                       r_top1=scores["route_top1"])
    audit.update(stream_audit, tau_local=TAU_L, tau_global=TAU_G,
        query_points=p["n"], Source_anchors=p["a"], query_chunk=query_chunk,
        source_views=4, query_views=3, uniform_observation_pairs=12,
        role_priors="equal: actual anchor area normalized separately FG and BG",
        output_cut="strict logdensity_FG minus logdensity_BG >0", minmax=False,
        CLS_shuffle_seed=int(shuffle_seed), CLS_shuffle_permutation=permutation.tolist(),
        CLS_shuffle_groups="whole0fixed;coarse1..16;fine17..80;role81..82",
        CLS_shuffle_changed_rows=int((permutation != torch.arange(83)).sum()),
        Source_labels_and_areas_shuffled=False,
        diagnostics_definitions={"g":"global-only logdensity ratio", "l":"same-budget local-only logdensity ratio",
            "r":"top4 rolewise Source-template route/local-transfer logdensity ratio", "h":"CLSxlocal logdensity ratio",
            "h_minus_l":"intervention delta, NOT a decomposition; h generally differs from g+l",
            "r_top1":"top1 rolewise Source-template route/local-transfer logdensity ratio"},
        SAM6D_style_routing_not_full_paper_reproduction=True,
        sigmoid_not_calibrated_task_probability=True,
        encoder_calls_in_math_core=0, QueryGT_used=False,
        persistent_NxN_kernel=False, real_task_gain_verified=False,
        arithmetic_dtype=str(p["dtype"]))
    return dict(state="READY", scores=scores,
                probabilities={name:score.sigmoid() for name,score in scores.items()},
                masks={name:score > 0 for name,score in scores.items()},
                diagnostics=diagnostics, audit=audit)


build_five_group_scores = run_transfer_suite


def cpu_selfcheck():
    """Five tiny SERVER-only equation tests; not a model or quality experiment."""
    import sys
    from pathlib import Path
    if not sys.platform.startswith("linux") or not str(Path(__file__).resolve()).startswith("/root/"):
        raise RuntimeError("CPU numerical checks run only on the preparation SERVER")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU check may not initialize CUDA")
    torch.set_num_threads(1)
    dtype = torch.float64
    source_raw = torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,1.,0.]], dtype=dtype)
    query_raw = F.normalize(torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]], dtype=dtype), dim=-1)
    sg = torch.tensor([0.,0.,1.], dtype=dtype).expand(83,-1).clone()
    qg = torch.tensor([0.,0.,1.], dtype=dtype).expand(81,-1).clone()
    sg[[1,17,81]] = source_raw[0]
    sg[[2,18,82]] = source_raw[2]
    qg[[1,17]] = query_raw[0]
    qg[[2,18]] = query_raw[1]
    source_ids = torch.tensor([[0,0,0,0],[1,1,2,2],[17,17,18,18],[81,81,82,82]])
    query_ids = torch.tensor([[0,0,0],[1,2,1],[17,18,17]])
    inputs = dict(source_local=source_raw[None].expand(4,-1,-1).clone(),
        query_local=query_raw[None].expand(3,-1,-1).clone(), source_raw=source_raw, query_raw=query_raw,
        source_global=sg, query_global=qg, source_old_global=sg.clone(), query_old_global=qg.clone(),
        source_template_ids=source_ids, query_template_ids=query_ids,
        source_roles=torch.tensor([True,True,False,False]), source_area=torch.tensor([1.,3.,2.,6.],dtype=dtype))
    actual = run_transfer_suite(inputs, query_chunk=2,
        query_masked_inputs=dict(query_local=inputs["query_local"], query_global=qg))
    expected_raw = []
    for query in query_raw:
        kernel = ((query@source_raw.T)-1)/TAU_L
        fg = torch.logsumexp(kernel[:2]+torch.tensor([.25,.75],dtype=dtype).log(),dim=0)
        bg = torch.logsumexp(kernel[2:]+torch.tensor([.25,.75],dtype=dtype).log(),dim=0)
        expected_raw.append(fg-bg)
    assert torch.allclose(actual["scores"]["raw_local"],torch.stack(expected_raw),atol=2e-12,rtol=0)
    one = run_transfer_suite(inputs, query_chunk=1)
    assert all(torch.allclose(one["scores"][name],actual["scores"][name],atol=2e-12,rtol=0) for name in ARMS)
    checks=["literal_role_area_logdensity_and_stream_chunk_parity"]
    constant = dict(inputs, source_global=torch.ones_like(sg), query_global=torch.ones_like(qg))
    neutral = run_transfer_suite(constant)
    assert torch.equal(neutral["scores"]["main"],neutral["scores"]["regional_local"])
    assert torch.equal(actual["scores"]["cls_constant"],actual["scores"]["regional_local"])
    assert torch.equal(neutral["scores"]["global_only"],torch.zeros(3,dtype=dtype))
    checks.append("constant_CLS_exact_local_degeneracy_and_no_global_identity")
    scaled = dict(inputs, source_area=inputs["source_area"]*torch.tensor([11.,11.,37.,37.],dtype=dtype))
    scaling = run_transfer_suite(scaled)
    assert all(torch.allclose(scaling["scores"][name],one["scores"][name],atol=2e-12,rtol=0) for name in ARMS)
    assert torch.equal(actual["scores"]["main"],actual["scores"]["masked_main"])
    assert torch.equal(actual["scores"]["mean_main_views"],actual["scores"]["main"])
    permutation=source_cls_permutation()
    assert permutation[0]==0 and sorted(permutation[1:17].tolist())==list(range(1,17)) and sorted(permutation[17:81].tolist())==list(range(17,81)) and sorted(permutation[81:83].tolist())==[81,82]
    checks.append("independent_FG_BG_area_scale_invariance_G5_means_and_grouped_CLS_shuffle")
    no_bg = dict(inputs, source_roles=torch.ones(4,dtype=torch.bool))
    assert run_transfer_suite(no_bg)["state"]=="SOURCE_BACKGROUND_EMPTY"
    try:
        run_transfer_suite(dict(inputs,source_roles=torch.zeros(4,dtype=torch.bool)))
    except ValueError: pass
    else: raise AssertionError("FG empty was silently retained")
    bad = source_raw.clone();bad[0,0]=torch.nan
    try: run_transfer_suite(dict(inputs,source_raw=bad))
    except ValueError: pass
    else: raise AssertionError("Nonfinite observations silently repaired")
    checks.append("explicit_BG_empty_fallback_FG_empty_invalid_nonfinite_error")
    # A misleading region CLS can overturn PERFECT local feature evidence.
    misleading = dict(inputs, source_global=sg.clone(), query_global=qg.clone())
    misleading["source_global"][[0,1,17,81]]=torch.tensor([-1.,0.,0.],dtype=dtype)
    misleading["source_global"][[2,18,82]]=torch.tensor([1.,0.,0.],dtype=dtype)
    misleading["query_global"][:]=torch.tensor([1.,0.,0.],dtype=dtype)
    failure=run_transfer_suite(misleading)
    assert failure["scores"]["regional_local"][0]>0 and failure["scores"]["main"][0]<0
    checks.append("global_context_shortcut_counterexample_not_a_quality_guarantee")
    return dict(state="SERVER_REGIONAL_TRANSFER_FIVE_CPU_EQUATION_CHECKS_PASSED",checks=checks,
        CUDA_initialized=torch.cuda.is_initialized(), pretrained_model=False,
        QueryGT_used=False, full_host_or_real_observation_verified=False,
        method_quality_verified=False, maximum_query_chunk=128)


if __name__=="__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check",action="store_true",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError("Preserve old CPU evidence; use a fresh output")
    result=cpu_selfcheck();result["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(state=result["state"],checks=len(result["checks"]))))
