"""Two frozen single-reference inference cards; no encoder, fitting or file writes.

m02: source-background concepts must furnish a complete PUBLIC FoRIS explanation.
m09: each source-foreground mode faces its own soft source-background opponents.

These are compatibility scores, NOT posteriors, confidence calibration, or
guarantees about background absence. Constants are fixed before query scoring.
Only legal reference labels enter these functions. The supplied context owns
public source replay, query RGB, positional-debias policy and original CRF.
"""
import math
import torch
import torch.nn.functional as F


RECIPE = dict(version="identity_competition_v1", modes=6, kmeans_steps=12,
              coordinate_scale=.2, max_background_concepts=3,
              cosine_temperature=.1, competition_gain=1.,
              pure_foreground=.9, pure_background=.1, query_gt_used=False)


def _unit(x):
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Finite NxD feature matrix required")
    return F.normalize(x, dim=1)


def _pair(value):
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 2:
        raise ValueError("Actual one-reference [S,Q] Part1 tensor required")
    return tuple(_unit(v.flatten(1).T) for v in value)


def _coverage(mask, hw):
    if mask.ndim != 2:
        raise ValueError("Reference/work mask must be HW")
    return F.interpolate(mask[None, None].float(), size=hw,
                         mode="area")[0, 0]


def _components(mask):
    """Deterministic 4-connected labels, CPU metadata only, no numeric clustering."""
    rows = mask.detach().bool().cpu().tolist()
    h, w = len(rows), len(rows[0])
    result, seen = [], set()
    for y in range(h):
        for x in range(w):
            if not rows[y][x] or (y, x) in seen:
                continue
            stack, group = [(y, x)], []
            seen.add((y, x))
            while stack:
                yy, xx = stack.pop()
                group.append(yy*w+xx)
                for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    ny, nx = yy+dy, xx+dx
                    if (0 <= ny < h and 0 <= nx < w and rows[ny][nx]
                            and (ny, nx) not in seen):
                        seen.add((ny, nx)); stack.append((ny, nx))
            result.append(torch.tensor(sorted(group), device=mask.device,
                                       dtype=torch.long))
    return result


def _partition(x, max_modes=6):
    """Fixed deterministic FPS-seeded spherical kmeans; source-only positions."""
    x = _unit(x)
    if len(x) == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    seeds = [0]
    best = (1 - x @ x[0]).clamp_min(0)
    for _ in range(1, min(max_modes, len(x))):
        index = int(best.argmax())
        if float(best[index]) <= 1e-7:
            break
        seeds.append(index)
        best = torch.minimum(best, (1-x @ x[index]).clamp_min(0))
    centers = x[seeds].clone()
    labels = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for _ in range(RECIPE["kmeans_steps"]):
        labels = (x @ centers.T).argmax(1)
        centers = torch.stack([_unit(x[labels == k].mean(0, keepdim=True))[0]
                              if (labels == k).any() else centers[k]
                              for k in range(len(centers))])
    # Remove empty categories, without random replacement or a prototype-count bonus.
    _, labels = labels.unique(sorted=True, return_inverse=True)
    return labels


def _prototypes(x, labels=None):
    labels = _partition(x, RECIPE["modes"]) if labels is None else labels
    if labels.numel() == 0:
        return x[:0]
    return torch.stack([_unit(x[labels == k].mean(0, keepdim=True))[0]
                        for k in range(int(labels.max())+1)])


def _signed_margin(query, foreground, complement, chunk=256):
    """Same bounded cosine definition for each alternative, independent of minmax."""
    if len(foreground) == 0 or len(complement) == 0:
        return None
    return torch.cat([(block @ foreground.T).max(1).values
                      - (block @ complement.T).max(1).values
                      for block in query.split(chunk)])


def _background_concepts(ctx):
    # Source Part1's apply/debias policy also observes query-global features.
    # Keep concept assignment genuinely source-only in native RAW coordinates;
    # each alternative compatibility is later read in its own actual Part1.
    ref, _ = _pair(ctx.features["final_raw"])
    cov = ctx.reference_coverage
    h, w = cov.shape
    fg, bg = cov.flatten() >= .9, cov.flatten() <= .1
    if not fg.any() or not bg.any():
        return [], dict(state="NO_PURE_REFERENCE_ROLE", concepts=0)
    ids = torch.nonzero(bg).flatten()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=ref.device),
                            torch.linspace(-1, 1, w, device=ref.device), indexing="ij")
    xy = torch.stack((yy.flatten(), xx.flatten()), 1)
    clustered = torch.cat((ref[ids], .2*xy[ids]), 1)
    labels = _partition(clustered)
    fg_mean = _unit(ref[fg].mean(0, keepdim=True))[0]
    proposals = []
    for label in range(int(labels.max())+1):
        mask = torch.zeros(h*w, dtype=torch.bool, device=ref.device)
        mask[ids[labels == label]] = True
        for component in _components(mask.view(h, w)):
            descriptor = _unit(ref[component].mean(0, keepdim=True))[0]
            hardness = float(descriptor @ fg_mean)
            proposals.append((hardness, int(component[0]), component))
    proposals.sort(key=lambda item: (-item[0], item[1]))
    masks, rows = [], []
    for hardness, _, component in proposals[:3]:
        grid = torch.zeros(h*w, dtype=torch.bool, device=ref.device)
        grid[component] = True
        work = F.interpolate(grid.view(1, 1, h, w).float(),
                             size=ctx.reference_mask.shape, mode="nearest")[0, 0].bool()
        # A background hypothesis may NEVER relabel legal reference foreground.
        work = work & ~ctx.reference_mask.bool()
        if work.any():
            masks.append(work)
            rows.append(dict(grid_tokens=len(component), work_pixels=int(work.sum()),
                             source_hardness=hardness,
                             intersects_reference_foreground=False))
    return masks, dict(state="SOURCE_ONLY_SPATIAL_CONCEPTS", concepts=len(masks),
                       rows=rows, source_assignment_coordinates="native_raw_before_Part1",
                       physical_semantic_concepts_guaranteed=False)


def _complete_background_map(query, foreground, complement, full_mask, hw):
    """Read each complete prediction region again as a unit descriptor.

    The region cosine/complement margin can exceed local point margins. Thus
    full evidence can strengthen exclusion as well as reject a point confuser.
    Nonpositive regions are absent explanations, even if minmax produced a mask.
    """
    if tuple(full_mask.shape) == tuple(hw):
        grid_mask = full_mask.bool()
    else:
        grid_mask = _coverage(full_mask, hw) > .5
    result = torch.zeros(hw[0]*hw[1], device=query.device, dtype=query.dtype)
    present = []
    if not len(foreground) or not len(complement):
        return result.view(hw), present
    for ids in _components(grid_mask):
        descriptor = _unit(query[ids].mean(0, keepdim=True))
        margin = _signed_margin(descriptor, foreground, complement)[0]
        if float(margin) > 0:
            result[ids] = margin
            present.append(dict(tokens=len(ids), signed_region_margin=float(margin)))
    return result.view(hw), present


def _alternative_packet(ctx):
    # An episode-local SMALL map cache avoids repeating identical PUBLIC decodes
    # for main/naive arms. No encoder features or on-disk assets are cached here.
    key = "_ten_identity_competition_alt_v1"
    existing = getattr(ctx, key, None)
    if existing is not None:
        return existing
    masks, selection = _background_concepts(ctx)
    hw = tuple(ctx.reference_coverage.shape)
    points, fulls, rows = [], [], []
    for mask in masks:
        alternative = ctx.replay(ref_mask=mask)
        if "part1" not in alternative:
            raise ValueError("Alternative PUBLIC replay must expose its ACTUAL Part1")
        reference, query = _pair(alternative["part1"])
        cov = _coverage(mask, hw).flatten()
        positive, negative = reference[cov >= .9], reference[cov <= .1]
        point = _signed_margin(query, positive, negative)
        if point is None:
            rows.append(dict(state="EMPTY_ALTERNATIVE_ROLE", full_explanation=False))
            continue
        complete, present = _complete_background_map(query, positive, negative,
                                                       alternative["mask"], hw)
        points.append(point.clamp_min(0).view(hw))
        fulls.append(complete)
        rows.append(dict(state="PUBLIC_COMPLETE_ALTERNATIVE", full_explanation=bool(present),
                         regions=present, own_Part1_policy=True,
                         independent_minmax_confidence_used=False))
    device, dtype = ctx.ref_tokens.device, ctx.ref_tokens.dtype
    zero = torch.zeros(hw, device=device, dtype=dtype)
    packet = dict(point=torch.stack(points).max(0).values if points else zero,
                  full=torch.stack(fulls).max(0).values if fulls else zero,
                  audit=dict(selection=selection, alternatives=rows,
                             absent_background_allowed=True,
                             minmax_response_is_not_common_confidence=True))
    setattr(ctx, key, packet)
    return packet


def _attach(result, audit):
    return {**result, "audit": {**result.get("audit", {}), **audit}}


def _run02(ctx, point_only=False, refund_only=False):
    packet = _alternative_packet(ctx)
    correction = (packet["point"] if refund_only else -packet["point"] if point_only
                  else packet["point"]-packet["full"])
    def replace(original_result, **kwargs):
        if original_result is None:
            return None
        score = original_result[0]
        if score.shape != correction.shape:
            raise ValueError("Source early-score grid changed")
        return score + correction.to(score)  # fixed gain=1 in signed cosine units
    result = ctx.replay(part2_override=replace)
    name = ("m02_refund_only" if refund_only else "m02_point_naive" if point_only
            else "m02_complete_competition")
    return _attach(result, dict(method=name,
        recipe=RECIPE.copy(), **packet["audit"], complete_FG_replay=True,
        correction_positive=int((correction>0).sum()), correction_negative=int((correction<0).sum()),
        correction_abs_max=float(correction.abs().max()),
        posterior_claimed=False, exact_refund_of_native_HN_penalty_claimed=False,
        native_sf_sbn_and_stage1_retained=True, no_query_gt=True))


def run02(ctx):
    return _run02(ctx)


def run02_naive(ctx):
    return _run02(ctx, point_only=True)


def run02_refund_only(ctx):
    """A strong control against attributing a mere positive score shift to identity."""
    return _run02(ctx, refund_only=True)


def _mode_scores(query, foreground_modes, background_modes, policy="conditional"):
    """Normalized LSE: duplicates do not create arbitrary log(number of modes)."""
    t = RECIPE["cosine_temperature"]
    a, b = query @ foreground_modes.T, query @ background_modes.T
    f = t*(torch.logsumexp(a/t, 1)-math.log(len(foreground_modes)))
    if policy == "conditional":
        log_opponents = torch.log_softmax((foreground_modes @ background_modes.T)/t, 1)
        opponent = t*torch.logsumexp(b[:, None, :]/t+log_opponents[None, :, :], 2)
        evidence = a-opponent
        r = t*(torch.logsumexp(evidence/t, 1)-math.log(len(foreground_modes)))
        # f-r is a bounded soft effective competitor, NOT a posterior.
        back = f-r
    elif policy == "symmetric":
        back = t*(torch.logsumexp(b/t, 1)-math.log(len(background_modes)))
        r = f-back
    elif policy == "mean_bg_modes":
        bg_mean = _unit(background_modes.mean(0, keepdim=True))[0]
        back = query @ bg_mean
        r = f-back
    else:
        raise ValueError("Unknown mode comparison policy")
    return r, (f+1)/2, (back+1)/2


def _run09(ctx, policy):
    # Source role banks are constructed in the same ACTUAL FG Part1 frame.
    reference, query = _pair(ctx.features["final"])
    raw_reference, _ = _pair(ctx.features["final_raw"])
    coverage = ctx.reference_coverage.flatten()
    fg, bg = reference[coverage >= .9], reference[coverage <= .1]
    if len(fg) == 0 or len(bg) == 0:
        return _attach(ctx.replay(), dict(method="m09", state="NO_PURE_ROLE_NATIVE_FALLBACK",
                                         no_query_gt=True, recipe=RECIPE.copy()))
    # Freeze grouping on S alone, then read means in the FG hypothesis's actual
    # Part1 frame. No query can change the source class/mode assignments.
    positive = _prototypes(fg, _partition(raw_reference[coverage >= .9]))
    negative = _prototypes(bg, _partition(raw_reference[coverage <= .1]))
    r, sf, sbn = _mode_scores(query, positive, negative, policy)
    hw = tuple(ctx.reference_coverage.shape)
    def replace(original_result, **kwargs):
        if original_result is None:
            return None
        score, old_sf, old_sbn, mu_fg, denoised = original_result
        return (r.view(hw).to(score), sf.view(hw).to(old_sf), sbn.view(hw).to(old_sbn),
                mu_fg, denoised)
    result = ctx.replay(part2_override=replace)
    return _attach(result, dict(method="m09", policy=policy, recipe=RECIPE.copy(),
        foreground_modes=len(positive), background_modes=len(negative),
        score_range=[float(r.min()),float(r.max())], sf_range=[float(sf.min()),float(sf.max())],
        sbn_range=[float(sbn.min()),float(sbn.max())],
        compatibility_not_calibrated_probability=True, native_stage1_retained=True,
        source_mode_assignment_coordinates="native_raw_before_Part1",
        prototype_readout_coordinates="FG_hypothesis_actual_native_Part1",
        sf_sbn_updated_together=True, full_PUBLIC_downstream_replayed=True, no_query_gt=True))


def run09(ctx):
    return _run09(ctx, "conditional")


def run09_naive(ctx):
    """Strong cheap control: same source mode bank, symmetric density ratio."""
    return _run09(ctx, "symmetric")


def run09_mean_background_modes(ctx):
    """Unconditional centroid of the SAME BG mode bank, not source top20% HN."""
    return _run09(ctx, "mean_bg_modes")


METHODS = [
    dict(id="m02", function=run02, naive=run02_naive,
         controls={"refund_only": run02_refund_only}, card=[
        "Assumption: some point-background matches fail a complete alternative identity explanation.",
        "Numeric: no independent minmax confidence; each BG subset replays its own actual Part1/public stages; signed correction may have both signs. No pp forecast.",
        "Match: main must beat full FoRIS, point penalty AND refund-only control; otherwise an apparent gain may be mere positive score shifting, not complete-competitor value or exact HN refund.",
        "Mismatch: coherent confusers can still mimic targets; nonpositive/absent concepts are empty; adverse task result stops this fixed recipe without gain/threshold sweeps."]),
    dict(id="m09", function=run09, naive=run09_naive,
         controls={"mean_bg_modes": run09_mean_background_modes}, card=[
        "Assumption: a foreground mode's nearest source-background modes are more relevant opponents than an unconditional bank.",
        "Numeric: temperature .1, up to six FPS-kmeans modes per label, normalized LSE and bounded compatibility maps; no base-class or query-label fit and no pp forecast.",
        "Match: main must beat same-bank symmetric density and complete FoRIS; small opponent weights alone are not novel evidence.",
        "Mismatch: correlated FG/BG modes can suppress true parts; no advantage over symmetric density rejects foreground-conditioned opponent assignment in this fixed recipe."])
]


def cpu_checks():
    """Run ONLY in the existing server CPU environment; no image/model/CUDA."""
    torch.set_num_threads(1)
    e1, e2 = torch.tensor([1.,0.]), torch.tensor([0.,1.])
    q = torch.stack((e1,e2,(e1+e2).div(math.sqrt(2))))
    p, b = e1[None], e2[None]
    r, sf, sbn = _mode_scores(q,p,b)
    assert torch.allclose(r, q[:,0]-q[:,1], atol=1e-6)
    assert (sf>=0).all() and (sf<=1).all() and (sbn>=0).all() and (sbn<=1).all()
    symmetric = _mode_scores(q,p,b,"symmetric")[0]
    swapped = _mode_scores(q,b,p,"symmetric")[0]
    assert torch.equal(symmetric,-swapped)
    duplicate = _mode_scores(q,p.repeat(3,1),b.repeat(4,1),"symmetric")[0]
    assert torch.allclose(duplicate,symmetric,atol=1e-6)
    margin = _signed_margin(q,p,b)
    assert torch.equal(margin,q[:,0]-q[:,1])
    empty, _ = _complete_background_map(q,p,b,torch.zeros(1,3,dtype=torch.bool),(1,3))
    assert torch.equal(empty,torch.zeros_like(empty))
    # Complete region can outweigh a weak local match, yielding negative correction.
    q2 = _unit(torch.tensor([[1.,0.],[.8,.6],[.72,.69]]))
    point = _signed_margin(q2,p,b).clamp_min(0)
    full, regions = _complete_background_map(q2,p,b,torch.ones(1,3,dtype=torch.bool),(1,3))
    delta = point-full.flatten()
    assert (delta>0).any() and (delta<0).any() and len(regions)==1
    absent, rows = _complete_background_map(torch.stack((e2,e2)),p,b,
                                          torch.ones(1,2,dtype=torch.bool),(1,2))
    assert torch.count_nonzero(absent)==0 and not rows
    x = _unit(torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]))
    assert torch.equal(_partition(x),_partition(x))
    groups = _components(torch.tensor([[1,0,1],[1,0,1]],dtype=torch.bool))
    assert [v.tolist() for v in groups]==[[0,3],[2,5]]
    return dict(state="CPU_TENSOR_CHECKS_PASSED",checks=9,cuda_initialized=torch.cuda.is_initialized(),
                full_public_source_tested=False, task_gain_tested=False, recipe=RECIPE)


if __name__ == "__main__":
    import json
    print(json.dumps(cpu_checks(),sort_keys=True),flush=True)
