"""The model's own selection, and both public pipelines rebuilt on it one stage at a time.

Input per episode: the frozen encoder's final-layer tokens of the reference and the query as tapped (before any
projection), the reference mask and the query colours, the positional basis. No query label is read.

Rows (each a 64 x 64 field and the mask its own pipeline renders from it at 1024)
  model.raw_nn     every query token takes the label of its nearest reference token (cosine, raw tokens): the
                   edit origin. Its field is the nearest-foreground minus nearest-background similarity.
  model.raw_mean   similarity to the reference foreground mean minus similarity to the background mean, above 0.
  insid3.*         INSID3 (models/insid3.py) on the same tokens, without its CRF: the projected nearest-token
                   vote, its candidates, the seed cluster, the final aggregation.
  foris.*          FoRIS (models/foris.py, public defaults) without its CRF: the stage-2 response, after the vote
                   term, after the seed-cluster prior, after the penalty, the final response; and the final
                   response with one term removed at a time. `foris@never` repeats the chain with no projection.
The terms of both pipelines are also returned as separate fields, so that any re-weighting, any decision rule and
any mixture of the two is a point of one family (`family_fields`, `ANCHORS`).

Two properties of the FoRIS source are reproduced as they are: its stage-1 feature gate multiplies each query token
by a positive scalar which the later per-token normalisation removes, so the gate does not reach the output; its
vote normalises the reference tokens along the image height (dim=2 of a [1, C, h, w] tensor), not along channels.
The runner checks the rebuilt stages against the stage outputs cached from the public code.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

CONFIG = dict(origin="model.raw_nn", tau=.6, lse_temperature=.07, bg_weight=.55, vote_boost=.20, prior_boost=.25,
              disagreement=.08, coupling=.05, fg_boost=.20, conflict=.18, uncertainty_power=1.5, penalty_max=.22,
              neg_cap=.12, hard_bg=.2, debias_below=.8, merge_threshold=.2, color_weight=.35, position_weight=.20,
              log_floor=1e-3, query_gt_in_inference=False)
TERMS = ("fg", "bg", "vote", "prior", "penalty", "delta")
FAMILY = ("foris.fg", "foris.bg", "foris.vote", "foris.prior", "foris.penalty", "foris.delta",
          "insid3.lcross", "insid3.lintra", "insid3.larea", "model.raw_nn", "model.raw_mean")
# A point of the family: weights over FAMILY, how the response is scaled, how it is rendered, the threshold.
ANCHORS = {
    "foris.pre": dict(w={"foris.fg": 1., "foris.bg": -CONFIG["bg_weight"], "foris.vote": CONFIG["vote_boost"],
                         "foris.prior": CONFIG["prior_boost"], "foris.penalty": -1., "foris.delta": 1.},
                      scale="minmax", render="field", t=.5),
    "insid3.final": dict(w={"insid3.lcross": 1., "insid3.lintra": 1., "insid3.larea": 1.},
                         scale="abs", render="mask", t=math.log(CONFIG["merge_threshold"])),
    "model.raw_nn": dict(w={"model.raw_nn": 1.}, scale="abs", render="mask", t=0.),
    "model.raw_mean": dict(w={"model.raw_mean": 1.}, scale="abs", render="mask", t=0.),
}


def down_bilinear(mask):
    """utils.data.downsample_mask of both sources: bilinear > .5, else nearest, else the centre token. -> [4096]"""
    m = mask[None, None].float()
    d = F.interpolate(m, (64, 64), mode="bilinear", align_corners=False)[0, 0] > .5
    if d.sum() == 0:
        d = F.interpolate(m, (64, 64), mode="nearest")[0, 0] > .5
        if d.sum() == 0:
            cy, cx = (torch.argwhere(mask > 0).float().mean(0) / (mask.shape[-1] // 64)).int()
            d[cy, cx] = True
    return d.reshape(-1)


def down_nearest(mask):
    return (F.interpolate(mask[None, None].float(), (64, 64), mode="nearest")[0, 0] > .5).reshape(-1)


def cluster(x, tau):
    """utils.clustering.agglomerative_clustering: average linkage on 1 - cosine, no merge at or above 1 - tau."""
    from sklearn.cluster import AgglomerativeClustering
    if x.shape[0] <= 1:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    d = (1.0 - (x @ x.T).clamp(-1, 1)).cpu().numpy()
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=float(1.0 - tau)).fit_predict(d)
    return torch.from_numpy(labels).long().to(x.device)


def cluster_mean(values, labels, k):
    total = torch.zeros((k,) + values.shape[1:], device=values.device, dtype=values.dtype).index_add_(0, labels, values)
    count = torch.bincount(labels, minlength=k).clamp_min(1).to(values.dtype)
    return total / count.reshape((k,) + (1,) * (values.ndim - 1))


def prototypes(x, labels):
    return F.normalize(cluster_mean(x, labels, int(labels.max()) + 1), dim=1)


def debias(x, basis):
    """Both sources: project off the positional subspace, then renormalise each token."""
    return F.normalize(x - (x @ basis) @ basis.T, dim=1)


def unit(x):
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def model_rows(q, r, fg):
    """The model with no design added: nearest-token label transfer, and the two-mean margin."""
    sim = q @ r.T
    vote = fg[sim.argmax(1)]
    near_fg = sim[:, fg].max(1).values
    near_bg = sim[:, ~fg].max(1).values if bool((~fg).any()) else torch.full_like(near_fg, -1.)
    mean_bg = F.normalize(r[~fg].mean(0), dim=0) if bool((~fg).any()) else torch.zeros_like(r[0])
    mean = q @ F.normalize(r[fg].mean(0), dim=0) - q @ mean_bg
    return {"model.raw_nn": near_fg - near_bg, "model.raw_mean": mean}, {"model.raw_nn": vote, "model.raw_mean": mean > 0}


def insid3(q, r, fg, basis, tau):
    """models/insid3.py predict_mask without the CRF. -> fields, token masks"""
    c, n = CONFIG, q.shape[0]
    qd, rd = debias(q, basis), debias(r, basis)
    proto = F.normalize(rd[fg].mean(0), dim=0)
    forward_sim = qd @ proto
    forward = forward_sim > 0
    if not bool(forward.any()):
        forward = forward_sim > float(torch.quantile(forward_sim, .9))
    vote = fg[(qd @ rd.T).argmax(1)]
    cand = forward & vote
    zero, empty = torch.zeros(n, device=q.device, dtype=q.dtype), torch.zeros(n, dtype=torch.bool, device=q.device)
    masks = {"insid3.vote": vote, "insid3.candidates": cand, "insid3.seed": empty, "insid3.final": empty}
    floor = math.log(c["log_floor"])
    fields = {"insid3.combined": zero, "insid3.lcross": zero + floor, "insid3.lintra": zero + floor, "insid3.larea": zero + floor}
    if not bool(cand.any()):
        return fields, masks, dict(clusters=0, candidates=0)
    labels = cluster(q, tau); k = int(labels.max()) + 1          # clustered in the raw space
    protos_deb, protos_raw = prototypes(qd, labels), prototypes(q, labels)
    counts = torch.bincount(labels[cand], minlength=k).to(q.dtype)
    area = counts / torch.bincount(labels, minlength=k).to(q.dtype)
    seed = int((protos_deb @ proto).masked_fill(counts == 0, -math.inf).argmax())
    intra = protos_raw[seed] @ protos_raw.T                     # to the seed, raw space
    cross = cluster_mean(forward_sim, labels, k)                 # to the reference, projected space
    area[seed] = 1.0
    combined = cross * intra * area
    log = lambda v: torch.log(v.clamp_min(c["log_floor"]))[labels]
    fields = {"insid3.combined": combined[labels], "insid3.lcross": log(cross), "insid3.lintra": log(intra), "insid3.larea": log(area)}
    masks.update({"insid3.seed": labels == seed, "insid3.final": combined[labels] > c["merge_threshold"]})
    return fields, masks, dict(clusters=k, candidates=int(cand.sum()), seed_tokens=int((labels == seed).sum()))


def foris(q, r, fg, near, rgb, basis, tau, mode="source"):
    """models/foris.py predict up to the final response. mode: 'source' (its own rule), 'always' or 'never' project.
    -> the six additive terms [4096], info"""
    c = CONFIG
    if mode == "source":
        if bool(near.any()):
            mean = q.mean(0)
            semantic = float(torch.dot(F.normalize(r[near].mean(0), dim=0), mean) / (mean.norm() + 1e-6))
            apply = semantic < c["debias_below"]
        else:
            semantic, apply = None, True
    else:
        semantic, apply = None, mode == "always"
    if apply:
        q, r = debias(q, basis), debias(r, basis)
    # Part 2, stage 2: clustered foreground prototypes against the orthogonalised hard-negative background mean.
    f, b = r[fg], r[~fg]
    mu_fg = F.normalize(f.mean(0), dim=0)
    if b.shape[0]:
        hard = b[(b @ mu_fg).topk(max(1, int(c["hard_bg"] * b.shape[0]))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    f = F.normalize(f, dim=1)
    protos = prototypes(f, cluster(f, tau))
    orth = mu_bg - (mu_bg * mu_fg).sum() * mu_fg
    mu_bg = orth / orth.norm().clamp_min(1e-8)
    t = max(1e-4, c["lse_temperature"])
    sim_fg = t * torch.logsumexp((q @ protos.T) / t, dim=1)
    sim_bg = q @ mu_bg
    sf, sbn = unit(sim_fg), unit(sim_bg)
    # Part 3: nearest-token vote (reference tokens normalised along height, as in the source), seed-cluster prior.
    rh = r.reshape(64, 64, -1)
    rh = (rh / rh.norm(dim=0, keepdim=True).clamp_min(1e-12)).reshape(4096, -1)
    vote = fg[(q @ rh.T).argmax(1)]
    grid = torch.zeros(64, 64, dtype=torch.bool, device=q.device); grid[::2, ::2] = True
    cand = vote | grid.reshape(-1)
    line = torch.linspace(-1., 1., 64, device=q.device, dtype=q.dtype)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    position = F.normalize(torch.stack([yy, xx], -1).reshape(-1, 2), dim=1)
    joint = F.normalize(torch.cat([q, F.normalize(rgb, dim=1) * c["color_weight"], position * c["position_weight"]], 1), dim=1)
    labels = cluster(joint, tau); k = int(labels.max()) + 1
    if k <= 1:
        prior, seed = cand.to(q.dtype), 0
    else:
        protos_q = prototypes(q, labels)
        counts = torch.bincount(labels[cand], minlength=k).to(q.dtype)
        area = counts / torch.bincount(labels, minlength=k).to(q.dtype).clamp_min(1)
        cross = cluster_mean(q @ mu_fg, labels, k)
        seed = int((cross * area).masked_fill(counts == 0, -math.inf).argmax())
        combined = cross * (protos_q[seed] @ protos_q.T).clamp_min(0) * area
        combined[seed] = combined.max().clamp_min(1e-6)
        lo, hi = combined.min(), combined.max()
        prior = ((combined - lo) / (hi - lo).clamp_min(1e-6))[labels]
    votef = vote.to(q.dtype)
    # Part 4: disagreement penalty, then a per-cluster boost or suppression (clusters of the query tokens alone).
    penalty = c["disagreement"] * ((sf - votef).abs() + (sf - prior).abs()) + c["coupling"] * torch.minimum(sf, sbn)
    penalty = (penalty * (1.0 - 2.0 * (sf - .5).abs()).clamp(0, 1) ** c["uncertainty_power"]).clamp_max(c["penalty_max"])
    part4 = cluster(q, tau); k4 = int(part4.max()) + 1
    fg_mean, bg_mean = cluster_mean(sf, part4, k4), cluster_mean(sbn, part4, k4)
    delta = (c["fg_boost"] * (fg_mean - bg_mean).clamp_min(0) - c["conflict"] * torch.minimum(fg_mean, bg_mean)).clamp_min(-c["neg_cap"])[part4]
    terms = dict(fg=sim_fg, bg=sim_bg, vote=votef, prior=prior, penalty=penalty, delta=delta)
    return terms, dict(projected=bool(apply), semantic=semantic, prior_clusters=k, part4_clusters=k4,
                       reference_prototypes=int(protos.shape[0]), prior_seed_tokens=int((labels == seed).sum()))


def response(terms, drop=None, upto=None):
    """The FoRIS response from its terms: all of them, without one, or only the stages up to `upto`."""
    c = CONFIG
    parts = dict(fg=terms["fg"], bg=-c["bg_weight"] * terms["bg"], vote=c["vote_boost"] * (terms["vote"] - .5),
                 prior=c["prior_boost"] * (terms["prior"] - .5), penalty=-terms["penalty"], delta=terms["delta"])
    keep = TERMS if upto is None else TERMS[:TERMS.index(upto) + 1]
    return sum(parts[k] for k in keep if k != drop)


def render_field(field, t=.5):
    """FoRIS: bilinear to 1024, then the threshold. field [64, 64] or [4096]."""
    x = torch.as_tensor(field, dtype=torch.float32).reshape(1, 1, 64, 64)
    return F.interpolate(x, (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > t


def render_mask(mask):
    """INSID3 without CRF: the token mask, bilinear to 1024, above one half."""
    return render_field(torch.as_tensor(mask).float(), .5)


@torch.inference_mode()
def episode(q, r, ref_mask, rgb, basis, *, device="cpu", tau=None, never=True):
    """q, r [4096, 1024] as tapped; ref_mask bool [1024, 1024]; rgb [4096, 3] in [0, 1]; basis [1024, 500].
    -> fields {name: float32 [64, 64]}, masks {name: bool [1024, 1024]}, info"""
    tau = CONFIG["tau"] if tau is None else tau
    q = F.normalize(torch.as_tensor(q, device=device).float(), dim=1)
    r = F.normalize(torch.as_tensor(r, device=device).float(), dim=1)
    basis = torch.as_tensor(basis, device=device).float()
    ref_mask = torch.as_tensor(ref_mask, device=device).bool()
    rgb = torch.as_tensor(rgb, device=device).float()
    if q.shape != (4096, 1024) or r.shape != q.shape or ref_mask.shape != (1024, 1024) or rgb.shape != (4096, 3):
        raise ValueError("Expected 64 x 64 tokens of width 1024, a 1024 x 1024 reference mask and 4096 colours")
    if not bool(ref_mask.any()):
        raise ValueError("Empty reference mask")
    fg, near = down_bilinear(ref_mask), down_nearest(ref_mask)
    fields, tokens, masks, info = {}, {}, {}, {"reference_tokens": int(fg.sum())}
    f, m = model_rows(q, r, fg); fields.update(f); tokens.update(m)
    f, m, info["insid3"] = insid3(q, r, fg, basis, tau); fields.update(f); tokens.update(m)
    for mode in ("source", "never") if never else ("source",):
        terms, receipt = foris(q, r, fg, near, rgb, basis, tau, mode)
        tag = "foris" if mode == "source" else "foris@never"
        info[tag] = receipt
        if mode == "source":
            fields.update({f"foris.{k}": v for k, v in terms.items()})
        stages = {f"{tag}.s2": response(terms, upto="bg"), f"{tag}.s3_vote": response(terms, upto="vote"),
                  f"{tag}.s3": response(terms, upto="prior"), f"{tag}.s4_penalty": response(terms, upto="penalty"),
                  f"{tag}.pre": response(terms)}
        if mode == "source":
            stages.update({f"foris.pre-{k}": response(terms, drop=k) for k in TERMS[1:]})
        for name, score in stages.items():
            fields[name] = score
            masks[name] = render_field(unit(score))
    masks.update({name: render_mask(mask) for name, mask in tokens.items()})
    out = {}
    for name, value in fields.items():
        value = value.float().reshape(64, 64).cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite field: {name}")
        out[name] = value
    return out, {k: v.cpu().numpy() for k, v in masks.items()}, info


def family_fields(fields):
    """[len(FAMILY), 4096] float32: the terms every point of the family weighs."""
    return np.stack([np.asarray(fields[name], np.float32).reshape(4096) for name in FAMILY])


def self_check():
    """Synthetic tokens only: the rebuilt chain runs, and each anchor reproduces its pipeline's own mask."""
    g = torch.Generator().manual_seed(0)
    centres = F.normalize(torch.randn(6, 1024, generator=g), dim=1)
    layout = (torch.arange(64)[:, None] // 22 + 3 * (torch.arange(64)[None, :] // 32)).reshape(-1)
    def image(shift):
        idx = torch.roll(layout.reshape(64, 64), shift, 1).reshape(-1)
        return centres[idx] + .35 * torch.randn(4096, 1024, generator=g) / 32, idx
    (q, _), (r, ridx) = image(9), image(0)
    ref_mask = F.interpolate((ridx == 1).reshape(1, 1, 64, 64).float(), (1024, 1024), mode="nearest")[0, 0] > .5
    basis = torch.linalg.qr(torch.randn(1024, 500, generator=g)).Q
    rgb = torch.rand(4096, 3, generator=g)
    fields, masks, info = episode(q, r, ref_mask, rgb, basis)
    fam = torch.from_numpy(family_fields(fields))
    for name, a in ANCHORS.items():
        score = sum(w * fam[FAMILY.index(k)] for k, w in a["w"].items())
        score = unit(score) if a["scale"] == "minmax" else score
        got = render_field(score, a["t"]) if a["render"] == "field" else render_mask(score > a["t"])
        assert np.array_equal(got.numpy(), masks[name]), name
    assert set(ANCHORS) <= set(masks) and all(v.shape == (1024, 1024) for v in masks.values())
    return f"stage_bank self-check passed: {len(fields)} fields, {len(masks)} masks, FoRIS projected={info['foris']['projected']}"


if __name__ == "__main__":
    print(self_check())
