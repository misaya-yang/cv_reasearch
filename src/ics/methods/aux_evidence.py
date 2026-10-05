"""Label-free evidence library for edits of a host mask: the rows of the sweep.

Every map is one byte per query token (higher = more likely the target). The maps are enumerated, not chosen:
family x token space x parameter. Each is graded both as an auxiliary inside another proposal's edit set and as a
proposal of its own on the whole image. Nothing here reads query truth.

  R  reference matching       nearest / top-10 / prototype / 4-part margins, one-sided scores, label transfer by
                              1, 5, 20 neighbours, cycle consistency                         (cross-image spaces)
  Q  same-image anchors       k-NN margin to confirmed target against confirmed background; anchors: the host mask
                              or its confident core; k in 1, 5, 20; pairs closer than 0, 2, 4 tokens excluded;
                              prototype margin; one-sided scores                              (every space)
  G  grouping                 k-means segments (8, 16, 32, 64): host cover and mean host score of the rest of the
                              segment; diffusion of the host mask over the 10-NN graph (1, 3, 10 steps)
  H  host and geometry        host score and its stage scores, the cached reference maxima, local score contrast,
                              signed distance to the host mask
  V  other proposals          how many proposals take the token; each proposal's own field
  N  placebo                  seeded noise at three scales: what a sweep this wide finds in nothing

Spaces: `last` (cached debiased last layer); with saved layers, each layer raw (`l16`) and each consecutive
difference (`d16_24` = unit(unit(l24) - unit(l16))). A space serves cross-image matching only when its positional
basis is given and projected out. Fixed constants, none fitted.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn.functional as F

CONFIG = {
    "name": "auxiliary_evidence_library", "levels": 256, "source_foreground": 0.5,
    "reference": dict(top=10, parts=4, votes=[1, 5, 20], cycle_scale=4.0),
    "anchors": dict(k=[1, 5, 20], radius=[0, 2, 4], kinds=["host", "core"]),
    "grouping": dict(segments=[8, 16, 32, 64], iterations=20, xy_weight=0.5, graph_k=10, walk_steps=[1, 3, 10]),
    "host": dict(contrast_window=5, distance_scales=[1.0, 4.0]),
    "placebo": dict(per_scale=8, scales=[1, 4, 16]),
    "margin_to_unit": "0.5 + margin, clipped to [0, 1]", "cosine_to_unit": "(cosine + 1) / 2",
    "missing": "a map that cannot be computed in an episode is the constant 128 there",
    "controls": ["H.score", "H.ref_margin", "F.*"],
    "query_gt_in_inference": False, "fitted_parameters": 0,
}
PACKET_FIELDS = ("s2", "s3", "fg_max", "bg_max", "fwd_sim")  # label-free; truth, native and pre are never opened here
_GRID = {}


def is_control(name):
    return name in ("H.score", "H.ref_margin") or name.startswith("F.")


def is_placebo(name):
    return name.startswith("N.")


def token_cover(mask):
    m = np.asarray(mask, dtype=bool)
    if m.shape != (1024, 1024):
        raise ValueError("Expected a 1024x1024 mask")
    return m.reshape(64, 16, 64, 16).mean((1, 3)).astype(np.float32)


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-6)


def _grid(device):
    """Token coordinates and the Chebyshev distance between every pair of tokens."""
    if device not in _GRID:
        yy, xx = torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing="ij")
        yy, xx = yy.reshape(-1), xx.reshape(-1)
        _GRID[device] = (yy, xx, torch.maximum((yy[:, None] - yy[None]).abs(), (xx[:, None] - xx[None]).abs()))
    return _GRID[device]


def _kmeans(x, k, iterations):
    """Lloyd from a farthest-point start: deterministic, no random generator."""
    k = min(k, x.shape[0]); centre, dist = [0], torch.full((x.shape[0],), float("inf"), device=x.device)
    for _ in range(k - 1):
        dist = torch.minimum(dist, (x - x[centre[-1]]).square().sum(1)); centre.append(int(dist.argmax()))
    c = x[centre].clone()
    for _ in range(iterations):
        label = torch.cdist(x, c).argmin(1); count = torch.bincount(label, minlength=k).float()
        new = torch.zeros_like(c).index_add_(0, label, x) / count.clamp_min(1)[:, None]
        c = torch.where(count[:, None] > 0, new, c)
    return torch.cdist(x, c).argmin(1), c


def spaces(q, r, layers=None, bases=None, device="cpu"):
    """name -> unit query tokens, unit reference tokens, and whether cross-image matching is meaningful there."""
    t = lambda x: F.normalize(torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x), device=device).float(), dim=1)
    bases = bases or {}
    def debias(x, name):
        b = bases.get(name)
        if b is None: return x, False
        b = torch.as_tensor(b, device=device).float()
        if b.shape[0] != x.shape[1]: raise ValueError(f"{name}: the positional basis belongs to another representation")
        return F.normalize(x - (x @ b) @ b.T, dim=1), True
    out = {"last": dict(q=t(q), r=t(r), cross=True)}
    if layers is not None:
        ids = sorted(int(k[1:]) for k in layers if k[0] == "q")
        raw = {i: (t(layers[f"q{i}"]), t(layers[f"r{i}"])) for i in ids}
        for i in ids:
            (a, da), (b, db) = debias(raw[i][0], f"l{i}"), debias(raw[i][1], f"l{i}")
            out[f"l{i}"] = dict(q=a, r=b, cross=da and db)
        for lo, hi in zip(ids, ids[1:]):
            name = f"d{lo}_{hi}"; key = name if name in bases else ("delta" if (lo, hi) == (16, 24) and "delta" in bases else name)
            dq, dr = F.normalize(raw[hi][0] - raw[lo][0], dim=1), F.normalize(raw[hi][1] - raw[lo][1], dim=1)
            if key != name and key in bases: bases = dict(bases, **{name: bases[key]})
            (a, da), (b, db) = debias(dq, name), debias(dr, name)
            out[name] = dict(q=a, r=b, cross=da and db)
    for name, s in out.items():
        if s["q"].shape[0] != 4096 or s["r"].shape[0] != 4096 or not (torch.isfinite(s["q"]).all() and torch.isfinite(s["r"]).all()):
            raise ValueError(f"{name}: expected finite 4096-token maps")
    return out


class _Maps:
    """Collects named maps in [0, 1]; a map that could not be computed stays None and becomes the neutral level."""
    def __init__(self): self.items, self.missing = {}, []
    def put(self, name, value):
        if name in self.items: raise ValueError(f"Duplicate map {name}")
        self.items[name] = value
        if value is None: self.missing.append(name)
    margin = staticmethod(lambda m: (.5 + m).clamp(0, 1))
    cosine = staticmethod(lambda c: ((c + 1) / 2).clamp(0, 1))


def _reference(out, s, name, cov):
    cfg = CONFIG["reference"]; fg = cov >= CONFIG["source_foreground"]
    names = ["nn_margin", "nn_fg", "nn_bg", "top_margin", "proto_margin", "parts_margin", "cycle"] + [f"vote{k}" for k in cfg["votes"]]
    if not fg.any() or fg.all():
        for n in names: out.put(f"R.{n}@{name}", None)
        return
    q, r = s["q"], s["r"]; sim = q @ r.T; sf, sb = sim[:, fg], sim[:, ~fg]
    nf, nb = sf.max(1).values, sb.max(1).values
    out.put(f"R.nn_margin@{name}", out.margin(nf - nb)); out.put(f"R.nn_fg@{name}", out.cosine(nf)); out.put(f"R.nn_bg@{name}", out.cosine(-nb))
    k = min(cfg["top"], sf.shape[1], sb.shape[1])
    out.put(f"R.top_margin@{name}", out.margin(sf.topk(k, 1).values.mean(1) - sb.topk(k, 1).values.mean(1)))
    out.put(f"R.proto_margin@{name}", out.margin(q @ F.normalize(r[fg].mean(0), dim=0) - q @ F.normalize(r[~fg].mean(0), dim=0)))
    cf, cb = (F.normalize(_kmeans(r[m], cfg["parts"], 10)[1], dim=1) for m in (fg, ~fg))
    out.put(f"R.parts_margin@{name}", out.margin((q @ cf.T).max(1).values - (q @ cb.T).max(1).values))
    yy, xx, _ = _grid(q.device); j = sim.argmax(1); back = sim.argmax(0)[j]
    d = torch.maximum((yy - yy[back]).abs(), (xx - xx[back]).abs()).float()
    out.put(f"R.cycle@{name}", (torch.exp(-d / cfg["cycle_scale"]) * cov[j]).clamp(0, 1))
    order = sim.topk(max(cfg["votes"]), 1).indices
    for k in cfg["votes"]: out.put(f"R.vote{k}@{name}", cov[order[:, :k]].mean(1).clamp(0, 1))


def _anchors(out, s, name, cover, score):
    cfg = CONFIG["anchors"]; q = s["q"]; sim = q @ q.T; _, _, cheb = _grid(q.device); kmax = max(cfg["k"])
    inside, outside = cover >= .5, cover == 0
    kinds = {"host": (inside, outside)}
    full = cover >= 1
    if full.any() and outside.any():
        kinds["core"] = (full & (score >= score[full].median()), outside & (score <= score[outside].median()))
    for kind in cfg["kinds"]:
        fg, bg = kinds.get(kind, (None, None)); ok = fg is not None and bool(fg.any()) and bool(bg.any())
        if ok:
            out.put(f"Q.{kind}.proto@{name}", out.margin(q @ F.normalize(q[fg].mean(0), dim=0) - q @ F.normalize(q[bg].mean(0), dim=0)))
        else:
            out.put(f"Q.{kind}.proto@{name}", None)
        for rad in cfg["radius"]:
            if not ok:
                for k in cfg["k"]: out.put(f"Q.{kind}.knn{k}.r{rad}@{name}", None)
                if rad == 2: out.put(f"Q.{kind}.fg.r2@{name}", None); out.put(f"Q.{kind}.bg.r2@{name}", None)
                continue
            far = sim.masked_fill(cheb <= rad, -2.0)  # neighbours share content through the receptive field
            tf, tb = (far[:, m].topk(min(kmax, int(m.sum())), 1).values for m in (fg, bg))
            def mean(t, k):
                t = t[:, :k]; valid = t > -1.5
                return (t * valid).sum(1) / valid.sum(1).clamp_min(1), valid.any(1)
            for k in cfg["k"]:
                (mf, vf), (mb, vb) = mean(tf, k), mean(tb, k)
                out.put(f"Q.{kind}.knn{k}.r{rad}@{name}", out.margin(torch.where(vf & vb, mf - mb, torch.zeros_like(mf))))
                if rad == 2 and k == 5:
                    out.put(f"Q.{kind}.fg.r2@{name}", out.cosine(torch.where(vf, mf, torch.zeros_like(mf))))
                    out.put(f"Q.{kind}.bg.r2@{name}", out.cosine(torch.where(vb, -mb, torch.zeros_like(mb))))
            del far
    return sim


def _grouping(out, s, name, cover, score, sim):
    cfg = CONFIG["grouping"]; q = s["q"]; yy, xx, _ = _grid(q.device)
    x = torch.cat([q, cfg["xy_weight"] * torch.stack([yy, xx], 1).float() / 63], 1)
    for k in cfg["segments"]:
        label, _ = _kmeans(x, k, cfg["iterations"]); count = torch.bincount(label, minlength=k).float()
        for tag, v in (("host", cover), ("score", score)):
            total = torch.zeros(k, device=q.device).index_add_(0, label, v)
            rest = torch.where(count[label] > 1, (total[label] - v) / (count[label] - 1).clamp_min(1), torch.full_like(v, .5))
            out.put(f"G.seg{k}.{tag}@{name}", rest.clamp(0, 1))
    near = sim.clone(); near.fill_diagonal_(-2.0); idx = near.topk(cfg["graph_k"], 1).indices; del near
    h = cover.clone()
    for step in range(1, max(cfg["walk_steps"]) + 1):
        h = h[idx].mean(1)
        if step in cfg["walk_steps"]: out.put(f"G.walk{step}@{name}", h.clamp(0, 1))


def _host(out, cover, score, packet, device):
    cfg = CONFIG["host"]; t = lambda a: torch.as_tensor(np.asarray(a, np.float32).reshape(-1), device=device)
    out.put("H.score", score)
    for k in ("s2", "s3"):
        out.put(f"H.{k}", t(minmax(packet[k])) if k in packet and np.asarray(packet[k]).size == 4096 else None)
    have = all(k in packet for k in ("fg_max", "bg_max"))
    out.put("H.ref_fg", out.cosine(t(packet["fg_max"])) if have else None)
    out.put("H.ref_bg", out.cosine(-t(packet["bg_max"])) if have else None)
    out.put("H.ref_margin", out.margin(t(packet["fg_max"]) - t(packet["bg_max"])) if have else None)
    out.put("H.fwd_sim", out.cosine(t(packet["fwd_sim"])) if "fwd_sim" in packet else None)
    w = cfg["contrast_window"]; grid = score.reshape(1, 1, 64, 64)
    out.put("H.contrast", out.margin((grid - F.avg_pool2d(grid, w, 1, w // 2, count_include_pad=False)).reshape(-1)))
    yy, xx, _ = _grid(device); pts = torch.stack([yy, xx], 1).float(); inside = cover >= .5
    if inside.any() and (~inside).any():
        signed = torch.cdist(pts, pts[inside]).min(1).values - torch.cdist(pts, pts[~inside]).min(1).values
        for scale in cfg["distance_scales"]: out.put(f"H.near{scale:g}", torch.sigmoid(-signed / scale))
    else:
        for scale in cfg["distance_scales"]: out.put(f"H.near{scale:g}", None)


def _placebo(out, key, device):
    cfg = CONFIG["placebo"]; rng = np.random.RandomState(zlib.crc32(key.encode()) & 0x7fffffff)
    for scale in cfg["scales"]:
        for i in range(cfg["per_scale"]):
            n = 64 // scale; a = torch.from_numpy(rng.rand(1, 1, n, n).astype(np.float32))
            if scale > 1: a = F.interpolate(a, (64, 64), mode="bilinear", align_corners=False)
            out.put(f"N.s{scale}.{i}", torch.as_tensor(minmax(a.numpy().reshape(-1)), device=device))


@torch.inference_mode()
def library(key, q, r, cov, score, host, proposers, *, packet=None, layers=None, bases=None, device="cpu"):
    """All maps of one episode.

    host: 1024x1024 bool. proposers: name -> (1024x1024 bool mask, optional 64x64 continuous field).
    packet: label-free cached fields (PACKET_FIELDS). Returns (names, uint8 [M, 4096], info)."""
    cov, score = np.asarray(cov, np.float32), np.asarray(score, np.float32)
    if cov.shape != (64, 64) or score.shape != (64, 64) or not (np.isfinite(cov).all() and np.isfinite(score).all()):
        raise ValueError("Reference coverage and host score must be finite 64x64 maps")
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32).reshape(-1), device=device)
    cover, sc, cv, out = t(token_cover(host)), t(minmax(score)), t(cov), _Maps()
    _host(out, cover, sc, packet or {}, device)
    sp = spaces(q, r, layers, bases, device)
    for name, s in sp.items():
        if s["cross"]: _reference(out, s, name, cv)
        sim = _anchors(out, s, name, cover, sc)
        _grouping(out, s, name, cover, sc, sim)
        del sim
    covers = [t(token_cover(mask)) for mask, _ in proposers.values()]
    out.put("V.count", torch.stack(covers).mean(0) if covers else None)
    for p, (_, field) in proposers.items():
        if field is not None and (np.asarray(field).shape != (64, 64) or not np.isfinite(field).all()):
            raise ValueError(f"{p}: the proposal field must be a finite 64x64 map")
        out.put(f"F.{p}", t(minmax(field)) if field is not None else None)
    _placebo(out, key, device)
    names = sorted(out.items); maps = np.full((len(names), 4096), 128, np.uint8)  # sorted: the order must not depend on which maps were missing
    for i, n in enumerate(names):
        v = out.items[n]
        if v is None: continue
        v = v.float().reshape(-1)
        if v.shape[0] != 4096 or not torch.isfinite(v).all(): raise ValueError(f"{n}: evidence must be a finite 4096-token map")
        maps[i] = (v.clamp(0, 1) * 256).clamp(max=255).to(torch.uint8).cpu().numpy()
    return names, maps, dict(spaces={k: bool(v["cross"]) for k, v in sp.items()}, missing=out.missing)
