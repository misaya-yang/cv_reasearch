"""Minimal cache-only lab harness: episodes, features, masks, FoRIS outputs, metrics.

Everything is read from existing caches; no encoder is initialised here.
CLI-1024 frame: predictions are 1024x1024, GT is nearest-resized to 1024x1024
(same as official inference.py with resize_to_orig_size=False).
"""
from __future__ import annotations
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "cv_data"
BINDINGS = ASSETS / "a/consensus_reference_calibration6380_v1_20261011/source_bindings.json"
BASIS = ASSETS / "native_assets/positional_basis.pt"
DEV = torch.device("mps")
DOMAINS = ("coco", "lvis", "pascal_part", "paco_part", "suim")


@lru_cache(1)
def bindings():
    rows = json.loads(BINDINGS.read_text())
    keep = []
    for r in rows:
        d, c = r["dataset"], r["cohort"]
        if d in ("coco", "lvis", "paco_part") or (d in ("pascal_part", "suim") and c == "native2000"):
            keep.append(r)
    return keep


def subset(per_domain=200, domains=DOMAINS, offset=0):
    """Deterministic stride sample per domain (no labels or scores consulted)."""
    out = []
    for d in domains:
        rows = [r for r in bindings() if r["dataset"] == d]
        if per_domain is None or per_domain >= len(rows):
            out += rows
            continue
        idx = (np.arange(per_domain) * len(rows) / per_domain + offset).astype(int) % len(rows)
        out += [rows[i] for i in idx]
    return out


@lru_cache(1)
def basis():
    return torch.load(BASIS, map_location="cpu")["basis"].float().to(DEV)


def feats(binding, role):
    """Raw O/24 tokens, (4096, 1024) float32 on device. role in {reference, query}."""
    key = "reference_raw" if role == "reference" else "query_raw_identity"
    return torch.from_numpy(np.load(binding["fit"][key]["payload_path"])["O/24"]).to(DEV)


def mask1024(path):
    """Binary mask nearest-resized to 1024x1024 (official set_reference / GT alignment)."""
    m = torch.from_numpy(np.array(Image.open(path).convert("L")) > 0)
    return F.interpolate(m[None, None].float(), size=(1024, 1024), mode="nearest")[0, 0] > 0.5


def coverage64(m1024):
    """Exact soft coverage of each 16x16 patch, (64, 64) in [0, 1]."""
    return F.avg_pool2d(m1024[None, None].float(), 16)[0, 0]


def foris_cli(binding):
    z = np.load(binding["old_arms"]["B"]["path"])
    return torch.from_numpy(np.unpackbits(z[binding["old_arms"]["B"]["cli_key"]])[: 1024 * 1024].reshape(1024, 1024).astype(bool))


def arm_cli(binding, arm):
    z = np.load(binding["old_arms"][arm]["path"])
    return torch.from_numpy(np.unpackbits(z[binding["old_arms"][arm]["cli_key"]])[: 1024 * 1024].reshape(1024, 1024).astype(bool))


def unit(x):
    return F.normalize(x, dim=-1)


def debias(x):
    """Project unit features off the positional subspace and renormalise."""
    b = basis()
    x = unit(x)
    return unit(x - (x @ b) @ b.T)


def up(field64, size=1024):
    return F.interpolate(field64[None, None].float(), size=(size, size), mode="bilinear", align_corners=False)[0, 0]


def iu(pred, gt):
    pred, gt = pred.bool(), gt.bool()
    return int((pred & gt).sum()), int((pred | gt).sum())


class Meter:
    """Class-pooled mIoU per domain (official) and per-episode mean IoU."""

    def __init__(self):
        self.I = defaultdict(lambda: defaultdict(float))
        self.U = defaultdict(lambda: defaultdict(float))
        self.eps = defaultdict(list)

    def add(self, binding, i, u):
        d, c = binding["dataset"], (binding["row"]["fold"], binding["row"]["global_class_id"])
        self.I[d][c] += i
        self.U[d][c] += u
        self.eps[d].append(100.0 * i / max(u, 1))

    def miou(self, d):
        return float(np.mean([100.0 * self.I[d][c] / max(self.U[d][c], 1.0) for c in self.I[d]]))

    def mean(self, d):
        return float(np.mean(self.eps[d]))

    def row(self, domains=DOMAINS):
        return {d: (round(self.miou(d), 2), round(self.mean(d), 2)) for d in domains if d in self.eps}


def best_threshold_iou(score, gt, n=None):
    """Exact GT-optimal single threshold on a score field (diagnostic only). Returns (I, U, t)."""
    s, g = score.flatten(), gt.flatten()
    order = torch.argsort(s, descending=True)
    inter = torch.cumsum(g[order].float(), 0)
    union = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32) + g.sum().float() - inter
    k = int((inter / union).argmax())
    return int(inter[k]), int(union[k]), float(s[order[k]])


def auroc(score, gt, stride=7):
    s, g = score.flatten()[::stride], gt.flatten()[::stride]
    if g.all() or not g.any():
        return float("nan")
    r = torch.argsort(torch.argsort(s)).float()
    n1, n0 = g.sum().float(), (~g).sum().float()
    return float((r[g].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def free():
    if DEV.type == "mps":
        torch.mps.empty_cache()


def kmeans(x, K, iters=8, w=None):
    """Weighted spherical k-means over a fixed-shape token set (w = 0 excludes a token).

    Fixed shapes keep the MPS graph cache bounded. Deterministic farthest-point initialisation.
    Returns (K, d) unit centroids; centroids that never receive mass repeat an existing one.
    """
    if w is None:
        w = torch.ones(x.shape[0], device=x.device)
    live = w > 0
    neg = torch.full_like(w, -1e9)
    d = torch.where(live, 1 - x @ unit((x * w[:, None]).sum(0)), neg)
    c = [x[d.argmax()]]
    d = torch.where(live, 1 - x @ c[0], neg)
    for _ in range(K - 1):
        c.append(x[d.argmax()])
        d = torch.minimum(d, torch.where(live, 1 - x @ c[-1], neg))
    c = torch.stack(c)
    xw = x * w[:, None]
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        s = torch.zeros_like(c).index_add_(0, a, xw)
        c = torch.where((s.norm(dim=1) > 1e-8)[:, None], unit(s), c)
    return c


def minmax(s):
    return (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)


EXT = {"isaid": "extended_foris_seed0_20261010", "isic": "extended_foris_seed0_20261010",
       "fundus": "fundus_foris_seed0_20261010", "lung": "chest_foris_seed0_20261010"}
EXT_DOMAINS = ("isaid", "isic", "lung", "fundus", "deepglobe_road")
ALL_DOMAINS = DOMAINS + EXT_DOMAINS


@lru_cache(1)
def ext_bindings():
    """Extended benchmarks in the same binding shape (cached O24 + complete FoRIS output)."""
    out = []
    for run in sorted(set(EXT.values())):
        base = ASSETS / "a" / run
        for i, row in enumerate(json.loads((base / "manifest.json").read_text())):
            rec, pred = base / "records" / f"{i:06d}.json", base / "predictions" / f"{i:06d}.npz"
            if not (rec.exists() and pred.exists()):
                continue
            raw = {}
            for e in json.loads(rec.read_text())["raw"]:
                entry = json.loads(Path(e["entry_path"]).read_text())
                raw[e["role"]] = {"payload_path": str(Path(e["entry_path"]).parent / entry["file"])}
            if "reference" not in raw or "query" not in raw:
                continue
            arm = {"path": str(pred), "cli_key": "cli/foris.crf"}
            out.append({"dataset": row["dataset"], "index": f"{run[:4]}{i:05d}", "row": row, "cohort": "ext",
                        "fit": {"reference_raw": raw["reference"], "query_raw_identity": raw["query"]},
                        "old_arms": {"B": arm},
                        "score64": {"path": str(base / "fields" / f"{i:06d}.npz"), "key": "score"}})
    for r in json.loads(BINDINGS.read_text()):
        if r["dataset"] == "deepglobe_road":
            out.append(r)
    return out


def ext_subset(per_domain=None, domains=EXT_DOMAINS):
    out = []
    for d in domains:
        rows = [r for r in ext_bindings() if r["dataset"] == d]
        if per_domain is not None and per_domain < len(rows):
            idx = (np.arange(per_domain) * len(rows) / per_domain).astype(int)
            rows = [rows[i] for i in idx]
        out += rows
    return out


def ignore1024(binding):
    """Official ignore region (iSAID only), nearest-resized to 1024, or None."""
    p = binding["row"].get("query_ignore_mask_path")
    return mask1024(p) if p else None
