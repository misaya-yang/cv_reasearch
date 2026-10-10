"""Shared legal inputs and rendering. No query labels or host scores are inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from ics.methods.direct_dino_features import resize, sample_grid, unit


@dataclass(frozen=True)
class Episode:
    q: np.ndarray
    r: np.ndarray
    wf: np.ndarray
    wvalid: np.ndarray
    q_hw: tuple[int, int]
    r_hw: tuple[int, int]
    q_valid: np.ndarray
    query_geometry: dict = field(default_factory=dict)
    reference_geometry: dict = field(default_factory=dict)
    original_shape: tuple[int, int] = (1024, 1024)
    q_rgb: np.ndarray | None = None
    r_rgb: np.ndarray | None = None
    reference_mask: np.ndarray | None = None
    producer: dict = field(default_factory=dict)
    source_id: str = "synthetic"

    @property
    def wb(self):
        return self.wvalid - self.wf


@dataclass
class Result:
    margin: np.ndarray
    info: dict = field(default_factory=dict)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grid_coverage(hw, geometry):
    if not geometry:
        return np.ones(np.prod(hw))
    view = int(geometry["view_side"])
    sh, sw = geometry["resized_hw"]
    oy, ox = geometry["padding_top_left"]
    if min(sh, sw) <= 0 or min(oy, ox) < 0 or oy + sh > view or ox + sw > view:
        raise ValueError("Invalid physical geometry")
    y0 = np.arange(hw[0]) * view / hw[0]
    x0 = np.arange(hw[1]) * view / hw[1]
    cy = np.maximum(0, np.minimum(y0 + view / hw[0], oy + sh) - np.maximum(y0, oy)) / (view / hw[0])
    cx = np.maximum(0, np.minimum(x0 + view / hw[1], ox + sw) - np.maximum(x0, ox)) / (view / hw[1])
    return (cy[:, None] * cx[None, :]).ravel()


def validate(ep):
    for shape in (ep.q_hw, ep.r_hw, ep.original_shape):
        if len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
            raise ValueError("Positive integer H/W required")
    if (ep.q.ndim != 2 or ep.r.ndim != 2 or ep.q.shape[1] != ep.r.shape[1]
            or len(ep.q) != np.prod(ep.q_hw) or len(ep.r) != np.prod(ep.r_hw)
            or ep.wf.shape != (len(ep.r),) or ep.wvalid.shape != ep.wf.shape
            or ep.q_valid.shape != (len(ep.q),)):
        raise ValueError("Aligned native feature grids and reference weights required")
    for a in (ep.q, ep.r, ep.wf, ep.wvalid, ep.q_valid):
        if not np.isfinite(a).all():
            raise ValueError("Non-finite legal input")
    if (np.any(ep.wf < 0) or np.any(ep.wvalid < ep.wf) or np.any(ep.wvalid > 1)
            or np.any(ep.q_valid < 0) or np.any(ep.q_valid > 1) or ep.wvalid.sum() <= 0):
        raise ValueError("Invalid complete-reference or padding weights")
    if ep.query_geometry and not np.allclose(grid_coverage(ep.q_hw, ep.query_geometry), ep.q_valid, atol=1e-12):
        raise ValueError("Query validity and physical image transform disagree")
    for x in (ep.q, ep.r):
        norm = np.linalg.norm(x, axis=1)
        if not np.all((norm < 1e-12) | (np.abs(norm - 1) < 1e-6)):
            raise ValueError("Episode features must already be unit-normalized")
    return ep


def load_episode(row):
    path = Path(row["feature_pack"])
    if row.get("sha256") and sha(path) != row["sha256"]:
        raise ValueError("Input hash changed")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"q", "r", "foreground_weight", "valid_weight",
                            "query_geometry_json", "producer_json"}:
            raise ValueError("Unexpected inputs: sealed direct-DINO pack required")
        q, r = z["q"].copy(), z["r"].copy()
        if q.dtype not in (np.dtype("float32"), np.dtype("float64")) or r.dtype != q.dtype:
            raise ValueError("Native FP32 or its explicit FP64 unit export required; no FP16")
        wf, valid = z["foreground_weight"].copy(), z["valid_weight"].copy()
        geometry, producer = json.loads(z["query_geometry_json"].item()), json.loads(z["producer_json"].item())
    if producer.get("FoRIS_Part1_applied") is not False or producer.get("kind") not in (
            "frozen_DINOv3_FP32_native_final_LN_patches_then_unit", "frozen_DINOv3_FP32_native_final_LN_patches"):
        raise ValueError("Direct native DINO provenance required, do not substitute processed cache")
    side = producer.get("model_input_side")
    if side not in (128, 1024) or q.shape != ((side // 16) ** 2, 1024) or r.shape != q.shape:
        raise ValueError("Native DINOv3-L dimensions and resolution disagree with producer")
    checkpoint = producer.get("model_assets", producer).get("checkpoint_sha256", "")
    if len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise ValueError("Known frozen DINO checkpoint hash required")
    q, r = unit(q), unit(r)
    image_hashes = producer.get("source_image_hashes")
    if image_hashes and len(image_hashes) != 2:
        raise ValueError("Reference/query source hashes must have a fixed order")
    for key, expected in (("r_rgb", image_hashes[0] if image_hashes else None),
                          ("q_rgb", image_hashes[1] if image_hashes else geometry.get("RGB_source_sha256")),
                          ("reference_mask", producer.get("reference_mask_sha256"))):
        if key in row and expected and row.get(key + "_sha256") != expected:
            raise ValueError("RGB/reference mask differs from native feature producer")
    qside, rside = int(round(len(q) ** .5)), int(round(len(r) ** .5))
    ep = Episode(q=q, r=r, wf=wf, wvalid=valid, q_hw=(qside, qside), r_hw=(rside, rside),
                 q_valid=grid_coverage((qside, qside), geometry), query_geometry=geometry,
                 reference_geometry=row.get("reference_geometry", {}), original_shape=tuple(row["original_shape"]),
                 q_rgb=load_image(row, "q_rgb"), r_rgb=load_image(row, "r_rgb"),
                 reference_mask=load_image(row, "reference_mask", binary=True),
                 producer=producer, source_id=row["id"])
    if ep.q_rgb is not None and ep.q_rgb.shape[:2] != ep.original_shape:
        raise ValueError("Original query RGB geometry differs from feature source")
    if ep.reference_mask is not None and (ep.r_rgb is None or ep.reference_mask.shape != ep.r_rgb.shape[:2]):
        raise ValueError("Complete original reference mask must align with original reference RGB")
    if ep.reference_geometry:
        if not np.allclose(grid_coverage(ep.r_hw, ep.reference_geometry), ep.wvalid, atol=1e-12):
            raise ValueError("Reference validity and physical image transform disagree")
        if ep.r_rgb is not None and list(ep.r_rgb.shape[:2]) != ep.reference_geometry.get("original_hw"):
            raise ValueError("Original reference RGB differs from recorded geometry")
    for a in (ep.q, ep.r, ep.wf, ep.wvalid, ep.q_valid):
        a.setflags(write=False)
    for a in (ep.q_rgb, ep.r_rgb, ep.reference_mask):
        if a is not None:
            a.setflags(write=False)
    return validate(ep)


def load_image(row, key, binary=False):
    if key not in row:
        return None
    from PIL import Image
    path = Path(row[key])
    if not row.get(key + "_sha256") or sha(path) != row[key + "_sha256"]:
        raise ValueError("Original image/mask hash binding required")
    with Image.open(path) as image:
        a = np.asarray(image if binary else image.convert("RGB")).copy()
    if binary:
        if a.ndim != 2 or not np.isin(a, (0, 1, 255)).all():
            raise ValueError("Complete binary reference mask required")
        return a != 0
    return a


def rgb_view(ep, role):
    """Original RGB -> recorded native encoder view; padding is known, not data."""
    from PIL import Image
    array = ep.q_rgb if role == "q" else ep.r_rgb
    g = ep.query_geometry if role == "q" else ep.reference_geometry
    if array is None or not g:
        raise ValueError("Original RGB and its physical transform unavailable")
    sh, sw = g["resized_hw"]
    oy, ox = g["padding_top_left"]
    out = np.broadcast_to(np.array((124, 116, 104), np.uint8), (g["view_side"], g["view_side"], 3)).copy()
    out[oy:oy + sh, ox:ox + sw] = np.asarray(Image.fromarray(array).resize((sw, sh), Image.Resampling.BILINEAR))
    return out


def prototype_margin(ep):
    """Shared inexpensive control; not counted as a new mechanism."""
    if ep.wf.sum() == 0:
        return np.full(ep.q_hw, -1.)
    if ep.wb.sum() == 0:
        return np.full(ep.q_hw, 1.)
    fg = unit(np.sum(ep.r * ep.wf[:, None], axis=0))
    bg = unit(np.sum(ep.r * ep.wb[:, None], axis=0))
    return (ep.q @ (fg - bg)).reshape(ep.q_hw)


def pairwise(a, b, block=256):
    """Cosine matrix; callers must account for O(len(a)*len(b)) storage."""
    a, b = unit(a), unit(b)
    out = np.empty((len(a), len(b)), float)
    for start in range(0, len(a), block):
        out[start:start + block] = a[start:start + block] @ b.T
    return out


def neighbors(hw, diagonal=False):
    ids = np.arange(np.prod(hw)).reshape(hw)
    a, b = [ids[:-1].ravel(), ids[:, :-1].ravel()], [ids[1:].ravel(), ids[:, 1:].ravel()]
    if diagonal:
        a += [ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()]
        b += [ids[1:, 1:].ravel(), ids[1:, :-1].ravel()]
    return np.concatenate(a), np.concatenate(b)


def render(ep, result):
    """Native signed margin -> physical 64-grid -> 1024 -> original binary mask."""
    margin = np.asarray(result.margin, float).reshape(ep.q_hw)
    if not np.isfinite(margin).all():
        raise ValueError("Method must return finite complete signed margin")
    g = ep.query_geometry
    if g:
        view = g["view_side"]
        sh, sw = g["resized_hw"]
        oy, ox = g["padding_top_left"]
        y = (oy + (np.arange(64) + .5) * sh / 64) * ep.q_hw[0] / view - .5
        x = (ox + (np.arange(64) + .5) * sw / 64) * ep.q_hw[1] / view - .5
        xx, yy = np.meshgrid(x, y)
        field = sample_grid(margin, yy, xx)
    else:
        field = resize(margin, (64, 64))
    field = field.astype(np.float32)
    work = resize(field, (1024, 1024)) > 0
    original = resize(work.astype(np.float32), ep.original_shape) > .5
    return dict(margin=field, work=work, original=original)
