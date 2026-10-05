"""Fixed edit-region auxiliary baseline; no labels, models, or parameter search.

The helper is the unchanged transport_query_witness probability field. Its
thresholded use is exactly an agreement rule, not an independent new method.
Only evaluation calls the component/error-count functions below.
"""
from __future__ import annotations

import numpy as np

PROPOSALS = {
    "E": ("latent_native", "predict", "latent_native"),
    "Efixed": ("latent_native", "control", "latent_native.control"),
    "Etwo": ("latent_native", "two_slot_control", "latent_native.two_slot_em.control"),
    "Bkernel": ("structure_conditioned", "kernel_control", "structure_conditioned.kernel.control"),
    "delta": (None, None, "multilayer.delta.control"),
    "rcg": (None, None, "rcg"),
}
MODES = ("noaux", "aux", "native_score")
ACTIONS = ("add_TP", "add_FP", "delete_FP", "delete_TP")
BUCKETS = ("wrong_object", "boundary_leakage", "whole_object_missed", "extent_incomplete")


def recipes():
    """Fixed before outcomes: each source alone, and each adder with RCG deletion."""
    out = {}
    for alias in PROPOSALS:
        for mode in MODES:
            suffix = ".control" if mode == "native_score" else ""
            for kind in ("only_add", "only_delete"):
                out[f"edit.{alias}.{mode}.{kind}{suffix}"] = {
                    "A": alias if kind == "only_add" else None,
                    "B": alias if kind == "only_delete" else None, "mode": mode}
    pairs = [(a, a) for a in PROPOSALS] + [(a, "rcg") for a in PROPOSALS if a != "rcg"]
    for a, b in pairs:
        for mode in MODES:
            suffix = ".control" if mode == "native_score" else ""
            out[f"edit.{a}__{b}.{mode}.combined{suffix}"] = {"A": a, "B": b, "mode": mode}
    # Coverage alone can be bought by editing more area. These controls keep
    # each raw proposal's count but remove its localization information.
    for alias in PROPOSALS:
        for kind in ("only_add", "only_delete"):
            out[f"edit.{alias}.global_native_count.{kind}.control"] = {
                "A": alias if kind == "only_add" else None,
                "B": alias if kind == "only_delete" else None,
                "mode": "global_native_count"}
    return out


def resize_continuous(field, shape=(1024, 1024)):
    import torch
    import torch.nn.functional as F
    value = np.asarray(field, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite two-dimensional continuous field")
    return F.interpolate(torch.from_numpy(value.copy())[None, None], shape,
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def matched_count(region, score, count, *, high):
    """Exact accepted count; score order, then ascending flat pixel index."""
    indices = np.flatnonzero(region)
    if not 0 <= count <= len(indices):
        raise ValueError("Accepted count is outside proposal region")
    out = np.zeros_like(region, dtype=bool)
    if count:
        values = np.asarray(score).ravel()[indices]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite native scores")
        order = np.lexsort((indices, -values if high else values))
        out.ravel()[indices[order[:count]]] = True
    return out


def accepted_edits(native, proposals, helper_fg, native_score):
    """No GT: A=proposal outside N, B=proposal exclusion inside N."""
    edits = {}
    if native.shape != helper_fg.shape or native.shape != native_score.shape:
        raise ValueError("Native, helper and continuous native score shapes differ")
    if not np.isfinite(native_score).all():
        raise ValueError("Nonfinite native scores")
    # Sort each full domain once, retaining deterministic row-major ties.
    global_orders = {}
    for name, region, high in (("A", ~native, True), ("B", native, False)):
        indices = np.flatnonzero(region)
        values = native_score.ravel()[indices]
        global_orders[name] = indices[np.lexsort((indices, -values if high else values))]
    for alias, proposal in proposals.items():
        if proposal.shape != native.shape:
            raise ValueError("Proposal shape differs from native")
        a, b = proposal & ~native, native & ~proposal
        aa, bb = a & helper_fg, b & ~helper_fg
        for mode, addition, deletion in (
            ("noaux", a, b), ("aux", aa, bb),
            ("native_score", matched_count(a, native_score, int(aa.sum()), high=True),
             matched_count(b, native_score, int(bb.sum()), high=False)),
        ):
            edits[f"{alias}.{mode}.A"] = addition
            edits[f"{alias}.{mode}.B"] = deletion
        for name, count in (("A", int(a.sum())), ("B", int(b.sum()))):
            selected = np.zeros_like(native, dtype=bool)
            selected.ravel()[global_orders[name][:count]] = True
            edits[f"{alias}.global_native_count.{name}"] = selected
    return edits


def compose(native, edits, recipe):
    addition = edits[f"{recipe['A']}.{recipe['mode']}.A"] if recipe["A"] else False
    deletion = edits[f"{recipe['B']}.{recipe['mode']}.B"] if recipe["B"] else False
    return (native & ~np.asarray(deletion)) | addition


def action_counts(addition, deletion, truth):
    return dict(add_TP=int((addition & truth).sum()), add_FP=int((addition & ~truth).sum()),
                delete_FP=int((deletion & ~truth).sum()), delete_TP=int((deletion & truth).sum()))


def error_buckets(native, truth):
    """Opus diagnostic: 8-connected components with strictly <10% overlap."""
    from scipy import ndimage
    def low_components(mask, other):
        labels, n = ndimage.label(mask, np.ones((3, 3), bool))
        size = np.bincount(labels.ravel(), minlength=n + 1)
        hit = np.bincount(labels.ravel(), weights=other.ravel(), minlength=n + 1)
        low = hit / np.maximum(size, 1) < .1
        low[0] = False
        return low[labels]
    wrong = low_components(native, truth)
    missed = low_components(truth, native)
    return {
        "wrong_object": wrong & ~truth,
        "boundary_leakage": native & ~wrong & ~truth,
        "whole_object_missed": missed & ~native,
        "extent_incomplete": truth & ~missed & ~native,
    }


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def retention(raw, kept):
    """Rates are undefined, not zero, when the relevant proposal count is zero."""
    return {
        "addition_FP_removed_fraction": ratio(raw["add_FP"] - kept["add_FP"], raw["add_FP"]),
        "addition_TP_lost_fraction": ratio(raw["add_TP"] - kept["add_TP"], raw["add_TP"]),
        "deletion_TP_protected_fraction": ratio(raw["delete_TP"] - kept["delete_TP"], raw["delete_TP"]),
        "deletion_FP_removal_lost_fraction": ratio(raw["delete_FP"] - kept["delete_FP"], raw["delete_FP"]),
    }
