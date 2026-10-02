"""Explicit adapters for independently CPU-validated research branches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from baselines import shape_plan

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/projection_merge"))
sys.path.insert(0, str(ROOT / "research/compute_structure"))
from projection_merge import MergeCache, factor_merge_predict, cached_merge_predict
from prototype import ResearchCache, DensePhaseCache, predict, dense_phase_predict
from merged_phase import MergedPhaseCache, merged_phase_predict

METHODS = ("factor_merged", "cached_merged", "dense_assoc_merged", "factor_phase",
           "dense_phase", "cached_phase", "factor_implicit", "factor_implicit_phase",
           "cached_merged_phase", "factor_merged_phase")


def plans_for(method, model, n, t, order, write):
    if method.startswith("factor"):
        return None
    dense = "cached" if method.startswith("cached") else "dense_assoc"
    return shape_plan(model, n, t, dense, order, write)


@dataclass
class CacheFactory:
    method: str
    plans: list | None

    def build(self, model, image, dense, position):
        if self.method in ("cached_merged_phase", "factor_merged_phase"):
            return MergedPhaseCache.build(model, image, dense, position,
                method="factor" if self.method.startswith("factor") else "cached", plans=self.plans)
        if self.method.endswith("merged"):
            kind = "factor" if self.method == "factor_merged" else "cached" if self.method == "cached_merged" else "dense_assoc"
            return MergeCache.build(model, image, dense, position, method=kind, plans=self.plans)
        if self.method.startswith("factor"):
            return ResearchCache.build(model, image, dense, position, include_statistics=False,
                                       include_phase="phase" in self.method)
        return DensePhaseCache.build(model, image, dense, position)


def forward(method, model, cache, sparse, attention, plans):
    if method in ("cached_merged_phase", "factor_merged_phase"):
        return merged_phase_predict(model, cache, sparse, attention)
    if method == "factor_merged":
        return factor_merge_predict(model, cache, sparse, attention)
    if method.endswith("merged"):
        return cached_merge_predict(model, cache, sparse, attention)
    if method.startswith("factor"):
        return predict(model, cache, sparse, attention, implicit_attention="implicit" in method,
                       statistic_ln=False, phase_layout="phase" in method)
    return dense_phase_predict(model, cache, sparse, attention,
        method="cached" if method == "cached_phase" else "dense_assoc", plans=plans)


def coverage(method, attention):
    if "implicit" in method:
        return "Implicit factor read/write use explicit contractions after first layer; SDPA applies only where actually called"
    if method.startswith("factor"):
        return "Projected factor writes are explicit; eligible reads/self-attention use " + attention
    return "Dense baseline eligible routes use " + attention + "; associated sparse writes remain explicit"
