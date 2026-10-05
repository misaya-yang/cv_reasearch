"""Exact frozen class-I/U paired bootstrap, with shared resampling cached.

Statistical contract: scripts/sam3_stitch_gpu_v2.py::paired (SHA276a53...).
Only arithmetic execution is vectorized: same photo-connected groups/order,
PCG64/default_rng(0) integer draws, repeated group multiplicities, per-class
summed I/U, sampled absent classes omitted, and np.percentile(2.5,97.5).
This module imports no model, tensor runtime or dataset/annotation loader.
"""
from __future__ import annotations

from collections import OrderedDict
import operator

# Plans contain no image, mask, label score, I/U or prediction value. The key
# binds precisely the classes/Source+Query photo identities/order and draw count.
_CACHE = OrderedDict()
_MAX_PLANS = 8


def clear_cache():
    _CACHE.clear()


def _sampling_plan(recs, draws):
    import numpy as np
    draws = operator.index(draws)
    key = (draws, tuple((row["c"], row.get("support"), row.get("query")) for row in recs))
    if key in _CACHE:
        plan = _CACHE.pop(key); _CACHE[key] = plan; return plan
    if not recs: raise ValueError("No paired observations")
    cls = np.array([row["c"] for row in recs]); ids = np.unique(cls)
    parent = list(range(len(recs)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]; index = parent[index]
        return index
    photo_owner = {}
    for index, record in enumerate(recs):
        for role in ("support", "query"):
            photo = record.get(role)
            if photo is None: continue
            if photo in photo_owner: parent[find(index)] = find(photo_owner[photo])
            else: photo_owner[photo] = index
    grouped = {}
    for index in range(len(recs)): grouped.setdefault(find(index), []).append(index)
    groups = tuple(np.asarray(group,dtype=int) for group in grouped.values())
    owner = np.empty(len(recs),dtype=int)
    for group, indices in enumerate(groups): owner[indices] = group
    class_index = np.searchsorted(ids,cls)
    present = np.zeros((len(groups),len(ids)),dtype=np.int64)
    np.add.at(present,(owner,class_index),1)
    # Preserve the public one RNG call per draw exactly. This small loop runs
    # once per common cohort, while the costly per-class arithmetic below is
    # vectorized/reused across all comparisons. No bulk-RNG equivalence assumed.
    rng = np.random.default_rng(0)
    picked = np.empty((draws,len(groups)),dtype=int)
    for draw in range(draws):
        picked[draw] = rng.integers(0,len(groups),len(groups))
    weights = np.zeros((draws,len(groups)),dtype=np.int64)
    if draws:
        np.add.at(weights,(np.repeat(np.arange(draws),len(groups)),picked.reshape(-1)),1)
    sampled_present = weights @ present > 0
    plan = dict(classes=cls,class_ids=ids,class_index=class_index,group_index=owner,
                groups=groups,present=present,weights=weights,sampled_present=sampled_present,
                draws=draws)
    _CACHE[key] = plan
    while len(_CACHE)>_MAX_PLANS: _CACHE.popitem(last=False)
    return plan


def _class_totals(values, plan):
    import numpy as np
    result = np.zeros((len(plan["groups"]),len(plan["class_ids"]),2),dtype=np.float64)
    np.add.at(result,(plan["group_index"],plan["class_index"]),values)
    return result


def _sample_scores(totals, plan):
    import numpy as np
    # This multiplies exact group-resampling counts, not probabilistic weights
    # or a normal/sandwich approximation. Query original pixel I/U are integers.
    sampled = np.einsum("dg,gci->dci",plan["weights"],totals,optimize=True)
    present = plan["sampled_present"]
    ratios = sampled[...,0] / np.maximum(sampled[...,1],1)
    return 100 * (ratios * present).sum(axis=1) / present.sum(axis=1)


def paired(recs, get, base, draws=2000):
    """Public-compatible dictionary; cache ONLY common exact resampling plans."""
    import numpy as np
    if not recs: raise ValueError("No paired observations")
    plan = _sampling_plan(recs,draws)
    a = np.array([get(row) for row in recs],float)
    b = np.array([base(row) for row in recs],float)
    a_totals,b_totals = _class_totals(a,plan),_class_totals(b,plan)
    def score(totals):
        full = totals.sum(axis=0)
        return 100 * np.mean(full[:,0] / np.maximum(full[:,1],1))
    gain = score(a_totals)-score(b_totals)
    boot = _sample_scores(a_totals,plan)-_sample_scores(b_totals,plan)
    groups = plan["groups"]
    return dict(miou=score(a_totals),gain=gain,
                ci95=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))] if len(groups)>1 else None,
                bootstrap_unit="connected support/query photograph groups",groups=len(groups),
                largest_group=max(map(len,groups)),draws=draws)
