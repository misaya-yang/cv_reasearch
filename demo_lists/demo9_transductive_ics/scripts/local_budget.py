#!/usr/bin/env python3
"""The error budget from files already on this machine: no encoder, no server, CPU seconds.

  python scripts/local_budget.py --run results/extent_v1/run --maps results/evidence_v1/maps \
      --replay results/self_support_v0/replay.json --out results/self_support_v0/local_budget.json

Every row is a patch-level mask scored against the truth (64 x 64, no refinement). "share" rows take as many
patches as the target truly has, the highest by the named evidence: they show what the evidence could give if the
target's share of the query were known. Rows marked LABELS use query labels or labels of other images; they are
bounds, not methods.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_extent import SMALL, load, miou
from evidence_audit import normalise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    cls, area = np.array([r["c"] for r in recs]), np.array([r["area"] for r in recs])
    rows = {}

    def add(name, j, mask, tf):
        i = float(tf[mask].sum())
        rows.setdefault(name, np.zeros((n, 2)))[j] = (i, float(tf.sum() + mask.sum() - i))
    for j, r in enumerate(recs):
        name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
        z, m = np.load(a.run / "packets" / (name + ".npz")), np.load(a.maps / (name + ".npz"))
        tf = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).ravel()
        count = max(int((tf > 0.5).sum()), 1)
        sn = normalise(z["score"].astype(np.float32)).ravel()
        f = lambda k: m[k].astype(np.float32)
        share = lambda v: np.isin(np.arange(4096), np.argsort(-np.nan_to_num(v, nan=-9))[:count])
        add("FoRIS, its own cut", j, sn > 0.5, tf)
        add("FoRIS, true share  LABELS", j, share(sn), tf)
        add("nearest reference label, its own cut", j, f("fg") > f("bg"), tf)
        add("nearest reference label, true share  LABELS", j, share(f("fg") - f("bg")), tf)
        add("query's own core against its sure background, own cut", j, f("grp_core") > f("grp_out"), tf)
        add("query's own core against its sure background, true share  LABELS", j, share(f("grp_core") - f("grp_out")), tf)
        add("truly found part against sure background, own cut  LABELS", j, np.nan_to_num(f("grp_true"), nan=-9) > f("grp_out"), tf)
        add("truly found part against sure background, true share  LABELS", j, share(f("grp_true") - f("grp_out")), tf)
        add("truly found part against reference background, true share  LABELS", j, share(f("grp_true") - f("bg")), tf)
        add("supervised class classifier (1200 other images), true share  LABELS", j, share(f("probe_class")), tf)
        add("labelled pool of the class against the rest, true share  LABELS", j, share(np.maximum(f("fg"), f("pos_oracle")) - np.maximum(f("bg"), f("neg_oracle"))), tf)
        add("grid ceiling (any patch-constant mask)", j, tf > 0.5, tf)
    rep = json.loads(a.replay.read_text())["rows"]
    key = {(r["fold"], r["e"], r["c"]): r["iu"]["ORACLE_found_true_background|k5"] for r in rep}
    rows["every correct patch labelled, only FoRIS's errors re-decided inside the query  LABELS"] = np.array([key[r["fold"], r["e"], r["c"]] for r in recs])
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    base = rows["FoRIS, its own cut"]
    out = dict(state="ANALYSED", episodes=n, scope="patch level, no refinement; exploratory", rows={})
    for k, t in rows.items():
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
        sc = lambda msk: float(miou(t[msk, 0], t[msk, 1], cls[msk])[0])
        out["rows"][k] = dict(patch_miou=sc(np.ones(n, bool)), over_foris=float(sc(np.ones(n, bool)) - miou(base[:, 0], base[:, 1], cls)[0]),
                              ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], small_targets=sc(area < SMALL), large_targets=sc(area >= SMALL))
        v = out["rows"][k]
        print("%-82s %6.2f  %+6.2f [%+.2f, %+.2f]  small %5.1f  large %5.1f" % (k, v["patch_miou"], v["over_foris"], *v["ci95"], v["small_targets"], v["large_targets"]))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
