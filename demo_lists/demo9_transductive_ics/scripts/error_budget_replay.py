#!/usr/bin/env python3
"""Where the gap to a perfect answer sits, in one currency. Replay on cached features; uses query labels; no method.

The correct decision for a query patch is the Bayes rule on the query: it needs what the target looks like in the
query, what the background looks like in the query, and how much of the query is target. A reference gives all three
for another image. This replay swaps the query's true version in, one at a time, under one fixed rule (mean of the
5 highest similarities to a positive set minus the same for a negative set; a query patch never uses itself or the
8 patches around it, so a target of fewer than about 10 patches has nothing left to vote and reads as a miss):

  ref_fg | ref_bg        everything from the reference (nearest-label transfer, the base every method starts from)
  ref_fg | query_bg      the query's true background replaces the reference's      -> cost of the background gap
  query_fg | ref_bg      the query's true target replaces the reference's target   -> cost of the appearance gap
  query_fg | query_bg    both from the query: what this representation can separate at all (its ceiling)
Each is read twice: with the natural cut (margin above 0; only meaningful when both sets come from the same image)
and with the true share of the query (the top patches by margin, as many as the target has) -> cost of the share gap.
The FoRIS score is read the same two ways, so the rows are comparable with the published method.

  python scripts/error_budget_replay.py --cache cache/evidence_v1 --run results/extent_v1/run --out results/error_budget_v0/budget.json
  python scripts/error_budget_replay.py --report results/error_budget_v0/budget.json

Any other representation (another layer, features after scale alignment) can be given the same test by pointing
--cache at a folder with the same feat/*.pt layout: the representation to build on is the one with the highest
ceiling and the smallest appearance gap. Patch level (64 x 64), no refinement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_extent import SMALL, load, miou  # noqa: E402

K = 5


def replay(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load(a.run)[:a.limit]
    near, rows = None, []
    with torch.no_grad():
        for r in recs:
            name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
            feat = torch.load(Path(a.cache) / "feat" / (name + ".pt"), map_location="cpu", weights_only=False)
            z = np.load(Path(a.run) / "packets" / (name + ".npz"))
            q, ref = feat["q"].to(dev).float(), feat["r"].to(dev).float()
            n = q.shape[0]
            side = int(round(n ** 0.5))
            ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
            tf = torch.from_numpy(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)).astype(np.float32)).to(dev).flatten()
            truth = tf > 0.5
            s = torch.from_numpy(z["score"].astype(np.float32)).to(dev).flatten()
            sn = (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)
            fgm = torch.from_numpy(z["cov"]).to(dev).flatten() >= 0.5
            if near is None or near.shape[0] != n:
                y, x = torch.arange(n, device=dev) // side, torch.arange(n, device=dev) % side
                near = ((y[:, None] - y[None]).abs() <= 1) & ((x[:, None] - x[None]).abs() <= 1)
            G, C = (q @ q.T).masked_fill(near, -1.0), q @ ref.T
            top = lambda S: S.topk(min(K, S.shape[1]), dim=1).values.mean(1) if S.shape[1] else S.new_full((n,), -1.0)
            sets = dict(ref_fg=top(C[:, fgm]), ref_bg=top(C[:, ~fgm]), query_fg=top(G[:, truth]), query_bg=top(G[:, ~truth]))
            count = int(truth.sum())
            iu = lambda m: [float(tf[m].sum()), float(tf.sum() + m.sum() - tf[m].sum())]

            def share(v):  # as many patches as the target has, the highest by v
                m = torch.zeros(n, dtype=torch.bool, device=dev)
                m[v.topk(max(count, 1)).indices] = True
                return m
            row = dict(fold=r["fold"], e=r["e"], c=r["c"], area=float(truth.float().mean()),
                       iu={"foris|natural": iu(sn > 0.5), "foris|share": iu(share(sn)), "ceiling_of_the_grid": iu(truth)},
                       similarity=dict(target_to_reference_target=float(sets["ref_fg"][truth].mean()) if count else None,
                                       target_to_own_target=float(sets["query_fg"][truth].mean()) if count else None,
                                       background_to_reference_background=float(sets["ref_bg"][~truth].mean()),
                                       background_to_own_background=float(sets["query_bg"][~truth].mean())))
            for pos, neg in (("ref_fg", "ref_bg"), ("ref_fg", "query_bg"), ("query_fg", "ref_bg"), ("query_fg", "query_bg")):
                margin = sets[pos] - sets[neg]
                row["iu"]["%s|%s|share" % (pos, neg)] = iu(share(margin))
                if pos[:3] == neg[:3]:
                    row["iu"]["%s|%s|natural" % (pos, neg)] = iu(margin > 0)
            rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(state="COMPLETED", episodes=len(rows), device=dev, k=K, rows=rows)))
    print(json.dumps(dict(state="COMPLETED", episodes=len(rows))))


def report(a):
    rows = json.loads(Path(a.report).read_text())["rows"]
    n = len(rows)
    cls, area = np.array([r["c"] for r in rows]), np.array([r["area"] for r in rows])
    T = {k: np.array([r["iu"][k] for r in rows]) for k in rows[0]["iu"]}
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    sc = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])
    ci = lambda t: [float(x) for x in np.percentile(miou(t[:, 0], t[:, 1], cls, w), [2.5, 97.5])]
    table = {k: dict(patch_miou=sc(T[k]), ci95=ci(T[k]), small_targets=sc(T[k], area < SMALL), large_targets=sc(T[k], area >= SMALL)) for k in T}
    g = lambda k: table[k]["patch_miou"]
    budget = dict(
        ceiling_of_this_representation=g("query_fg|query_bg|natural"),
        not_separable_even_with_the_query_s_own_labels=g("ceiling_of_the_grid") - g("query_fg|query_bg|natural"),
        appearance_gap=g("query_fg|query_bg|share") - g("ref_fg|query_bg|share"),
        background_gap=g("query_fg|query_bg|share") - g("query_fg|ref_bg|share"),
        both_gaps=g("query_fg|query_bg|share") - g("ref_fg|ref_bg|share"),
        share_gap_for_nearest_label_transfer=g("ref_fg|ref_bg|share") - g("ref_fg|ref_bg|natural"),
        share_gap_for_foris=g("foris|share") - g("foris|natural"),
        foris_over_nearest_label_transfer=g("foris|natural") - g("ref_fg|ref_bg|natural"))
    sim = {k: float(np.mean([r["similarity"][k] for r in rows if r["similarity"][k] is not None])) for k in rows[0]["similarity"]}
    out = dict(state="ANALYSED", episodes=n, scope="patch level, no refinement; uses query labels; a decomposition, not a method",
               rows=table, budget=budget, mean_similarity=sim)
    for k, v in table.items():
        print("%-28s %6.2f [%5.2f, %5.2f]  small %6.2f  large %6.2f" % (k, v["patch_miou"], *v["ci95"], v["small_targets"], v["large_targets"]))
    print(json.dumps(dict(budget={k: round(v, 2) for k, v in budget.items()}, mean_similarity={k: round(v, 3) for k, v in sim.items()})))
    if a.analysis:
        Path(a.analysis).write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--run")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--report")
    p.add_argument("--analysis", help="with --report: where to write the table and the budget")
    a = p.parse_args()
    report(a) if a.report else replay(a)


if __name__ == "__main__":
    main()
