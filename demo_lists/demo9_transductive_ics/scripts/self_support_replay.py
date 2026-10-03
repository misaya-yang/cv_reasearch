#!/usr/bin/env python3
"""Query self-support on cached features: the reference only names a seed, the query decides the extent itself.

FoRIS's score gives a trimap: sure target (the seed), sure background, unknown. Every unknown patch goes to the
side it resembles more inside the query (mean of its top-k similarities to each seed set; a patch never votes for
itself or its close neighbours). Variants change the seed (purified to its dominant set), the aggregation, the band
that is re-decided and the number of rounds. Seeds taken from the truth are upper bounds, not methods.

  python scripts/self_support_replay.py --cache cache/evidence_v1 --run results/extent_v1/run \
      --out results/self_support_v0/replay.json            # GPU if present; no encoder
  python scripts/self_support_replay.py --report results/self_support_v0/replay.json   # CPU

Patch level (64 x 64), no refinement. Development half: the 40 development tasks and fresh episodes with an even
index; confirmation half: fresh episodes with an odd index. A variant is chosen on the first and read on the second.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_extent import load, miou  # noqa: E402


def topk_mean(G, cols, k):
    if int(cols.sum()) == 0:
        return G.new_full((G.shape[0],), -1.0)
    S = G[:, cols]
    return S.topk(min(k, S.shape[1]), dim=1).values.mean(1)


def dominant(G, seed, weight, level, rounds=200):
    """Dominant set of the seed under replicator dynamics; affinities are similarities above `level`."""
    import torch
    idx = seed.nonzero()[:, 0]
    A = (G[idx][:, idx] - level).clamp_min(0)
    A.fill_diagonal_(0)
    x = weight[idx].clamp_min(1e-6)
    x = x / x.sum()
    for _ in range(rounds):
        y = x * (A @ x)
        s = y.sum()
        if float(s) <= 0:
            return seed
        x = y / s
    keep = x > 1e-3 * x.max()
    out = torch.zeros_like(seed)
    out[idx[keep]] = True
    return out if int(out.sum()) >= 4 else seed


def decide(G, sn, F0, B0, k, band, rounds):
    """Unknown patches inside the band go to the nearer seed set; the seeds grow with them between rounds."""
    unknown = (sn > band[0]) & (sn < band[1]) & ~F0 & ~B0
    F, B = F0, B0
    for _ in range(rounds):
        d = topk_mean(G, F, k) - topk_mean(G, B, k)
        F, B = F0 | (unknown & (d > 0)), B0 | (unknown & (d <= 0))
    return (sn > 0.5) & ~unknown | F & unknown | F0 & (sn > 0.5)


def neighbour_vote(G, sn, k, rounds, soft):
    """Every patch takes the mean label of its k most similar patches elsewhere in the query."""
    idx = G.topk(k, dim=1).indices
    y = sn if soft else (sn > 0.5).float()
    for _ in range(rounds):
        y = y[idx].mean(1)
        if not soft:
            y = (y > 0.5).float()
    return y > 0.5


def propagate(G, sn, k, alpha, steps=10):
    """Score diffusion on the query's own k-nearest-neighbour graph, anchored to the FoRIS score."""
    val, idx = G.topk(k, dim=1)
    w = val.clamp_min(0)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    f = sn
    for _ in range(steps):
        f = alpha * (w * f[idx]).sum(1) + (1 - alpha) * sn
    return f > 0.5


def replay(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load(a.run)[:a.limit]
    near, rows = None, []
    with torch.no_grad():
        for r in recs:
            name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
            q = torch.load(Path(a.cache) / "feat" / (name + ".pt"), map_location="cpu", weights_only=False)["q"].to(dev).float()
            z = np.load(Path(a.run) / "packets" / (name + ".npz"))
            n = q.shape[0]
            side = int(round(n ** 0.5))
            ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
            tf = torch.from_numpy(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)).astype(np.float32)).to(dev).flatten()
            s = torch.from_numpy(z["score"].astype(np.float32)).to(dev).flatten()
            sn = (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)
            if near is None or near.shape[0] != n:
                y, x = torch.arange(n, device=dev) // side, torch.arange(n, device=dev) % side
                near = ((y[:, None] - y[None]).abs() <= 2) & ((x[:, None] - x[None]).abs() <= 2)
            G = (q @ q.T).masked_fill(near, -1.0)
            order = sn.argsort()
            pick = lambda m, top: m if int(m.sum()) >= 16 else torch.zeros_like(m).index_fill(0, order[-16:] if top else order[:16], True)
            truth, native = tf > 0.5, sn > 0.5
            B0 = pick(sn <= 0.2, False)
            seeds = {"core": pick(sn >= 0.8, True), "core65": pick(sn >= 0.65, True)}
            level = float(G[seeds["core"]][:, B0].clamp_min(0).mean())  # how alike a seed patch and a background patch are
            seeds["dominant"] = dominant(G, seeds["core"], sn, level)
            seeds["dominant65"] = dominant(G, seeds["core65"], sn, level)
            orc = {"ORACLE_pure_core": seeds["core"] & truth, "ORACLE_core_plus_found": seeds["core"] | (native & truth),
                   "ORACLE_found": native & truth}
            seeds.update({k: (v if int(v.sum()) >= 4 else seeds["core"]) for k, v in orc.items()})
            iu = lambda m: [float(tf[m].sum()), float(tf.sum() + m.sum() - tf[m].sum())]
            row = dict(fold=r["fold"], e=r["e"], c=r["c"], dev40=bool(r.get("dev40")), iu=dict(native=iu(native)),
                       purity={k: float(truth[v].float().mean()) for k, v in seeds.items()},
                       cover={k: float((v & truth).sum() / truth.sum().clamp_min(1)) for k, v in seeds.items()})
            for seed, k, band, rounds in itertools.product(seeds, (1, 5), ((0.5, 0.8), (0.2, 0.8), (0.2, 1.01)), (1, 3)):
                if seed.startswith("ORACLE") and (k, rounds) != (5, 1):
                    continue
                row["iu"]["%s|k%d|%.1f-%.1f|r%d" % (seed, k, band[0], min(band[1], 1.0), rounds)] = iu(decide(G, sn, seeds[seed], B0, k, band, rounds))
            rejected = sn <= 0.5  # everything FoRIS rejects is the background model; only its own mask is re-decided
            for seed, k in itertools.product(("core", "core65"), (1, 5, 20)):
                row["iu"]["%s_vs_rejected|k%d" % (seed, k)] = iu(decide(G, sn, seeds[seed], rejected, k, (0.5, 1.01), 1))
            for k, rounds, soft in itertools.product((5, 10, 20), (1, 3), (False, True)):
                row["iu"]["vote|k%d|r%d|%s" % (k, rounds, "soft" if soft else "hard")] = iu(neighbour_vote(G, sn, k, rounds, soft))
            for k, alpha in itertools.product((10, 30), (0.5, 0.8, 0.95)):
                row["iu"]["propagate|k%d|a%.2f" % (k, alpha)] = iu(propagate(G, sn, k, alpha))
            row["iu"]["ORACLE_found_true_background|k5"] = iu(decide(G, sn, seeds["ORACLE_found"], ~truth & ~native if int((~truth & ~native).sum()) >= 16 else B0, 5, (0.2, 1.01), 1))
            rows.append(row)
    Path(a.out).write_text(json.dumps(dict(state="COMPLETED", episodes=len(rows), device=dev, rows=rows)))
    print(json.dumps(dict(state="COMPLETED", episodes=len(rows), variants=len(rows[0]["iu"]))))


def report(a):
    rows = json.loads(Path(a.report).read_text())["rows"]
    cls = np.array([r["c"] for r in rows])
    devm = np.array([r["dev40"] or r["e"] % 2 == 0 for r in rows])
    names = list(rows[0]["iu"])
    T = {k: np.array([r["iu"][k] for r in rows]) for k in names}
    sc = lambda t, m: float(miou(t[m, 0], t[m, 1], cls[m])[0])

    def ci(t, m):
        w = np.random.default_rng(0).multinomial(int(m.sum()), np.ones(int(m.sum())) / m.sum(), size=2000).astype(float)
        d = miou(t[m, 0], t[m, 1], cls[m], w) - miou(T["native"][m, 0], T["native"][m, 1], cls[m], w)
        return [float(x) for x in np.percentile(d, [2.5, 97.5])]
    print("episodes: development %d, confirmation %d; native %.2f / %.2f" % (devm.sum(), (~devm).sum(), sc(T["native"], devm), sc(T["native"], ~devm)))
    gain = {k: sc(T[k], devm) - sc(T["native"], devm) for k in names if k != "native"}
    free = sorted((k for k in gain if not k.startswith("ORACLE")), key=lambda k: -gain[k])
    print("label-free variants, development half (top 16 and bottom 3):")
    for k in free[:16] + free[-3:]:
        print("  %-34s %+6.2f [%+.2f, %+.2f]" % (k, gain[k], *ci(T[k], devm)))
    print("seeds from the truth (upper bounds), development half:")
    for k in names:
        if k.startswith("ORACLE"):
            print("  %-44s %+6.2f [%+.2f, %+.2f]" % (k, gain[k], *ci(T[k], devm)))
    best = free[0]
    print("chosen on development: %s -> confirmation half %+.2f [%+.2f, %+.2f]" % (best, sc(T[best], ~devm) - sc(T["native"], ~devm), *ci(T[best], ~devm)))
    for k in ("core", "dominant", "core65", "dominant65"):
        p = np.array([r["purity"][k] for r in rows])
        print("seed %-11s purity mean %.3f, >=0.9 in %.2f, <0.5 in %.2f; covers %.2f of the target" % (
            k, p.mean(), (p >= 0.9).mean(), (p < 0.5).mean(), np.mean([r["cover"][k] for r in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--run")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--report")
    a = p.parse_args()
    report(a) if a.report else replay(a)


if __name__ == "__main__":
    main()
