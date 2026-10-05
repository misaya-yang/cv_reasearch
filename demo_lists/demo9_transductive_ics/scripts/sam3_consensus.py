#!/usr/bin/env python3
"""Agreement of SAM3's two routes: a proposal of the exemplar route confirmed by the named route (CPU, saved outputs).

  python scripts/sam3_consensus.py --run RUN/dev --manifest SUITE/dev_episodes.json --foris F/episodes.jsonl --out X/consensus_dev.json

The exemplar route matches appearance and the named route matches a category word; their errors differ. Arms, all
without a query label, at original resolution:
  exemplar, routed   controls (sam3_route_score.py)
  cons_inside        exemplar proposals with score x (share of the proposal inside the named semantic map) >= 0.7 x top
  cons_match         exemplar proposals with score x (best named instance: its score x IoU with the proposal) >= 0.7 x top
  cons_pool          both routes' instance proposals, each weighted by its best counterpart in the other route
Diagnostics with labels: labels_all, best_route; AUC of each signal over all exemplar proposals and over those the
0.7 rule keeps.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ("cons_inside", "cons_match", "cons_pool")
ARMS = ("exemplar", "routed") + METHODS + ("labels_all", "best_route")


def auc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + .5 * (pos[:, None] == neg[None]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--foris", required=True); p.add_argument("--variant", default="open_fast")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    ann = Path(man["annotation_root"])
    root = Path(a.run) / "reference_use" / a.variant
    named = {tuple(j["key"]): j for j in map(json.loads, (root / "variant.jsonl").read_text().splitlines())}
    foris = {(j["fold"], j["e"], j["c"]): j["original_iu"]["native"] for j in map(json.loads, Path(a.foris).read_text().splitlines())}
    recs = [r for f in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, f.read_text().splitlines())]
    recs = [r for r in recs if (r["fold"], r["e"], r["c"]) in named and r.get("candidate_file")][:a.limit]

    def bits(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)
    sig = {k: ([], [], [], []) for k in ("score", "inside_sem", "match", "score_x_inside", "score_x_match")}
    for n, r in enumerate(recs):
        key = (r["fold"], r["e"], r["c"])
        truth = np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        size = truth.shape[::-1]
        full = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8)).resize(size, Image.NEAREST)) > 0
        iu = lambda m: [int((m & truth).sum()), int((m | truth).sum())]
        raw, meta = bits(Path(a.run) / r["candidate_file"], r["proposal_shape"]), r["proposal_metadata"]
        q = [i for i in range(len(meta)) if meta[i][1] > 0]
        j = named[key]
        nraw, nmeta = bits(root / j["candidate_file"], j["proposal_shape"]), j["meta"]
        E = [(full(raw[i]), meta[i][0]) for i in q]
        N = [(full(nraw[i]), nmeta[i][0]) for i in range(min(len(nmeta), j["semantic_row"])) if nraw[i].any()]
        sem = full(nraw[j["semantic_row"]])
        iou = lambda x, y: float((x & y).sum()) / max(int((x | y).sum()), 1)
        M = np.array([[iou(e, m) for m, _ in N] for e, _ in E]).reshape(len(E), len(N))
        se, sn = np.array([s for _, s in E]), np.array([s for _, s in N])
        inside = np.array([float((e & sem).sum()) / max(int(e.sum()), 1) for e, _ in E])
        match_e = (M * sn[None]).max(1) if len(N) and len(E) else np.zeros(len(E))     # support of each exemplar proposal
        match_n = (M * se[:, None]).max(0) if len(N) and len(E) else np.zeros(len(N))  # support of each named proposal
        on = [bool((e & truth).sum() > .5 * e.sum()) for e, _ in E]
        top = float(se.max()) if len(E) else 0.0
        for i in range(len(E)):
            for k, x in (("score", se[i]), ("inside_sem", inside[i]), ("match", match_e[i]), ("score_x_inside", se[i] * inside[i]), ("score_x_match", se[i] * match_e[i])):
                sig[k][0 if on[i] else 1].append(float(x))
                if se[i] >= .7 * top:
                    sig[k][2 if on[i] else 3].append(float(x))
        union = lambda masks: np.any(masks, 0) if len(masks) else np.zeros(truth.shape, bool)
        ex = union([E[i][0] for i in range(len(E)) if se[i] >= .7 * top])
        routed = ex if top >= j["query_top_score"] else sem

        def rel(masks, score):
            score = np.asarray(score, float)
            if not len(masks) or score.max() <= 0:
                return routed
            return union([m for m, s in zip(masks, score) if s >= .7 * score.max()])
        r["iu"] = dict(exemplar=iu(ex), routed=iu(routed), foris=foris[key],
                       cons_inside=iu(rel([e for e, _ in E], se * inside) if sem.any() else routed),
                       cons_match=iu(rel([e for e, _ in E], se * match_e)),
                       cons_pool=iu(rel([e for e, _ in E] + [m for m, _ in N], np.concatenate([se * match_e, sn * match_n]))),
                       labels_all=iu(union([e for (e, _), g in zip(E, on) if g] + [m for m, _ in N if (m & truth).sum() > .5 * m.sum()])),
                       best_route=max(iu(ex), iu(sem), key=lambda v: v[0] / max(v[1], 1)))
        if n % 40 == 0:
            print("%d/%d" % (n + 1, len(recs)), flush=True)
    out = dict(state="COMPLETED", episodes=len(recs), arms={},
               auc_all={k: auc(v[0], v[1]) for k, v in sig.items()}, auc_kept={k: auc(v[2], v[3]) for k, v in sig.items()},
               kept=dict(on=len(sig["score"][2]), off=len(sig["score"][3])))
    print("%d episodes, original resolution" % len(recs))
    print("exemplar proposals, on against off target, all: " + ", ".join("%s %.3f" % kv for kv in out["auc_all"].items()))
    print("  among those the 0.7 rule keeps (%d on, %d off): " % (out["kept"]["on"], out["kept"]["off"]) + ", ".join("%s %.3f" % kv for kv in out["auc_kept"].items()))
    for arm in ARMS + ("foris",):
        row = dict(miou=S.paired(recs, lambda r: r["iu"][arm], lambda r: r["iu"][arm], draws=1)["miou"])
        text = "  %-12s %6.2f" % (arm, row["miou"])
        for base in ("routed", "foris"):
            if base == arm:
                continue
            v = S.paired(recs, lambda r: r["iu"][arm], lambda r: r["iu"][base])
            folds = [S.paired([r for r in recs if r["fold"] == f], lambda r: r["iu"][arm], lambda r: r["iu"][base], draws=1)["gain"] for f in sorted({r["fold"] for r in recs})]
            d = [r["iu"][arm][0] / max(r["iu"][arm][1], 1) - r["iu"][base][0] / max(r["iu"][base][1], 1) for r in recs]
            row["over_" + base] = dict(gain=v["gain"], ci95=v["ci95"], per_fold=folds, up=sum(x > 0 for x in d), down=sum(x < 0 for x in d))
            text += "   over %s %+6.2f [%+.2f, %+.2f] folds %s up %d down %d" % (base, v["gain"], *v["ci95"], " ".join("%+.1f" % f for f in folds), row["over_" + base]["up"], row["over_" + base]["down"])
        out["arms"][arm] = row
        print(text)
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], iu=r["iu"])) + "\n" for r in recs))


if __name__ == "__main__":
    main()
