#!/usr/bin/env python3
"""Where from correspondence, extent from the segmenter: SAM3's proposals chosen by FoRIS's region (CPU, saved outputs).

  python scripts/sam3_where_extent.py --run RUN/dev --manifest SUITE/dev_episodes.json --field CACHE/test \
      --foris F/episodes.jsonl --out X/where_dev.json

A proposal's `inside` is the share of its area that lies in FoRIS's mask (patch level, from the decision cache).
Arms, all without a query label, at original resolution:
  exemplar, routed   controls (sam3_route_score.py)
  foris_field        FoRIS's patch-level mask itself (check of the field against FoRIS's own number)
  poe_ex             exemplar proposals with score x inside >= 0.7 x the episode's top score x inside
  poe_all            the same over the exemplar proposals and the instance proposals of the named route
  arb_iou            the complete mask (exemplar or named) with the larger IoU against FoRIS's mask
  arb_poe            the complete mask (exemplar or named) with the larger top confidence x inside
Diagnostics with labels: labels_all (pool proposals mostly on the target), best_route; AUC of each signal per proposal.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ("poe_ex", "poe_all", "arb_iou", "arb_poe")
ARMS = ("exemplar", "routed", "foris_field") + METHODS + ("labels_all", "best_route")


def auc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + .5 * (pos[:, None] == neg[None]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--field", required=True); p.add_argument("--foris", required=True); p.add_argument("--variant", default="open_fast")
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
    recs = [r for r in recs if (r["fold"], r["e"], r["c"]) in named and r.get("candidate_file")
            and (Path(a.field) / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))).exists()][:a.limit]

    def bits(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)
    sig = {k: ([], []) for k in ("score", "inside", "score_x_inside")}
    for n, r in enumerate(recs):
        key = (r["fold"], r["e"], r["c"])
        truth = np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        size = truth.shape[::-1]
        full = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8)).resize(size, Image.NEAREST)) > 0
        iu = lambda m: [int((m & truth).sum()), int((m | truth).sum())]
        with np.load(Path(a.field) / ("%d_%d_%d.npz" % key), allow_pickle=True) as z:
            field = np.asarray(Image.fromarray(z["maps"][0].astype(np.float32)).resize(size, Image.BILINEAR)) > .5
        inside = lambda m: float((m & field).sum()) / max(int(m.sum()), 1)
        raw, meta = bits(Path(a.run) / r["candidate_file"], r["proposal_shape"]), r["proposal_metadata"]
        q = [i for i in range(len(meta)) if meta[i][1] > 0]
        j = named[key]
        nraw, nmeta = bits(root / j["candidate_file"], j["proposal_shape"]), j["meta"]
        # the pool: (mask at original size, SAM3 score, route)
        pool = [(full(raw[i]), meta[i][0], "ex") for i in q]
        pool += [(full(nraw[i]), nmeta[i][0], "name") for i in range(min(len(nmeta), j["semantic_row"])) if nraw[i].any()]
        ins = [inside(m) for m, _, _ in pool]
        on = [bool((m & truth).sum() > .5 * m.sum()) for m, _, _ in pool]
        for (m, s, route), v, good in zip(pool, ins, on):
            if route == "ex":
                for k, x in (("score", s), ("inside", v), ("score_x_inside", s * v)):
                    sig[k][0 if good else 1].append(x)
        union = lambda idx: np.any([pool[i][0] for i in idx], 0) if idx else np.zeros(truth.shape, bool)
        # per proposal: score, inside, area, pixels on the target, 1 for the exemplar route; then the field and the target
        r["props"] = [[round(float(sc), 4), round(v, 4), int(m.sum()), int((m & truth).sum()), int(route == "ex")] for (m, sc, route), v in zip(pool, ins)]
        r["areas"] = dict(field=int(field.sum()), truth=int(truth.sum()), both=int((field & truth).sum()), image=int(truth.size))

        def poe(idx):
            if not idx or not field.any():
                return None
            score = {i: pool[i][1] * ins[i] for i in idx}
            top = max(score.values())
            return union([i for i in idx if score[i] >= .7 * top]) if top > 0 else None
        exi = [i for i in range(len(pool)) if pool[i][2] == "ex"]
        top = max((pool[i][1] for i in exi), default=0.0)
        ex = union([i for i in exi if pool[i][1] >= .7 * top])
        sem = full(nraw[j["semantic_row"]])
        routed = ex if top >= j["query_top_score"] else sem
        fiou = lambda m: float((m & field).sum()) / max(int((m | field).sum()), 1)
        r["iu"] = dict(exemplar=iu(ex), routed=iu(routed), foris_field=iu(field), foris=foris[key],
                       poe_ex=iu(v if (v := poe(exi)) is not None else ex),
                       poe_all=iu(v if (v := poe(list(range(len(pool))))) is not None else routed),
                       arb_iou=iu(routed if not field.any() else (ex if fiou(ex) >= fiou(sem) else sem)),
                       arb_poe=iu(routed if not field.any() else (ex if top * inside(ex) >= j["query_top_score"] * inside(sem) else sem)),
                       labels_all=iu(union([i for i in range(len(pool)) if on[i]])),
                       best_route=max(iu(ex), iu(sem), key=lambda v: v[0] / max(v[1], 1)))
        if n % 40 == 0:
            print("%d/%d" % (n + 1, len(recs)), flush=True)
    out = dict(state="COMPLETED", episodes=len(recs), arms={},
               auc_exemplar_proposals={k: dict(auc=auc(*v), on=len(v[0]), off=len(v[1])) for k, v in sig.items()})
    print("%d episodes, original resolution" % len(recs))
    print("exemplar proposals, on-target against off-target: " + ", ".join("%s %.3f" % (k, v["auc"]) for k, v in out["auc_exemplar_proposals"].items()))
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
    Path(a.out).with_suffix(".episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], iu=r["iu"], props=r["props"], areas=r["areas"], named_top=named[(r["fold"], r["e"], r["c"])]["query_top_score"])) + "\n" for r in recs))


if __name__ == "__main__":
    main()
