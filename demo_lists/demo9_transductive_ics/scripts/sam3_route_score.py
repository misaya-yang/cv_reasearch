#!/usr/bin/env python3
"""Original-resolution reading of the two routes and of their confidence routing (CPU).

  python scripts/sam3_route_score.py --run RUN/dev --manifest SUITE/dev_episodes.json [--foris F/episodes.jsonl] --out X/route_dev.json

Arms, all without a query label:
  exemplar        first pass (canvas, reference box), proposals scored >= 0.7 x the episode's top score
  top1_if_empty   first pass at the public 0.5, the top proposal when nothing passes (control)
  named           SAM3 on the query alone with the name it gave the reference from a general vocabulary: the semantic map
  routed          `exemplar` or `named`, whichever has the higher top confidence on the query (no level)
Class mIoU, paired 95% intervals over photograph groups (2000 draws, seed 0), gains per fold, and against FoRIS when
its per-episode records are given.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARMS = ("exemplar", "top1_if_empty", "named", "routed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--variant", default="open_fast"); p.add_argument("--foris"); p.add_argument("--fresh-from", nargs="*")
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    ann = Path(man["annotation_root"])
    root = Path(a.run) / "reference_use" / a.variant
    named = {tuple(j["key"]): j for j in map(json.loads, (root / "variant.jsonl").read_text().splitlines())}
    recs = [r for f in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, f.read_text().splitlines())]
    recs = [r for r in recs if (r["fold"], r["e"], r["c"]) in named and r.get("candidate_file")]
    if a.fresh_from:
        images = {r[k] for m in a.fresh_from for r in json.loads(Path(m).read_text())["episodes"] for k in ("support", "query")}
        recs = [r for r in recs if r["support"] not in images and r["query"] not in images]
    foris = {}
    if a.foris:
        foris = {(j["fold"], j["e"], j["c"]): j["original_iu"]["native"] for j in map(json.loads, Path(a.foris).read_text().splitlines())}

    def bits(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)
    for r in recs:
        truth = np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        full = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8)).resize(truth.shape[::-1], Image.NEAREST)) > 0
        iu = lambda m: [int((m & truth).sum()), int((m | truth).sum())]
        raw, meta = bits(Path(a.run) / r["candidate_file"], r["proposal_shape"]), r["proposal_metadata"]
        q = [i for i in range(len(meta)) if meta[i][1] > 0]
        top = max((meta[i][0] for i in q), default=0.0)
        union = lambda idx: full(raw[idx].any(0)) if idx else np.zeros(truth.shape, bool)
        ex = union([i for i in q if meta[i][0] >= .7 * top])
        j = named[(r["fold"], r["e"], r["c"])]
        sem = full(bits(root / j["candidate_file"], j["proposal_shape"])[j["semantic_row"]])
        r["iu"] = dict(exemplar=iu(ex), top1_if_empty=iu(union([i for i in q if meta[i][0] > .5] or ([max(q, key=lambda i: meta[i][0])] if q else []))),
                       named=iu(sem), routed=iu(ex if top >= j["query_top_score"] else sem))
        r["used_named"] = bool(top < j["query_top_score"])
        if foris:
            r["iu"]["foris"] = foris[(r["fold"], r["e"], r["c"])]
    out = dict(state="COMPLETED", episodes=len(recs), variant=a.variant, named_route_used_in=float(np.mean([r["used_named"] for r in recs])), arms={})
    print("%d episodes, original resolution; the named route is used in %.0f%% of them" % (len(recs), 100 * out["named_route_used_in"]))
    bases = ["exemplar"] + (["foris"] if foris else [])
    for arm in list(ARMS) + (["foris"] if foris else []):
        row = dict(miou=S.paired(recs, lambda r: r["iu"][arm], lambda r: r["iu"][arm], draws=1)["miou"])
        text = "  %-14s %6.2f" % (arm, row["miou"])
        for base in bases:
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
    Path(a.out).with_suffix(".episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], iu=r["iu"], used_named=r["used_named"])) + "\n" for r in recs))


if __name__ == "__main__":
    main()
