#!/usr/bin/env python3
"""Which of the shortlisted names to trust: reading of the top-k names pass (CPU, original resolution).

  python scripts/sam3_rerank_read.py --run RUN/dev --manifest SUITE/dev_episodes.json --variant topk_clean --out X/rerank_dev.json

The reference alone often fits several names (a hot dog is also a bun); the query can tell them apart. Per episode the
top-k names by the reference score (SAM3 confidence x IoU with the reference mask) were each given to SAM3 on the query.
Name rules, none uses a query label:
  top1      the best name on the reference (the method so far, with the cleaned vocabulary)
  qconf     the name with the largest reference score x top confidence on the query
  agree     the name with the largest reference score x IoU between its semantic map and the exemplar route's mask
  both      reference score x query confidence x that IoU
  union     the union of the semantic maps of the names whose reference score is >= 0.7 x the best
Each rule is read alone (`named_*`) and routed against the exemplar route by top confidence (`routed_*`).
Controls: exemplar, the earlier named/routed (variant open_fast), the true class name (privileged), the label-chosen name.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
RULES = ("top1", "qconf", "agree", "both", "union")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--variant", default="topk_clean"); p.add_argument("--foris"); p.add_argument("--fresh-from", nargs="*")
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    run = Path(a.run)
    man = json.loads(Path(a.manifest).read_text()); ann = Path(man["annotation_root"])
    load = lambda v: {tuple(j["key"]): j for j in map(json.loads, (run / "reference_use" / v / "variant.jsonl").read_text().splitlines())}
    have = lambda v: (run / "reference_use" / v / "variant.jsonl").exists()
    new = load(a.variant)
    old, st, nt = (load(v) if have(v) else {} for v in ("open_fast", "semantic_text", "native_text"))
    base = "routed_old" if old else "routed_by_instance"
    foris = {}
    if a.foris:
        foris = {(j["fold"], j["e"], j["c"]): j["original_iu"]["native"] for j in map(json.loads, Path(a.foris).read_text().splitlines())}
    recs = [r for f in sorted(run.glob("predictions_shard*.jsonl")) for r in map(json.loads, f.read_text().splitlines())]
    recs = [r for r in recs if (r["fold"], r["e"], r["c"]) in new and r.get("candidate_file")]
    if a.fresh_from:
        images = {r[k] for m in a.fresh_from for r in json.loads(Path(m).read_text())["episodes"] for k in ("support", "query")}
        recs = [r for r in recs if r["support"] not in images and r["query"] not in images]

    def bits(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)
    io = lambda v: v[0] / max(v[1], 1)
    for n, r in enumerate(recs):
        k = (r["fold"], r["e"], r["c"])
        truth = np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        full = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8)).resize(truth.shape[::-1], Image.NEAREST)) > 0
        iu = lambda m: [int((m & truth).sum()), int((m | truth).sum())]
        raw, meta = bits(run / r["candidate_file"], r["proposal_shape"]), r["proposal_metadata"]
        q = [i for i in range(len(meta)) if meta[i][1] > 0]
        top = max((meta[i][0] for i in q), default=0.0)
        keep = [i for i in q if meta[i][0] >= .7 * top]
        ex = full(raw[keep].any(0)) if keep else np.zeros(truth.shape, bool)
        j = new[k]
        rows = bits(run / "reference_use" / a.variant / j["candidate_file"], j["proposal_shape"])
        K = len(j["rerank"])
        sem = [full(rows[j["semantic_row"] + i]) for i in range(K)]
        qc = np.array([v[3] for v in j["rerank"]])
        inst = np.array([v[1] * v[2] for v in j["rerank"]])  # the best single instance on the reference
        wide = len(j["rerank"][0]) >= 7
        semr = np.array([v[4] * v[5] for v in j["rerank"]]) if wide else inst  # the name's semantic map on the reference
        uni = np.array([v[4] * v[6] for v in j["rerank"]]) if wide else inst   # the union of its confident instances
        ref = np.maximum(inst, np.maximum(semr, uni))
        agree = np.array([float((m & ex).sum()) / max(int((m | ex).sum()), 1) for m in sem])
        pick = dict(top1=int(ref.argmax()), by_instance=int(inst.argmax()), by_semantic=int(semr.argmax()), by_union=int(uni.argmax()), qconf=int((ref * qc).argmax()), agree=int((ref * agree).argmax()) if ex.any() and agree.max() > 0 else 0,
                    both=int((ref * qc * agree).argmax()) if ex.any() and agree.max() > 0 else int((ref * qc).argmax()))
        r["iu"] = dict(exemplar=iu(ex))
        for rule, i in pick.items():
            r["iu"]["named_" + rule] = iu(sem[i]); r["iu"]["routed_" + rule] = iu(ex if top >= qc[i] else sem[i])
        u = [i for i in range(K) if ref[i] >= .7 * ref.max()]
        um = np.any([sem[i] for i in u], 0)
        r["iu"]["named_union"] = iu(um); r["iu"]["routed_union"] = iu(ex if top >= max(qc[i] for i in u) else um)
        # votes: every name is a voter weighted by its reference score; `all` adds the exemplar route weighted by its top score
        w = ref / max(ref.sum(), 1e-9)
        V = np.tensordot(w, np.stack(sem).astype(np.float32), 1)
        for th in (.5, .35):
            r["iu"]["named_vote_%.2f" % th] = iu(V >= th)
        Va = (np.tensordot(ref, np.stack(sem).astype(np.float32), 1) + top * ex) / max(ref.sum() + top, 1e-9)
        r["iu"]["vote_all_0.50"] = iu(Va >= .5); r["iu"]["vote_all_0.35"] = iu(Va >= .35)
        pair = np.array([[float((x & y).sum()) / max(int((x | y).sum()), 1) for y in sem] for x in sem])
        med = int((pair @ ref).argmax())
        r["iu"]["named_medoid"] = iu(sem[med]); r["iu"]["routed_medoid"] = iu(ex if top >= qc[med] else sem[med])
        meda = int((pair @ ref + top * agree).argmax())
        r["iu"]["named_medoid_ex"] = iu(sem[meda])
        r["per_name"] = [[j["rerank"][i][0], float(j["rerank"][i][1]), float(j["rerank"][i][2]), float(qc[i]), float(agree[i]), float(pair[i] @ ref), int(sem[i].sum())] + iu(sem[i]) + [float(inst[i]), float(semr[i]), float(uni[i])] for i in range(K)]
        r["ex_top"] = float(top)
        best = max(range(K), key=lambda i: io(iu(sem[i])))
        r["iu"]["named_label_chosen"] = iu(sem[best]); r["iu"]["routed_label_chosen"] = iu(ex if top >= qc[best] else sem[best])
        o = old.get(k)
        if o:
            so = full(bits(run / "reference_use/open_fast" / o["candidate_file"], o["proposal_shape"])[o["semantic_row"]])
            r["iu"]["named_old"] = iu(so); r["iu"]["routed_old"] = iu(ex if top >= o["query_top_score"] else so)
        if k in st and k in nt:
            s_true = full(bits(run / "reference_use/semantic_text" / st[k]["candidate_file"], st[k]["proposal_shape"])[0])
            r["iu"]["named_true"] = iu(s_true); r["iu"]["routed_true"] = iu(ex if top >= max((m[0] for m in nt[k]["meta"]), default=0.0) else s_true)
            # the same true name read from its instance proposals instead of the semantic map (readout ablation, privileged)
            tr = bits(run / "reference_use/native_text" / nt[k]["candidate_file"], nt[k]["proposal_shape"]); ts = [m[0] for m in nt[k]["meta"]]
            pick_t = lambda idx: full(tr[idx].any(0)) if idx else np.zeros(truth.shape, bool)
            r["iu"]["true_instances_0.5"] = iu(pick_t([i for i in range(len(ts)) if ts[i] > .5]))
            r["iu"]["true_instances_rel0.7"] = iu(pick_t([i for i in range(len(ts)) if ts[i] >= .7 * max(ts)]))
        if foris:
            r["iu"]["foris"] = foris[k]
        r["names"] = dict(true=j["true_class"], top1=j["rerank"][0][0], old=(o or {}).get("name"), picks={x: j["rerank"][i][0] for x, i in pick.items()}, label_chosen=j["rerank"][best][0])
        if n % 100 == 0:
            print("%d/%d" % (n + 1, len(recs)), flush=True)
    base = "routed_old" if all("routed_old" in r["iu"] for r in recs) else "routed_by_instance"
    arms = [k for k in recs[0]["iu"] if all(k in r["iu"] for r in recs)]
    out = dict(state="COMPLETED", episodes=len(recs), variant=a.variant, arms={})
    print("%d episodes, original resolution, paired against %s" % (len(recs), base))
    for arm in arms:
        v = S.paired(recs, lambda r: r["iu"][arm], lambda r: r["iu"][base])
        folds = [S.paired([r for r in recs if r["fold"] == f], lambda r: r["iu"][arm], lambda r: r["iu"][base], draws=1)["gain"] for f in sorted({r["fold"] for r in recs})]
        d = [io(r["iu"][arm]) - io(r["iu"][base]) for r in recs]
        out["arms"][arm] = dict(miou=v["miou"], gain=v["gain"], ci95=v["ci95"], per_fold=folds, up=sum(x > .02 for x in d), down=sum(x < -.02 for x in d))
        print("  %-20s %6.2f  %+6.2f [%+.2f, %+.2f]  folds %s  up %d down %d" % (arm, v["miou"], v["gain"], *(v["ci95"] or [0, 0]), " ".join("%+.1f" % f for f in folds), out["arms"][arm]["up"], out["arms"][arm]["down"]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], iu=r["iu"], names=r["names"], per_name=r["per_name"], ex_top=r["ex_top"], truth=int(0))) + "\n" for r in recs))


if __name__ == "__main__":
    main()
