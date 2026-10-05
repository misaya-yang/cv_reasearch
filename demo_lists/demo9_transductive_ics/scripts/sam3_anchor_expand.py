#!/usr/bin/env python3
"""Anchor and expand: choose SAM3's query proposals by comparing them with each other inside the query (CPU).

  python scripts/sam3_anchor_expand.py --run RUN/dev --manifest SUITE/dev_episodes.json --regions X/regions_dev.npz \
      [--backward X/backward_dev.jsonl] [--frozen-from dev.json] --out X/anchor_dev.json

The method. The anchor is one proposal taken as the target. Every other proposal is kept when its frozen DINOv3 feature
is close to the anchor's: the comparison is inside one image, where instances of a concept share light, style and view;
the comparison with the reference image is used only to place the anchor. Rule names:
  top1                       the anchor alone (control: SAM3's most confident query proposal)
  relative_0.7               SAM3 score >= 0.7 x the top score (control: the per-episode level, no feature)
  expand_T                   anchor, plus every proposal whose cosine to the anchor is >= T
  filter_T                   relative_0.7, minus the proposals whose cosine to the anchor is < T
  both_T                     relative_0.7 filtered, plus proposals with cosine >= max(T, 0.8)
  nearer                     anchor, plus every proposal nearer to the anchor than to the reference background (no level)
The anchor is `score` (most confident) unless the rule ends in @NAME, NAME in ANCHORS.
Also printed: the anchor's accuracy per choice, the AUC of each signal on the non-anchor proposals, and two label
budgets (off-target proposals removed from relative_0.7; missed on-target proposals added).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
LEVELS = (0.5, 0.6, 0.7, 0.8)
ANCHORS = ("score", "ref", "votes", "score_ref")
CONTROLS = ("top1", "fallback_top1", "relative_0.7")


def rules():
    out = list(CONTROLS)
    for anchor in ANCHORS:
        tag = "" if anchor == "score" else "@" + anchor
        out += ["top1" + tag] if tag else []
        out += ["%s_%.1f%s" % (k, t, tag) for k in ("expand", "filter", "both") for t in LEVELS] + ["nearer" + tag]
    return out + ["budget_precision", "budget_recall", "labels"]


def anchor_of(kind, r):
    q, s, x = r["q"], r["score"], r["x"]
    if kind == "score":
        return max(q, key=lambda i: s[i])
    strong = [i for i in q if s[i] >= .5 * max(s[j] for j in q)]
    if kind == "ref":
        return max(strong, key=lambda i: x["to_ref"][i] - x["to_bg"][i])
    if kind == "votes":
        return max(strong, key=lambda i: x["ref_votes"][i])
    return max(q, key=lambda i: s[i] * max(x["to_ref"][i] - x["to_bg"][i] + 1, 1e-6))


def pick(rule, r):
    q, s, P = r["q"], r["score"], r["proposals"]
    if not q:
        return []
    top = max(s[i] for i in q)
    rel = [i for i in q if s[i] >= .7 * top]
    on = [i for i in q if P[i][2] > .5 * P[i][1]]
    if rule == "labels":
        return on
    if rule == "relative_0.7":
        return rel
    if rule == "fallback_top1":
        return [i for i in q if s[i] > .5] or [max(q, key=lambda i: s[i])]
    if rule == "budget_precision":
        return [i for i in rel if i in on] or on[:1]
    if rule == "budget_recall":
        return sorted(set(rel) | set(on))
    name, _, kind = rule.partition("@")
    a = anchor_of(kind or "score", r)
    sim = r["pooled"] @ r["pooled"][a]
    if name == "top1":
        return [a]
    if name == "nearer":
        return [i for i in q if i == a or sim[i] > r["x"]["to_bg"][i]]
    k, t = name.split("_")
    t = float(t)
    if k == "expand":
        return [i for i in q if i == a or sim[i] >= t]
    keep = [i for i in rel if i == a or sim[i] >= t]
    if k == "both":
        keep = sorted(set(keep) | {i for i in q if sim[i] >= max(t, .8)})
    return sorted(set(keep) | {a})


def auc(pos, neg):
    import numpy as np
    if not len(pos) or not len(neg):
        return None
    v = np.concatenate([pos, neg]).astype(float)
    order = v.argsort(kind="stable")
    rank = np.empty(len(v)); rank[order] = np.arange(1, len(v) + 1)
    for value in np.unique(v):
        m = v == value
        rank[m] = rank[m].mean()
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--regions", required=True)
    p.add_argument("--backward")
    p.add_argument("--frozen-from")
    p.add_argument("--fresh-from", nargs="*")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    z = np.load(a.regions)
    row = {tuple(int(v) for v in k): n for n, k in enumerate(z["keys"])}
    back = {}
    if a.backward and Path(a.backward).exists():
        back = {tuple(j["key"]): {r[0]: r for r in j["rows"]} for j in map(json.loads, Path(a.backward).read_text().splitlines())}
    recs = [json.loads(line) for f in sorted(Path(a.run).glob("episodes_shard*.jsonl")) for line in f.read_text().splitlines() if line]
    if a.fresh_from:
        images = {r[k] for m in a.fresh_from for r in json.loads(Path(m).read_text())["episodes"] for k in ("support", "query")}
        recs = [r for r in recs if r["support"] not in images and r["query"] not in images]
    recs = [r for r in recs if (r["fold"], r["e"], r["c"]) in row]
    rect, names = S.rectangles()[1], rules()
    for r in recs:
        n = row[(r["fold"], r["e"], r["c"])]
        P = r["proposals"]
        r["q"] = [i for i, v in enumerate(P) if v[1] > 0]
        r["score"] = [v[0] for v in P]
        r["pooled"] = z["pooled"][n].astype(np.float32)
        d = z["dense"][n]
        r["x"] = dict(to_ref=r["pooled"] @ z["ref_fg"][n].astype(np.float32), to_bg=r["pooled"] @ z["ref_bg"][n].astype(np.float32),
                      fg_dense=d[:, 0], bg_dense=d[:, 1], ref_votes=d[:, 2])
        count, h, w = r["proposal_shape"]
        with np.load(Path(a.run) / r["candidate_file"], allow_pickle=False) as c:
            raw = np.unpackbits(c["proposal_query"], axis=1)[:, :h * w].reshape(count, h, w).astype(bool)
        truth = S.on_canvas(np.asarray(Image.open(Path(man["annotation_root"]) / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1, rect)
        r["iu"] = {}
        for rule in names:
            idx = pick(rule, r)
            u = raw[idx].any(0) if idx else np.zeros((h, w), bool)
            r["iu"][rule] = [int((u & truth).sum()), int((u | truth).sum())]
    out = dict(state="COMPLETED", episodes=len(recs), reading="canvas", rules={})
    base = lambda r: r["iu"]["fallback_top1"]
    print("%d episodes, canvas reading, paired against fallback_top1" % len(recs))
    for rule in names:
        get = lambda r, rule=rule: r["iu"][rule]
        v = S.paired(recs, get, base)
        folds = [S.paired(sub, get, base, draws=1)["gain"] for f in sorted({r["fold"] for r in recs}) for sub in [[r for r in recs if r["fold"] == f]]]
        d = [get(r)[0] / max(get(r)[1], 1) - base(r)[0] / max(base(r)[1], 1) for r in recs]
        out["rules"][rule] = dict(miou=v["miou"], gain=v["gain"], ci95=v["ci95"], per_fold=folds, up=sum(x > 0 for x in d), down=sum(x < 0 for x in d))
        print("  %-22s %6.2f  %+6.2f [%+.2f, %+.2f]  folds %s  up %d down %d" % (rule, v["miou"], v["gain"], *(v["ci95"] or [0, 0]),
              " ".join("%+.1f" % f for f in folds), out["rules"][rule]["up"], out["rules"][rule]["down"]))
    # the anchor's accuracy, and the signals on the non-anchor proposals of episodes whose anchor (most confident) is on target
    on = lambda r, i: r["proposals"][i][2] > .5 * r["proposals"][i][1]
    out["anchor_on_target"] = {k: float(np.mean([on(r, anchor_of(k, r)) for r in recs if r["q"]])) for k in ANCHORS}
    print("anchor on target: " + ", ".join("%s %.3f" % kv for kv in out["anchor_on_target"].items()))
    sig = {}
    for r in recs:
        if not r["q"]:
            continue
        a0 = anchor_of("score", r)
        if not on(r, a0):
            continue
        sim, top = r["pooled"] @ r["pooled"][a0], r["score"][a0]
        b = back.get((r["fold"], r["e"], r["c"]), {})
        for i in r["q"]:
            if i == a0:
                continue
            v = dict(score=r["score"][i], to_anchor=sim[i], to_ref=r["x"]["to_ref"][i], ref_minus_bg=r["x"]["to_ref"][i] - r["x"]["to_bg"][i],
                     anchor_minus_bg=sim[i] - r["x"]["to_bg"][i], dense_fg_minus_bg=r["x"]["fg_dense"][i] - r["x"]["bg_dense"][i],
                     ref_votes=r["x"]["ref_votes"][i])
            if i in b:
                v["backward_best_iou"], v["backward_union_iou"] = b[i][3], b[i][2]
            for k, x in v.items():
                sig.setdefault(k, ([], []))[0 if on(r, i) else 1].append(float(x))
    out["auc_non_anchor"] = {k: dict(auc=auc(np.array(p), np.array(n)), on=len(p), off=len(n)) for k, (p, n) in sig.items()}
    print("non-anchor proposals, on-target against off-target:")
    for k, v in out["auc_non_anchor"].items():
        print("  AUC %-20s %s  (%d on, %d off)" % (k, "%.3f" % v["auc"] if v["auc"] is not None else "-", v["on"], v["off"]))
    family = [k for k in names if k not in CONTROLS and not k.startswith("budget") and k != "labels"]
    chosen = max(family, key=lambda k: out["rules"][k]["miou"])
    if a.frozen_from:
        dev = json.loads(Path(a.frozen_from).read_text())["rules"]
        chosen = max(family, key=lambda k: dev[k]["miou"])
    out["chosen"] = chosen
    for control in CONTROLS:
        v = S.paired(recs, lambda r: r["iu"][chosen], lambda r: r["iu"][control])
        out.setdefault("over", {})[control] = dict(gain=v["gain"], ci95=v["ci95"])
        print("%s %s: %.2f, %+.2f [%+.2f, %+.2f] over %s (%.2f)" % ("FROZEN on DEV" if a.frozen_from else "best here", chosen,
              out["rules"][chosen]["miou"], v["gain"], *v["ci95"], control, out["rules"][control]["miou"]))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
