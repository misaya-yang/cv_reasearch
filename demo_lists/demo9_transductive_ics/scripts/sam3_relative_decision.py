#!/usr/bin/env python3
"""A per-episode decision on SAM3's own proposals, read exactly (CPU; no model, no training, no query label in a rule).

  python scripts/sam3_relative_decision.py --run RUN/dev --manifest SUITE/dev_episodes.json --out results/sam3_relative_v1/dev.json
  python scripts/sam3_relative_decision.py --run RUN/confirm --manifest SUITE/confirm_episodes.json \
      --frozen-from results/sam3_relative_v1/dev.json --out results/sam3_relative_v1/confirm.json
  python scripts/sam3_relative_decision.py --records results/f4_v1/sam3_A_dev_episodes.jsonl --approximate --out /tmp/a.json
  python scripts/sam3_relative_decision.py --fixture /tmp/relative_fixture

RUN is the output folder of one `sam3_stitch.py` stage scored with --keep-candidates: `episodes_shard*.jsonl` (the
scored records, 20 most confident proposals of the visual arm each) and `candidates/*.npz` (their bitmaps inside the
query rectangle of the canvas). The public protocol keeps a proposal when its score exceeds 0.5, the same level for
every episode. The rules below use only the scores and the geometry of the canvas:

  absolute_0.5    score > 0.5                                   the public protocol (reference row)
  absolute_0.2 / 0.3 / 0.4   score > the level                  controls: another level, still the same for every episode
  fallback_top1   score > 0.5, else the top query proposal      no parameter
  relative_0.6 / 0.7 / 0.8   score >= a x the top query score   the level follows the episode
  labels          proposals mostly on the target                privileged diagnostic, never a method

A query proposal is one with area inside the query rectangle. Two readings per rule, both class mIoU with the
photograph-group bootstrap of `sam3_stitch.paired` against absolute_0.5: `canvas` (the union in the query rectangle,
the family of `proposal_reading`) and `original` (the same union at the query's own size, the family of the main
table). `stored_visual` is the run's own final mask; it can differ from absolute_0.5 when more than 20 proposals passed.
--approximate needs no candidate files: unions are taken as if proposals never overlapped (canvas reading only).
--frozen-from DEV.json marks the label-free rule that was best on DEV (original reading) as the one reading of this
report, and reads it against the fixed level that was best on DEV: a per-episode level counts only if it beats the
best single level. `ledger` sorts the episodes of a rule by what it kept (on-target proposals, off-target ones, none). The relative rules assume the target is present in the query, as the benchmark does.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
RULES = ("absolute_0.5", "absolute_0.2", "absolute_0.3", "absolute_0.4", "fallback_top1", "relative_0.6", "relative_0.7", "relative_0.8", "labels")
CONTROLS = ("absolute_0.2", "absolute_0.3", "absolute_0.4")  # fixed levels other than the public one; the best on DEV is the control
CANDIDATES = ("fallback_top1", "relative_0.6", "relative_0.7", "relative_0.8")  # what --frozen-from may choose


def pick(rule, proposals):
    """Indices of the proposals a rule keeps. A proposal row is [score, query area, overlap with truth, ...]."""
    q = [i for i, p in enumerate(proposals) if p[1] > 0]
    if rule.startswith("absolute_"):
        return [i for i in q if proposals[i][0] > float(rule.split("_")[1])]
    if rule == "labels":
        return [i for i in q if proposals[i][2] > .5 * proposals[i][1]]
    if not q:
        return []
    top = max(q, key=lambda i: proposals[i][0])
    if rule == "fallback_top1":
        return [i for i in q if proposals[i][0] > .5] or [top]
    return [i for i in q if proposals[i][0] >= float(rule.split("_")[1]) * proposals[top][0]]


def read(records, run, manifest, approximate):
    """Adds r['rule_iu'][rule] = dict(canvas=[I, U], original=[I, U]) to every record."""
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    rect = S.rectangles()[1]
    mismatch = 0
    for r in records:
        r["rule_iu"] = {}
        if approximate:
            t = r["canvas_truth"]
            exact = {"absolute_0.5": r["proposal_union_iu"]["0.5"], "absolute_0.3": r["proposal_union_iu"]["0.3"],
                     "labels": r["proposal_majority_truth_union_iu"]}  # the scored record already holds these unions exactly
            for rule in RULES:
                keep = [r["proposals"][i] for i in pick(rule, r["proposals"])]
                i = min(sum(p[2] for p in keep), t)  # overlapping proposals inflate the union: this reading is pessimistic
                r["rule_iu"][rule] = dict(canvas=list(exact.get(rule, [i, sum(p[1] for p in keep) + t - i])))
            continue
        n, h, w = r["proposal_shape"]
        with np.load(Path(run) / r["candidate_file"], allow_pickle=False) as z:
            raw = np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)
        truth = np.asarray(Image.open(Path(manifest["annotation_root"]) / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        tq = S.on_canvas(truth, rect)
        for rule in RULES:
            idx = pick(rule, r["proposals"])
            union = raw[idx].any(0) if idx else np.zeros((h, w), bool)
            full = np.asarray(Image.fromarray(union.astype(np.uint8)).resize(truth.shape[::-1], Image.NEAREST)) > 0
            r["rule_iu"][rule] = dict(canvas=[int((union & tq).sum()), int((union | tq).sum())],
                                      original=[int((full & truth).sum()), int((full | truth).sum())])
        mismatch += r["rule_iu"]["absolute_0.5"]["canvas"] != list(r["proposal_union_iu"]["0.5"])
    return mismatch


def table(records, approximate):
    import sam3_stitch as S
    out = {}
    for family in ("canvas",) if approximate else ("canvas", "original"):
        base = lambda r: r["rule_iu"]["absolute_0.5"][family]
        rows = {}
        for rule in RULES:
            get = lambda r, rule=rule: r["rule_iu"][rule][family]
            row = S.paired(records, get, base)
            row = dict(miou=row["miou"], gain=row["gain"], ci95=row["ci95"])
            row["per_fold"] = [S.paired(sub, get, base, draws=1)["gain"] for f in sorted({r["fold"] for r in records})
                               for sub in [[r for r in records if r["fold"] == f]]]
            d = [get(r)[0] / max(get(r)[1], 1) - base(r)[0] / max(base(r)[1], 1) for r in records]
            row["up"], row["down"] = sum(v > 0 for v in d), sum(v < 0 for v in d)
            row["episodes_with_nothing_kept"] = sum(get(r)[0] == 0 and not pick(rule, r["proposals"]) for r in records)
            rows[rule] = row
        out[family] = rows
    return out


def ledger(records, rule, family):
    """Episodes of a rule by what it kept; on target = a query proposal with more than half of its area on the truth."""
    cats = {}
    for r in records:
        P = r["proposals"]
        on = {i for i, p in enumerate(P) if p[1] > 0 and p[2] > .5 * p[1]}
        kept = set(pick(rule, P))
        name = ("no_on_target_proposal_in_top20" if not on else "nothing_kept" if not kept else "only_off_target_kept" if not kept & on
                else "on_target_plus_off_target" if kept - on else "on_target_but_some_missed" if on - kept else "clean")
        i, u = r["rule_iu"][rule][family]
        v = cats.setdefault(name, [0, 0.0]); v[0] += 1; v[1] += i / max(u, 1)
    return {k: dict(episodes=n, mean_iou=100 * t / n) for k, (n, t) in sorted(cats.items())}


def report(a):
    paths = [Path(a.records)] if a.records else sorted(Path(a.run).glob("episodes_shard*.jsonl"))
    records = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]
    if not records:
        raise SystemExit("no scored records under %s" % (a.records or a.run))
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    dropped = 0
    if a.fresh_from:  # keep only episodes that share no image with these manifests
        images = {r[k] for m in a.fresh_from for r in json.loads(Path(m).read_text())["episodes"] for k in ("support", "query")}
        fresh = [r for r in records if r["support"] not in images and r["query"] not in images]
        dropped, records = len(records) - len(fresh), fresh
        if not records:
            raise SystemExit("no fresh record left")
    mismatch = read(records, a.run, manifest, a.approximate)
    import sam3_stitch as S
    stored = lambda arm: S.paired(records, lambda r: r["original_iu"][arm], lambda r: r["original_iu"]["visual"], draws=1)["miou"]
    rep = dict(state="COMPLETED", episodes=len(records), approximate=bool(a.approximate), rules=table(records, a.approximate),
               stored_visual=stored("visual"), stored_text_privileged=stored("text"),
               absolute_0p5_differs_from_stored_union_in=mismatch, dropped_for_shared_images=dropped,
               scope="class mIoU; paired against absolute_0.5; 2000 draws over photograph groups, seed 0")
    family = "canvas" if a.approximate else "original"
    shown = max(CANDIDATES, key=lambda k: rep["rules"][family][k]["miou"])
    if a.frozen_from:
        dev = json.loads(Path(a.frozen_from).read_text())["rules"]["original"]
        shown = rep["frozen_rule"] = max(CANDIDATES, key=lambda k: dev[k]["miou"])
        control = max(CONTROLS, key=lambda k: dev[k]["miou"])
        over = S.paired(records, lambda r: r["rule_iu"][shown]["original"], lambda r: r["rule_iu"][control]["original"])
        rep["reading"] = dict(rule=shown, **rep["rules"]["original"][shown], control=control, control_row=rep["rules"]["original"][control],
                              gain_over_control=dict(gain=over["gain"], ci95=over["ci95"]))
    rep["ledger"] = {rule: ledger(records, rule, family) for rule in ("absolute_0.5", shown)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    Path(str(a.out) + ".episodes.jsonl").write_text("".join(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], rule_iu=r["rule_iu"])) + "\n" for r in records))
    for family, rows in rep["rules"].items():
        print("%s reading, %d episodes%s" % (family, len(records), " (approximate unions)" if a.approximate else ""))
        for rule, v in rows.items():
            print("  %-14s %6.2f  %+6.2f %s  folds %s  up %d down %d  empty %d" % (
                rule, v["miou"], v["gain"], "[%+.2f, %+.2f]" % tuple(v["ci95"]) if v["ci95"] else "", " ".join("%+.1f" % g for g in v["per_fold"]),
                v["up"], v["down"], v["episodes_with_nothing_kept"]))
    print("stored final masks: visual %.2f, text (privileged) %.2f; absolute_0.5 differs from the stored union in %d episodes" % (
        rep["stored_visual"], rep["stored_text_privileged"], mismatch))
    for rule, cats in rep["ledger"].items():
        print("ledger %s: %s" % (rule, "; ".join("%s %d (IoU %.1f)" % (k, v["episodes"], v["mean_iou"]) for k, v in cats.items())))
    if a.frozen_from:
        v = rep["reading"]
        print("FROZEN on DEV: %s -> %.2f, %+.2f [%+.2f, %+.2f] over absolute_0.5; over the best fixed level (%s, %.2f): %+.2f [%+.2f, %+.2f]" % (
            v["rule"], v["miou"], v["gain"], *v["ci95"], v["control"], v["control_row"]["miou"], v["gain_over_control"]["gain"], *v["gain_over_control"]["ci95"]))


def fixture(a):
    """Synthetic run through the exact path: files written in the layout of sam3_stitch, then read back."""
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    root = Path(a.fixture); (root / "run/candidates").mkdir(parents=True, exist_ok=True); (root / "ann").mkdir(exist_ok=True)
    x, y, w, h = S.rectangles()[1]
    rng, lines = np.random.default_rng(0), []
    for e in range(8):
        truth = np.zeros((120, 160), bool); truth[20:80, 30:110] = True
        Image.fromarray((truth * (e % 2 + 1)).astype(np.uint8)).save(root / "ann" / ("q%d.png" % e))
        tq = S.on_canvas(truth, (x, y, w, h))
        good, bad = tq.copy(), np.zeros_like(tq); bad[:, : w // 5] = True
        scores = [0.45, 0.30] if e < 4 else [0.80, 0.60]   # in the first four episodes nothing passes 0.5
        raw = np.stack([good, bad & ~tq])
        np.savez_compressed(root / "run/candidates" / ("%d.npz" % e), proposal_query=np.packbits(raw.reshape(2, h * w), axis=1))
        props = [[scores[i], int(raw[i].sum()), int((raw[i] & tq).sum()), 0, 0] for i in range(2)]
        union = lambda t: raw[[i for i in range(2) if scores[i] > t]].any(0) if any(s > t for s in scores) else np.zeros_like(tq)
        lines.append(json.dumps(dict(fold=e % 2, e=e, c=e % 2, query="q%d.jpg" % e, support="s%d.jpg" % e, proposals=props, proposal_shape=[2, h, w],
                                     candidate_file="candidates/%d.npz" % e, canvas_truth=int(tq.sum()),
                                     proposal_union_iu={str(t): [int((union(t) & tq).sum()), int((union(t) | tq).sum())] for t in (.1, .3, .5, .7)},
                                     original_iu=dict(visual=[0, int(truth.sum())], text=[int(truth.sum()), int(truth.sum())]))))
    (root / "run/episodes_shard0.jsonl").write_text("\n".join(lines) + "\n")
    (root / "manifest.json").write_text(json.dumps(dict(annotation_root=str(root / "ann"))))
    a.run, a.manifest, a.records, a.out, a.approximate, a.frozen_from = str(root / "run"), str(root / "manifest.json"), None, str(root / "dev.json"), False, None
    (root / "seen.json").write_text(json.dumps(dict(episodes=[dict(support="s0.jpg", query="other.jpg")])))  # shares an image with episode 0
    report(a)
    a.frozen_from, a.out = str(root / "dev.json"), str(root / "confirm.json")
    report(a)
    a.fresh_from, a.out = [str(root / "seen.json")], str(root / "fresh.json")
    report(a)
    fresh = json.loads((root / "fresh.json").read_text())
    rep = json.loads((root / "confirm.json").read_text()); o = rep["rules"]["original"]
    checks = [("absolute_0.5 reproduces the stored union", rep["absolute_0p5_differs_from_stored_union_in"] == 0),
              ("absolute_0.5 is empty in four episodes", o["absolute_0.5"]["episodes_with_nothing_kept"] == 4),
              ("fallback_top1 leaves no episode empty and beats absolute_0.5", o["fallback_top1"]["gain"] > 30 and o["fallback_top1"]["episodes_with_nothing_kept"] == 0),
              ("relative_0.8 drops the wrong proposal", o["relative_0.8"]["miou"] > 95),
              ("absolute_0.3 keeps the wrong proposal in the second half", o["absolute_0.3"]["miou"] < o["relative_0.8"]["miou"]),
              ("labels is the ceiling", all(o["labels"]["miou"] >= v["miou"] - 1e-9 for v in o.values())),
              ("--fresh-from drops the episode that shares an image", fresh["episodes"] == 7 and fresh["dropped_for_shared_images"] == 1),
              ("a rule was frozen from DEV and read against the best fixed level", rep["frozen_rule"] in CANDIDATES and rep["reading"]["control"] in CONTROLS
               and "gain_over_control" in rep["reading"] and "absolute_0.5" in rep["ledger"])]
    bad = [n for n, ok in checks if not ok]
    print("%d/%d checks pass%s" % (len(checks) - len(bad), len(checks), "" if not bad else "; FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="output folder of a sam3_stitch stage scored with --keep-candidates")
    p.add_argument("--records", help="one scored jsonl instead of --run (with --approximate)")
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--frozen-from")
    p.add_argument("--fresh-from", nargs="*", help="manifests whose images must not appear in a record that is read")
    p.add_argument("--approximate", action="store_true")
    p.add_argument("--fixture")
    a = p.parse_args()
    fixture(a) if a.fixture else report(a)


if __name__ == "__main__":
    main()
