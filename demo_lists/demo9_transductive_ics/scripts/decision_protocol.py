#!/usr/bin/env python3
"""Manifests for the main COCO-20i table without any image shared between fitting and scoring.

  python scripts/decision_protocol.py --like SUITE/dev_episodes.json --test 1000 --per-fold 3000 --out results/decision_main
Writes eval_episodes.json (the standard seed-0 list, the first `--test` episodes of every fold) and
train_episodes.json (the draws that follow, without any image of an evaluation episode of any fold and without a
repeated pair). A model that scores fold f is fitted on the training episodes of the other three folds, so it never
sees a class of fold f as a target, nor an image it is scored on. CPU, metadata only; no label pixel is read.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from extent_experiment import coco_episodes  # noqa: E402
from extent_train_cache import role_uids, uid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--like", type=Path, required=True, help="an existing manifest: data roots are copied from it")
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--per-fold", type=int, default=3000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    like = json.loads(a.like.read_text())
    common = {k: like[k] for k in ("data_root", "annotation_root", "foris_root", "projection_basis")}
    draws = {f: coco_episodes(common["data_root"], f, a.test + 30 * a.per_fold) for f in range(4)}
    for r in like["episodes"]:  # the list must be the one the earlier episodes came from
        if r["e"] < len(draws[r["fold"]]) and (r["c"], r["query"], r["support"]) != draws[r["fold"]][r["e"]]:
            raise SystemExit("episode list differs from %s" % a.like)
    test = [dict(fold=f, e=e, c=int(draws[f][e][0]), query=draws[f][e][1], support=draws[f][e][2]) for f in range(4) for e in range(a.test)]
    blocked = role_uids(test)
    rows, seen, stats = [], set(), {}
    for f in range(4):
        picked, rejected = 0, dict(blocked=0, repeated=0)
        for e in range(a.test, len(draws[f])):
            c, q, s = draws[f][e]
            if uid(q) in blocked or uid(s) in blocked:
                rejected["blocked"] += 1
            elif (f, c, q, s) in seen or q == s:
                rejected["repeated"] += 1
            else:
                rows.append(dict(fold=f, collection_fold=f, e=e, c=int(c), support=s, query=q, role="training"))
                seen.add((f, c, q, s))
                picked += 1
                if picked == a.per_fold:
                    break
        if picked != a.per_fold:
            raise SystemExit("draws exhausted for fold %d: %d" % (f, picked))
        mine = [r for r in rows if r["fold"] == f]
        stats[f] = dict(last=mine[-1]["e"], classes=len({r["c"] for r in mine}), images=len(role_uids(mine)), **rejected)
    if role_uids(rows) & blocked:
        raise SystemExit("image isolation failed")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "eval_episodes.json").write_text(json.dumps(dict(state="PREPARED", episodes=test, **common)))
    (a.out / "train_episodes.json").write_text(json.dumps(dict(state="PREPARED", episodes=rows, isolated_from="every image of eval_episodes.json", **common)))
    rep = dict(state="PREPARED", evaluation=len(test), evaluation_images=len(blocked), training=len(rows), training_images=len(role_uids(rows)), folds=stats)
    (a.out / "protocol.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
