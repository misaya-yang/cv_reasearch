#!/usr/bin/env python3
"""More training episodes for an existing image-isolated suite: the standard seed-0 draws that follow the ones the
suite already uses, without any image of the development or confirmation episodes and without repeating a pair.

  python scripts/decision_more_train.py --suite DIR --per-fold 1200 --out results/decision_v1/train_more_episodes.json
CPU, metadata only; no label pixel is read.
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
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--per-fold", type=int, default=1200)
    p.add_argument("--also", type=Path, nargs="*", default=[], help="earlier extra manifests whose pairs must not repeat")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sets = {k: json.loads((a.suite / (k + "_episodes.json")).read_text()) for k in ("train", "confirm", "dev")}
    old = sets["train"]["episodes"] + [r for f in a.also for r in json.loads(f.read_text())["episodes"]]
    blocked = role_uids(sets["dev"]["episodes"]) | role_uids(sets["confirm"]["episodes"])
    seen = {(r["fold"], r["c"], r["query"], r["support"]) for r in old}
    rows, stats = [], {}
    for f in range(4):
        first = max(r["e"] for r in old if r["fold"] == f) + 1
        draws = coco_episodes(sets["train"]["data_root"], f, first + 20 * a.per_fold)
        picked, rejected = 0, dict(blocked=0, repeated=0)
        for e in range(first, len(draws)):
            c, q, s = draws[e]
            if uid(q) in blocked or uid(s) in blocked:
                rejected["blocked"] += 1
            elif (f, c, q, s) in seen or q == s:
                rejected["repeated"] += 1
            else:
                for name in (s, q):
                    if not (Path(sets["train"]["data_root"]) / name).is_file():
                        raise SystemExit("missing image: " + name)
                rows.append(dict(fold=f, collection_fold=f, e=e, c=int(c), support=s, query=q, role="training"))
                seen.add((f, c, q, s))
                picked += 1
                if picked == a.per_fold:
                    break
        if picked != a.per_fold:
            raise SystemExit("draws exhausted for fold %d" % f)
        mine = [r for r in rows if r["fold"] == f]
        stats[f] = dict(first=first, last=mine[-1]["e"], classes=len({r["c"] for r in mine}), **rejected)
    if role_uids(rows) & blocked:
        raise SystemExit("image isolation failed")
    common = {k: sets["train"][k] for k in ("data_root", "annotation_root", "foris_root", "projection_basis")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(state="PREPARED", episodes=rows, isolated_from="development and confirmation images of " + str(a.suite), **common)))
    print(json.dumps(dict(state="PREPARED", episodes=len(rows), images=len(role_uids(rows)), folds=stats)))


if __name__ == "__main__":
    main()
